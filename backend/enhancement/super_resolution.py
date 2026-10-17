"""Superresolución x4 con Real-ESRGAN (mejora de calidad de vídeo).

Dos modelos oficiales, elegibles en `configs/app.yaml`:

| clave           | red              | uso por defecto          |
|-----------------|------------------|--------------------------|
| `general_x4v3`  | SRVGGNetCompact  | cámara en vivo y vídeo   |
| `x4plus`        | RRDBNet (ESRGAN) | imagen fija              |

Todo el pre y postproceso se hace en la GPU (conversión BGR->RGB, escalado
final, paso a uint8) para que a la CPU solo vuelva la imagen ya del tamaño
que se va a mostrar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .archs import RRDBNet, SRVGGNetCompact

MODEL_ZOO: dict[str, dict] = {
    "general_x4v3": {
        "file": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.5.0/realesr-general-x4v3.pth",
        "sha256": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        "build": lambda: SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4),
        "key": "params",
    },
    "x4plus": {
        "file": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth",
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "build": lambda: RRDBNet(num_feat=64, num_block=23, num_grow_ch=32),
        "key": "params_ema",
    },
}
SCALE = 4


def _torch_device(device) -> torch.device:
    if device in ("auto", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


class SuperResolver:
    """Carga un modelo Real-ESRGAN y amplía fotogramas BGR uint8."""

    def __init__(self, name: str, weights_dir: str | Path, device="auto",
                 half: bool = True):
        if name not in MODEL_ZOO:
            raise ValueError(f"modelo SR desconocido: {name!r} "
                             f"(opciones: {', '.join(MODEL_ZOO)})")
        spec = MODEL_ZOO[name]
        path = Path(weights_dir) / spec["file"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Faltan los pesos de superresolución: {path}\n"
                "  Ejecuta: python tools/download_models.py")
        self.name = name
        self.device = _torch_device(device)
        self.half = bool(half and self.device.type == "cuda")
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        state = state.get(spec["key"], state)
        net = spec["build"]()
        net.load_state_dict(state, strict=True)
        net.eval().to(self.device)
        self.net = net.half() if self.half else net
        self.warmup()

    @property
    def cuda(self) -> bool:
        return self.device.type == "cuda"

    @torch.no_grad()
    def warmup(self, h: int = 270, w: int = 480) -> None:
        dummy = np.zeros((h, w, 3), np.uint8)
        self.enhance(dummy)
        if self.cuda:
            torch.cuda.synchronize()

    @torch.no_grad()
    def enhance(self, frame: np.ndarray, input_max: int = 640,
                output_width: int | None = None,
                output_max: int | None = None) -> np.ndarray:
        """Amplía `frame` (BGR uint8) x4 y lo devuelve en BGR uint8.

        * `input_max`: lado mayor que entra a la red (se reduce si es mayor).
        * `output_width`: si se da, la salida se escala a ese ancho (vivo).
        * `output_max`: si se da, tope del lado mayor de la salida (uploads).
        """
        h, w = frame.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).flip(1)          # BGR -> RGB, NCHW
        x = x.half() if self.half else x.float()
        x = x / 255.0

        s = min(1.0, input_max / max(h, w))
        if s < 1.0:
            ih, iw = max(2, round(h * s)), max(2, round(w * s))
            x = F.interpolate(x, size=(ih, iw), mode="area")

        y = self.net(x).clamp_(0, 1)
        oh, ow = y.shape[-2:]
        if output_width:
            th, tw = round(oh * output_width / ow), output_width
        elif output_max and max(oh, ow) > output_max:
            r = output_max / max(oh, ow)
            th, tw = round(oh * r), round(ow * r)
        else:
            th, tw = oh, ow
        if (th, tw) != (oh, ow):
            mode = "area" if th < oh else "bicubic"
            y = F.interpolate(y, size=(th, tw), mode=mode,
                              **({} if mode == "area" else {"align_corners": False}))
            y = y.clamp_(0, 1)
        y = (y * 255.0).round_().to(torch.uint8)
        return y[0].flip(0).permute(1, 2, 0).contiguous().cpu().numpy()  # BGR
