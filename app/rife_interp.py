"""RIFE motion-compensated frame interpolation (v4.6 HD) para display SR fluido.

RIFE: Real-Time Intermediate Flow Estimation. Vendido bajo ``app/rife/``
(codigo canónico del repo oficial ``hzwer/ECCV2022-RIFE`` + weights de
``hzwer/RIFE`` ``RIFEv4.26_0921``). La carga de pesos es offline (24 MB en
``app/rife/train_log/flownet.pkl``), sin red en runtime.

Por qué se necesita: en la RTX 5050 la Real-ESRGAN 4x a 1280x720 cuesta ~267 ms
=> ~3.7 fps de contenido. El display async *hold* actual actualiza el
contenido a ~2.5 fps (stutter/judder en movimiento). RIFE interpola entre
keyframes SR a ~30 ms c/frame (fp16, 768x1280) => el display pasa a ~7-9 fps de
contenido con movimiento suave, SIN tocar la detección (YOLO/Mediapipe corren en
el grid nativo 640; la SR/RIFE son solo display).

Detección de movimiento excesivo: cuando dos key-frames SR difieren en más del
umbral configurable (default 12% DIF absoluta media), RIFE no interpela: eso
evita artefactos (fantasmales, desapariciones, ghosting) que aparecen con movimientos
rápidos o cambios de cámara. En esos casos interp retorna [] y el display vuelve al
comportamiento *hold* (key-frame SR sin interpolar).

Requisitos de tamaño: IFNet v4.26 necesita H,W divisibles por 64 (caso de fallo
comprobado con H=720: 720/64 no es entero, la IFNet multi-era falla en el warp).
``interp`` paddea a múltiplo de 64 de forma simétrica, interpela y recorta de
vuelta al tamaño origen. El fallback es gracioso: si RIFE no carga (falta
CUDA/pesos), ``ready`` es False y el llamador mantiene el comportamiento *hold*.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_ROOT, "rife")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


class RIFEInterpolator:
    """Interfaz mínima sobre el modelo RIFE HD v4.6 para interpolación de pares."""

    def __init__(self, weight_dir: str | None = None, device: str = "auto",
                 motion_threshold: float = 0.12):
        import torch as _torch
        if device == "auto":
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.weight_dir = weight_dir or os.path.join(_PKG, "train_log")
        self.model = None
        self.ready = False
        self.motion_threshold = motion_threshold
        if self.device == "cpu":
            return
        try:
            from train_log.RIFE_HDv3 import Model
            self.model = Model(local_rank=-1)
            self.model.load_model(self.weight_dir, -1)
            self.model.eval()
            self.model.device()
            _torch.cuda.empty_cache()
            self.ready = True
        except Exception as exc:
            self.model = None
            self.ready = False
            print(f"[RIFE] carga fallida, interp deshabilitada: {exc!r}",
                  flush=True)

    @staticmethod
    def _pad_size(h: int, w: int) -> tuple[int, int]:
        """Redondea al múltiplo de 64 superior (IFNet v4.26 lo exige)."""
        ph = ((h + 63) // 64) * 64
        pw = ((w + 63) // 64) * 64
        return ph, pw

    def _motion_magnitude(self, img0: np.ndarray, img1: np.ndarray) -> float:
        """Computa la media normalizada de diferencia absoluta entre dos frames.

        Sirve para detectar movimiento excesivo entre key-frames SR: si la
        diferencia supera ``motion_threshold`` el frame interpolado tenderá a
        tener ghosting/desapariciones, así que se salta la interpolación.
        """
        return float(
            np.abs(img0.astype(np.float32) - img1.astype(np.float32)).mean()
            / 255.0
        )

    def interp(self, img0: np.ndarray, img1: np.ndarray,
               num_steps: int = 1,
               times: list | None = None) -> list:
        """Genera ``num_steps`` frames intermedios entre img0 e img1.

        img0/img1: uint8 BGR (H,W,3), idéntico shape. Devuelve lista de uint8
        BGR del mismo shape. Si RIFE no está ready, los args son inválidos,
        o el movimiento supera el umbral, devuelve [].
        """
        if not self.ready or num_steps <= 0 or img0 is None or img1 is None:
            return []
        # Salta interpolación si el movimiento entre frames es excesivo:
        # RIFE produce artefactos (persona fantasma/desaparecida) que
        # degradan el display. En ese caso vuelve al modo *hold*.
        if self._motion_magnitude(img0, img1) > self.motion_threshold:
            return []
        if times is None:
            times = [(i + 1) / (num_steps + 1) for i in range(num_steps)]
        h, w = img0.shape[:2]
        ph, pw = self._pad_size(h, w)
        pb = (ph - h) // 2
        pt = (ph - h) - pb
        pl = (pw - w) - (pw - w) // 2
        pr = (pw - w) // 2

        def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(
                np.ascontiguousarray(bgr[:h, :w])).to(self.device,
                                                       dtype=torch.float32)
            t = t.permute(2, 0, 1)[None, ...] / 255.0
            t = torch.nn.functional.pad(t, (pl, pr, pt, pb),
                                        mode="constant", value=0)
            return t

        t0 = _to_tensor(img0)
        t1 = _to_tensor(img1)
        out: list = []
        # El autocast se abre una sola vez (no por timestep) para reducir
        # overhead de contexto: ~3-5 ms ahorrados c/2 steps en RTX 5050.
        with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16):
            for tt in times:
                f = self.model.inference(t0, t1, timestep=tt, scale=1.0)
                # NCHW → HWC, fp16 → fp32, GPU → CPU, tensor → numpy
                f = f[0].permute(1, 2, 0).float().cpu().numpy()
                if f.shape[0] != h or f.shape[1] != w:
                    f = f[pt:pt + h, pl:pl + w]
                f = np.clip(f * 255.0, 0, 255).astype(np.uint8)
                out.append(f)
        return out
