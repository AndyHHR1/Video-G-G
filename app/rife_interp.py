"""RIFE motion-compensated frame interpolation (v4.6 HD) para display SR fluido.

RIFE: Real-Time Intermediate Flow Estimation. Vendido bajo ``app/rife/``
(codigo canónico del repo oficial ``hzwer/ECCV2022-RIFE`` + weights de
``hzwer/RIFE`` ``RIFEv4.26_0921``). La carga de pesos es offline (24 MB en
``app/rife/train_log/flownet.pkl``), sin red en runtime.

Por qué se necesita: en la RTX 5050 la Real-ESRGAN 4x a 1280x720 cuesta ~267 ms
=> ~3.7 fps de contenido. El display async *hold* actual actualiza el
contenido a ~2.5 fps (stutter/judder en movimiento). RIFE interpola entre
keyframes SR a resolución reducida (~640 p, ~45 ms/c/frame fp32 en RTX 5050)
=> el display pasa a ~10-12 fps con movimiento suave, SIN tocar la detección
(YOLO/Mediapipe corren en el grid nativo 640; la SR/RIFE son solo display).

La inferencia a resolución reducida (max 640 p) acelera RIFE 3-4× sin
perder calidad perceptible: la SR 4× domina la nitidez y RIFE solo alisa
transiciones. Los frames interpolados se escalan de vuelta a 720p con
bilinear.

Detección de movimiento excesivo: el umbral de dif. absoluta media se
computa a 1/4 de resolución (filtra el noise textural de Real-ESRGAN) y
se baja a 5%. Cuando un key-frame SR difiere >5% del anterior (movimiento
moderado-adelantado), RIFE no interpela: evita artefactos (fantasmales,
desapariciones) que aparecen con movimientos rápidos o cambios de cámara.
En esos casos interp retorna [] y el display vuelve al modo *hold*.

Requisitos de tamaño: IFNet v4.26 necesita H,W divisibles por 64.
``interp`` escala a max_size manteniendo aspect ratio y redondea al
múltiplo de 64 superior. El fallback es gracioso: si RIFE no carga
(falta CUDA/pesos), ``ready`` es False y el llamador mantiene el *hold*.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_ROOT, "rife")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


class RIFEInterpolator:
    """Interfaz mínima sobre el modelo RIFE HD v4.6 para interpolación de pares."""

    def __init__(self, weight_dir: str | None = None, device: str = "auto",
                 motion_threshold: float = 0.05,
                 max_size: int = 640):
        import torch as _torch
        if device == "auto":
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.weight_dir = weight_dir or os.path.join(_PKG, "train_log")
        self.model = None
        self.ready = False
        self.motion_threshold = motion_threshold
        # RIFE inferencia a una resolución escalada (max_size×max_size).
        # El flownet (IFNet v4.6) es cuadrático en la resolución: 1280×720 →
        # ~120 ms, 640×360 → ~45 ms en RTX 5050. La diferencia visual es mínima
        # (la SR 4× ya introduce su propio blur/textura) y ganamos ~2.6× fps.
        self.max_size = max_size
        if self.device == "cpu":
            return
        try:
            from train_log.RIFE_HDv3 import Model
            self.model = Model(local_rank=-1)
            self.model.load_model(self.weight_dir, -1)
            self.model.eval()
            self.model.device()
            _torch.cuda.empty_cache()
            # Warmup CUDA: la primera llamada a inference() paga ~3 s de overhead
            # (cuDNN autotuning, lazy allocation). Se hace aqui con un tensor
            # dummy a 64×64 para que el primer frame real interp() no bloquee
            # ~3 s (que mataría el display por un frame) y el usuario no vea un
            # stutter congelado al activar el live SR.
            with _torch.no_grad():
                _dummy = _torch.zeros(1, 3, 64, 64, device=self.device)
                try:
                    self.model.inference(_dummy, _dummy, timestep=0.5, scale=1.0)
                except Exception:
                    pass
                _torch.cuda.empty_cache()
            self.ready = True
        except Exception as exc:
            self.model = None
            self.ready = False
            print(f"[RIFE] carga fallida, interp deshabilitada: {exc!r}",
                  flush=True)

    def _motion_magnitude(self, img0: np.ndarray, img1: np.ndarray) -> float:
        """Detecta movimiento entre key-frames SR.

        Se muestrea a 1/4 de resolución (160×90) para: (1) ser 16× más rápido
        que comparar full-res, y (2) filtrar el *noise* textural que Real-ESRGAN
        introduce frame-a-frame (texturas GAN de alta frecuencia inflan la DIF
        absoluta sin significar movimiento real). A 0.05 (5%) el umbral atrapa
        movimientos moderados (brazo, cabeza) donde RIFE produce ghosting.
        """
        s = 4
        if img0.shape[0] < 16 or img0.shape[1] < 16:
            return 0.0
        h, w = img0.shape[:2]
        sh = max(1, h // s)
        sw = max(1, w // s)
        g0 = cv2.resize(img0, (sw, sh), interpolation=cv2.INTER_AREA)
        g1 = cv2.resize(img1, (sw, sh), interpolation=cv2.INTER_AREA)
        return float(
            np.abs(g0.astype(np.float32) - g1.astype(np.float32)).mean() / 255.0
        )

    def interp(self, img0: np.ndarray, img1: np.ndarray,
               num_steps: int = 1,
               times: list | None = None) -> list:
        """Genera ``num_steps`` frames intermedios entre img0 e img1.

        img0/img1: uint8 BGR (H,W,3), idéntico shape. Devuelve lista de uint8
        BGR del mismo shape. Si RIFE no está ready, los args son inválidos,
        o el movimiento supera el umbral, devuelve [].

        A diferencia de la versión full-res, aquí RIFE inferencia a
        ``max_size`` (default 640) para ser ~3-4× más rápido. Los resultados
        se escalan de vuelta a la resolución original con bilinear (suficiente:
        la SR 4× ya domina la nitidez; RIFE solo alisa transiciones).
        """
        if not self.ready or num_steps <= 0 or img0 is None or img1 is None:
            return []
        if self._motion_magnitude(img0, img1) > self.motion_threshold:
            return []
        if times is None:
            times = [(i + 1) / (num_steps + 1) for i in range(num_steps)]
        h, w = img0.shape[:2]

        # Downscale a max_size manteniendo aspect ratio (IFNet v4.6 necesita
        # divisibles por 64). Inferenciar a 640 p es ~3.7× más rápido que 1280×720.
        scale = min(self.max_size / max(h, w), 1.0)
        dh = int(round(h * scale))
        dw = int(round(w * scale))
        dh = ((dh + 63) // 64) * 64
        dw = ((dw + 63) // 64) * 64
        img0_s = cv2.resize(img0, (dw, dh), interpolation=cv2.INTER_AREA)
        img1_s = cv2.resize(img1, (dw, dh), interpolation=cv2.INTER_AREA)

        def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(
                np.ascontiguousarray(bgr)).to(self.device,
                                              dtype=torch.float32)
            t = t.permute(2, 0, 1)[None, ...] / 255.0
            return t

        t0 = _to_tensor(img0_s)
        t1 = _to_tensor(img1_s)
        out: list = []
        with torch.no_grad():
            for tt in times:
                f = self.model.inference(t0, t1, timestep=tt, scale=1.0)
                # NCHW → HWC, GPU → CPU, tensor → numpy
                f = f[0].permute(1, 2, 0).float().cpu().numpy()
                f = np.clip(f * 255.0, 0, 255).astype(np.uint8)
                # Upscale de vuelta a resolución original SR (720p)
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
                out.append(f)
        return out
