"""Morfología (apertura/cierre) de la máscara de primer plano ANTES de la pose.

**Desactivada por defecto** (`skeleton.morphology: "off"`), porque no mejora:

YOLOv8-Pose trabaja sobre la imagen en color, no sobre una máscara. Se midió
con las imágenes de ejemplo de Ultralytics (bus.jpg, zidane.jpg), con y sin un
pasamanos sintético, la confianza media de los 17 puntos:

| variante                              | bus.jpg       | zidane.jpg    |
|---------------------------------------|---------------|---------------|
| original                              | 0.598         | 0.528         |
| apertura + cierre sobre la imagen     | 0.642 ¹       | 0.531         |
| máscara de personas + morfología      | 0.606         | 0.541         |
| con pasamanos                         | 0.368         | 0.465         |
| pasamanos + apertura / cierre         | 0.354 / 0.329 | 0.467 / 0.486 |
| pasamanos + máscara + morfología      | 0.388         | 0.461         |

¹ sube la media pero no la fracción de puntos fiables (0.57 en ambos).

Los resultados son mixtos y pequeños: con el pasamanos, el cierre sube 0.02 en
zidane.jpg pero baja 0.04 en bus.jpg; la máscara da como mucho +0.02 y cuesta
otra pasada de red. La fracción de puntos fiables (conf > 0.5) no mejora de
forma consistente. Como no mejora de forma fiable, se omite.

Queda disponible el modo `"mascara"` para medirlo con el vídeo REAL de la
escalera (cámara fija): sustracción de fondo MOG2 → apertura (quita ruido) →
cierre (rellena huecos, p. ej. el que deja el pasamanos) → dilatación, y se
atenúa el fondo fuera de la máscara. Para decidir con datos:

    python tools/eval_morfologia_pose.py video_escalera.mp4
"""
from __future__ import annotations

import cv2
import numpy as np


class ForegroundMorphology:
    def __init__(self, open_px: int = 5, close_px: int = 25,
                 dilate_px: int = 15, background_keep: float = 0.35):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300,
                                                     varThreshold=25,
                                                     detectShadows=True)
        k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))  # noqa: E731
        self.k_open, self.k_close, self.k_dil = k(open_px), k(close_px), k(dilate_px)
        self.keep = background_keep

    def mask(self, frame: np.ndarray) -> np.ndarray:
        m = self.bg.apply(frame)
        m = np.where(m == 255, 255, 0).astype(np.uint8)    # fuera sombras (127)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k_close)
        return cv2.dilate(m, self.k_dil)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        m = self.mask(frame)
        if not m.any():
            return frame                    # sin primer plano: no tocar nada
        out = (frame * self.keep).astype(np.uint8)
        out[m > 0] = frame[m > 0]
        return out
