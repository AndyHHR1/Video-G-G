"""Entrada de vídeo en vivo: cámara del servidor o flujo RTSP/HTTP."""
from __future__ import annotations

import time

import cv2


def parse_source(source):
    """'0' -> 0 (índice de cámara); cualquier otra cosa se deja como URL."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    return int(s) if s.isdigit() else s


def open_capture(source, attempts: int = 5) -> cv2.VideoCapture | None:
    """Abre una cámara o un flujo RTSP/HTTP reintentando con espera creciente
    (RQNF13: reconexión automática de la señal)."""
    src = parse_source(source)
    for i in range(max(1, attempts)):
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            return cap
        cap.release()
        time.sleep(min(1.0 * (i + 1), 8.0))
    return None
