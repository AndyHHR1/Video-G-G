"""Carga perezosa de los modelos del módulo Imagen (YOLO11n y YOLO11n-Pose).

En el repositorio original `detector.py` y `pose.py` cargaban los pesos al
importarse (y Ultralytics los descargaba de internet si faltaban). Aquí se
cargan en el PRIMER uso, desde `models/imagen/`, para que el módulo no ocupe
VRAM ni retrase el arranque del vídeo en vivo (importante en una GTX 1650 de
4 GB). Las funciones de análisis del grupo no cambian.
"""
from __future__ import annotations

import threading
from pathlib import Path

_CFG = {"detector": "yolo11n.pt", "pose": "yolo11n-pose.pt", "device": None}
_MODELS: dict = {}
_LOCK = threading.Lock()


def configurar(detector: str | Path | None = None, pose: str | Path | None = None,
               device=None) -> None:
    """Fija rutas y dispositivo. `device`: None/'auto', 'cpu', 'cuda', 0..."""
    if detector:
        _CFG["detector"] = str(detector)
    if pose:
        _CFG["pose"] = str(pose)
    _CFG["device"] = None if device in (None, "auto") else device
    liberar()


def dispositivo():
    return _CFG["device"]


def _cargar(clave: str):
    with _LOCK:
        if clave not in _MODELS:
            from ultralytics import YOLO
            _MODELS[clave] = YOLO(_CFG[clave])
        return _MODELS[clave]


def modelo_deteccion():
    return _cargar("detector")


def modelo_pose():
    return _cargar("pose")


def cargados() -> list[str]:
    return sorted(_MODELS)


def liberar() -> None:
    """Suelta los modelos (parada del sistema o cambio de configuración)."""
    with _LOCK:
        _MODELS.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass
