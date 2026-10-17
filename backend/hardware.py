"""Detección de la GPU y perfil de baja VRAM (GTX 1650 y similares).

El sistema se desarrolló en una RTX 5070 (8 GB), pero debe funcionar en una
GTX 1650 (Turing TU117, `sm_75`, 4 GB, sin Tensor Cores). La build de PyTorch
del proyecto (`cu128`) ya trae kernels para `sm_75`, así que no hace falta otra
instalación: basta con ajustar lo que en esa tarjeta cuesta demasiado.

Qué cambia el perfil `baja_vram` (todo editable en `configs/app.yaml`):

* **FP16 desactivado en Real-ESRGAN.** Las GTX 16xx tienen fallos conocidos
  con media precisión (salidas NaN / imágenes negras en varias redes); en FP32
  la red ligera sigue siendo usable.
* **Entrada más pequeña a la mejora en vivo** (`enhancement.live.input_max`):
  el coste de SRVGGNetCompact crece con los píxeles de entrada; 320 px en
  lugar de 480 px es ~2,2× menos trabajo y deja GPU libre para la detección.
* La detección NO se toca: el detector S7 v4 se entrenó a 640 px y bajar la
  resolución cambiaría su precisión (el modelo no se reentrena).

El perfil se elige con `hardware.profile`: `auto` (por defecto) lo activa si
la GPU tiene poca VRAM o es una GTX 16xx; `normal` y `baja_vram` lo fuerzan.
"""
from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

LOW_TIER_NAMES = ("GTX 16", "GTX 10", "MX4", "MX5", "T400", "T600", "T1000")


def gpu_info() -> dict:
    """Nombre, VRAM y capacidad de cómputo de la GPU 0 (o vacío sin CUDA)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        p = torch.cuda.get_device_properties(0)
        return {"name": p.name,
                "vram_gb": round(p.total_memory / 2 ** 30, 2),
                "capability": f"{p.major}.{p.minor}",
                "arch_ok": f"sm_{p.major}{p.minor}" in torch.cuda.get_arch_list()}
    except Exception:                                       # noqa: BLE001
        return {}


def choose_profile(profile: str, info: dict, low_vram_gb: float) -> str:
    profile = str(profile or "auto").lower()
    if profile in ("normal", "baja_vram"):
        return profile
    if not info:
        return "normal"                 # CPU: el perfil de GPU no aplica
    if info["vram_gb"] <= low_vram_gb:
        return "baja_vram"
    if any(tag in info["name"].upper() for tag in LOW_TIER_NAMES):
        return "baja_vram"
    return "normal"


def _set_dotted(obj, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not is_dataclass(obj) or not hasattr(obj, parts[-1]):
        raise KeyError(dotted)
    setattr(obj, parts[-1], value)


def _get_dotted(obj, dotted: str):
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def apply_profile(settings, info: dict | None = None) -> tuple[str, dict]:
    """Aplica el perfil de hardware a `settings` (in situ).

    Devuelve `(perfil, cambios)` donde `cambios` es {clave: (antes, después)}.
    """
    hw = settings.hardware
    info = gpu_info() if info is None else info
    name = choose_profile(hw.profile, info, hw.low_vram_gb)
    changes: dict = {}
    if name == "baja_vram" and settings.device != "cpu":
        for key, value in (hw.baja_vram or {}).items():
            try:
                before = _get_dotted(settings, key)
                _set_dotted(settings, key, value)
            except (AttributeError, KeyError):
                print(f"  [aviso] hardware.baja_vram: clave desconocida {key!r}")
                continue
            if before != value:
                changes[key] = (before, value)
    hw.active = name
    hw.gpu = info
    return name, changes
