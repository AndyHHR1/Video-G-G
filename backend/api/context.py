"""Estado compartido por las rutas: configuración y servicios cargados."""
from __future__ import annotations

from dataclasses import dataclass

from flask import current_app

from ..config import Settings


@dataclass
class Runtime:
    settings: Settings
    inferencer: object            # services.inferencer.Inferencer
    live_enhancer: object | None  # services.live_enhancer.LiveEnhancer
    panel_token: str = ""


def runtime() -> Runtime:
    return current_app.extensions["escaleras"]
