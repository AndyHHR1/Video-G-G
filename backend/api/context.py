"""Estado compartido por las rutas: configuración y servicios cargados."""
from __future__ import annotations

from dataclasses import dataclass, field

from flask import current_app

from ..config import Settings


@dataclass
class Runtime:
    settings: Settings
    inferencer: object            # services.inferencer.Inferencer
    live_enhancer: object | None  # services.live_enhancer.LiveEnhancer
    panel_token: str = ""
    lifecycle: object | None = None      # services.lifecycle.Lifecycle
    # Módulos aparte cargados (nombre -> estado) y fuentes de alertas extra
    # que registran (p. ej. las alertas acústicas). El vídeo no los importa.
    modules: dict = field(default_factory=dict)
    alert_sources: list = field(default_factory=list)


def runtime() -> Runtime:
    return current_app.extensions["escaleras"]
