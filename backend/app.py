"""Fábrica de la aplicación Flask.

Une las capas del sistema:

    frontend/templates  ──►  api/ (Flask blueprints)
                               │
                               ▼
                         services/ (hilo de inferencia, SR en vivo, vídeo)
                               │
                               ▼
                         perception/ + enhancement/ + privacy/
"""
from __future__ import annotations

import os

from flask import Flask

from .api import alerts, analysis, auth, system
from .api.context import Runtime
from .config import ROOT, Settings
from .perception.engine import resolve_device
from .services.inferencer import Inferencer, sr_enabled
from .services.live_enhancer import LiveEnhancer

FRONTEND_DIR = ROOT / "frontend"


def create_app(settings: Settings) -> Flask:
    """Carga modelos y servicios y devuelve la app lista para servir."""
    _check_models(settings)

    app = Flask(__name__,
                template_folder=str(FRONTEND_DIR / "templates"),
                static_folder=str(FRONTEND_DIR / "static"))
    app.config["MAX_CONTENT_LENGTH"] = settings.server.max_upload_mb * 1024 * 1024
    app.config["TEMPLATES_AUTO_RELOAD"] = settings.server.debug

    print("  cargando modelos en el hilo de inferencia...", flush=True)
    inferencer = Inferencer(settings)
    live = None
    if sr_enabled(settings):
        print("  cargando Real-ESRGAN para la cámara en vivo...", flush=True)
        live = LiveEnhancer(settings, resolve_device(settings.device))

    app.extensions["escaleras"] = Runtime(
        settings=settings, inferencer=inferencer, live_enhancer=live,
        panel_token=os.environ.get("PANEL_TOKEN", ""))

    for bp in (auth.bp, system.bp, analysis.bp, alerts.bp):
        app.register_blueprint(bp)
    return app


def _check_models(settings: Settings) -> None:
    missing = [p for p in (settings.detector_path, settings.coco_path)
               if not p.is_file()]
    if missing:
        lines = "\n".join(f"    - {p}" for p in missing)
        raise SystemExit(
            f"Faltan pesos de modelo:\n{lines}\n"
            "  Descárgalos con:  python tools/download_models.py")
