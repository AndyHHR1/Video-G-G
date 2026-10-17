"""Fábrica de la aplicación Flask.

Une las capas del sistema:

    frontend/templates  ──►  api/ (Flask blueprints)
                               │
                               ▼
                         services/ (hilo de inferencia, SR en vivo, cámara,
                                    ciclo de vida)
                               │
                               ▼
                         perception/ + enhancement/ + privacy/

Los módulos aparte (`modulos/audio`, `modulos/imagen`) se cargan al final y
de forma aislada: registran sus propias rutas y, si fallan, el vídeo en vivo
sigue funcionando sin cambios.
"""
from __future__ import annotations

import importlib
import os

from flask import Flask

from .api import alerts, analysis, auth, system
from .api.context import Runtime
from .config import ROOT, Settings
from .hardware import apply_profile
from .perception.engine import resolve_device
from .services.inferencer import Inferencer, sr_enabled
from .services.lifecycle import Lifecycle
from .services.live_enhancer import LiveEnhancer

FRONTEND_DIR = ROOT / "frontend"
MODULES = ("audio", "imagen")


def create_app(settings: Settings, lifecycle: Lifecycle | None = None) -> Flask:
    """Carga modelos y servicios y devuelve la app lista para servir."""
    _check_models(settings)
    lifecycle = lifecycle or Lifecycle()

    profile, changes = apply_profile(settings)
    gpu = settings.hardware.gpu
    if gpu:
        print(f"  GPU      : {gpu['name']} · {gpu['vram_gb']} GB · "
              f"sm_{gpu['capability'].replace('.', '')}"
              + ("" if gpu.get("arch_ok", True) else
                 "  [aviso] esta build de PyTorch no trae kernels para ella"),
              flush=True)
    print(f"  perfil   : {profile}"
          + "".join(f"\n             {k}: {a} -> {b}"
                    for k, (a, b) in changes.items()), flush=True)

    app = Flask(__name__,
                template_folder=str(FRONTEND_DIR / "templates"),
                static_folder=str(FRONTEND_DIR / "static"))
    app.config["MAX_CONTENT_LENGTH"] = settings.server.max_upload_mb * 1024 * 1024
    app.config["TEMPLATES_AUTO_RELOAD"] = settings.server.debug

    print("  cargando modelos en el hilo de inferencia...", flush=True)
    inferencer = Inferencer(settings)
    lifecycle.on_shutdown("inferencia de vídeo", inferencer.shutdown)
    live = None
    if sr_enabled(settings):
        print("  cargando Real-ESRGAN para la cámara en vivo...", flush=True)
        live = LiveEnhancer(settings, resolve_device(settings.device))
        lifecycle.on_shutdown("mejora Real-ESRGAN en vivo", live.stop)

    rt = Runtime(settings=settings, inferencer=inferencer, live_enhancer=live,
                 panel_token=os.environ.get("PANEL_TOKEN", ""),
                 lifecycle=lifecycle)
    app.extensions["escaleras"] = rt

    for bp in (auth.bp, system.bp, analysis.bp, alerts.bp):
        app.register_blueprint(bp)

    _load_modules(app, settings, rt)
    return app


def _load_modules(app: Flask, settings: Settings, rt: Runtime) -> None:
    """Carga cada módulo de `modulos/` sin que un fallo afecte al vídeo."""
    for name in MODULES:
        cfg = getattr(settings.modules, name)
        if not cfg.enabled:
            rt.modules[name] = {"ok": False, "estado": "desactivado (configs/app.yaml)"}
            continue
        try:
            mod = importlib.import_module(f"modulos.{name}")
            rt.modules[name] = mod.install(app, settings, rt, dict(cfg.options or {}))
            print(f"  módulo {name:<7}: {rt.modules[name].get('estado', 'activo')}",
                  flush=True)
        except Exception as exc:                            # noqa: BLE001
            rt.modules[name] = {"ok": False,
                                "estado": f"no disponible ({type(exc).__name__}: {exc})"}
            print(f"  módulo {name:<7}: NO disponible — {exc}. "
                  "El vídeo en vivo sigue funcionando.", flush=True)


def _check_models(settings: Settings) -> None:
    missing = [p for p in (settings.detector_path, settings.coco_path)
               if not p.is_file()]
    if missing:
        lines = "\n".join(f"    - {p}" for p in missing)
        raise SystemExit(
            f"Faltan pesos de modelo:\n{lines}\n"
            "  Descárgalos con:  python tools/download_models.py")
