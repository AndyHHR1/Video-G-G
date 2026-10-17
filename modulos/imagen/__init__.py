"""Módulo de IMAGEN — análisis de fotografías y fotogramas.

Procede del repositorio del grupo de imagen
(https://github.com/AndreCastAst/Repo-video-imagen-audio, rama `s7-imagen`) y
REEMPLAZA a la antigua pestaña «Imagen» del panel, que solo pasaba una foto por
el motor de vídeo.

Funciones (todas activas):

* Calidad óptica (luminosidad, contraste, nitidez, sombras, ruido, zonas
  oscuras / sobreexpuestas).
* Preprocesamiento adaptativo (modos original, adaptativo, CLAHE, gamma).
* YOLO11n: personas y celulares; YOLO11n-Pose: 17 articulaciones e
  inclinación del torso; asociación pose-persona por IoU, poses sin YOLO,
  proximidad celular-mano y estados posturales «para revisión».
* Análisis de fotografía subida y de fotogramas de la cámara del navegador.
* Vector de características `imagen-0.1.0` del último fotograma del vídeo en
  vivo (solo lectura).
* Bibliotecas de clip (`analizador_clip`) y recuperación en ROI, usadas por
  los scripts del grupo (`modulos/imagen/scripts/`).

Independencia del vídeo: usa sus propios modelos, cargados en el primer uso,
y nunca encola trabajo en el hilo de inferencia del vídeo.
"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint

HERE = Path(__file__).resolve().parent


def install(app, settings, rt, options: dict) -> dict:
    from .modulo_imagen import modelos
    from .rutas import bp

    det = settings.path(options.get("detector", "models/imagen/yolo11n.pt"))
    pose = settings.path(options.get("pose", "models/imagen/yolo11n-pose.pt"))
    missing = [str(p) for p in (det, pose) if not p.is_file()]
    if missing:
        raise FileNotFoundError("faltan pesos " + ", ".join(missing)
                                + " (python tools/download_models.py)")
    device = options.get("device", "auto")
    if device == "auto" and settings.device == "cpu":
        device = "cpu"
    modelos.configurar(det, pose, device)

    app.register_blueprint(bp)
    app.register_blueprint(Blueprint(
        "imagen_ui", __name__, static_folder=str(HERE / "static"),
        static_url_path="/modulos/imagen/static",
        template_folder=str(HERE / "templates")))
    if rt.lifecycle is not None:
        rt.lifecycle.on_shutdown("módulo de imagen", modelos.liberar)
    return {"ok": True,
            "estado": f"activo · YOLO11n + YOLO11n-Pose ({device}, carga en el primer uso)"}
