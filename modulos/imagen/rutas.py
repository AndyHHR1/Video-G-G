"""Rutas HTTP del módulo Imagen.

Basado en `app/imagen_pruebas.py` del repositorio del grupo de imagen
(AndreCastAst/Repo-video-imagen-audio, rama s7-imagen). Copia original en
`referencia/`.

| Ruta                             | Uso                                              |
|----------------------------------|--------------------------------------------------|
| POST /api/imagen/pruebas/frame   | fotografía o fotograma de cámara (multipart `frame`, `modo`) |
| GET  /api/imagen/caracteristicas | vector de características (`imagen-0.1.0`) del último fotograma del VÍDEO EN VIVO |
| GET  /api/imagen/estado          | modelos cargados y dispositivo                   |
| GET  /imagen/pruebas             | página de pruebas aislada del grupo              |

El módulo usa SUS propios modelos (YOLO11n y YOLO11n-Pose): nunca llama al
motor de vídeo. `caracteristicas` solo LEE el último resultado del vídeo en
vivo (no reinicia su seguimiento ni encola trabajo en su hilo), que era lo que
en el repo original obligaba a no alternar imagen y vídeo.
"""
from __future__ import annotations

import base64
import threading

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request

MAX_BYTES = 7 * 1024 * 1024
MAX_PIXELS = 12_000_000
MODOS = ("original", "adaptativo", "clahe", "gamma")

bp = Blueprint("imagen", __name__)
# YOLO de Ultralytics no es seguro entre hilos: un análisis a la vez.
_ANALISIS = threading.Lock()


@bp.get("/imagen/pruebas")
def pagina_imagen():
    return render_template("imagen/pruebas.html")


@bp.post("/api/imagen/pruebas/frame")
def frame_imagen():
    archivo = request.files.get("frame")
    if archivo is None:
        return jsonify({"estado": "error", "mensaje": "Falta frame"}), 400
    if request.content_length and request.content_length > MAX_BYTES:
        return jsonify({"estado": "error", "mensaje": "Fotografía demasiado grande"}), 413
    contenido = archivo.read(MAX_BYTES + 1)
    if len(contenido) > MAX_BYTES:
        return jsonify({"estado": "error", "mensaje": "Fotografía demasiado grande"}), 413
    frame = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"estado": "error", "mensaje": "Archivo no es una imagen válida"}), 400
    if frame.shape[0] * frame.shape[1] > MAX_PIXELS:
        return jsonify({"estado": "error", "mensaje": "Resolución demasiado grande"}), 413
    modo = request.form.get("modo", "adaptativo")
    if modo not in MODOS:
        return jsonify({"estado": "error", "mensaje": "Modo inválido"}), 400
    try:
        from .modulo_imagen.analisis_tiempo_real import analizar_frame
        with _ANALISIS:
            datos, anotada = analizar_frame(frame, modo=modo)
        ok, jpg = cv2.imencode(".jpg", anotada, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            datos["imagen_anotada"] = ("data:image/jpeg;base64,"
                                       + base64.b64encode(jpg).decode("ascii"))
        return jsonify(datos)
    except Exception as exc:                                # noqa: BLE001
        # Solo el tipo de error: no se exponen rutas internas ni trazas.
        current_app.logger.exception("Fallo en inferencia de Imagen")
        return jsonify({"estado": "error", "mensaje": "Fallo de análisis",
                        "tipo": type(exc).__name__}), 500


@bp.get("/api/imagen/caracteristicas")
def caracteristicas_video():
    """`build_features` (contrato imagen-0.1.0) sobre el último fotograma que
    analizó el vídeo en vivo. Solo lectura: no altera el vídeo."""
    from .modulo_imagen.features import build_features
    rt = current_app.extensions.get("escaleras")
    inf = getattr(rt, "inferencer", None)
    last = inf.last_result() if inf is not None else None
    if not last:
        return jsonify({"estado": "sin_datos",
                        "mensaje": "Aún no hay fotogramas del vídeo en vivo. "
                                   "Conecta la cámara en la pestaña «Cámara en vivo»."}), 404
    frame, result = last[0], last[1]
    return jsonify(build_features(frame, result, source="video_en_vivo",
                                  frame_id=result.get("frame_id")))


@bp.get("/api/imagen/estado")
def estado():
    from .modulo_imagen import modelos
    return jsonify({"modelos_cargados": modelos.cargados(),
                    "dispositivo": modelos.dispositivo() or "auto",
                    "modos": list(MODOS)})
