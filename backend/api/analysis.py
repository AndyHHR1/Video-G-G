"""Rutas de análisis de VIDEO EN VIVO.

El sistema solo trabaja con vídeo en vivo: no acepta vídeos ni imágenes
subidos (el análisis de fotografías vive en el módulo aparte `modulos/imagen`).

| Ruta          | Uso                                                     |
|---------------|---------------------------------------------------------|
| /api/frame    | fotograma JPEG de la webcam del navegador (en vivo)     |
| /api/stream   | MJPEG de la cámara conectada al SERVIDOR (o RTSP)       |
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, request

from ..rendering.draw import (annotate_status, encode_jpeg, normalise_boxes,
                              normalise_skeletons, summarise)
from ..services.video_io import open_capture
from .context import runtime

bp = Blueprint("analysis", __name__)


def _arg(name: str, default, cast):
    raw = request.args.get(name, request.form.get(name))
    try:
        return cast(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _body(dets, result, timing, **extra) -> dict:
    """JSON común de las respuestas en vivo (una sola fuente de verdad)."""
    body = {
        "detections": dets,
        "summary": summarise(dets),
        "riesgo": result["riesgo"],
        "postura": result["postura"],
        "obstaculos": result["obstaculos"],
        "alertas": result["alertas"],
        "tracks": result["tracks"],
        "sistema": result["sistema"],
        "timing": {**timing, "stages": result.get("stages", {})},
        "frame_id": result.get("frame_id"),
        "esqueletos": result.get("esqueletos", []),
    }
    body.update(extra)
    return body


# --------------------------------------------------------------------------- #
@bp.post("/api/frame")
def live_frame():
    """Fotograma de la cámara en vivo del navegador.

    El cuerpo es el JPEG crudo (`Content-Type: image/jpeg`): evita el parser
    multipart de Werkzeug, que costaba ~17 ms por fotograma. Parámetros en la
    query: `conf`, `overlay` (0 = el navegador dibuja las cajas), `sr`
    (1 = devolver la imagen mejorada con Real-ESRGAN) y `fid` (id del
    fotograma, que se devuelve como `frame_id`).
    """
    rt = runtime()
    raw = None
    if "file" in request.files:
        raw = request.files["file"].read()
    elif request.mimetype == "image/jpeg":
        raw = request.get_data(cache=False)
    if not raw:
        return jsonify({"error": "no se ha enviado ningún frame"}), 400
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "no se pudo decodificar el frame"}), 400

    conf = _arg("conf", rt.settings.inference.conf, float)
    overlay = _arg("overlay", "1", str) not in ("0", "false")
    want_sr = _arg("sr", "0", str) == "1"
    fid = request.args.get("fid")

    if overlay:
        # Modo clásico: el servidor devuelve el fotograma ya anotado.
        annotated, result, dets, timing = rt.inferencer.detect(
            image, conf, draw=True, fid=fid)
        annotate_status(annotated, dets, 0.0, result)
        return jsonify(_body(dets, result, timing,
                             image=encode_jpeg(annotated, quality=72).hex()))

    # Modo en vivo: detección asíncrona sobre el fotograma NATIVO; la mejora
    # de calidad corre en paralelo en su propio hilo (LiveEnhancer).
    enhancer = rt.live_enhancer
    if want_sr and enhancer is not None and enhancer.available:
        enhancer.submit(image, fid)
    got = rt.inferencer.detect(image, conf, draw=False, fid=fid, wait=False)
    if got is None:
        return jsonify({"detections": [], "summary": summarise([]),
                        "riesgo": "BAJO", "timing": {"predict": 0, "draw": 0},
                        "frame_id": None, "pendiente": True})
    annotated, result, dets, timing = got
    h, w = annotated.shape[:2]
    body = _body(normalise_boxes(dets, w, h), result, timing, size=[w, h],
                 esqueletos=normalise_skeletons(result.get("esqueletos", []), w, h))
    body["summary"] = summarise(dets)
    if want_sr and enhancer is not None and enhancer.available:
        sr_b64 = enhancer.get(result.get("frame_id"))
        if sr_b64:
            body["image"] = sr_b64
            body["timing"]["sr"] = enhancer.ms
    return jsonify(body)


# --------------------------------------------------------------------------- #
def _mjpeg(inferencer, source, conf: float, attempts: int, stopping):
    """Stream MJPEG de la cámara del servidor, con reconexión (RQNF13).

    Recibe el inferencer por parámetro: el generador se consume fuera del
    contexto de la petición, donde `current_app` ya no existe. Termina en
    cuanto se pide detener el sistema (`stopping`)."""
    cap = open_capture(source, attempts=1)
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0
    try:
        while cap is not None and not stopping():
            ok, frame = cap.read()
            if not ok:
                cap.release()
                inferencer.engine.camera_connected = False
                cap = open_capture(source, attempts=attempts)
                continue
            annotated, result, dets, _t = inferencer.detect(frame, conf)
            fps_n += 1
            elapsed = time.perf_counter() - fps_t
            if elapsed >= 0.5:
                fps, fps_n, fps_t = fps_n / elapsed, 0, time.perf_counter()
            annotate_status(annotated, dets, fps, result)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + encode_jpeg(annotated, quality=80) + b"\r\n")
    finally:
        if cap is not None:
            cap.release()


@bp.get("/api/stream")
def stream():
    rt = runtime()
    conf = _arg("conf", rt.settings.inference.conf, float)
    source = request.args.get("cam", rt.settings.camera.source)
    probe = open_capture(source, attempts=1)
    if probe is None:
        return jsonify({
            "error": "El servidor no tiene cámara. Usa el modo «Cámara de "
                     "este equipo», que usa la webcam de tu navegador, o "
                     "configura camera.source en configs/app.yaml."}), 503
    probe.release()
    lifecycle = rt.lifecycle
    return Response(_mjpeg(rt.inferencer, source, conf,
                           rt.settings.camera.reconnect_attempts,
                           lambda: lifecycle is not None and lifecycle.stopping),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
