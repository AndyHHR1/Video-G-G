"""Rutas de análisis: imagen, fotograma en vivo, vídeo y cámara del servidor.

| Ruta          | Uso                                                     |
|---------------|---------------------------------------------------------|
| /api/detect   | imagen fija (multipart `file`)                          |
| /api/frame    | fotograma JPEG de la webcam del navegador (en vivo)     |
| /api/video    | vídeo completo; devuelve el MP4 anotado                 |
| /api/stream   | MJPEG de la cámara conectada al SERVIDOR (o RTSP)       |
"""
from __future__ import annotations

import io
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, request, send_file

from ..rendering.draw import (annotate_status, encode_b64, encode_jpeg,
                              normalise_boxes, summarise)
from ..services.video_io import open_capture, transcode_h264
from .context import runtime

bp = Blueprint("analysis", __name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _arg(name: str, default, cast):
    raw = request.args.get(name, request.form.get(name))
    try:
        return cast(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _upload_sr(kind: str) -> str | None:
    """Modelo SR a aplicar en una subida, o None si la mejora está apagada."""
    rt = runtime()
    up = rt.settings.enhancement.uploads
    name = up.image_model if kind == "image" else up.video_model
    return name if name in rt.inferencer.enhancers else None


def build_response(dets, result, timing, **extra):
    """JSON común de todas las rutas de análisis (una sola fuente de verdad)."""
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
    }
    body.update(extra)
    return jsonify(body)


# --------------------------------------------------------------------------- #
@bp.post("/api/detect")
def detect_image():
    """Imagen fija. No genera alertas: una foto no es un suceso en el tiempo."""
    rt = runtime()
    if "file" not in request.files:
        return jsonify({"error": "no se ha enviado ningún archivo"}), 400
    raw = request.files["file"].read()
    if not raw:
        return jsonify({"error": "archivo vacío"}), 400
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "no se pudo decodificar la imagen"}), 400
    conf = _arg("conf", rt.settings.inference.conf, float)

    # Cada imagen es una escena nueva: no debe heredar caches de la anterior.
    rt.inferencer.reset()
    t0 = time.perf_counter()
    annotated, result, dets, timing = rt.inferencer.detect(
        image, conf, draw=True, force=True, allow_alerts=False,
        sr_model=_upload_sr("image"))
    timing["total"] = round((time.perf_counter() - t0) * 1000, 1)
    t_enc = time.perf_counter()
    payload = encode_jpeg(annotated).hex()
    timing["encode"] = round((time.perf_counter() - t_enc) * 1000, 1)
    return build_response(dets, result, timing,
                          inference_ms=timing["predict"],
                          size=[int(annotated.shape[1]), int(annotated.shape[0])],
                          image=payload)


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
            image, conf, draw=True, fid=fid,
            sr_model=_upload_sr("video") if want_sr else None)
        annotate_status(annotated, dets, 0.0, result)
        return build_response(dets, result, timing,
                              image=encode_jpeg(annotated, quality=72).hex())

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
    body = {
        "detections": normalise_boxes(dets, w, h),
        "summary": summarise(dets),
        "riesgo": result["riesgo"],
        "postura": result["postura"],
        "obstaculos": result["obstaculos"],
        "alertas": result["alertas"],
        "tracks": result["tracks"],
        "sistema": result["sistema"],
        "timing": {**timing, "stages": result.get("stages", {})},
        "frame_id": result.get("frame_id"),
        "size": [w, h],
    }
    if want_sr and enhancer is not None and enhancer.available:
        sr_b64 = enhancer.get(result.get("frame_id"))
        if sr_b64:
            body["image"] = sr_b64
            body["timing"]["sr"] = enhancer.ms
    return jsonify(body)


@bp.post("/api/video")
def process_video():
    """Procesa un vídeo completo y devuelve el MP4 anotado (con su audio)."""
    rt = runtime()
    if "file" not in request.files:
        return jsonify({"error": "no se ha enviado ningún archivo"}), 400
    upload = request.files["file"]
    raw = upload.read()
    if not raw:
        return jsonify({"error": "archivo vacío"}), 400
    conf = _arg("conf", rt.settings.inference.conf, float)

    suffix = Path(upload.filename or "entrada.mp4").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        suffix = ".mp4"
    tmpdir = Path(tempfile.gettempdir())
    src = tmpdir / f"sub_{uuid.uuid4().hex}{suffix}"
    raw_out = tmpdir / f"raw_{uuid.uuid4().hex}.mp4"
    final = tmpdir / f"out_{uuid.uuid4().hex}.mp4"
    src.write_bytes(raw)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        src.unlink(missing_ok=True)
        return jsonify({"error": "no se pudo decodificar el video"}), 400
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sr_model = _upload_sr("video")
    writer, frames, counts = None, 0, Counter()
    rt.inferencer.reset()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, result, dets, _t = rt.inferencer.detect(
                frame, conf, draw=True, sr_model=sr_model)
            frames += 1
            counts.update(x.get("class_name") or x.get("tipo", "?") for x in dets)
            if writer is None:
                # El tamaño se toma del PRIMER fotograma anotado: con SR es
                # mayor que el original.
                size = (annotated.shape[1], annotated.shape[0])
                writer = cv2.VideoWriter(str(raw_out),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, size)
                if not writer.isOpened():
                    return jsonify({"error": "no se pudo preparar la salida "
                                             f"de video ({size[0]}×{size[1]})"}), 400
            writer.write(annotated)
            if frames % 200 == 0:
                print(f"  {frames} frames...", flush=True)
        cap.release()
        if writer is not None:
            writer.release()
        ok, err = transcode_h264(raw_out, final, audio_source=src)
        if not ok:
            print(f"  [aviso] ffmpeg falló ({err}); se devuelve el intermedio")
        chosen = final if final.is_file() else raw_out
        payload = chosen.read_bytes() if chosen.is_file() else b""
    finally:
        cap.release()
        for p in (src, raw_out, final):
            p.unlink(missing_ok=True)

    if frames == 0 or not payload:
        return jsonify({"error": "el video no contenía frames utilizables"}), 400
    return send_file(io.BytesIO(payload), mimetype="video/mp4",
                     as_attachment=True, download_name="resultado.mp4")


# --------------------------------------------------------------------------- #
def _mjpeg(inferencer, source, conf: float, attempts: int):
    """Stream MJPEG de la cámara del servidor, con reconexión (RQNF13).

    Recibe el inferencer por parámetro: el generador se consume fuera del
    contexto de la petición, donde `current_app` ya no existe."""
    cap = open_capture(source, attempts=1)
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0
    try:
        while cap is not None:
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
    return Response(_mjpeg(rt.inferencer, source, conf,
                           rt.settings.camera.reconnect_attempts),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
