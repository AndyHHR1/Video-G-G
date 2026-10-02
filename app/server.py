#!/usr/bin/env python3
"""
Plataforma web local para probar el detector de riesgo en escaleras.

Carga el checkpoint de Ultralytics y expone tres modos de prueba:

  * imagen  -> sube un frame y recibe el resultado anotado
  * video   -> sube un video y descarga la version anotada
  * camara  -> webcam en vivo por streaming MJPEG

Las detecciones se acompanan de un panel de riesgo que traduce cada clase a
la categoria de riesgo de `Caso.md` (RQF02), de forma que la prueba sirva
para validar el diseno y no solo la precision del detector.

Uso:
    ./venv/bin/python app/server.py
    # abrir http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import hmac
import io
import os
import queue
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, Response, jsonify, redirect, render_template, request, send_file,
)

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Descripcion de cada clase, enlazada con los requisitos de `Caso.md`.
# `color` es BGR (como espera OpenCV).
# --------------------------------------------------------------------------- #
CLASS_INFO = {
    0: {
        "name": "Persona caída",
        "color": (60, 60, 235),        # rojo
        "risk": "ALTO",
        "req": "RQF02 (4) · RQNF05",
        "desc": "Persona tendida en el suelo tras una caída.",
    },
    1: {
        "name": "Postura no ergonómica",
        "color": (60, 170, 245),       # naranja
        "risk": "MEDIO",
        "req": "RQF03",
        "desc": "Postura sentada o agachada; no corresponde a tránsito de pie.",
    },
    2: {
        "name": "Postura erguida",
        "color": (90, 200, 90),        # verde
        "risk": "BAJO",
        "req": "RQF03",
        "desc": "Tránsito seguro: persona de pie o caminando.",
    },
    3: {
        "name": "Escalera",
        "color": (200, 120, 200),      # morado
        "risk": "CONTEXTO",
        "req": "—",
        "desc": "Tramo de escalera detectado en la escena.",
    },
    4: {
        "name": "Persona",
        "color": (220, 220, 220),      # blanco
        "risk": "NEUTRO",
        "req": "RQNF08",
        "desc": "Persona genérica. Es la entrada para el análisis de pose (MediaPipe).",
    },
    5: {
        "name": "Pérdida de equilibrio",
        "color": (40, 190, 255),       # amarillo
        "risk": "ALTO",
        "req": "RQF02 (4)",
        "desc": "Centro de gravedad inestable: pre-caída, antes de caer.",
    },
}

RISK_ORDER = {"ALTO": 3, "MEDIO": 2, "BAJO": 1, "NEUTRO": 0, "CONTEXTO": 0}

try:
    from risk_engine import RISK_COLOR
except Exception:                      # pragma: no cover
    RISK_COLOR = {"ALTO": (60, 60, 235), "MEDIO": (60, 170, 245),
                  "BAJO": (90, 200, 90), "CONTEXTO": (200, 120, 200)}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024   # 512 MB

# El modelo se carga una sola vez y se protege con un lock: Flask atiende en
# varios hilos y dos inferencias simultaneas sobre la misma GPU pueden fallar.
class Inferencer:
    """Ejecuta inferencia Y dibujo en un unico hilo trabajador persistente.

    Por que no se hace directamente en el hilo de cada peticion: Flask atiende
    cada peticion en un hilo distinto (`threaded=True`, necesario para que el
    stream de la webcam no bloquee al resto), y en un hilo recien creado hay que
    reenlazar el contexto CUDA. Medido en esta maquina:

        inferencia   ~29 ms en hilo ya enlazado  ->  ~82 ms en hilo nuevo
        dibujo cv2    0.13 ms en hilo ya enlazado ->  ~23 ms en hilo nuevo

    Total: ~85 ms por peticion frente a ~27 ms con este hilo unico. El hilo se
    crea una vez al arrancar y attend todas las peticiones.
    """

    def __init__(self, weights: str, device: int = 0,
                 obstacle_weights: str = "yolov8s.pt", conf: float = 0.25):
        self._weights = weights
        self._device = device
        self._obstacle_weights = obstacle_weights
        self._conf = conf
        self._q: queue.Queue = queue.Queue()
        self._model = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()          # no se atiende hasta tener modelo

    def _run(self) -> None:
        from ultralytics import YOLO
        # el modelo se carga DENTRO del hilo: asi el contexto CUDA se crea y
        # queda ligado a este mismo hilo
        self._model = YOLO(self._weights)
        self._model.to(self._device)
        # RQF02 (1)(2)(3): motor de riesgo completo. Se construye aqui tambien
        # para que MediaPipe y el detector de obstaculos nauten en el hilo
        # enlazado, que es donde rinde bien.
        from risk_engine import RiskEngine
        self._engine = RiskEngine(
            self._weights, obstacle_weights=self._obstacle_weights,
            device=self._device, conf=self._conf, enable_pose=True,
            anonymize=True,
        )
        self._ready.set()
        while True:
            job = self._q.get()
            if job is None:                       # señal de cierre
                return
            image, conf, imgsz, out = job
            try:
                out["value"] = self._job(image, conf, imgsz)
            except Exception as exc:              # noqa: BLE001
                out["error"] = exc
            finally:
                out["done"].set()

    def _job(self, image, conf, imgsz):
        """Motor completo + dibujo, ambos dentro del hilo trabajador.

        El dibujo tambien se hace aqui a proposito: medir en esta maquina
        muestra que cualquier operacion de OpenCV ejecutada desde un hilo
        recien creado cuesta ~23 ms frente a 0.13 ms en el hilo ya enlazado.
        """
        if conf != self._conf:
            self._conf = conf
            self._engine.conf = conf
        t0 = time.perf_counter()
        result = self._engine.step(image)
        t_predict = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        out = image.copy()
        detections = draw_all(out, result)
        t_draw = (time.perf_counter() - t1) * 1000
        return out, result, detections, {
            "predict": round(t_predict, 1), "draw": round(t_draw, 1),
        }

    def detect(self, image: np.ndarray, conf: float, imgsz: int):
        """Encola el trabajo y espera el resultado."""
        out = {"done": threading.Event()}
        self._q.put((image, conf, imgsz, out))
        out["done"].wait()
        if "error" in out:
            raise out["error"]
        return out["value"]

    def shutdown(self) -> None:
        self._q.put(None)


_inferencer = None
_settings = {"conf": 0.25, "imgsz": 640, "weights": "", "names": {}}


# --------------------------------------------------------------------------- #
# Codificacion
# --------------------------------------------------------------------------- #
def draw_all(frame: np.ndarray, result: dict) -> list[dict]:
    """Dibuja el resultado completo del motor de riesgo sobre el fotograma.

    `detecciones`   cajas del modelo propio (persona / escalera)
    `obstaculos`    objetos COCO sobre la escalera   -> RQF02 (1)
    `postura`       señales de keypoints             -> RQF02 (2)(3)
    `alertas`       historial de alertas confirmadas -> RQF07
    """
    out = []
    for d in result.get("detecciones", []):
        info = CLASS_INFO.get(
            d["class_id"],
            {"name": d["class_name"], "color": (200, 200, 200), "risk": "NEUTRO"},
        )
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), info["color"], 2)
        tid = d.get("track_id")
        label = f'#{tid} {info["name"]} {d["confidence"]:.0%}' if tid else \
                f'{info["name"]} {d["confidence"]:.0%}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ty = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty), info["color"], -1)
        cv2.putText(frame, label, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (20, 20, 20), 2, cv2.LINE_AA)
        out.append({**d, "class_name": info["name"], "risk": info["risk"]})

    for o in result.get("obstaculos", []):
        x1, y1, x2, y2 = o["bbox"]
        col = RISK_COLOR["ALTO"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f'OBSTACULO {o["objeto"]} {o["conf"]:.0%}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1),
                      col, -1)
        cv2.putText(frame, label, (x1 + 4, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
        out.append({"tipo": o["tipo"], "objeto": o["objeto"],
                    "confidence": o["conf"], "risk": "ALTO",
                    "bbox": o["bbox"], "ref": "RQF02 (1)"})

    for s in result.get("postura", []):
        col = RISK_COLOR.get(s["riesgo"], (220, 220, 220))
        label = f'{s["tipo"].upper()} {s["conf"]:.0%} [{s["ref"]}]'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y = frame.shape[0] - 6 - th
        cv2.rectangle(frame, (6, y - th - 6), (6 + tw + 8, y), col, -1)
        cv2.putText(frame, label, (10, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (20, 20, 20), 2, cv2.LINE_AA)
        out.append({"tipo": s["tipo"], "confidence": s["conf"],
                    "risk": s["riesgo"], "ref": s["ref"],
                    "detalle": s.get("detalle", "")})
    return out


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("no se pudo codificar la imagen")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Resumen de riesgo
# --------------------------------------------------------------------------- #
def summarise(detections: list[dict]) -> dict:
    """Traduce las detecciones a un veredicto de riesgo segun `Caso.md`."""
    # `detections` mezcla cajas de YOLO (con `class_name`), obstaculos y
    # señales de postura (que traen `tipo` en su lugar). Antes se accedia
    # `class_name` a pelo y reventaba con KeyError en cuanto habia un obstaculo.
    counts = Counter(d.get("class_name") or d.get("tipo", "?")
                     for d in detections)
    risks = {d["risk"] for d in detections
             if d.get("risk") not in ("CONTEXTO", "NEUTRO", None)}
    level = max(risks, key=lambda r: RISK_ORDER.get(r, 0), default="BAJO")

    notes = {
        "ALTO": "Se ha detectado una condición de caída o pérdida de equilibrio. "
                "Requiere alerta inmediata (RQF06).",
        "MEDIO": "Postura no ergonómica detectada. Conviene revisar (RQF04).",
        "BAJO": "No se detectan conductas de riesgo en la escena.",
    }
    return {
        "n": len(detections),
        "level": level,
        "riesgo": level,
        "counts": dict(counts),
        "message": notes.get(level, notes["BAJO"]),
    }


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template("index.html", classes=CLASS_INFO)


@app.post("/api/detect")
def api_detect():
    if "file" not in request.files:
        return jsonify({"error": "no se ha enviado ningún archivo"}), 400
    raw = request.files["file"].read()
    if not raw:
        return jsonify({"error": "archivo vacío"}), 400

    conf = float(request.form.get("conf", _settings["conf"]))
    imgsz = int(request.form.get("imgsz", _settings["imgsz"]))

    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "no se pudo decodificar la imagen"}), 400

    t0 = time.perf_counter()
    annotated, result, dets, timing = _inferencer.detect(image, conf, imgsz)
    t_all = (time.perf_counter() - t0) * 1000
    t_enc = time.perf_counter()
    payload = encode_jpeg(annotated).hex()
    timing["encode"] = round((time.perf_counter() - t_enc) * 1000, 1)
    timing["total"] = round(t_all, 1)

    return jsonify({
        "detections": dets,
        "summary": summarise(dets),
        "riesgo": result["riesgo"],
        "postura": result["postura"],
        "obstaculos": result["obstaculos"],
        "alertas": result["alertas"],
        "tracks": result["tracks"],
        "sistema": result["sistema"],
        "inference_ms": timing["predict"],
        "timing": timing,
        "size": [int(image.shape[1]), int(image.shape[0])],
        "image": payload,
    })


@app.post("/api/frame")
def api_frame():
    """Inferencia sobre un unico frame JPEG.

    Es el endpoint que usa el modo «cámara de este equipo»: el navegador
    captura con `getUserMedia`, pinta el frame en un canvas y lo envia aqui
    frame a frame. Hace falta porque la webcam del usuario esta en su
    navegador, no en esta maquina (que no tiene ninguna: no existe
    /dev/video*). El stream MJPEG de /api/stream solo sirve si el servidor
    tiene camara fisica.
    """
    if "file" not in request.files:
        return jsonify({"error": "no se ha enviado ningún frame"}), 400
    raw = request.files["file"].read()
    if not raw:
        return jsonify({"error": "frame vacío"}), 400

    conf = float(request.form.get("conf", _settings["conf"]))
    imgsz = int(request.form.get("imgsz", _settings["imgsz"]))
    overlay = request.form.get("overlay", "1") not in ("0", "false")

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "no se pudo decodificar el frame"}), 400

    annotated, result, dets, timing = _inferencer.detect(image, conf, imgsz)
    if overlay:
        annotate_status(annotated, dets, 0.0, result)
        return jsonify({
            "detections": dets,
            "summary": summarise(dets),
            "riesgo": result["riesgo"],
            "postura": result["postura"],
            "obstaculos": result["obstaculos"],
            "alertas": result["alertas"],
            "tracks": result["tracks"],
            "sistema": result["sistema"],
            "timing": timing,
            "image": encode_jpeg(annotated, quality=72).hex(),
        })

    # Modo `overlay=0`: se devuelven SOLO las cajas y el navegador dibuja
    # sobre el frame que ya tiene. Evita codificar y reenviar ~50 KB de JPEG
    # por frame, que era el cuello de botella: la inferencia son 16 ms pero el
    # ciclo completo llegaba a 45 ms.
    h, w = image.shape[:2]
    return jsonify({
        "detections": [
            {**d, "bbox": [
                round(d["bbox"][0] / w, 4), round(d["bbox"][1] / h, 4),
                round(d["bbox"][2] / w, 4), round(d["bbox"][3] / h, 4),
            ]}
            for d in dets
        ],
        "summary": summarise(dets),
        "riesgo": result["riesgo"],
        "postura": result["postura"],
        "obstaculos": result["obstaculos"],
        "alertas": result["alertas"],
        "tracks": result["tracks"],
        "sistema": result["sistema"],
        "timing": timing,
        "size": [w, h],
    })


def annotate_status(frame: np.ndarray, dets: list[dict], fps: float = 0.0,
                    result: dict | None = None) -> None:
    """Superpone la banda de estado: FPS, nº de detecciones y nivel de riesgo.

    Si se pasa el resultado del motor, el veredicto es el suyo, que ya
    incorpora postura y obstaculos; si no, se deduce solo de las cajas.
    """
    summary = summarise(dets)
    if result:
        summary["n"] = len(dets) + len(result.get("obstaculos", [])) \
            + len(result.get("postura", []))
        summary["level"] = result["riesgo"]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (30, 30, 30), -1)
    color = {
        "ALTO": (60, 60, 235), "MEDIO": (60, 170, 245),
        "BAJO": (90, 200, 90), "NEUTRO": (90, 200, 90),
    }.get(summary["level"], (200, 200, 200))
    cv2.putText(frame, f'FPS: {fps:.1f}   detecciones: {summary["n"]}',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                1, cv2.LINE_AA)
    cv2.putText(frame, f'Riesgo: {summary["level"]}',
                (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


@app.get("/api/cameras")
def api_cameras():
    """Informa de si el SERVIDOR tiene camara.

    La webcam del usuario la usa el navegador, asi que esto no la restringe:
    solo evita que se ofrezca un modo de camara que no puede funcionar.
    """
    found = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append({"index": i,
                          "name": cap.getBackendName() if hasattr(cap, "getBackendName") else f"cam {i}"})
        cap.release()
    return jsonify({"server_has_camera": bool(found), "devices": found})


def _transcode_h264(src: Path, dst: Path) -> tuple[bool, str]:
    """Recodifica a H.264 con el ffmpeg empaquetado en imageio-ffmpeg.

    `libx264` con `+faststart` deja el indice al principio del archivo, que es
    lo que permite empezar a reproducir antes de que termine de descargarse.
    """
    try:
        import subprocess

        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        r = subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
            capture_output=True, text=True, timeout=600,
        )
        return (r.returncode == 0 and dst.is_file()
                and dst.stat().st_size > 0, r.stderr[-200:])
    except Exception as exc:       # noqa: BLE001
        return False, str(exc)


@app.post("/api/video")
def api_video():
    if "file" not in request.files:
        return jsonify({"error": "no se ha enviado ningún archivo"}), 400
    raw = request.files["file"].read()
    if not raw:
        return jsonify({"error": "archivo vacío"}), 400

    conf = float(request.form.get("conf", _settings["conf"]))
    imgsz = int(request.form.get("imgsz", _settings["imgsz"]))

    # OpenCV 5 escribe siempre a disco: no acepta un buffer en memoria ni
    # como fuente (VideoCapture) ni como destino (VideoWriter). Se usan dos
    # temporales, uno para la entrada y otro para la salida, y se limpian al
    # terminar.
    suffix = Path(request.files["file"].filename or "entrada.mp4").suffix
    if suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        suffix = ".mp4"
    tmp = Path(tempfile.gettempdir()) / f"sub_{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(raw)

    cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        tmp.unlink(missing_ok=True)
        return jsonify({"error": "no se pudo decodificar el video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Codec de salida.
    #
    # OpenCV escribe `mp4v` (MPEG-4 Part 2) y los navegadores NO lo reproducen:
    # sale un reproductor con boton de play y pantalla negra.
    #
    # Este build de OpenCV (4.10, fijado por el conflicto con MediaPipe) no
    # trae ningun codificador que el navegador acepte: VP8 y VP9 abren el
    # VideoWriter pero escriben archivo vacio, y H.264 no existe. La solucion
    # es escribir con OpenCV y recodificar con el ffmpeg que trae
    # `imageio-ffmpeg`, que si incluye libx264.
    out_tmp = Path(tempfile.gettempdir()) / f"raw_{uuid.uuid4().hex}.mp4"
    final_tmp = Path(tempfile.gettempdir()) / f"out_{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(
        str(out_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        cap.release()
        tmp.unlink(missing_ok=True)
        return jsonify({"error": "no se pudo preparar la salida de video"}), 400

    total_counts, frames, t_infer = Counter(), 0, 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            annotated, result, _d, _t = _inferencer.detect(frame, conf, imgsz)
            t_infer += time.perf_counter() - t0
            frames += 1
            total_counts.update(x.get("class_name") or x.get("tipo", "?")
                                 for x in _d)
            total_counts.update(o["tipo"] for o in result.get("obstaculos", []))
            writer.write(annotated)
            if frames % 200 == 0:
                print(f"  {frames} frames...", flush=True)

        cap.release()
        writer.release()
        # OpenCV deja MPEG-4 Part 2, que ningun navegador reproduce: se
        # recodifica a H.264, que si es universal.
        ok, err = _transcode_h264(out_tmp, final_tmp)
        if not ok:
            print(f"  [aviso] ffmpeg fallo ({err}); se devuelve el intermedio")
        payload = (final_tmp if final_tmp.is_file() else out_tmp)
        payload = payload.read_bytes() if payload.is_file() else b""
    finally:
        tmp.unlink(missing_ok=True)
        out_tmp.unlink(missing_ok=True)
        final_tmp.unlink(missing_ok=True)

    if frames == 0 or not payload:
        return jsonify({"error": "el video no contenía frames utilizables"}), 400

    return send_file(
        io.BytesIO(payload), mimetype="video/mp4", as_attachment=True,
        download_name="resultado.mp4",
    )


def _mjpeg(camera_index: int, conf: float, imgsz: int):
    """Generador del stream de la webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        yield b""
        return
    
    fps_t, fps_n = time.perf_counter(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, result, _d, _t = _inferencer.detect(frame, conf, imgsz)
            fps_n += 1
            elapsed = time.perf_counter() - fps_t
            if elapsed >= 0.5:
                fps = fps_n / elapsed
                fps_n, fps_t = 0, time.perf_counter()

            annotate_status(annotated, dets, fps, result)

            frame_bytes = encode_jpeg(annotated, quality=80)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/api/stream")
def api_stream():
    conf = float(request.args.get("conf", _settings["conf"]))
    imgsz = int(request.args.get("imgsz", _settings["imgsz"]))
    cam = int(request.args.get("cam", 0))
    probe = cv2.VideoCapture(cam)
    opened = probe.isOpened()
    probe.release()
    if not opened:
        return jsonify({
            "error": "El servidor no tiene cámara. Usa el modo "
                     "«Cámara de este equipo», que usa la webcam de tu navegador."
        }), 503
    return Response(
        _mjpeg(cam, conf, imgsz),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------------------------------------------------- #
# Autenticacion (RQNF19)
#
# `Caso.md` exige "autenticación de usuario para acceso al panel". Se
# implementa con un secreto compartido que se compara en tiempo constante.
# El secreto NUNCA se escribe en el repositorio: se lee de la variable de
# entorno PANEL_TOKEN y, si no esta, el panel arranca en modo local sin
# autenticacion y lo dice en pantalla en lugar de fallar en silencio.
# --------------------------------------------------------------------------- #
PANEL_TOKEN = os.environ.get("PANEL_TOKEN", "")
AUTH_ON = bool(PANEL_TOKEN)
TOKEN_TTL = 8 * 3600


def _is_public_request() -> bool:
    """El stream y las imagenes de la propia pagina no exigen token.

    Se aceptan solo peticiones desde el propio navegador de la maquina, que es
    el escenario de desarrollo; cualquier otro origen va con token.
    """
    return request.remote_addr in ("127.0.0.1", "::1")


@app.before_request
def _guard():
    if not AUTH_ON:
        return None
    if request.path in ("/", "/login", "/api/health"):
        return None
    if request.path.startswith("/static/"):
        return None
    if request.path == "/api/stream" and _is_public_request():
        return None
    supplied = request.headers.get("X-Panel-Token", "") or request.cookies.get("panel", "")
    if supplied and hmac.compare_digest(supplied, PANEL_TOKEN):
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "token de panel requerido (RQNF19)"}), 401
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_ON:
        return redirect("/")
    if request.method == "POST":
        sent = request.form.get("token", "")
        if sent and hmac.compare_digest(sent, PANEL_TOKEN):
            resp = redirect("/")
            resp.set_cookie("panel", PANEL_TOKEN, httponly=True,
                            max_age=TOKEN_TTL, samesite="Lax")
            return resp
        return render_template("login.html", error="Token incorrecto"), 401
    return render_template("login.html", error=None)


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "weights": _settings["weights"],
        "device": "cuda:0" if _cv_cuda() else "cpu",
        "classes": len(CLASS_INFO),
        "auth": "activo" if AUTH_ON else "desactivado (PANEL_TOKEN sin definir)",
    })


def _cv_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/yolov8s_zoom/weights/best.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--obstacle-weights", default="yolov8s.pt",
                    help="Modelo COCO para RQF02 (1): obstaculos en la escalera")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.is_file():
        raise SystemExit(f"No se encuentra el checkpoint: {weights}")

    import yaml
    data_yaml = ROOT / "datasets/combined/data.yaml"
    names = {}
    if data_yaml.is_file():
        raw = yaml.safe_load(data_yaml.read_text()).get("names", {})
        names = {int(k): v for k, v in raw.items()} if isinstance(raw, dict) else dict(enumerate(raw))
    _settings.update(weights=str(weights), conf=args.conf, imgsz=args.imgsz, names=names)

    print("=" * 60)
    print("Plataforma de pruebas - Prevencion de Caidas en Escaleras")
    print("=" * 60)
    print(f"  pesos   : {weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights}")
    print(f"  clases  : {len(names)} ({', '.join(names.values())})")
    print(f"  conf    : {args.conf}   imgsz: {args.imgsz}")
    print(f"  URL     : http://{args.host}:{args.port}")
    print("=" * 60)

    global _inferencer
    print("  cargando pesos en el hilo trabajador...")
    _inferencer = Inferencer(
        str(weights), conf=args.conf,
        obstacle_weights=args.obstacle_weights,
    )
    print("  modelo listo")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()