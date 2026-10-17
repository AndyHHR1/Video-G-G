"""Rutas de sistema: página principal, salud, cámaras y parada del sistema."""
from __future__ import annotations

import time

import cv2
from flask import Blueprint, jsonify, render_template, request

from ..perception.catalog import CLASS_INFO
from .context import runtime

bp = Blueprint("system", __name__)
LOOPBACK = ("127.0.0.1", "::1", "localhost")


@bp.before_app_request
def refuse_while_stopping():
    """Mientras el sistema se detiene no se acepta trabajo nuevo."""
    lc = runtime().lifecycle
    if lc is None or not lc.stopping:
        return None
    if request.path.startswith(("/api/system/", "/api/health")):
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "el sistema se está deteniendo",
                        "deteniendo": True}), 503
    return None


@bp.get("/")
def index():
    rt = runtime()
    return render_template("index.html", classes=CLASS_INFO,
                           modules={k: v.get("ok", False)
                                    for k, v in rt.modules.items()})


def _device_label(device) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            idx = device if isinstance(device, int) else 0
            return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    except Exception:                                       # noqa: BLE001
        pass
    return "cpu"


@bp.get("/api/health")
def health():
    """Estado real del sistema (RQNF09, RQNF23, RQNF26)."""
    rt = runtime()
    inf = rt.inferencer
    engine = getattr(inf, "engine", None)
    detail = engine.status_detail() if engine else {}
    live = rt.live_enhancer
    if live is not None and live.available:
        esrgan = f"activo ({live.model_name}, {'cuda' if live._sr.cuda else 'cpu'}" \
                 f"{', fp16' if live._sr.half else ', fp32'})"
    elif live is not None and live.error:
        esrgan = f"desactivado ({live.error.splitlines()[0]})"
    else:
        esrgan = "desactivado (enhancement.enabled)"
    s = rt.settings
    lc = rt.lifecycle
    return jsonify({
        "ok": True,
        "estado": "DETENIENDO" if lc is not None and lc.stopping
                  else detail.get("estado", "DESCONOCIDO"),
        "detalle": detail,
        "modo": "solo vídeo en vivo",
        "weights": str(s.models.detector),
        "device": _device_label(inf.device),
        "hardware": {"perfil": s.hardware.active, "gpu": s.hardware.gpu},
        "classes": len(CLASS_INFO),
        "auth": "activo" if rt.panel_token else "desactivado (PANEL_TOKEN sin definir)",
        "esrgan": esrgan,
        "modulos": rt.modules,
    })


@bp.get("/api/cameras")
def cameras():
    """Indica si el SERVIDOR tiene cámaras locales (índices 0-3)."""
    found = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append({"index": i, "name": cap.getBackendName()})
        cap.release()
    return jsonify({"server_has_camera": bool(found), "devices": found,
                    "configured_source": str(runtime().settings.camera.source)})


# --------------------------------------------------------------------------- #
@bp.get("/api/system/status")
def system_status():
    rt = runtime()
    lc = rt.lifecycle
    return jsonify({
        "deteniendo": bool(lc and lc.stopping),
        "detenido": bool(lc and lc.stopped),
        "motivo": lc.reason if lc else "",
        "activo_s": round(time.time() - lc.started_at, 1) if lc else None,
    })


@bp.post("/api/system/stop")
def system_stop():
    """Función interna para PARAR todo el proyecto (botón «Detener sistema»).

    Sin PANEL_TOKEN solo se acepta desde el propio equipo: con
    `--host 0.0.0.0` cualquiera de la red podría apagar el panel."""
    rt = runtime()
    if not rt.panel_token and request.remote_addr not in LOOPBACK:
        return jsonify({"error": "la parada remota exige PANEL_TOKEN; "
                                 "deténlo desde el propio equipo"}), 403
    lc = rt.lifecycle
    if lc is None:
        return jsonify({"error": "ciclo de vida no disponible"}), 500
    started = lc.shutdown("botón del panel", background=True)
    return jsonify({"ok": True, "deteniendo": True,
                    "ya_en_curso": not started}), 202
