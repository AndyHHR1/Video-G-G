"""Rutas de sistema: página principal, salud y cámaras del servidor."""
from __future__ import annotations

import cv2
from flask import Blueprint, jsonify, render_template

from ..perception.catalog import CLASS_INFO
from .context import runtime

bp = Blueprint("system", __name__)


@bp.get("/")
def index():
    return render_template("index.html", classes=CLASS_INFO)


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
        esrgan = f"activo ({live.model_name}, {'cuda' if live._sr.cuda else 'cpu'})"
    elif live is not None and live.error:
        esrgan = f"desactivado ({live.error.splitlines()[0]})"
    else:
        esrgan = "desactivado (enhancement.enabled)"
    s = rt.settings
    return jsonify({
        "ok": True,
        "estado": detail.get("estado", "DESCONOCIDO"),
        "detalle": detail,
        "weights": str(s.models.detector),
        "device": _device_label(inf.device),
        "classes": len(CLASS_INFO),
        "auth": "activo" if rt.panel_token else "desactivado (PANEL_TOKEN sin definir)",
        "esrgan": esrgan,
        "esrgan_uploads": sorted(inf.enhancers),
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
