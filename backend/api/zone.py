"""Zona de la escalera y pasamanos dibujada en el panel (perception/zone.py).

    GET    /api/zona   -> {"escalera": [[x, y], ...], "pasamanos": [[[x, y], ...]]}
    POST   /api/zona   <- el mismo JSON (coordenadas 0..1); se valida y se guarda
    DELETE /api/zona   -> borra la zona (el sistema vuelve a analizar todo)

Protegida por PANEL_TOKEN como el resto de /api (ver auth.py).
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..perception.zone import Zone
from .context import runtime

bp = Blueprint("zone", __name__)


def _engine():
    return getattr(runtime().inferencer, "engine", None)


@bp.get("/api/zona")
def get_zone():
    eng = _engine()
    return jsonify(eng.zone.to_dict() if eng is not None else Zone().to_dict())


@bp.post("/api/zona")
def save_zone():
    eng = _engine()
    if eng is None:
        return jsonify({"error": "el motor aún no está listo"}), 503
    try:
        zone = Zone.from_dict(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"error": f"zona no válida: {exc}"}), 400
    zone.save(eng.zone_path)
    eng.set_zone(zone)
    return jsonify({"ok": True, **zone.to_dict()})


@bp.delete("/api/zona")
def delete_zone():
    eng = _engine()
    if eng is None:
        return jsonify({"error": "el motor aún no está listo"}), 503
    try:
        eng.zone_path.unlink()
    except FileNotFoundError:
        pass
    eng.set_zone(Zone())
    return jsonify({"ok": True, **Zone().to_dict()})
