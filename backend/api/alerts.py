"""Rutas del panel de alertas (RQF08, RQNF15, RQNF16).

Las alertas viven en `outputs/alerts/` (configurable): una carpeta por día con
las imágenes de evidencia y un `alertas.jsonl` con una línea por alerta. Así
el panel sobrevive a un reinicio del servidor.

Los módulos aparte pueden aportar sus propias alertas (p. ej. las acústicas de
`modulos/audio`) registrando una *fuente* en `Runtime.alert_sources`: un objeto
con `read() -> list[dict]`, `delete(ids | None) -> int` y `zip_name`. Este
archivo no importa ningún módulo: solo mezcla lo que le registren.
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from .context import runtime

bp = Blueprint("alerts", __name__)

ZIP_README = (
    "Paquete de alertas del sistema de prevención de caídas en escaleras.\n"
    "Las imágenes están anotadas con las detecciones y los rostros\n"
    "pixelados (RQNF14, Ley N.º 29733).\n"
    "alertas.jsonl lleva una línea por alerta con fecha, clase,\n"
    "severidad, confianza, duración e identidad de seguimiento.\n"
)


def _dir() -> Path:
    return runtime().settings.alerts_dir


def _read_log() -> list[dict]:
    log = _dir() / "alertas.jsonl"
    if not log.is_file():
        return []
    out = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _inside(base: Path, target: Path) -> bool:
    """True si `target` está dentro de `base` (evita rutas tipo ../../)."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _sort_key(a: dict) -> float:
    """Marca de tiempo comparable: el vídeo escribe hora local sin zona y el
    audio ISO-8601 en UTC; ambas se pasan a segundos desde epoch."""
    try:
        return datetime.fromisoformat(str(a.get("timestamp", ""))
                                      .replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _extra_rows() -> list[dict]:
    rows = []
    for src in runtime().alert_sources:
        try:
            rows.extend(src.read())
        except Exception:                                   # noqa: BLE001
            continue
    return rows


@bp.get("/api/alerts")
def list_alerts():
    items = []
    for a in _read_log():
        ev = Path(a.get("evidence") or "")
        exists = bool(a.get("evidence")) and ev.is_file()
        a["imagen"] = (f"/api/alerta-imagen/{ev.parent.name}/{ev.name}"
                       if exists else None)
        a.setdefault("source", "video")
        items.append(a)
    items.extend(_extra_rows())
    items.sort(key=_sort_key, reverse=True)
    return jsonify({"alertas": items[:100], "total": len(items)})


@bp.delete("/api/alertas")
def delete_alerts():
    """Sin cuerpo borra TODAS; con `{"ids": [...]}` solo las elegidas.
    Imagen y línea de registro se borran siempre juntas (RQNF15)."""
    d = _dir()
    body = request.get_json(silent=True) or {}
    ids = body.get("ids") or request.args.getlist("id") or None
    source = body.get("source") or request.args.get("source")  # "audio" = solo esa
    extra_deleted = 0
    for src in runtime().alert_sources:
        if source and getattr(src, "name", "") != source:
            continue
        try:
            extra_deleted += src.delete(ids)
        except Exception:                                   # noqa: BLE001
            pass
    if source and source != "video":
        return jsonify({"borradas": "todas" if not ids else extra_deleted,
                        "total": len(_read_log()) + len(_extra_rows())})

    if not ids:
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        (d / "alertas.jsonl").touch()
        return jsonify({"borradas": "todas", "total": 0})

    deleted, kept = 0, []
    for a in _read_log():
        if a.get("alert_id") not in ids:
            kept.append(a)
            continue
        ev = Path(a.get("evidence") or "")
        if a.get("evidence") and _inside(d, ev) and ev.is_file():
            ev.unlink()
        deleted += 1
    d.mkdir(parents=True, exist_ok=True)
    with (d / "alertas.jsonl").open("w", encoding="utf-8") as fh:
        for a in kept:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    for sub in d.iterdir():
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()
    return jsonify({"borradas": deleted + extra_deleted,
                    "total": len(kept) + len(_extra_rows())})


@bp.get("/api/alerta-imagen/<dia>/<nombre>")
def alert_image(dia: str, nombre: str):
    base = _dir()
    target = base / dia / nombre
    if not _inside(base, target) or not target.is_file():
        return jsonify({"error": "no encontrada"}), 404
    return send_file(str(target.resolve()), mimetype="image/jpeg")


@bp.get("/api/alertas.zip")
def alerts_zip():
    """Todas las alertas (imágenes + registro) en un ZIP generado en memoria."""
    d = _dir()
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if d.is_dir():
            for f in sorted(d.glob("*/*.jpg")):
                z.write(f, arcname=f"{f.parent.name}/{f.name}")
                n += 1
            log = d / "alertas.jsonl"
            if log.is_file():
                z.write(log, arcname="alertas.jsonl")
        for src in runtime().alert_sources:
            try:
                rows = src.read()
            except Exception:                               # noqa: BLE001
                continue
            if rows:
                z.writestr(getattr(src, "zip_name", "alertas_extra.jsonl"), "".join(
                    json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                n += len(rows)
        z.writestr("README.txt", ZIP_README)
    if n == 0:
        return jsonify({"error": "aun no hay alertas registradas"}), 404
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="alertas.zip")
