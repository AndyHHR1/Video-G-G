"""Dibujo de resultados y codificación de imágenes.

Convierte la salida del motor (`RiskEngine.step`) en:
* un fotograma anotado (cajas, etiquetas, señales de postura), y
* la lista plana de detecciones que consume la interfaz web.
"""
from __future__ import annotations

import base64
from collections import Counter

import cv2
import numpy as np

from ..perception.catalog import CLASS_INFO, RISK_COLOR, RISK_ORDER

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_INK = (20, 20, 20)


def _label(frame, text: str, x: int, y: int, color) -> None:
    """Etiqueta con fondo de color cuya esquina inferior izquierda es (x, y)."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 2)
    y = max(y, th + 6)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 5), _FONT, 0.5, _INK, 2, cv2.LINE_AA)


def _obstacle_style(o: dict) -> tuple[tuple, str, bool]:
    es_riesgo = o.get("riesgo") == "ALTO"
    if o.get("tipo") == "telefono":
        return RISK_COLOR["MEDIO"], "TELEFONO", es_riesgo
    if es_riesgo:
        return RISK_COLOR["ALTO"], "OBSTACULO", True
    return RISK_COLOR["CONTEXTO"], "ESCENA", False


def draw_all(frame: np.ndarray, result: dict) -> list[dict]:
    """Dibuja el resultado completo sobre `frame` (in place).

    Devuelve las detecciones enriquecidas para la interfaz.
    """
    out = []
    for d in result.get("detecciones", []):
        info = CLASS_INFO.get(d["class_id"], {"name": d["class_name"],
                                              "color": (200, 200, 200),
                                              "risk": "NEUTRO"})
        # El color sigue el riesgo REAL de la persona (que puede venir de una
        # señal de postura), no solo el de su clase.
        riesgo = d.get("riesgo") or info["risk"]
        color = RISK_COLOR.get(riesgo, info["color"])
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        parts = [f'#{d["track_id"]}'] if d.get("track_id") else []
        parts.append(info["name"])
        if d.get("senales"):
            parts.append("+".join(s.replace("_", " ") for s in d["senales"]))
        parts.append(f'{d["confidence"]:.0%}')
        _label(frame, " ".join(parts), x1, y1, color)
        out.append({**d, "class_name": info["name"], "risk": riesgo,
                    "color": color})

    for o in result.get("obstaculos", []):
        color, prefix, es_riesgo = _obstacle_style(o)
        x1, y1, x2, y2 = o["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _label(frame, f'{prefix} {o["objeto"]} {o["conf"]:.0%}', x1, y1, color)
        out.append({"tipo": o["tipo"], "objeto": o["objeto"],
                    "confidence": o["conf"],
                    "risk": "ALTO" if es_riesgo else "CONTEXTO",
                    "bbox": o["bbox"],
                    "ref": "RQF02 (1)" if es_riesgo else "escena"})

    for s in result.get("postura", []):
        color = RISK_COLOR.get(s["riesgo"], (220, 220, 220))
        text = f'{s["tipo"].upper()} {s["conf"]:.0%} [{s["ref"]}]'
        (_tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 2)
        _label(frame, text, 6, frame.shape[0] - 6 - th, color)
        out.append(_posture_item(s))
    return out


def _posture_item(s: dict) -> dict:
    return {"tipo": s["tipo"], "confidence": s["conf"], "risk": s["riesgo"],
            "ref": s["ref"], "detalle": s.get("detalle", "")}


def detections_only(result: dict) -> list[dict]:
    """Misma lista que `draw_all` pero sin dibujar (cámara en vivo: el
    navegador pinta las cajas sobre su propio lienzo)."""
    dets = [{**d,
             "class_name": CLASS_INFO.get(d["class_id"], {}).get("name", d["class_name"]),
             "risk": d.get("riesgo", "NEUTRO")}
            for d in result["detecciones"]]
    dets += [_posture_item(s) for s in result["postura"]]
    dets += [{**o, "confidence": o["conf"], "risk": o.get("riesgo", "CONTEXTO"),
              "ref": "RQF02 (1)"} for o in result["obstaculos"]]
    return dets


def summarise(detections: list[dict]) -> dict:
    """Veredicto de riesgo a partir de la lista de detecciones."""
    counts = Counter(d.get("class_name") or d.get("tipo", "?") for d in detections)
    risks = {d["risk"] for d in detections
             if d.get("risk") not in ("CONTEXTO", "NEUTRO", None)}
    level = max(risks, key=lambda r: RISK_ORDER.get(r, 0), default="BAJO")
    notes = {
        "ALTO": "Se ha detectado una condición de caída o pérdida de equilibrio. "
                "Requiere alerta inmediata (RQF06).",
        "MEDIO": "Postura no ergonómica detectada. Conviene revisar (RQF04).",
        "BAJO": "No se detectan conductas de riesgo en la escena.",
    }
    return {"n": len(detections), "level": level, "riesgo": level,
            "counts": dict(counts), "message": notes.get(level, notes["BAJO"])}


def annotate_status(frame: np.ndarray, dets: list[dict], fps: float = 0.0,
                    result: dict | None = None) -> None:
    """Banda superior con FPS, número de detecciones y nivel de riesgo."""
    summary = summarise(dets)
    if result:
        summary["n"] = (len(dets) + len(result.get("obstaculos", []))
                        + len(result.get("postura", [])))
        summary["level"] = result["riesgo"]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (30, 30, 30), -1)
    color = {"ALTO": RISK_COLOR["ALTO"], "MEDIO": RISK_COLOR["MEDIO"],
             "BAJO": RISK_COLOR["BAJO"], "NEUTRO": RISK_COLOR["BAJO"]
             }.get(summary["level"], (200, 200, 200))
    cv2.putText(frame, f'FPS: {fps:.1f}   detecciones: {summary["n"]}',
                (10, 20), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Riesgo: {summary["level"]}',
                (10, 39), _FONT, 0.55, color, 2, cv2.LINE_AA)


def normalise_boxes(dets: list[dict], w: int, h: int) -> list[dict]:
    """Cajas en píxeles -> coordenadas 0..1 (las señales sin caja se dejan)."""
    out = []
    for d in dets:
        if "bbox" in d:
            x1, y1, x2, y2 = d["bbox"]
            d = {**d, "bbox": [round(x1 / w, 4), round(y1 / h, 4),
                               round(x2 / w, 4), round(y2 / h, 4)]}
        out.append(d)
    return out


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("no se pudo codificar la imagen")
    return buf.tobytes()


def encode_b64(image: np.ndarray, quality: int = 82) -> str:
    """JPEG en base64 (lo que el navegador pinta directamente)."""
    return base64.b64encode(encode_jpeg(image, quality)).decode("ascii")
