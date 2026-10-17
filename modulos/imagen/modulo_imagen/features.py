"""Contrato básico de Imagen (v0.1). Coordenadas de cajas en píxeles."""
from datetime import datetime, timezone
import math
import cv2
import numpy as np


def quality_metrics(frame):
    """Medidas ópticas, no probabilidades de accidente."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("Se requiere un fotograma BGR no vacío")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "luminosidad_media_0_255": round(float(np.mean(gray)), 2),
        "contraste_desviacion": round(float(np.std(gray)), 2),
        "nitidez_laplaciano_varianza": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


def build_features(frame, result, *, source="desconocida", frame_id=None, timestamp=None):
    """Adapta el resultado existente de RiskEngine; jamás inventa clases o coordenadas."""
    h, w = frame.shape[:2]
    entities = []
    for d in result.get("detecciones", []):
        bb = d.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        try:
            coords = [float(v) for v in bb]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in coords):
            continue
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            continue
        entities.append({
            "clase": str(d.get("class_name", "desconocida")),
            "confianza": d.get("confidence"),
            "bbox_xyxy_px": [round(v, 2) for v in coords],
            "track_id": d.get("track_id"),
            "riesgo_reportado_por_video": d.get("riesgo"),
            "origen": d.get("origen"),
        })
    signals = []
    for s in result.get("postura", []):
        if isinstance(s, dict):
            signals.append({k: s[k] for k in ("tipo", "conf", "riesgo", "detalle", "desc") if k in s})
    obstacles = []
    for o in result.get("obstaculos", []):
        if isinstance(o, dict):
            obstacles.append({k: o[k] for k in ("tipo", "conf", "bbox", "riesgo") if k in o})
    return {
        "schema_version": "imagen-0.1.0",
        "metadata_captura": {
            "fecha_hora_utc": timestamp or datetime.now(timezone.utc).isoformat(),
            "fuente": source, "frame_id": frame_id,
            "ancho_px": w, "alto_px": h,
        },
        "vector_caracteristicas": {
            "calidad_optica": quality_metrics(frame),
            "entidades": entities, "senales_pose": signals,
            "obstaculos": obstacles, "numero_personas_detectadas": sum(
                1 for e in entities if e["clase"].lower() in
                ("persona", "persona caída", "postura erguida", "postura no ergonómica", "pérdida de equilibrio")
            ),
        },
        "diagnostico_descriptivo": {
            "riesgo_reportado_por_video": result.get("riesgo"),
            "nota": "Caracteristicas visuales; no equivale a prediccion de accidente."
        },
    }
