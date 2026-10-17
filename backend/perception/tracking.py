"""Seguimiento de personas por solapamiento IoU (RQF05).

Se usa un tracker propio y no ByteTrack de Ultralytics: ByteTrack arrastra
estado interno entre llamadas de forma frágil y obliga a construirlo con un
objeto de argumentos. Aquí la asignación es explícita y corta:

1. empareja cada detección con la identidad viva de mayor IoU (> umbral),
2. crea una identidad nueva para cada detección sin pareja,
3. olvida las identidades que llevan más de `occlusion_seconds` sin verse.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TrackState:
    """Estado temporal de una identidad seguida (RQF05 / RQF06)."""
    track_id: int
    last_seen: float
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    risk_since: dict = field(default_factory=dict)   # riesgo -> inicio
    risk_seen: set = field(default_factory=set)      # {(tid, riesgo)}
    risk_visto: dict = field(default_factory=dict)   # riesgo -> última vez visto
    alerts: dict = field(default_factory=dict)       # riesgo -> última alerta


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_match: float = 0.25, occlusion_seconds: float = 1.0):
        self.iou_match = iou_match
        self.occlusion_seconds = occlusion_seconds
        self.tracks: dict[int, TrackState] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        """Asigna `track_id` a cada detección (in place) y la devuelve."""
        now = time.time()

        # 1) asociación con las identidades vivas, mejor IoU primero
        pairs = []
        for tid, st in self.tracks.items():
            if now - st.last_seen > self.occlusion_seconds:
                continue
            for det in detections:
                if det["track_id"] is not None:
                    continue
                v = iou(st.bbox, det["bbox"])
                if v > self.iou_match:
                    pairs.append((v, tid, det))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_t, used_d = set(), set()
        for _v, tid, det in pairs:
            if tid in used_t or id(det) in used_d:
                continue
            used_t.add(tid)
            used_d.add(id(det))
            det["track_id"] = tid

        # 2) identidades nuevas: un id propio por detección
        next_id = max(self.tracks, default=0) + 1
        for det in detections:
            if det["track_id"] is None:
                det["track_id"] = next_id
                next_id += 1

        # 3) refrescar estado y olvidar identidades perdidas
        for det in detections:
            tid = det["track_id"]
            st = self.tracks.get(tid)
            if st is None:
                st = self.tracks[tid] = TrackState(tid, now)
            st.last_seen = now
            st.bbox = det["bbox"]
        for tid, st in list(self.tracks.items()):
            if now - st.last_seen > self.occlusion_seconds:
                del self.tracks[tid]
        return detections
