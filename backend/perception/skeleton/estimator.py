"""YOLOv8-Pose (Ultralytics) con seguimiento de identidad.

* 17 keypoints COCO por persona, varias personas por fotograma.
* Identidad estable con ByteTrack (`model.track(persist=True)`): usa un filtro
  de Kalman, así que sostiene el ID durante oclusiones cortas (el pasamanos, una
  persona que cruza delante) mejor que un emparejamiento por IoU puro. Sus
  parámetros están en `configs/tracker_esqueletos.yaml`.
* Fusión de esqueletos partidos: un pasamanos horizontal puede cortar a una
  persona en dos detecciones (medido con un pasamanos sintético: 4 personas se
  convirtieron en 6). Si una detección solo tiene cuerpo superior y otra, justo
  debajo y alineada, solo cuerpo inferior, se unen en un único esqueleto.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .keypoints import LOWER, N_KP, UPPER


@dataclass
class PoseObs:
    track_id: int | None
    bbox: list
    conf: float
    xy: np.ndarray                        # (17, 2) píxeles
    kconf: np.ndarray                     # (17,)
    merged: bool = False
    extra: dict = field(default_factory=dict)


class PoseEstimator:
    def __init__(self, model_path: str, device, *, imgsz=640, conf=0.30,
                 iou=0.6, tracker="bytetrack.yaml", kp_conf_min=0.45,
                 min_keypoints=5, half=False):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.tracker = tracker
        self.kp_conf_min = kp_conf_min
        self.min_kp = min_keypoints
        self.half = half

    def reset(self) -> None:
        """Olvida las identidades (otra escena / reinicio del motor)."""
        pred = getattr(self.model, "predictor", None)
        for trk in getattr(pred, "trackers", None) or []:
            trk.reset()

    def __call__(self, frame: np.ndarray) -> list[PoseObs]:
        r = self.model.track(frame, persist=True, tracker=self.tracker,
                             imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                             device=self.device, half=self.half,
                             verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []
        xy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(xy.shape[:2])
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id
        ids = ids.int().cpu().tolist() if ids is not None else [None] * len(boxes)
        obs = [PoseObs(ids[i], [round(float(v), 1) for v in boxes[i]],
                       float(confs[i]), xy[i].astype(float), kc[i].astype(float))
               for i in range(len(boxes))]
        obs = merge_split(obs, self.kp_conf_min)
        return [o for o in obs if int((o.kconf >= self.kp_conf_min).sum()) >= self.min_kp
                or o.merged]


def _has(o: PoseObs, idx, thr) -> int:
    return int((o.kconf[idx] >= thr).sum())


def merge_split(obs: list[PoseObs], thr: float) -> list[PoseObs]:
    """Une las mitades superior e inferior de una persona cortada por el
    pasamanos (u otro objeto horizontal)."""
    upper = [o for o in obs if _has(o, UPPER, thr) >= 3 and _has(o, LOWER, thr) <= 2]
    lower = [o for o in obs if _has(o, LOWER, thr) >= 2 and _has(o, UPPER, thr) <= 1]
    used: set[int] = set()
    out_extra: list[PoseObs] = []
    for u in upper:
        best, best_gap = None, None
        ux1, uy1, ux2, uy2 = u.bbox
        uw, uh = ux2 - ux1, uy2 - uy1
        for lo in lower:
            if id(lo) in used:
                continue
            lx1, ly1, lx2, ly2 = lo.bbox
            overlap = min(ux2, lx2) - max(ux1, lx1)
            if overlap < 0.4 * min(uw, lx2 - lx1):
                continue                          # no están alineadas
            gap = ly1 - uy2
            if gap < -0.5 * uh or gap > 0.6 * uh:
                continue                          # no están una sobre otra
            if best is None or abs(gap) < best_gap:
                best, best_gap = lo, abs(gap)
        if best is None:
            continue
        used.add(id(best))
        used.add(id(u))
        xy, kc = u.xy.copy(), u.kconf.copy()
        take = best.kconf > kc
        xy[take], kc[take] = best.xy[take], best.kconf[take]
        bx = [min(u.bbox[0], best.bbox[0]), min(u.bbox[1], best.bbox[1]),
              max(u.bbox[2], best.bbox[2]), max(u.bbox[3], best.bbox[3])]
        out_extra.append(PoseObs(u.track_id if u.track_id is not None else best.track_id,
                                 bx, max(u.conf, best.conf), xy, kc, merged=True))
    keep = [o for o in obs if id(o) not in used]
    return keep + out_extra


__all__ = ["PoseEstimator", "PoseObs", "merge_split", "N_KP"]
