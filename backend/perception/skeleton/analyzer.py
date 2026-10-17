"""Orquesta el análisis de esqueletos: pose → filtrado/suavizado →
cinemática → equilibrio, por cada identidad seguida.

    fotograma ─► [morfología opcional] ─► PoseEstimator (YOLOv8-Pose + ByteTrack)
                                              │  17 puntos + track_id
                                              ▼
                 KeypointSmoother (conf mínima, saltos, oclusión, One Euro)
                                              ▼
                 Kinematics (tronco, cadera, cabeza, centro de masa vs pies)
                                              ▼
                 BalanceMonitor (votos + persistencia) ─► alerta
"""
from __future__ import annotations

import time

import numpy as np

from .balance import BalanceMonitor
from .filters import KeypointSmoother
from .keypoints import N_KP
from .kinematics import Kinematics


class _Track:
    def __init__(self, cfg, nadir):
        k, s = cfg.keypoints, cfg.smoothing
        self.smoother = KeypointSmoother(k.conf_min, k.hold_s, k.max_jump_torso,
                                         s.min_cutoff, s.beta, s.d_cutoff)
        self.kin = Kinematics(nadir, s.velocity_alpha, cfg.balance.baseline_s,
                              cfg.balance.feet_pad)
        self.last_seen = 0.0


class SkeletonAnalyzer:
    def __init__(self, cfg, device, estimator=None):
        self.cfg = cfg
        nadir = cfg.perspective.nadir
        self.nadir = None if nadir in (None, "", []) else [float(v) for v in nadir]
        self.estimator = estimator
        if self.estimator is None:
            from .estimator import PoseEstimator
            self.estimator = PoseEstimator(
                str(cfg.model_path), device, imgsz=cfg.imgsz, conf=cfg.conf,
                iou=cfg.iou, tracker=str(cfg.tracker_path),
                kp_conf_min=cfg.keypoints.conf_min,
                min_keypoints=cfg.min_keypoints)
        self.morph = None
        if str(cfg.morphology).lower() in ("mascara", "on", "true"):
            from .morphology import ForegroundMorphology
            self.morph = ForegroundMorphology()
        self.balance = BalanceMonitor(cfg.balance)
        self.tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        self.tracks.clear()
        self.balance = BalanceMonitor(self.cfg.balance)
        self.estimator.reset()

    def step(self, frame: np.ndarray, t: float | None = None) -> list[dict]:
        t = time.perf_counter() if t is None else t
        h, w = frame.shape[:2]
        src = self.morph(frame) if self.morph is not None else frame
        people = []
        for o in self.estimator(src):
            people.append(self._person(o, h, w, t))
        self._forget(t)
        return people

    # ------------------------------------------------------------------ #
    def _person(self, o, h: int, w: int, t: float) -> dict:
        nadir = None if self.nadir is None else \
            np.array([self.nadir[0] * w / h, self.nadir[1]])
        pts = o.xy / h                                     # isótropo: / altura
        out = {"id": o.track_id, "bbox": o.bbox, "conf": round(o.conf, 3),
               "unido": o.merged}
        if o.track_id is None:
            # Aún sin identidad confirmada: se dibuja, pero sin historia no hay
            # velocidades ni decisión temporal.
            valid = o.kconf >= self.cfg.keypoints.conf_min
            out.update(kps=_pack(o.xy, o.kconf, valid, np.zeros(N_KP, bool)),
                       metricas={}, equilibrio={"estado": "sin_id"})
            return out
        tr = self.tracks.get(o.track_id)
        if tr is None:
            tr = self.tracks[o.track_id] = _Track(self.cfg, nadir)
        tr.kin.nadir = nadir
        tr.last_seen = t
        x, valid, held = tr.smoother.update(pts, o.kconf, t, tr.kin.torso_len)
        m = tr.kin.update(x, valid, t)
        b = self.cfg.balance
        small = (m["torso"] or 0) * h < b.min_torso_px
        baseline = tr.kin.baseline_angle(b.baseline_min_samples)
        if small:
            eq = {"estado": "no_evaluable", "criterios": [], "motivo":
                  f"persona pequeña (torso < {b.min_torso_px} px)", "alerta": False}
        else:
            eq = self.balance.evaluate(o.track_id, m, baseline, t)
            # La línea base solo aprende del tránsito estable.
            if eq["estado"] == "estable" and m["tronco_deg"] is not None \
                    and (m["cadera_v"] or 0) < b.hip_down_speed:
                tr.kin.add_baseline(t, m["tronco_deg"])
        eq["linea_base_deg"] = None if baseline is None else round(baseline, 1)
        com = m.pop("com", None)
        m["torso_px"] = None if not m["torso"] else round(m["torso"] * h, 1)
        m.pop("torso", None)
        out.update(kps=_pack(x * h, o.kconf, valid, held), metricas=m,
                   equilibrio=eq,
                   com=None if com is None else [round(float(com[0] * h), 1),
                                                 round(float(com[1] * h), 1)])
        return out

    def _forget(self, t: float) -> None:
        for tid, tr in list(self.tracks.items()):
            if t - tr.last_seen > self.cfg.forget_s:
                del self.tracks[tid]
                self.balance.forget(tid)


def _pack(xy, conf, valid, held) -> list:
    """[x, y, conf, estado] por punto; estado 0 = descartado, 1 = visto,
    2 = mantenido por oclusión."""
    return [[round(float(xy[i][0]), 1), round(float(xy[i][1]), 1),
             round(float(conf[i]), 2), 0 if not valid[i] else (2 if held[i] else 1)]
            for i in range(N_KP)]
