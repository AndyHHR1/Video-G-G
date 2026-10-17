"""Esqueletos YOLOv8-Pose: filtrado, cinemática y pérdida de equilibrio.

Sin GPU ni modelos: un estimador falso entrega esqueletos sintéticos con la
geometría de una persona, así se prueban escenarios concretos (de pie, bajar
la escalera deprisa, caída, pasamanos tapando las piernas, vista picada).
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from backend.config import load_settings
from backend.perception.skeleton.analyzer import SkeletonAnalyzer
from backend.perception.skeleton.balance import BalanceMonitor
from backend.perception.skeleton.estimator import PoseObs, merge_split
from backend.perception.skeleton.filters import KeypointSmoother
from backend.perception.skeleton.kinematics import Kinematics, up_vector

H, W = 1080, 1920


def body(cx=900.0, hip_y=600.0, torso=90.0, lean_deg=0.0, conf=0.9,
         hidden=()):
    """17 puntos COCO de una persona. `lean_deg` inclina el tronco (y la
    cabeza) respecto a la vertical; las piernas quedan donde estaban."""
    a = math.radians(lean_deg)
    axis = np.array([math.sin(a), -math.cos(a)])        # cadera -> hombros
    side = np.array([math.cos(a), math.sin(a)])
    hip = np.array([cx, hip_y])
    sh = hip + axis * torso
    head = sh + axis * torso * 0.45
    p = np.zeros((17, 2))
    p[0] = head
    p[1], p[2] = head + side * -6, head + side * 6
    p[3], p[4] = head + side * -12, head + side * 12
    p[5], p[6] = sh - side * 20, sh + side * 20
    p[7], p[8] = p[5] - axis * torso * 0.45, p[6] - axis * torso * 0.45
    p[9], p[10] = p[7] - axis * torso * 0.4, p[8] - axis * torso * 0.4
    p[11], p[12] = hip - np.array([15, 0]), hip + np.array([15, 0])
    p[13], p[14] = p[11] + [0, torso * 0.6], p[12] + [0, torso * 0.6]
    p[15], p[16] = p[13] + [0, torso * 0.6], p[14] + [0, torso * 0.6]
    c = np.full(17, conf)
    for i in hidden:
        c[i] = 0.05
    x1, y1 = p.min(axis=0) - 10
    x2, y2 = p.max(axis=0) + 10
    return p, c, [float(x1), float(y1), float(x2), float(y2)]


class FakeEstimator:
    def __init__(self, frames):
        self.frames = list(frames)

    def __call__(self, _frame):
        out = []
        for tid, (p, c, box) in self.frames.pop(0):
            out.append(PoseObs(tid, box, 0.9, p, c))
        return out

    def reset(self):
        pass


def run(seq, fps=15.0, cfg=None):
    cfg = cfg or load_settings().skeleton
    an = SkeletonAnalyzer(cfg, "cpu", estimator=FakeEstimator(seq))
    frame = np.zeros((H, W, 3), np.uint8)
    return [an.step(frame, i / fps) for i in range(len(seq))]


def states(results, tid=1):
    return [next(p for p in r if p["id"] == tid)["equilibrio"]["estado"]
            for r in results]


class SmootherTest(unittest.TestCase):
    def test_low_confidence_points_are_dropped_then_held(self):
        s = KeypointSmoother(conf_min=0.45, hold_s=0.5)
        p, c, _ = body()
        x, valid, held = s.update(p / H, c, 0.0)
        self.assertTrue(valid.all())
        c2 = c.copy()
        c2[15:] = 0.1                       # tobillos tapados (pasamanos)
        x, valid, held = s.update(p / H, c2, 0.2)
        self.assertTrue(valid[15] and held[15])          # se mantienen
        x, valid, held = s.update(p / H, c2, 0.9)
        self.assertFalse(valid[15])                      # pasado hold_s, fuera

    def test_one_euro_removes_jitter(self):
        s = KeypointSmoother(min_cutoff=1.0, beta=1.5)
        rng = np.random.default_rng(0)
        p, c, _ = body()
        raw, out = [], []
        for i in range(60):
            noisy = p + rng.normal(0, 3, p.shape)
            x, _, _ = s.update(noisy / H, c, i / 15)
            raw.append(noisy[11, 0])
            out.append(x[11, 0] * H)
        self.assertLess(np.std(out[10:]), 0.6 * np.std(raw[10:]))

    def test_impossible_jump_rejected(self):
        s = KeypointSmoother(max_jump_torso=1.2)
        p, c, _ = body()
        s.update(p / H, c, 0.0, torso=90 / H)
        q = p.copy()
        q[9] += [400, 0]                    # muñeca «salta» 4 torsos: error
        x, valid, held = s.update(q / H, c, 1 / 15, torso=90 / H)
        self.assertTrue(held[9])
        self.assertAlmostEqual(x[9, 0] * H, p[9, 0], delta=1)


class KinematicsTest(unittest.TestCase):
    def test_trunk_angle_and_feet(self):
        k = Kinematics()
        for lean, expect in ((0, 0), (40, 40)):
            p, c, _ = body(lean_deg=lean)
            m = k.update(p / H, c > 0.45, 0.0)
            self.assertAlmostEqual(m["tronco_deg"], expect, delta=1)
        p, c, _ = body()
        self.assertEqual(Kinematics().update(p / H, c > 0.45, 0.0)["com_offset"], 0.0)

    def test_perspective_nadir_changes_vertical(self):
        # Vista picada: las verticales convergen bajo la imagen. Una persona a
        # la derecha aparece inclinada en la imagen pero está vertical.
        nadir = np.array([0.5 * W / H, 2.5])
        up = up_vector(np.array([0.8 * W / H, 0.5]), nadir)
        self.assertGreater(abs(up[0]), 0.1)
        self.assertLess(up[1], 0)

    def test_hidden_ankles_make_com_not_evaluable(self):
        p, c, _ = body(hidden=(15, 16))
        m = Kinematics().update(p / H, c > 0.45, 0.0)
        self.assertIsNone(m["com_offset"])
        self.assertIsNotNone(m["tronco_deg"])


class BalanceTest(unittest.TestCase):
    def test_standing_is_stable(self):
        seq = [[(1, body())] for _ in range(30)]
        self.assertEqual(set(states(run(seq))), {"estable"})

    def test_walking_down_stairs_fast_is_not_a_fall(self):
        # 1.2 torsos/s hacia abajo con el tronco erguido: un solo criterio
        # como mucho, nunca dos.
        seq = [[(1, body(cx=900 + 4 * i, hip_y=500 + 7.2 * i))] for i in range(45)]
        self.assertNotIn("perdida_equilibrio", states(run(seq)))

    def test_fall_triggers_alert_once(self):
        seq = [[(1, body())] for _ in range(45)]           # 3 s estable: línea base
        for i in range(1, 9):                              # tropiezo ~0.5 s
            seq.append([(1, body(cx=900 + 10 * i, hip_y=600 + 22 * i,
                                 lean_deg=min(70, 9 * i)))])
        res = run(seq)
        st = states(res)
        self.assertIn("perdida_equilibrio", st[45:])
        eqs = [next(p for p in r if p["id"] == 1)["equilibrio"] for r in res]
        self.assertEqual(sum(e.get("alerta", False) for e in eqs), 1)   # cooldown
        last = eqs[-1]
        self.assertIn("tronco", last["criterios"])
        self.assertIsNotNone(last["linea_base_deg"])

    def test_handrail_hiding_legs_does_not_alarm(self):
        seq = [[(1, body(hidden=(13, 14, 15, 16)))] for _ in range(40)]
        res = run(seq)
        self.assertEqual(set(states(res)), {"estable"})
        m = next(p for p in res[-1] if p["id"] == 1)["metricas"]
        self.assertIsNone(m["com_offset"])

    def test_two_people_keep_separate_state(self):
        seq = [[(1, body(cx=500)), (2, body(cx=1400))] for _ in range(45)]
        for i in range(1, 9):
            seq.append([(1, body(cx=500)),
                        (2, body(cx=1400 + 10 * i, hip_y=600 + 22 * i,
                                 lean_deg=min(70, 9 * i)))])
        res = run(seq)
        self.assertNotIn("perdida_equilibrio", states(res, 1))
        self.assertIn("perdida_equilibrio", states(res, 2))

    def test_single_criterion_is_not_enough(self):
        cfg = load_settings().skeleton.balance
        mon = BalanceMonitor(cfg)
        r = mon.evaluate(1, {"tronco_deg": 5, "cadera_baja_v": 3.0}, 4.0, 0.0)
        self.assertEqual(r["estado"], "estable")
        self.assertEqual(r["criterios"], ["cadera_cae"])


class MergeSplitTest(unittest.TestCase):
    def test_person_cut_by_handrail_is_merged(self):
        p, c, box = body()
        up_c, lo_c = c.copy(), c.copy()
        up_c[11:] = 0.05                    # mitad superior
        lo_c[:11] = 0.05                    # mitad inferior
        y_cut = p[11, 1] - 5
        up = PoseObs(1, [box[0], box[1], box[2], y_cut], 0.8, p, up_c)
        lo = PoseObs(2, [box[0], y_cut + 15, box[2], box[3]], 0.6, p, lo_c)
        out = merge_split([up, lo], 0.45)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].merged)
        self.assertEqual(out[0].track_id, 1)
        self.assertTrue((out[0].kconf >= 0.45).all())

    def test_two_people_side_by_side_not_merged(self):
        p1, c1, b1 = body(cx=500)
        p2, c2, b2 = body(cx=1400)
        out = merge_split([PoseObs(1, b1, .9, p1, c1), PoseObs(2, b2, .9, p2, c2)], 0.45)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
