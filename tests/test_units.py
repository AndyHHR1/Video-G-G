"""Pruebas unitarias rápidas (no necesitan GPU ni modelos).

    python -m unittest discover -s tests -v      (o: pytest tests)
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend.config import load_settings
from backend.perception.alerts import AlertManager
from backend.perception.pose import best_person_match, posture_from_geometry
from backend.perception.tracking import IoUTracker, iou


def _det(box):
    return {"bbox": box, "track_id": None, "confidence": 0.9}


class ConfigTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        s = load_settings(overrides=["engine.pose_every=4", "server.port=8080",
                                     "enhancement.enabled=false"])
        self.assertEqual(s.engine.pose_every, 4)
        self.assertEqual(s.server.port, 8080)
        self.assertIs(s.enhancement.enabled, False)
        self.assertTrue(str(s.detector_path).endswith("escaleras_s7_v4.pt"))

    def test_thresholds_match_requirements(self):
        th = load_settings().thresholds
        self.assertEqual((th.prefilter, th.confirm), (0.75, 0.85))   # RQF04


class TrackingTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_identity_is_stable_and_unique(self):
        t = IoUTracker()
        a, b = t.update([_det([0, 0, 100, 200]), _det([300, 0, 400, 200])])
        self.assertNotEqual(a["track_id"], b["track_id"])
        a2, = t.update([_det([5, 0, 105, 200])])            # se mueve un poco
        self.assertEqual(a2["track_id"], a["track_id"])

    def test_occlusion_forgets_after_timeout(self):
        t = IoUTracker(occlusion_seconds=1.0)
        first, = t.update([_det([0, 0, 100, 200])])
        with mock.patch("backend.perception.tracking.time.time",
                        return_value=time.time() + 5):
            again, = t.update([_det([0, 0, 100, 200])])
        self.assertNotEqual(first["track_id"], again["track_id"])


def _pose(nose_y, hip, ankle):
    # 11 keypoints: nariz, hombros, muñecas, caderas, rodillas, tobillos
    lm = [(0.5, nose_y), (0.45, 0.3), (0.55, 0.3), (0.45, 0.5), (0.55, 0.5),
          (hip[0] - .03, hip[1]), (hip[0] + .03, hip[1]),
          (0.47, 0.75), (0.53, 0.75),
          (ankle[0] - .03, ankle[1]), (ankle[0] + .03, ankle[1])]
    return [{"landmarks": lm, "signals": []}]


class PoseGeometryTest(unittest.TestCase):
    def test_standing(self):
        self.assertEqual(posture_from_geometry(_pose(0.1, (0.5, 0.55), (0.5, 0.95))),
                         "persona_erguida")

    def test_fallen(self):
        # eje cadera-tobillo casi horizontal
        self.assertEqual(posture_from_geometry(_pose(0.8, (0.5, 0.8), (0.9, 0.85))),
                         "persona_caido")

    def test_no_pose(self):
        self.assertIsNone(posture_from_geometry([]))

    def test_best_person_match_picks_overlapping_box(self):
        lm = _pose(0.1, (0.5, 0.55), (0.5, 0.95))[0]["landmarks"]
        people = [[0, 0, 100, 100], [400, 40, 600, 500]]
        self.assertEqual(best_person_match(lm, people, (500, 1000)), 1)


class AlertTest(unittest.TestCase):
    def test_alert_after_persistence_with_evidence_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = IoUTracker()
            mgr = AlertManager(tracker, Path(tmp),
                               persist_seconds={"ALTO": 1.2, "MEDIO": 0.6},
                               persist_default=1.0, grace_seconds=1.0,
                               cooldown_seconds=15.0)
            det, = tracker.update([_det([0, 0, 50, 100])])
            t0 = time.time()
            with mock.patch("backend.perception.alerts.time.time", return_value=t0):
                mgr.observe(det["track_id"], "caida", det)
            self.assertFalse(mgr.has_pending)                 # aún no persiste
            with mock.patch("backend.perception.alerts.time.time", return_value=t0 + 1.3):
                mgr.observe(det["track_id"], "caida", det)
            self.assertTrue(mgr.has_pending)
            mgr.flush_evidence(np.zeros((100, 100, 3), np.uint8))
            log = (Path(tmp) / "alertas.jsonl").read_text().splitlines()
            alert = json.loads(log[0])
            self.assertEqual(alert["severity"], "ALTO")
            self.assertTrue(Path(alert["evidence"]).is_file())  # RQNF15


if __name__ == "__main__":
    unittest.main()
