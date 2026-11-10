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
        self.assertTrue(str(s.detector_path).endswith("escalera_v5.pt"))
        self.assertTrue(str(s.posture_path).endswith("postura_v5.pt"))

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


class HardwareProfileTest(unittest.TestCase):
    """Perfil para la GTX 1650 (4 GB, Turing sm_75)."""

    def test_gtx1650_selects_low_vram(self):
        from backend.hardware import choose_profile
        gtx = {"name": "NVIDIA GeForce GTX 1650", "vram_gb": 3.81, "capability": "7.5"}
        self.assertEqual(choose_profile("auto", gtx, 4.5), "baja_vram")
        big = {"name": "NVIDIA GeForce RTX 5070 Laptop GPU", "vram_gb": 7.9,
               "capability": "12.0"}
        self.assertEqual(choose_profile("auto", big, 4.5), "normal")
        self.assertEqual(choose_profile("auto", {}, 4.5), "cpu")         # sin GPU
        self.assertEqual(choose_profile("baja_vram", big, 4.5), "baja_vram")

    def test_apply_profile_disables_fp16_and_shrinks_live_sr(self):
        from backend.hardware import apply_profile
        s = load_settings()
        gtx = {"name": "NVIDIA GeForce GTX 1650", "vram_gb": 3.81,
               "capability": "7.5", "arch_ok": True}
        name, changes = apply_profile(s, gtx)
        self.assertEqual(name, "baja_vram")
        self.assertFalse(s.enhancement.half)
        self.assertEqual(s.enhancement.live.input_max, 320)
        self.assertEqual(s.inference.imgsz, 640)     # la detección no cambia
        self.assertIn("enhancement.half", changes)

    def test_cpu_profile_lightens_skeleton_and_resolves_path(self):
        from backend.hardware import apply_profile
        s = load_settings()
        name, changes = apply_profile(s, {})          # sin GPU
        self.assertEqual(name, "cpu")
        self.assertTrue(str(s.skeleton.model_path).endswith("yolov8n-pose.pt"))
        self.assertGreaterEqual(s.skeleton.every, 4)
        self.assertIn("skeleton.model", changes)
        # GPU presente pero device=cpu forzado: también perfil cpu
        s2 = load_settings(overrides=["device=cpu"])
        big = {"name": "NVIDIA GeForce RTX 5070", "vram_gb": 7.9,
               "capability": "12.0", "arch_ok": True}
        self.assertEqual(apply_profile(s2, big)[0], "cpu")


class EngineCadenceTest(unittest.TestCase):
    """Una etapa sin resultados (escena sin escalera / sin personas) debe
    respetar su cadencia: antes corría en TODOS los fotogramas."""

    def test_empty_stage_respects_cadence(self):
        from backend.perception.engine import RiskEngine

        calls = {"stairs": 0, "people": 0}

        class FakePosture:
            names = {3: "escalera"}
            id_of = {"escalera": 3}

            def stairs(self, frame, max_boxes=2):
                calls["stairs"] += 1
                return []                              # nunca hay escalera

            def classify_many(self, frame, boxes):
                return [("", 0.0)] * len(boxes)

        class FakeCoco:
            def people(self, frame):
                calls["people"] += 1
                return []                              # nunca hay personas

            def obstacles(self, frame):
                return []

        s = load_settings()
        eng = RiskEngine.__new__(RiskEngine)
        eng.__dict__.update(
            settings=s, device="cpu", conf=0.25, th=s.thresholds,
            pose_every=10 ** 6, people_every=6, stairs_every=13,
            obstacle_every=10 ** 6, classify_every=6, max_stairs_boxes=2,
            posture=FakePosture(), coco=FakeCoco(), skeleton=None,
            skeleton_every=1, _frame_no=0, _last_people=None,
            _last_stairs=None, _last_obstacles=[], _last_geom=None,
            _last_landmarks=None, _last_class=None, _last_skeletons=None,
            _fall_held={}, _errors={"gpu": 0, "pose": 0, "deteccion": 0},
            _error_prints={}, camera_connected=True)
        from backend.perception.tracking import IoUTracker
        from unittest import mock
        eng.tracker = IoUTracker()
        eng.pose = mock.Mock(analyse=lambda f: [])
        eng.alerts = mock.MagicMock()
        eng.alerts.update.return_value = []
        frame = np.zeros((54, 96, 3), np.uint8)
        for _ in range(26):
            try:
                eng.step(frame, allow_alerts=False)
            except Exception:                          # noqa: BLE001
                pass    # el resto del pipeline usa piezas falsas; solo se cuentan llamadas
        # 26 fotogramas: escalera en el 1.º (no había resultado) + 13 y 26
        self.assertGreaterEqual(calls["stairs"], 2)    # la etapa sí se alcanza
        self.assertLessEqual(calls["stairs"], 3)
        # personas: 1.º + 6, 12, 18, 24
        self.assertLessEqual(calls["people"], 5)


class LifecycleTest(unittest.TestCase):
    def test_shutdown_runs_hooks_once_in_reverse_order(self):
        from backend.services.lifecycle import Lifecycle
        calls = []
        lc = Lifecycle()
        lc.on_shutdown("primero", lambda: calls.append(1))
        lc.on_shutdown("falla", lambda: 1 / 0)
        lc.on_shutdown("último", lambda: calls.append(3))
        stopped = []
        lc.bind_server(lambda: stopped.append(True))
        self.assertTrue(lc.shutdown("prueba"))
        self.assertFalse(lc.shutdown("otra vez"))
        self.assertEqual(calls, [3, 1])               # un fallo no corta la parada
        self.assertTrue(lc.stopped and stopped)


if __name__ == "__main__":
    unittest.main()
