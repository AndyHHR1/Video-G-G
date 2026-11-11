"""Zona de escalera y pasamanos dibujada en el panel (sin GPU)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.perception.zone import Zone, point_segment_distance  # noqa: E402

W, H = 1000, 500
# escalera: trapecio en el centro de la imagen
STAIRS = [[0.3, 0.2], [0.7, 0.2], [0.8, 0.9], [0.2, 0.9]]
RAIL = [[0.25, 0.3], [0.25, 0.85]]          # pasamanos vertical a la izquierda


class ZoneTest(unittest.TestCase):
    def test_validation(self):
        z = Zone.from_dict({"escalera": STAIRS, "pasamanos": [RAIL]})
        self.assertTrue(z.has_stairs and z.has_rails)
        for bad in ({"escalera": [[0.1, 0.1], [0.2, 0.2]]},          # < 3 puntos
                    {"escalera": [[0.1, 0.1], [2, 0.2], [0.3, 0.3]]},  # fuera de 0..1
                    {"pasamanos": [[[0.1, 0.1]]]},                      # línea de 1 punto
                    {"escalera": "x"}, [1, 2], None):
            with self.assertRaises(ValueError):
                Zone.from_dict(bad)
        self.assertFalse(Zone.from_dict({}).has_stairs)               # zona vacía = sin zona

    def test_contains_and_feet(self):
        z = Zone.from_dict({"escalera": STAIRS})
        self.assertTrue(z.contains(500, 300, W, H))
        self.assertFalse(z.contains(50, 300, W, H))
        # persona dentro: sus pies (borde inferior) están en la escalera
        self.assertTrue(z.foot_inside([450, 100, 550, 400], W, H))
        # persona al fondo, a la derecha y fuera
        self.assertFalse(z.foot_inside([880, 100, 960, 400], W, H))
        # sin zona dibujada todo cuenta como dentro (comportamiento anterior)
        self.assertTrue(Zone().foot_inside([880, 100, 960, 400], W, H))

    def test_rail_segments_and_distance(self):
        z = Zone.from_dict({"pasamanos": [RAIL]})
        (a, b), = z.rail_segments(W, H)
        self.assertEqual((a, b), ((250.0, 150.0), (250.0, 425.0)))
        self.assertAlmostEqual(point_segment_distance((260, 300), a, b), 10.0)
        self.assertAlmostEqual(point_segment_distance((250, 500), a, b), 75.0)

    def test_save_load_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "zona.json"
            Zone.from_dict({"escalera": STAIRS, "pasamanos": [RAIL]}).save(p)
            self.assertEqual(Zone.load(p).to_dict()["escalera"], STAIRS)
            p.write_text("{basura")
            self.assertFalse(Zone.load(p).has_stairs)          # corrupto -> sin zona


class RailSignalTest(unittest.TestCase):
    """Con el pasamanos dibujado, se mide la distancia real de la mano."""

    def setUp(self):
        from backend.config import SkeletonSignalsCfg
        from backend.perception.skeleton.signals import skeleton_signals
        self.cfg, self.f = SkeletonSignalsCfg(), skeleton_signals
        self.rails = Zone.from_dict({"pasamanos": [RAIL]}).rail_segments(W, H)

    @staticmethod
    def sk(wrist_left):
        k = [[0.0, 0.0, 0.9, 0] for _ in range(17)]
        for i, (x, y) in {0: (320, 120), 5: (300, 160), 6: (340, 160),
                          11: (305, 260), 12: (335, 260), 9: wrist_left,
                          10: (345, 280)}.items():
            k[i] = [x, y, 0.9, 1]
        return {"id": 1, "kps": k, "metricas": {"torso_px": 100.0}}

    def test_hand_on_drawn_rail(self):
        sigs = self.f(self.sk((255, 250)), self.cfg, stairs_present=True, rails=self.rails)
        self.assertNotIn("sin_pasamanos", {s["tipo"] for s in sigs})

    def test_hand_away_from_drawn_rail(self):
        sigs = self.f(self.sk((300, 290)), self.cfg, stairs_present=True, rails=self.rails)
        self.assertIn("sin_pasamanos", {s["tipo"] for s in sigs})


if __name__ == "__main__":
    unittest.main()
