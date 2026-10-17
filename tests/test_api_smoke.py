"""Prueba de humo de la aplicación completa (modelos reales).

Se salta sola si faltan los pesos. Usa las imágenes de ejemplo que trae
Ultralytics, así que no versiona ninguna imagen de personas.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2

from backend.config import load_settings

SETTINGS = load_settings()
HAVE_MODELS = SETTINGS.detector_path.is_file() and SETTINGS.coco_path.is_file()


@unittest.skipUnless(HAVE_MODELS, "faltan pesos: python tools/download_models.py")
class ApiSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ultralytics
        from backend.app import create_app
        cls.tmp = tempfile.TemporaryDirectory()
        settings = load_settings(overrides=[f"alerts.dir={cls.tmp.name}"])
        cls.client = create_app(settings).test_client()
        assets = Path(ultralytics.__file__).parent / "assets"
        cls.bus = (assets / "bus.jpg").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health(self):
        d = self.client.get("/api/health").get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(d["classes"], 6)

    def test_index_renders(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_detect_image(self):
        import io
        r = self.client.post("/api/detect", data={"file": (io.BytesIO(self.bus), "bus.jpg")},
                             content_type="multipart/form-data")
        d = r.get_json()
        self.assertEqual(r.status_code, 200, d)
        self.assertIn(d["riesgo"], ("BAJO", "MEDIO", "ALTO"))
        self.assertTrue(any("bbox" in x for x in d["detections"]))

    def test_live_frame_normalised_boxes(self):
        img = cv2.imdecode(__import__("numpy").frombuffer(self.bus, "uint8"), 1)
        jpg = cv2.imencode(".jpg", cv2.resize(img, (960, 540)))[1].tobytes()
        d = None
        for fid in range(1, 15):                 # asíncrono: los primeros pueden ir vacíos
            d = self.client.post(f"/api/frame?overlay=0&sr=1&fid={fid}", data=jpg,
                                 content_type="image/jpeg").get_json()
            if not d.get("pendiente") and d["detections"]:
                break
        boxes = [x["bbox"] for x in d["detections"] if "bbox" in x]
        self.assertTrue(boxes)
        self.assertTrue(all(0 <= v <= 1 for b in boxes for v in b))

    def test_alert_routes(self):
        self.assertEqual(self.client.get("/api/alerts").status_code, 200)
        self.assertEqual(self.client.delete("/api/alertas").status_code, 200)


if __name__ == "__main__":
    unittest.main()
