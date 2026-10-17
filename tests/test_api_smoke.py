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
        cls.app = create_app(settings)
        cls.client = cls.app.test_client()
        assets = Path(ultralytics.__file__).parent / "assets"
        cls.bus = (assets / "bus.jpg").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.app.extensions["escaleras"].lifecycle.shutdown("fin de pruebas")
        cls.tmp.cleanup()

    def test_health(self):
        d = self.client.get("/api/health").get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(d["classes"], 6)

    def test_index_renders(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_upload_routes_removed(self):
        """Solo vídeo en vivo: ya no existen las rutas de imagen/vídeo subidos."""
        for route in ("/api/detect", "/api/video"):
            self.assertIn(self.client.post(route).status_code, (404, 405), route)

    def test_health_reports_profile_and_modules(self):
        d = self.client.get("/api/health").get_json()
        self.assertIn(d["hardware"]["perfil"], ("normal", "baja_vram"))
        self.assertTrue(d["modulos"]["audio"]["ok"], d["modulos"]["audio"])
        self.assertTrue(d["modulos"]["imagen"]["ok"], d["modulos"]["imagen"])

    def test_index_has_module_tabs_and_stop_button(self):
        html = self.client.get("/").get_data(as_text=True)
        for needle in ('data-p="cam"', 'data-p="audio"', 'data-p="imagen"',
                       'id="bsys"', 'id="p-audio"', 'id="p-imagen"'):
            self.assertIn(needle, html)
        for gone in ('data-p="img"', 'data-p="vid"', "/api/detect", "/api/video"):
            self.assertNotIn(gone, html)

    def test_imagen_module_photo(self):
        import io
        r = self.client.post("/api/imagen/pruebas/frame",
                             data={"frame": (io.BytesIO(self.bus), "bus.jpg"),
                                   "modo": "adaptativo"},
                             content_type="multipart/form-data")
        d = r.get_json()
        self.assertEqual(r.status_code, 200, d)
        self.assertEqual(d["estado"], "ok")
        self.assertGreaterEqual(len(d["personas"]), 1)
        self.assertTrue(d["imagen_anotada"].startswith("data:image/jpeg;base64,"))

    def test_live_frame_normalised_boxes(self):
        img = cv2.imdecode(__import__("numpy").frombuffer(self.bus, "uint8"), 1)
        jpg = cv2.imencode(".jpg", cv2.resize(img, (960, 540)))[1].tobytes()
        d = None
        import time
        for fid in range(1, 80):                 # asíncrono: los primeros pueden ir vacíos
            d = self.client.post(f"/api/frame?overlay=0&sr=1&fid={fid}", data=jpg,
                                 content_type="image/jpeg").get_json()
            if not d.get("pendiente") and d["detections"]:
                break
            time.sleep(0.05)                     # ritmo de cámara, no ráfaga
        boxes = [x["bbox"] for x in d["detections"] if "bbox" in x]
        self.assertTrue(boxes)
        self.assertTrue(all(0 <= v <= 1 for b in boxes for v in b))
        # El módulo Imagen lee (sin alterar) el último fotograma en vivo.
        f = self.client.get("/api/imagen/caracteristicas").get_json()
        self.assertEqual(f["schema_version"], "imagen-0.1.0")
        self.assertEqual(f["metadata_captura"]["fuente"], "video_en_vivo")

    def test_alert_routes(self):
        self.assertEqual(self.client.get("/api/alerts").status_code, 200)
        self.assertEqual(self.client.delete("/api/alertas").status_code, 200)


if __name__ == "__main__":
    unittest.main()
