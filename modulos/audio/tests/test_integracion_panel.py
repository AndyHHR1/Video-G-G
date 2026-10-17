"""Integración del módulo de audio con el panel de Video-G-G, sin modelos.

Sustituye a las dos pruebas del repo original que usaban su servidor antiguo
(app/server.py): aquí se monta el backend real de alertas/autenticación/vídeo
con un inferencer simulado y se instala el módulo como lo hace backend/app.py.
"""
import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from flask import Flask

from backend.api import alerts, analysis, auth, system
from backend.api.context import Runtime
from backend.config import load_settings
from backend.services.lifecycle import Lifecycle


class FakeInferencer:
    device = "cpu"
    engine = None
    stopped = False

    def detect(self, *args, **kwargs):
        return None

    def last_result(self):
        return None


@pytest.fixture
def panel(tmp_path):
    import modulos.audio as audio_mod
    settings = load_settings(overrides=[f"alerts.dir={tmp_path / 'visual'}"])
    app = Flask(__name__, template_folder="../../../frontend/templates")
    rt = Runtime(settings=settings, inferencer=FakeInferencer(), live_enhancer=None,
                 lifecycle=Lifecycle())
    app.extensions["escaleras"] = rt
    for bp in (auth.bp, system.bp, analysis.bp, alerts.bp):
        app.register_blueprint(bp)
    status = audio_mod.install(app, settings, rt, {
        "alerts_file": str(tmp_path / "audio" / "alertas.jsonl"), "ml_detector": False})
    rt.modules["audio"] = status
    store = rt.alert_sources[0]
    yield app.test_client(), rt, store
    rt.lifecycle.shutdown("fin de prueba")


def test_alertas_audio_en_panel_zip_y_borrado(panel):
    client, rt, store = panel
    store.append(dict(alert_id="audio-test", timestamp="2026-10-16T00:00:00+00:00",
                      source="audio", risk_type="posible_caida_acustica",
                      description="Posible caída"))
    assert client.get("/modulos/audio/static/audio-live.js").status_code == 200
    assert client.get("/modulos/audio/static/audio-worklet.js").status_code == 200
    assert client.get("/api/alerts").json["total"] == 1
    exported = client.get("/api/alertas.zip")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.data)) as archive:
        assert json.loads(archive.read("alertas_audio.jsonl"))["source"] == "audio"
    assert client.delete("/api/alertas", json={"ids": ["audio-test"]}).json["borradas"] == 1
    assert client.get("/api/alerts").json["total"] == 0


def test_borrar_solo_audio_conserva_video(panel):
    client, rt, store = panel
    vis = rt.settings.alerts_dir
    vis.mkdir(parents=True, exist_ok=True)
    (vis / "alertas.jsonl").write_text(json.dumps(
        {"alert_id": "v1", "timestamp": "2026-10-16T10:00:00"}) + "\n", encoding="utf-8")
    store.append(dict(alert_id="a1", timestamp="2026-10-16T15:00:00+00:00", source="audio"))
    assert client.get("/api/alerts").json["total"] == 2
    client.delete("/api/alertas?source=audio")
    rows = client.get("/api/alerts").json["alertas"]
    assert [r["alert_id"] for r in rows] == ["v1"]


def test_autenticacion_protege_audio(panel):
    client, rt, _ = panel
    rt.panel_token = "test-only"
    assert client.post("/api/audio/sessions").status_code == 401
    assert client.get("/api/audio/sessions/missing",
                      headers={"X-Panel-Token": "test-only"}).status_code == 404


def test_video_sigue_respondiendo_durante_audio(panel):
    client, rt, _ = panel
    ok, jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
    assert ok
    key = client.post("/api/audio/sessions").json["stream_id"]
    for seq in range(6):
        r = client.post(f"/api/audio/sessions/{key}/chunks",
                        data=np.zeros(8000, "<f4").tobytes(),
                        content_type="application/octet-stream",
                        headers={"X-Audio-Sequence": str(seq),
                                 "X-Audio-Offset": str(seq * 8000)})
        assert r.status_code in (202, 429)
        frame = client.post("/api/frame?overlay=0&sr=0", data=jpeg.tobytes(),
                            content_type="image/jpeg")
        assert frame.status_code == 200


def test_archivo_de_video_rechazado(panel):
    client, _, _ = panel
    r = client.post("/api/audio/analyze-file",
                    data={"file": (io.BytesIO(b"\x00" * 64), "clip.mp4")},
                    content_type="multipart/form-data")
    assert r.status_code == 415


def test_simulacion_registra_alertas(panel):
    client, _, store = panel
    r = client.post("/api/audio/simulate", json={"sensitivity": "media"})
    assert r.status_code == 200, r.json
    cats = {e["category"] for e in r.json["events"]}
    assert "caida" in cats
    assert r.json["alerts_registered"] == len(store.read()) > 0


def test_parada_del_sistema_cierra_audio(panel):
    client, rt, _ = panel
    client.post("/api/audio/sessions")
    manager_sessions = client.get("/api/audio/status").json["sesiones_activas"]
    assert manager_sessions == 1
    assert client.post("/api/system/stop").status_code == 202
    assert rt.lifecycle.wait(10)
    assert client.post("/api/frame", data=b"x", content_type="image/jpeg").status_code == 503
