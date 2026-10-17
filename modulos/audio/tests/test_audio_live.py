"""Integración HTTP simulada: sin cámara ni inferencia visual real.

La integración con el panel de Video-G-G está en test_integracion_panel.py."""
import io
import json
import threading
import time
import zipfile

import cv2
import numpy as np
import pytest
from flask import Flask

from modulos.audio.audio_live import AudioEventStore, AudioSessionManager, audio_blueprint


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


class FakeDetector:
    def __init__(self):
        self.resets = 0
        self.calls = 0

    def reset(self):
        self.resets += 1

    def process(self, samples):
        self.calls += 1
        if np.max(samples) > .5:
            return [dict(impact_offset_s=.5, scores={'score_final': .8}, rules=[], predicted_class='caida')]
        return []


@pytest.fixture
def service(tmp_path):
    store = AudioEventStore(tmp_path / 'audio.jsonl')
    detectors = []
    def factory():
        detector = FakeDetector()
        detectors.append(detector)
        return detector
    manager = AudioSessionManager(store, factory)
    app = Flask(__name__)
    app.register_blueprint(audio_blueprint(manager))
    yield app.test_client(), manager, store, detectors
    manager.close()


def send(client, key, seq, samples=None):
    if samples is None:
        samples = np.zeros(8000, dtype='<f4')
    return client.post(f'/api/audio/sessions/{key}/chunks', data=samples.astype('<f4').tobytes(),
                       content_type='application/octet-stream',
                       headers={'X-Audio-Sequence':str(seq), 'X-Audio-Offset':str(seq*8000)})


def test_start_event_stop_restart(service):
    client, manager, store, detectors = service
    key = client.post('/api/audio/sessions').json['stream_id']
    assert send(client, key, 0, np.ones(8000)*.8).status_code == 202
    eventually(lambda: len(store.read()) == 1)
    row = store.read()[0]
    assert row['type'] == 'posible_caida' and row['source'] == 'audio'
    assert row['stream_id'] == key and 'confidence' not in row
    session = manager.get(key)
    assert client.post(f'/api/audio/sessions/{key}/stop').status_code == 200
    session.worker.join(3)
    assert not session.worker.is_alive() and detectors[0].resets >= 2
    assert send(client, key, 1).status_code == 404
    new_key = client.post('/api/audio/sessions').json['stream_id']
    assert new_key != key and len(manager.sessions) == 1
    assert send(client, new_key, 0).status_code == 202
    eventually(lambda: manager.get(new_key).processed == 1)
    assert len(store.read()) == 1


def test_validate_pcm_sequence_and_gap(service):
    client, manager, store, detectors = service
    key = client.post('/api/audio/sessions').json['stream_id']
    assert client.post(f'/api/audio/sessions/{key}/chunks', data=b'bad').status_code == 400
    assert send(client, key, 0, np.ones(8000)*np.nan).status_code == 400
    assert send(client, key, 0, np.ones(8000)*2).status_code == 400
    assert send(client, key, 0).status_code == 202
    assert send(client, key, 0).status_code == 400
    eventually(lambda: manager.get(key).processed == 1)
    assert send(client, key, 3).status_code == 202
    eventually(lambda: manager.get(key).processed == 2)
    assert detectors[0].resets == 2  # primera ventana y pérdida de audio
    assert manager.get(key).snapshot()['state'] == 'silencio'
    assert not store.read()


def test_gap_keeps_session_alert_cooldown(service):
    client, manager, store, _ = service
    key = client.post('/api/audio/sessions').json['stream_id']
    send(client, key, 0, np.ones(8000)*.8)
    eventually(lambda: len(store.read()) == 1)
    send(client, key, 3, np.ones(8000)*.8)
    eventually(lambda: manager.get(key).processed == 2)
    assert len(store.read()) == 1


def test_queue_bounded_stop_discards_late_event(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Slow(FakeDetector):
        def process(self, samples):
            entered.set()
            release.wait(3)
            return super().process(samples)
    store = AudioEventStore(tmp_path / 'events.jsonl')
    manager = AudioSessionManager(store, Slow)
    session = manager.create()
    try:
        session.submit(0, 0, np.ones(8000)*.8)
        assert entered.wait(2)
        for seq in range(1, 12):
            session.submit(seq, seq*8000, np.ones(8000)*.8)
        assert session.queue.qsize() == 4 and session.dropped == 7
        manager.stop(session.id)
        release.set()
        session.worker.join(3)
        assert not session.worker.is_alive() and not store.read()
    finally:
        release.set()
        manager.close()


def test_expiry_limit_and_dependency_failure(tmp_path):
    store = AudioEventStore(tmp_path / 'events.jsonl')
    manager = AudioSessionManager(store, FakeDetector, max_sessions=1, idle_timeout=.05)
    try:
        first = manager.create()
        with pytest.raises(RuntimeError):
            manager.create()
        eventually(lambda: first.stopped.is_set(), timeout=4)
    finally:
        manager.close()
    def broken():
        raise ImportError('modelo/dependencia no disponible')
    manager = AudioSessionManager(store, broken)
    try:
        session = manager.create()
        eventually(lambda: session.state == 'error')
        assert 'ImportError' in session.snapshot()['error']
    finally:
        manager.close()
