"""Sesiones de audio acotadas y análisis acústico en tiempo real e histórico.

Implementa la arquitectura acústica de `final.py` (UPAO):
- Detección en vivo de pisadas y cadencia (caminando vs corriendo).
- Detección de impacto súbito de caída (pico RMS y factor de cresta).
- Detección de gritos y pedidos de auxilio (frecuencia fundamental F0 con pYIN).
- Registro persistente de alertas en `alertas.jsonl` y visualización inmediata.
- Soporte para análisis de archivos de audio/video y sesión sintética de prueba.
- Clasificador ML de caídas del grupo (`audio_fall_detector`, MLP calibrado de
  seis clases) en paralelo al detector por reglas: ambos analizan el mismo PCM.

Integración en Video-G-G: este archivo vive en `modulos/audio/` y solo se
comunica con el vídeo a través de las rutas `/api/audio/*` y de la fuente de
alertas que registra `modulos/audio/__init__.py`. No importa nada del vídeo.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Blueprint, jsonify, request

from .acoustic_perception import (
    SR,
    ClassificationThresholds,
    SensitivityConfig,
    StreamingAcousticDetector,
    analyze_session_full,
    build_simulated_session,
    extraer_audio_archivo,
    mejorar_audio,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MESSAGE = 'ALERTA: Se ha detectado un sonido acústico de riesgo en escaleras.'


ML_MESSAGE = 'ALERTA: Se ha detectado un sonido compatible con una posible caída.'
# Separación mínima entre dos alertas del MISMO tipo en una sesión. Cada
# detector tiene además su propio cooldown interno (3-4 s reglas, 30 s ML).
EVENT_COOLDOWN_S = 2.0
AUDIO_SUFFIXES = {'.wav', '.mp3', '.m4a', '.ogg', '.oga', '.flac', '.opus', '.aac', '.weba'}


class AudioEventStore:
    """Registro JSONL de alertas acústicas (solo metadatos, nunca audio).

    También actúa como *fuente de alertas* del panel de vídeo: `/api/alerts`
    las mezcla con las visuales usando `read()`, `delete()` y `zip_name`."""
    name = 'audio'
    zip_name = 'alertas_audio.jsonl'

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def read(self):
        with self.lock:
            if not self.path.exists():
                return []
            rows = []
            for line in self.path.read_text(encoding='utf-8', errors='replace').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
            return rows

    def append(self, event):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')

    def delete(self, ids=None):
        with self.lock:
            rows = self.read()
            keep = [r for r in rows if ids is not None and r.get('alert_id') not in ids]
            if self.path.exists():
                self.path.write_text(
                    ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in keep),
                    encoding='utf-8',
                )
            return len(rows) - len(keep)


class AudioSession:
    def __init__(self, detector_factory, store, idle_timeout=15, sensitivity="media"):
        self.id = uuid.uuid4().hex
        self.store = store
        self.idle_timeout = idle_timeout
        self.sensitivity = sensitivity
        self.last_seen = time.monotonic()
        self.stopped = threading.Event()
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=4)
        self.state = 'iniciando'
        self.error = None
        self.last_sequence = -1
        self.processed = 0
        self.dropped = 0
        self.processing_ms = 0.0
        self.rms = 0.0
        self.live_analysis = None
        self.last_event_offset = {}      # risk_type -> offset de la última alerta
        self.worker = threading.Thread(
            target=self._run, args=(detector_factory,), daemon=True, name=f'audio-{self.id[:8]}'
        )
        self.worker.start()

    def submit(self, sequence, offset, samples):
        with self.lock:
            if self.stopped.is_set():
                raise KeyError(self.id)
            if sequence <= self.last_sequence:
                raise ValueError('Secuencia duplicada o fuera de orden')
            self.last_sequence = sequence
            self.last_seen = time.monotonic()
            try:
                self.queue.put_nowait((sequence, offset, samples))
            except queue.Full:
                self.dropped += 1
                return False
            return True

    def snapshot(self):
        with self.lock:
            snap = dict(
                stream_id=self.id,
                state=self.state,
                error=self.error,
                processed_chunks=self.processed,
                dropped_chunks=self.dropped,
                processing_ms=round(self.processing_ms, 2),
                rms=round(self.rms, 6),
            )
            if self.live_analysis:
                snap['live_analysis'] = self.live_analysis
            return snap

    def stop(self):
        with self.lock:
            self.stopped.set()
            self.state = 'detenido'
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass

    def _run(self, factory):
        detector = None
        try:
            if callable(factory):
                try:
                    detector = factory(self.sensitivity)
                except TypeError:
                    detector = factory()
            else:
                detector = factory
            expected = None
            origin = 0.0
            with self.lock:
                if not self.stopped.is_set():
                    self.state = 'esperando_audio'
            while not self.stopped.is_set():
                try:
                    item = self.queue.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() - self.last_seen > self.idle_timeout:
                        self.stop()
                    continue
                if item is None:
                    break
                seq, offset, samples = item
                if expected is None or offset != expected:
                    detector.reset()
                    origin = offset / 16000
                expected = offset + len(samples)
                started = time.perf_counter()
                events = detector.process(samples)
                with self.lock:
                    if self.stopped.is_set():
                        break
                    self.processed += 1
                    self.processing_ms = (time.perf_counter() - started) * 1000
                    self.rms = float(np.sqrt(np.mean(samples ** 2)))
                    
                    # Estado en vivo y último análisis
                    if hasattr(detector, 'last_result') and detector.last_result:
                        self.live_analysis = detector.last_result
                        self.state = detector.last_result.get('category', 'analizando')
                    else:
                        self.state = 'silencio' if self.rms < 1e-4 else 'analizando'

                    for result in events:
                        impact_offset = origin + result.get('impact_offset_s', 0.0)
                        raw_cat = result.get('type') or result.get('predicted_class') or 'caida'
                        category = 'posible_caida' if raw_cat in ('caida', 'posible_caida') else raw_cat
                        kind = result.get('risk_type', f'{category}_acustica')
                        # Cooldown de 2.0 s por TIPO de alerta en la misma sesión:
                        # el clasificador ML informa el impacto ~7 s después, así
                        # que no debe chocar con la alerta inmediata por reglas.
                        if abs(impact_offset - self.last_event_offset.get(kind, -1e9)) < EVENT_COOLDOWN_S:
                            continue
                        event_id = 'audio-' + uuid.uuid4().hex
                        severity = result.get('severity') or ('ALTO' if raw_cat in ('caida', 'posible_caida', 'pedido_auxilio') else 'MEDIO')
                        
                        desc_map = {
                            'posible_caida': '¡ALERTA CRÍTICA: Impacto acústico súbito compatible con caída en escaleras!',
                            'caida': '¡ALERTA CRÍTICA: Impacto acústico súbito compatible con caída en escaleras!',
                            'pedido_auxilio': '¡ALERTA CRÍTICA: Grito o pedido de auxilio detectado!',
                            'corriendo': 'ADVERTENCIA: Alta frecuencia de pisadas; persona corriendo en escaleras.',
                            'caminando': 'Tránsito peatonal seguro: persona caminando normalmente.',
                        }
                        desc = result.get('description') or desc_map.get(raw_cat, DEFAULT_MESSAGE)
                        
                        self.store.append(dict(
                            alert_id=event_id,
                            event_id=event_id,
                            type=category,
                            risk_type=kind,
                            source='audio',
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            stream_id=self.id,
                            severity=severity,
                            description=desc,
                            impact_offset_s=round(impact_offset, 3),
                            risk_ref=result.get('risk_ref', 'Módulo de Percepción de Sonido en Escaleras (final.py)'),
                            alarm_activated=result.get('alarm_activated', severity == 'ALTO'),
                            technical=result.get('technical') or {
                                k: result[k] for k in ('scores', 'rules', 'predicted_class')
                                if k in result},
                            detector=result.get('detector', 'reglas'),
                            imagen=None,
                        ))
                        self.last_event_offset[kind] = impact_offset
        except Exception as exc:
            LOGGER.exception('Fallo en sesión acústica %s', self.id)
            with self.lock:
                if not self.stopped.is_set():
                    self.state = 'error'
                    self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if detector is not None and hasattr(detector, 'reset'):
                detector.reset()


class CombinedAudioDetector:
    """Ejecuta sobre el mismo PCM el detector por reglas (final.py) y, si está
    disponible, el clasificador ML del grupo (`audio_fall_detector`).

    * Reglas: respuesta inmediata (bloques de 500 ms) para pisadas, carrera,
      impacto y gritos; alimenta las métricas en vivo del panel.
    * ML: espera los 7 s posteriores reales a un impacto y aplica el MLP de
      seis clases + evaluación posterior + fusión originales.
    """

    def __init__(self, acoustic, ml=None):
        self.acoustic = acoustic
        self.ml = ml

    @property
    def last_result(self):
        return self.acoustic.last_result

    def reset(self):
        self.acoustic.reset()
        if self.ml is not None:
            self.ml.reset()

    def process(self, samples):
        events = list(self.acoustic.process(samples))
        for ev in events:
            ev.setdefault('detector', 'reglas')
        ml_events = []
        if self.ml is not None:
            try:
                ml_events = self.ml.process(samples)
            except Exception:                               # noqa: BLE001
                # Un fallo del MLP no debe tumbar el detector por reglas.
                LOGGER.exception('Clasificador ML de audio desactivado en esta sesión')
                self.ml = None
        if ml_events:
            for ev in ml_events:
                events.append(dict(
                    # Severidad original de la integración ML del grupo: MEDIO
                    # (posible caída que requiere verificación humana).
                    type='posible_caida', risk_type='posible_caida_ml',
                    severity='MEDIO', alarm_activated=True,
                    impact_offset_s=ev['impact_offset_s'],
                    description=ML_MESSAGE,
                    risk_ref='Clasificador ML de caídas (audio_fall_detector) · '
                             'requiere verificación humana',
                    technical={'scores': ev.get('scores'), 'rules': ev.get('rules'),
                               'predicted_class': ev.get('predicted_class')},
                    detector='ml'))
        return events


class AudioSessionManager:
    def __init__(self, store, detector_factory=None, max_sessions=4, idle_timeout=15,
                 default_sensitivity="media", use_ml=False):
        self.store = store
        self.default_sensitivity = default_sensitivity
        self.detector_factory = detector_factory or self._make_detector
        self.sessions = {}
        self.lock = threading.Lock()
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout
        self.closed = threading.Event()
        self.reaper = None
        self.use_ml = use_ml
        self.classifier = None
        self.classifier_lock = threading.Lock()
        self.ml_error = None

    # --- clasificador ML (audio_fall_detector) ------------------------------
    def _ml_detector(self):
        """StreamingFallDetector que comparte una única cabeza MLP cargada."""
        if not self.use_ml or self.ml_error:
            return None
        try:
            from threadpoolctl import threadpool_limits

            from .audio_fall_detector.classifier import FallClassifierHead
            from .audio_fall_detector.streaming import (StreamingFallDetector,
                                                        runtime_config)
            with self.classifier_lock:
                if self.classifier is None:
                    self.classifier = FallClassifierHead(runtime_config())
            detector = StreamingFallDetector(self.classifier)
        except Exception as exc:                            # noqa: BLE001
            self.ml_error = f'{type(exc).__name__}: {exc}'
            LOGGER.warning('Clasificador ML de audio no disponible: %s', exc)
            return None
        original = detector.process

        def process(samples):
            # Una inferencia espectral a la vez y BLAS acotado para proteger el vídeo.
            with self.classifier_lock, threadpool_limits(limits=1, user_api='blas'):
                return original(samples)
        detector.process = process
        return detector

    def _make_detector(self, sensitivity=None):
        sens = sensitivity or self.default_sensitivity
        return CombinedAudioDetector(StreamingAcousticDetector(sensitivity=sens),
                                     self._ml_detector())

    @property
    def ml_status(self):
        if not self.use_ml:
            return 'desactivado'
        if self.ml_error:
            return f'no disponible ({self.ml_error})'
        return 'activo' if self.classifier is not None else 'pendiente'

    def warmup(self):
        """Precalienta kernels JIT de librosa (onsets y pYIN) y carga el MLP
        para que el primer evento en vivo no pague la compilación."""
        LOGGER.info('Precalentando kernels acústicos...')
        detector = self._make_detector(self.default_sensitivity)
        silence = np.zeros(8000, dtype=np.float32)
        detector.process(silence)
        tone = (0.05 * np.sin(2 * np.pi * 500 * np.arange(8000) / 16000)).astype(np.float32)
        detector.process(tone)
        if detector.ml is not None:
            from threadpoolctl import threadpool_limits
            with self.classifier_lock, threadpool_limits(limits=1, user_api='blas'):
                self.classifier.predict_proba(np.zeros(24000, dtype=np.float32))
                t = np.arange(8000, dtype=np.float32) / 16000
                detector.ml.confirmator.detect_vocal_distress(
                    (.1 * np.sin(2 * np.pi * 450 * t)).astype(np.float32))
        detector.reset()
        LOGGER.info('Kernels acústicos precalentados correctamente.')

    def _reap(self):
        while not self.closed.wait(2):
            with self.lock:
                for key, value in list(self.sessions.items()):
                    if value.stopped.is_set() or time.monotonic() - value.last_seen > self.idle_timeout:
                        value.stop()
                        del self.sessions[key]

    def create(self, sensitivity=None):
        with self.lock:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError('Límite de sesiones de audio alcanzado')
            if self.reaper is None:
                self.reaper = threading.Thread(target=self._reap, daemon=True, name='audio-cleanup')
                self.reaper.start()
            sens = sensitivity or self.default_sensitivity
            session = AudioSession(self.detector_factory, self.store, self.idle_timeout, sensitivity=sens)
            self.sessions[session.id] = session
            return session

    def get(self, key):
        with self.lock:
            return self.sessions[key]

    def stop(self, key):
        with self.lock:
            session = self.sessions.pop(key, None)
        if session:
            session.stop()

    def close(self):
        self.closed.set()
        with self.lock:
            for session in self.sessions.values():
                session.stop()
            self.sessions.clear()


def audio_blueprint(manager):
    bp = Blueprint('audio_live', __name__)

    @bp.post('/api/audio/sessions')
    def create():
        try:
            data = request.get_json(silent=True) or {}
            sensitivity = data.get('sensitivity') or request.args.get('sensitivity') or manager.default_sensitivity
            return jsonify(manager.create(sensitivity=sensitivity).snapshot()), 201
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 429

    @bp.get('/api/audio/sessions/<key>')
    def status(key):
        try:
            return jsonify(manager.get(key).snapshot())
        except KeyError:
            return jsonify(error='Sesión finalizada'), 404

    @bp.post('/api/audio/sessions/<key>/chunks')
    def chunk(key):
        if request.content_length != 8000 * 4:
            return jsonify(error='Se requieren 8000 muestras float32 LE mono a 16000 Hz'), 400
        try:
            seq = int(request.headers.get('X-Audio-Sequence', ''))
            offset = int(request.headers.get('X-Audio-Offset', ''))
            if seq < 0 or offset != seq * 8000:
                raise ValueError('Secuencia u offset inválidos')
            samples = np.frombuffer(request.get_data(), dtype='<f4').copy()
            if not np.all(np.isfinite(samples)) or np.max(np.abs(samples)) > 1.05:
                raise ValueError('PCM inválido')
            session = manager.get(key)
            if session.snapshot()['state'] == 'error':
                return jsonify(session.snapshot()), 503
            accepted = session.submit(seq, offset, samples)
            return jsonify(session.snapshot()), 202 if accepted else 429
        except KeyError:
            return jsonify(error='Sesión finalizada'), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @bp.post('/api/audio/sessions/<key>/stop')
    def stop(key):
        manager.stop(key)
        return jsonify(state='detenido')

    @bp.get('/api/audio/status')
    def module_status():
        with manager.lock:
            n = len(manager.sessions)
        return jsonify(sesiones_activas=n, max_sesiones=manager.max_sessions,
                       sensibilidad=manager.default_sensitivity,
                       clasificador_ml=manager.ml_status,
                       alertas=len(manager.store.read()))

    @bp.post('/api/audio/sensitivity')
    def set_sensitivity():
        data = request.get_json(silent=True) or {}
        sens = data.get('sensitivity', 'media')
        if sens in ('alta', 'media', 'baja'):
            manager.default_sensitivity = sens
            return jsonify(ok=True, sensitivity=sens)
        return jsonify(error='Sensibilidad no válida (debe ser alta, media o baja)'), 400

    @bp.post('/api/audio/simulate')
    def simulate():
        """Genera y analiza la sesión sintética de final.py, registrando las alertas."""
        try:
            data = request.get_json(silent=True) or {}
            sens = data.get('sensitivity', 'media')
            audio, ground_truth = build_simulated_session()
            events, _ = analyze_session_full(audio, sr=SR, sensitivity=sens)
            
            # Registrar alertas de riesgo en el almacén
            now_iso = datetime.now(timezone.utc).isoformat()
            alerts_added = 0
            for ev in events:
                if ev['risk_level'] in ('ALTO', 'MEDIO'):
                    alert_id = 'audio-sim-' + uuid.uuid4().hex[:10]
                    manager.store.append(dict(
                        alert_id=alert_id,
                        event_id=alert_id,
                        type=ev['category'],
                        risk_type=f"{ev['category']}_acustica",
                        source='audio',
                        timestamp=now_iso,
                        stream_id='simulacion',
                        severity=ev['risk_level'],
                        description=ev['description'],
                        impact_offset_s=ev['start'],
                        risk_ref=f"Simulación UPAO final.py · [{ev['start']}s - {ev['end']}s]",
                        alarm_activated=ev['alarm_activated'],
                        technical=ev['metrics'],
                        imagen=None,
                    ))
                    alerts_added += 1

            return jsonify({
                "ok": True,
                "duration_s": round(len(audio) / SR, 2),
                "events": events,
                "ground_truth": ground_truth,
                "alerts_registered": alerts_added,
            })
        except Exception as exc:
            LOGGER.exception("Error en simulación")
            return jsonify(error=str(exc)), 500

    @bp.post('/api/audio/analyze-file')
    def analyze_file():
        """Analiza un archivo de AUDIO subido según el pipeline de final.py."""
        if 'file' not in request.files:
            return jsonify(error='No se envió ningún archivo'), 400
        file = request.files['file']
        if not file.filename:
            return jsonify(error='Nombre de archivo vacío'), 400

        # El sistema no acepta vídeos subidos: este análisis es solo de audio.
        if Path(file.filename).suffix.lower() not in AUDIO_SUFFIXES:
            return jsonify(error='Solo se aceptan archivos de audio '
                                 f'({", ".join(sorted(AUDIO_SUFFIXES))})'), 415

        sens = request.form.get('sensitivity', 'media')
        improve = request.form.get('improve_audio', 'true').lower() in ('true', '1', 'yes')

        suffix = Path(file.filename).suffix or '.wav'
        temp_f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            file.save(temp_f.name)
            temp_f.close()
            
            audio, duration_s = extraer_audio_archivo(temp_f.name, sr=SR)
            if improve:
                audio = mejorar_audio(audio, sr=SR)

            events, _ = analyze_session_full(audio, sr=SR, sensitivity=sens)

            # Registrar alertas de riesgo
            now_iso = datetime.now(timezone.utc).isoformat()
            alerts_added = 0
            for ev in events:
                if ev['risk_level'] in ('ALTO', 'MEDIO'):
                    alert_id = 'audio-file-' + uuid.uuid4().hex[:10]
                    manager.store.append(dict(
                        alert_id=alert_id,
                        event_id=alert_id,
                        type=ev['category'],
                        risk_type=f"{ev['category']}_acustica",
                        source='audio',
                        timestamp=now_iso,
                        stream_id=Path(file.filename).name[:16],
                        severity=ev['risk_level'],
                        description=ev['description'],
                        impact_offset_s=ev['start'],
                        risk_ref=f"Archivo {Path(file.filename).name} · [{ev['start']}s - {ev['end']}s]",
                        alarm_activated=ev['alarm_activated'],
                        technical=ev['metrics'],
                        imagen=None,
                    ))
                    alerts_added += 1

            return jsonify({
                "ok": True,
                "filename": file.filename,
                "duration_s": round(duration_s, 2),
                "events": events,
                "alerts_registered": alerts_added,
            })
        except Exception as exc:
            LOGGER.exception("Error analizando archivo de audio")
            return jsonify(error=str(exc)), 500
        finally:
            try:
                os.remove(temp_f.name)
            except Exception:
                pass

    return bp

