"""Módulo de AUDIO — percepción acústica en escaleras.

Procede del repositorio del grupo de audio (https://github.com/brigitteve/percepcion).
Solo se trajo su parte acústica; su versión antigua de vídeo NO se migró.

Funciones (todas activas):

* Audio en vivo del micrófono del navegador (AudioWorklet, PCM 16 kHz, bloques
  de 500 ms) con detección de pisadas (caminando/corriendo), impacto súbito de
  caída, gritos o pedido de auxilio (pYIN), vúmetro y métricas en vivo.
* Clasificador ML de caídas (`audio_fall_detector`: MLP calibrado de 6 clases,
  vetos, evaluación posterior y fusión) en paralelo al detector por reglas.
* Sensibilidad alta / media / baja (RF-07 de final.py).
* Demostración con la sesión sintética de final.py (con ground truth).
* Análisis de un archivo de audio (o la pista de audio de un archivo) con
  mejora de audio opcional.
* Alertas acústicas persistentes (JSONL), integradas en el panel de alertas,
  la descarga ZIP y el borrado.

Independencia del vídeo: no importa nada de `backend/`. Se comunica con el
resto solo mediante sus rutas `/api/audio/*` y la fuente de alertas que
registra en `runtime.alert_sources`.
"""
from __future__ import annotations

import threading
from pathlib import Path

from flask import Blueprint

HERE = Path(__file__).resolve().parent


def install(app, settings, rt, options: dict) -> dict:
    from .audio_live import AudioEventStore, AudioSessionManager, audio_blueprint

    store = AudioEventStore(settings.path(
        options.get("alerts_file", "outputs/audio_alerts/alertas.jsonl")))
    manager = AudioSessionManager(
        store,
        max_sessions=int(options.get("max_sessions", 4)),
        default_sensitivity=str(options.get("sensitivity", "media")),
        use_ml=bool(options.get("ml_detector", True)))

    app.register_blueprint(audio_blueprint(manager))
    # Archivos del panel (JS, CSS, worklet) y su plantilla HTML.
    app.register_blueprint(Blueprint(
        "audio_ui", __name__, static_folder=str(HERE / "static"),
        static_url_path="/modulos/audio/static",
        template_folder=str(HERE / "templates")))

    rt.alert_sources.append(store)
    if rt.lifecycle is not None:
        rt.lifecycle.on_shutdown("módulo de audio", manager.close)

    status = {"ok": True, "estado": "activo (precalentando)"}

    def _warm():
        try:
            manager.warmup()
            status["estado"] = f"activo · clasificador ML {manager.ml_status}"
        except Exception as exc:                            # noqa: BLE001
            status["estado"] = f"activo sin precalentar ({exc})"

    # En segundo plano: el arranque del vídeo no espera a librosa/numba.
    threading.Thread(target=_warm, name="audio-warmup", daemon=True).start()
    return status
