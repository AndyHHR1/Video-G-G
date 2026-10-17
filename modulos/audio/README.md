# Módulo de audio — percepción acústica en escaleras

Origen: <https://github.com/brigitteve/percepcion> (grupo de audio). Se trajo
**solo la parte acústica**; la versión antigua de vídeo de ese repositorio
(`app/server.py`, `app/risk_engine.py`, su `index.html`, `runs/`, `yolov8s.pt`)
**no se migró**: el vídeo de este proyecto es el de `backend/`.

## Qué hace (todo activo)

| Función | Dónde |
|---|---|
| Micrófono en vivo del navegador: AudioWorklet → PCM mono 16 kHz, bloques de 500 ms | `static/audio-worklet.js`, `static/audio-live.js` |
| Panel «🎙 Audio»: vúmetro, métricas en vivo, chip de estado, sensibilidad, pitido, notificaciones del SO, alertas | `static/audio-panel.js`, `static/audio-panel.css`, `templates/audio/panel.html` |
| Detector por reglas de `final.py`: pisadas/cadencia (caminando, corriendo), impacto súbito (pico RMS + factor de cresta), grito / pedido de auxilio (F0 con pYIN) | `acoustic_perception.py` |
| Clasificador ML de caídas: disparador RMS, MLP calibrado de 6 clases (caída, portazo, objeto que cae, pasos, voz/TV, ruido), vetos, evaluación de los 7 s posteriores y fusión | `audio_fall_detector/` (modelo `models/fall_classifier.joblib`, **sin reentrenar**) |
| Sesiones acotadas, cola, cooldown por tipo de alerta, simulación de `final.py`, análisis de archivo de audio | `audio_live.py` |
| Integración con el panel: rutas, archivos estáticos, fuente de alertas, parada | `__init__.py` (`install`) |

Las reglas y el MLP analizan el **mismo** audio en paralelo
(`CombinedAudioDetector`): las reglas responden en cada bloque y alimentan las
métricas; el MLP espera los 7 s posteriores reales al impacto, como lo diseñó
el grupo. Las alertas (solo metadatos, nunca audio) se guardan en
`outputs/audio_alerts/alertas.jsonl` y aparecen también en el panel de alertas
de la cámara, en el ZIP y en el borrado.

## Independencia del vídeo

No importa nada de `backend/`. `backend/app.py` llama a
`modulos.audio.install(...)` dentro de un `try`: si faltan sus dependencias, la
pestaña no aparece y el vídeo sigue igual. Su procesamiento es CPU (librosa,
scikit-learn) con BLAS limitado a un hilo durante el MLP.

## Rutas

`/api/audio/sessions` (POST), `/api/audio/sessions/<id>` (GET),
`/api/audio/sessions/<id>/chunks` (POST, 32000 bytes float32 LE),
`/api/audio/sessions/<id>/stop` (POST), `/api/audio/sensitivity` (POST),
`/api/audio/status` (GET), `/api/audio/simulate` (POST),
`/api/audio/analyze-file` (POST, solo archivos de audio).

## Dependencias y pruebas

```bash
pip install -r requirements/modulos.txt
python -m pytest modulos/audio/tests
node --test modulos/audio/tests/audio-worklet.test.cjs
```

El grupo fijó `scikit-learn==1.9.1`, que exige Python ≥ 3.11. Con Python 3.10 se
usa 1.7.2; `tests/test_audio_model.py` comprueba que los pesos (SHA-256) y las
predicciones de referencia son idénticos.

`referencia/` guarda el cuaderno original (`final_colab.py`) y el informe de
integración del grupo, como documentación.

## Limitaciones (del grupo, se mantienen)

Las pruebas usan señales sintéticas. Los umbrales de `final.py` no están
calibrados en una escalera real y los índices técnicos no son probabilidades
clínicas: toda alerta acústica requiere verificación humana.
