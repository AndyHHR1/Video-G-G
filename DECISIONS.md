# Decisiones de Diseño — Prevención de Caídas y Riesgos en Escaleras

Estado actual del sistema, funcionalidades disponibles, decisiones técnicas vigentes y limitaciones conocidas.

Este documento reúne en un solo lugar la información de `Caso.md`, las decisiones arquitectónicas, las elecciones de implementación y los resultados medidos. **Cada decisión está anclada al código real** con referencias `archivo.py:línea`.

---

## 1. Arquitectura General

```
Frame (cámara / imagen / vídeo)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Inferencer (threading persistente, app/server.py:109)   │
│  │                                                        │
│  │  YOLOv8 propio → persona / escalera     (RQF02)         │
│  │  YOLOv8 COCO   → obstáculos sobre escalón  (RQF02.1)   │
│  │  MediaPipe Pose → sujeción, distracción, tambaleo      │
│  └───────────────────────────────────────────────────────┘
│    │
│    ▼
│  RiskEngine.step()  (app/risk_engine.py:971)
│  │  1. Detección COCO de personas        _find_people() :917
│  │  2. Detección escaleras              self.det.predict() :1014
│  │  3. Análisis de pose                 _analyse_pose()  :398
│  │  4. Detección obstáculos             _obstacles()     :504
│  │  5. Clasificación posture            _classify_person() :941
│  │  6. Tracking IoU                     _track()         :345
│  │  7. Persistencia + histéresis        _persist() _forget_stale_risks()
│  │  8. Evaluación de alertas            _evalua_alerta() :597
│  |  9. Flush evidencia                  flush_evidence()  :640
│  └─ return dict con detecciones, postura, obstáculos, alertas, sistema
│
│  Inferencer._job() dibuja sobre el frame   _job() :202
│  Inferencer.flush_evidence() guarda alerta  flush_evidence() :640
│
│  Servidor devuelve JPEG (overlay=1) o JSON con cajas (overlay=0)
```

> **Hilo trabajador persistente**: `app/server.py:109-240` implementa `Inferencer`, una cola con un hilo dedicado. Flask atiende en multihilo (`threaded=True` en `app.run()`, `server.py:1035`), y cada hilo nuevo debe re-enlazar contexto CUDA (~82 ms vs ~29 ms). El `Inferencer._run()` carga el modelo dentro del hilo para que CUDA quede ligado a él y dure toda la vida del proceso.

---

## 2. Funcionalidades Disponibles

### 2.1 Modo Imagen
```python
# server.py:441
@app.post("/api/detect")
def api_detect():
```
- Subida de JPEG → detección + pose + obstáculos + dibujo con cajas.
- `force=True` ejecuta **todo** en un single frame (pose, personas, escalera, obstáculos) → necesario porque la cadencia normal salta inferencias (`server.py:465-466`).
- `allow_alerts=False` → una imagen fija **no genera alertas** (no es un suceso temporal, `risk_engine.py:1221`).
- Respuesta: JSON con `detections`, `summary` (`respuesta()`, `server.py:383`), `riesgo` global, y `image` (JPEG codificado en hex).

### 2.2 Modo Vídeo
```python
# server.py:771
@app.post("/api/video")
def api_video():
```
- Recibe MP4 → `cv2.VideoCapture` frame a frame (`server.py:825`).
- Cada frame: `_inferencer.detect(frame, ...)` (`server.py:829`).
- Output: H.264 (`_transcode_h264()` `server.py:748`) → `send_file()`.
- **Cadencias activas**: personas cada 3 frames (`people_every`, `risk_engine.py:241`), pose cada 2-4 frames (`pose_every`), escalera cada 12 (`stairs_every`), obstáculos cada 15 (`obstacle_every`).

### 2.3 Modo Cámara en Vivo (del navegador)
```python
# server.py:479
@app.post("/api/frame")
def api_frame():
```
- El navegador captura con `navigator.mediaDevices.getUserMedia()` → envía JPEG por POST.
- **Async mode** (`Inferencer.set_async()`, `server.py:520`): el navegador recibe el **último resultado** disponible sin bloquear, mientras el motor sigue trabajando a su ritmo (`DECISIONES_3.md` §5 — 110 fps de *throughput* async vs 18.6 fps síncrono).

### 2.4 Modo Cámara del Servidor (MJPEG)
```python
# server.py:862
def _mjpeg(camera_index: int, conf: float, imgsz: int):
```
- Stream MJPEG si el servidor tiene cámara física (`server.py:906`).
- Si no → `GET /api/stream` responde **503** con mensaje ("El servidor no tiene cámara. Usa el modo «Cámara de este equipo»") (`server.py:902`).

### 2.5 Alertas y Evidencia
```python
# risk_engine.py:597
def _evalua_alerta(self, tid: int, risk: str, det: dict) -> None:
```
- Alertas se guardan en `runs/alerts/<AAAA-MM-DD>/` con fotograma anotado + línea en `alertas.jsonl` (`risk_engine.py:657-698`).
- **Anti-avalancha**: `ALERT_COOLDOWN = 15.0s` por identidad y tipo (`risk_engine.py:88`).
- **Anonimización**: rostros pixelados con MediaPipe Face Detection antes de guardar (`risk_engine.py:703-730`).
- **Auto-cleanup**: conserva 7 días o 200 imágenes (`risk_engine.py:637-638`).

---

## 3. Taxonomía de Clases (6 clases)

| id | Clase | Origen | Riesgo | RQF |
|----|-------|--------|--------|-----|
| 0 | `persona_caido` | `fall:0` + Le2i `Lie` (91%) | ALTO | RQF02(4), RQNF05 |
| 1 | `persona_sentado` | `fall:2` | MEDIO | RQF03 |
| 2 | `persona_erguida` | `fall:1` + Le2i `Stand` + `tracking` | BAJO | RQF03 |
| 3 | `escalera` | `stairs:0` | CONTEXTO | — |
| 4 | `persona` | `tracking` | NEUTRO | entrada MediaPipe |
| 5 | `persona_desequilibrio` | Le2i `Likefall` (100%) | ALTO | RQF02(4) — pre-caída |

**Mapeo de clases** (`scripts/prepare_dataset.py:54-61`):
```python
CLASSES = {
    0: "persona_caido",
    1: "persona_sentado",
    2: "persona_erguida",
    3: "escalera",
    4: "persona",
    5: "persona_desequilibrio",
}
```

**Clases de riesgo** (`risk_engine.py:118-121`):
```python
FALL_CLASSES = {"persona_caido": "caida", "persona_desequilibrio": "perdida_equilibrio"}
NONERGO_CLASSES = {"persona_sentados": "postura_no_erguida"}
SAFE_CLASSES = {"persona_erguida"}
```

---

## 4. Motor de Riesgo (RiskEngine)

### 4.1 Tracking IoU (RQF05)
```python
# risk_engine.py:345
def _track(self, detections: list[dict]) -> list[dict]:
```
- IoU > 0.25 para asociar detecciones con tracks vivos (`risk_engine.py:363`, `_iou()` en `risk_engine.py:1271`).
- **Oclusión tolerada**: `OCCLUSION_SECONDS = 1.0` (`risk_engine.py:84`). Tras 1s sin detección, el track se elimina.
- Cada detección recibe un ID único creciente (`risk_engine.py:377-380`).
- **No se usa ByteTrack de Ultralytics**: arrastra estado interno de forma frágil entre llamadas. El seguimiento IoU propio (~30 líneas) es legible y auditable y cumple RQF05 (`DECISIONES_3.md` §4).

### 4.2 Geometría de postura (respaldo)
```python
# risk_engine.py:770-794
GEOM_CAIDO_INCLINA = 25.0      # inclinación eje tobillo-cadera > 25° = caído
GEOM_CAIDO_CABEZA = 0.55       # nariz/cadera normalizada < 0.55 = caído
GEOM_DESEQUILIBRIO_CABEZA = 0.75  # cabeza sobre cadera
GEOM_INESTABLE_INCLINA = 12.0  # >12° = pre-caída (85% caídas, 6% falsos positivos)
```
- Método: `_postura_por_geometria()` (`risk_engine.py:828`).
- MediaPipe Pose en modo **tracking** (`static_image_mode=False`, `risk_engine.py:307`) → pose estable entre frames.
- La geometría **manda sobre** el clasificador por recorte (validado: 88% de pie / 88% caído) (`risk_engine.py:1084`).

### 4.3 Señales de pose (RQF02.2, RQF02.3)
```python
# risk_engine.py:442-485
def _analyse_pose(self, image: np.ndarray) -> list[dict]:
```

| Señal | Criterio | Riesgo | Código |
|-------|----------|--------|--------|
| `sin_pasamanos` | wrist_low (ninguna muñeca sobre hombro) + arm_out (brazo écartado >1.2×) | MEDIO | `risk_engine.py:451` |
| `distraccion` | head_tilt > 0.06 + hand_at_head (<0.22w de nariz); **o** teléfono COCO cerca nariz | MEDIO | `risk_engine.py:463-472`, `1208` |
| `tambaleo` | lean > 0.09 o knee_asym > 0.10 | ALTO | `risk_engine.py:481` |

### 4.4 Obstáculos (RQF02.1)
```python
# risk_engine.py:127-131
OBSTACLE_RISK = {
    "backpack": "mochila", "handbag": "bolso", "suitcase": "maleta",
    "bottle": "botella", "book": "libro", "umbrella": "paraguas",
}
```
- Detector COCO separado (`yolov8s.pt`), `CONF_OBSTACLE = 0.25` (`risk_engine.py:54`).
- **Teléfono**: no es obstáculo → evidencia de distracción solo si caja cerca nariz (`risk_engine.py:1196-1214`).

### 4.5 Persistencia y Alertas (RQF06, RQF07)
```python
# risk_engine.py:76-82
PERSIST_SECONDS = {"ALTO": 1.2, "MEDIO": 0.6}  # duración mínima para alertar
PERSIST_GRACE = 1.0                            # histéresis: tolerancia apago
ALERT_COOLDOWN = 15.0                          # enfriamiento por track_id + riesgo
```
- **Doble umbral RQF04** (`risk_engine.py:47-48`): `CONF_PREFILTER = 0.75`, `CONF_CONFIRM = 0.85` — gobiernan la **confirmación**, no el disparo de alertas.
- **Puerta de alerta**: `CONF_ALERTA = 0.35` (`risk_engine.py:61`) — ajustado después de medir que con 0.75 no se disparaba ninguna alerta (medido: 0 alertas tras 25 frames).
- `_evalua_sobrevivientes()` (`risk_engine.py:576`): los riesgos mantenidos por histéresis se evalúan incluso sin detección en ese frame.

### 4.6 Anonimización (RQNF14, Ley 29733)
```python
# risk_engine.py:703
def _anonymise(self, frame: np.ndarray) -> np.ndarray:
```
- MediaPipe Face Detection (`model_selection=0`, `min_detection_confidence=0.4`).
- Pixelación: resize a 1/12 → resize de vuelta (`risk_engine.py:710-729`).

---

## 5. Entrenamiento

### 5.1 Configuración (`configs/train.yaml`)
```yaml
model: yolov8s              # 6 clases, 640px, ~100 FPS en GPU libre
epochs: 100
imgsz: 640
batch: 12
seed: 42                    # reproducibilidad (train.py:351-353)
device: 0
use_balanced: true          # balanceo por cajas, no por imágenes
max_boxes_per_class: 1500   # techo para clase dominante

# Augmentations (RQNF12: resiliencia a iluminación)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0
fliplr: 0.5
```

### 5.2 Balanceo por cajas (corrige bug anterior)
```python
# train.py:107-109
keep_rate[c] = 1.0 if n <= target_max else target_max / n
# Recorte por CONTAR CAJAS, no imágenes (bug en parte 2)
```
- Sin balanceo: 61:1 ratio `persona:persona_erguida`, 70% cajas `persona`.
- Con techo 1500: **14:1**, 22% `persona` (`DECISIONES_2.md` §2).

### 5.3 Métricas (modelo entregado: `runs/yolov8s_zoom/weights/best.pt`)
| Métrica | val | test |
|---------|-----|------|
| Precision | 0.791 | 0.767 |
| Recall | 0.772 | 0.719 |
| **mAP@0.5** | **0.810** | **0.747** |
| mAP@0.5:0.95 | 0.525 | 0.490 |

> **RQNF10** (mAP@0.5 > 75%): ✓ en val. En test 0.747 (0.003 del umbral). La clase que lo frena es `persona`, cuyo test es pseudo-etiquetado de detector. Sobre clases con GT humano: **0.756** (`scripts/train.py:213`).

### 5.4 Dataset
- **4744 imágenes** (2505 train / 867 val / 1372 test) (`DECISIONES.md` §5).
- Fuente: `fall/` (posturas), `stairs/` (escaleras), `tracking/MOT17` (personas CCTV), `Le2i` (pre-caída).
- **Fuga 0** verificada por md5 (`DECISIONES.md` §5).
- Le2i procesado con `--exclude Fall,Lie` (ambas clases excluidas del entrenamiento; `DECISIONES_3.md` §2.3).

---

## 6. Estado Operativo del Sistema (RQNF09, RQNF26)
```python
# risk_engine.py:875
def _estado_operativo(self) -> str:
```
| Estado | Condición |
|--------|-----------|
| **OPERATIVO** | Modelo cargado, cámara conectada, 0 fallos recientes |
| **DEGRADADO** | Sin cámara, o fallos recientes (GPU/pose/detección) |
| **NO OPERATIVO** | Modelo no cargado (`det`/`obs` es None) |

- `GET /api/health` expone estado + detalle (`server.py:968-981`).
- Panel muestra estado con color, refresca cada 3s.

---

## 7. Autenticación (RQNF19)
```python
# server.py:921
PANEL_TOKEN = os.environ.get("PANEL_TOKEN", "")
```
- **Secreto compartido**, `hmac.compare_digest` (tiempo constante) (`server.py:946`).
- Nunca en el repo: si no está definido, panel abierto y `/api/health` lo indica (`server.py:980`).
- `/api/stream` exento de token solo para `127.0.0.1` (`server.py:943`).

---

## 8. Rendimiento

| Operación | Tiempo | Notas |
|-----------|--------|-------|
| Inferencia YOLOv8s | ~16 ms | GPU RTX 5050, imgsz 640 |
| MediaPipe Pose (CPU) | ~18 ms | Modo tracking; CPU porque no hay ruta GPU |
| Detección obstáculos | ~14 ms | Cada 15 frames |
| Frame sin persona | 35-46 fps | GPU libre |
| Frame con persona | 8-13 fps | MediaPipe bottleneck |
| Cámara en vivo (HTTP) | **~18.6 fps** | Async mode, 41 ms/ciclo |

> **RQNF01 (≥30 fps)**: el motor completo mide **18-20 fps con personas**. No se cumple de forma sostenida, `DECISIONES_3.md` §7. La latencia (<500 ms) sí se cumple con holgura (~41 ms).

---

## 9. Audio Sincronizado con Alertas (RQNF-AUDIO)

> Se extrae un clip de audio de **0.5 s antes de la caída hasta 0.5 s después** (`AUDIO_PRE_FALL = 0.5`, `AUDIO_POST_FALL = 0.5` en `audio_extractor.py:67-68`) como evidencia adicional de cada alerta.

**Modos de operación:**

| Modo | Fuente de audio | Implementación |
|------|----------------|----------------|
| **Video** (`/api/video`) | Pista de audio del archivo MP4/ AVI subido | `extract_video_audio()` usa ffmpeg (vía `imageio-ffmpeg`) para extraer 0.5s antes → 0.5s después del timestamp del frame (`audio_extractor.py:69-117`) |
| **Cámara en vivo** (`/api/frame`, `/api/browser-camera`) | Micrófono del navegador | `getUserMedia({audio:true})` captura audio; `MediaRecorder` envía chunks WAV al endpoint `/api/audio`; `LiveAudioBuffer` mantiene un ring buffer de 10 s (`audio_extractor.py:141-146`) |
| **Cámara del servidor** (`/api/stream`) | No disponible | El servidor no tiene micrófono asociado al dispositivo de video |

**Integración con alertas:**
- `audio_manager` (singleton en `audio_extractor.py:310`) se importa en `server.py:97`.
- En `_job()` (`server.py:234`), justo después de `flush_evidence()`, se llama a `_extract_alert_audio()` que registra el timestamp y extrae el clip.
- La ruta del clip se guarda en `alert.audio_evidence` (`risk_engine.py:174`), que se serializa en `alertas.jsonl`.
- Los archivos de audio se sirven en `/api/alerta-audio/<dia>/<nombre>` y aparecen como reproductores `<audio>` en el panel de alertas.
- Al borrar alertas (`DELETE /api/alertas`), también se eliminan los archivos de audio asociados (`server.py:715-722`).

**Limitaciones:**
- El navegador debe conceder permiso de micrófono. Si se niega, el sistema funciona sin audio (fallback a `audio:false`).
- El ring buffer de audio en vivo mantiene 10 s de historia: si la caída se detecta con más de 10 s de retraso, el pre-fragmento no está disponible.

---

## 10. Limitaciones Conocidas

| Limitación | Evidencia |
|------------|-----------|
| `persona_desequilibrio` (pre-caída) es la peor clase | mAP50 test: 0.368 (`DECISIONES.md` §7) |
| 8% falsos positivos `persona_caido` en personas de pie | `DECISIONES_3.md` §16; calibrado con gente de frente, no desde ángulo de escalera |
| Tracking IoU: identity swap en cruces | No hay ReID; 4% error en cruces simulados |
| Sin webcam del servidor | `/api/stream` responde 503 si no hay cámara (`server.py:902`) |
| RQNF10 test: 0.747 (0.003 del 75%) | Clase `persona` con pseudo-etiqueta en test |
| RQNF19: auth por secreto compartido, no usuarios/roles | Diseño consciente y documentado |
| RQNF11 (horario campus) / RQNF27 (actuación física) | Dependen del despliegue físico, no del código |

---

## 11. Reproducibilidad

```bash
# Entorno (nota: RTX 5050 Blackwell → CUDA 12.8)
python3 -m venv venv
./venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128
./venv/bin/pip install -r requirements.txt

# Pipeline completo
./venv/bin/python scripts/prepare_le2i.py --exclude Fall,Lie
./venv/bin/python scripts/prepare_dataset.py --clean
./venv/bin/python scripts/add_zoom_variants.py --split train
./venv/bin/python scripts/train.py
./venv/bin/python scripts/evaluate.py --weights runs/yolov8s_zoom/weights/best.pt

# Servidor web
PANEL_TOKEN="secreto-aleatorio" ./venv/bin/python app/server.py --host 0.0.0.0 --port 5000
```

- Semilla fija: 42 (Python, NumPy, Torch, Ultralytics) (`train.py:351-353`).
- Trazabilidad: `datasets/combined/manifest.csv`, `datasets/combined/config.json`, `datasets/le2i_yolo/validacion.json`.
