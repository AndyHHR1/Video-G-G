# Registro de decisiones — rama `expo`

Cambios hechos al pasar de `semana-7` a `expo`, con el motivo y la evidencia.
El modelo (Modelo S7 v4) **no se tocó**.

## 1. Reorganización modular

| Antes (`semana-7`) | Ahora (`expo`) |
|---|---|
| `app/server.py` (1 217 líneas: HTTP, hilos, dibujo, vídeo, auth) | `backend/api/*`, `backend/services/*`, `backend/rendering/draw.py` |
| `app/risk_engine.py` (1 385 líneas) | `backend/perception/{engine,detectors,pose,tracking,alerts,catalog}.py` + `backend/privacy/anonymizer.py` |
| `app/real_esrgan.py` + `build_esrgan.py` (ONNX) | `backend/enhancement/{archs,super_resolution}.py` (PyTorch puro) |
| `app/templates/` | `frontend/templates/` (contenido idéntico) |
| `scripts/`, `configs/train*.yaml` | `training/`, `training/configs/` |
| Constantes en el código | `configs/app.yaml` |
| `runs/yolov8s_zoom/weights/best.pt` (base v3) | `models/base/yolov8s_zoom_v3.pt` |
| — (modelo v4 solo en el release) | `models/detector/escaleras_s7_v4.pt` |
| `requirements.txt` (pines para Python ≥ 3.11) | `requirements/{base,torch-cu128,torch-cpu,dev}.txt` (Python 3.10–3.12) |

**Verificación de equivalencia.** Se ejecutaron el motor de `semana-7` y el
nuevo sobre la misma secuencia de 120 fotogramas con las mismas cadencias.
Con la corrección 2.1 desactivada, los conteos de riesgo y de clases son
**idénticos** (`MEDIO 112 / ALTO 8`; `persona_caido·MEDIO 242`,
`persona·NEUTRO 183`).

## 2. Errores corregidos

### 2.1 La pose nunca se emparejaba con la persona correcta
`_best_person_match` leía `self._last_pose_landmarks`, que **nunca se
asignaba**. Con más de una persona en escena, la geometría de pose no se
aplicaba a nadie y todas las personas dependían del clasificador por recorte,
que confunde «de pie» con «caído». Ahora se guardan los keypoints de la última
pose y se emparejan por IoU.

Efecto medido en la misma secuencia: `persona_caido` (falsos, eran personas de
pie) baja de 242 a 139 detecciones y aparecen 109 `persona_erguida` correctas.

### 2.2 La evidencia con rostros no se guardaba
El anonimizador usaba `bb.xmax`/`bb.ymax`, atributos que **no existen** en
MediaPipe (`RelativeBoundingBox` tiene `xmin`, `ymin`, `width`, `height`). Cada
vez que se detectaba una cara, se lanzaba una excepción y la alerta quedaba
**sin imagen ni línea de registro**.

### 2.3 El modelo de rostros no veía caras a distancia de cámara
Solo se usaba el modelo de corto alcance (0–2 m). Ahora: corto + largo alcance,
búsqueda en el recorte de cada persona y pixelado de la cabeza como respaldo.

### 2.4 La evidencia en vivo se guardaba sin cajas
En la cámara en vivo el servidor no dibuja (lo hace el navegador), así que la
imagen de la alerta salía sin anotaciones. Ahora se dibuja una copia solo
cuando hay una alerta pendiente.

### 2.5 Peticiones que podían quedarse colgadas
Al descartar fotogramas atrasados de la cola, también se podían descartar
trabajos **síncronos** (imagen o vídeo subido) cuya petición seguía esperando.
Ahora solo se descartan trabajos asíncronos de la cámara en vivo.

### 2.6 El servidor arrancaba con el modelo equivocado
`--weights` apuntaba por defecto al modelo base v3; el v4 solo existía en el
release. Ahora `configs/app.yaml` apunta al Modelo S7 v4.

### 2.7 El estado del sistema nunca mostraba fallos de pose/detección
Los contadores de fallo se ponían a cero antes de calcular el estado. Ahora el
estado del fotograma refleja sus fallos y luego se reinician.

## 3. Mejora de vídeo en vivo fluida

Ver README §5.4 y §7. Resultado: de ~2 fps (ONNX RRDB a 320 px) a ~29 fps con
`realesr-general-x4v3` en FP16, en un hilo separado y emparejado por
`frame_id`.

## 4. Entorno

- Lambda Stack: `flatbuffers` de Debian con versión inválida → se instala uno
  válido en el venv.
- Lambda Stack: PyTorch de sistema sin `sm_120` → torch cu128 en el venv.
- Lambda Stack: TensorFlow de sistema importado por mediapipe rompe torch si
  carga antes → `backend/__init__.py` importa torch primero.
- `~/.local` se colaba en el venv con numpy 1.21 → `PYTHONNOUSERSITE=1` en los
  scripts.
- mediapipe arrastraba `opencv-contrib-python` 4.11 → pin a 4.10.0.84, igual
  que `opencv-python`.
