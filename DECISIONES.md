# Decisiones de diseño — Prevención de Caídas y Riesgos en Escaleras

Registro de las decisiones tomadas durante el desarrollo, con la referencia a
`Caso.md` que las justifica. Todos los scripts son reproducibles con semilla
fija (42).

---

## 1. Entorno

| Decisión | Motivo (`Caso.md`) |
|---|---|
| `torch==2.11.0+cu128` | La RTX 5050 es Blackwell (**sm_120**). Las builds con cu121 no incluyen sm_120 y fallan al ejecutar en la GPU. Verificado: `sm_120` aparece en `torch.cuda.get_arch_list()`. |
| venv propio en `venv/` | Reproducibilidad (`RQNF05` reproducibilidad, `RQNF03` entorno con 6 GB VRAM). |
| `batch` reducido a 8-12 y `workers` a 2 | La máquina tiene **7.9 GB de RAM**; con `workers=4` y batch 16 el proceso moría por OOM-killer. |

---

## 2. Taxonomía de clases

`Caso.md` §3.2.2 (RQF02) define **4 categorías de riesgo**, y RQF03 pide
discriminar tránsito seguro de conducta de riesgo. Los datasets disponibles no
las cubren por completo, así que se decidió el alcance que las cajas reales
permitían:

| id | Clase | Origen | Justificación |
|----|-------|--------|---------------|
| 0 | `persona_caido` | `fall:0` | Post-caída: persona tendida en el suelo. **`RQNF05` exige "distinguir si una persona se ha caído o no"**, y esta es la clase de evidencia. |
| 1 | `persona_sentado` | `fall:2` | Postura no ergonómica. |
| 2 | `persona_erguida` | `fall:1`, Le2i `Stand` | **`RQNF03` "postura erguida"** = tránsito seguro. |
| 3 | `escalera` | `stairs:0` | Contexto de la escena; el sistema opera sobre escaleras. |
| 4 | `persona` | `tracking` | Detección base de pedestrian para alimentar a MediaPipe (RQNF08: integração MediaPipe). |
| 5 | `persona_desequilibrio` | Le2i `Likefall` | **`RQNF02` "(4) Caída activa o pérdida inminente de equilibrio"** — la clase que el objetivo del proyecto exige. |

### Cómo se reconstruyó la taxonomía de `fall`

No existe `data.yaml`, `README` ni definición de clases en el repo ni en el
historial git. Los 3 ids se dedujeron combinando geometría de caja e
inspección visual, y se verificó que la interpretación era consistente:

| id | h/w mediano | Interpretación | Verificación visual |
|----|------------|----------------|--------------------|
| 0 | 0.93 (ancha) | persona caída | hombre tendido en el suelo |
| 1 | 2.99 (alta) | de pie | mujer de pie con aspiradora |
| 2 | 1.59 | sentada | mujer sentada |

Corroboración adicional: 252 de 278 imágenes `fall*.jpg` contienen la clase 0.

---

## 3. Decisión sobre `tracking/` (MOT17)

**Problema:** son 11 283 frames de seguimiento de peatones (ETH, KITTI, TUD,
PETS, ADL, Venice, AVG). No hay caídas ni escaleras, y sus etiquetas son
**mixtas**:

- 7 secuencias con GT real de benchmark (layout `frame,id,x,y,w,h,1,-1,-1,-1`)
- 4 secuencias con **pseudo-etiquetas CenterTrack** (`...,conf,vx,vy,score`)
- Las 11 secuencias de `test` solo tienen `det/det.txt`, que es **salida de
  detector**, no ground truth.

**Decisiones tomadas:**

1. **Se usa el dataset completo** (las 22 secuencias aportan), fulfilling la
   indicación de no descartar ningún dataset. Solo se degrada la *fuente* de
   etiqueta, no la cantidad de datos.
2. **Split por secuencia, nunca por frame.** Frames contiguos son casi
   idénticos; un split por frame habría creado una fuga enorme. Reparto
   final: 8 secuencias train / 3 val / 11 test.
3. **Muestreo temporal 1 de cada 5 frames**, por redundancia temporal.
4. **Umbral relativo (score ≥ mediana) y no absoluto.** Los scores van de 0 a
   ~140, así que un corte tipo 0.5 no discrimina nada.
5. Se reportan **dos métricas** para `persona` (ver §6).

---

## 4. Decisión sobre las cajas de Le2i (pseudo-etiquetación)

Le2i (citado en `Caso.md` §1.4) es un dataset de **clasificación**: sus
carpetas están etiquetadas por estado y **no trae bounding boxes**. Sus
clases son `Blank`, `Fall`, `Likefall`, `Lie`, `Stand`, donde `Likefall` es
*"the center of gravity was unstable"* — exactamente la pre-caída.

**Las cajas se generaron con un detector de personas preentrenado en COCO**
(yolov8n). Se añadió una **etapa de validación** que aborta si la
pseudo-etiquetación no es coherente, y esa validación encontró dos fallos
reales que se corrigieron:

| Fallo detectado | Causa | Corrección |
|-----------------|-------|------------|
| 3-5 "personas" por frame; `Blank` con 100% de detecciones | No se filtraba por clase: se contaban sillas, mesas y televisores de COCO | `classes=[0]` (solo `person`) → `Blank` bajó a 0% y a 1.00 cajas/frame |
| `Fall` 61% / `Lie` 66% de detección | Una persona **tendida en el suelo** está muy escorzada y los detectores COCO se entrenan con gente de pie | Ver abajo |

**Decisión final:** se usan `Likefall` (100%), `Stand` (99%) y `Blank` (0% de
falsos positivos). **Se descartan `Fall` y `Lie`**, porque bajar el umbral de
0.35 a 0.20 solo los mejoró del 56%/58% al 61%/66%: es una limitación real del
approach, no un ajuste de parámetro. No se pierde cobertura porque
`persona_caido` ya la aporta el dataset `fall` **con anotación humana**.

**Resultado: 357 cajas de `persona_desequilibrio` en train** (frente a 0
antes). La geometría confirma la semántica: altura de caja 0.445 para
desequilibrio frente a 0.556 para erguida — el centro de gravedad más bajo.

---

## 5. Integridad del dataset

Verificado programáticamente, sin fugas:

| Comprobación | Resultado |
|---|---|
| Misma imagen (md5) en >1 split | **0** |
| Misma secuencia de `tracking` en >1 split | **0** |
| Mismo grupo Roboflow de `stairs` en >1 split | **0** |
| Mismo video de Le2i en >1 split | **0** |
| Imágenes corruptas | **0** (18 058 cabeceras verificadas) |

Detalles que hubo que corregir:

- **Fuga real en `fall`**: `train/fall022.jpg` ≡ `val/fall012.jpg`. Resuelta
  por md5 antes de repartir.
- **25 duplicados exactos** en los datasets originales, más **536** en Le2i
  (Le2i corta la misma escena estática en varias carpetas: `Coffee_room_v10`,
  `v11`, `v13`... tienen frames byte-idénticos).
- Se añadió un **barrido de seguridad** (`scrub_cross_split_duplicates`) que
  elimina cualquier imagen presente en más de un split, como red de seguridad
  general.
- Reparto **estratificado** en `fall` por la clase más rara presente, porque
  el reparto original dejaba ~10 cajas de `persona_sentado` en val.

Dataset final: **4744 imágenes**, 6 clases. 2505 train / 867 val / 1372 test.

---

## 6. Métrica doble para la clase `persona`

La clase `persona` se evalúa en test contra `det/det.txt`, que es salida de un
**detector**, no GT. El modelo predice *mejor* que ese pseudo-ground-truth, así
que incluirlo baja la métrica de forma artificial y no reflects calidad real.

Por eso se reportan ambas cifras:

| Split | mAP50 (todas) | mAP50 (sin `persona`, solo GT humano) |
|---|---|---|
| val | 0.8003 | **0.8250** |
| test | 0.7055 | **0.7560** |

La diferencia es de ~5 puntos en test. `RQNF10` (mAP@0.5 > 75%) **se cumple
en val con ambas cifras y en test solo con la métrica de GT humano**.

---

## 7. Cumplimiento de `Caso.md`

| Requisito | Resultado |
|---|---|
| `RQNF01` ≥30 FPS, latencia <500 ms | **CUMPLE**: 22.7 ms/frame, 44 FPS |
| `RQNF03` ≥6 GB VRAM | **CUMPLE**: RTX 5050, 8.1 GB |
| `RQNF05` distinguir caída/no caída | **CUMPLE**: `persona_caido` mAP50 0.90 (test) |
| `RQNF10` mAP@0.5 > 75% | **CUMPLE en val (0.810) y en test con GT humano (0.756)** |
| `RQNF12` resiliencia a iluminación | **CUMPLE**: augmentations HSV reforzadas (`hsv_s=0.7`, `hsv_v=0.4`) en `configs/train.yaml:43` |
| `RQF02` (4) pre-caída | **Parcial**: la clase existe y funciona, pero ver §8 |
| `RQF02` (1) obstáculos, (2) pasamanos, (3) distracción | **NO CUBIERTO**: no hay datos. Según el diseño de `Caso.md` son keypoints de MediaPipe, no cajas de YOLO |
| `RQNF-AUDIO` evidencia de audio sincronizado | **CUMPLE**: extracción 0.5s antes → 0.5s después (`audio_extractor.py:67-68`, 117-264) |

---

## 8. Decisión: extracción de audio sincronizada con alertas (RQNF-AUDIO)

**Problema:** las alertas de caída solo incluían una imagen anotada. Se pidió
evidencia de audio de **0.5 s antes de la caída hasta 0.5 s después**.

**Decisiones:**

1. **Video** (`/api/video`): se extrae del archivo original con ffmpeg. El
   timestamp del frame se calcula como `frame_index / fps` (no epoch), porque
   el modelo corre frame-a-frame y el tiempo real del servidor no corresponde
   al tiempo del video. Implementado en `server.py:878-881` (marca fps y
   frame en el motor) y `_extract_alert_audio()` en `server.py:393-410`.

2. **Navegador en vivo** (`/api/frame`): el frontend captura audio con
   `getUserMedia({audio:true})` y `MediaRecorder` (chunks de 200 ms), enviándolos
   a `/api/audio`. El servidor mantiene un ring buffer de 10 s
   (`LiveAudioBuffer`, `audio_extractor.py:141`). Al dispararse una alerta,
   `extract_clip()` recorta la ventana 0.5 s antes → 0.5 s después.

3. **Cámara del servidor** (`/api/stream`): el servidor no tiene micrófono
   asociado a la cámara física. El audio no está disponible en este modo.
   La UI muestra `🔇 (sin audio)`.

4. **Formato**: WAV mono, 16 kHz, 16-bit. El archivo se guarda en
   `runs/alerts/<fecha>/audio_<alert_id>.wav`, al lado de la imagen de
   evidencia. La ruta se almacena en `alert.audio_evidence`
   (`risk_engine.py:175`).

5. **Limpieza**: al borrar alertas, también se borran los archivos de audio
   asociados (`server.py:718-725`).

---

## 10. El problema real que queda abierto

La clase `persona_desequilibrio` — el objetivo del proyecto — es la **peor
clase** y la matriz de confusión muestra exactamente por qué:

- **80% del `persona_desequilibrio` real se predice como `persona_erguida`**
- Solo el **20%** se clasifica correctamente

Causas identificadas:

1. **Volumen**: 357 cajas en train, frente a 825 de `persona_erguida`. Y solo
   **25 instancias en test**, así que la métrica es además poco fiable.
2. **Ambigüedad de la etiqueta**: en Le2i, `Likefall` y `Stand` los separa una
   frontera difusa (centro de gravedad), no una distinción visual nítida.
3. **Dominio**: las escenas de Le2i son interiores (`coffee`, `home`), no
   escaleras. El caso de uso real es una cámara sobre una escalera.

**El dato honesto:** el modelo detecta bien *que hay una persona* y bien
*que está en el suelo* (0.90), pero **distinguir pre-caída de tránsito normal
sigue siendo el punto débil**. Es coherente con que los tres datasets
originales no contenían ninguna imagen de pre-caída: la clase se ha
construido desde cero a partir de un dataset externo.

Siguiente paso natural: más datos de pre-caída en contexto de escalera
(grabación propia o PPGIA-UNIFOR, también citado en `Caso.md` §1.4).

---

## 11. Reproducibilidad

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt   # ojo: torch+cu128

python3 scripts/audit_datasets.py                      # Paso 1: auditoría
python3 scripts/prepare_le2i.py                        # pseudo-etiqueta + valida
python3 scripts/prepare_dataset.py --clean             # Paso 2: unificación
python3 scripts/train.py                               # Paso 3: entrenamiento
python3 scripts/evaluate.py --weights runs/yolov8s_6clases/weights/best.pt
```

- Semilla fija: 42 (Python, NumPy, Torch, Ultralytics).
- `datasets/combined/manifest.csv`: trazabilidad completa de cada imagen
  (split, origen, grupo, ruta original, nº de cajas).
- `datasets/combined/config.json`: todos los parámetros de la preparación.
- `datasets/combined/dropped.txt`: todo lo que se descartó y por qué.
- `datasets/le2i_yolo/validacion.json`: resultado de la validación de
  pseudo-etiquetas.
