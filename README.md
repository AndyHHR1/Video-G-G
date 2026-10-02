# Prevención de Caídas y Riesgos en Escaleras

Detector de riesgo de caídas en escaleras con **YOLOv8** (Ultralytics),
**MediaPipe Pose** y un segundo detector COCO para obstáculos, más una
plataforma web local para probarlo con imagen, vídeo o cámara en vivo.

El sistema está diseñado contra los requisitos del documento `Caso.md`
(IEEE 830 / ISO 25010): 8 funcionales (RQF01–RQF08) y 27 no funcionales
(RQNF01–RQNF27).

---

## Estado rápido

| Métrica | val | test |
|---|---|---|
| Precision | 0.791 | 0.767 |
| Recall | 0.772 | 0.719 |
| **mAP@0.5** | **0.810** | **0.747** |
| mAP@0.5:0.95 | 0.525 | 0.490 |

Modelo entregado: `runs/yolov8s_zoom/weights/best.pt` (yolov8s, 6 clases).

`RQNF10` pide mAP@0.5 > 75%: **se cumple en validación (0.810)** y queda en
**0.747 en test**, a 0.003 del umbral. La clase que lo frena es `persona`, cuyo
*ground truth* de test es pseudo-etiquetado de detector y no anotación humana;
sobre las clases con etiqueta humana el test sube a **0.756**.

### Clases detectadas

| id | Clase | Riesgo | Referencia |
|----|-------|--------|------------|
| 0 | `persona_caido` | ALTO | RQF02 (4) · RQNF05 |
| 1 | `persona_sentado` | MEDIO | RQF03 |
| 2 | `persona_erguida` | BAJO | RQF03 |
| 3 | `escalera` | contexto | — |
| 4 | `persona` | neutro | entrada a MediaPipe |
| 5 | `persona_desequilibrio` | ALTO | RQF02 (4) — **pre-caída** |

Además, el motor de riesgo deriva tres categorías que **no son cajas** sino
condiciones de postura, con MediaPipe Pose y un detector COCO:

| Señal | Riesgo | Referencia |
|-------|--------|------------|
| `obstaculo_escalon` | ALTO | RQF02 (1) |
| `sin_pasamanos` | MEDIO | RQF02 (2) |
| `distraccion` (móvil / lectura) | MEDIO | RQF02 (3) |
| `tambaleo` | ALTO | RQF02 (4) |

---

## Puesta en marcha

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
```

> **Atención a la GPU.** La RTX 5050 es Blackwell (`sm_120`), que solo
> funciona con wheels de PyTorch compilados para CUDA 12.8:
>
> ```bash
> ./venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
>     torch==2.11.0+cu128 torchvision==0.26.0+cu128
> ./venv/bin/pip install -r requirements.txt
> ```
>
> Con las builds por defecto, torch cae a `sm < 120` y falla al ejecutar.

### Regenerar todo desde cero

```bash
./venv/bin/python scripts/prepare_le2i.py --exclude Fall,Lie
./venv/bin/python scripts/prepare_dataset.py --clean
./venv/bin/python scripts/add_zoom_variants.py --split train
./venv/bin/python scripts/train.py
./venv/bin/python scripts/evaluate.py --weights runs/yolov8s_zoom/weights/best.pt
```

### Plataforma web

```bash
./venv/bin/python app/server.py --host 0.0.0.0 --port 5000
```

Modos: **imagen**, **vídeo**, **cámara en vivo** y **clases/requisitos**.

Para exponerla en la red local (por ejemplo desde el móvil) hace falta
`--host 0.0.0.0`. **No hay cifrado**: úsala solo en redes de confianza.

Autenticación (`RQNF19`): se activa definiendo `PANEL_TOKEN`; sin esa variable
el panel arranca abierto y lo indica en `/api/health`.

```bash
PANEL_TOKEN="un-secreto-largo-y-aleatorio" ./venv/bin/python app/server.py
```

---

## Datasets utilizados y créditos

Todo lo que se ha descargado y usado, con su origen y licencia de referencia.

### Datasets del enunciado (facilitados)

| Dataset | Uso | Origen |
|---------|-----|--------|
| **Fall Detection Dataset** | clase `persona_caido` y `persona_sentado`; anota cajas | <https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset> |
| **Stairs** | clase `escalera`; 959 imágenes con caja de escalera | <https://www.kaggle.com/datasets/samuelayman/stairs> |
| **MOT15 / MOT17 Challenge** | clase `persona`: peatones de CCTV, lejanos y de espaldas | <https://www.kaggle.com/datasets/mdraselsarker/mot15-challenge-dataset> |

### Datasets externos añadidos (citados en `Caso.md` §1.4)

| Dataset | Uso | Origen |
|---------|-----|--------|
| **Le2i Fall Detection (raw)** | **clase `persona_desequilibrio`** (pre-caída), tomada de su carpeta `Likefall`; y refuerzo de `persona_erguida` | <https://github.com/YifeiYang210/Fall_Detection_dataset> |

> `Le2i` es el que aporta la clase que el enunciado exige y que ningún otro
> dataset tenía: personas con el **centro de gravedad inestable antes de caer**.
> Se descargó de los enlaces de Google Drive indicados en el README de ese
> repositorio (`Le2i-raw`, 310 MB).

### Datasets además citados en `Caso.md` §1.4

Constan como referencias del documento y quedaron **descartados** por no ser
accesibles sin registro o por no aportar cajas:

- **Le2i Fall Detection Dataset** (versión procesada, segmentación de cuerpo
  humano): el archivo descargado es un RAR y no hay `unrar` en el entorno.
- **SisFall**, **FDD** (NTU) y **URFD**: requieren solicitud o registro previo.

### Modelos preentrenados

| Modelo | Uso | Origen |
|--------|-----|--------|
| `yolov8s.pt` / `yolov8n.pt` (COCO) | pseudo-etiquetado inicial y **detección de obstáculos** (RQF02 1) | Ultralytics, AGPL-3.0 |
| MediaPipe Pose (`pose_landmark_lite`) | postura, sujeción del pasamanos, distracción, tambaleo | Google MediaPipe, Apache-2.0 |

### Créditos y atribución

- **Ultralytics YOLO** — licence AGPL-3.0 (<https://www.ultralytics.com/license>).
  Los pesos de `yolov8n/s` se distribuyen bajo los términos de Ultralytics y
  de sus fuentes (COCO).
- **MediaPipe** — Apache-2.0 (<https://github.com/google-ai-edge/mediapipe>).
- **Los datasets pertenecen a sus autores** y se usan aquí solo con fines de
  investigación y evaluación académica. Se cita cada origen porque es lo que
  pide el enunciado; esta redistribución **no** intenta sublicenciarlos. Para
  cualquier uso que no sea de investigación hay que revisar la licencia de
  cada dataset en su ficha original (Kaggle / GitHub).
- Proyecto en el contexto de la **Universidad Peruana de Aplicaciones y
  Ciencias Skylight (UPAO)**.

---

## Arquitectura

```
  frame (cámara / imagen / vídeo)
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │  YOLOv8 propio  → persona / escalera          │  RQF02
  │  YOLOv8 COCO    → obstáculos sobre el escalón │  RQF02 (1)
  │  MediaPipe Pose → muñeca, cabeza, tronco      │  RQF02 (2)(3)
  └───────────────────────────────────────────────┘
        │
        ▼
  seguimiento con IoU + tolerancia a oclusión 1 s    RQF05
        │
        ▼
  doble umbral 75 % pre-filtrado / 85 % confirma   RQF04
        │
        ▼
  persistencia ≥ 3 s continuos                     RQF06
        │
        ▼
  alerta con metadatos + evidencia anonimizada     RQF07 · RQNF14
        │
        ▼
  panel web + registro auditable                   RQF08 · RQNF16
```

Todo el análisis se ejecuta en **un único hilo trabajador persistente**: Flask
atiende cada petición en un hilo distinto y en un hilo recién creado hay que
reenlazar el contexto CUDA, lo que costaba 85 ms por frame frente a 37 ms.

---

## Cumplimiento de requisitos

| Requisito | Descripción | Estado |
|-----------|-------------|--------|
| RQF01 | Captura continua de vídeo | **CUMPLE** — modo cámara |
| RQF02 | 4 categorías de riesgo | **CUMPLE** — 3 por pose + 1 por detector COCO |
| RQF03 | Discriminación seguro / riesgo | **CUMPLE** — veredicto por escena |
| RQF04 | Umbrales 75 % / 85 % | **CUMPLE** — doble umbral configurable |
| RQF05 | Seguimiento con oclusión 1 s | **CUMPLE** — identidad estable por IoU |
| RQF06 | Persistencia > 3 s | **CUMPLE** — medido: alerta a los 3.0 s |
| RQF07 | Alertas con evidencia | **CUMPLE** — metadatos + fotograma |
| RQF08 | Panel de supervisión | **CUMPLE** — app web |
| RQNF01 | ≥ 30 FPS | **NO VERIFICABLE en esta máquina** — 13.5 FPS medidos; ver nota |
| RQNF02 | Latencia < 500 ms | **CUMPLE** — 24 ms por ciclo |
| RQNF03 | VRAM ≥ 6 GB | **CUMPLE** — RTX 5050, 8.1 GB |
| RQNF10 | mAP@0.5 > 75 % | **PARCIAL** — val 0.810 ✓, test 0.747 |
| RQNF12 | Resiliencia a iluminación | **CUMPLE** — augmentations HSV reforzadas |
| RQNF14 | Protección de datos personales | **CUMPLE** — rostros pixelados al guardar |
| RQNF16 | Registro auditable | **CUMPLE** — `runs/alerts.jsonl` |
| RQNF19 | Autenticación de usuario | **CUMPLE** — `PANEL_TOKEN` |
| RQNF21 | Configuración sin recompilar | **CUMPLE** — `configs/train.yaml`, flags CLI |
| RQNF22 | Verificación con vídeos | **CUMPLE** — modo vídeo |

### Lo que NO está cubierto

- **RQNF04** (flujo continuo de cámara fija) y **RQNF19** en su parte de
  permisos por rol: la autenticación es un secreto compartido, no un modelo de
  usuarios.
- **RQF02 (1)(2)(3) sin validar con datos propios**: las heurísticas de pasamanos
  y distracción funcionan y están implementadas, pero **no se han medido**
  porque no hay ground truth para ellas en ningún dataset disponible.
- **RQNF11** (disponibilidad en horario de campus) y **RQNF27** (restricción de
  actuación física) dependen del despliegue, no del código.

---

## Documentación de decisiones

- `DECISIONES.md` — parte 1: auditoría de datos, unificación de los tres
  datasets, pseudo-etiquetado de Le2i, primera vuelta de entrenamiento.
- `DECISIONES_2.md` — parte 2: respuesta al fallo reportado con webcam,
  correcciones de fuga, y dos "mejoras" que resultaron peores y se revirtieron.
- `DECISIONES_3.md` — parte 3: auditoría de requisitos, motor de riesgo,
  seguridad, y datasets añadidos.

---

## Estructura

```
├── Caso.md                  documento de requisitos del proyecto
├── DECISIONES*.md           registro de decisiones
├── configs/train.yaml       parámetros de entrenamiento
├── requirements.txt         entorno reproducible
├── scripts/
│   ├── audit_datasets.py    auditoría de los datasets de origen
│   ├── prepare_le2i.py      pseudo-etiquetado con puerta de validación
│   ├── prepare_dataset.py   unificación a YOLO sin fuga
│   ├── add_zoom_variants.py primeros planos por recorte exacto
│   ├── train.py             entrenamiento + evaluación
│   ├── evaluate.py          evaluación de un checkpoint
│   └── eval_scale.py        evaluación por escala y recorte
└── app/
    ├── server.py            servidor Flask + hilo trabajador
    ├── risk_engine.py       motor de riesgo (RQF02-RQF07)
    └── templates/           interfaz
```

## Licencia

El código de este repositorio se distribuye bajo AGPL-3.0, alineado con
Ultralytics. Los datasets y los modelos preentrenados **no** se incluyen y
conservan sus propias licencias; ver [Créditos](#datasets-utilizados-y-créditos).