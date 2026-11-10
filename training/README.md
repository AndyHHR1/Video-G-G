# Entrenamiento

## Modelo v5 (actual, rama `modelo-v5`)

El motor usa el modelo propio para dos tareas distintas, así que v5 son **dos
redes pequeñas** entrenadas solo con `stairs_fallhuman`:

| Modelo | Tarea | Base | Test | Latencia GPU / CPU | Tamaño |
|---|---|---|---|---|---|
| `models/detector/escalera_v5.pt` | detección de `escalera` (1 clase) a 512 px | YOLO11n | mAP50 0.802 · P 0.79 · R 0.77 | 11 / 26 ms | 5.5 MB |
| `models/detector/postura_v5.pt` | clasificación del recorte de cada persona: `persona_caido` · `persona_sentado` · `persona_erguida` (224 px) | YOLO11n-cls | recall 95 % · 89 % · 99 % (media 94.3 %) | 7 / 10 ms | 3.2 MB |

Por qué así:

* **Postura = clasificador de recortes.** COCO ya localiza a cada persona; el
  v4 pasaba cada recorte por un detector completo a 640 px. Un clasificador es
  la herramienta correcta y cuesta una fracción.
* **Los peatones de `tracking/` (MOT15) pasan a ser `persona_erguida`.** En v4
  eran una clase `persona` aparte que competía con `persona_erguida`; aquí
  aportan lo que faltaba (cámara alta, personas pequeñas o de espaldas).
* **Las variantes `s` se probaron y se descartaron:** postura-s fue peor (media
  89.4 %, sobreajusta con tan pocos datos) y escalera-s solo +2.5 puntos de
  mAP50 al doble de coste en CPU.
* `persona_desequilibrio` (pre-caída) no tiene datos en `stairs_fallhuman`; la
  detecta el esqueleto (`backend/perception/skeleton/`), no el modelo.

Advertencia honesta: el test de postura es pequeño (41 caído, 18 sentado, 15
de pie de fotos + 150 peatones) y las fotos de `fall/` son de stock; falta
validar con vídeo real de la escalera (vista en diagonal).

```bash
scripts/get_dataset.sh                    # o: hf download andres31416/stairs_fallhuman --repo-type dataset --local-dir stairs_fallhuman
python training/prepare_v5.py             # -> datasets/v5_postura, datasets/v5_escalera
bash training/run_v5_all.sh               # entrena n y s de ambos en la GPU -> runs/v5/*/v5_metrics.json
python tools/bench_engine.py              # rendimiento del motor completo (añade --set device=cpu)
```

## Histórico (v3 / v4)

Scripts y configuraciones con los que se construyeron los datasets y se
entrenaron los modelos. **No hace falta ejecutarlos para usar el sistema**, y en
la rama `expo` el Modelo S7 v4 es definitivo: se conservan para trazabilidad
(ISO/IEC 42001) y reproducibilidad.

| Paso | Script | Salida |
|---|---|---|
| 1. Auditoría de datasets | `audit_datasets.py` | informe por consola |
| 2. Le2i → clase pre-caída | `prepare_le2i.py` | `datasets/le2i_yolo/` |
| 3. Unificación a YOLO | `prepare_dataset.py --clean` | `datasets/combined/` |
| 4. Variantes de primer plano | `add_zoom_variants.py --split train` | `images/train_zoom` |
| 5. Entrenamiento base v3 | `train.py` (`configs/train.yaml`) | `models/base/yolov8s_zoom_v3.pt` |
| 6. Fotogramas de escaleras UPAO | `extract_escaleras_frames.py` (lee `data/Escaleras/`) | `datasets/escaleras_dataset/` |
| 7. Fine-tuning v4 | `train_escaleras.py --config training/configs/train_escaleras_v4.yaml` | Modelo S7 v4 |
| Evaluación | `evaluate.py`, `eval_scale.py` | métricas |

Otros: `extract_frames.py` / `label_video_frames.py` (clips propios),
`build_kaggle_variant.py` (variante descartada).

Los pasos 1-4 necesitan el dataset de Hugging Face, que se clona como
repositorio aparte (ver `data/README.md`):

```bash
scripts/get_dataset.sh          # -> ./stairs_fallhuman
```

Se ejecutan desde la raíz del repositorio, con el venv activo:

```bash
python training/evaluate.py --weights models/detector/escaleras_s7_v4.pt
```
