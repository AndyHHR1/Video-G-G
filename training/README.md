# Entrenamiento (histórico)

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

Se ejecutan desde la raíz del repositorio, con el venv activo:

```bash
python training/evaluate.py --weights models/detector/escaleras_s7_v4.pt
```
