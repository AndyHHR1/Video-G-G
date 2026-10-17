# Datos (no versionados en este repositorio)

Los datos se mantienen **separados del código**: ni el dataset ni los vídeos se
suben a este repositorio (pesan mucho y contienen personas reales, Ley N.º 29733).
Solo hacen falta para **reproducir el entrenamiento**; para usar el sistema no
se necesitan, porque el Modelo S7 v4 ya está en `models/detector/`.

## 1. Dataset de entrenamiento (Hugging Face)

Repositorio aparte:
[`andres31416/stairs_fallhuman`](https://huggingface.co/datasets/andres31416/stairs_fallhuman)

```bash
scripts/get_dataset.sh          # clona en ./stairs_fallhuman (requiere git-lfs)
# equivalente manual:
git lfs install
git clone https://huggingface.co/datasets/andres31416/stairs_fallhuman
```

| Carpeta | Origen | Clases que aporta |
|---|---|---|
| `fall/` | Fall Detection Dataset (Kaggle) | `persona_caido`, `persona_sentado`, `persona_erguida` |
| `stairs/final_stairs/` | Stairs (Kaggle) | `escalera` |
| `tracking/` | MOT15 (Kaggle) | `persona` |

`stairs_fallhuman/` queda en la raíz del proyecto (es la ruta por defecto de
`training/audit_datasets.py` y `training/prepare_dataset.py`) y está en
`.gitignore`: es **otro repositorio git**, se actualiza con `git pull` dentro de
esa carpeta sin mezclarse con este.

## 2. Vídeos de las escaleras UPAO (fine-tuning v4)

No están en Hugging Face; se comparten aparte. Se colocan así:

```
data/
└── Escaleras/
    ├── Estándar/   vídeos desde la posición REAL de la cámara de seguridad
    │               (se reparten por vídeo en train / val / test)
    └── Extra/      vídeos desde otras posiciones: solo entrenamiento
```

Los usa `training/extract_escaleras_frames.py`. Los vídeos de `Estándar/`
también sirven para probar el sistema en la pestaña **Vídeo** del panel (RQNF22).
