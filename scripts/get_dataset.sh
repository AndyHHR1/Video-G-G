#!/usr/bin/env bash
# Clona el dataset de entrenamiento desde Hugging Face como repositorio APARTE.
#
#   scripts/get_dataset.sh            -> ./stairs_fallhuman  (ignorado por git)
#   scripts/get_dataset.sh RUTA       -> otra carpeta
#
# Solo hace falta para reproducir el entrenamiento (training/). Para usar el
# sistema no se necesita: el Modelo S7 v4 ya está en models/detector/.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-stairs_fallhuman}"
REPO="https://huggingface.co/datasets/andres31416/stairs_fallhuman"

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
    echo "Falta git-lfs (las imágenes del dataset se guardan con LFS)."
    echo "  Ubuntu/WSL: sudo apt install git-lfs   ·   Windows/macOS: https://git-lfs.com"
    exit 1
fi
git lfs install --skip-repo >/dev/null

if [ -d "$DEST/.git" ]; then
    echo "Ya existe $DEST: actualizando..."
    git -C "$DEST" pull --ff-only
else
    git clone "$REPO" "$DEST"
fi
echo "Dataset listo en $DEST (fall/, stairs/, tracking/)."
