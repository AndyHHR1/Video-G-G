#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/$(dirname "$0")" 2>/dev/null || cd "$(dirname "$0")"
cd "$(dirname "$0")"
ROOT="$(pwd)"
cd "$ROOT"

echo "============================================================"
echo "  Setup - Sistema de Prevencion de Caídas en Escaleras"
echo "  Rama: supuesto-ideal"
echo "============================================================"

# --- Verificar GPU NVIDIA con CUDA 12.8 (sm_120 = RTX 5050) ---
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] No se detectó NVIDIA. Se requiere GPU con >= 6 GB VRAM."
    exit 1
fi

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 6000 ]; then
    echo "[ADVERTENCIA] VRAM detectada: ${VRAM} MB. El requisito mínimo es 6 GB."
fi
echo "✓ GPU detectada: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (${VRAM} MB VRAM)"

# --- Crear entorno virtual ---
if [ ! -d venv ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

PIP="./venv/bin/pip"

# --- Instalar PyTorch con CUDA 12.8 (compatible con sm_120 / RTX 5050) ---
echo "Instalando PyTorch + CUDA 12.8..."
$PIP install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" 2>&1 | tail -2

# --- Instalar dependencias ---
echo "Instalando dependencias de requirements.txt..."
$PIP install -r requirements.txt 2>&1 | tail -3

# --- Verificar modelos ---
echo ""
echo "Verificando modelos..."
MISSING=""

if [ ! -f "yolov8s.pt" ]; then
    echo "  Descargando yolov8s.pt (COCO, detector de obstáculos)..."
    $PIP install gdown 2>&1 | tail -1
    MISSING=""
else
    echo "  ✓ yolov8s.pt presente"
fi

if [ ! -f "runs/yolov8s_zoom/weights/best.pt" ]; then
    echo "  [ADVERTENCIA] runs/yolov8s_zoom/weights/best.pt no encontrado."
    echo "  Descargue el modelo entrenado o entrene con:"
    echo "    ./venv/bin/python scripts/train.py --set name=yolov8s_zoom"
    MISSING="model"
else
    echo "  ✓ runs/yolov8s_zoom/weights/best.pt presente"
fi

if [ -z "$MISSING" ]; then
    echo "  ✓ Todos los modelos listos"
fi

# --- Datasets ---
echo ""
echo "Datasets:"
if [ -d "stairs_fallhuman" ]; then
    echo "  ✓ stairs_fallhuman/ (presente)"
else
    echo "  ⨯ stairs_fallhuman/ (ausente - clonar de HuggingFace)"
    echo "    git lfs install"
    echo "    git clone https://huggingface.co/datasets/andres31416/stairs_fallhuman"
fi

if [ -d "datasets/_external/Le2i-raw" ]; then
    echo "  ✓ Le2i-raw/ (presente)"
else
    echo "  ⨯ Le2i-raw/ (ausente - descargar desde Google Drive)"
    echo "    Ver scripts/prepare_le2i.py --help"
fi

echo ""
echo "============================================================"
if [ -z "$MISSING" ]; then
    echo "  LISTO. Inicie el servidor con:"
    echo ""
    echo "  ./venv/bin/python app/server.py --host 127.0.0.1 --port 5000"
    echo ""
    echo "  Abra: http://127.0.0.1:5000"
else
    echo "  FALTA: modelo entrenado. Entrene antes de iniciar."
fi
echo "============================================================"
