#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
echo "============================================================"
echo "  Setup - Sistema de Prevención de Caídas en Escaleras (V1)"
echo "  Rama: supuesto-ideal / tag: supuesto-ideal-v1.0"
echo "  Directorio: $ROOT"
echo "============================================================"

# --- Verificar GPU NVIDIA con CUDA 12.8 ---
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] No se detectó NVIDIA. Se requiere GPU con >= 6 GB VRAM."
    echo "        El modelo YOLOv8 y MediaPipe necesitan aceleración por hardware."
    echo "        En modo CPU el rendimiento no cumple RQNF01 (>30 FPS)."
    exit 1
fi

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 6000 ]; then
    echo "[ADVERTENCIA] VRAM detectada: ${VRAM} MB. El requisito mínimo es 6 GB (RQNF02)."
fi
echo "✓ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (${VRAM} MB VRAM)"

# --- Verificar Python 3.9+ ---
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python3 no encontrado."
    exit 1
fi
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python: $PYVER"

# --- Crear entorno virtual ---
if [ ! -d venv ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi
echo "✓ venv listo"

PIP="$ROOT/venv/bin/pip"

# --- Instalar PyTorch con CUDA 12.8 ---
echo ""
echo "Instalando PyTorch + CUDA 12.8 (compatible RTX 5050 / sm_120)..."
$PIP install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" 2>&1 | tail -1
echo "✓ PyTorch instalado"

# --- Instalar dependencias (en orden para evitar conflicto protobuf/mediapipe) ---
echo ""
echo "Instalando dependencias..."
echo "  (nota: mediapipe 0.10.21 requiere protobuf<5; se instala por separado)"

# 1. Paquetes sin protobuf ni mediapipe
$PIP install \
    "ultralytics==8.4.170" \
    "opencv-python==4.10.0.84" \
    "pillow==12.3.0" \
    "numpy==1.26.4" \
    "matplotlib==3.11.2" \
    "PyYAML==6.0.3" \
    "Flask==3.1.3" \
    "Werkzeug==3.1.5" \
    "imageio-ffmpeg==0.6.0" \
    "onnxruntime-gpu==1.30.0" \
    "psutil==7.2.2" \
    "polars==1.44.2" \
    "polars-runtime-32==1.44.2" \
    "scipy==1.16.3" \
    "sympy==1.14.0" \
    "networkx==3.6.1" \
    "filelock==3.32.3" \
    "fsspec==2026.7.0" \
    "setuptools==78.1.0" \
    "six==1.17.0" \
    "Jinja2==3.1.6" \
    "MarkupSafe==3.0.3" \
    "mpmath==1.3.0" \
    "requests==2.34.2" \
    2>&1 | tail -1
echo "  ✓ Paquetes principales instalados"

# 2. MediaPipe + protobuf compatible (<5)
$PIP install "mediapipe==0.10.21" "protobuf>=4.25.3,<5" 2>&1 | tail -1
echo "  ✓ MediaPipe + protobuf corregido"

# 3. CUDA utilities
$PIP install "cuda-bindings==12.9.7" "cuda-pathfinder==1.6.0" "triton==3.6.0" 2>&1 | tail -1
echo "  ✓ CUDA utilities"

# --- Verificar modelos ---
echo ""
echo "Verificando modelos..."
MISSING=""

if [ ! -f "$ROOT/yolov8s.pt" ]; then
    echo "  Descargando yolov8s.pt (COCO para obstáculos, 22MB)..."
    $PIP install gdown 2>&1 | tail -1
    # YOLOv8s COCO weights (22MB) - download from Ultralytics
    $PIP run python -c "
from ultralytics import YOLO; YOLO('yolov8s.pt')
" 2>&1 | tail -2 || echo "  (Se descargará automáticamente al primer uso)"
else
    echo "  ✓ yolov8s.pt presente (22MB)"
fi

if [ -f "$ROOT/runs/yolov8s_zoom/weights/best.pt" ]; then
    SIZE=$(du -h "$ROOT/runs/yolov8s_zoom/weights/best.pt" | cut -f1)
    echo "  ✓ runs/yolov8s_zoom/weights/best.pt presente ($SIZE)"
else
    echo "  ⨯ best.pt no encontrado - entrenar con:"
    echo "    ./venv/bin/python scripts/train.py --set name=yolov8s_zoom"
    MISSING="model"
fi

# --- Datasets (info para el usuario) ---
echo ""
echo "Datasets:"
echo "  stairs_fallhuman:  $( [ -d 'stairs_fallhuman' ] && echo '✓ presente' || echo '✗ clonar con: git clone https://huggingface.co/datasets/andres31416/stairs_fallhuman' )"
echo "  Le2i-raw:          $( [ -d 'datasets/_external/Le2i-raw' ] && echo '✓ presente' || echo '✗ descargar con: ./venv/bin/python scripts/prepare_le2i.py --help' )"
echo "  Combined dataset:  $( [ -d 'datasets/combined' ] && echo '✓ generado' || echo '✗ generar con: ./venv/bin/python scripts/prepare_dataset.py --clean' )"

# --- Verificación final ---
echo ""
echo "============================================================"
if [ -z "$MISSING" ]; then
    echo "  ✓ LISTO - Inicie el servidor:"
    echo ""
    echo "  ./venv/bin/python app/server.py --host 127.0.0.1 --port 5000"
    echo ""
    echo "  Abra: http://127.0.0.1:5000"
    echo "============================================================"
    exit 0
else
    echo "  ✗ FALTA: modelo entrenado."
    echo "           Ejecute scripts/train.py antes de iniciar."
    echo "============================================================"
    exit 1
fi
