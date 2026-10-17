#!/usr/bin/env bash
# =============================================================================
#  Instalación del entorno - Sistema de prevención de caídas en escaleras
# =============================================================================
#  Uso:
#    scripts/setup.sh                 venv estándar, GPU si hay NVIDIA
#    scripts/setup.sh --lambda        venv que reutiliza Lambda Stack
#                                     (python3 -m venv --system-site-packages)
#    scripts/setup.sh --cpu           fuerza PyTorch solo CPU
#    scripts/setup.sh --venv NOMBRE   nombre de la carpeta del venv (.venv)
#
#  Todo queda dentro del venv: no se instala nada en el sistema.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

VENV=".venv"
LAMBDA=0
FORCE_CPU=0
while [ $# -gt 0 ]; do
    case "$1" in
        --lambda) LAMBDA=1 ;;
        --cpu)    FORCE_CPU=1 ;;
        --venv)   VENV="$2"; shift ;;
        -h|--help) sed -n 2,13p "$0"; exit 0 ;;
        *) echo "Opción desconocida: $1"; exit 2 ;;
    esac
    shift
done

# Los paquetes de ~/.local no deben colarse en el venv (Lambda Stack y pip
# --user suelen dejar versiones incompatibles de numpy allí).
export PYTHONNOUSERSITE=1

echo "== Python"
command -v python3 >/dev/null || { echo "Falta python3"; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Se requiere Python >= 3.10"'
python3 --version

echo "== GPU"
HAS_GPU=0
if [ "$FORCE_CPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 \
        && nvidia-smi -L >/dev/null 2>&1; then
    HAS_GPU=1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Sin GPU NVIDIA (o --cpu): se instala PyTorch CPU."
fi

echo "== Entorno virtual: $VENV"
if [ ! -d "$VENV" ]; then
    if [ "$LAMBDA" -eq 1 ]; then
        python3 -m venv --system-site-packages "$VENV"
    else
        python3 -m venv "$VENV"
    fi
fi
PY="$ROOT/$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

if [ "$LAMBDA" -eq 1 ]; then
    # Lambda Stack trae un flatbuffers de Debian con versión no estándar
    # ("1.12.1-git..."), que pip >= 24.1 rechaza. Uno válido dentro del venv
    # tiene prioridad sobre el del sistema.
    "$PY" -m pip install -q --ignore-installed "flatbuffers>=2.0"
fi

echo "== PyTorch"
if [ "$HAS_GPU" -eq 1 ]; then
    "$PY" -m pip install -r requirements/torch-cu128.txt
else
    "$PY" -m pip install -r requirements/torch-cpu.txt
fi

echo "== Dependencias"
"$PY" -m pip install -r requirements/base.txt

echo "== Módulos de audio e imagen"
"$PY" -m pip install -r requirements/modulos.txt

echo "== Modelos"
"$PY" tools/download_models.py

echo "== Verificación"
"$PY" tools/check_env.py || true

cat <<MSG

Listo. Para arrancar el panel:
    scripts/run.sh                      (o: source $VENV/bin/activate && python -m backend)
y abre http://127.0.0.1:5000
MSG
