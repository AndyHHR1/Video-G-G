#!/usr/bin/env bash
# Entrena y evalúa las variantes del Modelo v5 en la GPU, una tras otra.
#   training/run_v5_all.sh                 -> todas
#   training/run_v5_all.sh postura         -> solo las de postura
#   resultados: runs/v5/<nombre>/v5_metrics.json
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
# WSL2 limita la memoria "fijada" (pinned) del anfitrión: el hilo pin-memory del
# DataLoader de clasificación daba "CUDA error: out of memory" con la GPU casi
# vacía. Sin pin memory el coste es despreciable para estos tamaños.
export PIN_MEMORY=False
PY=.venv/bin/python
WHAT="${1:-all}"
run() { echo "=== $(date +%T) $*"; $PY training/train_v5.py "$@" 2>&1 | grep -E "V5_METRICS|Error|error|Traceback|EarlyStopping|epochs completed" ; }
if [ "$WHAT" = all ] || [ "$WHAT" = postura ]; then
  run --task postura  --model yolo11n-cls.pt --name postura_n
  run --task postura  --model yolo11s-cls.pt --name postura_s
fi
if [ "$WHAT" = all ] || [ "$WHAT" = escalera ]; then
  run --task escalera --model yolo11n.pt     --name escalera_n
  run --task escalera --model yolo11s.pt     --name escalera_s
fi
echo "=== $(date +%T) FIN"
