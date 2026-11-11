"""Aceleración en procesador con OpenVINO (equipos sin GPU NVIDIA).

En CPU, OpenVINO ejecuta los mismos modelos YOLO bastante más rápido que
PyTorch (medido en este proyecto, mediana por inferencia):

| modelo                 | PyTorch | OpenVINO |
|------------------------|---------|----------|
| escalera_v5 (512 px)   | 173 ms  |  49 ms   |
| postura_v5 (224 px)    |  56 ms  |  17 ms   |
| COCO yolov8s (512 px)  | 343 ms  | 192 ms   |
| YOLOv8n-Pose (480 px)  | 313 ms  |  79 ms   |

La primera vez que el sistema arranca en CPU cada modelo se exporta a
OpenVINO (unos segundos) y se guarda junto al original como
`<nombre>_<imgsz>_openvino_model/` (no se versiona). Las siguientes veces se
carga directamente. Con GPU, sin el paquete `openvino`, o con
`hardware.openvino_cpu: false`, se usa el modelo PyTorch tal cual.

La exportación es de tamaño fijo: quien use un modelo acelerado debe predecir
SIEMPRE con el mismo `imgsz` con el que se exportó.

INT8 (`int8=True`): solo para el COCO que localiza personas y objetos, que es
el modelo más pesado en CPU. Medido (misma carga, OpenVINO FP32 vs INT8):
168 -> 87 ms; en coco128 la clase persona queda igual (mAP50 0.800 -> 0.803;
precisión 0.884 -> 0.912, recall 0.687 -> 0.661). Se calibra con `coco8`
(se descarga sola, ~1 MB). Postura (no gana), esqueleto (sin forma fiable de
medir su precisión) y escalera (gana poco y se calibraría con el dataset) se
quedan en FP32. `hardware.openvino_int8_coco: false` lo desactiva.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_ENABLED = True
INT8_COCO = True


def configure(enabled: bool, int8_coco: bool = True) -> None:
    global _ENABLED, INT8_COCO
    _ENABLED = bool(enabled)
    INT8_COCO = bool(int8_coco)


def available() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except Exception:                                       # noqa: BLE001
        return False


def cpu_weights(weights: str, device, imgsz: int, dynamic: bool = False,
                int8: bool = False) -> str:
    """Ruta del modelo OpenVINO equivalente (exportándolo si hace falta) si el
    dispositivo es CPU y OpenVINO está disponible; si no, `weights` tal cual.

    `dynamic=True` exporta con lote variable: necesario si se predicen VARIAS
    imágenes a la vez (el clasificador de postura recibe todas las personas
    en un lote); con tamaño fijo OpenVINO solo acepta lotes de 1."""
    if str(device) != "cpu" or not _ENABLED or os.environ.get("VGG_NO_OPENVINO"):
        return weights
    src = Path(weights)
    if src.suffix != ".pt" or not src.is_file() or not available():
        return weights
    tag = ("dyn" if dynamic else str(imgsz)) + ("_int8" if int8 else "")
    out = src.with_name(f"{src.stem}_{tag}_openvino_model")
    if (out / "metadata.yaml").is_file():
        return str(out)
    if int8:
        try:
            import nncf  # noqa: F401  (cuantización INT8)
        except Exception:                                   # noqa: BLE001
            return cpu_weights(weights, device, imgsz, dynamic=dynamic)
    try:
        from ultralytics import YOLO
        print(f"  OpenVINO: exportando {src.name} a {imgsz} px"
              f"{' INT8' if int8 else ''} (solo la primera vez)...", flush=True)
        extra = {"int8": True, "data": "coco8.yaml"} if int8 else {}
        exported = Path(YOLO(str(src)).export(format="openvino", imgsz=imgsz,
                                              half=False, dynamic=dynamic,
                                              verbose=False, **extra))
        if exported.resolve() != out.resolve():
            if out.exists():
                shutil.rmtree(out)
            exported.rename(out)
        return str(out)
    except Exception as exc:                                # noqa: BLE001
        if int8:     # p. ej. sin internet para descargar coco8: FP32
            print(f"  [aviso] INT8 no disponible para {src.name} ({exc}); se usa FP32",
                  flush=True)
            return cpu_weights(weights, device, imgsz, dynamic=dynamic)
        print(f"  [aviso] OpenVINO no disponible para {src.name}: {exc}", flush=True)
        return weights
