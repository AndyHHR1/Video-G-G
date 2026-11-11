#!/usr/bin/env python3
"""Descarga y verifica (SHA-256) todos los pesos que necesita el sistema.

    python tools/download_models.py            # descarga lo que falte
    python tools/download_models.py --check    # solo verifica, no descarga

Modelos:
  * models/detector/escalera_v5.pt      Modelo v5 · escalera (versionado en git)
  * models/detector/postura_v5.pt       Modelo v5 · postura  (versionado en git)
  * models/coco/yolov8s.pt              YOLOv8s COCO (Ultralytics)
  * models/sr/realesr-general-x4v3.pth  Real-ESRGAN compacto (vídeo en vivo)
  * models/pose/yolov8{s,n}-pose.pt     YOLOv8-Pose (esqueletos y equilibrio)
  * models/imagen/yolo11n.pt            YOLO11n (módulo Imagen)
  * models/imagen/yolo11n-pose.pt       YOLO11n-Pose (módulo Imagen)
  * modulos/audio/.../fall_classifier.joblib  MLP acústico (módulo Audio)

Los dos del Modelo v5 se entrenan con training/train_v5.py y vienen
versionados en el repositorio; el resto son preentrenados públicos.

Solo usa la librería estándar, así que funciona antes de instalar nada.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    # Modelo v5: versionado en el repositorio (no tiene URL de descarga).
    {
        "path": "models/detector/escalera_v5.pt",
        "sha256": "20349281fa4cbde7e88780266775a42934d7054cc331e1ea5b7e1e6f1cd2fb85",
        "url": None,
        "desc": "Modelo v5 · escalera (YOLO11n, 1 clase)",
    },
    {
        "path": "models/detector/postura_v5.pt",
        "sha256": "783d207b1be291078a53ba98135b5e71baab21ab1595bae185ece30a81c491ee",
        "url": None,
        "desc": "Modelo v5 · postura (YOLO11n-cls, 3 clases)",
    },
    {
        "path": "models/coco/yolov8s.pt",
        "sha256": "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a",
        "url": "https://github.com/ultralytics/assets/releases/download/"
               "v8.3.0/yolov8s.pt",
        "desc": "YOLOv8s COCO (personas y objetos)",
    },
    {
        "path": "models/sr/realesr-general-x4v3.pth",
        "sha256": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.5.0/realesr-general-x4v3.pth",
        "desc": "Real-ESRGAN general x4 v3 (vídeo en vivo)",
    },
    {
        "path": "models/pose/yolov8s-pose.pt",
        "sha256": "234314cd8baf62616791aceb9ea6ad5c19f26cf6c0d8f3a1bfce1e23b186cfb3",
        "url": "https://github.com/ultralytics/assets/releases/download/"
               "v8.3.0/yolov8s-pose.pt",
        "desc": "YOLOv8s-Pose (esqueletos: 17 puntos por persona)",
    },
    {
        "path": "models/pose/yolov8n-pose.pt",
        "sha256": "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212",
        "url": "https://github.com/ultralytics/assets/releases/download/"
               "v8.3.0/yolov8n-pose.pt",
        "desc": "YOLOv8n-Pose (alternativa más rápida, skeleton.model)",
    },
    {
        "path": "models/imagen/yolo11n.pt",
        "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        "url": "https://github.com/ultralytics/assets/releases/download/"
               "v8.3.0/yolo11n.pt",
        "desc": "YOLO11n COCO (módulo Imagen: personas y celulares)",
    },
    {
        "path": "models/imagen/yolo11n-pose.pt",
        "sha256": "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
        "url": "https://github.com/ultralytics/assets/releases/download/"
               "v8.3.0/yolo11n-pose.pt",
        "desc": "YOLO11n-Pose (módulo Imagen: articulaciones)",
    },
    {
        "path": "modulos/audio/audio_fall_detector/models/fall_classifier.joblib",
        "sha256": "68527b88f75142693c56ee798f981611febb5d25fd46316e6c799aca7e2eb9da",
        "url": "https://github.com/brigitteve/percepcion/raw/main/"
               "audio_fall_detector/models/fall_classifier.joblib",
        "desc": "Clasificador acústico de caídas (módulo Audio, versionado)",
    },
]


def sha256(data_or_path) -> str:
    h = hashlib.sha256()
    if isinstance(data_or_path, (bytes, bytearray)):
        h.update(data_or_path)
    else:
        with open(data_or_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    print(f"    descargando {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def ensure(model: dict, check_only: bool) -> bool:
    dest = ROOT / model["path"]
    if dest.is_file():
        ok = sha256(dest) == model["sha256"]
        print(f"  [{'OK' if ok else 'HASH DISTINTO'}] {model['path']}  ({model['desc']})")
        return ok
    if check_only:
        print(f"  [FALTA] {model['path']}  ({model['desc']})")
        return False
    if not model.get("url"):
        print(f"  [FALTA] {model['path']}  ({model['desc']}): viene versionado "
              "en el repositorio; recupéralo con git (git checkout -- models/).")
        return False
    print(f"  [..] {model['path']}  ({model['desc']})")
    data = fetch(model["url"])
    if "member" in model:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            data = tar.extractfile(model["member"]).read()
    if sha256(data) != model["sha256"]:
        print("    ERROR: la suma SHA-256 no coincide; no se guarda el archivo.")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"    guardado ({len(data) / 1e6:.1f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="solo verificar")
    args = ap.parse_args()
    print("Modelos del sistema:")
    results = [ensure(m, args.check) for m in MODELS]
    if all(results):
        print("Todo correcto.")
        return 0
    print("Hay modelos que faltan o no coinciden.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
