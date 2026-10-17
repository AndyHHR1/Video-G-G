#!/usr/bin/env python3
"""Descarga y verifica (SHA-256) todos los pesos que necesita el sistema.

    python tools/download_models.py            # descarga lo que falte
    python tools/download_models.py --check    # solo verifica, no descarga

Modelos:
  * models/detector/escaleras_s7_v4.pt  "Modelo S7 v4" (release del proyecto)
  * models/coco/yolov8s.pt              YOLOv8s COCO (Ultralytics)
  * models/sr/realesr-general-x4v3.pth  Real-ESRGAN compacto (vivo / vídeo)
  * models/sr/RealESRGAN_x4plus.pth     Real-ESRGAN RRDB (imagen fija)

Solo usa la librería estándar, así que funciona antes de instalar nada.
El detector viene versionado en el repositorio; si falta, se extrae del
release v1.2.0-escaleras-v4 de GitHub.
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
RELEASE = ("https://github.com/AndyHHR1/Video-G-G/releases/download/"
           "v1.2.0-escaleras-v4/escaleras_release_v1.2.0.tar.gz")

MODELS = [
    {
        "path": "models/detector/escaleras_s7_v4.pt",
        "sha256": "b6a04f2f28d1f7bcc6282053db891c387ac0b121f9d2fcca4534c691f606d7ed",
        "url": RELEASE, "member": "best.pt",
        "desc": "Modelo S7 v4 (YOLOv8s, 6 clases)",
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
        "path": "models/sr/RealESRGAN_x4plus.pth",
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth",
        "desc": "Real-ESRGAN x4plus (imagen fija)",
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
