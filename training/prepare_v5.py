#!/usr/bin/env python3
"""
Modelo v5 - Preparación de datos a partir de `stairs_fallhuman`.

El motor usa el modelo propio para DOS tareas distintas (ver
backend/perception/engine.py), así que en v5 se entrenan dos modelos pequeños
en lugar de un detector de 6 clases:

1. **Postura** (clasificación de recortes): COCO ya localiza a cada persona;
   el modelo solo decide su postura sobre el recorte. Un clasificador es la
   herramienta correcta para eso y cuesta ~2 ms por persona, frente a pasar el
   recorte por un detector completo a 640 px.
       datasets/v5_postura/{train,val,test}/{persona_caido,persona_sentado,persona_erguida}/
   - `fall/`     -> recortes de las 3 posturas.
   - `tracking/` -> recortes de peatones (MOT15) como `persona_erguida`: van de
     pie o caminando, y aportan justo lo que faltaba (cámara alta, personas
     pequeñas, de espaldas). Antes eran una clase `persona` aparte que chocaba
     con `persona_erguida`.

2. **Escalera** (detección, 1 clase) + imágenes de fondo sin escalera para
   reducir falsos positivos.
       datasets/v5_escalera/{images,labels}/{train,val,test}/  + data.yaml

El reparto train/val/test es el de training/prepare_dataset.py (sin fuga:
duplicados por hash eliminados, variantes de Roboflow y secuencias de vídeo
agrupadas en un solo split). Los datos originales se abren solo en lectura.

Uso:
    python training/prepare_v5.py            # --src stairs_fallhuman --out datasets
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_dataset as pdz  # noqa: E402  (reparto sin fuga ya probado)

SEED = 42
POSTURE = {0: "persona_caido", 1: "persona_sentado", 2: "persona_erguida"}
TRACKING_CLASS = pdz.TRACKING_CLASS          # 4 -> peatón genérico
PAD = 0.08            # mismo margen que PostureDetector.classify (inferencia)
MIN_SIDE = 20         # recortes más pequeños no tienen información útil
TRACK_MIN_H = 72      # peatones MOT: solo los que se ven con algo de detalle
TRACK_MIN_ASPECT = 1.4  # alto/ancho: de pie (descarta cajas cortadas o raras)
# Tope de recortes de peatones por split, relativo a la clase de postura más
# numerosa de `fall` en ese split: aportan variedad sin aplastar al resto.
TRACK_CAP_FACTOR = 2.0
TRACK_PER_SEQ = 400   # reparto entre secuencias (ninguna domina)
BG_FRACTION = 0.12    # fondos sin escalera respecto a las imágenes con escalera
BALANCE_TARGET = 0.6  # train: cada postura llega al menos al 60 % de la mayor


def crop(img, box, pad=PAD):
    """Recorte YOLO normalizado (cx, cy, w, h) con margen, como en inferencia."""
    H, W = img.shape[:2]
    _, cx, cy, bw, bh = box
    x1, x2 = (cx - bw / 2) * W, (cx + bw / 2) * W
    y1, y2 = (cy - bh / 2) * H, (cy + bh / 2) * H
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, x2 = int(max(0, x1 - pw)), int(min(W, x2 + pw))
    y1, y2 = int(max(0, y1 - ph)), int(min(H, y2 + ph))
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return img[y1:y2, x1:x2]


def build_postura(samples, out: Path) -> dict:
    root = out / "v5_postura"
    if root.exists():
        shutil.rmtree(root)
    for sp in ("train", "val", "test"):
        for name in POSTURE.values():
            (root / sp / name).mkdir(parents=True, exist_ok=True)

    counts: dict = defaultdict(Counter)
    # 1) fall/: las tres posturas
    for s in samples:
        if s.origin != "fall":
            continue
        img = cv2.imread(str(s.src_img))
        if img is None:
            continue
        for i, b in enumerate(s.boxes):
            if b[0] not in POSTURE:
                continue
            c = crop(img, b)
            if c is None:
                continue
            name = POSTURE[b[0]]
            cv2.imwrite(str(root / s.split / name /
                            f"fall_{Path(s.out_name).stem}_{i}.jpg"), c,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[s.split][name] += 1

    # 2) tracking/: peatones como persona_erguida, con tope por split
    rng = random.Random(SEED)
    cands: dict = defaultdict(lambda: defaultdict(list))   # split -> seq -> [(s, i)]
    for s in samples:
        if s.origin != "tracking":
            continue
        for i, b in enumerate(s.boxes):
            cands[s.split][s.group].append((s, i))
    for sp, by_seq in cands.items():
        biggest = max(counts[sp].values(), default=0)
        cap = int(max(biggest * TRACK_CAP_FACTOR, 150))
        chosen = []
        for seq in sorted(by_seq):
            items = by_seq[seq]
            rng.shuffle(items)
            chosen += items[:TRACK_PER_SEQ]
        rng.shuffle(chosen)
        n = 0
        cache: dict = {}
        for s, i in chosen:
            if n >= cap:
                break
            key = str(s.src_img)
            img = cache.get(key)
            if img is None:
                img = cv2.imread(key)
                if img is None:
                    continue
                cache = {key: img}          # solo la última imagen en memoria
            H, W = img.shape[:2]
            _, cx, cy, bw, bh = s.boxes[i]
            hpx, wpx = bh * H, bw * W
            if hpx < TRACK_MIN_H or wpx <= 0 or hpx / wpx < TRACK_MIN_ASPECT:
                continue
            # descartar peatones cortados por el borde de la imagen
            if cx - bw / 2 < 0.005 or cx + bw / 2 > 0.995 or \
               cy - bh / 2 < 0.005 or cy + bh / 2 > 0.995:
                continue
            c = crop(img, s.boxes[i])
            if c is None:
                continue
            cv2.imwrite(str(root / sp / "persona_erguida" /
                            f"mot_{Path(s.out_name).stem}_{i}.jpg"), c,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[sp]["persona_erguida (mot)"] += 1
            n += 1

    # 3) Balanceo SOLO en train: las clases minoritarias se repiten (enlaces)
    #    hasta BALANCE_TARGET de la mayor. Con los aumentos aleatorios de cada
    #    época las copias no son idénticas; val y test quedan intactos.
    train = root / "train"
    sizes = {d.name: len(list(d.iterdir())) for d in train.iterdir()}
    top = max(sizes.values(), default=0)
    for name, k in sizes.items():
        files = sorted(p for p in (train / name).iterdir() if not p.is_symlink())
        need = int(top * BALANCE_TARGET) - k
        rep = 0
        while need > 0 and files:
            for p in files:
                if need <= 0:
                    break
                (train / name / f"{p.stem}_dup{rep}{p.suffix}").symlink_to(p.name)
                need -= 1
            rep += 1
        counts["train"][f"{name} (tras balanceo)"] = len(list((train / name).iterdir()))
    return {sp: dict(c) for sp, c in counts.items()}


def build_escalera(samples, out: Path) -> dict:
    root = out / "v5_escalera"
    if root.exists():
        shutil.rmtree(root)
    for sp in ("train", "val", "test"):
        (root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (root / "labels" / sp).mkdir(parents=True, exist_ok=True)

    counts: dict = defaultdict(Counter)

    def put(s, boxes, tag):
        dst = root / "images" / s.split / f"{tag}_{s.out_name}"
        dst.symlink_to(os.path.relpath(s.src_img, dst.parent))
        (root / "labels" / s.split / f"{dst.stem}.txt").write_text(
            "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                    for _, cx, cy, w, h in boxes))

    for s in samples:
        if s.origin == "stairs":
            boxes = [b for b in s.boxes if b[0] == 3]
            if boxes:
                put(s, boxes, "st")
                counts[s.split]["escalera"] += 1
                counts[s.split]["cajas"] += len(boxes)
    # fondos: fotos de interior (fall) y calle (tracking) sin escalera anotada
    rng = random.Random(SEED)
    for sp in ("train", "val", "test"):
        pool = [s for s in samples if s.split == sp and s.origin in ("fall", "tracking")]
        rng.shuffle(pool)
        k = int(counts[sp]["escalera"] * BG_FRACTION)
        for s in pool[:k]:
            put(s, [], "bg")
            counts[sp]["fondo"] += 1

    (root / "data.yaml").write_text(
        "# Modelo v5 - escalera (generado por training/prepare_v5.py)\n"
        f"path: {root.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: escalera\n", encoding="utf-8")
    return {sp: dict(c) for sp, c in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("stairs_fallhuman"))
    ap.add_argument("--out", type=Path, default=Path("datasets"))
    args = ap.parse_args()
    src, out = args.src.resolve(), args.out.resolve()
    if not src.is_dir():
        raise SystemExit(f"No existe el dataset: {src}")
    out.mkdir(parents=True, exist_ok=True)

    b = pdz.Builder(src, out / "_v5_tmp")
    print("Leyendo stairs_fallhuman (solo lectura)...")
    pdz.load_fall(b)
    pdz.load_stairs(b)
    pdz.load_tracking(b)
    print(f"  muestras: {len(b.samples)}")
    print(f"  duplicados resueltos: {b.dedupe_preferring_rare_classes()}")
    b.assign_splits()
    print(f"  fuga entre splits eliminada: {b.scrub_cross_split_duplicates()}")

    rep = {"postura": build_postura(b.samples, out),
           "escalera": build_escalera(b.samples, out),
           "seed": SEED, "pad": PAD, "track_min_h": TRACK_MIN_H,
           "track_cap_factor": TRACK_CAP_FACTOR, "bg_fraction": BG_FRACTION}
    (out / "v5_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
