#!/usr/bin/env python3
"""
Paso 3 - Entrenamiento y evaluacion de YOLOv8.

Lee `configs/train.yaml`, entrena el detector y evalua sobre `val` y `test`,
guardando pesos, metricas y graficas. Todo el entrenamiento es reproducible:
semilla fija y configuracion versionada.

Uso:
    python3 scripts/train.py
    python3 scripts/train.py --model yolov8s --epochs 150
    python3 scripts/train.py --config configs/train.yaml --set use_balanced=false
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Balanceo de clases
# --------------------------------------------------------------------------- #
def build_balanced_dataset(cfg: dict) -> Path | None:
    """Crea un subconjunto de entrenamiento con la clase mas frecuente submuestreada.

    El dataset unificado esta muy desbalanceado: las clases que dominan son
    `persona` (CCTV, muchas cajas por fotograma) y `escalera`, frente a las
    clases de postura `persona_caido` / `persona_sentado` / `persona_erguida`,
    que aportan menos de un tercio de las cajas de entrenamiento.

    Aqui se identifica la clase con mas IMAGENES propias (no la que mas cajas
    tiene: `persona` aparece en pocos fotogramas pero con muchas cajas cada
    uno) y se conserva solo una fraccion de ellas. Es la clase que mas
    diluye el gradiente sobre el resto. Sin tocar los originales: lo que se
    crean son symlinks nuevos bajo `images/train_balanced/`.

    Devuelve la ruta del data.yaml generado, o None si no aplica.
    """
    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    root = data_path.parent
    names = _read_names(data_path)

    train_img = root / "images" / "train"
    train_lbl = root / "labels" / "train"
    out_img = root / "images" / "train_balanced"
    out_lbl = root / "labels" / "train_balanced"
    for d in (out_img, out_lbl):
        d.mkdir(parents=True, exist_ok=True)

    # Clase dominante = la presente en mas imagenes propias (no la que mas
    # cajas aporta). `persona` tiene muchisimas cajas pero se concentra en
    # pocos fotogramas de CCTV con varios sujetos cada uno, asi que la
    # metrica adecuada para decidir a quien submuestrear es el conteo de
    # imagenes, no el de instancias.
    per_class = Counter()
    img_classes: dict[str, set] = defaultdict(set)
    for lbl in train_lbl.glob("*.txt"):
        cls = set()
        for line in lbl.read_text().splitlines():
            if line.strip():
                cls.add(int(float(line.split()[0])))
        img_classes[lbl.stem] = cls
        per_class.update(cls)

    if not per_class:
        print("  [aviso] no hay cajas en train; no se balancea")
        return None

    major = per_class.most_common(1)[0][0]
    frac = float(cfg["keep_majority_fraction"])
    rng = random.Random(int(cfg["seed"]))

    kept = dropped = 0
    by_stem = {p.stem: p for p in train_img.iterdir() if p.is_file()}
    for stem, cls in sorted(img_classes.items()):
        src_img = by_stem.get(stem)
        if src_img is None:
            continue
        # si la imagen contiene la clase mayoritaria Y solo ella, se submuestrea
        if cls == {major} and rng.random() > frac:
            dropped += 1
            continue
        _link(src_img, out_img / src_img.name)
        _link(train_lbl / f"{stem}.txt", out_lbl / f"{stem}.txt")
        kept += 1

    print(
        f"  balanceo: clase mayoritaria '{names.get(major, major)}' "
        f"submuestreada al {frac:.0%}  (train {kept} imgs, {dropped} descartadas)"
    )

    out_yaml = root / "data_balanced.yaml"
    out_yaml.write_text(
        f"# Generado por scripts/train.py (NO editar a mano).\n"
        f"path: {root.resolve()}\n"
        f"train: images/train_balanced\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in names.items()),
        encoding="utf-8",
    )
    return out_yaml


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def _read_names(data_path: Path) -> dict[int, str]:
    data = yaml.safe_load(data_path.read_text())
    raw = data.get("names", {})
    if isinstance(raw, list):
        return dict(enumerate(raw))
    return {int(k): v for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Evaluacion
# --------------------------------------------------------------------------- #
# Clases cuyas etiquetas de test proceden de anotacion HUMANA. La clase
# `persona` queda fuera a proposito: sus 11 secuencias de test solo tienen
# `det/det.txt`, que es salida de un detector, no ground truth. Como el modelo
# predice mejor que ese pseudo-ground-truth, incluirla baja artificialmente la
# metrica. Se reportan las dos cifras: con y sin ella.
GT_HUMAN_CLASSES = ("persona_caido", "persona_sentado", "persona_erguida",
                    "escalera", "persona_desequilibrio")


def evaluate(model, data: str, split: str, device, imgsz, batch, out_dir: Path) -> dict:
    """Evalua el modelo y devuelve un diccionario de metricas.

    `out_dir` recibe las graficas de cada evaluacion para que pesos,
    metricas y figuras queden juntos en la carpeta de resultados.

    Ademas de la metrica global se calcula `mAP50_gt_humano`: el promedio de
    AP50 restringido a las clases con anotacion humana, excluyendo `persona`.
    """
    res = model.val(
        data=data, split=split, device=device, imgsz=imgsz,
        batch=batch, plots=True, verbose=False,
        project=str(out_dir), name=f"eval_{split}",
    )
    box = res.box
    names = res.names

    per_class = {}
    for i, cls_idx in enumerate(box.ap_class_index):
        cid = int(cls_idx)
        per_class[names[cid]] = {
            "mAP50": round(float(box.ap50[i]), 4),
            "mAP50-95": round(float(box.ap[i]), 4),
        }

    honest = [v for k, v in per_class.items() if k in GT_HUMAN_CLASSES]
    out = {
        "split": split,
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "per_class": per_class,
    }
    if honest:
        out["mAP50_gt_humano"] = round(
            sum(v["mAP50"] for v in honest) / len(honest), 4
        )
        out["mAP50_95_gt_humano"] = round(
            sum(v["mAP50-95"] for v in honest) / len(honest), 4
        )
    return out


def rank_worst(per_class: dict, metric: str = "mAP50") -> list[tuple[str, float]]:
    """Ordena las clases de peor a mejor para la metrica indicada."""
    return sorted(
        ((k, v[metric]) for k, v in per_class.items()),
        key=lambda kv: kv[1],
    )


def save_report(path: Path, cfg: dict, metrics: dict) -> None:
    """Volca un informe legible junto a los artefactos del entrenamiento."""
    lines = [
        "Informe de evaluacion - Prevencion de Caidas y Riesgos en Escaleras",
        "=" * 64,
        f"modelo        : {cfg['model']}",
        f"epocas        : {cfg['epochs']}  imgsz: {cfg['imgsz']}  batch: {cfg['batch']}",
        f"seed          : {cfg['seed']}",
        f"dataset       : {cfg.get('data_final', cfg['data'])}",
        "",
    ]
    for split in ("val", "test"):
        m = metrics.get(split)
        if not m:
            continue
        lines += [
            f"--- {split.upper()} ---",
            f"  precision : {m['precision']:.4f}",
            f"  recall    : {m['recall']:.4f}",
            f"  mAP50     : {m['mAP50']:.4f}",
            f"  mAP50-95  : {m['mAP50-95']:.4f}",
        ]
        if "mAP50_gt_humano" in m:
            lines += [
                f"  mAP50 (solo clases con GT humano, sin `persona`): "
                f"{m['mAP50_gt_humano']:.4f}",
                f"  mAP50-95 (ídem)                                 : "
                f"{m['mAP50_95_gt_humano']:.4f}",
            ]
        lines += ["", "  clase                 mAP50   mAP50-95"]
        for name, v in sorted(m["per_class"].items(), key=lambda kv: kv[1]["mAP50"]):
            lines.append(f"  {name:20} {v['mAP50']:6.4f}   {v['mAP50-95']:7.4f}")
        worst = rank_worst(m["per_class"])
        if worst:
            lines += ["", f"  peor clase ({split}): {worst[0][0]} ({worst[0][1]:.4f})"]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    ap.add_argument("--set", nargs="*", default=[],
                    help="Sobrescribe claves, p.ej. --set epochs=50 batch=8")
    ap.add_argument("--model", help="Sobrescribe el modelo, p.ej. yolov8s")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--no-balanced", action="store_true")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Reanuda desde un checkpoint last.pt (p.ej. tras "
                         "interrumpir el entrenamiento). El resto de "
                         "argumentos se recupera del propio checkpoint.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    resuming = args.resume is not None
    for kv in args.set:
        k, _, v = kv.partition("=")
        cur = cfg.get(k)
        if isinstance(cur, bool):
            cfg[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            cfg[k] = int(v)
        elif isinstance(cur, float):
            cfg[k] = float(v)
        else:
            cfg[k] = v
    if args.model:
        cfg["model"] = args.model
        cfg["name"] = f"{args.model}_balanced" if cfg.get("use_balanced") else args.model
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.imgsz:
        cfg["imgsz"] = args.imgsz
    if args.no_balanced:
        cfg["use_balanced"] = False

    # reproducibilidad
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch
    from ultralytics import YOLO

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("=" * 64)
    print("ENTRENAMIENTO YOLOv8 - Prevencion de Caidas en Escaleras")
    print("=" * 64)
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- datos ------------------------------------------------------------ #
    data = cfg["data"]
    if resuming:
        # Al reanudar, Ultralytics restaura dataset, batch y optimizador desde
        # el checkpoint. Aqui solo se reconstruye el directorio balanceado para
        # que las rutas sigan existiendo.
        print("\n[1/4] Reanudando: se reconstruye el dataset balanceado...")
        if cfg.get("use_balanced"):
            build_balanced_dataset(cfg)
        data = cfg["balanced_data"] if cfg.get("use_balanced") else cfg["data"]
    elif cfg.get("use_balanced"):
        print("\n[1/4] Preparando dataset balanceado...")
        alt = build_balanced_dataset(cfg)
        if alt:
            data = str(alt)
    cfg["data_final"] = data

    print(f"\n[2/4] Dataset: {data}")

    # --- entrenamiento ---------------------------------------------------- #
    run_dir = Path(ROOT / cfg["project"]) / cfg["name"]

    if resuming:
        ckpt = args.resume if args.resume.is_absolute() else ROOT / args.resume
        if not ckpt.is_file():
            raise SystemExit(f"No existe el checkpoint: {ckpt}")
        print(f"\n[3/4] Reanudando desde {ckpt}...")
        model = YOLO(str(ckpt))
        model.train(resume=True)   # el resto sale del checkpoint
    else:
        print(f"\n[3/4] Entrenando {cfg['model']} ({cfg['epochs']} épocas, "
              f"imgsz={cfg['imgsz']}, batch={cfg['batch']})...")
        model = YOLO(f"{cfg['model']}.pt")
        model.train(
            data=data,
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            seed=seed,
            device=cfg["device"],
            workers=cfg["workers"],
            patience=cfg["patience"],
            project=str(ROOT / cfg["project"]),
            name=cfg["name"],
            exist_ok=True,
            hsv_h=cfg["hsv_h"], hsv_s=cfg["hsv_s"], hsv_v=cfg["hsv_v"],
            degrees=cfg["degrees"], translate=cfg["translate"], scale=cfg["scale"],
            shear=cfg["shear"], flipud=cfg["flipud"], fliplr=cfg["fliplr"],
            mosaic=cfg["mosaic"], close_mosaic=cfg["close_mosaic"], mixup=cfg["mixup"],
            lr0=cfg["lr0"], lrf=cfg["lrf"], momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"], warmup_epochs=cfg["warmup_epochs"],
            optimizer=cfg["optimizer"], cos_lr=cfg["cos_lr"],
            val=cfg["val"], plots=cfg["plots"], save_period=cfg["save_period"],
            verbose=True,
        )

    # --- evaluacion ------------------------------------------------------- #
    print("\n[4/4] Evaluando el mejor modelo...")
    best_path = run_dir / "weights" / "best.pt"
    out_dir = best_path.parent.parent
    best = YOLO(str(best_path))

    metrics = {}
    for split in ("val", "test"):
        metrics[split] = evaluate(
            best, data, split, cfg["device"], cfg["imgsz"], cfg["batch"], out_dir
        )

    save_report(out_dir / "INFORME.txt", cfg, metrics)
    (out_dir / "metricas.json").write_text(
        json.dumps({"config": cfg, "metricas": metrics},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # resumen en consola
    print("\n" + "=" * 64)
    print("RESULTADOS")
    print("=" * 64)
    for split in ("val", "test"):
        m = metrics[split]
        print(f"\n{split.upper()}")
        print(f"  precision {m['precision']:.4f} | recall {m['recall']:.4f} | "
              f"mAP50 {m['mAP50']:.4f} | mAP50-95 {m['mAP50-95']:.4f}")
        for name, v in sorted(m["per_class"].items(), key=lambda kv: kv[1]["mAP50"]):
            print(f"    {name:20} mAP50={v['mAP50']:.4f}  mAP50-95={v['mAP50-95']:.4f}")
    print(f"\nArtefactos: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())