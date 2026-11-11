#!/usr/bin/env python3
"""
Modelo v5 - Entrenamiento, evaluación en test y medida de velocidad.

    python training/train_v5.py --task postura  --model yolo11n-cls --name postura_n
    python training/train_v5.py --task escalera --model yolo11n     --name escalera_n

Entrena en la GPU (`--device 0`), guarda en runs/v5/<name>/ y escribe
runs/v5/<name>/v5_metrics.json con:
  * métricas en el split de TEST (nunca visto en el entrenamiento)
  * latencia por inferencia en GPU y en CPU (para equipos modestos)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = {"postura": ROOT / "datasets" / "v5_postura",
        "escalera": ROOT / "datasets" / "v5_escalera" / "data.yaml"}

# Aumentos: la cámara es fija pero cambia la luz (RQNF12) y la persona puede
# aparecer en cualquier lado; las escaleras son direccionales (sin flipud).
COMMON = dict(seed=42, deterministic=False, plots=True, exist_ok=True,
              hsv_h=0.015, hsv_s=0.6, hsv_v=0.45, fliplr=0.5, flipud=0.0)
TASK_ARGS = {
    "postura": dict(imgsz=224, epochs=80, batch=64, patience=20,
                    auto_augment="randaugment", erasing=0.3, degrees=10.0,
                    translate=0.1, scale=0.3, dropout=0.2, lr0=0.001,
                    optimizer="AdamW", cos_lr=True, warmup_epochs=3),
    "escalera": dict(imgsz=512, epochs=100, batch=16, patience=25,
                     degrees=5.0, translate=0.1, scale=0.5, shear=2.0,
                     mosaic=1.0, close_mosaic=10, mixup=0.05, cos_lr=True),
}


def latency(model, task: str, imgsz: int, device, n: int = 60) -> float:
    """Milisegundos por imagen (mediana) con una imagen sintética."""
    rng = np.random.default_rng(0)
    shape = (256, 128, 3) if task == "postura" else (540, 960, 3)
    img = rng.integers(0, 255, shape, dtype=np.uint8)
    for _ in range(5):
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
        ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=DATA, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.task == "postura":
        # En WSL2 la memoria "fijada" (pinned) del anfitrión es limitada y el
        # hilo pin-memory del DataLoader de clasificación fallaba con
        # "CUDA error: out of memory" con la GPU casi vacía. El entrenador de
        # clasificación de Ultralytics no expone la opción (el de detección sí),
        # así que se fija aquí a False.
        import functools
        import importlib
        for mod in ("ultralytics.models.yolo.classify.train",
                    "ultralytics.models.yolo.classify.val"):
            m_ = importlib.import_module(mod)
            if hasattr(m_, "build_dataloader"):
                m_.build_dataloader = functools.partial(m_.build_dataloader,
                                                        pin_memory=False)

    hp = {**COMMON, **TASK_ARGS[args.task]}
    for k in ("epochs", "imgsz", "batch"):
        if getattr(args, k):
            hp[k] = getattr(args, k)

    project = ROOT / "runs" / "v5"
    model = YOLO(args.model)
    model.train(data=str(DATA[args.task]), device=args.device,
                workers=args.workers, project=str(project), name=args.name, **hp)

    best = project / args.name / "weights" / "best.pt"
    m = YOLO(str(best))
    out = {"task": args.task, "base": args.model, "weights": str(best.relative_to(ROOT)),
           "hp": {k: v for k, v in hp.items() if not isinstance(v, Path)}}

    if args.task == "postura":
        r = m.val(data=str(DATA["postura"]), split="test", imgsz=hp["imgsz"],
                  device=args.device, plots=True, project=str(project),
                  name=f"{args.name}_test", exist_ok=True)
        cm = r.confusion_matrix.matrix            # filas = predicho, cols = real
        names = m.names
        per = {}
        for i, nm in names.items():
            real = cm[:, i].sum()
            per[nm] = {"recall": round(float(cm[i, i] / real), 4) if real else None,
                       "n_test": int(real)}
        out["test"] = {"top1": round(float(r.top1), 4),
                       "top5": round(float(r.top5), 4), "per_class": per,
                       "confusion_pred_x_real": cm.astype(int).tolist(),
                       "names": names}
    else:
        r = m.val(data=str(DATA["escalera"]), split="test", imgsz=hp["imgsz"],
                  device=args.device, plots=True, project=str(project),
                  name=f"{args.name}_test", exist_ok=True)
        out["test"] = {"mAP50": round(float(r.box.map50), 4),
                       "mAP50_95": round(float(r.box.map), 4),
                       "precision": round(float(r.box.mp), 4),
                       "recall": round(float(r.box.mr), 4)}

    out["latency_ms"] = {"gpu": latency(m, args.task, hp["imgsz"], args.device),
                         "cpu": latency(m, args.task, hp["imgsz"], "cpu", n=30)}
    out["size_mb"] = round(best.stat().st_size / 1e6, 2)
    (project / args.name / "v5_metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("V5_METRICS", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
