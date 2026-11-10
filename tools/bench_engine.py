#!/usr/bin/env python3
"""Mide el rendimiento real del motor de vídeo (RiskEngine.step) por fotograma.

    python tools/bench_engine.py                       # GPU si hay
    python tools/bench_engine.py --set device=cpu      # solo procesador
    python tools/bench_engine.py --frames DIR --n 300

Usa como "vídeo" una carpeta de fotogramas (por defecto una secuencia de
peatones del split de TEST de stairs_fallhuman), reescalados a 960 px como
los que envía el panel. No escribe alertas ni evidencias.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_FRAMES = ROOT / "stairs_fallhuman/tracking/test/AVG-TownCentre/img1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    help="clave.sub=valor de configs/app.yaml")
    args = ap.parse_args()

    import backend  # noqa: F401  (importa torch antes que mediapipe)
    import cv2
    from backend.config import load_settings
    from backend.perception.engine import RiskEngine

    files = sorted(p for p in args.frames.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        print(f"Sin fotogramas en {args.frames}")
        return 1
    frames = []
    for p in files[: args.n + args.warmup]:
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        if w > 960:
            im = cv2.resize(im, (960, round(h * 960 / w)), interpolation=cv2.INTER_AREA)
        frames.append(im)
    while len(frames) < args.n + args.warmup:
        frames += frames[: args.n + args.warmup - len(frames)]

    from backend.hardware import apply_profile
    s = load_settings(overrides=args.overrides)
    profile, changes = apply_profile(s)           # igual que backend/app.py
    eng = RiskEngine(s)
    print(f"perfil: {profile} {changes or ''}")
    print(f"dispositivo: {eng.device} · fotogramas: {len(frames)} "
          f"({frames[0].shape[1]}x{frames[0].shape[0]})")
    for f in frames[: args.warmup]:
        eng.step(f, allow_alerts=False)

    ts, stages = [], {}
    for f in frames[args.warmup:]:
        t = time.perf_counter()
        r = eng.step(f, allow_alerts=False)
        ts.append((time.perf_counter() - t) * 1000)
        for k, v in (r.get("stages") or {}).items():
            if isinstance(v, (int, float)):
                stages.setdefault(k, []).append(v)
    ts.sort()
    p50, p95 = ts[len(ts) // 2], ts[int(len(ts) * 0.95)]
    mean = statistics.fmean(ts)
    print(f"motor: media {mean:.1f} ms · p50 {p50:.1f} ms · p95 {p95:.1f} ms "
          f"-> {1000 / mean:.1f} fps")
    if stages:
        print("etapas (media ms): " + " · ".join(
            f"{k} {statistics.fmean(v):.1f}" for k, v in stages.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
