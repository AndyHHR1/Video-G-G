#!/usr/bin/env python3
"""¿Mejora la morfología (apertura/cierre de la máscara) la pose en MI escalera?

    python tools/eval_morfologia_pose.py video_escalera.mp4 [--frames 600]

Pasa el mismo vídeo por YOLOv8-Pose dos veces: tal cual y con la máscara de
primer plano limpiada con morfología (`skeleton.morphology: mascara`), y
compara:

  * confianza media de los keypoints y fracción de puntos fiables (> 0.5),
  * personas por fotograma,
  * identidades distintas (menos = seguimiento más estable).

Si la columna «mascara» no gana con claridad, dejar `morphology: "off"`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend  # noqa: E402,F401  (torch antes que mediapipe)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.config import load_settings  # noqa: E402
from backend.perception.engine import resolve_device  # noqa: E402
from backend.perception.skeleton.estimator import PoseEstimator  # noqa: E402
from backend.perception.skeleton.morphology import ForegroundMorphology  # noqa: E402


def run(video: str, frames: int, morph: bool) -> dict:
    settings = load_settings()
    sk = settings.skeleton
    est = PoseEstimator(str(sk.model_path), resolve_device(settings.device),
                        imgsz=sk.imgsz, conf=sk.conf, iou=sk.iou,
                        tracker=str(sk.tracker_path),
                        kp_conf_min=sk.keypoints.conf_min,
                        min_keypoints=sk.min_keypoints)
    fg = ForegroundMorphology() if morph else None
    cap = cv2.VideoCapture(video)
    confs, people, ids = [], [], set()
    n = 0
    while n < frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        src = fg(frame) if fg is not None else frame
        obs = est(src)
        people.append(len(obs))
        for o in obs:
            confs.append(o.kconf)
            if o.track_id is not None:
                ids.add(o.track_id)
    cap.release()
    c = np.concatenate(confs) if confs else np.zeros(1)
    return {"fotogramas": n, "conf_media": float(c.mean()),
            "kp>0.5": float((c > 0.5).mean()),
            "personas/fotograma": float(np.mean(people)) if people else 0.0,
            "identidades": len(ids)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=600)
    a = ap.parse_args()
    base = run(a.video, a.frames, False)
    morph = run(a.video, a.frames, True)
    print(f"{'medida':22s} {'sin morfología':>15s} {'mascara':>10s}")
    for k in base:
        print(f"{k:22s} {base[k]:15.3f} {morph[k]:10.3f}")
    better = (morph["kp>0.5"] - base["kp>0.5"] > 0.02
              and morph["identidades"] <= base["identidades"])
    print("\nRecomendación:", "skeleton.morphology: mascara" if better
          else 'skeleton.morphology: "off" (no mejora de forma clara)')
    return 0


if __name__ == "__main__":
    sys.exit(main())
