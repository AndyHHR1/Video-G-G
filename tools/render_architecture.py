#!/usr/bin/env python3
"""Genera docs/img/arquitectura.png (diagrama de arquitectura del sistema).

    python tools/render_architecture.py

Se dibuja con matplotlib para que el diagrama sea reproducible y se pueda
actualizar junto con el código.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img" / "arquitectura.png"

INK = "#1f2933"
MUTED = "#52606d"
LAYERS = {
    "src":   ("#e8eef7", "#3e6aa8"),
    "front": ("#e7f4ee", "#2f7d57"),
    "api":   ("#fdf1e3", "#b26b16"),
    "svc":   ("#f3ecfa", "#6b47a3"),
    "core":  ("#fbe9ea", "#a8323a"),
    "out":   ("#eef0f2", "#4b5563"),
}


def box(ax, x, y, w, h, title, lines=(), layer="core", title_size=10.5):
    fill, edge = LAYERS[layer]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc=fill, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 0.02, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=edge)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.052 - i * 0.025, t, ha="center", va="top",
                fontsize=8.2, color=INK)


def arrow(ax, x1, y1, x2, y2, label=None, color=MUTED, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, lw=1.2, color=color))
    if label:
        ax.text((x1 + x2) / 2 + 0.006, (y1 + y2) / 2, label, fontsize=7.6,
                color=color, va="center")


def band(ax, y, h, label):
    ax.text(0.012, y + h / 2, label, rotation=90, ha="center", va="center",
            fontsize=8.5, color=MUTED, fontweight="bold")


def main() -> None:
    fig = plt.figure(figsize=(13, 9.6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, "Sistema de prevención de caídas en escaleras · UPAO — arquitectura",
            ha="center", va="top", fontsize=14, fontweight="bold", color=INK)

    # 1. Fuentes
    band(ax, 0.86, 0.09, "ENTRADA")
    for i, (t, sub) in enumerate([("Webcam del navegador", "getUserMedia → JPEG"),
                                  ("Cámara IP / USB", "RTSP · índice (servidor)"),
                                  ("Imagen", "JPG / PNG"),
                                  ("Vídeo pregrabado", "MP4 / AVI (RQNF22)")]):
        box(ax, 0.04 + i * 0.24, 0.865, 0.21, 0.085, t, [sub], "src", 9.5)

    # 2. Frontend
    band(ax, 0.745, 0.09, "FRONTEND")
    box(ax, 0.04, 0.75, 0.93, 0.09, "frontend/templates/index.html  ·  panel de supervisión (RQF08)",
        ["Pestañas Imagen · Vídeo · Cámara en vivo · Clases   |   lienzo 960 px con cajas, "
         "riesgo, alertas, estado del sistema, «SR en vivo»"], "front")
    for i in range(4):
        arrow(ax, 0.145 + i * 0.24, 0.865, 0.145 + i * 0.24, 0.842)

    # 3. API
    band(ax, 0.6, 0.12, "API  (Flask)")
    apis = [("analysis.py", ["/api/detect  /api/frame", "/api/video  /api/stream"]),
            ("alerts.py", ["/api/alerts  /api/alertas", "ZIP · imagen de evidencia"]),
            ("system.py", ["/  ·  /api/health", "/api/cameras"]),
            ("auth.py", ["PANEL_TOKEN (RQNF19)", "hmac.compare_digest"])]
    for i, (t, ls) in enumerate(apis):
        box(ax, 0.04 + i * 0.24, 0.6, 0.21, 0.115, t, ls, "api", 10)
    arrow(ax, 0.5, 0.75, 0.5, 0.718, "  HTTP · JSON · JPEG crudo")

    # 4. Servicios
    band(ax, 0.43, 0.13, "SERVICIOS")
    box(ax, 0.04, 0.43, 0.42, 0.135, "services/inferencer.py — hilo CUDA único",
        ["cola de trabajos · modo asíncrono en vivo (frame_id)",
         "SR previa en imagen/vídeo · dibujo · evidencia"], "svc")
    box(ax, 0.49, 0.43, 0.23, 0.135, "services/live_enhancer.py",
        ["hilo propio · ranura «solo el último»", "Real-ESRGAN x4 ≈22 ms/cuadro",
         "caché por frame_id"], "svc")
    box(ax, 0.75, 0.43, 0.22, 0.135, "services/video_io.py",
        ["H.264 con ffmpeg (+audio)", "RTSP con reconexión (RQNF13)"], "svc")
    arrow(ax, 0.145, 0.6, 0.145, 0.567, "detect()")
    arrow(ax, 0.23, 0.6, 0.55, 0.567, "submit() si sr=1")

    # 5. Núcleo
    band(ax, 0.13, 0.28, "NÚCLEO")
    box(ax, 0.04, 0.13, 0.66, 0.28, "perception/engine.py — RiskEngine.step()", [], "core", 10.5)
    inner = [
        ("detectors.CocoDetector", ["YOLOv8s COCO", "personas · objetos (RQF02-1)"]),
        ("detectors.PostureDetector", ["Modelo S7 v4 · 6 clases", "postura · escalera"]),
        ("pose.PoseAnalyzer", ["MediaPipe Pose · 33 puntos", "pasamanos · distracción · tambaleo"]),
        ("tracking.IoUTracker", ["IoU > 0.25", "oclusión 1 s (RQF05)"]),
        ("alerts.AlertManager", ["persistencia ALTO 1.2 s / MEDIO 0.6 s", "cooldown 15 s (RQF06 · RQF07)"]),
        ("veredicto + estado", ["BAJO · MEDIO · ALTO", "OPERATIVO / DEGRADADO (RQNF09)"]),
    ]
    for i, (t, ls) in enumerate(inner):
        x = 0.06 + (i % 3) * 0.212
        y = 0.27 if i < 3 else 0.15
        box(ax, x, y, 0.195, 0.095, t, ls, "core", 8.8)
    box(ax, 0.73, 0.28, 0.24, 0.13, "enhancement/",
        ["SRVGGNetCompact (general-x4v3)", "RRDBNet (x4plus)", "PyTorch puro · FP16"], "core", 10)
    box(ax, 0.73, 0.13, 0.24, 0.13, "privacy/anonymizer.py",
        ["MediaPipe Face (corto + largo alcance)", "búsqueda por persona + cabeza",
         "pixelado (Ley N.º 29733)"], "core", 10)
    arrow(ax, 0.25, 0.43, 0.25, 0.412)
    arrow(ax, 0.6, 0.43, 0.8, 0.412)
    arrow(ax, 0.695, 0.19, 0.73, 0.19)

    # 6. Salidas
    band(ax, 0.02, 0.09, "SALIDA")
    box(ax, 0.04, 0.025, 0.45, 0.085, "outputs/alerts/<AAAA-MM-DD>/  +  alertas.jsonl",
        ["evidencia anotada y anonimizada · registro append-only (RQNF16)"], "out", 9.5)
    box(ax, 0.52, 0.025, 0.45, 0.085, "configs/app.yaml  ·  models/",
        ["umbrales, cadencias, persistencia, modelos (RQNF21) · pesos verificados SHA-256"],
        "out", 9.5)
    arrow(ax, 0.3, 0.13, 0.3, 0.112)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"guardado {OUT}")


if __name__ == "__main__":
    main()
