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
    "mod":   ("#e6f6f8", "#1d7a8a"),
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
    for i, (t, sub, layer) in enumerate([
            ("Webcam del navegador", "vídeo en vivo · JPEG 960 px", "src"),
            ("Cámara IP / USB", "RTSP · índice (servidor)", "src"),
            ("Micrófono del navegador", "módulo audio · PCM 16 kHz", "mod"),
            ("Fotografía / cámara", "módulo imagen · JPG / PNG", "mod")]):
        box(ax, 0.04 + i * 0.24, 0.865, 0.21, 0.085, t, [sub], layer, 9.5)

    # 2. Frontend
    band(ax, 0.745, 0.09, "FRONTEND")
    box(ax, 0.04, 0.75, 0.93, 0.09, "frontend/templates/index.html  ·  panel de supervisión (RQF08)",
        ["Pestañas Cámara en vivo · Audio · Imagen · Clases   |   cajas, riesgo, alertas "
         "(vídeo + audio), «SR en vivo», botón «Detener sistema»"], "front")
    for i in range(4):
        arrow(ax, 0.145 + i * 0.24, 0.865, 0.145 + i * 0.24, 0.842)

    # 3. API
    band(ax, 0.6, 0.12, "API  (Flask)")
    apis = [("analysis.py", ["/api/frame  /api/stream", "solo vídeo en vivo"]),
            ("alerts.py", ["/api/alerts  /api/alertas", "ZIP · fuentes extra (audio)"]),
            ("system.py", ["/  ·  /api/health", "/api/system/stop"]),
            ("auth.py", ["PANEL_TOKEN (RQNF19)", "hmac.compare_digest"])]
    for i, (t, ls) in enumerate(apis):
        box(ax, 0.04 + i * 0.24, 0.6, 0.21, 0.115, t, ls, "api", 10)
    arrow(ax, 0.5, 0.75, 0.5, 0.718, "  HTTP · JSON · JPEG crudo")

    # 4. Servicios
    band(ax, 0.43, 0.13, "SERVICIOS")
    box(ax, 0.04, 0.43, 0.27, 0.135, "services/inferencer.py",
        ["hilo CUDA único · cola", "asíncrono en vivo (frame_id)",
         "dibujo · evidencia"], "svc", 9.5)
    box(ax, 0.33, 0.43, 0.19, 0.135, "live_enhancer.py",
        ["hilo propio · Real-ESRGAN x4", "FP16 · FP32 en GTX 16xx",
         "caché por frame_id"], "svc", 9.5)
    box(ax, 0.54, 0.43, 0.17, 0.135, "video_io · lifecycle",
        ["RTSP con reconexión", "parada ordenada", "hardware.py (GTX 1650)"], "svc", 9.5)
    box(ax, 0.73, 0.43, 0.24, 0.135, "modulos/  (aparte del vídeo)",
        ["audio/: reglas final.py + MLP", "imagen/: YOLO11 + YOLO11-Pose",
         "se cargan aislados"], "mod", 9.5)
    arrow(ax, 0.145, 0.6, 0.145, 0.567, "detect()")
    arrow(ax, 0.23, 0.6, 0.42, 0.567, "submit() si sr=1")
    arrow(ax, 0.983, 0.75, 0.983, 0.567)
    ax.text(0.992, 0.66, "/api/audio · /api/imagen", rotation=90, ha="center",
            va="center", fontsize=7.6, color=MUTED)

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
        ["SRVGGNetCompact (general-x4v3)", "vídeo en vivo", "PyTorch puro · FP16/FP32"], "core", 10)
    box(ax, 0.73, 0.13, 0.24, 0.13, "privacy/anonymizer.py",
        ["MediaPipe Face (corto + largo alcance)", "búsqueda por persona + cabeza",
         "pixelado (Ley N.º 29733)"], "core", 10)
    arrow(ax, 0.25, 0.43, 0.25, 0.412)
    arrow(ax, 0.42, 0.43, 0.8, 0.412)
    arrow(ax, 0.695, 0.19, 0.73, 0.19)

    # 6. Salidas
    band(ax, 0.02, 0.09, "SALIDA")
    box(ax, 0.04, 0.025, 0.45, 0.085, "outputs/alerts/  +  outputs/audio_alerts/",
        ["evidencia anotada y anonimizada · registros JSONL (RQNF16)"], "out", 9.5)
    box(ax, 0.52, 0.025, 0.45, 0.085, "configs/app.yaml  ·  models/",
        ["umbrales, cadencias, persistencia, modelos (RQNF21) · pesos verificados SHA-256"],
        "out", 9.5)
    arrow(ax, 0.3, 0.13, 0.3, 0.112)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"guardado {OUT}")


if __name__ == "__main__":
    main()
