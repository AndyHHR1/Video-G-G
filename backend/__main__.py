"""Punto de entrada:  python -m backend [opciones]

Ejemplos:
    python -m backend
    python -m backend --host 0.0.0.0 --port 8000
    python -m backend --device cpu --no-sr
    python -m backend --profile baja_vram          # fuerza el perfil GTX 1650
    python -m backend --set engine.pose_every=4 --set thresholds.alert=0.4
    python -m backend --stop                       # detiene un servidor en marcha
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import urllib.request

from .config import DEFAULT_CONFIG, load_settings


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m backend",
        description="Panel de prevención de caídas en escaleras (UPAO). "
                    "Vídeo en vivo + módulos de audio e imagen.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="Archivo YAML de configuración (por defecto configs/app.yaml)")
    ap.add_argument("--host", help="Dirección de escucha (server.host)")
    ap.add_argument("--port", type=int, help="Puerto (server.port)")
    ap.add_argument("--device", help="auto | cpu | cuda | índice de GPU")
    ap.add_argument("--profile", choices=("auto", "normal", "baja_vram"),
                    help="Perfil de hardware (hardware.profile). baja_vram = GTX 1650")
    ap.add_argument("--weights", help="Pesos del detector (models.detector)")
    ap.add_argument("--conf", type=float, help="Umbral por defecto del detector")
    sr = ap.add_mutually_exclusive_group()
    sr.add_argument("--sr", "--esrgan", dest="sr", action="store_true",
                    default=None, help="Fuerza la mejora Real-ESRGAN")
    sr.add_argument("--no-sr", dest="sr", action="store_false",
                    help="Desactiva la mejora Real-ESRGAN")
    ap.add_argument("--no-audio", action="store_true", help="No carga el módulo de audio")
    ap.add_argument("--no-imagen", action="store_true", help="No carga el módulo de imagen")
    ap.add_argument("--debug", action="store_true", help="Modo depuración de Flask")
    ap.add_argument("--stop", action="store_true",
                    help="Detiene el servidor que está corriendo en --host/--port y sale")
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR",
                    help="Sobrescribe cualquier valor del YAML (repetible)")
    return ap.parse_args(argv)


def stop_remote(host: str, port: int) -> int:
    """Pide a un servidor en marcha que se detenga (POST /api/system/stop)."""
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    url = f"http://{host}:{port}/api/system/stop"
    req = urllib.request.Request(url, method="POST", data=b"")
    token = os.environ.get("PANEL_TOKEN")
    if token:
        req.add_header("X-Panel-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read() or b"{}")
        print("  parada solicitada:", "ya estaba en curso"
              if body.get("ya_en_curso") else "deteniendo el sistema")
        return 0
    except Exception as exc:                                # noqa: BLE001
        print(f"  no se pudo contactar con {url}: {exc}")
        return 1


def main(argv=None) -> None:
    args = parse_args(argv)
    overrides = list(args.set)
    for flag, key in (("host", "server.host"), ("port", "server.port"),
                      ("device", "device"), ("weights", "models.detector"),
                      ("conf", "inference.conf"), ("sr", "enhancement.enabled"),
                      ("profile", "hardware.profile")):
        value = getattr(args, flag)
        if value is not None:
            overrides.append(f"{key}={value}")
    if args.debug:
        overrides.append("server.debug=true")
    if args.no_audio:
        overrides.append("modules.audio.enabled=false")
    if args.no_imagen:
        overrides.append("modules.imagen.enabled=false")
    settings = load_settings(args.config, overrides)
    s = settings.server
    if args.stop:
        sys.exit(stop_remote(s.host, s.port))

    # Menos fragmentación de memoria en GPUs de 4 GB (GTX 1650).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from werkzeug.serving import make_server

    from .app import create_app          # import tardío: carga torch/ultralytics
    from .services.lifecycle import Lifecycle

    print("=" * 64)
    print(" Sistema de prevención de caídas en escaleras · UPAO")
    print(" Vídeo en vivo · módulo de audio · módulo de imagen")
    print("=" * 64)
    print(f"  detector : {settings.models.detector}")
    print(f"  COCO     : {settings.models.coco}")
    print(f"  device   : {settings.device}")
    print(f"  SR       : {settings.enhancement.enabled}")
    lifecycle = Lifecycle()
    app = create_app(settings, lifecycle)

    server = make_server(s.host, s.port, app, threaded=True)
    lifecycle.bind_server(server.shutdown)

    def _on_signal(signum, _frame):
        name = signal.Signals(signum).name
        if not lifecycle.shutdown(f"señal {name}", background=True):
            print("  (ya se está deteniendo; espera unos segundos)", flush=True)

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    print(f"  listo    : http://{s.host}:{s.port}")
    print("  detener  : botón «Detener sistema» del panel, Ctrl+C o "
          "python -m backend --stop")
    print("=" * 64, flush=True)
    try:
        server.serve_forever()
    finally:
        if not lifecycle.stopping:            # salida por excepción
            lifecycle.shutdown("salida del servidor")
        lifecycle.wait(15)
        server.server_close()


if __name__ == "__main__":
    main()
