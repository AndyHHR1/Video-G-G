"""Punto de entrada:  python -m backend [opciones]

Ejemplos:
    python -m backend
    python -m backend --host 0.0.0.0 --port 8000
    python -m backend --device cpu --no-sr
    python -m backend --set engine.pose_every=4 --set thresholds.alert=0.4
"""
from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG, load_settings


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m backend",
        description="Panel de prevención de caídas en escaleras (UPAO).")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="Archivo YAML de configuración (por defecto configs/app.yaml)")
    ap.add_argument("--host", help="Dirección de escucha (server.host)")
    ap.add_argument("--port", type=int, help="Puerto (server.port)")
    ap.add_argument("--device", help="auto | cpu | cuda | índice de GPU")
    ap.add_argument("--weights", help="Pesos del detector (models.detector)")
    ap.add_argument("--conf", type=float, help="Umbral por defecto del detector")
    sr = ap.add_mutually_exclusive_group()
    sr.add_argument("--sr", "--esrgan", dest="sr", action="store_true",
                    default=None, help="Fuerza la mejora Real-ESRGAN")
    sr.add_argument("--no-sr", dest="sr", action="store_false",
                    help="Desactiva la mejora Real-ESRGAN")
    ap.add_argument("--debug", action="store_true", help="Modo depuración de Flask")
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR",
                    help="Sobrescribe cualquier valor del YAML (repetible)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    overrides = list(args.set)
    for flag, key in (("host", "server.host"), ("port", "server.port"),
                      ("device", "device"), ("weights", "models.detector"),
                      ("conf", "inference.conf"), ("sr", "enhancement.enabled")):
        value = getattr(args, flag)
        if value is not None:
            overrides.append(f"{key}={value}")
    if args.debug:
        overrides.append("server.debug=true")
    settings = load_settings(args.config, overrides)

    from .app import create_app          # import tardío: carga torch/ultralytics
    s = settings.server
    print("=" * 64)
    print(" Sistema de prevención de caídas en escaleras · UPAO")
    print("=" * 64)
    print(f"  detector : {settings.models.detector}")
    print(f"  COCO     : {settings.models.coco}")
    print(f"  device   : {settings.device}")
    print(f"  SR       : {settings.enhancement.enabled}")
    app = create_app(settings)
    print(f"  listo    : http://{s.host}:{s.port}")
    print("=" * 64, flush=True)
    app.run(host=s.host, port=s.port, debug=s.debug, threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()
