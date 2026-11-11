#!/usr/bin/env python3
"""Arrancador del sistema: levanta el servidor, espera a que responda y abre
el panel en el navegador. Es lo que ejecuta el acceso directo que crea el
instalador (iniciar.bat / iniciar.sh).

    python tools/launch.py                 # 127.0.0.1:5000
    python tools/launch.py --port 8000 --no-browser

Cerrar esta ventana (o Ctrl+C) detiene el sistema de forma ordenada.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def open_browser(url: str) -> None:
    """En WSL el navegador está en Windows: se abre con explorer.exe."""
    if is_wsl():
        try:
            subprocess.Popen(["explorer.exe", url], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    webbrowser.open(url)


def health(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=2) as r:
            return json.load(r)
    except Exception:                                       # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    args, extra = ap.parse_known_args()
    try:                                  # mensajes al momento, aun redirigidos
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                       # noqa: BLE001
        pass
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"

    print("=" * 64)
    print(" Sistema de prevención de caídas en escaleras")
    print("=" * 64)
    if health(url):
        print(f"El sistema ya está en marcha: {url}")
        if not args.no_browser:
            open_browser(url)
        return 0

    env = {**os.environ, "PYTHONNOUSERSITE": "1",
           # configuración de Ultralytics dentro del proyecto, no en el perfil
           "YOLO_CONFIG_DIR": os.environ.get("YOLO_CONFIG_DIR",
                                             str(ROOT / ".cache" / "ultralytics"))}
    cmd = [sys.executable, "-m", "backend", "--host", args.host,
           "--port", str(args.port), *extra]
    print("Arrancando (la primera vez tarda más: carga los modelos)...")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)

    def stop(*_):
        if proc.poll() is None:
            print("\nDeteniendo el sistema...")
            proc.send_signal(signal.SIGINT if platform.system() != "Windows"
                             else signal.CTRL_C_EVENT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    signal.signal(signal.SIGTERM, stop)

    try:
        for _ in range(180):                    # hasta ~3 min (primera carga)
            if proc.poll() is not None:
                print("El servidor se cerró al arrancar. Revisa los mensajes de arriba.")
                return proc.returncode or 1
            h = health(url)
            if h:
                print(f"\nListo: {url}  ·  {h.get('device', '')}")
                print("Para detener el sistema: cierra esta ventana o pulsa Ctrl+C,")
                print("o usa el botón «Detener sistema» del panel.\n")
                if not args.no_browser:
                    open_browser(url)
                break
            time.sleep(1)
        return proc.wait()
    except KeyboardInterrupt:
        stop()
        return 0


if __name__ == "__main__":
    sys.exit(main())
