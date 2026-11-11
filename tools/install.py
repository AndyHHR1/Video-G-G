#!/usr/bin/env python3
"""Instalador del sistema (parte común a Windows, Linux, WSL y macOS).

No se ejecuta a mano: lo lanzan `instalar.bat` (Windows) o `instalar.sh`
(Linux/WSL/macOS) con el Python 3.12 del entorno `.venv`, después de preparar
`uv`. Pasos (todos se pueden repetir: lo que ya está bien no se toca):

  1. Detecta la GPU NVIDIA -> PyTorch con CUDA 12.8, si no -> PyTorch CPU.
  2. Comprueba cada paquete de requirements/ (nombre y versión) y SOLO
     instala lo que falta o no cumple la versión.
  3. Descarga y verifica (SHA-256) los modelos que falten.
  4. Diagnóstico (tools/check_env.py).
  5. Crea el arrancador: iniciar.bat / iniciar.sh (+ acceso directo lo crea
     el script de cada sistema).

Todo queda dentro de la carpeta del proyecto (.venv, .python, .cache).
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements"
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def has_nvidia() -> bool:
    if IS_MAC:
        return False
    exe = shutil.which("nvidia-smi") or (
        "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else None)
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            say(f"  GPU: {out.stdout.strip().splitlines()[0]}")
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    return False


# --------------------------------------------------------------------------- #
# Comprobación de paquetes
# --------------------------------------------------------------------------- #
_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(.*?)\s*(#.*)?$")


def read_requirements(path: Path) -> list[tuple[str, str]]:
    """[(nombre, especificador)] de un requirements, siguiendo `-r otro.txt`."""
    out: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-r "):
            out += read_requirements(path.parent / line[3:].strip())
            continue
        m = _LINE.match(line)
        if m:
            out.append((m.group(1), (m.group(3) or "").split(";")[0].strip()))
    return out


def _vtuple(v: str) -> tuple:
    base = v.split("+")[0]
    return tuple(int(p) if p.isdigit() else 0
                 for p in re.split(r"[.\-]", base)[:4])


def satisfies(installed: str, spec: str) -> bool:
    """Comprueba la versión instalada contra `==x`, `>=x`, `<x`, `>x`, `<=x`,
    `!=x` separados por comas. Usa `packaging` si está; si no, comparación
    numérica simple (suficiente para los pines de este proyecto)."""
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        return Version(installed) in SpecifierSet(spec)
    except Exception:                                       # noqa: BLE001
        pass
    iv = _vtuple(installed)
    for part in spec.split(","):
        part = part.strip()
        m = re.match(r"(==|>=|<=|!=|>|<)\s*(.+)", part)
        if not m:
            continue
        op, ver = m.groups()
        if op == "==" and "+" in ver:             # torch==2.11.0+cu128
            if installed != ver:
                return False
            continue
        rv = _vtuple(ver)
        ok = {"==": iv[:len(rv)] == rv, ">=": iv >= rv, "<=": iv <= rv,
              ">": iv > rv, "<": iv < rv, "!=": iv[:len(rv)] != rv}[op]
        if not ok:
            return False
    return True


def check(path: Path) -> list[str]:
    """Imprime el estado de cada paquete y devuelve los que faltan/no cumplen."""
    missing = []
    for name, spec in read_requirements(path):
        try:
            v = metadata.version(name)
        except metadata.PackageNotFoundError:
            say(f"    [FALTA] {name} {spec}")
            missing.append(name)
            continue
        if satisfies(v, spec):
            say(f"    [OK]    {name} {v}")
        else:
            say(f"    [VERSIÓN] {name} {v} (se necesita {spec})")
            missing.append(name)
    return missing


def uv_install(uv: str, args: list[str]) -> None:
    py = sys.executable
    cmd = [uv, "pip", "install", "--python", py, *args]
    env = {**os.environ, "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "600")}
    for attempt in (1, 2, 3):                 # redes lentas: reintentos
        r = subprocess.run(cmd, env=env)
        if r.returncode == 0:
            return
        say(f"  reintento {attempt}...")
    raise SystemExit("ERROR: no se pudieron instalar los paquetes (revisa la conexión).")


# --------------------------------------------------------------------------- #
# Arrancador
# --------------------------------------------------------------------------- #
def write_launchers() -> list[Path]:
    made = []
    if IS_WIN:
        bat = ROOT / "iniciar.bat"
        bat.write_text(
            "@echo off\r\n"
            "rem Arranca el sistema y abre el panel (creado por instalar.bat)\r\n"
            "cd /d \"%~dp0\"\r\n"
            "title Sistema de prevencion de caidas en escaleras\r\n"
            "if not exist \".venv\\Scripts\\python.exe\" (\r\n"
            "  echo Falta el entorno. Ejecuta primero instalar.bat\r\n"
            "  pause & exit /b 1\r\n)\r\n"
            "\".venv\\Scripts\\python.exe\" tools\\launch.py %*\r\n"
            "if errorlevel 1 pause\r\n", encoding="ascii")
        made.append(bat)
    else:
        sh = ROOT / "iniciar.sh"
        sh.write_text(
            "#!/usr/bin/env bash\n"
            "# Arranca el sistema y abre el panel (creado por instalar.sh)\n"
            "cd \"$(dirname \"$(readlink -f \"$0\")\")\"\n"
            "[ -x .venv/bin/python ] || { echo 'Falta el entorno. Ejecuta primero ./instalar.sh'; "
            "read -rp 'Enter para salir...'; exit 1; }\n"
            "export PYTHONNOUSERSITE=1\n"
            ".venv/bin/python tools/launch.py \"$@\" || read -rp 'Enter para salir...'\n",
            encoding="utf-8")
        sh.chmod(0o755)
        made.append(sh)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uv", required=True, help="ruta al ejecutable de uv")
    ap.add_argument("--cpu", action="store_true", help="forzar PyTorch CPU")
    args = ap.parse_args()

    say("\n== 1/5 Hardware")
    gpu = not args.cpu and has_nvidia()
    torch_req = REQ / ("torch-cu128.txt" if gpu else "torch-cpu.txt")
    say(f"  PyTorch: {'CUDA 12.8 (GPU NVIDIA)' if gpu else 'CPU'}")

    say("\n== 2/5 Paquetes")
    reqs = [torch_req, REQ / "base.txt", REQ / "modulos.txt"]
    if not gpu:
        reqs.append(REQ / "accel.txt")        # OpenVINO: solo sin GPU
    for req in reqs:
        if not req.is_file():
            continue
        say(f"  {req.relative_to(ROOT)}:")
        missing = check(req)
        if missing:
            say(f"  -> instalando {len(missing)} paquete(s)...")
            uv_install(args.uv, ["-r", str(req)])
        else:
            say("  -> todo instalado")

    say("\n== 3/5 Modelos")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "download_models.py")])
    if r.returncode != 0:
        say("  AVISO: algún modelo falta o no coincide (ver arriba).")

    say("\n== 4/5 Diagnóstico")
    subprocess.run([sys.executable, str(ROOT / "tools" / "check_env.py")])

    say("\n== 5/5 Arrancador")
    for p in write_launchers():
        say(f"  creado {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
