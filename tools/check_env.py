#!/usr/bin/env python3
"""Diagnóstico del entorno: versiones, GPU, modelos y compatibilidad.

    python tools/check_env.py

Útil antes de una demo o al instalar en otra máquina.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ok = True


def line(status: bool, text: str) -> None:
    global ok
    ok &= status
    print(f"  [{'OK' if status else '!!'}] {text}")


print(f"Python {sys.version.split()[0]}  ({sys.executable})")
line(sys.version_info >= (3, 10), "Python >= 3.10")

import os  # noqa: E402
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# torch debe importarse antes que mediapipe (ver backend/__init__.py)
try:
    import torch
    cuda = torch.cuda.is_available()
    line(True, f"torch {torch.__version__}  CUDA={'sí' if cuda else 'no'}")
    if cuda:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        line(True, f"GPU {name}  sm_{cap[0]}{cap[1]}  {vram:.1f} GB VRAM")
        line(vram >= 6, "VRAM >= 6 GB (RQNF03)")
except Exception as exc:                                    # noqa: BLE001
    line(False, f"torch no importa: {exc}")

for mod, want in (("numpy", "1.26"), ("cv2", "4.10"), ("mediapipe", "0.10.21"),
                  ("ultralytics", "8.4"), ("flask", None), ("yaml", None)):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", None) if mod != "flask" else None
        if ver is None:
            from importlib.metadata import version
            ver = version({"cv2": "opencv-python", "yaml": "PyYAML"}.get(mod, mod))
        line(want is None or str(ver).startswith(want), f"{mod} {ver}")
    except Exception as exc:                                # noqa: BLE001
        line(False, f"{mod}: {exc}")

try:
    import imageio_ffmpeg
    line(Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file(), "ffmpeg (imageio-ffmpeg)")
except Exception as exc:                                    # noqa: BLE001
    line(False, f"ffmpeg: {exc}")

print("Modelos:")
import subprocess  # noqa: E402
r = subprocess.run([sys.executable, str(ROOT / "tools" / "download_models.py"),
                    "--check"], capture_output=True, text=True)
print("\n".join("  " + x for x in r.stdout.strip().splitlines()[1:]))
ok &= r.returncode == 0

print("\nEntorno correcto." if ok else "\nHay problemas (ver [!!]).")
sys.exit(0 if ok else 1)
