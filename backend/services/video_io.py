"""Entrada y salida de vídeo: recodificación H.264 y cámara del servidor."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2


def transcode_h264(src: Path, dst: Path,
                   audio_source: Path | None = None) -> tuple[bool, str]:
    """Recodifica `src` a H.264 (+faststart) y, si existe, añade el audio del
    vídeo original.

    OpenCV 4.10 solo escribe MPEG-4 Part 2 (`mp4v`), que los navegadores no
    reproducen; el ffmpeg empaquetado en `imageio-ffmpeg` sí trae libx264.
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [exe, "-y", "-loglevel", "error", "-i", str(src)]
        video = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if audio_source and audio_source.is_file():
            cmd += ["-i", str(audio_source), *video,
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "aac", "-b:a", "128k", str(dst)]
        else:
            cmd += [*video, str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        ok = r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0
        return ok, r.stderr[-200:]
    except Exception as exc:                                # noqa: BLE001
        return False, str(exc)


def parse_source(source):
    """'0' -> 0 (índice de cámara); cualquier otra cosa se deja como URL."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    return int(s) if s.isdigit() else s


def open_capture(source, attempts: int = 5) -> cv2.VideoCapture | None:
    """Abre una cámara o un flujo RTSP/HTTP reintentando con espera creciente
    (RQNF13: reconexión automática de la señal)."""
    src = parse_source(source)
    for i in range(max(1, attempts)):
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            return cap
        cap.release()
        time.sleep(min(1.0 * (i + 1), 8.0))
    return None
