#!/usr/bin/env python3
"""
Extracción de audio sincronizado con las alertas de caída.

Este módulo se integra con el flujo existente de `RiskEngine` sin modificar
su lógica: se encarga exclusivamente de capturar y extraer clips de audio
alrededor de los eventos de caída detectados.

Modos de operación:

  * VIDEO  — Cuando el video subido contiene pista de audio, se extrae un clip
    de 0.5 s ANTES de la caída hasta 0.5 s DESPUÉS (configurable) usando el
    timestamp del fotograma donde se disparó la alerta.

  * NAVEGADOR — El navegador captura audio+vídeo con `getUserMedia` (audio:true).
    Envía paquetes WAV/Opus en streaming al endpoint `/api/audio`. El servidor
    mantiene un ring buffer de los últimos N segundos; al dispararse una alerta
    se extrae la ventana de 0.5 s antes → 0.5 s después de `alert._ts`.

  * SERVIDOR — El endpoint `/api/stream` usa una cámara física del servidor
    sin micrófono asociado. El audio no está disponible en este modo. La UI
    muestra la limitación.

La extracción para video reutiliza el ffmpeg empaquetado en `imageio-ffmpeg`,
igual que ya hace el servidor para la transcodificación de vídeo.
"""

from __future__ import annotations

import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALERT_DIR = ROOT / "runs" / "alerts"

# Ventana de audio alrededor de la caída (segundos).
AUDIO_PRE_FALL = 0.5
AUDIO_POST_FALL = 0.5


@dataclass
class AudioClip:
    """Metadatos del clip de audio extraído para una alerta."""
    alert_id: str
    path: str
    start_ts: float
    end_ts: float
    duration: float


def _ffmpeg_exe() -> str:
    """Devuelve la ruta al binario de ffmpeg empaquetado."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_video_audio(
    video_path: str | Path,
    fall_timestamp: float,
    alert_id: str,
    pre: float = AUDIO_PRE_FALL,
    post: float = AUDIO_POST_FALL,
) -> AudioClip | None:
    """Extrae un clip de audio de un archivo de video.

    Args:
        video_path: Ruta al video original (con pista de audio).
        fall_timestamp: Tiempo en segundos del fotograma donde se disparó
                        la alerta.
        alert_id: ID de la alerta para nombrar el archivo.
        pre: Segundos de audio *antes* de la caída.
        post: Segundos de audio *después* de la caída.

    Returns:
        AudioClip con la ruta al archivo WAV, o None si el video no tiene
        pista de audio o ffmpeg falla.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        return None

    start = max(0.0, fall_timestamp - pre)
    duration = pre + post

    # Verificar si el video tiene pista de audio.
    probe = subprocess.run(
        [_ffmpeg_exe(), "-i", str(video_path), "-hide_banner"],
        capture_output=True, text=True,
    )
    if "Audio:" not in probe.stderr:
        return None

    dia = time.strftime("%Y-%m-%d")
    out_dir = ALERT_DIR / dia
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audio_{alert_id}.wav"

    r = subprocess.run(
        [_ffmpeg_exe(), "-y", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", str(video_path),
         "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
         str(out_path)],
        capture_output=True, text=True, timeout=60,
    )

    if not r.returncode == 0 or not out_path.is_file():
        if out_path.is_file():
            out_path.unlink(missing_ok=True)
        return None

    return AudioClip(
        alert_id=alert_id,
        path=str(out_path),
        start_ts=start,
        end_ts=start + duration,
        duration=duration,
    )


@dataclass
class _AudioFrame:
    ts: float
    data: bytes


class LiveAudioBuffer:
    """Ring buffer de audio para el modo cámara en vivo (navegador).

    Recibe paquetes de audio WAV (mono, 16-bit, 16 kHz) del cliente y los
    almacena con su timestamp. Mantiene `keep_seconds` segundos de historia
    para poder extraer la ventana PRE y POST de una caída.
    """

    def __init__(self, keep_seconds: float = 10.0) -> None:
        self._buf: deque[_AudioFrame] = deque()
        self._keep = keep_seconds
        self._lock = threading.Lock()
        self._samples_per_sec = 16000
        self._bytes_per_sample = 2  # 16-bit
        self._channels = 1

    def push_wav(self, wav_bytes: bytes, recv_ts: float | None = None) -> None:
        """Almacena un paquete WAV. `recv_ts` es el tiempo de recepción.

        Se extrae el PCM raw del WAV (se descarta la cabecera) para poder
        concatenar varios chunks sin generar cabeceras duplicadas. El WAV de
        salida se construye al vuelo en `extract_clip`.
        """
        if not wav_bytes:
            return
        recv_ts = recv_ts or time.time()
        pcm = self._strip_wav_header(wav_bytes)
        if not pcm:
            return
        with self._lock:
            self._buf.append(_AudioFrame(ts=recv_ts, data=pcm))
            self._prune()

    @staticmethod
    def _strip_wav_header(wav_bytes: bytes) -> bytes:
        """Extrae los datos PCM de un archivo WAV (44-byte header por defecto)."""
        import struct
        if len(wav_bytes) < 44 or not wav_bytes[:4] == b"RIFF":
            return wav_bytes
        # Leer el tamaño de los datos desde el header.
        data_offset = 44
        pos = 12
        while pos < len(wav_bytes) - 8:
            chunk_id = wav_bytes[pos:pos + 4]
            chunk_size = struct.unpack_from("<I", wav_bytes, pos + 4)[0]
            if chunk_id == b"data":
                data_offset = pos + 8
                break
            pos += 8 + chunk_size
            if chunk_size == 0:
                break
        return wav_bytes[data_offset:]

    def _prune(self) -> None:
        """Descarta frames más antiguos que `keep_seconds`."""
        cutoff = time.time() - self._keep
        while self._buf and self._buf[0].ts < cutoff:
            self._buf.popleft()

    def extract_clip(
        self,
        fall_ts: float,
        alert_id: str,
        pre: float = AUDIO_PRE_FALL,
        post: float = AUDIO_POST_FALL,
    ) -> AudioClip | None:
        """Extrae un clip WAV de la ventana alrededor de `fall_ts`.

        Returns:
            AudioClip con la ruta al archivo WAV, o None si no hay audio
            suficiente en el buffer.
        """
        start = fall_ts - pre
        end = fall_ts + post
        target_duration = pre + post

        with self._lock:
            # Necesitamos al menos `pre` segundos antes de la caída.
            if not self._buf or (self._buf[0].ts > start + 0.5):
                return None

            frames = list(self._buf)
            # No limpiamos todo el buffer: conservamos los frames post-caida
            # para futuras alertas. Se consumen solo los que se usan.
            used = []
            for f in frames:
                if f.ts < start - 0.2:
                    continue
                if f.ts > end:
                    break
                used.append(f)
            if used:
                first_used_ts = used[0].ts
                while self._buf and self._buf[0].ts < first_used_ts:
                    self._buf.popleft()

        # Concatenar PCM de los frames seleccionados.
        collected = bytearray()
        for f in used:
            collected.extend(f.data)

        min_bytes = self._samples_per_sec * self._bytes_per_sample * \
            min(target_duration, 0.3)
        if len(collected) < min_bytes:
            return None

        dia = time.strftime("%Y-%m-%d")
        out_dir = ALERT_DIR / dia
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"audio_{alert_id}.wav"

        # Escribir como PCM raw (los headers WAV se descartan en push_wav).
        try:
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(self._channels)
                wf.setsampwidth(self._bytes_per_sample)
                wf.setframerate(self._samples_per_sec)
                wf.writeframes(bytes(collected))
        except Exception:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

        return AudioClip(
            alert_id=alert_id,
            path=str(out_path),
            start_ts=start,
            end_ts=end,
            duration=target_duration,
        )

    def clear(self) -> None:
        """Vacía el buffer (se llama en reset)."""
        with self._lock:
            self._buf.clear()


class AudioEvidenceManager:
    """Coordina la extracción de audio tanto en modo video como en vivo.

    Se instancia una vez en el servidor y se pasa a cada flujo que pueda
    generar alertas. El `RiskEngine` no necesita conocer este gestor: el
    servidor llama a sus métodos cuando dispara una alerta.
    """

    def __init__(self) -> None:
        self._video_path: str | None = None
        self._fall_ts_map: dict[str, float] = {}
        self._live_buffer = LiveAudioBuffer(keep_seconds=10.0)
        self._lock = threading.Lock()

    def set_video_source(self, video_path: str | None) -> None:
        """Registra la ruta del video actual (modo /api/video)."""
        with self._lock:
            self._video_path = video_path

    def clear_video_source(self) -> None:
        with self._lock:
            self._video_path = None

    def feed_live_audio(self, wav_bytes: bytes) -> None:
        """Recibe un paquete de audio del navegador (modo /api/frame)."""
        self._live_buffer.push_wav(wav_bytes)

    def register_fall_timestamp(self, alert_id: str, ts: float) -> None:
        """Asocia un timestamp de caída a un ID de alerta.

        Se llama desde el servidor cuando un frame produce una alerta.
        El audio se extrae después en `extract_for_alert`.
        """
        with self._lock:
            self._fall_ts_map[alert_id] = ts

    def extract_for_alert(self, alert) -> AudioClip | None:
        """Extrae el clip de audio para una alerta recién emitida.

        Args:
            alert: Objeto `Alert` de risk_engine.py (tiene `.alert_id`, `._ts`).

        Returns:
            AudioClip si se extrajo audio, None en caso contrario.
        """
        alert_id = getattr(alert, "alert_id", None)
        ts = getattr(alert, "_ts", None)
        if alert_id is None or ts is None:
            return None

        with self._lock:
            fall_ts = self._fall_ts_map.get(alert_id, ts)

        # Modo video: extraer del archivo original.
        with self._lock:
            video_path = self._video_path

        if video_path:
            clip = extract_video_audio(
                video_path, fall_ts, alert_id,
                pre=AUDIO_PRE_FALL, post=AUDIO_POST_FALL,
            )
            if clip:
                alert.audio_evidence = clip.path
                return clip

        # Modo vivo: extraer del ring buffer.
        clip = self._live_buffer.extract_clip(
            fall_ts, alert_id,
            pre=AUDIO_PRE_FALL, post=AUDIO_POST_FALL,
        )
        if clip:
            alert.audio_evidence = clip.path
            return clip

        return None

    def reset(self) -> None:
        """Limpia el estado (se llama al resetear el motor)."""
        with self._lock:
            self._fall_ts_map.clear()
        self._live_buffer.clear()


# Instancia global del gestor de audio.
audio_manager = AudioEvidenceManager()
