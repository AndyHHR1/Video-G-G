"""Mejora de calidad del vídeo en vivo (Real-ESRGAN x4) en su propio hilo.

La superresolución en vivo NO pasa por el hilo de detección: así la detección
mantiene su ritmo y la mejora va en paralelo. Cada fotograma que llega con
"SR en vivo" activado se deposita en una ranura de "solo el último"; el hilo
lo amplía con `realesr-general-x4v3` (~22 ms en una RTX 5070 con FP16; en
una GTX 1650 el perfil `baja_vram` usa FP32 y entrada de 320 px), lo
reduce al ancho del panel y guarda el JPEG en base64 asociado a su `frame_id`.

La API devuelve la versión mejorada del MISMO fotograma que analizó el
detector cuando la tiene en caché, y si no la más reciente: así las cajas
quedan alineadas con la imagen.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from ..config import Settings
from ..rendering.draw import encode_b64


class LiveEnhancer:
    CACHE = 16

    def __init__(self, settings: Settings, device):
        self.cfg = settings.enhancement.live
        self._settings = settings
        self._device = device
        self._sr = None
        self.error: str | None = None
        self.ms: float = 0.0
        self._cond = threading.Condition()
        self._slot: tuple | None = None
        self._cache: OrderedDict = OrderedDict()
        self._latest: tuple | None = None
        self._stop = False
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-sr",
                                        daemon=True)
        self._thread.start()
        self._ready.wait()

    @property
    def available(self) -> bool:
        return self._sr is not None

    @property
    def model_name(self) -> str:
        return self.cfg.model

    def submit(self, frame: np.ndarray, fid) -> None:
        if not self.available:
            return
        with self._cond:
            self._slot = (frame, None if fid is None else str(fid))
            self._cond.notify()

    def get(self, fid=None) -> str | None:
        """JPEG base64 mejorado del fotograma `fid`, o el último disponible."""
        with self._cond:
            if fid is not None and str(fid) in self._cache:
                return self._cache[str(fid)]
            return self._latest[1] if self._latest else None

    def clear(self) -> None:
        with self._cond:
            self._cache.clear()
            self._latest = None
            self._slot = None

    def stop(self, timeout: float = 5.0) -> None:
        """Detiene el hilo y libera la red de la GPU."""
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout)
        self.clear()
        self._sr = None

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        import time
        try:
            from ..enhancement.super_resolution import SuperResolver
            self._sr = SuperResolver(self.cfg.model, self._settings.sr_dir,
                                     self._device,
                                     half=self._settings.enhancement.half)
        except Exception as exc:                            # noqa: BLE001
            self.error = str(exc)
            print(f"  [aviso] SR en vivo desactivado: {exc}", flush=True)
        self._ready.set()
        if self._sr is None:
            return
        while True:
            with self._cond:
                while self._slot is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                frame, fid = self._slot
                self._slot = None
            try:
                t0 = time.perf_counter()
                img = self._sr.enhance(frame, input_max=self.cfg.input_max,
                                       output_width=self.cfg.output_width)
                b64 = encode_b64(img, self.cfg.jpeg_quality)
                self.ms = round((time.perf_counter() - t0) * 1000, 1)
            except Exception as exc:                        # noqa: BLE001
                self.error = str(exc)
                continue
            with self._cond:
                self._latest = (fid, b64)
                if fid is not None:
                    self._cache[fid] = b64
                    while len(self._cache) > self.CACHE:
                        self._cache.popitem(last=False)
