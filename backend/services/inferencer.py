"""Hilo trabajador único para toda la inferencia.

Por qué un único hilo persistente y no inferir en el hilo de cada petición:
Flask atiende cada petición en un hilo distinto y, en un hilo recién creado,
CUDA tiene que volver a enlazar su contexto. Medido en este proyecto:

    inferencia     ~29 ms en hilo ya enlazado  ->  ~82 ms en hilo nuevo
    dibujo OpenCV  0.13 ms                      ->  ~23 ms

Los modelos se cargan DENTRO de este hilo para que su contexto CUDA quede
ligado a él, y todas las peticiones encolan su trabajo aquí.

Modo asíncrono (cámara en vivo): la petición encola su fotograma y devuelve
al instante el ÚLTIMO resultado calculado, sin esperar. El navegador mantiene
así su ritmo de 30 fps aunque la inferencia vaya algo más lenta, y cada
respuesta lleva el `frame_id` del fotograma que realmente se analizó para que
el cliente dibuje las cajas sobre ese fotograma.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from ..config import Settings
from ..perception.engine import RiskEngine, resolve_device
from ..rendering.draw import draw_all, detections_only


def sr_enabled(settings: Settings) -> bool:
    """Resuelve `enhancement.enabled` (True / False / 'auto')."""
    flag = settings.enhancement.enabled
    if isinstance(flag, str) and flag.lower() == "auto":
        import torch
        return torch.cuda.is_available() and settings.device != "cpu"
    return bool(flag)


class Inferencer:
    MAX_QUEUE = 3           # en vivo solo importa el fotograma más reciente

    def __init__(self, settings: Settings):
        self.settings = settings
        self.conf = settings.inference.conf
        self.device = None
        self.engine: RiskEngine | None = None
        self.load_error: Exception | None = None
        self._q: queue.Queue = queue.Queue()
        self._last: tuple | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="inferencer",
                                        daemon=True)
        self._thread.start()
        self._ready.wait()
        if self.load_error:
            raise self.load_error

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        s = self.settings
        self.device = resolve_device(s.device)
        self.engine = RiskEngine(s, conf=self.conf)
        # Calentamiento: la primera inferencia paga la inicialización de CUDA
        # y de MediaPipe (varios segundos); mejor aquí que en el primer
        # fotograma del usuario.
        warm = np.full((540, 960, 3), 120, np.uint8)
        warm[130:460, 300:430] = 40
        for force in (True, True, False):
            self.engine.step(warm, force=force, allow_alerts=False)
        self.engine.reset()

    def _run(self) -> None:
        try:
            self._load()
        except Exception as exc:                            # noqa: BLE001
            self.load_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self._q.get()
            if job is None:
                self._release()
                return
            frame, opts, out, _droppable = job
            try:
                if opts.get("reset"):
                    self.engine.reset()
                    out["value"] = None
                else:
                    out["value"] = self._process(frame, **opts)
                    with self._lock:
                        self._last = out["value"]
            except BaseException as exc:                    # noqa: BLE001
                # El hilo sobrevive a cualquier fallo: si muriera, todas las
                # peticiones siguientes quedarían colgadas.
                out["error"] = exc
            finally:
                out["done"].set()

    def _process(self, frame: np.ndarray, conf: float, draw: bool = True,
                 force: bool = False, allow_alerts: bool = True, fid=None):
        if conf != self.conf:
            self.conf = self.engine.conf = conf
        t0 = time.perf_counter()
        self.engine.camera_connected = True
        result = self.engine.step(frame, force=force, allow_alerts=allow_alerts)
        result["frame_id"] = fid
        t_predict = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        if draw:
            annotated = frame.copy()
            dets = draw_all(annotated, result)
        else:
            annotated, dets = frame, detections_only(result)
        # La evidencia de las alertas se guarda YA dibujada, también en vivo
        # (donde el navegador pinta las cajas y aquí no se dibuja nada).
        if self.engine.has_pending_evidence:
            evidence = annotated
            if not draw:
                evidence = frame.copy()
                draw_all(evidence, result)
            self.engine.flush_evidence(evidence)
        t_draw = (time.perf_counter() - t1) * 1000
        return annotated, result, dets, {"predict": round(t_predict, 1),
                                         "draw": round(t_draw, 1)}

    # ------------------------------------------------------------------ #
    # API pública (llamada desde los hilos de Flask)
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray, conf: float, *, draw: bool = True,
               force: bool = False, allow_alerts: bool = True, fid=None,
               wait: bool = True):
        """Encola un fotograma. Con `wait=False` devuelve el último resultado
        disponible (o `None`) sin esperar al de este fotograma."""
        if self.stopped:
            raise RuntimeError("el sistema se está deteniendo")
        self._trim()
        out = {"done": threading.Event()}
        opts = dict(conf=conf, draw=draw, force=force,
                    allow_alerts=allow_alerts, fid=fid)
        self._q.put((frame, opts, out, not wait))
        if not wait:
            with self._lock:
                return self._last
        out["done"].wait()
        if "error" in out:
            raise out["error"]
        return out["value"]

    def _trim(self) -> None:
        """Si el motor no llega al ritmo de la cámara, descarta fotogramas en
        vivo atrasados. Solo se descartan trabajos asíncronos: uno síncrono
        (modo overlay o cámara del servidor) tiene una petición esperando."""
        if self._q.qsize() <= self.MAX_QUEUE:
            return
        pending = []
        while True:
            try:
                pending.append(self._q.get_nowait())
            except queue.Empty:
                break
        for job in pending:
            if job is not None and job[3]:
                job[2]["value"] = None
                job[2]["done"].set()
            else:
                self._q.put(job)

    def reset(self) -> None:
        """Limpia las caches del motor desde su propio hilo."""
        if self.stopped:
            return
        out = {"done": threading.Event()}
        self._q.put((None, {"reset": True}, out, False))
        out["done"].wait()

    def last_result(self) -> tuple | None:
        """Último (fotograma, resultado) analizado en vivo, solo lectura.
        Lo usa el módulo Imagen para su vector de características."""
        with self._lock:
            if not self._last:
                return None
            return self._last[0], self._last[1]

    def clear_last(self) -> None:
        with self._lock:
            self._last = None

    # ------------------------------------------------------------------ #
    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Detiene el hilo trabajador y libera los modelos de la GPU.

        Las peticiones en vivo que estuvieran en cola se descartan; la que se
        esté procesando termina antes de que el hilo salga."""
        if self._stopped.is_set():
            return
        self._stopped.set()
        pending = []
        while True:
            try:
                pending.append(self._q.get_nowait())
            except queue.Empty:
                break
        for job in pending:
            if job is not None:
                job[2]["error"] = RuntimeError("sistema detenido")
                job[2]["done"].set()
        self._q.put(None)
        self._thread.join(timeout)
        with self._lock:
            self._last = None

    def _release(self) -> None:
        """Suelta los modelos desde su propio hilo (contexto CUDA)."""
        self.engine = None
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass
