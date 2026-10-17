"""Ciclo de vida: parada ordenada de todo el sistema.

`Lifecycle.shutdown()` es la función interna para PARAR el proyecto. La usan:

* el botón «Detener sistema» del panel   -> POST /api/system/stop
* la línea de comandos                   -> python -m backend --stop
* Ctrl+C / SIGTERM en la consola del servidor

Orden de parada: primero se deja de aceptar trabajo (las rutas en vivo y el
stream MJPEG lo comprueban con `stopping`), después se cierran los módulos
aparte (audio, imagen), la mejora en vivo y el hilo de inferencia (que suelta
los modelos de la GPU) y, por último, el servidor HTTP. Cada paso está aislado:
si uno falla, los demás se ejecutan igual.
"""
from __future__ import annotations

import threading
import time
from typing import Callable


class Lifecycle:
    def __init__(self) -> None:
        self._hooks: list[tuple[str, Callable[[], None]]] = []
        self._stopping = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._server_stop: Callable[[], None] | None = None
        self.reason: str = ""
        self.started_at = time.time()

    # ------------------------------------------------------------------ #
    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    @property
    def stopped(self) -> bool:
        return self._done.is_set()

    def on_shutdown(self, name: str, fn: Callable[[], None]) -> None:
        """Registra un paso de parada. Se ejecutan en orden inverso al de
        registro (lo último que arrancó es lo primero que se cierra)."""
        with self._lock:
            self._hooks.append((name, fn))

    def bind_server(self, stop: Callable[[], None]) -> None:
        """El servidor HTTP registra cómo pararse (werkzeug `shutdown`)."""
        self._server_stop = stop

    # ------------------------------------------------------------------ #
    def shutdown(self, reason: str = "solicitado", *, background: bool = False
                 ) -> bool:
        """Detiene el sistema. Devuelve False si ya se estaba deteniendo.

        Con `background=True` vuelve enseguida (lo usa la ruta HTTP, que
        tiene que responder antes de que el servidor se apague)."""
        with self._lock:
            if self._stopping.is_set():
                return False
            self._stopping.set()
            self.reason = reason
        if background:
            threading.Thread(target=self._run, name="shutdown",
                             daemon=True).start()
        else:
            self._run()
        return True

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def _run(self) -> None:
        # Un instante para que la respuesta HTTP que pidió la parada salga.
        time.sleep(0.3)
        print(f"\n  deteniendo el sistema ({self.reason})...", flush=True)
        for name, fn in reversed(self._hooks):
            t0 = time.perf_counter()
            try:
                fn()
                print(f"    - {name}: detenido "
                      f"({(time.perf_counter() - t0) * 1000:.0f} ms)", flush=True)
            except Exception as exc:                        # noqa: BLE001
                print(f"    - {name}: error al detener ({exc})", flush=True)
        self._done.set()
        if self._server_stop is not None:
            try:
                self._server_stop()
            except Exception as exc:                        # noqa: BLE001
                print(f"    - servidor HTTP: {exc}", flush=True)
        print("  sistema detenido.", flush=True)
