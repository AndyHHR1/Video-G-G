"""Filtrado de keypoints: confianza mínima, saltos imposibles, oclusión y
suavizado de trayectoria con el filtro One Euro.

Por qué One Euro (Casiez et al., 2012): un promedio móvil fijo obliga a elegir
entre quitar temblor (mucho suavizado → retraso) o seguir movimientos rápidos
(poco suavizado → temblor). One Euro adapta la frecuencia de corte a la
velocidad: quieto suaviza mucho y en un movimiento brusco (un tropiezo) casi
no retrasa, que es justo lo que necesita la detección de pérdida de equilibrio.

Las coordenadas se filtran normalizadas por la altura del fotograma, así los
parámetros no dependen de la resolución de la cámara.
"""
from __future__ import annotations

import math

import numpy as np

from .keypoints import N_KP


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class KeypointSmoother:
    """Estado de los 17 puntos de UNA persona seguida.

    Para cada punto:
      * confianza < `conf_min`          -> no se usa esta observación;
      * salto > `max_jump` (en torsos)  -> se descarta (cambio izq/der, ruido);
      * sin observación válida          -> se mantiene la última posición
        filtrada durante `hold_s` (el pasamanos tapa piernas o manos un
        instante) y se marca como «estimado»; pasado ese tiempo, el punto deja
        de ser válido;
      * observación válida              -> filtro One Euro.
    """

    def __init__(self, conf_min=0.45, hold_s=0.6, max_jump_torso=1.2,
                 min_cutoff=1.0, beta=1.5, d_cutoff=1.0):
        self.conf_min = conf_min
        self.hold_s = hold_s
        self.max_jump = max_jump_torso
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = np.zeros((N_KP, 2))         # posición filtrada
        self.dx = np.zeros((N_KP, 2))        # velocidad filtrada
        self.last_obs = np.full(N_KP, -np.inf)
        self.has = np.zeros(N_KP, bool)
        self.t_prev: float | None = None

    def update(self, pts: np.ndarray, conf: np.ndarray, t: float,
               torso: float | None = None):
        """`pts` (17, 2) normalizados, `conf` (17,). Devuelve
        `(x, valid, estimated)`: posiciones filtradas, máscara de puntos
        utilizables y máscara de los que se mantienen por oclusión."""
        dt = max(1e-3, t - self.t_prev) if self.t_prev is not None else None
        stale = (t - self.last_obs) > self.hold_s
        ok = conf >= self.conf_min
        if torso and self.max_jump:
            # El salto permitido crece con el tiempo transcurrido.
            jump = np.linalg.norm(pts - self.x, axis=1) / torso
            limit = self.max_jump * (1 + (dt or 0) * 4)
            ok &= ~(self.has & ~stale & (jump > limit))
        fresh = ok & (~self.has | stale)     # primera vez o tras oclusión larga
        self.x[fresh] = pts[fresh]
        self.dx[fresh] = 0.0
        run = ok & ~fresh
        if dt is not None and run.any():
            dx_raw = (pts[run] - self.x[run]) / dt
            a_d = _alpha(self.d_cutoff, dt)
            self.dx[run] = a_d * dx_raw + (1 - a_d) * self.dx[run]
            speed = np.linalg.norm(self.dx[run], axis=1, keepdims=True)
            cutoff = self.min_cutoff + self.beta * speed
            tau = 1.0 / (2.0 * np.pi * cutoff)
            a = 1.0 / (1.0 + tau / dt)
            self.x[run] = a * pts[run] + (1 - a) * self.x[run]
        elif run.any():
            self.x[run] = pts[run]
        self.last_obs[ok] = t
        self.has |= ok
        self.t_prev = t

        held = self.has & ~ok & ((t - self.last_obs) <= self.hold_s)
        valid = ok | held
        return self.x.copy(), valid, held
