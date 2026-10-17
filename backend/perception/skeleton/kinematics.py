"""Medidas biomecánicas por persona a partir del esqueleto suavizado.

Todas las magnitudes se expresan en **torsos** (distancia media
hombros-caderas de esa persona) en vez de píxeles. Con la cámara alta en
diagonal (vista picada) una persona cerca de la cámara mide el doble que una
lejana: en píxeles una misma caída daría velocidades distintas según dónde
ocurra; en torsos, no.

Vista picada y la vertical
--------------------------
Con la cámara inclinada hacia abajo, las verticales del mundo no son
verticales en la imagen: convergen hacia un punto bajo la imagen (el *nadir*).
Si se configura ese punto (`skeleton.perspective.nadir`), la «vertical» de
cada persona es la dirección que se aleja del nadir pasando por su cadera.
Sin él se usa la vertical de la imagen y, además, el ángulo del tronco se
compara con la **línea base propia de cada persona** (su mediana mientras
camina estable), lo que cancela casi toda la inclinación aparente que añade
la perspectiva.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .keypoints import ANKLES, COM_SEGMENTS, HEAD, HIPS, SHOULDERS


def mean_point(x: np.ndarray, valid: np.ndarray, idx) -> np.ndarray | None:
    sel = [i for i in idx if valid[i]]
    if not sel:
        return None
    return x[sel].mean(axis=0)


def up_vector(at: np.ndarray, nadir) -> np.ndarray:
    """Dirección «arriba» del mundo en el punto `at` de la imagen."""
    if nadir is None:
        return np.array([0.0, -1.0])
    v = at - np.asarray(nadir, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, -1.0])


def angle_deg(v: np.ndarray, ref: np.ndarray) -> float:
    c = float(np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def center_of_mass(x: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """Centro de masa aproximado: media ponderada de los segmentos visibles.
    Exige ver el tronco; los segmentos tapados (pasamanos) no cuentan."""
    acc, wsum, has_trunk = np.zeros(2), 0.0, False
    for name, idx, w in COM_SEGMENTS:
        p = mean_point(x, valid, idx)
        if p is None:
            continue
        has_trunk |= name == "tronco"
        acc += w * p
        wsum += w
    return acc / wsum if has_trunk and wsum > 0 else None


class _Deriv:
    """Velocidad y aceleración de un punto con tiempos reales y suavizado
    exponencial (las derivadas amplifican el ruido)."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.p = self.v = self.a = None
        self.t = None

    def update(self, p: np.ndarray | None, t: float):
        if p is None:
            self.p = None                  # hueco: se reinicia sin inventar
            self.v = self.a = None
            return None, None
        if self.p is None or self.t is None or t <= self.t:
            self.p, self.t = p, t
            return None, None
        dt = t - self.t
        v = (p - self.p) / dt
        v = v if self.v is None else self.alpha * v + (1 - self.alpha) * self.v
        a = None
        if self.v is not None:
            a = (v - self.v) / dt
            a = a if self.a is None else self.alpha * a + (1 - self.alpha) * self.a
        self.p, self.t, self.v, self.a = p, t, v, a
        return v, a

    def reset(self):
        self.p = self.v = self.a = self.t = None


class Kinematics:
    """Estado cinemático de una persona seguida."""

    def __init__(self, nadir=None, alpha: float = 0.5, baseline_s: float = 3.0,
                 feet_pad: float = 0.15):
        self.nadir = nadir
        self.feet_pad = feet_pad          # medio pie a cada lado, en torsos
        self.hip = _Deriv(alpha)
        self.head = _Deriv(alpha)
        self.torso_len: float | None = None
        self.baseline: deque = deque()
        self.baseline_s = baseline_s

    def baseline_angle(self, min_samples: int) -> float | None:
        if len(self.baseline) < min_samples:
            return None
        return float(np.median([a for _, a in self.baseline]))

    def add_baseline(self, t: float, angle: float) -> None:
        self.baseline.append((t, angle))
        while self.baseline and t - self.baseline[0][0] > self.baseline_s:
            self.baseline.popleft()

    def update(self, x: np.ndarray, valid: np.ndarray, t: float) -> dict:
        """`x` normalizado por la ALTURA de la imagen en ambos ejes (por eso
        las distancias son isótropas). Devuelve las medidas del fotograma."""
        sh = mean_point(x, valid, SHOULDERS)
        hp = mean_point(x, valid, HIPS)
        hd = mean_point(x, valid, HEAD)
        m: dict = {"tronco_deg": None, "torso": None, "cadera_v": None,
                   "cadera_baja_v": None, "cadera_a": None, "cabeza_v": None,
                   "cabeza_a": None, "com_offset": None, "com": None}

        if sh is not None and hp is not None:
            L = float(np.linalg.norm(sh - hp))
            if L > 1e-4:
                self.torso_len = L if self.torso_len is None \
                    else 0.8 * self.torso_len + 0.2 * L
                up = up_vector(hp, self.nadir)
                m["tronco_deg"] = round(angle_deg(sh - hp, up), 1)
        L = self.torso_len
        m["torso"] = L
        if not L:
            self.hip.update(None, t)
            self.head.update(None, t)
            return m

        anchor = hp if hp is not None else sh
        up = up_vector(anchor, self.nadir)
        down = -up
        lateral = np.array([-up[1], up[0]])

        v, a = self.hip.update(hp, t)
        if v is not None:
            m["cadera_v"] = round(float(np.linalg.norm(v)) / L, 2)
            m["cadera_baja_v"] = round(float(np.dot(v, down)) / L, 2)
        if a is not None:
            m["cadera_a"] = round(float(np.linalg.norm(a)) / L, 2)
        v, a = self.head.update(hd, t)
        if v is not None:
            m["cabeza_v"] = round(float(np.linalg.norm(v)) / L, 2)
        if a is not None:
            m["cabeza_a"] = round(float(np.linalg.norm(a)) / L, 2)

        # Centro de masa frente a la base de apoyo (tobillos), medido en el
        # eje lateral: perpendicular a la vertical local.
        com = center_of_mass(x, valid)
        if com is not None:
            m["com"] = com
            feet = [x[i] for i in ANKLES if valid[i]]
            if feet:
                c = float(np.dot(com, lateral))
                f = [float(np.dot(p, lateral)) for p in feet]
                # La base de apoyo es el pie entero, no solo el tobillo.
                pad = self.feet_pad * L
                lo, hi = min(f) - pad, max(f) + pad
                out = 0.0 if lo <= c <= hi else min(abs(c - lo), abs(c - hi))
                m["com_offset"] = round(out / L, 2)
        return m
