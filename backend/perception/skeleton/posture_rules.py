"""Seguro de postura por reglas del esqueleto, independiente del modelo.

El clasificador de postura (Modelo v5) puede equivocarse. Estas reglas miran
solo la GEOMETRÍA de los 17 puntos de cada persona (YOLOv8-Pose) y sirven de
seguro en los dos sentidos (ver `RiskEngine._link_posture`):

* si el esqueleto ve a alguien **caído** y el modelo no, gana el esqueleto;
* si el modelo dice «caído» pero el esqueleto ve a la persona **claramente de
  pie**, la caída del modelo baja a «REVISAR» y no escala a alerta.

Todas las alturas se miden a lo largo de la vertical local (con el *nadir*
de la vista picada si está configurado) y en **torsos** de esa persona, así no
dependen de lo lejos que esté de la cámara. Referencias de una persona de pie
(torso = hombros-caderas ≈ 0.5 m): cabeza ≈ 1.4 torsos sobre la cadera,
cadera ≈ 1.65 sobre los tobillos y 0.8 sobre las rodillas, muñecas ≈ 1.45
sobre los tobillos.

Caído (umbrales en `configs/app.yaml → skeleton.postura`):

| criterio          | se activa si                                                    |
|-------------------|-----------------------------------------------------------------|
| manos_a_pies      | muñecas y tobillos a la misma altura (≤ `hands_feet_torso`)     |
| cabeza_a_cadera   | la cabeza no sube más de `head_hip_torso` sobre la cadera       |
| cuerpo_tendido    | los puntos ocupan más a lo ancho que a lo alto (≥ `lying_ratio`) |
| eje_horizontal    | el eje hombros→tobillos se inclina ≥ `axis_deg` de la vertical   |

Hace falta ≥ 3 criterios, o 2 si uno de ellos es de cuerpo tendido
(`cuerpo_tendido` / `eje_horizontal`): agacharse a recoger algo junta las
manos con los pies y baja la cabeza a la cadera, pero no tiende el cuerpo.

Postura incorrecta (sentado / agachado), ≥ `min_votes` de:

| criterio           | se activa si                                                   |
|--------------------|----------------------------------------------------------------|
| cadera_a_rodillas  | la cadera baja a la altura de las rodillas (≤ `hip_knee_torso`)|
| rodillas_dobladas  | AMBAS rodillas cerradas (cadera-rodilla-tobillo ≤ `knee_angle_deg`) |
| cuerpo_compacto    | cadera y tobillos juntos en vertical (≤ `compact_torso`)       |

Subir un escalón dobla UNA rodilla, por eso se exigen las dos. Un criterio
que no se puede evaluar (tobillos tapados por el pasamanos) no vota. El
estado debe mantenerse `persist_*_s` segundos para confirmarse.
"""
from __future__ import annotations

import numpy as np

from .keypoints import ANKLES, HEAD, HIPS, KNEES, N_KP, SHOULDERS
from .kinematics import angle_deg, mean_point, up_vector

WRISTS = [9, 10]
LEGS = ((11, 13, 15), (12, 14, 16))          # cadera, rodilla, tobillo
LYING = {"cuerpo_tendido", "eje_horizontal"}


class PostureRules:
    def __init__(self, cfg):
        self.cfg = cfg
        self._since: dict[int, tuple[str, float]] = {}

    def forget(self, tid: int) -> None:
        self._since.pop(tid, None)

    def evaluate(self, tid, x: np.ndarray, valid: np.ndarray,
                 torso: float | None, nadir, t: float) -> dict:
        """`x`: 17 puntos normalizados por la altura de la imagen (isótropos);
        `valid`: puntos utilizables; `torso`: largo del torso en esas unidades."""
        c = self.cfg
        empty = {"estado": "indeterminado", "pendiente": None, "criterios": [],
                 "criterios_caida": [], "criterios_postura": [], "evaluables": 0,
                 "sostenido_s": 0.0}
        hp = mean_point(x, valid, HIPS)
        sh = mean_point(x, valid, SHOULDERS)
        anchor = hp if hp is not None else sh
        if not torso or anchor is None:
            self._since.pop(tid, None)
            return empty

        up = up_vector(anchor, nadir)
        lat = np.array([-up[1], up[0]])

        def h(p) -> float:                     # altura en torsos
            return float(np.dot(p, up)) / torso

        hd = mean_point(x, valid, HEAD)
        kn = mean_point(x, valid, KNEES)
        an = mean_point(x, valid, ANKLES)
        wr = mean_point(x, valid, WRISTS)
        evaluable = 0

        # ---- caído -------------------------------------------------------
        fall: list[str] = []
        if wr is not None and an is not None:
            evaluable += 1
            if abs(h(wr) - h(an)) <= c.hands_feet_torso:
                fall.append("manos_a_pies")
        if hd is not None and hp is not None:
            evaluable += 1
            if h(hd) - h(hp) <= c.head_hip_torso:
                fall.append("cabeza_a_cadera")
        vis = [i for i in range(N_KP) if valid[i]]
        if len(vis) >= c.min_points:
            evaluable += 1
            pts = x[vis]
            ext_v = float(np.ptp(pts @ up))
            ext_h = float(np.ptp(pts @ lat))
            if ext_v > 1e-6 and ext_h / ext_v >= c.lying_ratio:
                fall.append("cuerpo_tendido")
        bottom = an if an is not None else kn
        if sh is not None and bottom is not None:
            evaluable += 1
            if angle_deg(sh - bottom, up) >= c.axis_deg:
                fall.append("eje_horizontal")
        is_fall = len(fall) >= 3 or (len(fall) >= 2 and bool(LYING & set(fall)))

        # ---- postura incorrecta (sentado / agachado) ----------------------
        bad: list[str] = []
        if hp is not None and kn is not None:
            evaluable += 1
            if h(hp) - h(kn) <= c.hip_knee_torso:
                bad.append("cadera_a_rodillas")
        angles = [angle_deg(x[a] - x[b], x[k] - x[b])
                  for a, b, k in LEGS if valid[a] and valid[b] and valid[k]]
        if angles:
            evaluable += 1
            if max(angles) <= c.knee_angle_deg:
                bad.append("rodillas_dobladas")
        if hp is not None and an is not None:
            evaluable += 1
            if h(hp) - h(an) <= c.compact_torso:
                bad.append("cuerpo_compacto")
        is_bad = len(bad) >= c.min_votes

        # ---- de pie con claridad (para vetar un «caído» del modelo) -------
        upright = (not fall and not bad
                   and all(p is not None for p in (hd, hp, kn, an))
                   and h(hd) - h(hp) >= c.upright_head_torso
                   and h(hp) - h(an) >= c.upright_hip_ankle_torso)

        raw = ("caido" if is_fall else "postura_incorrecta" if is_bad
               else "erguido" if upright else "indeterminado")
        prev = self._since.get(tid)
        if prev is None or prev[0] != raw:
            self._since[tid] = (raw, t)
        held = t - self._since[tid][1]
        need = {"caido": c.persist_caido_s,
                "postura_incorrecta": c.persist_postura_s}.get(raw, 0.0)
        confirmed = held >= need
        return {"estado": raw if confirmed else "indeterminado",
                "pendiente": None if confirmed else raw,
                "criterios": fall if raw == "caido" else bad if raw == "postura_incorrecta" else [],
                "criterios_caida": fall, "criterios_postura": bad,
                "evaluables": evaluable, "sostenido_s": round(held, 2)}
