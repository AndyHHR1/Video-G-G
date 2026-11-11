"""Señales de conducta por persona a partir de su esqueleto (YOLOv8-Pose).

Sustituyen a las de MediaPipe Pose, que describía UNA sola persona por
fotograma y, con varias en escena, emparejaba su pose con la caja equivocada
(medido con peatones de MOT15: 168 de 192 caídas falsas venían de ahí). Aquí
cada persona seguida tiene sus propias señales.

Todas las distancias van en torsos de esa persona (como en kinematics.py), así
que no dependen de lo lejos que esté de la cámara:

| señal          | se activa si                                                        |
|----------------|---------------------------------------------------------------------|
| distraccion    | una muñeca junto a la cara (≤ `hand_face_torso`) con el codo        |
|                | doblado hacia arriba, o un teléfono (COCO) junto a la nariz         |
|                | (≤ `phone_face_torso`)                                              |
| sin_pasamanos  | hay escalera en la escena y NINGUNA mano se separa del cuerpo a la  |
|                | altura de la cintura/pecho (donde está la barandilla)               |

Un punto que no se ve (pasamanos, mochila) no decide: si no hay muñecas
visibles no se emite `sin_pasamanos`.
"""
from __future__ import annotations

import math

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 5, 6, 7, 8, 9, 10, 11, 12


def _pt(kps, i):
    """(x, y) en píxeles si el punto es utilizable (visto o mantenido)."""
    k = kps[i]
    return (k[0], k[1]) if k[3] else None


def _mid(a, b):
    if a and b:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return a or b


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def skeleton_signals(sk: dict, cfg, phones: list[dict] = (),
                     stairs_present: bool = False) -> list[dict]:
    """Señales de una persona. `sk`: esqueleto del SkeletonAnalyzer con `kps`
    en píxeles y `metricas.torso_px`; `phones`: teléfonos COCO (bbox en px)."""
    kps = sk.get("kps") or []
    torso = (sk.get("metricas") or {}).get("torso_px")
    if len(kps) < 17 or not torso:
        return []
    out: list[dict] = []
    nose = _pt(kps, NOSE)
    sh = _mid(_pt(kps, L_SH), _pt(kps, R_SH))
    hip = _mid(_pt(kps, L_HIP), _pt(kps, R_HIP))
    wrists = [w for w in (_pt(kps, L_WR), _pt(kps, R_WR)) if w]

    # --- RQF02 (3): distracción --------------------------------------------
    if nose:
        for wi, ei in ((L_WR, L_EL), (R_WR, R_EL)):
            w, e = _pt(kps, wi), _pt(kps, ei)
            # mano junto a la cara y codo por debajo de la mano (brazo
            # levantado hacia la cara, no colgando cerca por la perspectiva)
            if w and _dist(w, nose) <= cfg.hand_face_torso * torso \
                    and (e is None or e[1] > w[1]):
                out.append({"tipo": "distraccion", "conf": 0.55,
                            "detalle": f"persona {sk.get('id')}: mano junto a la cara"})
                break
        for ph in phones:
            cx = (ph["bbox"][0] + ph["bbox"][2]) / 2
            cy = (ph["bbox"][1] + ph["bbox"][3]) / 2
            d = _dist((cx, cy), nose) / torso
            if d <= cfg.phone_face_torso:
                out = [s for s in out if s["tipo"] != "distraccion"]
                out.append({"tipo": "distraccion",
                            "conf": round(min(0.9, ph.get("conf", 0.5) + 0.3), 2),
                            "detalle": f"persona {sk.get('id')}: teléfono a "
                                       f"{d:.1f} torsos de la cara"})
                break

    # --- RQF02 (2): no usa el pasamanos (solo con escalera en la escena) ----
    if stairs_present and wrists and hip and sh:
        top = sh[1] - cfg.rail_above_shoulder_torso * torso
        bottom = hip[1] + cfg.rail_below_hip_torso * torso
        holding = any(top <= w[1] <= bottom
                      and abs(w[0] - hip[0]) >= cfg.rail_lateral_torso * torso
                      for w in wrists)
        if not holding:
            out.append({"tipo": "sin_pasamanos", "conf": 0.5,
                        "detalle": f"persona {sk.get('id')}: ninguna mano en la "
                                   "barandilla"})
    return out
