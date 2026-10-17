"""Análisis de postura con MediaPipe Pose (RQF02 2-3-4, RQF03).

Las categorías "no usa el pasamanos" y "distracción" no son objetos que se
puedan encerrar en una caja: son condiciones de sujeción y de postura de la
cabeza. `Caso.md` prescribe MediaPipe Pose para ellas, y aquí se deducen de
los keypoints. También se clasifica la postura global (erguida, pre-caída,
caída) por geometría del eje tobillo-cadera-cabeza.

MediaPipe corre en CPU (XNNPACK) y su coste es casi fijo (~20 ms) sea cual
sea la resolución, así que la palanca de rendimiento es la cadencia con la que
se llama, no el tamaño de entrada (ver `engine.pose_every`).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Índices de MediaPipe Pose
NOSE = 0
LS, RS = 11, 12      # hombros
LW, RW = 15, 16      # muñecas
LH, RH = 23, 24      # caderas
LK, RK = 25, 26      # rodillas
LA, RA = 27, 28      # tobillos
# Orden en el que se guardan los 11 keypoints útiles en `landmarks`
KEYPOINTS = (NOSE, LS, RS, LW, RW, LH, RH, LK, RK, LA, RA)

# --------------------------------------------------------------------------- #
# Umbrales de postura por geometría
# --------------------------------------------------------------------------- #
# Medidos sobre una prueba real en escalera:
#
#   fase                inclinación   cabeza_sobre_cadera
#   caminando erguido       2.3°            1.02
#   tambaleándose           1.5°            0.75
#   caído en el suelo      51.5°            0.38
#
# `inclinación` es el ángulo del eje tobillo->cadera frente a la vertical;
# `cabeza_sobre_cadera` es la altura de la nariz sobre la cadera normalizada
# por la longitud de ese eje. Se descartó el aspecto ancho/alto del esqueleto:
# con el brazo en la barandilla una persona de pie daba 1.92, como un caído.
GEOM_CAIDO_INCLINA = 25.0
GEOM_CAIDO_CABEZA = 0.55
# 0.75 medido dentro del motor (MediaPipe en modo tracking): 8 % de falsas
# alarmas y 92 % de acierto "de pie", sin perder caídas (93 %).
GEOM_DESEQUILIBRIO_CABEZA = 0.75
# Inestabilidad (pre-caída) por inclinación, medido sobre el test set:
#   >  8° -> 19 % FP | 92 % caídas      > 12° ->  6 % FP | 85 % caídas  <- este
#   > 10° -> 13 % FP | 92 % caídas      > 16° ->  4 % FP | 70 % caídas
GEOM_INESTABLE_INCLINA = 12.0


class PoseAnalyzer:
    """Envuelve MediaPipe Pose en modo tracking (una persona por fotograma)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._pose = None
        if enabled:
            import mediapipe as mp
            # Modo TRACKING: 20 ms frente a 41 ms en modo imagen estática.
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )

    # ------------------------------------------------------------------ #
    def analyse(self, image: np.ndarray) -> list[dict]:
        """Devuelve `[{"landmarks": [...11 (x, y)], "signals": [...]}]` o `[]`.

        La persona se devuelve SIEMPRE que haya pose, aunque no tenga ninguna
        señal: sus keypoints son los que permiten deducir la postura erguida.
        """
        if not self.enabled or self._pose is None:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            # En modo tracking la primera llamada solo inicializa el tracker.
            res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return []

        lm = res.pose_landmarks.landmark          # 33 puntos normalizados
        h, w = image.shape[:2]

        def p(i):
            return lm[i] if i < len(lm) else None

        nose, ls, rs = p(NOSE), p(LS), p(RS)
        lw, rw = p(LW), p(RW)
        lh, rh = p(LH), p(RH)
        lk, rk = p(LK), p(RK)
        if not all(x is not None for x in (nose, ls, rs, lw, rw)):
            return []

        signals = []

        # --- RQF02 (2): no usa el pasamanos ---
        # Si ninguna muñeca sube a la altura de la barandilla y el brazo está
        # separado del tronco, no hay agarre.
        shoulder_y = (ls.y + rs.y) / 2
        wrist_high = min(lw.y, rw.y) < shoulder_y - 0.02
        span = abs(ls.x - rs.x) * w
        arm_out = (abs(lw.x - ls.x) * w) > span * 1.2
        if not wrist_high and arm_out:
            signals.append({
                "tipo": "sin_pasamanos", "conf": 0.55,
                "detalle": "ninguna muñeca a la altura de la barandilla",
            })

        # --- RQF02 (3): distracción (móvil o lectura) ---
        head_tilt = abs(nose.x - (ls.x + rs.x) / 2)
        hand_at_head = (
            math.hypot((lw.x - nose.x) * w, (lw.y - nose.y) * h) < 0.22 * w
            or math.hypot((rw.x - nose.x) * w, (rw.y - nose.y) * h) < 0.22 * w
        )
        if head_tilt > 0.06 and hand_at_head:
            signals.append({
                "tipo": "distraccion", "conf": 0.6,
                "detalle": "cabeza inclinada con mano junto a la cara",
            })
        elif hand_at_head and lw.y < shoulder_y:
            signals.append({
                "tipo": "distraccion", "conf": 0.45,
                "detalle": "mano a la altura de la cabeza",
            })

        # --- RQF02 (4): tambaleo ---
        lean = 0.0
        if lh and rh:
            lean = abs(nose.x - (lh.x + rh.x) / 2)
        knee_asym = abs((lk.y - lh.y) if (lk and lh) else 0) - \
            abs((rk.y - rh.y) if (rk and rh) else 0)
        if lean > 0.09 or abs(knee_asym) > 0.10:
            signals.append({
                "tipo": "tambaleo", "conf": 0.5,
                "detalle": f"inclinación tronco {lean:.2f}, "
                           f"asimetría rodillas {knee_asym:.2f}",
            })

        return [{
            "landmarks": [(lm[i].x, lm[i].y) for i in KEYPOINTS],
            "signals": signals,
        }]


# --------------------------------------------------------------------------- #
def posture_from_geometry(pose: list[dict], tambaleo: bool = False) -> str | None:
    """Deduce la postura de los keypoints. `None` si no hay pose utilizable.

    Índices de `landmarks`: 0 nariz, 1/2 hombros, 3/4 muñecas, 5/6 caderas,
    7/8 rodillas, 9/10 tobillos.
    """
    if not pose or len(pose[0].get("landmarks", [])) < 11:
        return None
    pts = pose[0]["landmarks"]
    hip = ((pts[5][0] + pts[6][0]) / 2, (pts[5][1] + pts[6][1]) / 2)
    ank = ((pts[9][0] + pts[10][0]) / 2, (pts[9][1] + pts[10][1]) / 2)
    largo = math.hypot(hip[0] - ank[0], hip[1] - ank[1])
    if largo <= 0:
        return None
    inclinacion = math.degrees(
        math.atan2(abs(hip[0] - ank[0]), abs(hip[1] - ank[1])))
    cabeza = (hip[1] - pts[0][1]) / largo
    if inclinacion > GEOM_CAIDO_INCLINA or cabeza < GEOM_CAIDO_CABEZA:
        return "persona_caido"
    if (inclinacion > GEOM_INESTABLE_INCLINA
            or cabeza < GEOM_DESEQUILIBRIO_CABEZA
            or tambaleo):
        return "persona_desequilibrio"
    return "persona_erguida"


def best_person_match(landmarks: list | None, people: list[list[int]],
                      frame_shape: tuple) -> int:
    """Índice de la caja de persona a la que pertenece la pose, o -1.

    MediaPipe describe a UNA persona. Con varias cajas se elige la de mayor
    IoU con la caja que envuelve al esqueleto completo (comparar un único
    keypoint fallaba cuando caía cerca de una detección falsa).
    """
    if not landmarks or not people:
        return -1
    h, w = frame_shape[:2]
    sx1, sy1 = min(p[0] for p in landmarks) * w, min(p[1] for p in landmarks) * h
    sx2, sy2 = max(p[0] for p in landmarks) * w, max(p[1] for p in landmarks) * h
    if sx2 - sx1 <= 0 or sy2 - sy1 <= 0:
        return -1
    best, best_iou = -1, 0.05
    for i, (x1, y1, x2, y2) in enumerate(people):
        ix1, iy1 = max(sx1, x1), max(sy1, y1)
        ix2, iy2 = min(sx2, x2), min(sy2, y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            continue
        union = (sx2 - sx1) * (sy2 - sy1) + (x2 - x1) * (y2 - y1) - inter
        v = inter / union
        if v > best_iou:
            best, best_iou = i, v
    return best
