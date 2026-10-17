"""Los 17 puntos COCO que devuelve YOLOv8-Pose y los huesos que los unen."""
from __future__ import annotations

NAMES = [
    "nariz", "ojo_izq", "ojo_der", "oreja_izq", "oreja_der",
    "hombro_izq", "hombro_der", "codo_izq", "codo_der", "muneca_izq",
    "muneca_der", "cadera_izq", "cadera_der", "rodilla_izq", "rodilla_der",
    "tobillo_izq", "tobillo_der",
]
N_KP = len(NAMES)

HEAD = [0, 1, 2, 3, 4]
SHOULDERS = [5, 6]
HIPS = [11, 12]
KNEES = [13, 14]
ANKLES = [15, 16]
UPPER = HEAD + SHOULDERS + [7, 8, 9, 10]
LOWER = HIPS + KNEES + ANKLES

# Huesos (índices 0-based), agrupados para darles color.
BONES = {
    "cabeza": [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
    "brazos": [(5, 7), (7, 9), (6, 8), (8, 10)],
    "tronco": [(5, 6), (5, 11), (6, 12), (11, 12)],
    "piernas": [(11, 13), (13, 15), (12, 14), (14, 16)],
}
BONE_COLOR_BGR = {
    "cabeza": (255, 200, 80), "brazos": (80, 220, 255),
    "tronco": (255, 255, 255), "piernas": (120, 255, 120),
}

# Segmentos para el centro de masa (fracción de masa corporal aproximada,
# tablas antropométricas de Dempster agrupadas a lo que ve la cámara).
COM_SEGMENTS = [
    ("cabeza", HEAD, 0.08),
    ("tronco", SHOULDERS + HIPS, 0.50),
    ("brazos", [5, 6, 7, 8, 9, 10], 0.10),
    ("muslos", HIPS + KNEES, 0.20),
    ("piernas", KNEES + ANKLES, 0.12),
]
