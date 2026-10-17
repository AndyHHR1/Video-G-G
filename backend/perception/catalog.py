"""Catálogo de clases y categorías de riesgo.

Tabla única que enlaza cada salida del sistema con su requisito de `Caso.md`.
La usan el motor (para decidir el riesgo), el dibujo (colores) y la API
(textos de la interfaz). Los colores van en BGR, como espera OpenCV.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Clases del detector propio "Modelo S7 v4" (YOLOv8s, 6 clases)
# --------------------------------------------------------------------------- #
CLASS_INFO: dict[int, dict] = {
    0: {
        "name": "Persona caída",
        "color": (60, 60, 235),        # rojo
        "risk": "ALTO",
        "req": "RQF02 (4) · RQNF05",
        "desc": "Persona tendida en el suelo tras una caída.",
    },
    1: {
        "name": "Postura no ergonómica",
        "color": (60, 170, 245),       # naranja
        "risk": "MEDIO",
        "req": "RQF03",
        "desc": "Postura sentada o agachada; no corresponde a tránsito de pie.",
    },
    2: {
        "name": "Postura erguida",
        "color": (90, 200, 90),        # verde
        "risk": "BAJO",
        "req": "RQF03",
        "desc": "Tránsito seguro: persona de pie o caminando.",
    },
    3: {
        "name": "Escalera",
        "color": (200, 120, 200),      # morado
        "risk": "CONTEXTO",
        "req": "—",
        "desc": "Tramo de escalera detectado en la escena.",
    },
    4: {
        "name": "Persona",
        "color": (220, 220, 220),      # blanco
        "risk": "NEUTRO",
        "req": "RQNF08",
        "desc": "Persona genérica. Es la entrada para el análisis de pose (MediaPipe).",
    },
    5: {
        "name": "Pérdida de equilibrio",
        "color": (40, 190, 255),       # amarillo
        "risk": "ALTO",
        "req": "RQF02 (4)",
        "desc": "Centro de gravedad inestable: pre-caída, antes de caer.",
    },
}

RISK_ORDER = {"ALTO": 3, "MEDIO": 2, "BAJO": 1, "NEUTRO": 0, "CONTEXTO": 0}

RISK_COLOR = {
    "ALTO": (60, 60, 235), "MEDIO": (60, 170, 245),
    "BAJO": (90, 200, 90), "CONTEXTO": (200, 120, 200),
}

# --------------------------------------------------------------------------- #
# Categorías de riesgo de RQF02 con su severidad
# --------------------------------------------------------------------------- #
RISK_CATALOG: dict[str, dict] = {
    "obstaculo_escalon": {
        "n": 1, "riesgo": "ALTO", "ref": "RQF02 (1)",
        "desc": "Objeto u obstáculo abandonado en los escalones.",
    },
    "sin_pasamanos": {
        "n": 2, "riesgo": "MEDIO", "ref": "RQF02 (2)",
        "desc": "Tránsito por la escalera sin sujetarse del pasamanos.",
    },
    "distraccion": {
        "n": 3, "riesgo": "MEDIO", "ref": "RQF02 (3)",
        "desc": "Descenso distraído: uso de teléfono o lectura al caminar.",
    },
    "perdida_equilibrio": {
        "n": 4, "riesgo": "ALTO", "ref": "RQF02 (4)",
        "desc": "Caída activa o pérdida inminente de equilibrio.",
    },
    "caida": {
        "n": 4, "riesgo": "ALTO", "ref": "RQF02 (4) · RQNF05",
        "desc": "Persona caída en el suelo.",
    },
    "postura_no_erguida": {
        "n": 3, "riesgo": "MEDIO", "ref": "RQF03",
        "desc": "Postura no ergonómica: no corresponde a tránsito de pie.",
    },
}

# Señal de pose que no es una categoría de RQF02 pero se reporta con ella.
TAMBALEO_META = {
    "n": 4, "riesgo": "ALTO", "ref": "RQF02 (4)",
    "desc": "Tambaleo: paso inestable o pérdida de equilibrio.",
}

# Clases del detector que implican riesgo.
FALL_CLASSES = {"persona_caido": "caida",
                "persona_desequilibrio": "perdida_equilibrio"}
NONERGO_CLASSES = {"persona_sentado": "postura_no_erguida"}
SAFE_CLASSES = {"persona_erguida"}

# --------------------------------------------------------------------------- #
# Objetos COCO relevantes en la escalera (RQF02 1)
# --------------------------------------------------------------------------- #
# Obstáculo real sobre el escalón -> riesgo ALTO.
OBSTACLE_RISK = {
    "backpack": "mochila", "handbag": "bolso", "suitcase": "maleta",
    "bottle": "botella", "book": "libro", "umbrella": "paraguas",
}
# El móvil no es un obstáculo: es evidencia de distracción (RQF02 3) cuando
# aparece junto a la cabeza de una persona.
PHONE_CLASSES = {"cell phone": "telefono"}
# Mobiliario: forma parte de la escena, se dibuja como CONTEXTO.
OBSTACLE_CONTEXT = {
    "chair": "silla", "couch": "sofá", "bench": "banco",
    "potted plant": "planta",
}
OBSTACLE_CLASSES = {**OBSTACLE_RISK, **OBSTACLE_CONTEXT, **PHONE_CLASSES}
