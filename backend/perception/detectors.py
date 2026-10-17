"""Detectores YOLO: dónde hay gente, qué postura tiene, escalera y objetos.

Reparto de trabajo entre los dos modelos:

* **YOLOv8s COCO** (`models/coco/yolov8s.pt`) localiza personas y objetos.
  Es un detector general: en una foto real de escalera encontraba a la
  persona al 0.92 mientras el modelo propio no pasaba de 0.08, porque la
  clase `escalera` saturaba el encuadre.
* **Modelo S7 v4** (`models/detector/escaleras_s7_v4.pt`) decide la POSTURA
  de cada persona recortada y detecta la escalera.

El modelo propio es el entregable final y no se modifica: aquí solo se usa
para inferencia.
"""
from __future__ import annotations

import cv2
import numpy as np

from .catalog import OBSTACLE_CLASSES, OBSTACLE_RISK, PHONE_CLASSES


def _names(model) -> dict[int, str]:
    raw = model.names
    return dict(enumerate(raw)) if isinstance(raw, list) else \
        {int(k): v for k, v in raw.items()}


class CocoDetector:
    """Personas y objetos de escena con YOLOv8 COCO (RQF02 1)."""

    def __init__(self, weights: str, device, obstacle_conf: float = 0.25,
                 person_conf: float = 0.20, person_imgsz: int = 512):
        from ultralytics import YOLO
        self.device = device
        self.model = YOLO(weights)
        self.model.to(device)
        self.names = _names(self.model)
        self.obstacle_conf = obstacle_conf
        self.person_conf = person_conf
        self.person_imgsz = person_imgsz
        self.ids = {k: v for k, v in self.names.items() if v in OBSTACLE_CLASSES}
        self.risk_ids = {k for k, v in self.ids.items() if v in OBSTACLE_RISK}
        self.phone_ids = {k for k, v in self.ids.items() if v in PHONE_CLASSES}

    def people(self, frame: np.ndarray) -> list[list[int]]:
        """Cajas de persona `[x1, y1, x2, y2]` (mínimo 8 px de lado)."""
        res = self.model.predict(frame, conf=self.person_conf, iou=0.5,
                                 imgsz=self.person_imgsz, device=self.device,
                                 verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            if self.names.get(int(b.cls[0])) != "person":
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if x2 - x1 > 8 and y2 - y1 > 8:
                out.append([x1, y1, x2, y2])
        return out

    def obstacles(self, frame: np.ndarray) -> list[dict]:
        """Objetos sobre la escalera: obstáculo (ALTO), teléfono o mobiliario."""
        res = self.model.predict(frame, conf=self.obstacle_conf, iou=0.5,
                                 device=self.device, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cid = int(b.cls[0])
            if cid not in self.ids:
                continue
            bbox = [int(v) for v in b.xyxy[0].tolist()]
            conf = round(float(b.conf[0]), 3)
            if cid in self.phone_ids:
                out.append({"tipo": "telefono", "objeto": self.names[cid],
                            "riesgo": "MEDIO", "conf": conf, "bbox": bbox})
                continue
            es_riesgo = cid in self.risk_ids
            out.append({
                "tipo": "obstaculo_escalon" if es_riesgo else "elemento_escena",
                "objeto": self.names[cid],
                "riesgo": "ALTO" if es_riesgo else "CONTEXTO",
                "conf": conf, "bbox": bbox,
            })
        return out


class PostureDetector:
    """Modelo propio de 6 clases: postura de la persona y escalera."""

    CROP_CONF = 0.20     # confianza mínima dentro del recorte de una persona
    STAIRS_CONF = 0.30

    def __init__(self, weights: str, device):
        from ultralytics import YOLO
        self.device = device
        self.model = YOLO(weights)
        self.model.to(device)
        self.names = _names(self.model)
        self.id_of = {v: k for k, v in self.names.items()}

    def stairs(self, frame: np.ndarray, max_boxes: int = 2) -> list[tuple]:
        """`[(conf, bbox), ...]` de la clase `escalera`, las mejores primero.

        Se limita el número de cajas: en una foto real el modelo devolvía seis
        cajas solapadas que cubrían todo el encuadre.
        """
        res = self.model.predict(frame, conf=self.STAIRS_CONF, iou=0.6,
                                 device=self.device, verbose=False)[0]
        found = []
        if res.boxes is not None:
            for b in res.boxes:
                if self.names.get(int(b.cls[0])) == "escalera":
                    found.append((float(b.conf[0]),
                                  [int(v) for v in b.xyxy[0].tolist()]))
        found.sort(reverse=True)
        return found[:max_boxes]

    def classify(self, frame: np.ndarray, box: list[int]) -> tuple[str, float]:
        """Postura de una persona recortándola y pasando el recorte al modelo.

        El recorte se ajusta a la caja con un margen del 8 % por lado y se
        reescala a 640x640. No se usa un cuadrado de lado max(ancho, alto):
        con una persona alta el recorte salía más ancho que la imagen y la
        escalera dominaba.
        """
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad = 0.08
        bw, bh = x2 - x1, y2 - y1
        ax1 = int(max(0, x1 - bw * pad)); ax2 = int(min(W, x2 + bw * pad))
        ay1 = int(max(0, y1 - bh * pad)); ay2 = int(min(H, y2 + bh * pad))
        if ax2 - ax1 < 16 or ay2 - ay1 < 16:
            return "", 0.0
        crop = cv2.resize(frame[ay1:ay2, ax1:ax2], (640, 640),
                          interpolation=cv2.INTER_LINEAR)
        res = self.model.predict(crop, conf=self.CROP_CONF, device=self.device,
                                 verbose=False)[0]
        best, bestc = "", 0.0
        if res.boxes is not None:
            for b in res.boxes:
                name = self.names.get(int(b.cls[0]), "")
                if name == "escalera":
                    continue
                c = float(b.conf[0])
                if c > bestc:
                    best, bestc = name, c
        return best, bestc
