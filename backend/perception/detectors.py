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


def _load(weights: str, device, imgsz: int, dynamic: bool = False,
          int8: bool = False):
    """YOLO en `device`; en CPU, la versión OpenVINO si está disponible
    (ver accel.py). Devuelve (modelo, acelerado)."""
    from ultralytics import YOLO
    from .accel import cpu_weights
    path = cpu_weights(weights, device, imgsz, dynamic=dynamic, int8=int8)
    model = YOLO(path)
    if path.endswith(".pt"):
        model.to(device)                 # los exportados no admiten .to()
    return model, not path.endswith(".pt")


def _names(model) -> dict[int, str]:
    raw = model.names
    return dict(enumerate(raw)) if isinstance(raw, list) else \
        {int(k): v for k, v in raw.items()}


class CocoDetector:
    """Personas y objetos de escena con YOLOv8 COCO (RQF02 1)."""

    def __init__(self, weights: str, device, obstacle_conf: float = 0.25,
                 person_conf: float = 0.20, person_imgsz: int = 512):
        self.device = device
        from . import accel as _accel
        # en CPU: INT8 (medido: casi 2x más rápido, persona igual de precisa)
        self.model, accel = _load(weights, device, person_imgsz,
                                  int8=_accel.INT8_COCO)
        # modelo OpenVINO de tamaño fijo: los objetos también a person_imgsz
        self.obstacle_imgsz = person_imgsz if accel else None
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
        extra = {"imgsz": self.obstacle_imgsz} if self.obstacle_imgsz else {}
        res = self.model.predict(frame, conf=self.obstacle_conf, iou=0.5,
                                 device=self.device, verbose=False, **extra)[0]
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
                # Un objeto en la escalera es un riesgo a vigilar, no una
                # emergencia como una caída: MEDIO (naranja), nunca ALTO.
                "riesgo": "MEDIO" if es_riesgo else "CONTEXTO",
                "conf": conf, "bbox": bbox,
            })
        return out


class PostureDetector:
    """Modelo propio v5: escalera (detector) + postura (clasificador).

    Son dos redes pequeñas en lugar de un detector de 6 clases:

    * `stairs()` usa un detector YOLO de 1 clase (`escalera`) sobre el
      fotograma completo, a 512 px (las escaleras ocupan buena parte del
      encuadre; más resolución no aporta y cuesta más).
    * `classify()` usa un clasificador YOLO-cls sobre el recorte de cada
      persona que ya localizó COCO: ~2 ms por persona, frente a pasar el
      recorte por un detector completo a 640 px como en v4.

    Los ids de clase se mantienen fijos (los usan el catálogo, el dibujo y
    la API): 0 persona_caido · 1 persona_sentado · 2 persona_erguida ·
    3 escalera · 4 persona (sin clasificar).
    """

    NAMES = {0: "persona_caido", 1: "persona_sentado", 2: "persona_erguida",
             3: "escalera", 4: "persona"}
    STAIRS_CONF = 0.35
    STAIRS_IMGSZ = 512
    POSTURE_IMGSZ = 224
    MIN_POSTURE_CONF = 0.40   # por debajo, la postura se da como no decidida

    def __init__(self, stairs_weights: str, posture_weights: str, device):
        self.device = device
        # detector de escalera y clasificador de postura (OpenVINO en CPU)
        self.model, _ = _load(stairs_weights, device, self.STAIRS_IMGSZ)
        # lote variable: classify_many() pasa todas las personas a la vez
        self.cls, _ = _load(posture_weights, device, self.POSTURE_IMGSZ, dynamic=True)
        stairs_names = set(_names(self.model).values())
        if "escalera" not in stairs_names:
            raise ValueError(f"{stairs_weights}: no tiene la clase 'escalera' "
                             f"({sorted(stairs_names)})")
        self.cls_names = _names(self.cls)
        unknown = set(self.cls_names.values()) - set(self.NAMES.values())
        if unknown:
            raise ValueError(f"{posture_weights}: clases desconocidas {unknown}")
        self.names = dict(self.NAMES)
        self.id_of = {v: k for k, v in self.names.items()}

    def stairs(self, frame: np.ndarray, max_boxes: int = 2) -> list[tuple]:
        """`[(conf, bbox), ...]` de la clase `escalera`, las mejores primero.

        Se limita el número de cajas: en una foto real el modelo devolvía seis
        cajas solapadas que cubrían todo el encuadre.
        """
        res = self.model.predict(frame, conf=self.STAIRS_CONF, iou=0.6,
                                 imgsz=self.STAIRS_IMGSZ,
                                 device=self.device, verbose=False)[0]
        found = []
        if res.boxes is not None:
            sn = _names(self.model)
            for b in res.boxes:
                if sn.get(int(b.cls[0])) == "escalera":
                    found.append((float(b.conf[0]),
                                  [int(v) for v in b.xyxy[0].tolist()]))
        found.sort(reverse=True)
        return found[:max_boxes]

    @staticmethod
    def _crop(frame: np.ndarray, box: list[int], pad: float = 0.08):
        """Recorte con un 8 % de margen por lado: el MISMO que se usó para
        generar los recortes de entrenamiento (training/prepare_v5.py)."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        ax1 = int(max(0, x1 - bw * pad)); ax2 = int(min(W, x2 + bw * pad))
        ay1 = int(max(0, y1 - bh * pad)); ay2 = int(min(H, y2 + bh * pad))
        if ax2 - ax1 < 16 or ay2 - ay1 < 16:
            return None
        return frame[ay1:ay2, ax1:ax2]

    def classify(self, frame: np.ndarray, box: list[int]) -> tuple[str, float]:
        """Postura de una persona: clase y probabilidad del clasificador."""
        return self.classify_many(frame, [box])[0]

    def classify_many(self, frame: np.ndarray,
                      boxes: list[list[int]]) -> list[tuple[str, float]]:
        """Postura de varias personas en UNA sola pasada (lote) del clasificador.

        Devuelve `("", prob)` cuando la clase ganadora no llega a
        `MIN_POSTURE_CONF`: mejor «sin clasificar» que una postura dudosa.
        """
        out: list[tuple[str, float]] = [("", 0.0)] * len(boxes)
        crops, idx = [], []
        for i, box in enumerate(boxes):
            c = self._crop(frame, box)
            if c is not None:
                crops.append(c); idx.append(i)
        if not crops:
            return out
        res = self.cls.predict(crops, imgsz=self.POSTURE_IMGSZ,
                               device=self.device, verbose=False)
        for i, r in zip(idx, res):
            p = r.probs
            if p is None:
                continue
            name, conf = self.cls_names.get(int(p.top1), ""), float(p.top1conf)
            out[i] = (name, conf) if conf >= self.MIN_POSTURE_CONF else ("", conf)
        return out
