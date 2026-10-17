"""Anonimización de rostros (RQNF14 · Ley N.º 29733 de Protección de Datos
Personales del Perú).

Toda imagen que el sistema ESCRIBE en disco (evidencia de alertas) pasa antes
por aquí. Dos capas, porque ante la duda se prefiere anonimizar de más:

1. **Rostros detectados** con MediaPipe Face Detection en sus dos modelos:
   corto alcance (0-2 m) y largo alcance (hasta ~5 m, el caso de una cámara
   de seguridad sobre la escalera). Medido: con solo el de corto alcance, las
   caras de una escena típica no se detectaban y la evidencia salía sin
   anonimizar.
2. **Respaldo por persona**: para cada persona detectada cuya cara no se haya
   encontrado (de espaldas, de perfil, lejos), se pixela la zona de la cabeza
   (parte superior de su caja).

El pixelado usa bloques de 1/12 del tamaño de la región: la persona no es
identificable, pero la escena (postura, caída, obstáculo) sigue sirviendo
como evidencia.
"""
from __future__ import annotations

import cv2
import numpy as np

HEAD_FRACTION = 0.22      # parte superior de la caja de persona = cabeza


class FaceAnonymizer:
    def __init__(self, enabled: bool = True, min_confidence: float = 0.4,
                 block: int = 12, margin: float = 0.08):
        self.enabled = enabled
        self.min_confidence = min_confidence
        self.block = block
        self.margin = margin
        self._detectors = None         # carga perezosa: solo si hay alertas

    def _get_detectors(self):
        if self._detectors is None:
            import mediapipe as mp
            fd = mp.solutions.face_detection
            self._detectors = [
                fd.FaceDetection(model_selection=sel,
                                 min_detection_confidence=self.min_confidence)
                for sel in (0, 1)]
        return self._detectors

    def _pixelate(self, frame: np.ndarray, x1, y1, x2, y2) -> None:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        roi = frame[y1:y2, x1:x2]
        small = cv2.resize(roi, (max(1, roi.shape[1] // self.block),
                                 max(1, roi.shape[0] // self.block)),
                           interpolation=cv2.INTER_AREA)
        frame[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)

    def faces(self, frame: np.ndarray, offset=(0, 0)
              ) -> list[tuple[int, int, int, int]]:
        """Cajas de rostro (con margen) en píxeles de la imagen completa."""
        h, w = frame.shape[:2]
        if h < 8 or w < 8:
            return []
        ox, oy = offset
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = []
        for det in self._get_detectors():
            try:
                res = det.process(rgb)
            except Exception:                               # noqa: BLE001
                continue
            for d in res.detections or []:
                bb = d.location_data.relative_bounding_box
                m = self.margin
                out.append((ox + int((bb.xmin - m) * w), oy + int((bb.ymin - m) * h),
                            ox + int((bb.xmin + bb.width + m) * w),
                            oy + int((bb.ymin + bb.height + m) * h)))
        return out

    def __call__(self, frame: np.ndarray,
                 people: list[list[int]] | None = None) -> np.ndarray:
        """Pixela los rostros in place.

        Además de buscar caras en la imagen completa, se buscan en la parte
        superior de cada persona detectada: en el recorte la cara ocupa más
        píxeles y el detector la encuentra aunque esté lejos o de perfil. Si
        aun así no aparece, se pixela la zona de la cabeza de su caja.
        """
        if not self.enabled:
            return frame
        original = frame.copy()          # detectar siempre sobre la imagen limpia
        faces = self.faces(original)
        for x1, y1, x2, y2 in people or []:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            top = original[y1:y1 + int((y2 - y1) * 0.6), x1:int(x2)]
            found = self.faces(top, offset=(x1, y1))
            faces += found
            if not found:
                self._pixelate(frame, x1, y1, x2, y1 + (y2 - y1) * HEAD_FRACTION)
        for box in faces:
            self._pixelate(frame, *box)
        return frame
