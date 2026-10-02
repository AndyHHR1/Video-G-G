#!/usr/bin/env python3
"""
Motor de riesgo: implementa los requisitos de `Caso.md` que YOLO por si solo
no cubre.

Reparto de responsabilidades:

  YOLOv8  -> RQF02 (caja + clase de la persona y de la escalera)
  aqui    -> RQF02 (1) obstaculos, (2) no uso del pasamanos, (3) distraccion
             RQF03  discriminacion transito seguro / conducta de riesgo
             RQF04  doble umbral 75% pre-filtrado / 85% confirmacion
             RQF05  seguimiento con ByteTrack y tolerancia a oclusion ~1 s
             RQF06  persistencia: la condicion debe durar > 3 s
             RQF07  alerta estructurada con evidencia fotografica
             RQNF14 anonimizacion de rostros
             RQNF16/RQNF23 registro auditable

El texto de `Caso.md` es explicito en este punto: "Media Pipe Pose para
estimacion de postura" es la via para las categorias (2) y (3), porque son
condiciones de SUJECION y de POSTURA DE CABEZA, no objetos que se puedan
delimitar con cajas.

Uso:
    from risk_engine import RiskEngine
    eng = RiskEngine(risk_weights, obstacle_weights)
    out = eng.step(frame_bgr)      # detecciones + riesgos + alertas
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Parametros exigidos por Caso.md
# --------------------------------------------------------------------------- #
CONF_PREFILTER = 0.75      # RQF04: umbral de pre-filtrado
CONF_CONFIRM = 0.85        # RQF04: umbral de confirmacion
# Los umbrales de RQF04 gobiernan la CONFIRMACION de un riesgo de persona, no
# la deteccion cruda de un objeto. Una mochila o una botella sobre un escalon
# no llega a 0.75 de confianza: son objetos pequenos y parcialmente ocultos.
# Con 0.75 el detector de obstaculos no encontraba practicamente nada, que es
# exactamente el fallo reportado.
CONF_OBSTACLE = 0.25
PERSIST_SECONDS = 3.0      # RQF06: persistencia minima
OCCLUSION_SECONDS = 1.0    # RQF05: oclusion tolerada
ALERT_COOLDOWN = 15.0      # evita rafaga de alertas repetidas

# Categorias de riesgo de RQF02, con la severidad que se les asigna.
RISK_CATALOG = {
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

# Clases de YOLO que generan riesgo de caída o pérdida de equilibrio.
FALL_CLASSES = {"persona_caido": "caida", "persona_desequilibrio": "perdida_equilibrio"}
NONERGO_CLASSES = {"persona_sentado": "postura_no_erguida"}
SAFE_CLASSES = {"persona_erguida"}

# Objetos de COCO que, si aparecen sobre la escalera, son obstáculos.
# Un cliente con el movil en la mano NO es un obstaculo; por eso `cell phone`
# queda fuera de esta lista y se trata aparte como distraccion.
# Objetos que constituyen un OBSTACULO real en la escalera -> riesgo ALTO.
OBSTACLE_RISK = {
    "backpack": "mochila", "handbag": "bolso", "suitcase": "maleta",
    "bottle": "botella", "book": "libro", "umbrella": "paraguas",
}

# Telefono movil. NO es un obstaculo: es la evidencia de RQF02 (3)
# "distracciones (uso del telefono o lectura mientras se camina)".
# Se detecta con COCO y se combina despues con la pose: un movil cerca de la
# cabeza de una persona es uso de telefono; un movil suelto en un escalon no.
PHONE_CLASSES = {"cell phone": "telefono"}

# Mobiliario: no es un riesgo por si mismo, pero forma parte de la escena y
# se dibuja como CONTEXTO. Antes se eliminaba del filtro, y eso hacia que
# las sillas y bancos que el detector si encontraba dejaran de aparecer: el
# usuario lo describio como "objetos que antes salian y ahora no".
OBSTACLE_CONTEXT = {
    "chair": "silla", "couch": "sofá", "bench": "banco",
    "potted plant": "planta",
}

# Union de ambas. NOTA: `box`, `traffic cone` y `cart` NO existen en el
# voculario de 80 clases de Ultralytics; se habian escrito por error y nunca
# pudieron coincidir con nada.
OBSTACLE_CLASSES = {**OBSTACLE_RISK, **OBSTACLE_CONTEXT, **PHONE_CLASSES}

RISK_COLOR = {
    "ALTO": (60, 60, 235), "MEDIO": (60, 170, 245),
    "BAJO": (90, 200, 90), "CONTEXTO": (200, 120, 200),
}


# --------------------------------------------------------------------------- #
# Alerta estructurada (RQF07) con evidencia (RQNF18)
# --------------------------------------------------------------------------- #
@dataclass
class Alert:
    """Estructura de una alerta, tal y como pide RQF07 y RQNF15."""
    alert_id: str
    timestamp: str
    risk_type: str
    risk_ref: str
    severity: str
    description: str
    track_id: int | None
    confidence: float
    bbox: list[float]
    evidence: str            # ruta del fotograma capturado
    duration_s: float
    system_status: str = "OPERATIVO"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
@dataclass
class TrackState:
    """Estado temporal de una identidad seguida (RQF05 / RQF06)."""
    track_id: int
    last_seen: float
    active: bool = True
    risks: deque = field(default_factory=lambda: deque(maxlen=300))
    risk_since: dict = field(default_factory=dict)
    alerts: dict = field(default_factory=dict)


class RiskEngine:
    """Pipeline completo: deteccion -> pose -> contexto -> persistencia -> alerta."""

    # Cuantas cajas de `escalera` se aceptan como mucho. Medido: en una foto
    # real de escalera el modelo devolvia 6 cajas solapadas que cubrian el
    # encuadre entero y no llegaba a considerar a la persona.
    MAX_STAIRS_BOXES = 2

    def __init__(
        self,
        risk_weights: str,
        obstacle_weights: str = "yolov8s.pt",
        device: int = 0,
        conf: float = 0.25,
        enable_pose: bool = True,
        anonymize: bool = True,
        pose_every: int = 2,
        obstacle_every: int = 10,
        stairs_every: int = 6,
        people_every: int = 1,
        person_imgsz: int = 512,
        person_conf: float = 0.30,
    ):
        import mediapipe as mp
        from ultralytics import YOLO

        self.device = device
        self.conf = conf
        self.anonymize = anonymize
        self.pose_every = pose_every
        self.obstacle_every = obstacle_every
        self.stairs_every = stairs_every
        self.people_every = people_every
        self.person_imgsz = person_imgsz
        self.person_conf = person_conf
        self._pose_errors = 0
        self._last_people: list = []
        # Postura deducida en el ultimo frame en el que se evaluo la pose.
        # La pose se calcula a una cadencia mas lenta que el frame (ver
        # `pose_every`): sin esta cache, en los frames intermedios no habria
        # keypoints y la clasificacion caeria a `persona` generica.
        self._last_geom: str | None = None
        self._last_stairs: list = []
        self._last_obstacles: list = []

        # --- RQF02: detector de personas y escalera -----------------------
        self.det = YOLO(risk_weights)
        self.det.to(device)
        # nombres del modelo propio -> id, para saber que clase es cual
        raw = self.det.names
        self.risk_names = dict(enumerate(raw)) if isinstance(raw, list) else \
            {int(k): v for k, v in raw.items()}
        self.id_of = {v: k for k, v in self.risk_names.items()}

        # --- RQF02 (1): obstaculos. Un modelo COCO aparte, porque las clases
        # del modelo propio no incluyen ningun objeto ----------------------
        self.obs = YOLO(obstacle_weights)
        self.obs.to(device)
        obs_raw = self.obs.names
        self.obs_names = dict(enumerate(obs_raw)) if isinstance(obs_raw, list) else \
            {int(k): v for k, v in obs_raw.items()}
        self.obs_ids = {k: v for k, v in self.obs_names.items()
                        if v in OBSTACLE_CLASSES}
        self.obs_risk_ids = {k for k, v in self.obs_ids.items()
                             if v in OBSTACLE_RISK}
        self.obs_phone_ids = {k for k, v in self.obs_ids.items()
                              if v in PHONE_CLASSES}

        # --- RQF02 (2) y (3): postura ------------------------------------
        self.enable_pose = enable_pose
        self.pose = None
        if enable_pose:
            # static_image_mode=True: deteccion por fotograma. Con False el
            # tracker no inicializa en la primera imagen de una secuencia y
            # no devuelve pose; ademas la imagen puede venir de un fichero
            # suelto y no de un video continuo.
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            # indices de MediaPipe Pose
            self.LW, self.RW = 15, 16      # muñecas
            self.LS, self.RS = 11, 12      # hombros
            self.LH, self.RH = 23, 24      # caderas
            self.NOSE = 0
            self.LK, self.RK = 25, 26      # rodillas
            self.LA, self.RA = 27, 28      # tobillos

        # --- estado -------------------------------------------------------
        self.tracks: dict[int, TrackState] = {}
        self.alerts: deque = deque(maxlen=200)
        self.log_path = Path("runs/alerts.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._alert_seq = 0

        # histéresis de ByteTrack tolerando la oclusión
        self._tracker = None
        self._tracker_ready = False

    # ------------------------------------------------------------------ #
    # Seguimiento (RQF05)
    # ------------------------------------------------------------------ #
    def _track(self, detections: list[dict]) -> list[dict]:
        """Asigna un ID estable a cada detección y tolera oclusiones cortas.

        No se usa ByteTrack de Ultralytics porque arrastra estado interno entre
        procesos de forma frágil; la asignación es una versión reducida y
        explícita: coincidencia por solapamiento IoU, tolerancia de oclusión de
        `OCCLUSION_SECONDS` y recuerdo de identidades recentemente perdidas.
        """
        now = time.time()
        # 1) association por IoU con las identities vivas
        pairs = []
        for tid, st in self.tracks.items():
            if now - st.last_seen > OCCLUSION_SECONDS:
                continue
            for det in detections:
                if det["track_id"] is not None:
                    continue
                iou = _iou(getattr(st, "bbox", [0, 0, 0, 0]), det["bbox"])
                if iou > 0.25:
                    pairs.append((iou, tid, det))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_t, used_d = set(), set()
        for _iou_v, tid, det in pairs:
            if tid in used_t or id(det) in used_d:
                continue
            used_t.add(tid)
            used_d.add(id(det))
            det["track_id"] = tid
        # 2) identidades nuevas
        for det in detections:
            if det["track_id"] is None:
                det["track_id"] = max(self.tracks, default=0) + 1
        # 3) actualizar estado y limpiar identidades perdidas
        for det in detections:
            tid = det["track_id"]
            st = self.tracks.get(tid)
            if st is None:
                st = self.tracks[tid] = TrackState(tid, now)
            st.last_seen = now
            st.active = True
            st.bbox = det["bbox"]   # type: ignore[attr-defined]
        for tid, st in list(self.tracks.items()):
            if now - st.last_seen > OCCLUSION_SECONDS:
                del self.tracks[tid]
        return detections

    # ------------------------------------------------------------------ #
    # Postura (RQF02 2 y 3, RQF03)
    # ------------------------------------------------------------------ #
    def _analyse_pose(self, image: np.ndarray) -> list[dict]:
        """Estima las condiciones que no se pueden encerrar en una caja.

        Tres señales, todas a partir de los keypoints de MediaPipe Pose:

          * `sin_pasamanos`: la muñeca baja del hombro Indicates que el brazo
            cuelga (no está sujetando nada a la altura de la barandilla).
          * `distraccion`: cabeza inclinada Y muñeca cerca de la cabeza, que es
            la postura de alguien mirando el móvil.
          * `tambaleo`: inclinación del tronco y asimetría entre rodillas, que
            es lo que delata un paso inestable.
        """
        out: list[dict] = []
        if not self.enable_pose or self.pose is None:
            return out
        res = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return out
        # `res.pose_landmarks` es un MENSAJE protobuf `NormalizedLandmarkList`
        # con los puntos de UNA sola persona (MediaPipe corre con num_poses=1).
        # No es ni iterable ni subscriptable: hay que bajar a su campo
        # `.landmark`, que si es un `RepeatedCompositeContainer` de 33 puntos.
        # Iterarlo directamente lanza TypeError y tumba el fotograma entero,
        # con lo que se pierden tambien las detecciones y los obstaculos ya
        # calculados.
        lm = res.pose_landmarks.landmark
        h, w = image.shape[:2]
        def p(i):
            return lm[i] if i < len(lm) else None
        nose, ls, rs = p(self.NOSE), p(self.LS), p(self.RS)
        lw, rw = p(self.LW), p(self.RW)
        lh, rh = p(self.LH), p(self.RH)
        lk, rk = p(self.LK), p(self.RK)
        if not all(x is not None for x in (nose, ls, rs, lw, rw)):
            return out      # pose incompleta: no hay nada que concluir

        signals = []
        # --- RQF02 (2): no usa el pasamanos ---
        # el pasamanos esta a la altura de la cadera/codo. Si ninguna
        # muñeca sube por encima de la altura del hombro, no hay agarre.
        shoulder_y = (ls.y + rs.y) / 2
        wrist_high = min(lw.y, rw.y) < shoulder_y - 0.02
        # un brazo muy écartado del tronco tambien indica que no se agarra
        span = abs(ls.x - rs.x) * w
        arm_out = (abs(lw.x - ls.x) * w) > span * 1.2
        if not wrist_high and arm_out:
            signals.append({
                "tipo": "sin_pasamanos", "conf": 0.55,
                "detalle": "ninguna muñeca a la altura de la barandilla",
            })

        # --- RQF02 (3): distraccion (movil o lectura) ---
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

        # --- tambaleo ---
        lean = 0.0
        if lh and rh:
            hip_mid = (lh.x + rh.x) / 2
            lean = abs(nose.x - hip_mid)
        knee_asym = abs((lk.y - lh.y) if (lk and lh) else 0) - \
            abs((rk.y - rh.y) if (rk and rh) else 0)
        if lean > 0.09 or abs(knee_asym) > 0.10:
            signals.append({
                "tipo": "tambaleo", "conf": 0.5,
                "detalle": f"inclinación tronco {lean:.2f}, asimetría rodillas {knee_asym:.2f}",
            })

        # Se devuelve la persona SIEMPRE, con sus keypoints, aunque no haya
        # ninguna señal de riesgo. Antes solo se añadia dentro de
        # `if signals:`, y los keypoints desaparecian justamente en el caso
        # que mas hace falta: una persona que camina recta y no dispara
        # ninguna señal, donde no habia forma de deducir la postura.
        out.append({
            "landmarks": [(lm[i].x, lm[i].y) for i in
                          (self.NOSE, self.LS, self.RS, self.LW, self.RW,
                           self.LH, self.RH, self.LK, self.RK,
                           self.LA, self.RA)],
            "signals": signals,
        })
        return out

    # ------------------------------------------------------------------ #
    # Obstáculos (RQF02 1)
    # ------------------------------------------------------------------ #
    def _obstacles(self, image: np.ndarray) -> list[dict]:
        """Detecta objetos abandonados sobre la escalera con un modelo COCO."""
        res = self.obs.predict(
            image, conf=CONF_OBSTACLE, iou=0.5, device=self.device, verbose=False
        )[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cid = int(b.cls[0])
            if cid not in self.obs_ids:
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            # el movil se separa: no es obstaculo, es evidencia de RQF02 (3)
            if cid in self.obs_phone_ids:
                out.append({
                    "tipo": "telefono",
                    "objeto": self.obs_names[cid],
                    "riesgo": "MEDIO",
                    "conf": round(float(b.conf[0]), 3),
                    "bbox": [x1, y1, x2, y2],
                })
                continue
            es_riesgo = cid in self.obs_risk_ids
            out.append({
                "tipo": "obstaculo_escalon" if es_riesgo else "elemento_escena",
                "objeto": self.obs_names[cid],
                "riesgo": "ALTO" if es_riesgo else "CONTEXTO",
                "conf": round(float(b.conf[0]), 3),
                "bbox": [x1, y1, x2, y2],
            })
        return out

    # ------------------------------------------------------------------ #
    # Persistencia y alertas (RQF06, RQF07)
    # ------------------------------------------------------------------ #
    def _persist(self, tid: int, risk: str, det: dict) -> None:
        now = time.time()
        st = self.tracks.get(tid)
        if st is None:
            return
        st.risk_since.setdefault(risk, now)
        dur = now - st.risk_since[risk]
        last = st.alerts.get(risk, 0.0)
        # RQF06: solo se alerta tras PERSIST_SECONDS de persistencia
        if dur >= PERSIST_SECONDS and now - last > ALERT_COOLDOWN:
            st.alerts[risk] = now
            meta = RISK_CATALOG.get(risk, {})
            self._alert_seq += 1
            evidence = self._save_evidence(now)
            alert = Alert(
                alert_id=f"ALT-{self._alert_seq:06d}",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                risk_type=risk,
                risk_ref=meta.get("ref", ""),
                severity=meta.get("riesgo", "MEDIO"),
                description=meta.get("desc", ""),
                track_id=tid,
                confidence=round(det.get("confidence", 0.0), 4),
                bbox=det["bbox"],
                evidence=evidence,
                duration_s=round(dur, 2),
            )
            self.alerts.append(alert)
            self._write_audit(alert)          # RQNF16

    def _write_audit(self, alert: Alert) -> None:
        """Registro auditable append-only (RQNF16, RQNF23)."""
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")

    def _save_evidence(self, now: float) -> str:
        """Guarda el fotograma como evidencia fotografica (RQF07, RQNF18)."""
        d = Path("runs/evidence")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"ev_{int(now * 1000)}.jpg"
        frame = getattr(self, "_last_frame", None)
        if frame is not None:
            cv2.imwrite(str(path), self._anonymise(frame.copy()))
        return str(path)

    # ------------------------------------------------------------------ #
    # Privacidad (RQNF14, Ley 29733)
    # ------------------------------------------------------------------ #
    def _anonymise(self, frame: np.ndarray) -> np.ndarray:
        """Pixeliza las caras detectadas antes de guardar o transmitir."""
        if not self.anonymize:
            return frame
        try:
            import mediapipe as mp
            if not hasattr(self, "_face"):
                self._face = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.4)
            res = self._face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception:
            return frame
        if res.detections:
            h, w = frame.shape[:2]
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x1 = int((bb.xmin - 0.08) * w); y1 = int((bb.ymin - 0.08) * h)
                x2 = int((bb.xmax + 0.08) * w); y2 = int((bb.ymax + 0.08) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = frame[y1:y2, x1:x2]
                small = cv2.resize(roi, (max(1, roi.shape[1] // 12),
                                         max(1, roi.shape[0] // 12)))
                frame[y1:y2, x1:x2] = cv2.resize(
                    small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
        return frame

    # ------------------------------------------------------------------ #
    # Umbrales (RQF04)
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_thresholds(dets: list[dict]) -> tuple[list[dict], list[dict]]:
        """Aplica el doble umbral 75% / 85% de RQF04.

        Devuelve `(confirmadas, preliminares)`: las de confianza >= 0.85
        cuentan como hallazgo firme; las de 0.75-0.85 quedan en cuarentena
        hasta que la persistencia las consolida.
        """
        conf = [d for d in dets if d["confidence"] >= CONF_CONFIRM]
        pre = [d for d in dets if CONF_PREFILTER <= d["confidence"] < CONF_CONFIRM]
        return conf, pre

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    # --- Clasificacion de postura por geometria (respaldo) -----------------
    # Medido sobre tres fases de una prueba real (persona en una escalera):
    #
    #   fase                inclinacion   cabeza_sobre_cadera
    #   caminando erguido       2.3 deg          1.02
    #   tambaleandose           1.5 deg          0.75
    #   caido en el suelo      51.5 deg          0.38
    #
    # `inclinacion` es el angulo del eje tobillo->cadera frente a la vertical:
    # de pie son 2 grados, tumbado 51. Discrimina "caido" sin ambiguedad.
    # `cabeza_sobre_cadera` es la altura de la nariz sobre la cadera,
    # normalizada por la longitud del eje.
    #
    # Se descarto medir el aspecto ancho/alto del esqueleto: con el brazo en
    # alto hacia la barandilla la caja se ensancha y daba 1.92 en una persona
    # de pie, confundiendose con alguien caido.
    #
    # LIMITACION: calibrado con TRES imagenes. No es una metrica de precision,
    # es un respaldo para cuando el detector no tiene opinion alguna, que es lo
    # que ocurre en fotos reales de escalera, fuera de su dominio.
    GEOM_CAIDO_INCLINA = 25.0
    GEOM_CAIDO_CABEZA = 0.55
    GEOM_DESEQUILIBRIO_CABEZA = 0.90

    def _postura_por_geometria(self, pose: list[dict]) -> str | None:
        """Deduce la postura de los keypoints. None si no hay pose.

        Indices de `landmarks`: 0 nariz, 1/2 hombros, 3/4 munecas,
        5/6 caderas, 7/8 rodillas, 9/10 tobillos.
        """
        if not pose or len(pose[0].get("landmarks", [])) < 11:
            return None
        pts = pose[0]["landmarks"]
        hip = ((pts[5][0] + pts[6][0]) / 2, (pts[5][1] + pts[6][1]) / 2)
        tob = ((pts[9][0] + pts[10][0]) / 2, (pts[9][1] + pts[10][1]) / 2)
        largo = math.hypot(hip[0] - tob[0], hip[1] - tob[1])
        if largo <= 0:
            return None
        inclinacion = math.degrees(
            math.atan2(abs(hip[0] - tob[0]), abs(hip[1] - tob[1])))
        cabeza = (hip[1] - pts[0][1]) / largo
        if inclinacion > self.GEOM_CAIDO_INCLINA or cabeza < self.GEOM_CAIDO_CABEZA:
            return "persona_caido"
        if cabeza < self.GEOM_DESEQUILIBRIO_CABEZA:
            return "persona_desequilibrio"
        return "persona_erguida"

    # --- Localizacion con COCO + clasificacion propia ---------------------- #
    def _find_people(self, frame: np.ndarray) -> list[list[int]]:
        """Detecta personas con el modelo COCO de 80 clases.

        Es un detector general: en la prueba con una foto real de escalera
        encontro a la persona al 0.92 mientras el modelo propio no llegaba a
        0.08, porque la clase `escalera` saturaba el encuadre. El reparto
        correcto es: COCO decide DONDE hay gente, el modelo propio decide QUE
        POSTURA tiene.
        """
        res = self.obs.predict(
            frame, conf=self.person_conf, iou=0.5, imgsz=self.person_imgsz,
            device=self.device, verbose=False
        )[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            if self.obs_names.get(int(b.cls[0])) != "person":
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if x2 - x1 > 8 and y2 - y1 > 8:
                out.append([x1, y1, x2, y2])
        return out

    def _classify_person(self, frame: np.ndarray, box: list[int]) -> tuple[str, float]:
        """Clasifica la postura recortando la persona y pasando el recorte.

        El recorte se ajusta a la caja con un margen proporcional POR LADO y se
        reescala a cuadrado. No se usa un cuadrado del lado
        max(ancho,alto): con una persona alta (354x1073 en la prueba) eso daba
        un recorte mas ancho que la imagen entera y la escalera dominaba.
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
        res = self.det.predict(crop, conf=0.20, device=self.device, verbose=False)[0]
        best, bestc = "", 0.0
        if res.boxes is not None:
            for b in res.boxes:
                name = self.risk_names.get(int(b.cls[0]), "")
                if name == "escalera":       # no describe a una persona
                    continue
                c = float(b.conf[0])
                if c > bestc:
                    best, bestc = name, c
        return best, bestc

    def step(self, frame: np.ndarray) -> dict:
        """Procesa un fotograma y devuelve el estado completo del sistema."""
        t0 = time.perf_counter()
        self._last_frame = frame

        self._frame_no = getattr(self, "_frame_no", 0) + 1
        run_extras = (self._frame_no % max(1, self.pose_every)) == 0
        # Los obstaculos van a una cadencia MAS LENTA que la pose: una mochila
        # en un escalon no aparece ni desaparece en tres fotogramas. Medido:
        # YOLO riesgo 19.4 ms, pose 20.0 ms y obstaculos 20.3 ms.
        run_obstacles = (self._frame_no % max(1, self.obstacle_every)) == 0
        run_people = (self._frame_no % max(1, self.people_every)) == 0
        run_stairs = (self._frame_no % max(1, self.stairs_every)) == 0

        # --- localizacion de personas con COCO (RQF02) ---
        #
        # Antes se pedia al modelo propio DONDE estaba la persona. En una foto
        # real de escalera no encontraba a nadie (0.08 de confianza maxima)
        # mientras COCO la encontraba al 0.92: la clase `escalera` saturaba
        # el encuadre con seis cajas solapadas. Se invierte el reparto: COCO
        # localiza, el modelo propio clasifica la postura.
        # La cache de personas solo se reutiliza si ya hay alguna: en la
        # primera imagen, o al cambiar de una a otra, la cache estaria vacia
        # o seria de OTRA imagen y la persona no se detectaria.
        people = self._find_people(frame) if (run_people or not self._last_people) \
            else self._last_people
        self._last_people = people

        # --- escalera, con tope de cajas y cadencia lenta ---
        # `escalera` es clase de CONTEXTO y no cambia entre fotogramas, pero
        # su deteccion cuesta 19 ms: la mitad del presupuesto. Se reutiliza la
        # ultima caja como con los obstaculos.
        if run_stairs or not self._last_stairs:
            res_s = self.det.predict(
                frame, conf=0.30, iou=0.6, device=self.device, verbose=False
            )[0]
            found = []
            if res_s.boxes is not None:
                for b in res_s.boxes:
                    if self.risk_names.get(int(b.cls[0])) == "escalera":
                        found.append((float(b.conf[0]),
                                      [int(v) for v in b.xyxy[0].tolist()]))
            found.sort(reverse=True)
            self._last_stairs = found[:self.MAX_STAIRS_BOXES]
        stairs = self._last_stairs

        # --- postura (RQF02 2 y 3) ---
        # La pose y los obstaculos se calculan cada `pose_every` frames: son
        # dos inferencias mas y a 30 FPS sostenidos no caben en el presupuesto.
        # Las condiciones que sostienen (sujeccion del pasamanos, distraccion,
        # La pose va envuelta: si falla, se pierde la señal de postura pero el
        # fotograma sigue produciendo detecciones y obstaculos. Antes un
        # TypeError aqui tumbaba `step()` entero y el endpoint devolvia 500.
        pose = []
        if run_extras:
            try:
                pose = self._analyse_pose(frame)
            except Exception as exc:       # noqa: BLE001
                self._pose_errors += 1
                if self._pose_errors <= 3:
                    print(f"  [aviso] analisis de pose fallo: "
                          f"{type(exc).__name__}: {exc}", flush=True)
        else:
            pose = []
        pose_signals = []
        for person in pose:
            for s in person["signals"]:
                meta = RISK_CATALOG.get(s["tipo"])
                if meta:
                    pose_signals.append({**s, **meta})
                else:   # tambaleo no esta en RQF02 pero se reporta aparte
                    pose_signals.append({
                        **s, "n": 4, "riesgo": "ALTO", "ref": "RQF02 (4)",
                        "desc": "Tambaleo: paso inestable o pérdida de equilibrio.",
                    })

        # --- obstaculos (RQF02 1) ---
        if run_obstacles:
            try:
                self._last_obstacles = self._obstacles(frame)
            except Exception as exc:       # noqa: BLE001
                if self._pose_errors <= 3:
                    print(f"  [aviso] deteccion de obstaculos fallo: "
                          f"{type(exc).__name__}: {exc}", flush=True)
        obstacles = self._last_obstacles

        # --- clasificacion de postura de cada persona ---
        #
        # Primero se recorta y se pasa por el modelo propio. Si el recorte no
        # da ninguna clase de persona --que es lo que pasa con las fotos reales
        # de escalera, fuera de su dominio-- se recurre a la geometria de la
        # pose. Antes, cuando no habia clase, se informaba como `persona`
        # generica: la caja aparecia pero sin ninguna postura.
        dets: list[dict] = []
        if pose:
            self._last_geom = self._postura_por_geometria(pose)
        geom = self._last_geom
        for box in people:
            name, conf_p = self._classify_person(frame, box)
            origen = "modelo"
            if not name and geom:
                name, conf_p, origen = geom, 0.55, "geometria de pose"
            if not name:
                name, conf_p, origen = "persona", 0.4, "sin clasificar"
            det = {
                "class_id": self.id_of.get(name, 4),
                "class_name": name,
                "confidence": round(float(conf_p), 4),
                "origen": origen,
                "bbox": box, "track_id": None,
                "riesgo": "NEUTRO", "ref": "",
            }
            if name in FALL_CLASSES:
                det["riesgo"] = RISK_CATALOG[FALL_CLASSES[name]]["riesgo"]
                det["ref"] = RISK_CATALOG[FALL_CLASSES[name]]["ref"]
                det["risk_type"] = FALL_CLASSES[name]
            elif name in NONERGO_CLASSES:
                det["riesgo"] = "MEDIO"
                det["ref"] = "RQF03"
                det["risk_type"] = NONERGO_CLASSES[name]
            elif name in SAFE_CLASSES:
                det["riesgo"] = "BAJO"
                det["ref"] = "RQF03"
            dets.append(det)

        dets = self._track(dets)
        confirmed, preliminary = self.apply_thresholds(dets)

        for conf_s, box in stairs:
            dets.append({
                "class_id": 3, "class_name": "escalera",
                "confidence": round(conf_s, 4), "bbox": box,
                "track_id": None, "riesgo": "CONTEXTO", "ref": "",
            })

        # --- telefono cerca de la cabeza: RQF02 (3) confirmado por evidencia --
        # Un movil detectado por COCO no basta: puede estar suelto. Se cruza
        # con los keypoints de la pose y solo si la caja del movil cae cerca
        # de la nariz se emite la señal de distraccion.
        telefonos = [o for o in obstacles if o.get("tipo") == "telefono"]
        if telefonos and pose:
            for person in pose:
                nose_x, nose_y = person["landmarks"][0]
                h, w = frame.shape[:2]
                for ph in telefonos:
                    cx = (ph["bbox"][0] + ph["bbox"][2]) / 2 / w
                    cy = (ph["bbox"][1] + ph["bbox"][3]) / 2 / h
                    if math.hypot(cx - nose_x, cy - nose_y) < 0.14:
                        pose_signals.append({
                            "tipo": "distraccion", "conf": min(0.9, ph["conf"] + 0.3),
                            "n": 3, "riesgo": "MEDIO", "ref": "RQF02 (3)",
                            "desc": "Uso de teléfono móvil detectado junto a la cabeza.",
                            "detalle": f"teléfono a {math.hypot(cx-nose_x, cy-nose_y):.2f} de la nariz",
                        })


        # --- persistencia y alertas (RQF06, RQF07) ---
        for d in dets:
            if "risk_type" in d and d["confidence"] >= CONF_PREFILTER:
                self._persist(d["track_id"], d["risk_type"], d)

        # --- nivel global ---
        # `escalera` es CONTEXTO y una persona NEUTRA no aporta: para el
        # veredicto solo cuentan las señales con consecuencia (RQF03)
        risks = [d["riesgo"] for d in dets if d["riesgo"] in ("ALTO", "MEDIO")]
        risks += [p["riesgo"] for p in pose_signals]
        # solo los obstaculos de riesgo suben el veredicto; el mobiliario
        # es informacion de escena
        risks += ["ALTO"] * sum(1 for o in obstacles
                                if o.get("riesgo") == "ALTO")
        level = max(risks, key=lambda r: {"ALTO": 3, "MEDIO": 2}.get(r, 0),
                                          default="BAJO")

        # --- estado del sistema (RQNF09, RQNF26) ---
        ms = (time.perf_counter() - t0) * 1000
        return {
            "detecciones": dets,
            "confirmadas": len(confirmed),
            "preliminares": len(preliminary),
            "postura": pose_signals,
            "obstaculos": obstacles,
            "alertas": [a.to_dict() for a in list(self.alerts)[-5:]],
            "riesgo": level,
            "tracks": len(self.tracks),
            "ms": round(ms, 1),
            "sistema": "OPERATIVO",
        }


# --------------------------------------------------------------------------- #
def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0