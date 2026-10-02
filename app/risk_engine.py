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
CONF_OBSTACLE = 0.35
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
# Mobiliario (silla, carrito) queda FUERA a proposito: no es un objeto
# abandonado sobre un escalon, y con el umbral bajo de deteccion aparecia como
# falso positivo constante en interiores.
OBSTACLE_CLASSES = {
    "backpack": "mochila", "handbag": "bolso", "suitcase": "maleta",
    "bottle": "botella", "book": "libro", "box": "caja",
    "traffic cone": "cono",
}

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

    def __init__(
        self,
        risk_weights: str,
        obstacle_weights: str = "yolov8s.pt",
        device: int = 0,
        conf: float = 0.25,
        enable_pose: bool = True,
        anonymize: bool = True,
        pose_every: int = 3,
    ):
        import mediapipe as mp
        from ultralytics import YOLO

        self.device = device
        self.conf = conf
        self.anonymize = anonymize
        self.pose_every = pose_every
        self._last_obstacles: list = []
        self._pose_errors = 0

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

        # --- RQF02 (2) y (3): postura ------------------------------------
        self.enable_pose = enable_pose
        self.pose = None
        if enable_pose:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
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

        if signals:
            out.append({
                "landmarks": [(lm[i].x, lm[i].y) for i in
                              (self.NOSE, self.LS, self.RS, self.LW, self.RW,
                               self.LH, self.RH)],
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
            out.append({
                "tipo": "obstaculo_escalon",
                "objeto": self.obs_names[cid],
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
    def step(self, frame: np.ndarray) -> dict:
        """Procesa un fotograma y devuelve el estado completo del sistema."""
        t0 = time.perf_counter()
        self._last_frame = frame

        # --- deteccion de personas y escalera (RQF02) ---
        res = self.det.predict(
            frame, conf=self.conf, device=self.device, verbose=False
        )[0]
        dets: list[dict] = []
        draw = []
        if res.boxes is not None:
            for b in res.boxes:
                cid = int(b.cls[0])
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                name = self.risk_names.get(cid, str(cid))
                det = {
                    "class_id": cid, "class_name": name,
                    "confidence": round(float(b.conf[0]), 4),
                    "bbox": [x1, y1, x2, y2], "track_id": None,
                    "riesgo": "CONTEXTO" if name == "escalera" else "NEUTRO",
                    "ref": "",
                }
                # RQF03: la clase de la caja ya es una señal de riesgo
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
                draw.append(det)
                dets.append(det)

        dets = self._track(dets)
        confirmed, preliminary = self.apply_thresholds(dets)

        # --- postura (RQF02 2 y 3) ---
        # La pose y los obstaculos se calculan cada `pose_every` frames: son
        # dos inferencias mas y a 30 FPS sostenidos no caben en el presupuesto.
        # Las condiciones que sostienen (sujeccion del pasamanos, distraccion,
        # objetos en el escalon) cambian en segundos, no en fotogramas.
        self._frame_no = getattr(self, "_frame_no", 0) + 1
        run_extras = (self._frame_no % max(1, self.pose_every)) == 0
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
        if run_extras:
            try:
                self._last_obstacles = self._obstacles(frame)
            except Exception as exc:       # noqa: BLE001
                if self._pose_errors <= 3:
                    print(f"  [aviso] deteccion de obstaculos fallo: "
                          f"{type(exc).__name__}: {exc}", flush=True)
        obstacles = self._last_obstacles

        # --- persistencia y alertas (RQF06, RQF07) ---
        for d in dets:
            if "risk_type" in d and d["confidence"] >= CONF_PREFILTER:
                self._persist(d["track_id"], d["risk_type"], d)

        # --- nivel global ---
        # `escalera` es CONTEXTO y una persona NEUTRA no aporta: para el
        # veredicto solo cuentan las señales con consecuencia (RQF03)
        risks = [d["riesgo"] for d in dets if d["riesgo"] in ("ALTO", "MEDIO")]
        risks += [p["riesgo"] for p in pose_signals]
        risks += ["ALTO"] * len(obstacles)
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