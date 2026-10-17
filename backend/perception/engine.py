"""Motor de riesgo: une detectores, pose, seguimiento y alertas.

    fotograma
       │
       ├─ CocoDetector.people()      ¿dónde hay personas?          RQF02
       ├─ PostureDetector.stairs()   ¿dónde está la escalera?      contexto
       ├─ PoseAnalyzer.analyse()     pasamanos, distracción, tambaleo  RQF02 (2)(3)(4)
       ├─ CocoDetector.obstacles()   objetos sobre el escalón      RQF02 (1)
       ├─ postura de cada persona    geometría de pose > recorte   RQF03
       ├─ IoUTracker.update()        identidad estable             RQF05
       ├─ AlertManager               persistencia + alerta         RQF06 · RQF07
       └─ veredicto global + estado del sistema                    RQNF09 · RQNF26

CADENCIAS. Cada etapa tiene su propia frecuencia (`engine.*_every`): una
mochila en un escalón o la sujeción del pasamanos no cambian en tres
fotogramas, y ejecutar todo en cada frame no cabe en el presupuesto de 33 ms.
Entre ejecuciones se reutiliza el último resultado de esa etapa. Con
`force=True` (imagen fija) se ejecuta todo.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..config import Settings
from ..privacy.anonymizer import FaceAnonymizer
from .alerts import AlertManager
from .catalog import (FALL_CLASSES, NONERGO_CLASSES, RISK_CATALOG, RISK_ORDER,
                      SAFE_CLASSES, TAMBALEO_META)
from .detectors import CocoDetector, PostureDetector
from .pose import PoseAnalyzer, best_person_match, posture_from_geometry
from .tracking import IoUTracker


def resolve_device(device):
    """'auto' -> 0 si hay CUDA, si no 'cpu'."""
    if device == "auto":
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    return device


class RiskEngine:
    def __init__(self, settings: Settings, conf: float | None = None,
                 enable_pose: bool = True):
        s = settings
        self.settings = s
        self.device = resolve_device(s.device)
        self.conf = conf if conf is not None else s.inference.conf
        e, th, ps = s.engine, s.thresholds, s.persistence

        self.pose_every = e.pose_every
        self.people_every = e.people_every
        self.stairs_every = e.stairs_every
        self.obstacle_every = e.obstacle_every
        self.classify_every = e.classify_every
        self.max_stairs_boxes = e.max_stairs_boxes
        self.th = th

        # --- componentes -------------------------------------------------
        self.posture = PostureDetector(str(s.detector_path), self.device)
        self.coco = CocoDetector(str(s.coco_path), self.device,
                                 obstacle_conf=th.obstacle,
                                 person_conf=e.person_conf,
                                 person_imgsz=e.person_imgsz)
        self.pose = PoseAnalyzer(enabled=enable_pose)
        self.tracker = IoUTracker(iou_match=ps.iou_match,
                                  occlusion_seconds=ps.occlusion_seconds)
        self.alerts = AlertManager(
            self.tracker, s.alerts_dir,
            persist_seconds={"ALTO": ps.alto_seconds, "MEDIO": ps.medio_seconds},
            persist_default=ps.default_seconds,
            grace_seconds=ps.grace_seconds,
            cooldown_seconds=ps.cooldown_seconds,
            keep_days=s.alerts.keep_days, max_images=s.alerts.max_images,
            anonymizer=FaceAnonymizer(enabled=s.privacy.anonymize_faces),
        )

        # --- caches entre fotogramas --------------------------------------
        self._frame_no = 0
        self._last_people: list = []
        self._last_stairs: list = []
        self._last_obstacles: list = []
        self._last_geom: str | None = None
        self._last_landmarks: list | None = None
        self._last_class: tuple | None = None
        # fotogramas seguidos en que un track mantiene "caída" por recorte
        self._fall_held: dict = {}

        # --- estado operativo (RQNF09, RQNF26) ----------------------------
        self._errors = {"gpu": 0, "pose": 0, "deteccion": 0}
        self._error_prints = {"pose": 0, "deteccion": 0}
        self.camera_connected = False

    # ------------------------------------------------------------------ #
    @property
    def names(self) -> dict[int, str]:
        return self.posture.names

    def reset(self) -> None:
        """Limpia las caches de fotograma (cada imagen subida es otra escena)."""
        self._last_people = []
        self._last_stairs = []
        self._last_obstacles = []
        self._last_geom = None
        self._last_landmarks = None
        self._last_class = None
        self._fall_held.clear()

    @property
    def has_pending_evidence(self) -> bool:
        return self.alerts.has_pending

    def flush_evidence(self, annotated: np.ndarray) -> None:
        self.alerts.flush_evidence(annotated, people=self._last_people)

    # ------------------------------------------------------------------ #
    # Estado del sistema
    # ------------------------------------------------------------------ #
    def status(self) -> str:
        if self.posture is None or self.coco is None:
            return "NO OPERATIVO"
        if not self.camera_connected or any(self._errors.values()):
            return "DEGRADADO"
        return "OPERATIVO"

    def status_detail(self) -> dict:
        return {
            "estado": self.status(),
            "fallos_gpu": self._errors["gpu"],
            "fallos_pose": self._errors["pose"],
            "fallos_deteccion": self._errors["deteccion"],
            "camara": self.camera_connected,
            "tracks": len(self.tracker.tracks),
            "alertas_emitidas": len(self.alerts.history),
        }

    def _fail(self, kind: str, exc: Exception) -> None:
        self._errors[kind] += 1
        self._error_prints[kind] = self._error_prints.get(kind, 0) + 1
        if self._error_prints[kind] <= 3:
            print(f"  [aviso] {kind} falló: {type(exc).__name__}: {exc}",
                  flush=True)

    # ------------------------------------------------------------------ #
    # Umbrales RQF04
    # ------------------------------------------------------------------ #
    def apply_thresholds(self, dets: list[dict]) -> tuple[list, list]:
        """(confirmadas >= confirm, preliminares entre prefilter y confirm)."""
        confirmed = [d for d in dets if d["confidence"] >= self.th.confirm]
        prelim = [d for d in dets
                  if self.th.prefilter <= d["confidence"] < self.th.confirm]
        return confirmed, prelim

    # ------------------------------------------------------------------ #
    # Caídas por recorte: confirmación temporal
    # ------------------------------------------------------------------ #
    def _confirm_falls(self, dets: list[dict]) -> None:
        """Una caída vista SOLO por el recorte (sin pose) pasa de MEDIO
        "REVISAR" a ALTO únicamente si se mantiene `fall_confirm_frames`
        fotogramas seguidos con el mismo track. La geometría de pose es una
        señal fiable y ya salió a ALTO sin pasar por aquí."""
        present = set()
        for d in dets:
            if not d.get("_fall_pending"):
                continue
            tid = d.get("track_id")
            if tid is None:
                d["riesgo"] = "MEDIO"
                continue
            present.add(tid)
            held = self._fall_held[tid] = self._fall_held.get(tid, 0) + 1
            if held >= self.th.fall_confirm_frames:
                d["riesgo"] = "ALTO"
                d["ref"] = (f"caída confirmada ({held} frames, "
                            f"modelo {d['confidence']:.2f})")
        for tid in list(self._fall_held):
            if tid not in present:
                del self._fall_held[tid]
        for d in dets:
            d.pop("_fall_pending", None)

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def step(self, frame: np.ndarray, force: bool = False,
             allow_alerts: bool = True) -> dict:
        """Procesa un fotograma y devuelve el estado completo del sistema."""
        t0 = time.perf_counter()
        stages: dict = {}
        self._frame_no += 1
        n = self._frame_no

        def due(every: int) -> bool:
            return force or n % max(1, every) == 0

        # --- 1. personas (COCO) ------------------------------------------
        people = self.coco.people(frame) \
            if (due(self.people_every) or not self._last_people) \
            else self._last_people
        self._last_people = people
        stages["people"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- 2. escalera (contexto, cadencia lenta) -----------------------
        t = time.perf_counter()
        if due(self.stairs_every) or not self._last_stairs:
            self._last_stairs = self.posture.stairs(frame, self.max_stairs_boxes)
        stairs = self._last_stairs
        stages["stairs"] = round((time.perf_counter() - t) * 1000, 2)

        # --- 3. pose (RQF02 2-3-4) ----------------------------------------
        # Si MediaPipe falla se pierde la señal de postura, pero el fotograma
        # sigue produciendo cajas y obstáculos.
        t = time.perf_counter()
        pose: list = []
        if due(self.pose_every):
            try:
                pose = self.pose.analyse(frame)
            except Exception as exc:                        # noqa: BLE001
                self._fail("pose", exc)
        stages["pose"] = round((time.perf_counter() - t) * 1000, 2)

        pose_signals = []
        for person in pose:
            for sig in person["signals"]:
                meta = RISK_CATALOG.get(sig["tipo"], TAMBALEO_META)
                pose_signals.append({**sig, **meta})

        # --- 4. obstáculos (RQF02 1) --------------------------------------
        t = time.perf_counter()
        if due(self.obstacle_every):
            try:
                self._last_obstacles = self.coco.obstacles(frame)
            except Exception as exc:                        # noqa: BLE001
                self._fail("deteccion", exc)
        obstacles = self._last_obstacles
        stages["obstacles"] = round((time.perf_counter() - t) * 1000, 2)

        # --- 5. postura de cada persona -----------------------------------
        # La geometría de la pose manda sobre el recorte. Medido en test:
        #   de pie: recorte 31 % vs geometría 64 %; caído: 78 % vs 93 %.
        # La pose se empareja con la caja que mejor encaja; el resto de
        # personas se clasifican por recorte.
        if pose:
            self._last_geom = posture_from_geometry(pose)
            self._last_landmarks = pose[0]["landmarks"]
        geom = self._last_geom
        pose_idx = best_person_match(self._last_landmarks, people, frame.shape)
        if pose_idx < 0 and len(people) == 1:
            pose_idx = 0          # una sola persona: la pose es suya

        run_class = (n % max(1, self.classify_every) == 0
                     or self._last_class is None)
        t = time.perf_counter()
        dets: list[dict] = []
        for i, box in enumerate(people):
            is_pose = geom is not None and i == pose_idx
            if is_pose:
                name, conf_p, origin = geom, 0.55, "geometria de pose"
            elif run_class or not self._last_class \
                    or abs(self._last_class[0][0] - box[0]) > 64:
                name, conf_p = self.posture.classify(frame, box)
                self._last_class = (tuple(box), name, conf_p)
                origin = "modelo"
            else:
                _, name, conf_p = self._last_class
                origin = "modelo (cache)"
            if not name:
                name, conf_p, origin = "persona", 0.4, "sin clasificar"

            det = {
                "class_id": self.posture.id_of.get(name, 4),
                "class_name": name,
                "confidence": round(float(conf_p), 4),
                "origen": origin,
                "bbox": box, "track_id": None,
                "riesgo": "NEUTRO", "ref": "",
            }
            if name in FALL_CLASSES:
                det["risk_type"] = FALL_CLASSES[name]
                if is_pose:
                    meta = RISK_CATALOG[det["risk_type"]]
                    det["riesgo"], det["ref"] = meta["riesgo"], meta["ref"]
                elif conf_p < self.th.fall_crop_conf:
                    # recorte débil: se marca para revisión, no alerta
                    det["riesgo"] = "MEDIO"
                    det["ref"] = (f"REVISAR (modelo {conf_p:.2f} < "
                                  f"{self.th.fall_crop_conf}, sin pose)")
                else:
                    det["riesgo"] = "MEDIO"
                    det["ref"] = f"caída? modelo {conf_p:.2f} (sin pose, confirmar)"
                    det["_fall_pending"] = True
            elif name in NONERGO_CLASSES:
                det["riesgo"], det["ref"] = "MEDIO", "RQF03"
                det["risk_type"] = NONERGO_CLASSES[name]
            elif name in SAFE_CLASSES:
                det["riesgo"], det["ref"] = "BAJO", "RQF03"
            dets.append(det)
        stages["classify"] = round((time.perf_counter() - t) * 1000, 2)

        # Las señales de pose pertenecen a la persona emparejada: su caja
        # toma el mayor riesgo entre su clase y sus señales.
        if pose_signals and 0 <= pose_idx < len(dets):
            d = dets[pose_idx]
            current = d.get("riesgo", "NEUTRO")
            d["senales"] = [x["tipo"] for x in pose_signals]
            d["riesgo"] = max([current] + [x["riesgo"] for x in pose_signals],
                              key=lambda r: RISK_ORDER.get(r, 0))
            if d["riesgo"] != current:
                d["riesgo_por_clase"] = current

        # --- 6. seguimiento + confirmación de caídas ----------------------
        self.tracker.update(dets)
        self._confirm_falls(dets)
        for conf_s, box in stairs:
            dets.append({
                "class_id": 3, "class_name": "escalera",
                "confidence": round(conf_s, 4), "bbox": box,
                "track_id": None, "riesgo": "CONTEXTO", "ref": "",
            })
        confirmed, preliminary = self.apply_thresholds(dets)

        # --- 7. teléfono junto a la cabeza => distracción (RQF02 3) -------
        phones = [o for o in obstacles if o.get("tipo") == "telefono"]
        if phones and pose:
            h, w = frame.shape[:2]
            for person in pose:
                nose_x, nose_y = person["landmarks"][0]
                for ph in phones:
                    cx = (ph["bbox"][0] + ph["bbox"][2]) / 2 / w
                    cy = (ph["bbox"][1] + ph["bbox"][3]) / 2 / h
                    dist = math.hypot(cx - nose_x, cy - nose_y)
                    if dist < 0.14:
                        pose_signals.append({
                            "tipo": "distraccion",
                            "conf": min(0.9, ph["conf"] + 0.3),
                            "n": 3, "riesgo": "MEDIO", "ref": "RQF02 (3)",
                            "desc": "Uso de teléfono móvil detectado junto a la cabeza.",
                            "detalle": f"teléfono a {dist:.2f} de la nariz",
                        })

        # --- 8. persistencia y alertas (RQF06, RQF07) ---------------------
        active = {(d["track_id"], d["risk_type"]) for d in dets
                  if "risk_type" in d and d["confidence"] >= self.th.prefilter}
        self.alerts.forget_stale(active)
        if allow_alerts:
            for d in dets:
                if "risk_type" in d and d["confidence"] >= self.th.alert:
                    self.alerts.observe(d["track_id"], d["risk_type"], d)
            self.alerts.evaluate_survivors(dets)

        # --- 9. veredicto global ------------------------------------------
        risks = [d["riesgo"] for d in dets if d["riesgo"] in ("ALTO", "MEDIO")]
        risks += [p["riesgo"] for p in pose_signals]
        risks += ["ALTO"] * sum(1 for o in obstacles if o.get("riesgo") == "ALTO")
        level = max(risks, key=lambda r: RISK_ORDER.get(r, 0), default="BAJO")

        # El estado refleja los fallos de ESTE fotograma; después se olvidan
        # para que un fallo aislado no deje el sistema degradado para siempre.
        system_status = self.status()
        self._errors = {k: 0 for k in self._errors}

        ms = (time.perf_counter() - t0) * 1000
        stages["total"] = round(ms, 1)
        return {
            "detecciones": dets,
            "confirmadas": len(confirmed),
            "preliminares": len(preliminary),
            "postura": pose_signals,
            "obstaculos": obstacles,
            "alertas": self.alerts.recent(5),
            "riesgo": level,
            "tracks": len(self.tracker.tracks),
            "ms": round(ms, 1),
            "stages": stages,
            "sistema": system_status,
        }
