"""Motor de riesgo: une detectores, pose, seguimiento y alertas.

    fotograma
       │
       ├─ CocoDetector.people()      ¿dónde hay personas?          RQF02
       ├─ PostureDetector.stairs()   ¿dónde está la escalera?      contexto
       ├─ SkeletonAnalyzer.step()    YOLOv8-Pose: 17 puntos por persona, ID,
       │                             equilibrio y seguro de postura  RQF02 (4)
       ├─ CocoDetector.obstacles()   objetos sobre el escalón      RQF02 (1)
       ├─ postura de cada persona    clasificador del Modelo v5    RQF03
       ├─ señales por esqueleto      pasamanos, distracción        RQF02 (2)(3)
       ├─ IoUTracker.update()        identidad estable             RQF05
       ├─ AlertManager               persistencia + alerta         RQF06 · RQF07
       └─ veredicto global + estado del sistema                    RQNF09 · RQNF26

CADENCIAS. Cada etapa tiene su propia frecuencia (`engine.*_every`): una
mochila en un escalón o la sujeción del pasamanos no cambian en tres
fotogramas, y ejecutar todo en cada frame no cabe en el presupuesto de 33 ms.
Entre ejecuciones se reutiliza el último resultado de esa etapa. Un
planificador ejecuta como mucho UNA etapa pesada por fotograma (`_schedule`)
para que no se sumen en picos. Con `force=True` (imagen fija) se ejecuta todo.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..config import Settings
from ..privacy.anonymizer import FaceAnonymizer
from .alerts import AlertManager
from .catalog import (FALL_CLASSES, NONERGO_CLASSES, RISK_CATALOG, RISK_ORDER,
                      SAFE_CLASSES)
from .detectors import CocoDetector, PostureDetector
from .tracking import IoUTracker, iou

BALANCE_RISK = "desequilibrio_pose"


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

        self.people_every = e.people_every
        self.stairs_every = e.stairs_every
        self.obstacle_every = e.obstacle_every
        self.classify_every = e.classify_every
        self.max_stairs_boxes = e.max_stairs_boxes
        self.th = th

        # --- componentes -------------------------------------------------
        from . import accel
        accel.configure(getattr(s.hardware, "openvino_cpu", True),
                        getattr(s.hardware, "openvino_int8_coco", True))
        self.posture = PostureDetector(str(s.detector_path),
                                       str(s.posture_path), self.device)
        # La confianza de PERSONA la controla la barra «Sensibilidad» del panel
        # (self.conf); arranca con inference.conf de configs/app.yaml.
        self.coco = CocoDetector(str(s.coco_path), self.device,
                                 obstacle_conf=th.obstacle,
                                 person_conf=self.conf,
                                 person_imgsz=e.person_imgsz)
        # MediaPipe Pose ya no se usa en el motor: describía UNA persona y
        # emparejaba su pose con la caja equivocada en escenas con gente. Las
        # señales salen del esqueleto de cada persona (skeleton/signals.py).
        # (MediaPipe Face sigue anonimizando las evidencias.)
        self.signals_cfg = s.skeleton.senales
        # Zona de escalera y pasamanos dibujada en el panel (perception/zone.py)
        from .zone import Zone
        self.zone_path = s.path("configs/zona_local.json")
        self.zone = Zone.load(self.zone_path)
        self.skeleton = None
        self.skeleton_every = max(1, s.skeleton.every)
        if s.skeleton.enabled and enable_pose:
            try:
                from .skeleton import SkeletonAnalyzer
                self.skeleton = SkeletonAnalyzer(s.skeleton, self.device)
            except Exception as exc:                        # noqa: BLE001
                print(f"  [aviso] esqueletos YOLOv8-Pose desactivados: {exc}",
                      flush=True)
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
            # BalanceMonitor ya exige `persist_s`; aquí no se suma más espera.
            persist_by_risk={BALANCE_RISK: 0.0},
            cooldown_by_risk={BALANCE_RISK: s.skeleton.balance.cooldown_s},
        )

        # --- caches entre fotogramas --------------------------------------
        self._frame_no = 0
        # None = la etapa aún no se ha ejecutado (ver step(): cadencias)
        self._last_people: list | None = None
        self._last_stairs: list | None = None
        self._last_obstacles: list = []
        # planificador: próximo fotograma en que toca cada etapa pesada
        self._next_due: dict[str, int] = {}
        # [(caja, postura, prob), ...] de la última pasada del clasificador
        self._last_class: list | None = None
        self._last_skeletons: list | None = None
        # fotogramas seguidos en que un track mantiene "caída" por recorte
        self._fall_held: dict = {}

        # --- estado operativo (RQNF09, RQNF26) ----------------------------
        self._errors = {"gpu": 0, "pose": 0, "deteccion": 0}
        self._error_prints = {"pose": 0, "deteccion": 0}
        self.camera_connected = False

    # ------------------------------------------------------------------ #
    # Sensibilidad (barra «Sensibilidad» del panel = parámetro `conf`)
    # ------------------------------------------------------------------ #
    @property
    def conf(self) -> float:
        return self._conf

    @conf.setter
    def conf(self, value) -> None:
        """Confianza mínima para aceptar que algo es una PERSONA (detector
        COCO). Más baja = más sensible (personas lejanas o tapadas, algo más
        de falsas detecciones); más alta = más estricta. Antes este valor se
        guardaba pero ninguna etapa lo usaba."""
        v = min(0.95, max(0.05, float(value)))
        self._conf = v
        coco = self.__dict__.get("coco")
        if coco is not None:
            coco.person_conf = v

    @property
    def names(self) -> dict[int, str]:
        return self.posture.names

    def reset(self) -> None:
        """Limpia las caches de fotograma (cada imagen subida es otra escena)."""
        self._last_people = None
        self._last_stairs = None
        self._last_obstacles = []
        self._next_due = {}
        self._last_class = None
        self._last_skeletons = None
        self._fall_held.clear()
        if self.skeleton is not None:
            self.skeleton.reset()

    # ------------------------------------------------------------------ #
    # Planificador de etapas pesadas
    # ------------------------------------------------------------------ #
    # Prioridad a igualdad de retraso: personas (todo depende de ellas),
    # esqueleto (equilibrio y señales), escalera, obstáculos.
    STAGES = ("people", "skeleton", "stairs", "obstacles")

    def _schedule(self, n: int, force: bool = False) -> set[str]:
        periods = {"people": self.people_every, "skeleton": self.skeleton_every,
                   "stairs": self.stairs_every, "obstacles": self.obstacle_every}
        active = [k for k in self.STAGES
                  if k != "skeleton" or self.skeleton is not None]
        if force:
            for k in active:
                self._next_due[k] = n + max(1, periods[k])
            return set(active)
        due = [k for k in active if self._next_due.get(k, 0) <= n]
        if not due:
            return set()
        # la más atrasada primero; a igualdad, por prioridad (orden de STAGES)
        k = min(due, key=lambda s: (self._next_due.get(s, 0), self.STAGES.index(s)))
        self._next_due[k] = n + max(1, periods[k])
        return {k}

    def set_zone(self, zone) -> None:
        """Nueva zona desde el panel (cambio de referencia: seguro entre hilos)."""
        self.zone = zone

    # ------------------------------------------------------------------ #
    # Señales por esqueleto -> caja de su persona
    # ------------------------------------------------------------------ #
    def _skeleton_signals(self, skeletons: list, dets: list, phones: list,
                          stairs_present: bool, w: int = 0, h: int = 0) -> list[dict]:
        from .skeleton.signals import skeleton_signals
        out: list[dict] = []
        people = [d for d in dets if d.get("class_id") != 3]
        zone = getattr(self, "zone", None)
        rails = zone.rail_segments(w, h) if zone is not None and w else []
        for sk in skeletons:
            # con zona dibujada, solo cuentan las personas en la escalera
            if zone is not None and w and not zone.foot_inside(sk["bbox"], w, h):
                continue
            sigs = skeleton_signals(sk, self.signals_cfg, phones, stairs_present,
                                    rails=rails)
            if not sigs:
                continue
            sigs = [{**s_, **RISK_CATALOG[s_["tipo"]]} for s_ in sigs]
            out.extend(sigs)
            best = max(people, key=lambda d: iou(d["bbox"], sk["bbox"]),
                       default=None)
            if best is None or iou(best["bbox"], sk["bbox"]) < 0.3:
                continue
            current = best.get("riesgo", "NEUTRO")
            best["senales"] = sorted(set(best.get("senales", []))
                                     | {s_["tipo"] for s_ in sigs})
            best["riesgo"] = max([current] + [s_["riesgo"] for s_ in sigs],
                                 key=lambda r: RISK_ORDER.get(r, 0))
            if best["riesgo"] != current:
                best["riesgo_por_clase"] = current
        return out

    @property
    def has_pending_evidence(self) -> bool:
        return self.alerts.has_pending

    def flush_evidence(self, annotated: np.ndarray) -> None:
        self.alerts.flush_evidence(annotated, people=self._last_people or [])

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
    # Esqueletos: pérdida de equilibrio -> caja de persona -> alerta
    # ------------------------------------------------------------------ #
    def _link_balance(self, skeletons: list, dets: list, signals: list) -> None:
        """Lleva cada pérdida de equilibrio del esqueleto a la caja de persona
        que mejor solapa (la que sigue el IoUTracker y usa AlertManager). Si
        COCO no vio a esa persona en este fotograma, se añade su caja."""
        for sk in skeletons:
            eq = sk.get("equilibrio") or {}
            if eq.get("estado") != "perdida_equilibrio":
                continue
            people = [d for d in dets if d.get("class_id") != 3]
            best = max(people, key=lambda d: iou(d["bbox"], sk["bbox"]),
                       default=None)
            if best is None or iou(best["bbox"], sk["bbox"]) < 0.3:
                best = {"class_id": 4, "class_name": "Persona",
                        "confidence": 0.0, "origen": "esqueleto",
                        "bbox": [int(v) for v in sk["bbox"]], "track_id": None,
                        "riesgo": "NEUTRO", "ref": ""}
                dets.append(best)
                self.tracker.update([best])
            reason = ", ".join(eq.get("criterios", [])) or "tronco extremo"
            if best.get("risk_type") != "caida":
                best["risk_type"] = BALANCE_RISK
                best["riesgo"] = "ALTO"
                best["ref"] = f"esqueleto {sk['id']}: {reason}"
                best["confidence"] = max(best["confidence"], eq.get("score", 0.0))
            best["esqueleto_id"] = sk["id"]
            meta = RISK_CATALOG[BALANCE_RISK]
            signals.append({"tipo": BALANCE_RISK, "conf": eq.get("score", 0.0),
                            "riesgo": meta["riesgo"], "ref": meta["ref"],
                            "desc": meta["desc"],
                            "detalle": f"persona {sk['id']}: {reason}"})

    # ------------------------------------------------------------------ #
    # Esqueletos: seguro de postura por reglas (independiente del modelo)
    # ------------------------------------------------------------------ #
    def _link_posture(self, skeletons: list, dets: list, signals: list) -> None:
        """Aplica el veredicto geométrico de cada esqueleto a la caja de
        persona que mejor solapa (skeleton/posture_rules.py):

        * `caido`: la caja pasa a «Persona caída», ALTO, aunque el modelo
          dijera otra cosa. Si COCO no vio a esa persona, se añade su caja.
        * `postura_incorrecta`: una caja de riesgo BAJO/NEUTRO sube a MEDIO.
        * `erguido` (claramente de pie, todos los puntos clave visibles): una
          caída del recorte del modelo o de la geometría de MediaPipe pasa a
          «de pie» y no genera alerta (el «desequilibrio» de MediaPipe solo si
          el monitor de equilibrio del esqueleto también la ve estable).
        """
        for sk in skeletons:
            pst = sk.get("postura") or {}
            estado = pst.get("estado")
            if estado not in ("caido", "postura_incorrecta", "erguido"):
                continue
            people = [d for d in dets if d.get("class_id") != 3]
            best = max(people, key=lambda d: iou(d["bbox"], sk["bbox"]),
                       default=None)
            matched = best is not None and iou(best["bbox"], sk["bbox"]) >= 0.3
            crit = ", ".join(pst.get("criterios", []))

            if estado == "erguido":
                # Veto: el esqueleto ve a la persona claramente de pie (todos
                # los puntos clave a su altura normal) y la caja dice caída.
                #  * recorte del modelo: siempre;
                #  * geometría de MediaPipe: describe UNA persona y en escenas
                #    con gente se empareja con la caja equivocada (medido con
                #    peatones de MOT15: 168 de 192 falsas caídas venían de
                #    ahí). Su «desequilibrio» solo se veta si el monitor de
                #    equilibrio del esqueleto también la ve estable.
                if not matched:
                    continue
                origin = str(best.get("origen", ""))
                name = best.get("class_name")
                eq = (sk.get("equilibrio") or {}).get("estado")
                veto = (name == "persona_caido"
                        and (origin.startswith("modelo")
                             or origin == "geometria de pose")) or \
                       (name == "persona_desequilibrio"
                        and origin == "geometria de pose" and eq == "estable")
                if veto:
                    conf_m = best.get("confidence", 0.0)
                    best.pop("_fall_pending", None)
                    best.pop("risk_type", None)       # no genera alerta
                    # se conservan las señales que no son de caída
                    # (distracción, pasamanos) que ya subieron su riesgo
                    keep = [s_["riesgo"] for s_ in signals
                            if s_.get("tipo") in best.get("senales", [])
                            and s_.get("tipo") != "tambaleo"]
                    best.update(class_id=2, class_name="persona_erguida",
                                riesgo=max(["BAJO"] + keep,
                                           key=lambda r: RISK_ORDER.get(r, 0)),
                                clase_modelo=name, veto_esqueleto=True,
                                ref=(f"esqueleto {sk['id']} de pie ({origin} "
                                     f"veía {name.replace('persona_', '')} "
                                     f"{conf_m:.2f})"))
                continue

            if not matched:
                if estado != "caido":
                    continue              # sin caja: solo la caída se añade
                best = {"class_id": 4, "class_name": "persona",
                        "confidence": 0.0, "origen": "esqueleto",
                        "bbox": [int(v) for v in sk["bbox"]], "track_id": None,
                        "riesgo": "NEUTRO", "ref": ""}
                dets.append(best)
                self.tracker.update([best])
            best["esqueleto_id"] = sk["id"]

            if estado == "caido":
                meta = RISK_CATALOG["caida_esqueleto"]
                if best.get("class_name") != "persona_caido":
                    best["clase_modelo"] = best.get("class_name")
                best.update(class_id=0, class_name="persona_caido",
                            risk_type="caida", riesgo="ALTO",
                            ref=f"esqueleto {sk['id']}: {crit}",
                            confidence=max(best.get("confidence", 0.0), 0.6))
                best.pop("_fall_pending", None)   # ya confirmada por reglas
            else:
                if RISK_ORDER.get(best.get("riesgo", "NEUTRO"), 0) >= RISK_ORDER["MEDIO"]:
                    continue                  # ya tiene un riesgo igual o mayor
                meta = RISK_CATALOG["postura_esqueleto"]
                best.update(risk_type="postura_no_erguida", riesgo="MEDIO",
                            ref=f"esqueleto {sk['id']}: {crit}")
            signals.append({"tipo": "caida_esqueleto" if estado == "caido"
                            else "postura_esqueleto",
                            "conf": 0.6, "riesgo": meta["riesgo"],
                            "ref": meta["ref"], "desc": meta["desc"],
                            "detalle": f"persona {sk['id']}: {crit}"})

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

        # --- planificador: como mucho UNA etapa pesada por fotograma -------
        # Antes cada etapa corría a su cadencia (n % every == 0) y, cuando
        # coincidían varias en el mismo fotograma, se sumaban: picos de
        # 100-130 ms (p95) frente a ~20 ms de mediana. Ahora, de las etapas
        # que tocan, se ejecuta la más atrasada (a igualdad, por prioridad) y
        # el resto espera al fotograma siguiente. Una etapa que nunca se ha
        # ejecutado (`None`) toca ya. Con `force` (imagen fija) se ejecuta todo.
        run = self._schedule(n, force)

        # --- 1. personas (COCO) ------------------------------------------
        t = time.perf_counter()
        if "people" in run:
            self._last_people = self.coco.people(frame)
        people = self._last_people or []
        stages["people"] = round((time.perf_counter() - t) * 1000, 2)

        # --- 2. escalera (contexto, cadencia lenta) -----------------------
        t = time.perf_counter()
        if "stairs" in run:
            self._last_stairs = self.posture.stairs(frame, self.max_stairs_boxes)
        stairs = self._last_stairs or []
        stages["stairs"] = round((time.perf_counter() - t) * 1000, 2)

        # --- 3. esqueletos YOLOv8-Pose (varias personas, con ID) ----------
        # Dan, por persona: equilibrio, seguro de postura y las señales de
        # pasamanos / distracción (antes MediaPipe, que solo veía a UNA).
        t = time.perf_counter()
        if self.skeleton is not None and "skeleton" in run:
            try:
                self._last_skeletons = self.skeleton.step(frame)
            except Exception as exc:                        # noqa: BLE001
                self._fail("pose", exc)
        skeletons = self._last_skeletons or []
        stages["skeleton"] = round((time.perf_counter() - t) * 1000, 2)

        # --- 4. obstáculos (RQF02 1) --------------------------------------
        t = time.perf_counter()
        if "obstacles" in run:
            try:
                self._last_obstacles = self.coco.obstacles(frame)
            except Exception as exc:                        # noqa: BLE001
                self._fail("deteccion", exc)
        obstacles = self._last_obstacles
        stages["obstacles"] = round((time.perf_counter() - t) * 1000, 2)

        # --- 5. postura de cada persona (clasificador del Modelo v5) ------
        # Se clasifica cuando se refrescan las cajas de personas (entre medias
        # las cajas no cambian y la caché por solape acierta siempre). El
        # seguro de postura del esqueleto corrige después (_link_posture).
        run_class = "people" in run or not self._last_class
        t = time.perf_counter()
        need = list(range(len(people)))
        by_idx: dict[int, tuple] = {}
        if need:
            if run_class:
                preds = self.posture.classify_many(frame, [people[i] for i in need])
                by_idx = {i: (*p, "modelo") for i, p in zip(need, preds)}
                self._last_class = [(people[i], *by_idx[i][:2]) for i in need]
            else:
                fresh = []
                for i in need:
                    prev = max(self._last_class or [],
                               key=lambda c: iou(c[0], people[i]), default=None)
                    if prev is not None and iou(prev[0], people[i]) > 0.3:
                        by_idx[i] = (prev[1], prev[2], "modelo (cache)")
                    else:
                        fresh.append(i)          # persona nueva: se clasifica ya
                if fresh:
                    preds = self.posture.classify_many(frame,
                                                       [people[i] for i in fresh])
                    for i, p in zip(fresh, preds):
                        by_idx[i] = (*p, "modelo")
                        self._last_class = (self._last_class or []) + [(people[i], *p)]
        dets: list[dict] = []
        for i, box in enumerate(people):
            is_pose = False
            name, conf_p, origin = by_idx.get(i, ("", 0.0, "modelo"))
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

        # --- 5b. señales por esqueleto (pasamanos, distracción) ------------
        # Cada señal va a la caja de SU persona (la que más solapa con su
        # esqueleto), que toma el mayor riesgo entre su clase y sus señales.
        H, W = frame.shape[:2]
        zone = self.zone
        if zone.has_stairs:
            # objetos: solo es obstáculo lo que está DENTRO de la escalera
            obstacles = [o if o.get("tipo") != "obstaculo_escalon"
                         or zone.foot_inside(o["bbox"], W, H)
                         else {**o, "tipo": "elemento_escena", "riesgo": "CONTEXTO",
                               "fuera_zona": True}
                         for o in obstacles]
        phones = [o for o in obstacles if o.get("tipo") == "telefono"]
        pose_signals = self._skeleton_signals(
            skeletons, dets, phones,
            stairs_present=bool(stairs) or zone.has_stairs, w=W, h=H)

        # --- 6. seguimiento + confirmación de caídas ----------------------
        self.tracker.update(dets)
        # El seguro del esqueleto va ANTES de confirmar caídas: puede vetar
        # una caída del modelo (persona claramente de pie) o imponerla.
        self._link_posture(skeletons, dets, pose_signals)
        self._confirm_falls(dets)
        self._link_balance(skeletons, dets, pose_signals)
        if zone.has_stairs:
            # Personas fuera de la escalera dibujada: se ven, pero no generan
            # riesgos ni alertas (gente que pasa por detrás, otra sala...).
            for d in dets:
                if d.get("class_id") == 3 or zone.foot_inside(d["bbox"], W, H):
                    continue
                d.pop("risk_type", None)
                d.pop("_fall_pending", None)
                d.update(riesgo="NEUTRO", ref="fuera de la escalera",
                         fuera_zona=True)
        for conf_s, box in stairs:
            dets.append({
                "class_id": 3, "class_name": "escalera",
                "confidence": round(conf_s, 4), "bbox": box,
                "track_id": None, "riesgo": "CONTEXTO", "ref": "",
            })
        confirmed, preliminary = self.apply_thresholds(dets)

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
        # Objetos en la escalera: MEDIO (riesgo a vigilar), nunca ALTO por sí
        # solos; el rojo queda para caídas y pérdida de equilibrio.
        risks += ["MEDIO"] * sum(1 for o in obstacles
                                 if o.get("tipo") == "obstaculo_escalon")
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
            "esqueletos": skeletons,
            "obstaculos": obstacles,
            "alertas": self.alerts.recent(5),
            "riesgo": level,
            "tracks": len(self.tracker.tracks),
            "ms": round(ms, 1),
            "stages": stages,
            "sistema": system_status,
        }
