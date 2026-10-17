"""Persistencia temporal del riesgo y emisión de alertas (RQF06, RQF07).

Una condición de riesgo solo se convierte en alerta cuando:

1. su confianza supera la puerta de alerta (`thresholds.alert`),
2. persiste el tiempo exigido por su severidad (ALTO 1.2 s, MEDIO 0.6 s),
   con una histéresis (`grace_seconds`) para que un parpadeo del clasificador
   no reinicie la cuenta,
3. no se ha emitido otra alerta del mismo tipo para la misma persona en los
   últimos `cooldown_seconds`.

La evidencia (fotograma ANOTADO y anonimizado) y la línea de auditoría
(`alertas.jsonl`) se escriben juntas en `flush_evidence()`, después de dibujar
las cajas, para que la imagen y el registro nunca se desincronicen (RQNF15,
RQNF16, RQNF18).
"""
from __future__ import annotations

import datetime
import json
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .catalog import RISK_CATALOG
from .tracking import IoUTracker


@dataclass
class Alert:
    """Alerta estructurada (RQF07) con su evidencia (RQNF18)."""
    alert_id: str
    timestamp: str
    risk_type: str
    risk_ref: str
    severity: str
    description: str
    track_id: int | None
    confidence: float
    bbox: list
    evidence: str | None
    duration_s: float
    system_status: str = "OPERATIVO"

    def to_dict(self) -> dict:
        return asdict(self)


class AlertManager:
    def __init__(self, tracker: IoUTracker, alert_dir: Path, *,
                 persist_seconds: dict, persist_default: float,
                 grace_seconds: float, cooldown_seconds: float,
                 keep_days: int = 7, max_images: int = 200,
                 anonymizer=None):
        self.tracker = tracker
        self.dir = Path(alert_dir)
        self.persist_seconds = persist_seconds
        self.persist_default = persist_default
        self.grace = grace_seconds
        self.cooldown = cooldown_seconds
        self.keep_days = keep_days
        self.max_images = max_images
        self.anonymizer = anonymizer
        self.history: deque = deque(maxlen=200)
        self._seq = 0
        # Alertas disparadas en este fotograma, a la espera de su imagen.
        self._pending: list[tuple[Alert, float]] = []

    # ------------------------------------------------------------------ #
    # Persistencia
    # ------------------------------------------------------------------ #
    def forget_stale(self, active: set[tuple[int, str]]) -> None:
        """Olvida los riesgos que llevan más de `grace` segundos sin verse."""
        now = time.time()
        for tid, st in self.tracker.tracks.items():
            for risk in list(st.risk_since):
                if (tid, risk) in active:
                    st.risk_visto[risk] = now
                    continue
                if now - st.risk_visto.get(risk, now) > self.grace:
                    del st.risk_since[risk]
                    st.risk_seen.discard((tid, risk))
                    st.risk_visto.pop(risk, None)

    def observe(self, tid: int, risk: str, det: dict) -> None:
        """Registra que el riesgo está presente y evalúa si ya alerta."""
        st = self.tracker.tracks.get(tid)
        if st is None:
            return
        now = time.time()
        if (tid, risk) not in st.risk_seen:
            st.risk_since[risk] = now
            st.risk_seen.add((tid, risk))
        st.risk_visto[risk] = now
        self._evaluate(tid, risk, det)

    def evaluate_survivors(self, dets: list[dict]) -> None:
        """Evalúa también los riesgos que la histéresis mantiene vivos aunque
        este fotograma no los haya detectado (el clasificador parpadea)."""
        by_id = {d.get("track_id"): d for d in dets if "risk_type" in d}
        for tid, st in self.tracker.tracks.items():
            for risk in list(st.risk_since):
                if st.alerts.get(risk):
                    continue
                det = by_id.get(tid)
                if det is not None and det.get("risk_type") == risk:
                    continue                     # ya lo evaluó observe()
                if det is None:
                    det = {"confidence": 0.0, "bbox": list(st.bbox)}
                self._evaluate(tid, risk, det)

    def _evaluate(self, tid: int, risk: str, det: dict) -> None:
        now = time.time()
        st = self.tracker.tracks.get(tid)
        if st is None or risk not in st.risk_since:
            return
        dur = now - st.risk_since[risk]
        meta = RISK_CATALOG.get(risk, {})
        severity = meta.get("riesgo", "MEDIO")
        required = self.persist_seconds.get(severity, self.persist_default)
        if dur < required or now - st.alerts.get(risk, 0.0) <= self.cooldown:
            return
        st.alerts[risk] = now
        self._seq += 1
        alert = Alert(
            alert_id=f"ALT-{self._seq:06d}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            risk_type=risk,
            risk_ref=meta.get("ref", ""),
            severity=severity,
            description=meta.get("desc", ""),
            track_id=tid,
            confidence=round(det.get("confidence", 0.0), 4),
            bbox=det["bbox"],
            evidence=None,                    # se completa en flush_evidence
            duration_s=round(dur, 2),
        )
        self.history.append(alert)
        self._pending.append((alert, now))

    def recent(self, n: int = 5) -> list[dict]:
        return [a.to_dict() for a in list(self.history)[-n:]]

    # ------------------------------------------------------------------ #
    # Evidencia y auditoría
    # ------------------------------------------------------------------ #
    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def flush_evidence(self, annotated: np.ndarray,
                       people: list | None = None) -> None:
        """Guarda la imagen anotada de las alertas pendientes + su registro.
        `people` son las cajas de persona, para anonimizar sus cabezas."""
        for alert, ts in self._pending:
            try:
                alert.evidence = self._save_image(ts, annotated, people)
                self._append_audit(alert)
            except Exception as exc:                        # noqa: BLE001
                print(f"  [aviso] no se pudo guardar la evidencia: {exc}",
                      flush=True)
        self._pending.clear()

    def _save_image(self, ts: float, frame: np.ndarray,
                    people: list | None = None) -> str:
        day = datetime.date.fromtimestamp(ts).isoformat()
        d = self.dir / day
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"alerta_{int(ts * 1000)}.jpg"
        img = frame.copy()
        if self.anonymizer is not None:
            img = self.anonymizer(img, people)
        cv2.imwrite(str(path), img)
        self._prune()
        return str(path)

    def _append_audit(self, alert: Alert) -> None:
        """Registro append-only en `alertas.jsonl` (RQNF16)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "alertas.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")

    def _prune(self) -> None:
        """Conserva `keep_days` días y como mucho `max_images` imágenes."""
        if not self.dir.is_dir():
            return
        limit = datetime.date.today() - datetime.timedelta(days=self.keep_days)
        for sub in self.dir.iterdir():
            if not sub.is_dir():
                continue
            try:
                if datetime.date.fromisoformat(sub.name) < limit:
                    shutil.rmtree(sub, ignore_errors=True)
            except ValueError:
                continue
        imgs = sorted(self.dir.glob("*/*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in imgs[:max(0, len(imgs) - self.max_images)]:
            old.unlink(missing_ok=True)
