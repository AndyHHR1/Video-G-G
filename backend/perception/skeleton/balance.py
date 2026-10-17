"""Decisión de pérdida de equilibrio a partir de las medidas del esqueleto.

Criterios (umbrales en `configs/app.yaml → skeleton.balance`):

| criterio   | se activa si                                                        |
|------------|---------------------------------------------------------------------|
| tronco     | desvío respecto a la línea base de la persona >= `trunk_dev_deg`,   |
|            | o, sin línea base, ángulo absoluto >= `trunk_abs_deg`               |
| com_fuera  | centro de masa fuera de la base de apoyo > `com_margin` torsos      |
| cadera_cae | velocidad de la cadera hacia abajo >= `hip_down_speed` torsos/s     |
| cadera_acc | aceleración de la cadera >= `hip_accel` torsos/s²                   |
| cabeza_acc | aceleración de la cabeza >= `head_accel` torsos/s²                  |

Hay pérdida de equilibrio cuando se cumplen al menos `min_votes` criterios
(o el tronco supera `trunk_extreme_deg`) durante `persist_s` segundos. Un
solo criterio no basta: bajar la escalera deprisa mueve la cadera hacia abajo
y agacharse inclina el tronco, pero ninguno de los dos es un desequilibrio.
Un criterio que no se puede evaluar (tobillos tapados por el pasamanos) no
vota ni a favor ni en contra.
"""
from __future__ import annotations


class BalanceMonitor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.since: dict[int, float] = {}
        self.last_alert: dict[int, float] = {}

    def evaluate(self, tid: int, m: dict, baseline: float | None,
                 t: float) -> dict:
        c = self.cfg
        votes: list[str] = []
        evaluable = 0
        dev = None
        if m.get("tronco_deg") is not None:
            evaluable += 1
            if baseline is not None:
                dev = round(abs(m["tronco_deg"] - baseline), 1)
                if dev >= c.trunk_dev_deg:
                    votes.append("tronco")
            elif m["tronco_deg"] >= c.trunk_abs_deg:
                votes.append("tronco")
        for key, name, thr in (("com_offset", "com_fuera", c.com_margin),
                               ("cadera_baja_v", "cadera_cae", c.hip_down_speed),
                               ("cadera_a", "cadera_acc", c.hip_accel),
                               ("cabeza_a", "cabeza_acc", c.head_accel)):
            val = m.get(key)
            if val is None:
                continue
            evaluable += 1
            if (val > thr) if key == "com_offset" else (val >= thr):
                votes.append(name)

        extreme = (m.get("tronco_deg") or 0) >= c.trunk_extreme_deg \
            or (dev or 0) >= c.trunk_extreme_deg
        unstable = len(votes) >= c.min_votes or extreme
        if unstable:
            self.since.setdefault(tid, t)
        else:
            self.since.pop(tid, None)
        held = t - self.since[tid] if tid in self.since else 0.0
        state = "estable"
        alert = False
        if unstable:
            state = "inestable"
            if held >= c.persist_s:
                state = "perdida_equilibrio"
                if t - self.last_alert.get(tid, -1e9) >= c.cooldown_s:
                    self.last_alert[tid] = t
                    alert = True
        score = min(0.99, 0.5 + 0.12 * len(votes) + (0.2 if extreme else 0.0)) \
            if unstable else round(len(votes) / max(1, evaluable), 2)
        return {"estado": state, "criterios": votes, "evaluables": evaluable,
                "desvio_tronco_deg": dev, "sostenido_s": round(held, 2),
                "score": round(score, 2), "alerta": alert}

    def forget(self, tid: int) -> None:
        self.since.pop(tid, None)
        self.last_alert.pop(tid, None)
