"""Zona de la escalera y líneas del pasamanos, dibujadas en el panel.

El usuario marca sobre la imagen de la cámara:

* `escalera`: polígono (coordenadas 0..1 de la imagen) con el tramo de
  escalera y sus descansos. Con él:
    - una persona cuyos pies (centro del borde inferior de su caja) quedan
      FUERA no genera riesgos ni alertas (se sigue viendo, como contexto);
    - un objeto solo es «obstáculo en la escalera» si está dentro;
    - la escena «tiene escalera» aunque el detector no la vea.
* `pasamanos`: una o varias líneas (polilíneas 0..1). Con ellas, «sin
  pasamanos» deja de ser una aproximación: una mano está en el pasamanos si
  queda a menos de `rail_touch_torso` torsos de la línea.

Se guarda en `configs/zona_local.json` (propia de cada instalación, no se
versiona). Sin zona, el sistema funciona como antes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MAX_POINTS = 64          # por polígono / línea
MAX_RAILS = 8


def _clean_points(pts, minimum: int) -> list[list[float]]:
    if not isinstance(pts, list) or len(pts) > MAX_POINTS:
        raise ValueError("lista de puntos no válida")
    out = []
    for p in pts:
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)):
            raise ValueError("cada punto debe ser [x, y] numérico")
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("las coordenadas van de 0 a 1")
        out.append([round(x, 4), round(y, 4)])
    if out and len(out) < minimum:
        raise ValueError(f"hacen falta al menos {minimum} puntos")
    return out


class Zone:
    def __init__(self, escalera=None, pasamanos=None):
        self.escalera: list[list[float]] = escalera or []
        self.pasamanos: list[list[list[float]]] = pasamanos or []

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data) -> "Zone":
        """Valida lo que llega del panel (lanza ValueError si no es válido)."""
        if not isinstance(data, dict):
            raise ValueError("se esperaba un objeto JSON")
        esc = _clean_points(data.get("escalera") or [], 3)
        rails = data.get("pasamanos") or []
        if not isinstance(rails, list) or len(rails) > MAX_RAILS:
            raise ValueError("pasamanos no válido")
        rails = [r for r in (_clean_points(r, 2) for r in rails) if r]
        return cls(esc, rails)

    def to_dict(self) -> dict:
        return {"escalera": self.escalera, "pasamanos": self.pasamanos}

    @property
    def has_stairs(self) -> bool:
        return len(self.escalera) >= 3

    @property
    def has_rails(self) -> bool:
        return bool(self.pasamanos)

    @classmethod
    def load(cls, path: Path) -> "Zone":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------ #
    def contains(self, x: float, y: float, w: int, h: int) -> bool:
        """¿El punto (píxeles) está dentro de la escalera? Sin zona: sí."""
        if not self.has_stairs:
            return True
        px, py = x / w, y / h
        inside = False
        pts = self.escalera
        j = len(pts) - 1
        for i in range(len(pts)):          # regla par-impar (ray casting)
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > py) != (yj > py) and \
                    px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi:
                inside = not inside
            j = i
        return inside

    def foot_inside(self, bbox, w: int, h: int) -> bool:
        """Pies de una caja [x1, y1, x2, y2] (centro del borde inferior)."""
        x1, _y1, x2, y2 = bbox
        return self.contains((x1 + x2) / 2, y2, w, h)

    def rail_segments(self, w: int, h: int) -> list[tuple]:
        """Segmentos del pasamanos en píxeles: [((x1, y1), (x2, y2)), ...]."""
        segs = []
        for line in self.pasamanos:
            for a, b in zip(line, line[1:]):
                segs.append(((a[0] * w, a[1] * h), (b[0] * w, b[1] * h)))
        return segs


def point_segment_distance(p, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))
