"""Seguro de postura por reglas del esqueleto (sin GPU, esqueletos sintéticos).

Coordenadas normalizadas por la altura de la imagen, y hacia abajo. Torso
(hombros-caderas) = 0.1, así 1 torso = 0.1 unidades.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import PostureRulesCfg  # noqa: E402
from backend.perception.skeleton.posture_rules import PostureRules  # noqa: E402

T = 0.1   # un torso


def skel(nose, sh, el, wr, hip, kn, an):
    """17 puntos COCO a partir de pares (izq, der) por articulación."""
    x = np.zeros((17, 2))
    x[0] = nose
    x[1] = x[3] = (nose[0] - 0.01, nose[1] - 0.005)
    x[2] = x[4] = (nose[0] + 0.01, nose[1] - 0.005)
    for (i, j), (a, b) in (((5, 6), sh), ((7, 8), el), ((9, 10), wr),
                           ((11, 12), hip), ((13, 14), kn), ((15, 16), an)):
        x[i], x[j] = a, b
    return x


def standing():
    # de pie: cabeza 1.4 torsos sobre la cadera, cadera 1.65 sobre tobillos
    return skel(nose=(0.50, 0.26), sh=((0.45, 0.30), (0.55, 0.30)),
                el=((0.44, 0.36), (0.56, 0.36)), wr=((0.44, 0.42), (0.56, 0.42)),
                hip=((0.47, 0.40), (0.53, 0.40)), kn=((0.47, 0.48), (0.53, 0.48)),
                an=((0.47, 0.565), (0.53, 0.565)))


def lying():
    # el mismo cuerpo girado 90° alrededor de la cadera: tendido en el suelo
    x = standing()
    c = np.array([0.50, 0.40])
    rot = np.array([[0, -1], [1, 0]])
    return (x - c) @ rot.T + c


def bending():
    # agachado a recoger algo: piernas rectas, tronco horizontal, manos al suelo
    return skel(nose=(0.64, 0.42), sh=((0.58, 0.38), (0.60, 0.40)),
                el=((0.60, 0.47), (0.62, 0.48)), wr=((0.60, 0.55), (0.62, 0.555)),
                hip=((0.48, 0.40), (0.50, 0.40)), kn=((0.48, 0.48), (0.50, 0.48)),
                an=((0.48, 0.565), (0.50, 0.565)))


def sitting_step():
    # sentado en un escalón: cadera a la altura de las rodillas, rodillas a 90°
    return skel(nose=(0.50, 0.26), sh=((0.45, 0.30), (0.55, 0.30)),
                el=((0.44, 0.36), (0.56, 0.36)), wr=((0.52, 0.40), (0.60, 0.40)),
                hip=((0.47, 0.40), (0.53, 0.40)), kn=((0.57, 0.41), (0.63, 0.41)),
                an=((0.57, 0.49), (0.63, 0.49)))


def climbing():
    # subiendo un escalón: UNA rodilla levantada y doblada, la otra recta
    return skel(nose=(0.50, 0.26), sh=((0.45, 0.30), (0.55, 0.30)),
                el=((0.44, 0.36), (0.56, 0.36)), wr=((0.44, 0.42), (0.56, 0.42)),
                hip=((0.47, 0.40), (0.53, 0.40)), kn=((0.50, 0.44), (0.53, 0.48)),
                an=((0.49, 0.52), (0.53, 0.565)))


ALL = np.ones(17, bool)


class PostureRulesTest(unittest.TestCase):
    def setUp(self):
        self.r = PostureRules(PostureRulesCfg())

    def ev(self, x, valid=ALL, t=10.0, tid=1):
        return self.r.evaluate(tid, x, valid, T, None, t)

    def confirmed(self, x, valid=ALL, tid=1):
        self.ev(x, valid, t=0.0, tid=tid)
        return self.ev(x, valid, t=5.0, tid=tid)       # sostenido 5 s

    def test_standing_is_upright(self):
        out = self.confirmed(standing())
        self.assertEqual(out["estado"], "erguido")
        self.assertEqual(out["criterios_caida"], [])
        self.assertEqual(out["criterios_postura"], [])

    def test_lying_is_fall_with_hands_at_feet_height(self):
        out = self.confirmed(lying())
        self.assertEqual(out["estado"], "caido")
        for crit in ("manos_a_pies", "cabeza_a_cadera", "cuerpo_tendido",
                     "eje_horizontal"):
            self.assertIn(crit, out["criterios"])

    def test_bending_to_pick_up_is_not_a_fall(self):
        # manos a los pies y cabeza a la cadera, pero el cuerpo no está tendido
        out = self.confirmed(bending())
        self.assertNotEqual(out["estado"], "caido")
        self.assertIn("manos_a_pies", out["criterios_caida"])
        self.assertNotIn("cuerpo_tendido", out["criterios_caida"])

    def test_sitting_on_a_step_is_bad_posture(self):
        out = self.confirmed(sitting_step())
        self.assertEqual(out["estado"], "postura_incorrecta")
        self.assertIn("cadera_a_rodillas", out["criterios"])
        self.assertIn("rodillas_dobladas", out["criterios"])

    def test_climbing_one_bent_knee_is_still_upright(self):
        out = self.confirmed(climbing())
        self.assertEqual(out["estado"], "erguido")
        self.assertNotIn("rodillas_dobladas", out["criterios_postura"])

    def test_hidden_ankles_do_not_vote(self):
        # pasamanos tapando los tobillos: solo queda cadera_a_rodillas (1 voto)
        valid = ALL.copy()
        valid[[15, 16]] = False
        out = self.confirmed(sitting_step(), valid)
        self.assertEqual(out["criterios_postura"], ["cadera_a_rodillas"])
        self.assertNotEqual(out["estado"], "postura_incorrecta")

    def test_fall_needs_persistence(self):
        first = self.ev(lying(), t=0.0)
        self.assertEqual(first["estado"], "indeterminado")
        self.assertEqual(first["pendiente"], "caido")
        later = self.ev(lying(), t=0.7)
        self.assertEqual(later["estado"], "caido")

    def test_state_change_restarts_persistence(self):
        self.confirmed(standing())
        out = self.ev(lying(), t=5.1)                  # acaba de caer
        self.assertEqual(out["pendiente"], "caido")

    def test_perspective_scale_independent(self):
        # la misma persona a mitad de tamaño (más lejos): mismo veredicto
        x = lying()
        c = x.mean(axis=0)
        small = (x - c) * 0.5 + c
        self.r.evaluate(2, small, ALL, T * 0.5, None, 0.0)
        out = self.r.evaluate(2, small, ALL, T * 0.5, None, 5.0)
        self.assertEqual(out["estado"], "caido")


class LinkPostureTest(unittest.TestCase):
    """Fusión esqueleto <-> modelo en RiskEngine._link_posture."""

    def setUp(self):
        from backend.perception.engine import RiskEngine
        from backend.perception.tracking import IoUTracker
        self.eng = RiskEngine.__new__(RiskEngine)
        self.eng.tracker = IoUTracker()

    @staticmethod
    def det(name, origin="modelo", riesgo="BAJO", pending=False):
        d = {"class_id": {"persona_caido": 0, "persona_erguida": 2}.get(name, 4),
             "class_name": name, "confidence": 0.7, "origen": origin,
             "bbox": [100, 100, 200, 400], "track_id": 1, "riesgo": riesgo,
             "ref": ""}
        if pending:
            d["_fall_pending"] = True
            d["risk_type"] = "caida"
        return d

    @staticmethod
    def sk(estado, criterios=("manos_a_pies", "cuerpo_tendido")):
        return {"id": 7, "bbox": [102, 98, 198, 402],
                "postura": {"estado": estado, "criterios": list(criterios)}}

    def test_skeleton_fall_overrides_model_upright(self):
        d = self.det("persona_erguida")
        signals = []
        self.eng._link_posture([self.sk("caido")], [d], signals)
        self.assertEqual(d["class_name"], "persona_caido")
        self.assertEqual(d["riesgo"], "ALTO")
        self.assertEqual(d["risk_type"], "caida")
        self.assertEqual(d["clase_modelo"], "persona_erguida")
        self.assertEqual(signals[0]["tipo"], "caida_esqueleto")

    def test_upright_skeleton_vetoes_model_crop_fall(self):
        for pending, origin in ((True, "modelo"), (False, "modelo (cache)")):
            d = self.det("persona_caido", origin=origin, riesgo="MEDIO",
                         pending=pending)
            self.eng._link_posture([self.sk("erguido", ())], [d], [])
            self.assertNotIn("_fall_pending", d)    # ya no escala a ALTO
            self.assertNotIn("risk_type", d)        # ni genera alerta
            self.assertEqual(d["class_name"], "persona_erguida")
            self.assertEqual(d["riesgo"], "BAJO")
            self.assertEqual(d["clase_modelo"], "persona_caido")
            self.assertTrue(d["veto_esqueleto"])

    def test_upright_skeleton_vetoes_mediapipe_fall_keeping_other_signals(self):
        # MediaPipe empareja su única pose con la caja equivocada en multitudes
        d = self.det("persona_caido", origin="geometria de pose", riesgo="ALTO")
        d["senales"] = ["distraccion", "tambaleo"]
        signals = [{"tipo": "distraccion", "riesgo": "MEDIO"},
                   {"tipo": "tambaleo", "riesgo": "ALTO"}]
        self.eng._link_posture([self.sk("erguido", ())], [d], signals)
        self.assertEqual(d["class_name"], "persona_erguida")
        self.assertEqual(d["riesgo"], "MEDIO")      # queda la distracción
        self.assertTrue(d["veto_esqueleto"])

    def test_mediapipe_imbalance_vetoed_only_if_skeleton_balance_stable(self):
        sk = self.sk("erguido", ())
        sk["equilibrio"] = {"estado": "inestable"}
        d = self.det("persona_desequilibrio", origin="geometria de pose",
                     riesgo="ALTO")
        self.eng._link_posture([sk], [d], [])
        self.assertEqual(d["class_name"], "persona_desequilibrio")   # sin veto
        sk["equilibrio"] = {"estado": "estable"}
        self.eng._link_posture([sk], [d], [])
        self.assertEqual(d["class_name"], "persona_erguida")

    def test_bad_posture_raises_low_risk_only(self):
        d = self.det("persona_erguida", riesgo="BAJO")
        self.eng._link_posture([self.sk("postura_incorrecta", ("cadera_a_rodillas",
                                                               "rodillas_dobladas"))],
                               [d], [])
        self.assertEqual(d["riesgo"], "MEDIO")
        self.assertEqual(d["risk_type"], "postura_no_erguida")
        high = self.det("persona_caido", riesgo="ALTO")
        self.eng._link_posture([self.sk("postura_incorrecta")], [high], [])
        self.assertEqual(high["riesgo"], "ALTO")      # no baja un riesgo mayor

    def test_skeleton_fall_without_box_adds_one(self):
        dets = []
        self.eng._link_posture([self.sk("caido")], dets, [])
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["class_name"], "persona_caido")
        self.assertEqual(dets[0]["origen"], "esqueleto")


if __name__ == "__main__":
    unittest.main()
