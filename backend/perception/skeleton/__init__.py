"""Esqueletos con YOLOv8-Pose: 17 keypoints por persona con identidad,
suavizado, cinemática (tronco, cadera, cabeza, centro de masa vs pies) y
alerta de pérdida de equilibrio. Ver analyzer.py para el flujo completo."""
from .analyzer import SkeletonAnalyzer

__all__ = ["SkeletonAnalyzer"]
