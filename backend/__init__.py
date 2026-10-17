"""Backend del sistema de prevención de caídas en escaleras (UPAO).

Paquetes:
    api/          rutas HTTP (Flask)
    services/     hilos de trabajo: inferencia, SR en vivo, E/S de vídeo
    perception/   motor de riesgo: detectores, pose, seguimiento, alertas
    enhancement/  superresolución Real-ESRGAN x4
    privacy/      anonimización de rostros (Ley N.º 29733)
    rendering/    dibujo y codificación de resultados
"""
__version__ = "2.0.0"

# Lambda Stack instala TensorFlow a nivel de sistema y mediapipe lo importa.
# Si TensorFlow carga primero su NCCL, torch falla al arrancar
# ("undefined symbol: ncclCommShrink"). Importar torch antes que nada lo evita.
import os as _os

# Silencia los mensajes de arranque de TensorFlow/absl (no se usan).
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("GLOG_minloglevel", "2")

try:
    import torch  # noqa: F401
except ImportError:          # tools que solo leen la configuración
    pass
