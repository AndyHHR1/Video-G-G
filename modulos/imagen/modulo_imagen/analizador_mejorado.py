"""Integración opcional, sin modificar analizador.py ni Video."""
from .analizador import analizar_imagen
from .preprocesamiento import mejorar_imagen


def analizar_imagen_con_mejora(frame, *, modo="adaptativo"):
    mejorada, proceso = mejorar_imagen(frame, modo=modo)
    resultado, anotada = analizar_imagen(mejorada)
    resultado["preprocesamiento"] = proceso
    resultado["calidad_imagen_original"] = proceso["antes"]
    # Coordenadas siguen referidas al tamaño original (no se redimensiona).
    return resultado, anotada
