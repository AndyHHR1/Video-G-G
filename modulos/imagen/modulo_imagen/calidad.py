"""Métricas de calidad visual. No etiquetan riesgos ni diagnostican objetos."""
import cv2
import numpy as np


def validar_bgr(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
        raise ValueError('Se requiere una imagen BGR uint8 de 3 canales no vacía')


def medir_calidad(frame):
    validar_bgr(frame)
    # Medición económica en miniatura; preserva la imagen original sin cambios.
    h, w = frame.shape[:2]
    escala = min(1.0, 640.0 / max(h, w))
    mini = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1 else frame
    gris = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)
    brillo = float(gris.mean())
    contraste = float(gris.std())
    oscuras = float(np.mean(gris < 60)*100)
    sobreexp = float(np.mean(gris > 245)*100)
    # Cuantiles: dan una medida aproximada de sombras aunque el brillo global sea correcto.
    p10, p50, p90 = [float(v) for v in np.percentile(gris, (10,50,90))]
    sombras = oscuras > 22 and p90 > 125
    lap = cv2.Laplacian(gris, cv2.CV_64F)
    nitidez = float(lap.var())
    # Indicador heurístico de ruido de altas frecuencias en regiones suavizadas.
    ruido = float(np.median(np.abs(gris.astype(np.float32) - cv2.medianBlur(gris,3).astype(np.float32))))
    return {
        'luminosidad': round(brillo,2), 'contraste':round(contraste,2),
        'nitidez_laplaciana':round(nitidez,2),
        'porcentaje_zonas_oscuras':round(oscuras,2),
        'porcentaje_zonas_sobreexpuestas':round(sobreexp,2),
        'percentil_10_gris':round(p10,2), 'mediana_gris':round(p50,2),
        'percentil_90_gris':round(p90,2),
        'sombras_fuertes_estimadas':bool(sombras),
        'ruido_mad_estimado':round(ruido,2),
        'brillo_bajo_estimado':bool(brillo < 80),
        'contraste_bajo_estimado':bool(contraste < 40),
        'advertencia':'Umbrales heurísticos, no calibrados con etiquetas; nitidez y ruido dependen de escena y resolución.'
    }
