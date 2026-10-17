"""Analiza fotogramas de clips de incidentes sin tracking ni detección de accidentes.

Video conserva la responsabilidad de capturar, identificar la escalera y elegir el
intervalo de interés. La función es reutilizable sin Flask ni una cámara abierta.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from .analisis_tiempo_real import analizar_frame
from .recuperacion_roi import recuperar_en_roi


def tiempos_muestreo(duracion_s: float, intervalo_s: float = 1.0, tiempos_interes=None):
    if duracion_s <= 0:
        raise ValueError('La duración debe ser positiva')
    if intervalo_s <= 0:
        raise ValueError('El intervalo debe ser positivo')
    if tiempos_interes is not None:
        valores = [float(t) for t in tiempos_interes]
        if any(t < 0 or t > duracion_s for t in valores):
            raise ValueError('Hay tiempos fuera del clip')
        return sorted(set(round(min(t, duracion_s), 3) for t in valores))
    salida = []
    t = 0.0
    while t < duracion_s:
        salida.append(round(t, 3))
        t += intervalo_s
    return salida or [0.0]


def analizar_clip(ruta_clip, *, modo='adaptativo', intervalo_s=1.0,
                  tiempos_interes=None, guardar_en=None, contexto_video=None,
                  max_duracion_s=10.5):
    """Devuelve (resumen JSON-serializable, fotogramas procesados).

    El clip se lee con OpenCV. `contexto_video` conserva metadatos del detector
    de escaleras de Video; nunca ejecuta RiskEngine ni su tracking.
    """
    ruta = Path(ruta_clip)
    captura = cv2.VideoCapture(str(ruta))
    if not captura.isOpened():
        raise ValueError(f'No se pudo abrir el clip: {ruta.name}')
    try:
        fps = float(captura.get(cv2.CAP_PROP_FPS))
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            raise ValueError('No se puede determinar FPS/duración del video')
        duracion = total / fps
        if duracion > max_duracion_s:
            raise ValueError(f'El clip excede {max_duracion_s:g} segundos')
        tiempos = tiempos_muestreo(duracion, intervalo_s, tiempos_interes)
        carpeta = Path(guardar_en) if guardar_en else None
        if carpeta:
            carpeta.mkdir(parents=True, exist_ok=True)
        resultados = []
        imagenes = []
        ultimo_candidato = None
        perdidas_consecutivas = 0
        inicio = time.perf_counter()
        for i, segundo in enumerate(tiempos):
            numero = min(int(round(segundo * fps)), total - 1)
            captura.set(cv2.CAP_PROP_POS_FRAMES, numero)
            ok, original = captura.read()
            if not ok or original is None:
                resultados.append({'segundo_s': segundo, 'estado': 'no_decodificado'})
                continue
            # Pueden usarse tanto clips crudos como seleccionados por Video.
            dato, anotada = analizar_frame(original, modo=modo)
            dato['segundo_s'] = round(numero / fps, 3)
            dato['fotograma_indice'] = numero
            # Si las poses y YOLO global fallan, una segunda inferencia enfocada
            # en la ultima region observada puede recuperar una persona acostada.
            actual = dato['segundo_s']
            roi_resultado = None
            if ultimo_candidato is not None and 0 < actual - ultimo_candidato['segundo_s'] <= 2.0:
                previa = ultimo_candidato['bbox']
                def coincide_area(c):
                    b = c.get('bbox') or []
                    if len(b) != 4: return False
                    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                    cx0, cy0 = (previa[0]+previa[2])/2, (previa[1]+previa[3])/2
                    return abs(cx-cx0) <= max(110, (previa[2]-previa[0])*1.5) and abs(cy-cy0) <= max(100, (previa[3]-previa[1])*1.5)
                hay_actual = any(coincide_area(p) for p in dato.get('personas',[])+dato.get('poses_sin_asociar',[]))
                if not hay_actual:
                    roi_resultado = recuperar_en_roi(original, previa)
                    dato['recuperacion_roi'] = roi_resultado or {'estado':'sin_deteccion_actual'}
                    if roi_resultado:
                        b = roi_resultado['bbox']
                        x1,y1,x2,y2 = map(int,b)
                        cv2.rectangle(anotada,(x1,y1),(x2,y2),(255,0,255),2)
                        cv2.putText(anotada, 'YOLO ROI - revisar', (x1,max(20,y1-7)), cv2.FONT_HERSHEY_SIMPLEX,.48,(255,0,255),2)
                        dato['advertencias'].append('Persona recuperada en ROI con YOLO; verificar identidad con Video.')
            observaciones = [p for p in dato.get('personas', []) if p.get('estado_postural') in ('postura_horizontal_posible','inclinacion_para_revision')]
            observaciones += [p for p in dato.get('poses_sin_asociar', []) if p.get('estado_postural') in ('postura_horizontal_posible','inclinacion_para_revision')]
            if observaciones:
                ultimo_candidato = {'segundo_s':dato['segundo_s'], 'bbox':observaciones[0]['bbox']}
                perdidas_consecutivas = 0
                dato['revision_incidente'] = 'indicio_visual_sin_confirmar'
            elif roi_resultado:
                ultimo_candidato = {'segundo_s': dato['segundo_s'], 'bbox': roi_resultado['bbox']}
                dato['revision_incidente'] = 'persona_recuperada_roi_sin_confirmar_incidente'
            elif ultimo_candidato and not dato.get('poses_sin_asociar') and not any(p.get('estado_postural') in ('postura_horizontal_posible','inclinacion_para_revision') for p in dato.get('personas', [])):
                # Se informa una perdida durante un breve periodo, sin fingir deteccion.
                segundos_perdidos = dato['segundo_s'] - ultimo_candidato['segundo_s']
                if 0 < segundos_perdidos <= 2.0:
                    perdidas_consecutivas += 1
                    dato['revision_incidente'] = 'deteccion_perdida_tras_indicio'
                    dato['ultima_observacion_estimada'] = {**ultimo_candidato, 'es_deteccion_actual':False}
                    dato['advertencias'].append('Se perdio la observacion posterior a una postura inusual; no hay deteccion confirmada en este fotograma.')
                    cv2.putText(anotada, 'OBSERVACION PERDIDA - revisar', (12, max(22, anotada.shape[0]-18)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,140,255), 2)
                else:
                    ultimo_candidato = None
                    dato['revision_incidente'] = 'sin_evidencia_actual'
            else:
                dato['revision_incidente'] = 'sin_indicio_suficiente'
            dato.pop('imagen_anotada', None)
            if carpeta:
                cv2.imwrite(str(carpeta / f'frame_{i:03d}_original.jpg'), original)
                cv2.imwrite(str(carpeta / f'frame_{i:03d}_anotado.jpg'), anotada)
            resultados.append(dato)
            imagenes.append(anotada)
        return {
            'modulo': 'imagen', 'estado': 'ok', 'clip': ruta.name,
            'duracion_s': round(duracion, 3), 'fps_origen': round(fps, 3),
            'fotogramas_planificados': len(tiempos),
            'fotogramas_analizados': len(imagenes),
            'tiempo_total_ms': round((time.perf_counter() - inicio)*1000, 1),
            'contexto_video': contexto_video or {},
            'resultados': resultados,
            'nota': 'No confirma caídas; la interpretación temporal pertenece a Video/Predicción.'
        }, imagenes
    finally:
        captura.release()
