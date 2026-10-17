"""Análisis por fotograma sin abrir cámara ni tocar RiskEngine/Video."""
import math
import time
import cv2
from .calidad import validar_bgr
from .preprocesamiento import mejorar_imagen
from .detector import detectar_imagen
from .pose import analizar_postura
from .analizador import calcular_iou


def _conf_punto(puntos, indice, umbral=0.5):
    if not isinstance(puntos, list) or len(puntos) <= indice:
        return None
    p = puntos[indice]
    if len(p) < 3 or float(p[2]) < umbral:
        return None
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


def _dist(a, b):
    return math.dist(a, b) if a is not None and b is not None else None


def _asociar_telefono(cel, personas):
    """Proximidad relativa: pista geométrica, NO prueba de uso/distracción."""
    x1,y1,x2,y2 = cel['bbox']
    centro = [(x1+x2)/2, (y1+y2)/2]
    candidatos = []
    for p in personas:
        bbox = p['bbox']
        ancho = max(float(bbox[2])-float(bbox[0]), 1)
        alto = max(float(bbox[3])-float(bbox[1]), 1)
        pose = p.get('pose') or {}
        puntos = pose.get('puntos_corporales', [])
        manos = [_conf_punto(puntos, 9), _conf_punto(puntos, 10)]
        d = [_dist(centro, mano) for mano in manos]
        d = [v for v in d if v is not None]
        # Si la pose no es visible, no inferimos proximidad.
        if d:
            rel = min(d) / max(ancho, alto)
            candidatos.append((rel, p['detection_id']))
    if not candidatos:
        return {'estado': 'no_evaluable', 'persona_id': None, 'distancia_relativa_mano': None}
    dist, idx = min(candidatos)
    return {'estado': 'cerca_de_mano' if dist <= .20 else 'sin_proximidad_clara',
            'persona_id': idx, 'distancia_relativa_mano': round(dist, 3)}



def _evaluar_postura(pose):
    """Indicio visual, nunca confirmacion de accidente."""
    angulo = pose.get('inclinacion_torso')
    caja = pose.get('bbox') or []
    if len(caja) != 4:
        return 'no_evaluable'
    ancho = max(0., float(caja[2]) - float(caja[0]))
    alto = max(1., float(caja[3]) - float(caja[1]))
    # La proporcion de la caja es una pista, no una verificacion de caida.
    horizontal = ancho / alto >= 1.15
    if (angulo is not None and angulo >= 55) or horizontal:
        return 'postura_horizontal_posible'
    if angulo is not None and angulo >= 35:
        return 'inclinacion_para_revision'
    return 'sin_indicio_suficiente'


def _anotar_pose_huerfana(imagen, pose, estado):
    caja = pose.get('bbox') or []
    if len(caja) != 4:
        return
    x1, y1, x2, y2 = map(int, caja)
    h, w = imagen.shape[:2]
    x1, y1 = max(0, min(w-1, x1)), max(0, min(h-1, y1))
    x2, y2 = max(0, min(w-1, x2)), max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    color = (0, 140, 255)
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
    texto = 'POSE sin YOLO'
    if estado == 'postura_horizontal_posible':
        texto = 'POSE horizontal - REVISAR'
    elif estado == 'inclinacion_para_revision':
        texto = 'POSE inclinada - REVISAR'
    cv2.putText(imagen, texto, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 2)

def analizar_frame(frame, *, modo='adaptativo'):
    """Retorna (dict serializable, imagen BGR anotada); modelos existentes se reutilizan.

    Nunca emite etiqueta de caída o distracción confirmada por una fotografía.
    """
    validar_bgr(frame)
    t0 = time.perf_counter()
    procesada, info = mejorar_imagen(frame, modo=modo)
    t_pre = time.perf_counter()
    # Mismos píxeles y coordenadas para el detector y la pose.
    _, objetos = detectar_imagen(procesada)
    t_det = time.perf_counter()
    imagen_pose, poses = analizar_postura(procesada)
    t_pose = time.perf_counter()
    personas = []
    usados = set()
    for obj in objetos:
        if obj['clase'] != 'person':
            continue
        candidates = [(calcular_iou(obj['bbox'], p['bbox']), i) for i, p in enumerate(poses)
                      if i not in usados and 'bbox' in p]
        score, idx = max(candidates, default=(0.0, None))
        pose = poses[idx] if idx is not None and score >= 0.3 else None
        if pose is not None:
            usados.add(idx)
        pnts = pose.get('puntos_corporales', []) if pose else []
        visible = {nombre: _conf_punto(pnts, j) is not None for nombre,j in
                   [('hombro_izquierdo',5),('hombro_derecho',6),('muneca_izquierda',9),('muneca_derecha',10),
                    ('cadera_izquierda',11),('cadera_derecha',12)]}
        personas.append({'detection_id':len(personas),'bbox':obj['bbox'],
                         'confianza':obj['confianza'],'pose':pose,
                         'puntos_visibles':visible,
                         'pose_asociada':pose is not None,
                         'estado_postural':_evaluar_postura(pose) if pose else 'no_evaluable'})
    huerfanas = []
    for idx, pose in enumerate(poses):
        if idx in usados:
            continue
        estado_postural = _evaluar_postura(pose)
        huerfanas.append({**pose, 'origen':'solo_pose', 'estado_postural':estado_postural})
        _anotar_pose_huerfana(imagen_pose, pose, estado_postural)
    telefonos = []
    for obj in objetos:
        if obj['clase'] == 'cell phone':
            telefonos.append({**obj, 'relacion_manos':_asociar_telefono(obj, personas)})
    advertencias = []
    if huerfanas:
        advertencias.append(f'{len(huerfanas)} pose(s) sin coincidencia con YOLO; revisar recuadros naranjas.')
    if any(p['estado_postural'] in ('postura_horizontal_posible','inclinacion_para_revision') for p in personas) or any(p['estado_postural'] in ('postura_horizontal_posible','inclinacion_para_revision') for p in huerfanas):
        advertencias.append('Hay indicios visuales de postura inusual: posible incidente para revisar con Video, NO confirmado.')
    if not personas and poses:
        advertencias.append('Se estimaron poses, pero no se asociaron a una detección de persona con seguridad.')
    if any(not p['pose_asociada'] for p in personas):
        advertencias.append('Hay personas detectadas sin pose asociada; no se puede inferir su postura.')
    if any(p['pose_asociada'] and not all(p['puntos_visibles'][k] for k in ('muneca_izquierda','muneca_derecha')) for p in personas):
        advertencias.append('Una o ambas muñecas no son visibles con confianza suficiente.')
    if not telefonos:
        advertencias.append('No se detectaron celulares; esto NO significa que no haya uno.')
    if info['antes']['brillo_bajo_estimado']:
        advertencias.append('Luminosidad baja estimada: detección potencialmente limitada.')
    if info['antes']['porcentaje_zonas_sobreexpuestas'] > 25:
        advertencias.append('Muchas zonas sobreexpuestas: pueden faltar detalles visuales.')
    for obj in objetos:
        if obj['clase'] in ('person','cell phone'):
            x1,y1,x2,y2 = map(int,obj['bbox'])
            cv2.rectangle(imagen_pose,(x1,y1),(x2,y2),(0,200,250) if obj['clase']=='cell phone' else (70,200,70),2)
            cv2.putText(imagen_pose,f"{obj['clase']} {obj['confianza']:.2f}",(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
    return {'modulo':'imagen', 'version':'pruebas-0.3', 'estado':'ok',
            'dimensiones':{'ancho':int(frame.shape[1]),'alto':int(frame.shape[0])},
            'calidad_original':info['antes'],'preprocesamiento':info,
            'personas':personas,'poses_sin_asociar':huerfanas,
            'celulares':telefonos,'otros_objetos':[o for o in objetos if o['clase'] not in ('person','cell phone')],
            'advertencias':advertencias,
            'tiempos_ms': {'preprocesamiento':round((t_pre-t0)*1000,1), 'deteccion':round((t_det-t_pre)*1000,1), 'pose':round((t_pose-t_det)*1000,1)},
            'tiempo_procesamiento_ms':round((time.perf_counter()-t0)*1000,1)}, imagen_pose
