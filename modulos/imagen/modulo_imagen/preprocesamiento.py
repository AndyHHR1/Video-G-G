"""Correcciones adaptativas ligeras. Mantiene BGR para detección YOLO."""
import cv2
import numpy as np
from .calidad import medir_calidad, validar_bgr


def _gamma(img, valor):
    lut = np.clip(255*(np.arange(256,dtype=np.float32)/255)**valor,0,255).astype(np.uint8)
    return cv2.LUT(img,lut)


def _clahe(img, clip=1.6):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)


def mejorar_imagen(frame, *, modo='adaptativo'):
    """Devuelve (BGR mejorada, metadatos). No modifica el frame recibido."""
    validar_bgr(frame)
    if modo not in ('original','adaptativo','clahe','gamma'):
        raise ValueError('Modo inválido: original, adaptativo, clahe o gamma')
    antes=medir_calidad(frame)
    img=frame.copy()
    op=[]
    if modo=='gamma':
        img=_gamma(img,.78); op.append('gamma=0.78')
    elif modo=='clahe':
        img=_clahe(img); op.append('clahe_lab=1.6')
    elif modo=='adaptativo':
        # Evaluar zonas recortadas y no oscurecer imágenes correctas.
        oscuro = antes['luminosidad'] < 78
        sombras = antes['sombras_fuertes_estimadas']
        contraste_bajo = antes['contraste'] < 38
        ruido_alto = antes['ruido_mad_estimado'] > 9
        if ruido_alto and (oscuro or sombras):
            img=cv2.medianBlur(img,3); op.append('mediana_3')
        if oscuro:
            gamma = .68 if antes['luminosidad'] < 45 else .83
            img=_gamma(img,gamma); op.append(f'gamma={gamma}')
        if sombras or (contraste_bajo and antes['luminosidad'] < 110):
            img=_clahe(img,clip=1.5); op.append('clahe_lab=1.5')
        # La escala de grises solo se usa para medir: YOLO recibe BGR.
    despues=medir_calidad(img)
    return img, {'modo':modo, 'operaciones':op, 'antes':antes, 'despues':despues,
                 'color_salida':'BGR', 'correccion_aplicada':bool(op)}
