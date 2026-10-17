"""Segunda busqueda localizada; no identifica accidentes ni crea detecciones ficticias."""
import time
import cv2


def crear_roi(bbox, dimensiones, *, factor=2.4, minimo=240):
    """ROI expandida, dentro de la imagen. dimensiones=(alto, ancho)."""
    h, w = dimensiones[:2]
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rw = min(w, max(minimo, (x2 - x1) * factor))
    rh = min(h, max(minimo, (y2 - y1) * factor))
    left = max(0, min(int(round(cx - rw/2)), w - int(rw)))
    top = max(0, min(int(round(cy - rh/2)), h - int(rh)))
    return left, top, min(w, left + int(rw)), min(h, top + int(rh))


def _intersecta(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    area = max(1, (b[2]-b[0])*(b[3]-b[1]))
    return inter / area


def recuperar_en_roi(frame, bbox_anterior, *, confianza=.16, imgsz=960):
    """Busca SOLO personas en la region previa. Requiere una prediccion YOLO actual.

    Se ejecuta a demanda en clips y no altera el detector global ni el tracking de Video.
    """
    from .modelos import dispositivo, modelo_deteccion
    modelo = modelo_deteccion()
    roi = crear_roi(bbox_anterior, frame.shape)
    x1,y1,x2,y2 = roi
    recorte = frame[y1:y2, x1:x2]
    if recorte.size == 0: return None
    t0 = time.perf_counter()
    resultados = modelo.predict(source=recorte, classes=[0], conf=confianza, imgsz=imgsz, device=dispositivo(), verbose=False)
    candidatos=[]
    for caja in resultados[0].boxes:
        local = caja.xyxy[0].tolist()
        global_box = [round(local[0]+x1, 2), round(local[1]+y1, 2), round(local[2]+x1, 2), round(local[3]+y1, 2)]
        centro = ((global_box[0]+global_box[2])/2, (global_box[1]+global_box[3])/2)
        if _intersecta(global_box, bbox_anterior) < .12 and not (bbox_anterior[0] <= centro[0] <= bbox_anterior[2] and bbox_anterior[1] <= centro[1] <= bbox_anterior[3]):
            continue
        candidatos.append({'bbox':global_box,'confianza':round(float(caja.conf[0]),3),'origen':'roi_yolo','clase':'person'})
    if not candidatos: return None
    mejor = max(candidatos, key=lambda c:c['confianza'])
    mejor['roi_usada'] = list(roi)
    mejor['tiempo_roi_ms'] = round((time.perf_counter()-t0)*1000,1)
    return mejor
