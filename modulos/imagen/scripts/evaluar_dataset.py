"""Evaluación exploratoria de detección y pose en fotografías de escaleras.
No calcula precision/recall sin anotaciones humanas, ni clasifica accidentes.
Ejecutar desde la raíz del repositorio: python scripts/evaluar_dataset.py ...
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))  # raíz de Video-G-G
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (aa + bb - inter) if aa + bb > inter else 0.0


def optical_quality(frame):
    import numpy as np
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        'luminosidad_media_0_255': round(float(np.mean(gray)), 2),
        'contraste_desviacion': round(float(np.std(gray)), 2),
        'nitidez_laplaciana_varianza': round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


def torso_angle(xy, cf, threshold=0.5):
    import math
    if len(xy) < 13 or cf is None or any(float(cf[k]) < threshold for k in (5, 6, 11, 12)):
        return None
    sx = (float(xy[5][0]) + float(xy[6][0])) / 2
    sy = (float(xy[5][1]) + float(xy[6][1])) / 2
    hx = (float(xy[11][0]) + float(xy[12][0])) / 2
    hy = (float(xy[11][1]) + float(xy[12][1])) / 2
    if abs(sx - hx) + abs(sy - hy) < 1e-6:
        return None
    return round(math.degrees(math.atan2(abs(sx-hx), abs(hy-sy))), 2)


def extract_pose(result):
    items = []
    if result.boxes is None or result.keypoints is None:
        return items
    boxes = result.boxes.xyxy.cpu().numpy()
    coords = result.keypoints.xy.cpu().numpy()
    conf = result.keypoints.conf
    scores = conf.cpu().numpy() if conf is not None else None
    for i in range(min(len(boxes), len(coords))):
        cf = scores[i] if scores is not None else None
        points = [[round(float(x), 2), round(float(y), 2), round(float(cf[j]), 3) if cf is not None else None]
                  for j, (x, y) in enumerate(coords[i])]
        items.append({'bbox': [round(float(v), 2) for v in boxes[i]],
                      'inclinacion_torso_grados': torso_angle(coords[i], cf),
                      'keypoints_xy_conf': points})
    return items


def match_people(detections, poses, min_iou=0.5):
    """Matching greedy uno-a-uno dentro de la MISMA imagen; no es tracking."""
    matches = sorted([(iou(d['bbox'], p['bbox']), di, pi)
                      for di, d in enumerate(detections) if d['clase'] == 'person'
                      for pi, p in enumerate(poses)], reverse=True)
    taken_d, taken_p = set(), set()
    for score, di, pi in matches:
        if score < min_iou:
            break
        if di not in taken_d and pi not in taken_p:
            detections[di]['pose'] = poses[pi]
            taken_d.add(di)
            taken_p.add(pi)
    return detections


def run(args):
    src, dst = Path(args.input), Path(args.output)
    if not src.is_dir():
        raise SystemExit(f'No existe carpeta de entrada: {src}')
    files = sorted(p for p in src.rglob('*') if p.suffix.lower() in EXTS)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit('No encontré imágenes JPG/PNG en la carpeta de entrada.')
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'anotadas').mkdir(exist_ok=True)
    print(f'Imágenes: {len(files)}. Cargando modelos una sola vez...')
    detector = YOLO(args.modelo)
    pose_model = YOLO(args.pose)
    rows, details, counts = [], [], Counter()
    for index, path in enumerate(files, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            rows.append({'archivo': str(path.relative_to(src)), 'estado': 'no_se_pudo_leer'})
            continue
        start = time.perf_counter()
        det = detector.predict(frame, conf=args.conf, verbose=False, device=args.device)[0]
        pos = pose_model.predict(frame, conf=args.conf, verbose=False, device=args.device)[0]
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        detections = []
        if det.boxes is not None:
            for box in det.boxes:
                cls = int(box.cls[0].item())
                name = str(det.names[cls])
                if name not in ('person', 'cell phone'):
                    continue
                detections.append({'clase': name,
                                   'confianza': round(float(box.conf[0].item()), 4),
                                   'bbox': [round(float(v), 2) for v in box.xyxy[0].tolist()],
                                   'pose': None})
        poses = extract_pose(pos)
        match_people(detections, poses, args.iou)
        num_people = sum(d['clase'] == 'person' for d in detections)
        num_cells = sum(d['clase'] == 'cell phone' for d in detections)
        num_pose = sum(d['clase'] == 'person' and d['pose'] is not None for d in detections)
        quality = optical_quality(frame)
        relative = path.relative_to(src)
        output_name = '__'.join(relative.parts)
        annotated = pos.plot()
        # Asegurar que cajas de celulares también estén visibles sobre la salida Pose.
        for d in detections:
            if d['clase'] == 'cell phone':
                x1,y1,x2,y2 = map(int,d['bbox'])
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,165,255),2)
                cv2.putText(annotated,f"celular {d['confianza']:.2f}",(x1,max(12,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,165,255),2)
        cv2.imwrite(str(dst / 'anotadas' / output_name), annotated)
        record = {'archivo': relative.as_posix(), 'estado': 'ok',
                  'personas': num_people, 'celulares': num_cells,
                  'personas_con_pose_asociada': num_pose,
                  'tiempo_inferencia_ms': elapsed_ms, **quality}
        rows.append(record)
        details.append({'archivo': relative.as_posix(), 'width': int(frame.shape[1]),
                        'height': int(frame.shape[0]), 'detecciones': detections,
                        'posturas_sin_asociar': max(0,len(poses)-num_pose),
                        'calidad': quality})
        counts['personas'] += num_people
        counts['celulares'] += num_cells
        print(f'[{index}/{len(files)}] {relative}: {num_people} personas, {num_cells} celulares, {elapsed_ms} ms')
    fields = ['archivo', 'estado', 'personas', 'celulares', 'personas_con_pose_asociada',
              'tiempo_inferencia_ms', 'luminosidad_media_0_255',
              'contraste_desviacion', 'nitidez_laplaciana_varianza']
    with (dst / 'reporte.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (dst / 'detecciones.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'imagenes_totales': len(files), 'imagenes_analizadas': len(details),
               'personas_detectadas_total': counts['personas'],
               'celulares_detectados_total': counts['celulares'],
               'nota': 'Conteos de detecciones por fotograma, no personas únicas. No se ha calculado precisión/recall sin ground truth. No clasifica caídas ni accidentes.'}
    (dst / 'resumen.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('LISTO:', dst.resolve(), '\n', json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluación exploratoria de fotogramas con YOLO + YOLO-Pose')
    parser.add_argument('--input', default='datasets/escaleras/frames')
    parser.add_argument('--output', default='runs/imagen/evaluacion_01')
    parser.add_argument('--modelo', default="models/imagen/yolo11n.pt")
    parser.add_argument('--pose', default="models/imagen/yolo11n-pose.pt")
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--device', default='cpu', help='cpu o 0 para GPU NVIDIA configurada')
    parser.add_argument('--limit', type=int, default=0, help='0 = todas; 10 = prueba inicial')
    run(parser.parse_args())
