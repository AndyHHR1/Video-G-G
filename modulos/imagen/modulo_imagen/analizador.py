
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .detector import detectar_imagen
from .pose import analizar_postura


def calcular_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def calidad_imagen(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return {
        "luminosidad": round(float(np.mean(gris)), 2),
        "contraste": round(float(np.std(gris)), 2),
        "nitidez_laplaciana": round(
            float(cv2.Laplacian(gris, cv2.CV_64F).var()), 2
        )
    }


def analizar_imagen(frame):
    if frame is None or frame.size == 0:
        raise ValueError("La imagen esta vacia")

    # Ambos modelos analizan el mismo fotograma.
    _, detecciones = detectar_imagen(frame)
    imagen_anotada, posturas = analizar_postura(frame)

    personas = []
    indices_usados = set()

    for objeto in detecciones:
        if objeto["clase"] != "person":
            continue

        mejor_indice = None
        mejor_iou = 0.0

        for indice, postura in enumerate(posturas):
            if indice in indices_usados:
                continue

            if "bbox" not in postura:
                continue

            valor = calcular_iou(
                objeto["bbox"], postura["bbox"]
            )

            if valor > mejor_iou:
                mejor_iou = valor
                mejor_indice = indice

        pose_asociada = None

        if mejor_indice is not None and mejor_iou >= 0.5:
            pose_asociada = posturas[mejor_indice]
            indices_usados.add(mejor_indice)

        personas.append({
            "detection_id": len(personas),
            "bbox": objeto["bbox"],
            "confianza": objeto["confianza"],
            "pose": pose_asociada
        })

    otros_objetos = [
        objeto for objeto in detecciones
        if objeto["clase"] != "person"
    ]

    resultado = {
        "modulo": "imagen",
        "version": "0.2.0",
        "dimensiones": {
            "ancho": int(frame.shape[1]),
            "alto": int(frame.shape[0])
        },
        "personas": personas,
        "objetos": otros_objetos,
        "calidad_imagen": calidad_imagen(frame)
    }

    return resultado, imagen_anotada


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Uso: python -m app.modulo_imagen.analizador imagen.jpg"
        )

    ruta = Path(sys.argv[1])
    frame = cv2.imread(str(ruta))

    if frame is None:
        raise SystemExit("No se pudo abrir la imagen")

    datos, imagen = analizar_imagen(frame)

    carpeta = Path("runs/imagen")
    carpeta.mkdir(parents=True, exist_ok=True)

    archivo_json = carpeta / "resultado.json"
    archivo_jpg = carpeta / "resultado.jpg"

    archivo_json.write_text(
        json.dumps(datos, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    cv2.imwrite(str(archivo_jpg), imagen)

    print("ANALISIS COMPLETADO")
    print("Personas:", len(datos["personas"]))
    print("Otros objetos:", len(datos["objetos"]))
    print("JSON:", archivo_json)
    print("Imagen:", archivo_jpg)
