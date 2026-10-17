"""Ejecutar desde la raíz del proyecto: python -m scripts.comparar_calidad RUTA.jpg
Requiere opencv-python, numpy; con --yolo también ultralytics y pesos originales.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))  # raíz de Video-G-G
import argparse
import json
from pathlib import Path
import cv2
from modulos.imagen.modulo_imagen.preprocesamiento import mejorar_imagen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagen", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("runs/imagen/comparacion_calidad"))
    ap.add_argument("--yolo", action="store_true", help="Comparar detecciones; exige pesos YOLO instalados")
    args = ap.parse_args()
    frame = cv2.imread(str(args.imagen))
    if frame is None:
        ap.error(f"No se pudo abrir la imagen: {args.imagen}")
    args.salida.mkdir(parents=True, exist_ok=True)
    datos = {}
    imagenes = {}
    for modo in ("original", "adaptativo", "clahe", "gamma"):
        img, info = mejorar_imagen(frame, modo=modo)
        cv2.imwrite(str(args.salida / f"{modo}.jpg"), img)
        datos[modo] = info
        imagenes[modo] = img
    if args.yolo:
        from modulos.imagen.modulo_imagen.detector import detectar_imagen
        for modo, img in imagenes.items():
            marcada, dets = detectar_imagen(img)
            cv2.imwrite(str(args.salida / f"{modo}_yolo.jpg"), marcada)
            datos[modo]["detecciones_yolo"] = dets
    (args.salida / "comparacion.json").write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Completado:", args.salida.resolve())
    for modo, info in datos.items():
        print(modo, info["operaciones"], "->", info["despues"]["luminosidad"], "brillo")
        if args.yolo:
            print("  detecciones:", [(d["clase"], d["confianza"]) for d in info["detecciones_yolo"]])


if __name__ == "__main__":
    main()
