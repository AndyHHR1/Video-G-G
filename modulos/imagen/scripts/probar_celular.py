
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))  # raíz de Video-G-G
from pathlib import Path
import cv2
from ultralytics import YOLO

modelo = YOLO("models/imagen/yolo11n.pt")

imagen_path = Path(
    "datasets/escaleras/frames/video_06_t006.jpg"
)

imagen = cv2.imread(str(imagen_path))

if imagen is None:
    raise SystemExit(
        f"No se encontró la imagen: {imagen_path}"
    )

salida = Path("runs/imagen/prueba_celular")
salida.mkdir(parents=True, exist_ok=True)

pruebas = [
    ("A_original", 640, 0.35),
    ("B_sensible", 960, 0.15),
    ("C_resolucion", 1280, 0.10)
]

for nombre, tamano, confianza in pruebas:
    resultados = modelo.predict(
        source=imagen,
        imgsz=tamano,
        conf=confianza,
        verbose=False
    )

    resultado = resultados[0]
    celulares = []

    for caja in resultado.boxes:
        clase = int(caja.cls[0])
        etiqueta = resultado.names[clase]

        if etiqueta == "cell phone":
            celulares.append({
                "confianza": round(
                    float(caja.conf[0]), 3
                ),
                "bbox": [
                    round(float(v), 1)
                    for v in caja.xyxy[0].tolist()
                ]
            })

    print(
        f"{nombre}: {len(celulares)} celulares"
    )
    print(celulares)

    cv2.imwrite(
        str(salida / f"{nombre}.jpg"),
        resultado.plot()
    )

print("Pruebas finalizadas:", salida)
