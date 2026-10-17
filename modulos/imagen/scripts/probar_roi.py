
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))  # raíz de Video-G-G
import cv2
from pathlib import Path
from ultralytics import YOLO

modelo = YOLO("models/imagen/yolo11n.pt")

ruta = Path(
    "datasets/escaleras/frames/video_06_t006.jpg"
)

imagen = cv2.imread(str(ruta))

if imagen is None:
    raise SystemExit("No se pudo leer la imagen")

# Zona aproximada alrededor de manos y celular
# Coordenadas especificas para esta fotografia
x1, y1, x2, y2 = 325, 160, 455, 270

recorte = imagen[y1:y2, x1:x2]

if recorte.size == 0:
    raise SystemExit("El recorte esta vacio")

# Ampliar el recorte cuatro veces
recorte_grande = cv2.resize(
    recorte,
    None,
    fx=4,
    fy=4,
    interpolation=cv2.INTER_CUBIC
)

resultados = modelo.predict(
    recorte_grande,
    imgsz=960,
    conf=0.10,
    verbose=False
)

resultado = resultados[0]
celulares = []

for caja in resultado.boxes:
    clase = int(caja.cls[0])
    nombre = resultado.names[clase]

    if nombre == "cell phone":
        celulares.append({
            "confianza": round(
                float(caja.conf[0]), 3
            ),
            "bbox_original": [
                round(x1 + float(caja.xyxy[0][0]) / 4, 1),
                round(y1 + float(caja.xyxy[0][1]) / 4, 1),
                round(x1 + float(caja.xyxy[0][2]) / 4, 1),
                round(y1 + float(caja.xyxy[0][3]) / 4, 1)
            ]
        })

salida = Path("runs/imagen/prueba_roi")
salida.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(salida / "recorte.jpg"),
    recorte_grande
)

cv2.imwrite(
    str(salida / "deteccion.jpg"),
    resultado.plot()
)

print("Celulares detectados:", len(celulares))
print(celulares)
print("Resultados en:", salida)
