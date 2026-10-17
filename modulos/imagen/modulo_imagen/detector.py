"""Detección de personas y celulares con YOLO11n (COCO preentrenado)."""
import cv2

from .modelos import dispositivo, modelo_deteccion


def __getattr__(nombre):
    # Compatibilidad: el código original importaba `modelo` de este archivo.
    if nombre == "modelo":
        return modelo_deteccion()
    raise AttributeError(nombre)


def detectar_imagen(frame):
    modelo = modelo_deteccion()
    resultados = modelo.predict(
        source=frame,
        conf=0.35,
        device=dispositivo(),
        verbose=False
    )

    imagen_anotada = resultados[0].plot()

    detecciones = []

    for caja in resultados[0].boxes:
        clase_id = int(caja.cls[0])
        nombre = modelo.names[clase_id]
        confianza = float(caja.conf[0])
        coordenadas = caja.xyxy[0].tolist()

        if nombre in ("person", "cell phone"):
            detecciones.append({
                "clase": nombre,
                "confianza": round(confianza, 3),
                "bbox": [round(x, 2) for x in coordenadas]
            })

    return imagen_anotada, detecciones


if __name__ == "__main__":
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    print("Cámara iniciada. Presiona Q para salir.")

    try:
        while True:
            ok, frame = camara.read()

            if not ok:
                break

            imagen, objetos = detectar_imagen(frame)

            cv2.imshow("Modulo Imagen - YOLO", imagen)

            for objeto in objetos:
                print(objeto)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()
