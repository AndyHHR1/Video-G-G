"""Estimación de postura con YOLO11n-Pose (17 puntos COCO)."""
import cv2
import math

from .modelos import dispositivo, modelo_pose as _modelo_pose


def calcular_inclinacion(puntos):
    # YOLO Pose:
    # hombro izquierdo = 5, derecho = 6
    # cadera izquierda = 11, derecha = 12
    indices = [5, 6, 11, 12]

    if any(puntos[i][2] < 0.5 for i in indices):
        return None

    hombro_x = (puntos[5][0] + puntos[6][0]) / 2
    hombro_y = (puntos[5][1] + puntos[6][1]) / 2

    cadera_x = (puntos[11][0] + puntos[12][0]) / 2
    cadera_y = (puntos[11][1] + puntos[12][1]) / 2

    dx = hombro_x - cadera_x
    dy = cadera_y - hombro_y

    if abs(dx) + abs(dy) < 1e-6:
        return None

    angulo = math.degrees(math.atan2(abs(dx), abs(dy)))
    return round(angulo, 2)


def analizar_postura(frame):
    resultados = _modelo_pose().predict(
        frame,
        conf=0.35,
        device=dispositivo(),
        verbose=False
    )

    resultado = resultados[0]
    imagen_anotada = resultado.plot()
    personas = []

    if resultado.keypoints is not None:
        puntos_xy = resultado.keypoints.xy.cpu().numpy()
        puntos_conf = resultado.keypoints.conf

        if puntos_conf is not None:
            confianzas = puntos_conf.cpu().numpy()
        else:
            confianzas = None

        for i, puntos in enumerate(puntos_xy):
            if confianzas is None:
                continue

            articulaciones = [
                [float(x), float(y), float(c)]
                for (x, y), c in zip(
                    puntos, confianzas[i]
                )
            ]

            angulo = calcular_inclinacion(articulaciones)


            caja = resultado.boxes.xyxy[i].cpu().tolist()

            personas.append({
                "persona_indice": i,
                "bbox": [round(float(v), 2) for v in caja],
                "inclinacion_torso": angulo,
                "puntos_corporales": articulaciones
            })


            if angulo is not None:
                cv2.putText(
                    imagen_anotada,
                    f"Torso: {angulo:.1f} grados",
                    (15, 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

    return imagen_anotada, personas


if __name__ == "__main__":
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la camara")

    print("Pose Estimation iniciado. Presiona Q para salir.")

    try:
        while True:
            ok, frame = camara.read()

            if not ok:
                break

            imagen, personas = analizar_postura(frame)

            cv2.imshow(
                "Modulo Imagen - Pose Estimation",
                imagen
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()
