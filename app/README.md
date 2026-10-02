# Plataforma de pruebas local

Interfaz web para probar el detector de riesgo en escaleras sobre imagen, video
o cámara en vivo. Pensada para desarrollo: sirve para validar el comportamiento
del modelo en escenas reales, no solo las métricas del dataset.

## Arranque

```bash
./venv/bin/python app/server.py
```

Abrir **http://127.0.0.1:5000**

Opciones:

| Flag | Por defecto | Para qué |
|------|-------------|----------|
| `--weights` | `runs/yolov8s_6clases/weights/best.pt` | Probar otro checkpoint (p. ej. el baseline `yolov8n_balanced`) |
| `--conf` | `0.25` | Umbral por defecto |
| `--imgsz` | `640` | Resolución de inferencia |
| `--host` / `--port` | `127.0.0.1` / `5000` | Escucha |
| `--debug` | off | Recarga automática de Flask |

Para exponerla en la red local (p. ej. desde el móvil):

```bash
./venv/bin/python app/server.py --host 0.0.0.0 --port 5000
```

## Modos de prueba

**Imagen** — sube un frame y obtienes la imagen anotada, el desglose de
tiempos por etapa y la tabla de detecciones con su nivel de riesgo.

**Video** — se decodifica, se infiere frame a frame y se descarga el video ya
anotado. El progreso se ve en la consola del servidor.

**Cámara en vivo** — usa la **webcam de tu navegador**, no la del servidor. El
modelo corre en la máquina remota y el navegador captura cada frame con
`getUserMedia`, lo pinta en un canvas y lo envía a `/api/frame`, que lo devuelve
ya anotado con una banda de FPS y nivel de riesgo. Solo hay una petición en
vuelo a la vez: encolarlas acumularía retraso y el panel mostraría frames
viejos. Se puede elegir la cadencia de envío (5 / 10 / 15 fps).

Es el modo que corresponde al escenario de `Caso.md` (RQNF01: cámara fija
única), así que conviene apuntar la webcam a una escalera real y comprobar a
mano los FPS y las detecciones.

> **Requisito importante:** `getUserMedia` solo funciona en un **origen
> seguro**. Debes abrir la app en `http://localhost:5000` (con el reenvío de
> puertos de VS Code). Si la abres con la URL de tunneling de VS Code
> (`*.app.github.dev`), el navegador rechazará la cámara por política de
> contexto no seguro. La app lo detecta y te avisa con el motivo exacto.
>
> La app se inicia en `0.0.0.0` para que el reenvío de puertos funcione. Eso
> también la hace accesible desde la red local: no hay autenticación, así que
> úsala solo en redes de confianza.

## Clases y panel de riesgo

Cada clase se traduce a la categoría de riesgo de `Caso.md` y el panel muestra
el nivel peor detectado:

| Clase | Riesgo | Referencia |
|---|---|---|
| Persona caída | ALTO | RQF02 (4) · RQNF05 |
| Pérdida de equilibrio | ALTO | RQF02 (4) |
| Postura no ergonómica | MEDIO | RQF03 |
| Postura erguida | BAJO | RQF03 |
| Escalera | CONTEXTO | — |
| Persona | NEUTRO | RQNF08 (entrada a MediaPipe) |

**Lo que esta plataforma no cubre:** las categorías 1, 2 y 3 de RQF02
(obstáculos en los escalones, no uso del pasamanos, distracciones). No hay
datos para ellas y, según el diseño de `Caso.md`, se resuelven con keypoints
de MediaPipe, no con cajas de YOLO. El panel lo indica explícitamente para no
dar una falsa sensación de cobertura.

## API

| Ruta | Método | Descripción |
|------|--------|-------------|
| `/api/health` | GET | Pesos cargados, dispositivo y nº de clases |
| `/api/detect` | POST | multipart `file` + `conf`. Devuelve detecciones, resumen de riesgo, tiempos e imagen anotada en base64 |
| `/api/video` | POST | multipart `file` + `conf`. Devuelve `video/mp4` anotado |
| `/api/stream` | GET | `?conf=&imgsz=&cam=` → stream MJPEG |

## Nota de rendimiento

Toda la inferencia y el dibujo se ejecutan en **un único hilo trabajador
persistente** (`Inferencer`), no en el hilo de cada petición. El motivo es
medible en esta máquina: Flask atiende cada petición en un hilo distinto y en
un hilo recién creado hay que reenlazar el contexto CUDA, lo que cuesta

| Operación | Hilo ya enlazado | Hilo nuevo |
|-----------|------------------|------------|
| inferencia | ~29 ms | ~82 ms |
| dibujo OpenCV | 0.13 ms | ~23 ms |

Con el hilo único la petición queda en **~37 ms de mediana** (mínimo 21.7 ms)
en lugar de ~85 ms. La latencia sube y baja según la frecuencia de peticiones
porque la GPU baja de estado de energía cuando está ociosa; en el modo cámara
el stream la mantiene ocupada y el rendimiento es más estable.

## Limitaciones conocidas

- El servidor de desarrollo de Flask es de un solo proceso. Es suficiente
  para pruebas locales, pero no para uso concurrente real.
- Los videos se procesan enteros en memoria antes de devolverlos; para clips
  largos conviene usar el modo cámara o clips cortos.
- No hay autenticación ni límite de tasa: **no exponer a internet**.
