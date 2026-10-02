# Decisiones de diseño — Prevención de Caídas y Riesgos en Escaleras

Registro de las decisiones tomadas durante el desarrollo, con la referencia a
`Caso.md` que las justifica. Todos los scripts son reproducibles con semilla
fija (42).

- **Parte 1**: decisiones de la primera vuelta (unificación de los tres
  datasets originales, pseudo-etiquetado de Le2i con COCO, auditoría).
- **Parte 2** (este documento): decisiones de la segunda vuelta, nacidas de
  que el usuario probara el modelo con su webcam y fallara con gente cerca y
  de espaldas.

---

# PARTE 2

## 1. Taxonomía: se **mantienen** las 6 clases

**Síntoma reportado:** «detecta bien a las personas pero si están lejos; cuando
están cerca o de espaldas directamente no detecta».

**Diagnóstico medido.** Con las 6 clases, sobre 1228 imágenes de test con
persona:

| Métrica | Valor |
|---|---|
| Imágenes sin ninguna detección | 3 (0.2%) |
| Predicciones de clase `persona` | **76-80%** |
| Imágenes que reciben una clase de postura | 20% |

Es decir: **el modelo sí detectaba a la gente, pero la etiquetaba como
`Persona` genérica** el 76% de las veces. La caja se dibujaba y el sistema no
podía expresar riesgo.

**Causa.** `persona` aportaba 6621 cajas de las 10206 totales (65%). Es la clase
del CCTV de `tracking`, y con esa proporción el modelo la elige siempre: es la
opción de menor riesgo para él.

### Hipótesis probada y DESCARTADA

La primera hipótesis fue que el problema era la existencia misma de la clase
genérica, y que fusionarla en `persona_erguida` la eliminaría: cinco clases,
todo peatón con una postura, RQF03 satisfecho.

**Se entrenó y se midió. Fue un error:**

| clase (test) | 6 clases | 5 clases (fusionado) |
|---|---|---|
| `persona_erguida` | 0.679 | **0.498** |
| `persona_sentado` | 0.821 | 0.580 |
| `persona_caido` | 0.887 | 0.795 |
| `persona_desequilibrio` | 0.368 | 0.254 |
| **mAP50** | **0.706** | **0.587** |

Cayó **todo**, no solo la clase fusionada. La causa es que forzar una sola
clase sobre dos dominios muy distintos multiplica su varianza intra-clase:
`persona_erguida` pasó a cubrir a la vez fotos de stock de personas de pie
(grandes, iluminadas, posadas, área 0.08) y peatones de CCTV (pequeños, a
distancia, área 0.010). Un detector tiene que cubrir un rango de escala de
8x dentro de una misma etiqueta, y no puede hacerlo bien.

**La fusión queda revertida.** Se conservan las 6 clases.

**Cómo se controla entonces el colapso.** Con el techo de cajas del recorte
(`max_boxes_per_class`), que es donde sí se demuestra que funciona:

| | Ratio entre clases de persona | `persona` |
|---|---|---|
| Sin control (parte 1) | 61:1 | 70% de las cajas |
| Con techo de 1500 | **14:1** | **22% de las cajas** |

Además, `RQNF08` solo necesita **cajas** para alimentar a MediaPipe, no una
etiqueta «persona»: que el modelo etiquete a un peatón lejano como `persona`
no impide el análisis de pose posterior.

### Lo que sí explica el síntoma reportado

La evaluación por escala (`scripts/eval_scale.py`) mide **100% de detección en
todos los rangos de tamaño, incluidos los recortados, y tanto de espaldas
(escala del CCTV) como de frente**. Es decir, el comportamiento descrito
—«funciona de lejos, falla de cerca y de espaldas»— **no se reproduce en el
conjunto de datos**, aunque el usuario lo observa en su escena.

La explicación más honesta: la escena del usuario (webcam real, escalera,
alguien que pasa cerca y de espaldas) **no está representada en ningún dataset**
—ni en `stairs_fallhuman` ni en los externos—, y por tanto **tampoco está en
el conjunto de test**. Lo que se puede medir da 100%; lo que el usuario ve
queda fuera del alcance de cualquier métrica sobre estos datos. Queda
pendiente de datos reales, pospuesto por indicación expresa del usuario
(«estamos trabajando con lo del envelope»).

---

## 2. Bug de balanceo (recortaba la clase equivocada)

**Decisión anterior equivocada.** El recorte de clases se decidía contando
**imágenes** por clase, y se submuestreaba la clase con más imágenes. Medido,
eso elegía `persona_erguida` —porque Le2i `Stand` aporta muchas imágenes— y la
recortaba a la mitad, mientras `persona` (6634 cajas) quedaba intacta. Es
decir: se recortaba justo la clase de postura que más faltaba y se dejaba
intacta la dominante.

**Corrección.** El recorte se decide por **conteo de cajas**, que es lo que
compite en el gradiente, con un techo `max_boxes_per_class` y una tasa de
conservación adaptativa por clase (una tasa fija penalizaría a las clases menos
numerosas, que son las que más necesitan señal).

Balanceo real antes y después:

| | Ratio entre clases de persona | `persona_erguida` | `persona` |
|---|---|---|---|
| Antes (por imágenes) | 61:1 | recortada a la mitad | intacta |
| Ahora (por cajas, techo 1500) | **14:1** | 1278 cajas | **1487 cajas** |

---

## 3. Primeros planos sintéticos

**Síntoma reportado:** falla con gente cerca.

**Hallazgo medible en los datos.** Distribución del área de las cajas (fracción
del encuadre) y porcentaje de personas recortadas por el borde:

| Clase | Cajas | Recortadas | Área mediana |
|---|---|---|---|
| `persona` (CCTV) | 4570 | 8.1% | **0.010** (1% del frame) |
| `persona_erguida` | 824 | 19.8% | 0.080 |
| `persona_desequilibrio` | 357 | **0.0%** | 0.069 |

La clase de pre-caída **no tenía ni un solo ejemplo de primer plano**: el 100%
viene de Le2i, cinco escenas con cámara fija y plano abierto, cuerpo entero y
pequeño.

**Decisión.** Generar primeros planos por recorte (`scripts/add_zoom_variants.py`).
Un recorte es una **transformación geométrica exacta** de las cajas, así que se
generan primeros planos reales sin volver a anotar nada, y de paso el caso de
la persona parcialmente fuera de cuadro.

Efecto:

| Clase | Recortadas antes → después | Área mediana |
|---|---|---|
| `persona_desequilibrio` | 0.0% → 21.9% | 0.069 → 0.121 |
| `persona_erguida` | 19.8% → 28.5% | 0.080 → 0.111 |

**Resultado honesto: NO resolvió el problema reportado.** La evaluación por
escala (`scripts/eval_scale.py`) dio 100% de detección en todos los rangos de
tamaño, incluidos los recortados, **antes y después** de los primeros planos, y
el mAP50 de test bajó ligeramente (0.706 → 0.692).

La conclusión es que la vía sintética está agotada: el conjunto de test no
contiene escenas reales de escalera con webcam, y ninguna augmentation
sintética crea una escena que no existe. Los primeros planos se conservan
porque amplían la cobertura de escala a coste casi nulo, pero no son la
solución.

---

## 4. Re-etiquetado de Le2i con el modelo propio

En la parte 1 se descartaron las clases `Fall` y `Lie` de Le2i porque el
detector COCO no localizaba a una persona tendida (sujeto muy escorzado):
se quedaban en 61% y 66% de detección, y se prefirió no generar etiquetas
ruidosas. Eso dejaba `persona_caido` con ~250 cajas, la clase más escasa.

**Decisión.** Reetiquetar con el modelo del proyecto, que sí ve a la persona
tendida. Resultado:

| Clase | COCO (parte 1) | Modelo propio | Decisión |
|---|---|---|---|
| `Lie` | 66% | **91%** | se acepta |
| `Fall` | 61% | 65% | se descarta (caída en movimiento, con desenfoque) |

`persona_caido` pasó de 163 a 1246 cajas. Se mantiene la puerta de validación
que aborta si una clase no alcanza el 80%; `Fall` se descarta automáticamente
sin tumbar el resto.

> **⚠ Este punto queda DESMENTIDO por el 9.2.** Aunque `Lie` por sí solo es un
> dato válido y bien validado (91% de detección), **entrenar con él destruye la
> clase `persona_desequilibrio`**: `Lie` y `Likefall` son el mismo sujeto en
> momentos consecutivos. El dataset se regenera con `Lie` porque las correcciones
> de fuga y deduplicación son correctas, pero **el modelo entregado se entrenó
> sin `Lie`**. Ver el punto 9.2 antes de dar por buena esta decisión.

---

## 5. Fuga de datos en Le2i: clips solapados

**Bug encontrado.** `persona_desequilibrio` se quedaba **con cero instancias en
test**, lo que hacía su AP incalculable. Dos causas encadenadas:

1. Las carpetas de Le2i se llaman `<escena>_v<video>c<clip>`: `coffee_v11c23` y
   `coffee_v11c24` son clips 23 y 24 **del mismo video**. Repartir por clips
   metía frames casi idénticos en splits distintos.
   → Se agrupa por **video base** (`base_video()`).
2. Los clips se solapan tanto que **un mismo frame aparece en varias carpetas
   de estado**: el mismo fotograma figura como `Likefall` y como `Stand`. Con
   una deduplicación «el primero que llega gana», se quedaba `Stand` (que se
   recorre antes) y `persona_desequilibrio` perdía todo el test.
   → Se deduplica **priorizando la clase más escasa**
   (`dedupe_preferring_rare_classes()`).

También se añadió limpieza de la carpeta de salida en `prepare_le2i.py`: si no,
`prepare_dataset.py` (que lee el disco, no el manifest) arrastraba muestras de
repartos antiguos.

---

## 6. Métrica de evaluación

`persona_erguida` es la única clase que **mezcla** anotación humana (`fall`)
con pseudo-etiquetas (Le2i `Stand`, `tracking`), consecuencia de la fusión del
punto 1. Se reporta por tanto el mAP50 global y, aparte, el restringido a las
clases de etiqueta inequívocamente humana (`persona_sentado`, `escalera`).

Se mantiene el criterio de la parte 1: **toda cifra se reporta con la
procedencia de sus etiquetas indicada**, porque el conjunto de test contiene
pseudonotes de detector y eso penaliza la métrica de forma artificial.

---

## 7. Rendimiento de la plataforma web

**Problema medido.** Las peticiones tardaban ~85 ms con un modelo que en
aislado tardaba 19 ms. Causa: Flask atiende cada petición en un hilo distinto y
en un hilo recién creado hay que reenlazar el contexto CUDA.

| Operación | Hilo ya enlazado | Hilo nuevo |
|---|---|---|
| inferencia | 29 ms | 82 ms |
| dibujo OpenCV | 0.13 ms | 23 ms |

**Decisión.** Hilo trabajador persistente (`Inferencer`) que hace inferencia **y**
dibujo, con cola. De ~85 ms a **~37 ms de mediana**.

**Modo cámara.** La primera versión abría la cámara del **servidor**, que no
tiene ninguna (no existe `/dev/video*`). La webcam del usuario nunca se usaba.
Se reimplementó con `getUserMedia` en el navegador, envío frame a frame a
`/api/frame`, con una sola petición en vuelo (encolarlas acumula retraso). El
stream MJPEG se conserva solo para cuando el servidor tenga cámara, y ahora
responde **503 con el motivo** en lugar de fallar en silencio.

Detalle importante: el `<video>` de origen **no se oculta con `display:none`**
—un video oculto no carga metadata, se queda con `videoWidth = 0` y el canvas
sale 0×0, con lo que nunca se envía nada y la imagen queda en negro.

**Vídeo.** OpenCV escribe `mp4v` (MPEG-4 Part 2), que **los navegadores no
decodifican**: de ahí el reproductor con botón de play y pantalla negra. Este
build de OpenCV no trae H.264 (`avc1` no abre) y no hay `ffmpeg` en el
sistema, así que la salida pasa a **WebM con VP8**, que Chrome, Firefox y Edge
reproducen de forma nativa.

---

## 8. Límite de 30 FPS en la cámara

`RQNF01` exige ≥30 FPS. El selector llegaba a 15.

**Medición del cuello de botella.** Con la GPU libre, la inferencia son **16 ms**
(techo teórico de 60 FPS), pero el ciclo completo eran **45 ms**. Es decir, el
cuello **no era el modelo**: eran 29 ms de HTTP y codificación. El culpable
concreto es que el servidor devolvía el frame ya anotado como JPEG, unos
**100 KB por frame**, cuando el navegador ya tenía esa imagen en su canvas.

**Decisiones:**

1. **Modo `overlay=0`**: el servidor devuelve solo las cajas en coordenadas
   normalizadas y el navegador las dibuja sobre el canvas que ya tiene. La
   respuesta pasa de **104 698 a 340 bytes**.
2. **Recorte del frame en el cliente** a 640 px de ancho antes de enviarlo. El
   modelo hace letterbox a 640 de todos modos, así que mandar la imagen a
   resolución nativa solo costaba ancho de banda y decodificación.
3. **Selector de resolución** 640 / 512 / 448, aunque se comprobó que la
   inferencia es plano a partir de 640 (16.0 ms a 640 y a 512): con este
   modelo el selector no cambia el rendimiento, se mantiene por si cambia el
   modelo.
4. **Selector de cadencia** hasta 30 fps.
5. **FPS medido sobre frames realmente completados**, no sobre el intervalo de
   envío, para que el panel refleje el rendimiento real.

**Resultado medido:**

| modo | Ciclo | FPS |
|---|---|---|
| `overlay=1` (JPEG de vuelta) | 48.4 ms | 20.7 |
| **`overlay=0` (solo cajas)** | **23.6 ms** | **42.4** |

**CUMPLE RQNF01.** El modo `overlay=1` se conserva porque es el que necesita
el endpoint de imagen (subir una foto y verla anotada), donde no hay problema
de rendimiento.

---

## 9. La lección más importante: dos "mejoras" que resultaron peores

Se documentan porque son el resultado más útil de esta vuelta.

### 9.1 Fusionar la clase `persona` (ya  recounted en el punto 1 en el punto 1)

Entrenado y medido: **mAP50 de test 0.706 → 0.587**. Todas las clases bajaron.
Revertido.

### 9.2 Añadir Le2i `Lie` a `persona_caido`: destruyó la clase de pre-caída

La idea era buena y se validó por separado: con el modelo propio, `Lie` pasa
del 66% (COCO) al **91%** de detección, y `persona_caido` crecía de 163 a 1246
cajas. La clase más escasa dejaba de serlo.

Pero al entrenar con esos datos, sobre el **mismo** conjunto de validación:

| modelo | val mAP50 | `persona_desequilibrio` | `persona_erguida` |
|---|---|---|---|
| Sin Le2i `Lie` | **0.810** | **0.920** | 0.733 |
| Con Le2i `Lie` | 0.607 | 0.267 | 0.298 |

**`persona_desequilibrio` se desplomó de 0.920 a 0.267.** La causa es que `Lie`
y `Likefall` son **el mismo sujeto en momentos consecutivos de las mismas cinco
escenas**: antes de la caída y en el suelo. Al meter ambos en el entrenamiento,
caída y pre-caída se vuelven visualmente indistinguibles para el modelo. Las
fotos de stock de `fall` sí las separaban bien, precisamente porque su dominio
es completamente distinto.

**Conclusión.** Más datos no es mejor si los datos son visualmente contiguos a
otra clase. `persona_desequilibrio` y `persona_caido` necesitan fuentes que no
se solapen en el tiempo.

**Estado final:** se entrega `runs/yolov8s_zoom`, que es el mejor modelo
validado. El dataset actual incluye Le2i `Lie` (las correcciones de fuga y
dedup son correctas y se conservan), pero **para reproducir el modelo entregado
hay que regenerar Le2i sin `Lie`**, es decir con el detector COCO por defecto.

---

## 10. Qué NO se ha resuelto

- **La escena real sigue sin estar representada.** Todo el conjunto de test es
  CCTV, interiores de Le2i y fotografía de stock. No hay ni un ejemplo de
  webcam real sobre una escalera con alguien que pasa cerca o de espaldas. La
  evaluación por escala da 100% en todas las condiciones **medibles**, y aun
  así el usuario observa fallos en su escena: el problema está fuera de lo que
  los datos pueden medir. La única vía que lo cierra es grabar con la cámara
  real, y quedó pospuesto por indicación expresa del usuario («estamos
  trabajando con lo del envelope»).
- **`persona_sentado` sigue siendo la clase más escasa** (~100 cajas). Es un
  límite de los datos, no del modelo: el dataset `fall` aporta 75 cajas de esa
  clase.
- **`persona_desequilibrio` tiene 31 instancias en test.** Con los ~12 videos
  base de que parte Le2i `Likefall` no se puede cuadrar el reparto por frames
  sin romper la garantía de que un video entero cae en un solo split. Se
  reporta su AP con ese soporte tan bajo.
- **Las categorías 1, 2 y 3 de RQF02** (obstáculos, pasamanos, distracciones)
  siguen sin cubrir: no hay datos y, según el propio diseño de `Caso.md`, son
  keypoints de MediaPipe, no cajas de YOLO.

---

## 11. Reproducibilidad

```bash
python3 scripts/prepare_le2i.py --self-model runs/<prev>/weights/best.pt
python3 scripts/prepare_dataset.py --clean
python3 scripts/add_zoom_variants.py --split train
python3 scripts/train.py
python3 scripts/evaluate.py --weights runs/yolov8s_5clases/weights/best.pt
python3 scripts/eval_scale.py --weights A.pt --weights B.pt --compare
python3 app/server.py --host 0.0.0.0 --port 5000
```

- Semilla fija 42 en Python, NumPy, Torch y Ultralytics.
- `datasets/combined/manifest.csv`: trazabilidad de cada imagen.
- `datasets/combined/config.json`: parámetros de la preparación.
- `datasets/combined/dropped.txt`: todo lo descartado y por qué.
- `datasets/le2i_yolo/validacion.json`: resultado de la validación de
  pseudo-etiquetas.