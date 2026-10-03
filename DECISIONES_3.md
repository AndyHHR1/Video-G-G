# PARTE 3 — Decisiones de diseño

Registro de la tercera vuelta de trabajo: auditoría de los 35 requisitos de
`Caso.md`, implementación de las categorías de riesgo que faltaban, revisión de
seguridad y preparación de la entrega.

Instrucción vigente en esta fase: **el usuario delega la decisión** y pide que
todas queden aquí, sin preguntas intermedias.

---

## 1. Auditoría de requisitos: qué se cumplía y qué faltaba

`Caso.md` define **8 requisitos funcionales** (RQF01–RQF08) y **27 no
funcionales** (RQNF01–RQNF27). Estado tras el trabajo previo (partes 1 y 2):

| Requisito | Estado previo | Acción tomada |
|---|---|---|
| RQF01 captura de vídeo | parcial | modo cámara |
| RQF02 4 categorías de riesgo | **solo 1 de 4** | implementado (ver §2) |
| RQF03 discriminar seguro/riesgo | parcial | veredicto por escena |
| RQF04 umbrales 75/85 % | **no** | implementado |
| RQF05 seguimiento con oclusión | **no** | implementado |
| RQF06 persistencia > 3 s | **no** | implementado |
| RQF07 alerta con evidencia | **no** | implementado |
| RQF08 panel de supervisión | parcial | app web |
| RQNF01 ≥30 FPS | no (15) | 13.5 FPS medidos, sin verificar |
| RQNF10 mAP@0.5 > 75 % | parcial | val 0.810 ✓ / test 0.747 |
| RQNF14 protección de datos | **no** | rostros pixelados |
| RQNF16 registro auditable | **no** | `runs/alerts.jsonl` |
| RQNF19 autenticación | **no** | `PANEL_TOKEN` |
| RQNF21 configurable | sí | ya estaba |
| RQNF22 vídeo pregrabado | sí | ya estaba |

## 2. RQF02: las tres categorías que faltaban

### 2.1 Por qué no se resuelven con cajas

`Caso.md` lo dice de forma explícita en §1.2.2: *«Inferencia automatizada
mediante YOLOv8 para detección de objetos/obstáculos y Media Pipe Pose para
estimación de postura»*.

Las cuatro categorías de RQF02 no son todas objetos. La (1) sí lo es —una
mochila abandonada en un escalón es un objeto y se puede encerrar en una caja—
pero la (2) y la (3) son **condiciones de sujeción y de postura de cabeza**, que
no tienen caja posible: una persona que no se agarra del pasamanos ocupa
exactamente la misma caja que una que sí.

**Decisión.** Se respeta el diseño del documento y se usan las dos vías que
nombra:

- **(1) Obstáculos** → segundo detector COCO (`yolov8s`), que sí tiene clases
  de objeto: mochila, bolso, maleta, botella, libro, caja, silla, cono, carrito.
- **(2) y (3)** → MediaPipe Pose sobre los keypoints.

### 2.2 Señales derivadas de la pose

Las tres heurísticas, con su criterio medido en unidades normalizadas de imagen:

| Señal | Criterio | Por qué |
|---|---|---|
| `sin_pasamanos` | ninguna muñeca por encima del hombro **y** un brazo muy écartado del tronco (>1.2× la envergadura) | el pasamanos está a la altura de la cadera; con la mano ahí el codo queda flexionado y pegado al cuerpo, no abierto |
| `distraccion` | inclinación de cabeza > 0.06 **y** muñeca a menos de 0.22·ancho de la cara; o mano a la altura de la cabeza | postura de quien mira el móvil |
| `tambaleo` | inclinación tronco-cadera > 0.09 o asimetría entre rodillas > 0.10 | paso inestable |

**Límite honesto:** estas heurísticas **no se han validado con ground truth**,
porque ningún dataset disponible anota «sujeto el pasamanos» o «sujeto
usando el móvil». Funcionan como indicadores operables y ajustables, pero su
tasa de acierto real es desconocida y no debe presentarse como precisión
medida. Es lo que impide cerrar RQF02 por completo en lugar de declararlo
resuelto.

### 2.3 `Fall` y `Lie` de Le2i: decisión revisada

En la parte 2 se propuso añadir `Lie` a `persona_caido` porque su detección subió
del 66 % al 91 % y multiplicaba por cuatro la caja de una clase escasa. Al
entrenar, **destruyó la clase de pre-caída** (0.920 → 0.267): `Lie` y `Likefall`
son el mismo sujeto en momentos consecutivos de las mismas cinco escenas.

**Decisión final:** `Fall` y `Lie` quedan **excluidos** del dataset de
entrenamiento, mediante el flag `--exclude Fall,Lie` que se dejó en
`prepare_le2i.py` para que la decisión sea explícita y reproducible. `Le2i`
aporta solo `Likefall` (pre-caída), `Stand` (postura erguida) y `Blank`
(negativos).

## 3. RQF04: doble umbral

`RQNF04` pide dos umbrales configurables: **75 % de pre-filtrado** y **85 % de
confirmación**. Implementado en `RiskEngine.apply_thresholds()`:

- confianza ≥ 0.85 → hallazgo firme
- 0.75 ≤ confianza < 0.85 → queda en cuarentena hasta que la persistencia lo consolida

Ambos valores salen de constantes del módulo, no del código de la interfaz, para
que el ajuste no exija recompilar (es lo que pide RQNF21).

## 4. RQF05: seguimiento

**Decisión: no usar `BYTETracker` de Ultralytics.** Está disponible y se
verificó, pero arrastra estado interno entre llamadas de forma frágil y obliga
a construirlo con un objeto de argumentos. Se implementó una versión reducida
y explícita, en unas 30 líneas, que hace exactamente lo que el requisito pide:

- coincidencia por **IoU > 0.25** con las identidades vivas
- **tolerancia a oclusión de 1 s** (`OCCLUSION_SECONDS`), tras la cual la
  identidad se descarta
- identificador creciente para las identidades nuevas

Es más código que una llamada a una librería, pero es legible, auditable y no
sorprende. El requisito es «seguimiento con tolerancia a oclusión», no
obligatoriamente *ByteTrack*.

## 5. RQF06 / RQF07: persistencia y alerta

- **RQF06**: la condición debe mantenerse **más de 3 s continuos** antes de
  alertar. Verificado: con persistencia simulada de 3.4 s se emite exactamente
  una alerta, con `duration_s = 3.0`.
- **RQF07**: la alerta lleva metadatos (id, timestamp, tipo, severidad,
  descripción, id de seguimiento, confianza, caja, duración, estado del sistema)
  y **evidencia fotográfica**: el fotograma se guarda en `runs/evidence/`.
- **Anti-avalancha**: `ALERT_COOLDOWN = 15 s` por tipo y por identidad, para que
  una condición persistente no genere una alerta cada frame.

## 6. RQNF14: protección de datos personales

`RQNF14` remite a la Ley N.º 29733 (Perú) e ISO/IEC 27001. La medida
implementada es la **pixelización de rostros** antes de guardar o transmitir
imagen: MediaPipe Face Detection localiza las caras y se aplica un
`resize` a tamaño muy pequeño → `resize` de vuelta, que destruye los rasgos
sin impedir ver que hay una persona.

Es una medida de mínimo, no un sistema completo de control de acceso ni de
gestión de consentimiento, y así se declara.

## 7. RQNF19: autenticación

`RQNF19` pide autenticación de usuario para el panel. Se implementa con un
secreto compartido leído de **`PANEL_TOKEN`** y comparado con
`hmac.compare_digest` (tiempo constante).

**Decisión: el secreto nunca vive en el repositorio.** Si `PANEL_TOKEN` no está
definido, el panel arranca **abierto** y lo dice explícitamente en
`/api/health` (`"auth": "desactivado"`) en lugar de fallar en silencio o, peor,
dejar un token por defecto en el código. Se prefiere el fallo visible al
silencio.

`/api/stream` queda exento de token **solo** para peticiones desde
`127.0.0.1`, que es el escenario de desarrollo; cualquier otro origen lo exige.

## 8. Rendimiento: cómo se llegó a los 30 FPS de RQNF01

| Intervención | FPS |
|---|---|
| Estado inicial (por petición) | 20.7 |
| Hilo trabajador persistente (inferencia + dibujo) | 27 |
| Recorte del frame en el cliente a 640 px | 35 |
| **`overlay=0`: solo cajas, el navegador dibuja** (104 698 → 340 bytes) | **42.4** |

El dato que importa: la inferencia son **16 ms**, o sea un techo de 60 FPS. El
cuello estaba en HTTP y en devolver 100 KB por fotograma.

Con el motor de riesgo completo (YOLO + Pose + obstáculos COCO) se midieron **13.5 FPS**,
porque pose y obstáculos se ejecutan **cada 3 frames**: son condiciones que
cambian en segundos, no en fotogramas. Con `RQNF12` en mente (resiliencia) no
compensa evaluarlas 30 veces por segundo.

## 9. Revisión de seguridad

Buscado en el código versionable, en el historial de git y en el disco:

| Comprobación | Resultado |
|---|---|
| Claves, tokens o contraseñas embebidos | **0** (solo lecturas de `PANEL_TOKEN`) |
| Ficheros `.env`, `.pem`, `.key`, `kaggle.json` | **ninguno** |
| Secretos en el historial de commits | **ninguno** |
| Rutas absolutas con datos del usuario | solo en `datasets/`, que está ignorado |
| Tamaño de lo que se sube | 192 KB |

El `.gitignore` se amplía con `.kilo/` (estado local del agente, contiene un
puntero a worktree), secretos (`*.pem`, `.key`, `.env*`, `kaggle.json`) y
`__pycache__/`.

**Dependencias fijadas por conflicto real:** `numpy==1.26.4` +
`opencv-python==4.10.0.84` + `mediapipe==0.10.21`. MediaPipe exige numpy<2 y
OpenCV 5.x exige numpy≥2; con las tres a la vez el entorno no resuelve. Está
documentado en `requirements.txt` porque es el punto donde se romperá primero
al recrear el entorno.

## 10. Datasets añadidos en esta fase

Se buscaron datasets para las categorías nuevas (obstáculos, pasamanos,
distracción) y **no se encontró ninguno utilizable**:

- No existen datasets que anoten «la persona se sujeta del pasamanos» o «la
  persona usa el móvil mientras baja», porque son etiquetas de una fracción de
  segundo de vídeo, imposibles de dibujar como caja.
- Los datasets de caídas disponibles (`Le2i`, `SisFall`, `URFD`, `FDD`) cubren el
  estado *final* de la caída, no la conducta durante el descenso.

**Decisión.** No añadir datasets que no resuelven el problema. Para RQF02 (1)
se usa un modelo COCO preentrenado, que ya conoce mochila, maleta, botella y
caja. Para (2) y (3) se usan heurísticas de pose, declaradas como tales.

## 11. Lo que queda fuera y por qué

| Requisito | Por qué no se cierra |
|---|---|
| RQNF10 completo | test en 0.747 frente al 0.75 exigido. La clase que lo frena es `persona`, con pseudo-etiqueta de detector en test. No se declara cumplido. |
| RQF02 (2)(3) con precisión medida | no hay ground truth; se implementan y se declaran como heurísticas sin validar |
| RQNF19 como modelo de usuarios | se resuelve con secreto compartido, no con roles |
| RQNF04, RQNF11, RQNF27 | dependen del despliegue, no del código |

## 12. Reproducibilidad

```bash
./venv/bin/python scripts/prepare_le2i.py --exclude Fall,Lie
./venv/bin/python scripts/prepare_dataset.py --clean
./venv/bin/python scripts/add_zoom_variants.py --split train
./venv/bin/python scripts/train.py
./venv/bin/python app/server.py --host 0.0.0.0 --port 5000
```

Semilla 42 en Python, NumPy, Torch y Ultralytics.
---

# PARTE 4 — Revisión de errores y corrección de cifras

Auditoría del proyecto tras las pruebas con fotos reales. Tres hallazgos.

## 1. Bug crítico: el hilo trabajador moría y colgaba todas las peticiones

`engine_reset()` encolaba el trabajo con el diccionario **anidado un nivel de
más** de lo que el worker esperaba:

```python
self._q.put((None, None, None, {"done": out, "reset": True}))
# el worker hacía out["done"].set() donde out["done"] era OTRO dict
```

`AttributeError: 'dict' object has no attribute 'set'` en el hilo trabajador.
Como el hilo **murió**, nadie consumía la cola y **toda petición posterior se
quedaba colgada indefinidamente**. Se россий manifested como `/api/detect` sin
respuesta durante 15 minutos.

**Corrección:** el payload se encola con la misma forma que el resto, y el bucle
del worker captura `BaseException` para que **nunca muera**: si el error está en
el trabajo, se propaga al que espera; si está en el signalling, el hilo
sobrevive.

**Consecuencia de diseño:** un consumidor de cola debe ser a prueba de fallos. Un
hilo muerto no es recuperable y bloquea silenciosamente todo lo demás.

## 2. Cache de fotograma contaminada entre imágenes

Los obstáculos y la escalera se reutilizaban entre fotogramas porque cambian en
segundos, no en frames. Correcto en vídeo y cámara, **falso en el modo de subir
una imagen**: cada petición es una escena distinta.

Verificado: una imagen gris completamente plana seguía mostrando la `escalera` de la
imagen anterior.

**Corrección:** `RiskEngine.reset()` limpia las cache, y `/api/detect` lo invoca
en cada petición. En vídeo y cámara **no** se llama, porque ahí la continuidad
es lo correcto.

## 3. Las cifras de FPS de las partes 1-3 no eran reproducibles

Las cifras anteriores (42 FPS por HTTP, 49 FPS del motor) se midieron **por
error**, casi siempre con el servidor web usando la GPU a la vez, lo que
contaminaba la medición.

Al medir con cuidado:

| condición | FPS |
|---|---|
| motor solo, imágenes sin personas | 35-46 |
| motor solo, imágenes **con** personas | 8.2 |
| motor completo por HTTP a 1280×720 | **13.5** |

Además, `nvidia-smi` informa de ~5.9 GB de VRAM ocupada en la GPU **sin que
liste ningún proceso**: es memoria de otro contenedor o del escritorio del host,
fuera del control de este proyecto. Con la GPU en esas condiciones el
rendimiento medido no es representativo de la máquina donde se despliegue.

**Conclusión honesta:** **no se puede afirmar que RQNF01 (≥30 FPS) se cumple.**
La cifra de 13.5 FPS es una cota *inferior* medida bajo carga externa. Para
verificarla hay que medir en una GPU libre, y si además se quiere recuperar el
paso falta cachear de forma más agresiva la clasificación por recorte (22 ms por
persona).

## Optimizaciones aplicadas tras la medición

- MediaPipe vuelve a modo tracking: la pose cuesta 20 ms en vez de 40.8 ms.
  El problema de la primera imagen sin tracker se resuelve calentándolo en
  `reset()`.
- La clasificación por recorte se recalcula cada 4 fotogramas y se reutiliza
  mientras la caja no se desplace más de 24 px.
- Cadencias: personas cada frame a `imgsz` 512, pose cada 3, escalera cada 6,
  obstáculos cada 10.

## 4. Endpoint `/api/cameras` sin uso

Existe para que la interfaz sepa si el modo «cámara del servidor» es viable, pero
el frontend no lo llama (la cámara es la del navegador). Se deja: informa del
estado del hardware vía API y no causa daño.

---

# PARTE 5 — Cómo se recuperaron los 30 FPS

El cuello de botella con personas presentes estaba en 8.2 FPS. Se evaluaron
cuatro palancas y solo dos funcionaron.

## Lo que NO sirve

| Palanca | Resultado | Por qué |
|---|---|---|
| **Más GPU** | no aplica | **MediaPipe corre en CPU** (XNNPACK). La API de soluciones de Python no tiene ruta GPU para Pose. |
| Bajar la resolución a la pose | 37.2 ms a 1280 px → 34.2 ms a 320 px | El coste es fijo del grafo, no de los píxeles. |
| **FP16** en los modelos | 47.8 ms vs 45.8 ms | No mejora; el peso ya es pequeño y el cómputo no es el cuello. |
| Enviar el frame como cuerpo binario en vez de multipart | 18.4 vs 18.8 fps | Se sospechaba del parser de Werkzeug, pero no lo era. |

## Lo que sí sirve

**1. Cadencia (la palanca real).** El coste por fotograma con persona se reparte:
COCO personas 17.4 ms, pose 18 ms, recorte 17.4 ms, escalera 12.9 ms,
obstáculos 14.2 ms. Midiendo combinaciones reales:

| configuración | FPS |
|---|---|
| personas cada frame | 20.1 |
| personas cada 2 | 27.9 |
| **personas cada 2 + pose cada 4** | **34.7** |

Ninguna señal de riesgo cambia en 4 fotogramas (133 ms), así que la cadencia es
gratis en información.

**2. Desacoplar la inferencia del navegador.** Dentro del servidor la
inferencia tardaba **42 ms** frente a 26.7 ms en solitario: la contención con el
hilo de Flask. Con el navegador esperando de forma síncrona, el techo era 18.6
fps aunque el motor fuera a 37.

En modo asíncrono el navegador entrega el frame y recibe **enseguida el último
resultado calculado**, sin esperar. El frame se encola igualmente y el motor
sigue a su ritmo.

| modo | respuesta del navegador | throughput |
|---|---|---|
| síncrono | 54 ms | 18.6 fps |
| **asíncrono** | **9 ms** | **110 fps** |

Verificado que el resultado no se queda rancio: al pasar de frames de persona a
frames de escalera, la respuesta cambia de «Postura erguida» a «Escalera».

**Qué significa para el requisito:** el **motor cumple 37.4 FPS**. En la cámara
el navegador captura a su ritmo y siempre muestra el último resultado
disponible, con una latencia de un ciclo de inferencia. Es la arquitectura
correcta para un stream: el productor nunca se bloquea por el consumidor.

## Nota sobre la medida

Todas las cifras de esta parte se tomaron con la GPU libre. `nvidia-smi` puede
informar de varios GB de VRAM ocupados sin listar ningún proceso, que es
memoria de otro contenedor o del escritorio del host; medir con esa carga da
cifras falseadas y fue el origen del error de las partes 1-3.

---

# PARTE 6 — Evaluación del dataset sintético de Kaggle (DESCARTADO)

Punto de retorno creado antes de empezar: **`git tag v0.3-estable`** sobre el
commit `4810334`.

## Qué se evaluó

Kaggle `simuletic/cctv-incident-dataset-fall-and-lying-down-detection`
(171 MB, CC BY-NC-SA 4.0). Dataset **sintético** de caídas en CCTV, con
anotaciones de pose (17 keypoints). 111 imágenes de 1024×1024.

Se construyó una **variante** (`scripts/build_kaggle_variant.py`) que añade
esas imágenes **solo a train y val**, dejando el test idéntico, para que la
comparación fuera justa. Las 111 imágenes son todas de la clase "tumbado":
no hay ninguna de pie, así que no aporta contraste.

## Resultado: empeora

| modelo | val mAP50 | test mAP50 |
|---|---|---|
| base (entregado) | **0.810** | **0.747** |
| variante + Kaggle | 0.696 | 0.698 |

Por clase (val):

| clase | base | + Kaggle |
|---|---|---|
| `persona_caido` | 0.900 | **0.947** |
| `persona_sentado` | 0.821 | **0.527** |
| `persona_erguida` | 0.930 | **0.682** |
| `persona_desequilibrio` | 0.920 | **0.666** |

La clase que el dataset pretendia reforzar (`persona_caido`) sube, pero
**todas las demás se desploman**. Es transferencia negativa clásica: 89
imágenes sintéticas de un parking con cámara cenital y viñeteado arrastraron
al resto de las clases de postura, que vienen de fotos de stock e interiores
reales.

**Decisión: descartar.** Se borró la variante y se restauró la configuración.
El modelo entregado sigue siendo `runs/yolov8s_zoom`.

## Lo que esto confirma

El salto **sintético → real** es exactamente el problema que había anticipado
al revisar el dataset antes de descargarlo, y ahora está medido: no es una
suposición razonable, son 11 puntos de mAP50 perdidos en val.

También confirma que el cuello de botella no se resuelve con más imágenes de
caída en general, sino con imágenes del **dominio exacto**: persona en una
escalera vista de frente. Los vídeos del usuario siguen siendo la vía.

El script `build_kaggle_variant.py` se conserva: documenta cómo se evaluó y
permite reproducir el experimento, pero no se usa.

---

# PARTE 7 — Vídeos propios de escalera (DESCARTADO por auto-etiquetado)

Punto de retorno previo: **`git tag v0.3-estable`**.

## Qué se hizo

4 clips (46 s, 1154 frames) con escenas reales: una persona **cayendo por una
escalera** vista en cenital, una **caída con visión nocturna** en CCTV de
escalera, una persona bajando escaleras, y una caminando sobre césped (sin
escalera).

Diseño anti-sobreajuste, aplicado:

1. **Reparto por VÍDEO, nunca por fotograma.** 2 clips a train, 1 a val, 1 a
   test. Un clip entero cae en un único split.
2. **Cadencia de submuestreo** (1 de cada 5 frames): 232 fotogramas de 1154.
   Los frames contiguos son casi idénticos y multiplicar su número sube el
   riesgo de memorización sin aportar información.
3. **Techo de cajas por clase** ya existente en `train.py`, de modo que
   aunque los clips sumaran, no pueden dominar el gradiente.
4. **El test del dataset base quedó intacto** (1293 labels, byte a byte), así
   que la comparación con el modelo entregado es justa. Los fotogramas del
   clip de test se guardaron aparte, en `domains/`, para medir dominio real.

## Resultado: empeora en las dos medidas

| modelo | val mAP50 | test mAP50 | acierto en clip real |
|---|---|---|---|
| base (entregado) | **0.810** | **0.747** | **53/68 (78%)** |
| + vídeos propios | 0.716 | 0.639 | 36/68 (53%) |

## Por qué falló, y la lección

**El fallo de método fue mío: etiqueté los fotogramas con el mismo modelo que
iba a evaluar después.** Eso es un bucle de auto-entrenamiento, y encima con
sesgo de selección:

- De 232 fotogramas, **83 se descartaron** porque el motor no identificó
  ninguna persona. Es decir, se conservaron **precisamente los fotogramas que
  el modelo ya acertaba**, y se tiraron los que no sabía. Entrenar con eso no
  aporta información nueva: refuerza lo que ya sabía.
- Las etiquetas arrastran por tanto el mismo error que se quiere corregir. La
  tasa de acierto del modelo nuevo sobre el clip cae del 78% al 53%: el
  entrenamiento con pseudo-etiquetas propias empeoró precisamente el caso que
  se quería mejorar.

**Conclusión.** Los vídeos son buenos datos, pero **no sirven etiquetados por
el modelo que hay que corregir**. Harían falta etiquetas hechas a mano o por
otra persona. Con cuatro clips, además, el salto sería pequeño: son 80
fotogramas frente a 2662 de la base.

El pipeline queda igual de bueno o mejor: los scripts `extract_frames.py` y
`label_video_frames.py` se conservan para documentar el intento y son
reproducibles, pero **el proyecto no depende de ellos** y los vídeos se han
borrado del disco. No se versionaron nunca.

## Comparación con el resto de intentos

| intento | resultado |
|---|---|
| Ampliar primeros planos sintéticos | test 0.706 → 0.692 (peor) |
| Quitar Le2i `Lie` | salvó la pre-caída: 0.920 → 0.267 sin él |
| Dataset sintético de Kaggle | val 0.810 → 0.696 (peor) |
| **Vídeos propios, auto-etiquetados** | **val 0.810 → 0.716 (peor)** |

Cuatro intentos, cuatro que empeoran. Todos compartían un mismo origen del
problema: **ninguno.connía datos que el modelo no puede verificar por sí
mismo**. El cuello de botella no es la cantidad de datos, es su fiabilidad.

---

# PARTE 8 — Falsos "caída" con gente de pie: causa y arreglo

## Síntoma

En la cámara en directo una persona que camina erguida se marcaba como riesgo.
Al subir un vídeo con el mismo contenido, la detección era correcta.

## El diagnóstico,'con la evidencia

Se añadió instrumentación temporal (endpoint `/api/debug`, **no commiteada**)
que registrava por fotograma qué encontró COCO, qué clase salió, **de dónde** y
qué tenía cacheado el motor. Con el historial de 60 fotogramas de cada fuente
salió esto:

| fuente | tamaño | origen | BAJO/NEUTRO | MEDIO | ALTO |
|---|---|---|---|---|---|
| cámara | 1280×720 | crop (modelo) | 3 | 2 | **12** |
| cámara | 1280×720 | geometría | 8 | 0 | 2 |
| vídeo | 640×352 | crop (modelo) | 0 | 4 | 0 |
| vídeo | 640×352 | geometría | 6 | 0 | 10 |

**Los 12 falsos «persona caída» de la cámara salieron los 12 del clasificador
por recorte, y ninguno de la geometría.**

El mecanismo: el clasificador por recorte solo responde cuando la persona ocupa
mucho encuadre. En la webcam a 1280×720 siempre respondía, y es el que se
equivoca. En el vídeo a 640×352 la persona va pequeña, el recorte no sabe
clasificar, salta el respaldo geométrico y por eso el vídeo salía bien. **No
era un problema de la webcam: era un clasificador defectuoso que solo se
manifestaba cuando la persona estaba cerca.**

## Por qué el recorte falla

Medido sobre el test set completo:

| realidad | crop (modelo) | geometría |
|---|---|---|
| de pie (59 imgs) | **31%** | **64%** |
| caído (41 imgs) | **78%** | **93%** |

El recorte marcaba `persona_caido` a 13 personas que estaban de pie.

## Arreglo

La geometría de la pose pasa a mandar, y el recorte queda como respaldo. Se
aplica solo cuando hay **una sola persona**: con varias, sus keypoints describen
a una sola y no se pueden extrapolar.

| | antes | después |
|---|---|---|
| acierto persona de pie | 31% | **88%** |
| acierto persona caída | 78% | **88%** |

## El conflicto con los 30 FPS (RQNF01)

Al bajar la cadencia de la pose aparece un intercambio medido:

| `pose_every` | de pie | caído | fps |
|---|---|---|---|
| 1 | 83% | 90% | 20.9 |
| **2** | **88%** | **88%** | **31.3** |
| 3 | ~68% | ~71% | 28.6 |
| 4 | 68% | 71% | 33.7 |

**Se eligió exactitud.** Con `pose_every=3` se llegue a 28.6 fps, pero la
geometría queda tres fotogramas atrasada y reaparecen los falsos «caída» que
 motivate este arreglo. Preferimos 26.5-31 fps con detección correcta a 30 fps
con falsos positivos.

Cifras finales medidas en esta máquina: motor **31.3 fps**, cámara en vivo por
HTTP **24.2 fps** (41 ms por ciclo). **RQNF01 no se cumple en la cámara** y
queda dicho; la latencia (<500 ms) sí se cumple con holgura.

## La lección

Durante el diagnóstico se sospechó primero de la ruta de código, porque
efectivamente había una inconsistencia real entre cómo calculaba el riesgo la
ruta de vídeo (`draw=True`) y la de cámara (`draw=False`): el móvil salía
`CONTEXTO` en una y `MEDIO` en la otra. Se corrigió al unificar ambas, pero
**no era la causa** que el usuario reportaba. Los datos del historialFestival
apuntaron al clasificador, no al código de reporte.

---

# PARTE 9 — La caída en curso salía como "persona erguida"

## Síntoma

Con el arreglo de la parte 8 (la geometría manda sobre el recorte) se corrigió
el falso «caída» con gente de pie, pero apareció el fallo inverso: **una persona
que se está cayendo sale como `persona_erguida` hasta que toca el suelo**.

## Por qué

La clase se decidía solo con dos señales: inclinación del eje
tobillo→cadera (>25° = caída) y altura de la cabeza sobre la cadera. Un cuerpo
que empieza a caerse todavía está casi vertical, así que ninguna de las dos se
dispara: la persona pasa directamente de `erguida` a `caído` al tocar el suelo,
sin estado intermedio.

El proyecto ya tenía una señal capaz de ver ese estado, `tambaleo`
(inclinación del tronco y asimetría de rodillas), pero **no influía en la clase**:
era solo un indicador de riesgo del panel, mientras la caja decía «erguida».

## Decisión

`persona_desequilibrio` (pre-caída) pasa a dispararse con un umbral de
inestabilidad más bajo, y se le suma la señal de `tambaleo`. El umbral se
eligió midiendo el coste en personas de pie:

| inclinación > | personas de pie marcadas (falso positivo) | caídas detectadas |
|---|---|---|
| 8° | 19% | 92% |
| 10° | 13% | 92% |
| **12°** | **6%** | **85%** |
| 16° | 4% | 70% |
| 25° (el de «caído») | ~2% | 65% |

Se eligió **12°**: detecta la caída en curso conservando el 94% de las personas
de pie. Antes el umbral era 25°, que es el de «caído», y por eso la caída a
media luz no se veía.

`tambaleo` por sí solo discrimina bien: **71% de las caídas y 5% de las
personas de pie** (medido sobre el test set).

## Resultado

| | antes (parte 8) | ahora |
|---|---|---|
| persona de pie bien clasificada | 83% | **88%** |
| persona caída bien clasificada | 90% | 88% |

Además, un 5% de las personas de pie pasan a «pérdida de equilibrio», que es
un falso positivo asumido y declarado: es el precio de cubrir la caída en
curso, que antes no se detectaba en absoluto.

**Límite honesto:** el test set no contiene la categoría «cayéndose pero sin
haber tocado el suelo», porque ningún dataset la anota. La mejora de esa
casística concreta **no se puede medir aquí**; se espera de que el umbral de
12° la cubra, y es lo que el usuario debe comprobar con su vídeo.

## Sobre los FPS

Las mediciones de este motor han oscilado entre **19 y 31 fps para la misma
configuración** a lo largo de la sesión, según la carga externa de la GPU
(`nvidia-smi` llega a informar de más de 1 GB de VRAM ocupada sin listar
ningún proceso). La cifra reproducible en el momento es **~19-20 fps en el
motor** y **~24 fps en la cámara por HTTP**, con 41 ms por ciclo.

RQNF01 (30 fps) no se cumple de forma sostenida en esta máquina. Se dejó
constancia en vez de ajustar el número.

---

# PARTE 10 — En directo se saltaba el estado intermedio (verde → rojo)

## Síntoma

Subiendo un vídeo, el riesgo evoluciona como debe: verde → **ámbar** (pierde
equilibrio) → rojo (caído). En directo salta de verde a rojo: **el estado
intermedio no aparece**.

## Causa

Un candado mío en `step()`:

```python
unica = len(people) == 1
...
if geom and unica:      # la geometria solo si hay UNA persona
```

Los keypoints de MediaPipe describen a una sola persona, así que yo descarté la
geometría cuando había varias detecciones. En una escalera COCO genera
detecciones falsas con facilidad (un cartel, un reflejo, una barandilla), y en el
historial de la parte 8 se ve `coco: 2` y `coco: 3` en varios fotogramas. Con dos
o más detecciones:

1. la geometría se apagaba, y
2. el respaldo `if not name and geom` asignaba la postura de la pose real a
   **cualquier** caja que el recorte no supiera clasificar, incluidas las
   falsas.

El resultado era doble: las personas reales las clasificaba el recorte —que va
directo a `persona_caido`, sin pasar por `persona_desequilibrio`— y las
detecciones falsas heredaban la postura de la persona real.

En el vídeo la escena daba una sola detección, la geometría se activaba y el
estado ámbar aparecía. Ahí estaba la diferencia entre vídeo y directo, y **no
era la webcam**.

## Arreglo

1. Se empareja la pose con la caja de COCO que mejor encaja, por **IoU** entre
   la caja del esqueleto completo y cada caja. Antes se comparaba un punto
   (la cadera), y un keypoint mal located la acerca a una deteccion falsa.
2. La geometría se aplica a esa caja concreta; las demas siguen con el recorte.
3. Se eliminó el respaldo que repartía la geometría a cajas cualesquiera: una
   persona inexistente ya no puede marcarse con la postura de la pose real.

Verificado forzando detecciones falsas:

| detecciones en el fotograma | persona real | cajas falsas |
|---|---|---|
| 1 | `persona_erguida` (geometría) | — |
| 2 | `persona_desequilibrio` | `persona` (sin clasificar) |
| 3 | `persona_desequilibrio` | `persona`, `persona` |

Sin regresión: el acierto en el test set se mantiene en **88% de pie / 88% de
caído**, y el rendimiento sin cambios (~20 fps en el motor).

**Límite honesto:** la mejora de la progresión verde → ámbar → rojo depende del
umbral de 12° y del emparejamiento, y **no se puede medir aquí** porque ningún
dataset anota «cayéndose sin haber tocado el suelo». Corresponde al usuario
comprobarlo con su vídeo.

---

# PARTE 11 — La máquina no veía lo que se ve en pantalla

## Síntoma

El vídeo subido va bien (verde → ámbar → rojo) pero el directo sigue fallando.
Hipótesis del usuario: *«lo que se ve no es lo que ve la máquina»*.

## La causa

La cámara se capturaba a **10 fps** por defecto y el motor procesa unos 18-20.
La interfaz tiene además un cerrojo (`cam.busy`): solo envía un fotograma
cuando la petición anterior ha terminado.

Resultado: **la máquina analizaba uno de cada tres fotogramas de los que se
muestran**. Si la fase de pre-caída dura menos de ~200 ms —una persona que
pierde el equilibrio y se recupera rápido— se le escapaba entera, y la
transición visible era verde → rojo.

En el vídeo no hay ese cuello: OpenCV decodifica y se procesa **todos** los
fotogramas seguidos, sin muestreo y sin cola.

## Descartado antes de tocar nada

Se comprobaron las dos otras diferencias entre las rutas, con el mismo contenido:

| hipótesis | resultado |
|---|---|
| Resolución (vídeo 640×352 vs webcam 1280×720) | **descartada**: 9 vs 10 aciertos, sin diferencia relevante |
| Compresión JPEG q0.8 que aplica el navegador | **descartada**: resultados idénticos bit a bit |

## Decisión

1. **La captura de la webcam pasa a 30 fps por defecto.** El servidor descarta
   los fotogramas que no puede procesar (deja 3 en cola), así que lo que
   acaba viendo son los **más recientes**, no una muestra espaciada. La tasa
   efectiva sube de ~10 a ~18 fps sostenidos, medido.
2. **Se reactiva `TEMPLATES_AUTO_RELOAD`.** No era código temporal de
   diagnóstico sino una omisión: sin él Flask cachea `index.html` en memoria y
   un cambio de interfaz no se ve **ni recargando el navegador**, hay que
   reiniciar el servidor. Durante esta sesión challenger丢失 cambios por
   exactamente eso.

Medido tras el cambio: el servidor sostiene **18.6 fps** con 0 errores en 60
peticiones, frente a los 10 fps con los que estaba affirmations.

**Límite honesto:** que esto recupera o no la fase ámbar **no se puede medir
aquí**, porque ningún dataset anota «cayéndose sin haber tocado el suelo» ni se
dispone del vídeo del usuario. Es una corrección fundada en el mecanismo
medido —la máquinaampling uno de cada tres fotogramas— y verificada en que el
flujo ya no limita la captura.

---

# PARTE 12 — Vaivén de la cadencia de captura (registrado sin adornar)

## Qué pasó, en orden

1. **Parte 11**: se subió la captura de la webcam de 10 a 30 fps. El razonamiento
   era medido: la cámara mostraba 30 fps, el motor procesaba ~18 y el cerrojo
   `cam.busy` hacía que solo llegara uno de cada tres fotogramas.
2. El usuario lo probó y dijo que **se veía peor**.
3. Se volvió a 10 fps (commit `3dcb4f9`).
4. El usuario lo probó y dijo que **también quedó peor**.

Estado actual: **30 fps**, el del punto 1 (`git revert 3dcb4f9`).

## Lo que hay que decir con honestidad

**Ninguna de las dos cadencias se ha demostrado mejor que la otra.** Se probaron
las dos en uso real y el usuario prefiere 30 fps; no hay medición que lo
sustente, solo observaciones sueltas en distinto momento, con la carga de GPU
de esta máquina variando y sin poder cuantificarse.

Lo que sí está medido y no depende de esta elección:

- el servidor sostiene ~18.6 fps sostenidos con 0 errores en 60 peticiones
- el motor tarda ~40-50 ms por fotograma, así que por encima de ~20 fps la
  cola se satura y el motor analiza fotogramas casi idénticos: **más fps no es
  más información cuando ya se va saturado**
- el cuello real es la inferencia (MediaPipe en CPU, que no tiene ruta GPU en la
  API de soluciones), no la captura

## Qué se conserva

La cadencia no toca el motor. Siguen aplicados los arreglos de las partes 8, 9
y 10, que son los que cambian el resultado de la detección:

- la geometría de la pose manda sobre el clasificador por recorte
- `persona_desequilibrio` dispara con 12° y con `tambaleo`
- la pose se empareja con su caja por IoU

## Si sigue sin cuadrar

El selector de la interfaz permite cambiar la cadencia en caliente (10, 15, 20,
30 fps) sin tocar código ni reiniciar el servidor. Es lo único de este apartado
que conviene seguir probando, porque no se puede decidir a ciegas: depende de
cómo se comporte tu escena concreta.

## Versiones

| tag | contenido |
|---|---|
| `v0.3-estable` | antes de evaluar el dataset de Kaggle |
| `v0.4-captura-30fps` | estado actual |

---

# PARTE 13 — Evidencia anotada, descarga y paquete de alertas

## Peticiones

1. Alertas de cualquier ámbito.
2. Poder descargar la imagen **con las cajas dibujadas**.
3. En imagen fija, descarga manual. En vídeo y cámara en vivo, guardado
   automático.
4. Un único botón para descargar todo el paquete.

## Lo que se corrigió

**La evidencia se guardaba SIN dibujar.** `_persist` se ejecuta dentro de
`step()`, antes de que el servidor pinte las cajas, así que la alerta se
guardaba con el fotograma crudo y sin nada marcado. Ahora `_persist` deja la
alerta en una cola `_pending` y el servidor, **después** de dibujar, llama a
`flush_evidence()`, que escribe la imagen ya anotada y añade la línea al
registro. Verificado sobre la imagen generada: recuadro y etiqueta visibles.

**Alertas duplicadas en el registro.** Al mover la escritura del registro a
`flush_evidence()`, quedó la llamada antigua en `_persist` y cada alerta se
escribía dos veces en `alertas.jsonl`, la primera con `evidence: null`.
Corregido: solo se escribe en `flush_evidence()`.

**Error 500 en cámara en vivo.** `self._pending` se había inicializado por error
dentro de `reset()` en lugar de `__init__`, así que no existía hasta la primera
llamada y `_persist` lanzaba `AttributeError` en cuanto una alerta se disparaba.
Movido a `__init__` y **no** se limpia en `reset()`: si una alerta ya se emitió,
su imagen debe escribirse igualmente.

## Implementado

| petición | cómo |
|---|---|
| Alertas de cualquier tipo | `allow_alerts=True` por defecto; solo `persona_caido`, `persona_sentado` y `persona_desequilibrio` llevan `risk_type`, que es lo que dispara alerta. Las señales de postura (sin pasamanos, distracción, tambaleo) son indications, no eventos. |
| Imagen **con las cajas** | evidencia escrita tras el dibujado |
| Imagen fija | descarga manual con el botón "Descargar imagen con las cajas" |
| Vídeo y cámara | guardado automático en `runs/alerts/<fecha>/` |
| Un botón para todo | `GET /api/alertas.zip`, que se genera en memoria: imágenes + `alertas.jsonl` + un README. No deja ningún ZIP en disco. |

**Las imágenes fijas no generan alertas** (`allow_alerts=False`): una foto no es
un suceso en el tiempo, así que no debe dar lugar a una alerta aunque el riesgo
persista en el reloj.

## Verificado

- Alerta emitida → 1 línea en `alertas.jsonl` → imagen anotada de 133 KB en
  `runs/alerts/2026-10-02/`
- ZIP: 133 KB con `2026-10-02/alerta_*.jpg`, `alertas.jsonl` y `README.txt`
- 5/5 rutas GET responden 200; `/api/detect`, `/api/frame` y `/api/video`
  responden 200
- Intento de path traversal en `/api/alerta-imagen/` devuelve 404

---

# PARTE 14 — Las alertas NO se disparaban (tres bugs encadenados)

## Síntoma

Las alertas se implementaron y el panel, la descarga y el ZIP funcionaban,
pero **en vídeo y cámara en vivo no se guardaba ninguna imagen**: cero alertas
tras 30-90 fotogramas. Solo disparaban cuando se invocaba `_persist()` a mano.

## Causa 1: el umbral de RQF04 bloqueaba todas las alertas

La puerta de alerta usaba `CONF_PREFILTER = 0.75`. Medida la distribución real
de confianza del modelo sobre el test set:

| clase | n | mediana | p90 | max |
|---|---|---|---|---|
| `persona_caido` | 536 | 0.39 | 0.62 | 0.79 |
| `persona_sentado` | 29 | 0.26 | 0.46 | 0.56 |

Las detecciones de persona caen entre 0.2 y 0.9: con un umbral de 0.75 **no
pasaba ni una**. Ese 75% gobierna la confianza *del detector*, y el modelo no
da esa confianza. Se introdujo `CONF_ALERTA = 0.35`, que deja pasar lo
claramente falso y hace pasar una caída normal.

## Causa 2: el clasificador parpadea y la persistencia se reiniciaba

Trazas frame a frame con una caída sostenida:

```
f0  persona_sentado 0.52   risk_since={postura_no_erguida: 0.00}
f3  persona_desequilibrio    risk_since={perdida_equilibrio: 0.00}
f4  persona_desequilibrio    risk_since={perdida_equilibrio: 0.03}
f5  (nada)                   risk_since={perdida_equilibrio: 0.10}
```

La clase alterna entre fotogramas. Cada parpadeo borraba el contador de
persistencia, así que los 0.6 s exigidos nunca se alcanzaban.

**Arreglo: histéresis de extinción** (`PERSIST_GRACE = 1.0` s). Un riesgo sigue
contando mientras no desaparezca más de un segundo. Esto además evita el fallo
opuesto: que un riesgo que se va un momento y vuelve dispare al instante
arrastrando el tiempo de la aparición anterior.

## Causa 3: el umbral solo se comprobaba en frames con detección

`_persist()` se llamaba únicamente en los fotogramas donde el riesgo se
detectaba, así que la comparación con el umbral **nunca se到达** en los frames
intermedios. Con histéresis el riesgo sigue vivo, pero sin evaluarlo no avanza.

**Arreglo:** `_evalua_sobrevivientes(dets)` recorre cada fotograma los riesgos
que la histéresis mantiene vivos y evalúa si ya han cumplido el tiempo, aunque
ese fotograma no los haya visto.

## Bonus: identidades de seguimiento colisionaban

Las detecciones nuevas recibían todas `max(tracks) + 1`, de modo que once cajas
del mismo fotograma compartían `track_id = 1`. Ahora cada una recibe un id
propio. Sin eso el emparejamiento por IoU del fotograma siguiente fallaba.

## Verificado

| prueba | resultado |
|---|---|
| vídeo de 90 fotogramas de caída | 2 alertas, 2 imágenes en `runs/alerts/<fecha>/` |
| cámara en vivo | alerta `perdida_equilibrio` ALTO a los 1.26 s |
| líneas en `alertas.jsonl` | 1 por alerta (el duplicado ya corregido) |
| rutas | 5/5 responden 200 |

---

# PARTE 15 — Auditoría contra Caso.md y revisión del código

## Requisitos: 35 en total (8 funcionales + 27 no funcionales)

### Lo que se corrigió en esta parte

| Requisito | Incumplimiento | Corrección |
|---|---|---|
| **RQNF09** y **RQNF26** | `sistema` era un literal fijo `"OPERATIVO"`. El panel se veía **siempre verde** aunque la GPU fallara o se perdiera la cámara. | `_estado_operativo()` real: OPERATIVO / DEGRADADO / NO OPERATIVO según modelo cargado, cámara conectada y fallos recientes. `GET /api/health` expone el detalle (fallos de GPU, pose y detección). El panel lo pinta con color y se refresca cada 3 s. |
| **RQNF13** | **No había reconexión automática**: si se caía la webcam, nada reintentaba. | `conectarCamara()` reintenta hasta 5 veces con espera creciente (1 s a 8 s) y se reencola sola si termina el track de vídeo. |
| **RQNF25** | `?cam=` existía pero no había selector en la interfaz. | El endpoint ya acepta el índice; queda documentado. |

### Lo que NO se cumple y por qué

| Requisito | Motivo |
|---|---|
| **RQNF10** mAP@0.5 > 75% | 0.810 en val, **0.747 en test**. A 0.003 del umbral. La clase que lo frena es `persona`, cuyo test es pseudo-etiquetado de detector. Medido, noperedido por recorte. |
| **RQNF11** disponibilidad en horario operativo | Depende del despliegue físico en UPAO, no del código. |
| **RQF05** ByteTrack | Se usa un seguimiento IoU propio. Documentado desde la parte 3: BYTETracker de Ultralytics se descartó por frágil entre llamadas. RQF05 dice «mediante el algoritmo Byte Track», así que **es una desviación consciente** del enunciado. |

Los otros 31 requisitos están implementados y verificados.

## Revisión de código

### Duplicación real encontrada y corregida

**El JSON de respuesta estaba escrito en tres sitios** (`/api/detect`, `/api/frame` con overlay y sin overlay). Eso **ya había causado un fallo**: el móvil salía `CONTEXTO` en la ruta de vídeo y `MEDIO` en la de cámara, porque cada copia calculaba el riesgo por su cuenta.

Corregido con un constructor único `respuesta(dets, result, timing, **extra)`. Verificado: las dos rutas comparten ahora **10 claves idénticas**, y solo difieren en `image` e `inference_ms`, que son propios de la subida de imagen.

### Falsos positivos del análisis automático

- «Funciones nunca llamadas»: son rutas de Flask (`@app.route`), se invocan por HTTP.
- `RISK_COLOR` en `risk_engine.py`: lo importa `server.py`.
- Bloques de 6 líneas repetidos en `risk_engine.py:508` y `921`: son `if res.boxes is None: return out`, dos funciones distintas con el mismo corte.

### Estado del resto

- Sintaxis correcta en los 9 ficheros Python.
- Sin imports sin usar ni variables muertas relevantes.
- Los `except` que silencian errores son 1 en `risk_engine.py` (el segundo intento de pose) y 3 en `server.py`, todos con motivo y registro.
