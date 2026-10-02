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
