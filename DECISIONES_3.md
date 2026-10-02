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
| RQNF01 ≥30 FPS | no (15) | 42 FPS |
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
`resize` aggressively-downscale → `resize` de vuelta, que destruye los rasgos
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

Con el motor de riesgo completo (YOLO + Pose + obstáculos COCO) son **49 FPS**,
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