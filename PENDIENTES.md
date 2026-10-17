# Pendientes — Live cámara (post-correcciones A/B/sync)

Estado tras el sprint de 2026-10-16. Se aplicaron mitigaciones de falsos positivos,
instrumentación de profiling y sincronización de cajas por frame-id. Lo que **queda
pendiente** (y por qué), para decidir cómo continuar.

## Cumplido en este sprint
1. **False ALTO en personas de pie (A):** `risk_engine._classify_person` corre el
   modelo 6-clase a `conf=0.20` (débil; test `persona` mAP50=0.49, `caído`≈0) y
   clasificaba de pie como `persona_caido` → ALTO.
   - Piso de confianza `FALL_CROP_CONF=0.50` para caídas via recorte **sin pose**.
   - Confirmación temporal (`_confirm_caida`): necesita `FALL_CONFIRM_N=3` frames
     seguidos con el mismo `track_id` antes de ALTO; hasta allí, **MEDIO "REVISAR"**
     (no NEUTRO, no ALTO directo). La geometría de pose (MediaPipe) sigue directa a
     ALTO (es la señal confiable).
2. **Profilign (B1):** `step()` expone `timing.stages`
   (`people/stairs/pose/classify/obstacles/total`) en `/api/detect` y `/api/frame`.
3. **Sync caja/video por frame-id:** servidor devuelve `frame_id` (del frame que el
   worker analizó realmente) y cliente (`index.html`) mantiene un buffer de 10 blobs
   para dibujar las cajas sobre el frame emparejado → elimina el "desfase" de cajas.
4. Mejora de imagen ya aplicada previamente: filtro low-light **solo en el display**
   (`drawBoxes`), `toBlob` 1 s + `if(!blob) return` (sin lanzar el error de canvas).

## Validado
- `/api/detect` sobre `bus.jpg` (4 personas de pie/camina): `riesgo: BAJO`, sin ALTO falso.
- Profiling caliente (CUDA, fuerza todas las etapas, 4 personas): **~219 ms/frame**
  → people 26, stairs 27, pose 25, classify 114, obstacles 24 ms.
  - **FP16 (`half=True`) RECHAZADO:** mide peor en Turing (219→504 ms); el docstring
    `risk_engine.py` L219-220 ya lo había descartado. No se aplica.
  - **Cadencias:** dejadas como están (`pose_every=8, people_every=6, stairs_every=13,
    obstacle_every=11, classify_every=6`, img 512). Están sintonizadas a GTX1650 (fps);
    la false-ALTO la cubre A sin tocar cadencias. Nota: el docstring (L200-221) documenta
    `pose_every=2` como "configuración" (corrección) vs 26.5 fps; restaurarlo daba más
    fps en RTX pero **baja fps en GTX1650**. Se priorizó GTX1650 + A.
- Server-side `frame_id` validado: `/api/frame?fid=100` responde `frame_id: 100`.

## Pendiente (necesita decisión / datos del usuario)
### C. Reentrenamiento del modelo 6-clase (pospuesto)
El release asset `escaleras_release_v1.1.0.tar.gz` **solo contiene `best.pt` + scripts +
métricas, NO los videos `Escaleras/{Estándar,Extra}`**. El dataset 6-clase real se
construye con `scripts/prepare_dataset.py` sobre `stairs_fallhuman/`, cuyas 3 fuentes
son de Kaggle + Le2i (GitHub):

| Dataset | Clase aporta | URL (README L120-128) |
|---|---|---|
| Fall Detection Dataset | `persona_caido`, `persona_sentado` | https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset |
| Stairs | `escalera` | https://www.kaggle.com/datasets/samuelayman/stairs |
| MOT15 Challenge | `persona` | https://www.kaggle.com/datasets/mdraselsarker/mot15-challenge-dataset |
| Le2i Fall Detection (raw) | `persona_desequilibrio` (+refuerzo `persona_erguida`) | https://github.com/YifeiYang210/Fall_Detection_dataset |

- **No hay credenciales Kaggle en este entorno** (`~/.kaggle/kaggle.json` ausente, CLI
  `kaggle` no instalado) → no se pueden descargar las 3 fuentes Kaggle aquí.
- Le2i (GitHub, público) es clonable, pero solo aporta pre-caida/erguida → insuficiente
  para reentrenar las 6 clases.
- Pipeline disponible: `python3 scripts/prepare_dataset.py` → `datasets/escaleras_dataset/`
  → `python3 scripts/train_escaleras.py --epochs 100` (≈3-4 h en GTX1650; `configs/train_escaleras.yaml`).
- Si se reentrena, enfocarse en **más datos de personas de pie en escaleras/poca luz**
  (no solo repetir el entrenamiento), ya que la confusión caído↔erguida es el origen del
  false-ALTO.

### Validación en webcam real
El sandbox NO tiene cámara (`/dev/video*` inexistente), por lo que no se pudo validar
el live directo. Se validó el pipeline server-side sobre `bus.jpg`. Pendiente probar A y
el sync-de-frame-id con la cámara real del usuario.

### Cold-start / first-frame stall
La primera inferencia paga ~3 s de JIT CUDA (el warmup de `Inferencer._run` usa un
frame gris `_wu` sin personas → no calienta el path de recorte por persona). Opcional:
ampliar el warmup con un frame real con personas para eliminar el stall inicial.

## Recomendación (Claude)
A: aplicada. B: profiling primero (hecho) → FP16 descartado, cadencias conservadas.
Sync frame-id: aplicada. C: no ahora (necesita datos/credenciales). Reentrenar sólo si
después de A el modelo sigue mal en el escenario real, y priorizar datos de pie en
escaleras/poca luz.
