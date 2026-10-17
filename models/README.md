# Modelos

| Archivo | Qué es | Versionado | SHA-256 |
|---|---|---|---|
| `detector/escaleras_s7_v4.pt` | **Modelo S7 v4** — YOLOv8s, 6 clases. Modelo definitivo del proyecto (release `v1.2.0-escaleras-v4`). | sí | `b6a04f2f28d1f7bcc6282053db891c387ac0b121f9d2fcca4534c691f606d7ed` |
| `detector/escaleras_s7_v4.metrics.json` | Métricas y configuración del entrenamiento v4 (del release). | sí | — |
| `detector/escaleras_s7_v4.data.yaml` | Nombres de clase del dataset v4 (del release). | sí | — |
| `coco/yolov8s.pt` | YOLOv8s COCO (Ultralytics): personas y objetos. | sí | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` |
| `base/yolov8s_zoom_v3.pt` | Modelo base v3 (el que estaba en `runs/` de `semana-7`), punto de partida del fine-tuning v4. **No se usa en ejecución**: el sistema carga siempre el Modelo S7 v4. Se conserva solo para reproducir el entrenamiento. | sí | `938fe2b20ee426b22bbd1f7725a6ce9cff8c828fbb9fcd8128adca13d55b8c2f` |
| `sr/realesr-general-x4v3.pth` | Real-ESRGAN compacto (vídeo en vivo). | no (se descarga) | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` |
| `imagen/yolo11n.pt` | YOLO11n COCO (Ultralytics): personas y celulares del **módulo Imagen**. | no (se descarga) | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `imagen/yolo11n-pose.pt` | YOLO11n-Pose (Ultralytics): articulaciones del **módulo Imagen**. | no (se descarga) | `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` |

El clasificador acústico del **módulo Audio** está versionado junto a su código:
`modulos/audio/audio_fall_detector/models/fall_classifier.joblib`
(`68527b88f75142693c56ee798f981611febb5d25fd46316e6c799aca7e2eb9da`).

`RealESRGAN_x4plus.pth` ya no se usa: solo servía a las imágenes y vídeos
subidos, que se retiraron (el sistema es solo de vídeo en vivo). Si quedó en
`models/sr/` de una instalación anterior, se puede borrar.

Descargar o verificar todo:

```bash
python tools/download_models.py           # descarga lo que falte
python tools/download_models.py --check   # solo verifica los hashes
```

Ningún modelo se reentrena ni se modifica en esta rama (tampoco los de los módulos). Para usar otro
checkpoint sin tocar código: `python -m backend --weights ruta/al/modelo.pt`.
