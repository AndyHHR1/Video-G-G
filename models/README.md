# Modelos

| Archivo | Qué es | Versionado | SHA-256 |
|---|---|---|---|
| `detector/escaleras_s7_v4.pt` | **Modelo S7 v4** — YOLOv8s, 6 clases. Modelo definitivo del proyecto (release `v1.2.0-escaleras-v4`). | sí | `b6a04f2f28d1f7bcc6282053db891c387ac0b121f9d2fcca4534c691f606d7ed` |
| `detector/escaleras_s7_v4.metrics.json` | Métricas y configuración del entrenamiento v4 (del release). | sí | — |
| `detector/escaleras_s7_v4.data.yaml` | Nombres de clase del dataset v4 (del release). | sí | — |
| `coco/yolov8s.pt` | YOLOv8s COCO (Ultralytics): personas y objetos. | sí | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` |
| `base/yolov8s_zoom_v3.pt` | Modelo base v3 (el que estaba en `runs/` de `semana-7`), punto de partida del fine-tuning v4. **No se usa en ejecución**: el sistema carga siempre el Modelo S7 v4. Se conserva solo para reproducir el entrenamiento. | sí | `938fe2b20ee426b22bbd1f7725a6ce9cff8c828fbb9fcd8128adca13d55b8c2f` |
| `sr/realesr-general-x4v3.pth` | Real-ESRGAN compacto (vídeo en vivo). | no (se descarga) | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` |
| `sr/RealESRGAN_x4plus.pth` | Real-ESRGAN RRDB (imagen fija). | no (se descarga) | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` |

Descargar o verificar todo:

```bash
python tools/download_models.py           # descarga lo que falte
python tools/download_models.py --check   # solo verifica los hashes
```

El Modelo S7 v4 **no se reentrena ni se modifica** en esta rama. Para usar otro
checkpoint sin tocar código: `python -m backend --weights ruta/al/modelo.pt`.
