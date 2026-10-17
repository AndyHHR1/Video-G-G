# Módulo de imagen

Origen: <https://github.com/AndreCastAst/Repo-video-imagen-audio/tree/s7-imagen>
(grupo de imagen, rama `s7-imagen`). **Reemplaza la antigua pestaña «Imagen»**
del panel, que solo pasaba una foto por el motor de vídeo.

## Qué hace (todo activo)

| Función | Dónde |
|---|---|
| Calidad óptica: luminosidad, contraste, nitidez (Laplaciano), zonas oscuras / sobreexpuestas, sombras, ruido | `modulo_imagen/calidad.py` |
| Preprocesamiento: adaptativo, original, CLAHE, gamma (BGR para YOLO) | `modulo_imagen/preprocesamiento.py` |
| YOLO11n: personas y celulares | `modulo_imagen/detector.py` |
| YOLO11n-Pose: 17 articulaciones e inclinación del torso | `modulo_imagen/pose.py` |
| Análisis por fotograma: asociación pose-persona (IoU), poses sin YOLO, proximidad celular-mano, estados posturales «para revisión», advertencias | `modulo_imagen/analisis_tiempo_real.py` |
| Vector de características `imagen-0.1.0` | `modulo_imagen/features.py` |
| Clips (≤ 10.5 s) y recuperación de personas en ROI | `modulo_imagen/analizador_clip.py`, `recuperacion_roi.py` |
| Rutas HTTP y página de pruebas aislada | `rutas.py`, `templates/imagen/pruebas.html` |
| Pestaña «🖼 Imagen» del panel | `templates/imagen/panel.html`, `static/imagen-panel.js` |

Nunca emite «caída confirmada»: los estados posturales son indicios para
revisión; la interpretación temporal corresponde al vídeo.

## Cambios respecto al repositorio original

1. Los modelos se cargan **en el primer uso** y desde `models/imagen/`
   (`modulo_imagen/modelos.py`), no al importar ni descargándolos al vuelo: así
   no ocupan VRAM ni retrasan el arranque del vídeo (GTX 1650).
2. Su `/api/imagen/analizar` pasaba la foto por el motor de **vídeo** y
   reiniciaba su seguimiento. Se sustituyó por `/api/imagen/caracteristicas`,
   que aplica `build_features` al último fotograma del vídeo en vivo **sin
   alterarlo**.
3. Un cerrojo serializa los análisis (YOLO de Ultralytics no es seguro entre
   hilos).

## Rutas

`POST /api/imagen/pruebas/frame` (multipart `frame`, `modo`),
`GET /api/imagen/caracteristicas`, `GET /api/imagen/estado`,
`GET /imagen/pruebas` (página de pruebas del grupo).

## Pruebas y scripts

```bash
python -m pytest modulos/imagen/tests
python modulos/imagen/scripts/comparar_calidad.py modulos/imagen/ejemplos/prueba_imagen.jpg
python modulos/imagen/scripts/probar_clip_imagen.py clip.mp4      # análisis offline del grupo
```

Los scripts se ejecutan desde la raíz del repositorio. `ejemplos/` tiene las
dos fotos de prueba del grupo y `referencia/` su documentación y su página
original.
