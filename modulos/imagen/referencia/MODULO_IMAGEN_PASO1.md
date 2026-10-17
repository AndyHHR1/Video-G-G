# Módulo Imagen — paso 1 (escaleras)

Este parche agrega extracción de características ópticas + adaptación de detecciones, posturas y tracks **ya producidos** por `RiskEngine`. No modifica `/api/frame`, `/api/detect`, historial ni alertas. El tracking existente es IoU, no ByteTrack. Las señales pose pueden describir una sola persona por fotograma. El módulo no detecta por sí mismo agua ni piso mojado; eso necesita datos, etiquetas y validación.

## Instalar y ejecutar (Windows, PowerShell)

1. Abrir carpeta del proyecto en VS Code.
2. `py -3.11 -m venv .venv` (si tienes Python 3.11).
3. ` .\.venv\Scripts\python.exe -m pip install numpy==1.26.4 opencv-python==4.10.0.84 pytest`
4. ` .\.venv\Scripts\python.exe -m pytest tests/test_modulo_imagen.py -q`
5. Para el servidor también se necesitan las dependencias compatibles con tu CPU/GPU y los modelos `.pt` definidos por el proyecto general. **No instales los paquetes CUDA del requirements general a ciegas en Windows**.
6. Cuando el servidor esté activo, prueba una imagen en PowerShell:

```powershell
curl.exe -X POST -F "file=@C:\ruta\foto_escalera.jpg" http://127.0.0.1:5000/api/imagen/analizar
```

La ruta devuelve el JSON `imagen-0.1.0` con `metadata_captura`, `vector_caracteristicas` y `diagnostico_descriptivo`. Esta primera ruta es para **imágenes sueltas** y reinicia tracks entre llamadas. El siguiente paso será adaptar la salida de `/api/frame` para reutilizar el tracking persistente del video en directo, después de comprobar el rendimiento y la compatibilidad con la interfaz. Evita alternar llamadas de imagen suelta y video en la misma sesión de prueba porque comparten el motor.

## Validación

Comparar manualmente coordenadas de detección, `track_id`, señales pose y calidad óptica con las imágenes reales. No se ha demostrado todavía precisión de riesgos ni reconocimiento de agua. Para una grabación de ensayo, usar situaciones simuladas seguras sin caídas reales.
