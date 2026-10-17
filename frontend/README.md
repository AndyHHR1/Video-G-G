# Frontend

Panel web servido por Flask (`backend/api/system.py` → `render_template`).

- `templates/index.html` — panel: pestañas Imagen, Vídeo, Cámara en vivo y Clases.
- `templates/login.html` — acceso cuando se define `PANEL_TOKEN`.

Por requisito del cliente **la interfaz no se modificó** en la rama `expo`: el
contenido es idéntico al de `semana-7`. Todo cambio de comportamiento (por
ejemplo, la mejora de vídeo en vivo fluida) se hizo en el backend, respetando
el mismo contrato de la API (ver README principal, §12).
