# Frontend

Panel web servido por Flask (`backend/api/system.py` → `render_template`).

- `templates/index.html` — panel: pestañas Imagen, Vídeo, Cámara en vivo y Clases.
- `templates/login.html` — acceso cuando se define `PANEL_TOKEN`.

Por requisito del cliente **el plano de la interfaz es el mismo** que en
`semana-7`. El único cambio es el texto de la casilla de mejora de calidad
(antes «SR en vivo (≈4 fps)», ahora «Mejora de calidad (Real-ESRGAN x4)»),
porque la mejora ya funciona en tiempo real. El resto de cambios se hizo en el
backend, respetando el mismo contrato de la API (README principal, §12).
