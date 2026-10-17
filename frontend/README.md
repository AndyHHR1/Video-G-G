# Frontend

Panel web servido por Flask (`backend/api/system.py` → `render_template`).

- `templates/index.html` — panel: pestañas **Cámara en vivo**, **🎙 Audio**,
  **🖼 Imagen** y **Clases**, y el botón **«⏻ Detener sistema»**.
- `templates/login.html` — acceso cuando se define `PANEL_TOKEN`.

El vídeo es **solo en vivo**: se quitaron las pestañas de imagen y vídeo subidos.
Las pestañas Audio e Imagen no viven aquí: `index.html` solo incluye
`audio/panel.html` e `imagen/panel.html`, que aportan los módulos de `modulos/`
(cada uno con sus propios JS/CSS en `/modulos/<nombre>/static/`). Si un módulo
no carga, su pestaña no aparece y el resto del panel funciona igual.

El estilo es el mismo que en `semana-7`; la cámara en vivo no cambió salvo que
el monitor del micrófono de la webcam empieza silenciado y plegado (para no
realimentar al micrófono que analiza la pestaña Audio).
