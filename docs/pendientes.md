# Pendientes

Estado a la rama `expo` (2026-10-16).

## Resuelto en `expo`
- Falsos «persona caída» con varias personas en escena: la pose ahora se
  empareja con la persona correcta (ver `decisiones.md` §2.1).
- Mejora de vídeo en vivo fluida (~29 fps con Real-ESRGAN `general-x4v3`).
- Evidencias de alertas con rostros: se guardaban mal y sin anonimizar a
  distancia de cámara (ver `decisiones.md` §2.2–2.4).
- Arranque con el Modelo S7 v4 por defecto.

## Pendiente
1. **Validar en la escalera real** con la cámara definitiva: umbrales de
   pasamanos y distracción (reglas sobre keypoints, sin ground truth).
2. **Texto del botón «SR en vivo (≈4 fps)»**: quedó desactualizado; cambiarlo
   cuando se permita tocar el frontend.
3. **Usuarios y roles** (RQNF19 pide usuario y contraseña): hoy es un token
   compartido.
4. **HTTPS** si el panel se usa fuera de `localhost` (necesario además para la
   webcam del navegador desde otro equipo).
5. **Multicámara** (RQNF25): la arquitectura lo permite (un `RiskEngine` por
   cámara), pero la V1 procesa una sola.
6. **Cartel informativo** de videovigilancia en la escalera (Directiva
   01-2020-JUS/DGTAIPD): responsabilidad del despliegue.
