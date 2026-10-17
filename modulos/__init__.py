"""Módulos aparte del sistema, independientes del vídeo en vivo.

Cada subpaquete expone `install(app, settings, runtime, options) -> dict`, que
registra sus rutas Flask y devuelve su estado. `backend/app.py` los carga de
forma aislada: si uno falla, el vídeo en vivo sigue funcionando.

    modulos/audio   percepción acústica (repo brigitteve/percepcion)
    modulos/imagen  análisis de imagen   (repo AndreCastAst, rama s7-imagen)
"""
