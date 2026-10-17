# Datos (no versionados)

Esta carpeta no se sube al repositorio: contiene vídeos de personas reales
(Ley N.º 29733) y además pesan mucho. Estructura esperada:

```
data/
└── Escaleras/
    ├── Estándar/   vídeos grabados desde la posición REAL de la cámara de seguridad
    │               (se reparten por vídeo en train / val / test)
    └── Extra/      vídeos desde otras posiciones: solo sirven para entrenamiento
```

Solo se necesitan para **reproducir** el entrenamiento del Modelo S7 v4
(`training/extract_escaleras_frames.py`). Para usar el sistema no hacen falta:
los vídeos de `Estándar/` se pueden subir en la pestaña **Vídeo** del panel
para probarlo (RQNF22).
