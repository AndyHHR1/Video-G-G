# Marco normativo: protección de datos (Perú) y normas ISO/IEC

Este documento explica qué normas aplican al sistema y **cómo** se cumple cada
una en el código. No es asesoría legal: el despliegue real en el campus debe
validarlo el área legal de la UPAO.

---

## 1. Ley N.º 29733 — Ley de Protección de Datos Personales (Perú)

Promulgada en 2011. Regula el tratamiento de datos personales en bancos de datos
públicos y privados. La autoridad es la **Autoridad Nacional de Protección de
Datos Personales (ANPD)**, dentro del Ministerio de Justicia y Derechos Humanos.
Su reglamento vigente fue aprobado por el **D.S. N.º 016-2024-JUS**, que
reemplazó al D.S. N.º 003-2013-JUS.

### 1.1 ¿Por qué aplica a este sistema?

La imagen de una persona que permite identificarla es un **dato personal**. Una
cámara que graba a estudiantes, docentes y personal en una escalera realiza
**tratamiento** de datos personales, y las evidencias guardadas constituyen un
**banco de datos**.

### 1.2 Directiva de videovigilancia

La **Directiva N.º 01-2020-JUS/DGTAIPD** («Tratamiento de datos personales
mediante sistemas de videovigilancia») concreta las obligaciones para cámaras:

- **Informar** con carteles visibles que la zona está videovigilada, quién es el
  responsable y cómo ejercer los derechos.
- Usar las cámaras solo para la **finalidad** declarada (aquí: seguridad y
  prevención de accidentes).
- **Conservar** las imágenes un plazo limitado (la regla general de la Directiva
  es de 30 días) y suprimirlas después, salvo que sean necesarias como prueba de
  un incidente.
- Adoptar **medidas de seguridad** que impidan el acceso no autorizado.

### 1.3 Principios de la ley y cómo se cumplen

| Principio (Ley 29733) | Qué exige | Cómo lo cumple el sistema |
|---|---|---|
| Legalidad (Art. 4) | Tratamiento conforme a la ley | Uso exclusivo para seguridad en un recinto privado, con cartel informativo (despliegue). |
| Consentimiento (Art. 5) | Consentimiento o excepción legal | Videovigilancia por seguridad del recinto informada mediante cartel (Directiva 01-2020). |
| Finalidad (Art. 6) | Finalidad determinada y explícita | Solo detecta conductas de riesgo de caída. **No identifica** personas ni hace reconocimiento facial. |
| Proporcionalidad (Art. 7) | Datos adecuados y no excesivos | No se graba el flujo continuo; solo **un fotograma por alerta confirmada**. |
| Calidad (Art. 8) | Datos exactos y pertinentes | La evidencia lleva fecha, hora, tipo de riesgo e identificador de seguimiento. |
| Seguridad (Art. 9 y 16) | Medidas técnicas y organizativas | Panel con token, comparación en tiempo constante, control de rutas, evidencias fuera del repositorio. |
| Disposición de recurso (Art. 10) | Derechos ARCO | Borrado individual o total de alertas desde el panel. |

### 1.4 Anonimización (disociación)

La ley distingue los datos personales de los datos que **ya no permiten
identificar** a la persona. Anonimizar las evidencias reduce el riesgo y es
coherente con el principio de proporcionalidad. Implementación en
`backend/privacy/anonymizer.py`:

1. Detección de rostros con **MediaPipe Face Detection** en sus dos modelos
   (corto alcance 0–2 m y largo alcance hasta ~5 m, el caso típico de una
   cámara en una escalera).
2. Búsqueda adicional **dentro del recorte de cada persona** detectada: la cara
   ocupa más píxeles y se encuentra aunque esté lejos o de perfil.
3. **Respaldo**: si aun así no aparece la cara de una persona, se pixela la
   zona de su cabeza (parte superior de su caja).
4. Pixelado con bloques de 1/12 de la región (irreversible en la imagen
   guardada: los píxeles originales no se escriben en disco).

Se verificó con imágenes de prueba en las que el modelo de corto alcance no
encontraba ninguna cara: con los pasos 2 y 3 ambas personas quedan
anonimizadas.

### 1.5 Conservación

`configs/app.yaml → alerts.keep_days: 7` y `alerts.max_images: 200`. La
limpieza es automática al guardar cada evidencia. Es un plazo menor que el
general de la Directiva.

### 1.6 Ley N.º 31814 (inteligencia artificial)

La **Ley N.º 31814** (2023) promueve el uso de la inteligencia artificial en el
Perú con enfoque en derechos, y su reglamento (2025) clasifica los usos según su
riesgo. Un sistema de videovigilancia que **no identifica** personas y solo
apoya la decisión humana (sin actuar físicamente, RQNF27) se alinea con un uso
de riesgo controlado, siempre con supervisión humana de las alertas.

---

## 2. Normas ISO/IEC

Las normas ISO son de adopción **voluntaria**; ninguna es exigible por ley a una
universidad privada para este tipo de sistema. En este proyecto son
**obligatorias por el documento de requisitos (ERS)** del curso.

| Norma | Qué es | Evidencia en el proyecto |
|---|---|---|
| **IEEE 830-1998** (sucedida por **ISO/IEC/IEEE 29148**) | Especificación de requisitos de software | `docs/Caso.md` y la ERS `docs/informe/ERS IEEE 830 - Sistema de prevención de caídas en escaleras.docx` (39 requisitos). |
| **ISO/IEC 25010:2023** | Modelo de calidad del producto (9 características) | Cada requisito trazado a una característica (Apéndice C del informe; §10 del README). Modularidad (RQNF20) → capas del backend. |
| **ISO/IEC 25000 (SQuaRE)** | Marco general de calidad | Pruebas automáticas (`tests/`), diagnóstico (`tools/check_env.py`). |
| **ISO/IEC 42001:2023** | Sistema de gestión de IA (AIMS) | Modelo versionado y verificado por SHA-256; registro de cada alerta (`alertas.jsonl`); decisiones de diseño documentadas (`docs/decisiones.md`); supervisión humana. |
| **ISO/IEC 23894:2023** | Gestión del riesgo en IA | Análisis de falsos positivos/negativos (umbrales, confirmación temporal de caídas, estado «REVISAR»). |
| **ISO/IEC 22989:2022** | Conceptos y terminología de IA | Glosario del informe. |
| **ISO/IEC TR 24028:2020** | Confiabilidad de la IA | Aumentos de iluminación en el entrenamiento (RQNF12), estado DEGRADADO explícito ante fallos (RQNF26). |
| **ISO/IEC 27001:2022** | Seguridad de la información | Token de acceso fuera del código, protección contra *path traversal*, evidencias fuera del repositorio. |
| **ISO/IEC 27701** y **ISO/IEC 29100** | Gestión y marco de privacidad | Minimización de datos y anonimización (§1). |
| **ISO/IEC 20889:2018** | Técnicas de desidentificación | Enmascaramiento (pixelado) de rostros. |
| **ISO 8000** y **ISO/IEC 5259-3** | Calidad de datos (para analítica e IA) | Partición por vídeo sin fuga de datos; auditoría de datasets (`training/audit_datasets.py`). |
