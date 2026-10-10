UNIVERSIDAD PRIVADA ANTENOR ORREGO
FACULTAD DE INGENIERÍA
ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS E INTELIGENCIA ARTIFICIAL












Sistema Inteligente para Prevención de Caídas y Riesgos en Escaleras Basado en Video con YOLOv8 y MediaPipe (V4)


Estudiantes:
Hermenegildo Rumiche, Andy Humberto
Mendoza Avila, Jorge Luis
Lopez Gonzalez, Jorge
Coronel Ruiz, Brayan
Cortez Acon Jonaiker Mirosevic
Sifuentes Berrocal, Edy Anthony
Pereda Obando, Luis Go


Docente: 
Mendoza Corpus, Carlos Alfredo
	Curso
Percepción computacional
 


Trujillo - 2026


Índice
1. Introducción	4
1.1. Propósito	4
1.2. Ámbito del Sistema	4
1.2.1. Identificación del Sistema	4
1.2.2. Alcance Operativo (Lo que el sistema HARÁ)	5
1.2.3. Límites y Exclusiones del Sistema (Lo que el sistema NO HARÁ)	5
1.2.4. Beneficios, Objetivos y Metas	6
1.3. Definiciones, Acrónimos y Abreviaturas	6
1.4. Referencias	8
2. Descripción General	8
2.1. Perspectiva del Producto	8
2.2. Funciones del Producto	9
2.3. Características de los Usuarios	9
2.4. Restricciones	10
2.5. Suposiciones y Dependencias	11
2.6. Requisitos Futuros	12
3. Requisitos Específicos	13
3.1. Interfaces Externas	13
3.1.1. Interfaces de Usuario	13
3.1.2. Interfaces de Hardware y Software	15
3.2. Funciones	16
Justificación del Criterio de Organización:	16
3.2.1. Ingesta y Captura de Video	16
3.2.2. Análisis de Visión por Computadora e Inferencia	16
3.2.3. Seguimiento Espacio-Temporal y Evaluación de Riesgo	17
3.2.4. Gestión y Notificación de Alertas	18
3.3. Requisitos de Rendimiento	19
3.4. Restricciones de Diseño	20
3.5. Atributos del Sistema	22
3.5.1. Capacidad de Interacción / Usabilidad	22
3.5.2. Fiabilidad	22
3.5.3. Seguridad (Security)	24
3.5.4. Mantenibilidad	27
3.5.5. Flexibilidad / Portabilidad	29
3.5.6. Protección (Safety)	30
4. APÉNDICES	30






1. Introducción
La presente Especificación de Requisitos de Software (ERS) establece las bases técnicas, funcionales y de calidad para el diseño, desarrollo, evaluación y despliegue del software de percepción computacional orientado a la seguridad de tránsito peatonal y prevención de caídas en escaleras dentro del campus universitario.
1.1. Propósito
El propósito de este documento es definir de manera exhaustiva, formal y verificable los requisitos funcionales y no funcionales que rigen la construcción del "Sistema Inteligente para Prevención de Caídas y Riesgos en Escaleras Basado en Video con YOLOv8 y MediaPipe (V4)" en los accesos y tramos de escaleras de la Universidad Privada Antenor Orrego (UPAO).
El documento está estructurado para servir como guía oficial a las siguientes audiencias:
Equipo de Desarrollo e Ingeniería de Software: Constituye la especificación técnica de referencia para el diseño arquitectónico, estimación postural, entrenamiento de modelos visuales, desarrollo de componentes, pruebas de verificación y despliegue del sistema.
Docente Evaluador del curso de Percepción Computacional (UPAO): Funciona como el instrumento formal de auditoría y evaluación técnica sobre la aplicación de modelos de calidad (ISO/IEC 25010), gestión de IA (ISO/IEC 42001) y estándares de ingeniería de requerimientos (IEEE 830).
Personal de Seguridad y Supervisores de Campus: Representa la referencia funcional para los interesados (stakeholders) y usuarios finales respecto a los alcances de la herramienta de monitoreo de escaleras, despliegue de alertas tempranas y soporte a la prevención de caídas y tropiezos.
1.2. Ámbito del Sistema
1.2.1. Identificación del Sistema
Nombre del Software: Sistema Inteligente para Prevención de Caídas y Riesgos en Escaleras Basado en Video con YOLOv8 y MediaPipe (V4)
Tipo de Producto: Aplicación independiente (standalone) para procesamiento e inspección visual en tiempo real con aceleración por hardware (GPU).
1.2.2. Alcance Operativo (Lo que el sistema HARÁ)
Basado en las capacidades aprobadas en los Requisitos Funcionales (RQF01 al RQF08), el sistema realizará de manera automatizada las siguientes funciones:
Ingesta continua: Captura de la señal de video en vivo proveniente de una cámara fija instalada en un tramo de escalera de la UPAO (RQF01).
Análisis por Visión Computacional: Inferencia automatizada mediante YOLOv8 para detección de objetos/obstáculos y MediaPipe Pose para estimación esquelética, identificando cuatro (4) categorías de riesgo en tiempo real: objetos u obstáculos abandonados en escalones, no uso del pasamanos durante el tránsito, distracciones peligrosas (uso de celular o lectura mientras se camina) y eventos de caída activa o pérdida inminente de equilibrio (RQF02).
Discriminación de contexto: Clasificación activa para diferenciar entre tránsito seguro sostenido (sujeción del pasamanos, postura erguida) y conductas o estados de riesgo real en escaleras (RQF03).
Filtrado de certeza: Aplicación de umbrales de confianza configurables (75% pre-filtrado / 85% confirmación) para eliminar falsos positivos (RQF04).
Seguimiento espacio-temporal: Rastreo de identidades únicas (ID de seguimiento) mediante un algoritmo de seguimiento IoU propio (~30 líneas), tolerando oclusiones de hasta 1 segundo (RQF05).
Evaluación de persistencia: Validación de que una condición de peligro permanezca un tiempo mínimo según su nivel de riesgo (ALTO: 1.2 s, MEDIO: 0.6 s, con histéresis de 1.0 s) antes de calificarla como alerta confirmada (RQF06).
Gestión de alertas: Emisión y transmisión de notificaciones estructuradas con metadatos y evidencia fotográfica del incidente (RQF07).
Monitoreo visual: Despliegue de datos del evento e indicador permanente del estado operativo del sistema en el panel de supervisión (RQF08).
1.2.3. Límites y Exclusiones del Sistema (Lo que el sistema NO HARÁ)
En estricta concordancia con las restricciones y acuerdos del proyecto, el software queda acotado por los siguientes límites operacionales:
Sin actuación física autónoma: El software opera exclusivamente como un sistema pasivo de supervisión y apoyo a la decisión; no ejecutará ni controlará directamente actuadores físicos, barreras mecánicas de escaleras ni sirenas sonoras de alta potencia que puedan sobresaltar a los peatones o causar accidentes secundarios (RQNF27).
Sin operación 24/7: El sistema no operará de forma ininterrumpida las 24 horas del día, restringiendo su funcionamiento únicamente al horario operativo de tránsito en el campus (RQNF11).
Sin interoperabilidad externa: En esta versión (V1), el software es una solución aislada que no intercambiará datos ni se integrará con sistemas de videovigilancia globales de terceros (VMS/NVR), plataformas de control de accesos ni bases de datos institucionales de la UPAO.
Procesamiento de cámara única en V1: El sistema procesará el flujo continuo de una (1) sola cámara fija (RQF01, RQNF04); la capacidad de analizar múltiples flujos paralelos queda explícitamente diferida a la sección 2.6 (Requisitos Futuros).
1.2.4. Beneficios, Objetivos y Metas
Automatización del Monitoreo: Sustitución de la vigilancia humana continua en escaleras por un análisis automatizado y constante en GPU mediante YOLOv8 y MediaPipe.
Oportunidad de Respuesta: Reducción de la latencia en la notificación de alertas a menos de 500 ms tras la confirmación de la condición de riesgo.
Prevención Activa de Caídas: Incremento de la seguridad peatonal en escaleras mediante la detección oportuna de comportamientos e imprudencias de riesgo antes y durante la ocurrencia de una caída.
1.3. Definiciones, Acrónimos y Abreviaturas


Término / Acrónimo
Definición Resumida
Algoritmo de seguimiento IoU (V1)
Algoritmo de seguimiento multiobjeto en tiempo real (~30 líneas) que preserva la identidad (ID) de transeúntes u objetos en escaleras ante oclusiones mediante IoU > 0.25 y OCCLUSION_SECONDS = 1.0. En la versión V1 del sistema se implementa de manera propia en lugar del ByteTrack de Ultralytics, para evitar el arrastre de estado interno entre llamadas.
MediaPipe Pose
Framework de estimación de pose esquelética en tiempo real para detección de keypoints corporales y postura en el tránsito por escaleras.
ERS
Especificación de Requisitos de Software (documento normado por el estándar IEEE 830).
FPS
Frames Per Second / Fotogramas por Segundo (tasa de rendimiento de video. Requisito formal RQNF01: mínima 30 FPS; en V1 el motor mide 26.5 fps con personas).
ISO/IEC 25010
Estándar internacional que especifica el modelo de calidad del producto software en 9 características.
Ley N.º 29733
Ley de Protección de Datos Personales del Perú, aplicable al tratamiento de imágenes captadas en video.
mAP@0.5
Métrica de precisión promedio media en visión por computadora (mínima exigida > 75%).
VRAM
Memoria de acceso aleatorio de video de la GPU (requerimiento mínimo de 6 GB).
YOLOv8
Modelo de red neuronal convolucional estado del arte para detección y segmentación de objetos en tiempo real.

1.4. Referencias
El presente documento de especificación de requisitos se fundamenta y mantiene trazabilidad con las siguientes fuentes normativas, legales y datasets de referencia:
GitHub - Fall Detection Datasets (Le2i / SisFall / FDD / URFD): Repositorio público especializado (YifeiYang210/Fall_Detection_dataset) de datasets procesados para detección de caídas en video. Enlace de referencia: https://github.com/YifeiYang210/Fall_Detection_dataset
Congreso de la República del Perú: Ley N.º 29733 - Ley de Protección de Datos Personales del Perú.
IEEE Computer Society: IEEE Std 830-1998 - IEEE Recommended Practice for Software Requirements Specifications.
ISO/IEC 25010: Systems and software engineering — Systems and software quality requirements and evaluation (SQuaRE) — Product quality model.
ISO/IEC 27001: Information technology — Security techniques — Information security management systems — Requirements.
ISO/IEC 42001: Information technology — Artificial intelligence — Management system.
ISO/IEC 25000: Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Guide to SQuaRE.
ISO 8000: Data quality — International standard for industrial data quality.
PPGIA-UNIFOR TsetFall Dataset: Dataset público especializado para la detección de caídas y análisis de actividades diarias.
1.5. Visión General del Documento
El presente informe de ERS está organizado formalmente en cuatro (4) secciones estructuradas según el estándar IEEE 830:
Sección 1 (Introducción): Especifica el propósito del documento, ámbito del sistema (funciones, límites y beneficios), términos/acrónimos clave, referencias normativas y la visión general del informe.
Sección 2 (Descripción General): Presenta la perspectiva del producto en los tramos de escaleras, funciones principales, características del perfil de usuario, restricciones operativas y tecnológicas, suposiciones, dependencias y requisitos futuros (sección 2.6).
Sección 3 (Requisitos Específicos): Constituye el núcleo del informe y consolida los 35 requisitos aprobados (8 Funcionales: RQF01 a RQF08; y 27 No Funcionales: RQNF01 a RQNF27) organizados en las subsecciones del IEEE 830 (Interfaces Externas, Funciones por Jerarquía Funcional/Pipeline, Requisitos de Rendimiento, Restricciones de Diseño, Atributos del Sistema y Otros Requisitos), todos ellos trazados a las 9 características del modelo de calidad ISO/IEC 25010.
Sección 4 (Apéndices): Incorpora el material complementario oficial: el Glosario Extendido (Apéndice A), la Descripción del Pipeline de Datos y Arquitectura en texto estructurado (Apéndice B), y la Matriz de Trazabilidad Completa (Apéndice C).
2. Descripción General
El Sistema Inteligente para Prevención de Caídas y Riesgos en Escaleras es un programa desarrollado para fortalecer la seguridad peatonal dentro del campus de la Universidad Privada Antenor Orrego (UPAO). Su propósito es supervisar continuamente las zonas de escaleras a través del análisis del video en tiempo real capturado por una cámara de seguridad, identificando de manera oportuna situaciones de riesgo como caídas, tropiezos, imprudencias o presencia de obstáculos, y notificando estos eventos al personal encargado de la seguridad para una pronta intervención.
2.1. Perspectiva del Producto
El sistema es un programa software completamente independiente que opera de manera autónoma en las zonas de escaleras de la UPAO. En esta primera versión, no se conecta ni depende de ningún otro software, plataforma o sistema externo de la universidad; funciona utilizando únicamente la información que recibe directamente de la cámara fija instalada en el tramo de escaleras.
El entorno operativo del producto se define a través de sus relaciones de entrada y salida:
Entrada del sistema (Captura de video): El programa recibe como única fuente de entrada la señal de video que graba de forma continua una cámara de seguridad ubicada sobre los tramos de escaleras. La cámara registra el tránsito peatonal y transmite ese video al sistema para su revisión constante. El sistema está configurado inicialmente para trabajar con una cámara fija, con la posibilidad de sumar más escaleras en fases futuras.
Procesamiento de la información: El sistema examina permanentemente las imágenes mediante YOLOv8 y MediaPipe para observar la postura de los peatones, detectar obstáculos en escalones y evaluar comportamientos de riesgo o pérdida de equilibrio.
Salida del sistema (Panel de supervisión): Cuando el sistema confirma que existe un riesgo o una caída en la escalera, su salida es una notificación de alerta que se envía inmediatamente a un panel de supervisión. Este panel es una pantalla o interfaz destinada al personal encargado de la seguridad y a los supervisores del campus, donde se muestra la información detallada del evento para permitir una rápida auxilio o despeje de la vía.
2.2. Funciones del Producto
A grandes rasgos, el sistema realiza la supervisión automática de las escaleras mediante la revisión constante del video, identificando situaciones peligrosas y notificándolas a tiempo. Las funciones principales del producto se resumen a continuación:
Observación continua del video: El sistema recibe y revisa de forma ininterrumpida el flujo de video de las escaleras, monitoreando el flujo de personas y el estado de los escalones.
Identificación automática de situaciones de riesgo: El programa analiza las imágenes para reconocer automáticamente cuatro tipos específicos de peligros en escaleras:
Objetos u obstáculos en escalones: Mochilas, cajas, líquidos o elementos tirados en los peldaños que puedan provocar tropezones o resbalones.
No uso del pasamanos: Tránsito de personas descendiendo o ascendiendo sin sujetarse de la barandilla de protección.
Distracciones peligrosas: Peatones caminando por la escalera mientras observan el teléfono móvil o leen documentos.
Caída activa o pérdida de equilibrio: Detección de pérdida repentina de postura vertical, resbalón o cuerpo tendido sobre los escalones.
Verificación de la permanencia del riesgo: El sistema comprueba si la situación de peligro o el objeto detectado permanece en el video de forma continua un tiempo mínimo según su nivel (ALTO: 1.2 s, MEDIO: 0.6 s, con histéresis de 1.0 s), evitando falsas alarmas o avisos repetidos por movimientos pasajeros.
Generación automática de alertas: Una vez confirmado que la situación de riesgo ha persistido el tiempo suficiente, el programa genera de forma inmediata un aviso de alerta.
Visualización en el panel de supervisión: El sistema presenta la alerta en la pantalla del personal de seguridad de forma clara. Cada notificación muestra la cámara que captó el hecho, la fecha y hora exactas, el tipo de riesgo identificado y una foto del momento preciso en que se detectó el problema.
2.3. Características de los Usuarios
Identificación de los usuarios: El sistema está dirigido al personal de vigilancia, supervisores de infraestructura y encargados de gestión de riesgos de campus en la Universidad Privada Antenor Orrego (UPAO).
Nivel educativo y formación: Son profesionales o técnicos capacitados en normas de seguridad laboral, gestión de riesgos e inspección de ambientes de trabajo. No poseen formación en ingeniería de software, informática ni desarrollo de sistemas.
Experiencia técnica: Tienen un nivel de experiencia técnica básico a intermedio en el uso de herramientas informáticas cotidianas (manejo de computadoras de escritorio, navegación en pantallas y visualización de monitores de control). No cuentan con conocimientos en programación, administración de servidores ni configuración de redes.
Experiencia en la tarea: Poseen un alto conocimiento de los protocolos de seguridad en instalaciones universitarias, reconociendo fácilmente qué situaciones constituyen un riesgo en escaleras (como escalones obstaculizados, distracciones con celular o caídas).
Implicancia para el diseño del sistema: Debido a que los usuarios no son especialistas en tecnología, el panel de supervisión debe ser sumamente claro, visual e intuitivo. La información de las alertas se debe presentar de forma directa y fácil de interpretar (fotografía del incidente, fecha, hora y tipo de riesgo), de modo que el usuario pueda comprender la situación de inmediato sin requerir capacitaciones complejas ni configuraciones técnicas.
2.4. Restricciones
Políticas Institucionales y Marco Normativo de Gobernanza de IA:
Sistema de Gestión de IA (AIMS): El desarrollo y despliegue del sistema debe alinearse con la norma ISO/IEC 42001 (Artificial Intelligence Management System), estableciendo políticas de transparencia, trazabilidad de las inferencias visuales, rendición de cuentas y auditoría poscomercialización en el ciclo de vida del modelo de aprendizaje automático.
Gestión de Riesgos de IA: La identificación y mitigación de amenazas algorítmicas (tales como falsos positivos o falsos negativos en la detección de riesgos) se rigen bajo las directrices de la norma ISO/IEC 23894, manteniendo consistencia conceptual con la terminología estandarizada de ISO/IEC 22989.
Consideraciones acerca de la Seguridad y Protección de Datos:
Marco Legal Personal: Cumplimiento obligatorio de la Ley N.º 29733 (Ley de Protección de Datos Personales del Perú) en la captura, procesamiento y almacenamiento de los flujos de video y fotogramas donde aparezcan estudiantes, docentes y personal transeúnte en las escaleras.
Seguridad de la Información (SGSI): Aplicación de los controles de la norma ISO/IEC 27001 para salvaguardar la confidencialidad, integridad y disponibilidad de las grabaciones de video, evitando accesos no autorizados y protegiendo los canales de transmisión entre la cámara, el servidor de cómputo y el panel de supervisión.
Limitaciones del Hardware y Procesamiento:
El procesamiento de los modelos de visión por computadora (YOLOv8 para objetos y MediaPipe Pose para estimación esquelética corporal) y el seguimiento espacial (algoritmo de seguimiento IoU propio) requiere obligatoriamente un entorno de ejecución con aceleración por hardware basado en GPU NVIDIA con capacidad mínima de 6 GB de VRAM.
En la primera versión, el sistema está restringido al procesamiento de un único flujo continuo de video proveniente de una cámara fija instalada en un tramo de escalera.
Operaciones Paralelas y Rendimiento en Tiempo Real:
El sistema debe garantizar un procesamiento continuo en GPU a una tasa sostenida de al menos 30 FPS (requisito formal RQNF01). En la versión V1, el rendimiento real mide 26.5 fps con personas presentes (MediaPipe Pose en CPU es el cuello de botella, ~18 ms por frame); sin modo asíncrono se alcanzan ~20 fps. El requisito de 30 FPS no se cumple de forma sostenida en escenas con peatones.
La latencia máxima permitida para la emisión de la alerta hacia el panel de supervisión no superará los 500 ms tras confirmarse la persistencia temporal del riesgo (tiempo mínimo según nivel: ALTO 1.2 s, MEDIO 0.6 s, con histéresis 1.0 s).
Funciones de Auditoría y Control de Calidad del Software:
Modelo de Calidad de Software: Evaluación de las características del producto según ISO/IEC 25010:2023 (marco general de la familia ISO/IEC 25000 SQuaRE), asegurando altos estándares de idoneidad funcional, fiabilidad, eficiencia en el rendimiento y capacidad de interacción.
Trazabilidad de Eventos: El sistema debe registrar logs inalterables que contengan la cámara de origen, fecha, hora, tipo de riesgo, identificador de seguimiento y la captura del fotograma como evidencia auditable de la detección, cumpliendo con los requerimientos de trazabilidad de ISO/IEC 42001.
Resiliencia y Confiabilidad Algorítmica: Resistencia del modelo ante perturbaciones visuales (cambios repentinos de iluminación, oclusiones o ruido en la imagen) evaluada según el informe técnico ISO/IEC TR 24028.
Criticalidad de la Aplicación y Calidad de los Datos de Entrenamiento:
Dado que el software está orientado a la prevención de caídas y atención de emergencias en escaleras, la fiabilidad de las alertas depende estrictamente de la calidad de los datos utilizados en la fase de entrenamiento y ajuste fino (fine-tuning).
Se imponen como restricción los lineamientos de la norma ISO 8000 y la serie ISO/IEC 5259-3, utilizando datasets públicos especializados de detección de caídas (Le2i, SisFall, FDD, URFD, PPGIA-UNIFOR TsetFall Dataset) e imágenes de escaleras universitarias, exigiendo auditorías sobre la completitud del etiquetado y la precisión postural.
Interfaces con Otras Aplicaciones:
El programa opera de forma independiente y desacoplada de la red central de videovigilancia o de otros sistemas informáticos de la universidad en su versión inicial, limitando sus interfaces a la ingesta de video desde la cámara fija y al envío de eventos hacia el panel de supervisión.
Requisitos de Habilidad del Usuario:
El panel de supervisión debe diseñarse bajo principios de operabilidad e interactividad (ISO/IEC 25010), permitiendo su uso por parte del personal de seguridad de la UPAO sin requerir competencias en configuración de modelos de IA ni conocimientos de programación.
2.5. Suposiciones y Dependencias
Dependencias de la Infraestructura de Hardware y Red:
Se asume la disponibilidad continua y la estabilidad del flujo de video transmitido por la cámara IP enfocada en la escalera.
Se presupone el funcionamiento ininterrumpido de la red local del sector y del suministro eléctrico para la cámara, el equipo con GPU y la pantalla del panel de supervisión.
En caso de interrupción temporal de la señal de video, se asume que el mecanismo de reconexión automática del software restablecerá la ingesta sin requerir el reinicio manual del sistema.
Suposiciones sobre el Entorno Físico y la Escena Visual:
Se asume que el área de la escalera mantiene iluminación ambiental adecuada que permite visibilizar escalones, barandillas y posturas de los peatones.
Se presupone la conservación de los elementos estructurales de la escalera (como el pasamanos y bordes de escalón). Si la barandilla es modificada o reubicada, los criterios de detección postural deberán reconfigurarse.
Suposiciones sobre el Dataset y el Rendimiento del Modelo:
Se asume que los datasets de entrenamiento (secuencias de caídas Le2i, SisFall, FDD, URFD, PPGIA-UNIFOR TsetFall Dataset complementados con tomas de escaleras UPAO) son representativos para alcanzar una precisión promedio media (mAP@0.5 > 75%), cumpliendo con la norma ISO/IEC 5259-3.
Dependencias del Entorno de Software:
El correcto desempeño computacional (≥ 30 FPS) depende de la compatibilidad y correcta instalación de las librerías de aceleración por hardware (controladores NVIDIA CUDA / TensorRT) sobre el sistema operativo en el servidor de ejecución.
Disponibilidad del Personal de Operación:
Se presupone que el personal de seguridad o supervisores del campus mantendrán operativo el panel de control durante las horas de trabajo para recibir las notificaciones y ejecutar las acciones preventivas ante la generación de una alerta.
2.6. Requisitos Futuros
En esta subsección se esbozan las expansiones y mejoras funcionales que podrán ser analizadas, diseñadas e implementadas en versiones posteriores del software, sobre la base de la arquitectura inicial:
Escalabilidad Multicámara y Cobertura Distribuida:
Ampliación del motor de ingesta y procesamiento de video para escalar de la cámara fija inicial a un entorno multicámara continuo en todos los pabellones y edificios del campus UPAO.
Integración con Plataformas de Videovigilancia Institucionales (VMS/NVR):
Evaluación de interfaces de conexión con el sistema central de seguridad y videovigilancia de la universidad, permitiendo la sincronización de las alertas detectadas por el sistema con la infraestructura global de monitoreo de la UPAO.
Amplificación del Catálogo de Riesgos y Tipos de EPP:
Reentrenamiento del modelo (fine-tuning) para incorporar nuevos patrones de comportamiento de riesgo (como velocidad excesiva al bajar, saltos de peldaños o aglomeraciones sofocantes en escaleras de evacuación).
Notificaciones Móviles y Remotas de Alta Prioridad:
Implementación de un canal secundario de transmisión de alertas automáticas (mensajería instantánea o app móvil) hacia los brigadistas de primeros auxilios ante eventos de caída severa.
3. Requisitos Específicos
3.1. Interfaces Externas
3.1.1. Interfaces de Usuario
Especifica los requerimientos relacionados con las características visuales, de presentación de datos, de interacción y de seguridad en el acceso al panel de supervisión.

Especificación de Requerimiento Funcional
Detalle
Requerimiento funcional N°:
RQF08
Nombre:
Despliegue de datos en el panel de supervisión
Tipo:
Funcional
Prioridad:
Alta
Característica ISO 25010:
Adecuación Funcional > Pertinencia Funcional
Subsección IEEE 830:
3.1.1 Interfaces de Usuario
Descripción:
El sistema debe presentar visualmente en el panel de supervisión la información detallada de la alerta emitida, mostrando la captura del fotograma del incidente en la escalera junto con los datos asociados (cámara/ubicación, fecha, hora, categoría de riesgo e ID de seguimiento) para la revisión del personal de seguridad.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF06
Nombre:
Reconocibilidad visual de la información de alerta
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF08
Característica ISO 25010:
Capacidad de Interacción > Reconocibilidad de Adecuación
Subsección IEEE 830:
3.1.1 Interfaces de Usuario
Descripción:
El panel de supervisión debe presentar la información de cada alerta en un diseño visual estructurado que consolide en un solo bloque la cámara/escalera de origen, fecha, hora exacta, categoría de riesgo detectada, ID de seguimiento y la captura del fotograma de la imprudencia o caída, permitiendo reconocer de inmediato la situación.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF08
Nombre:
Protección visual frente a errores de interpretación
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF08
Característica ISO 25010:
Capacidad de Interacción > Protección frente a Errores de Usuario
Subsección IEEE 830:
3.1.1 Interfaces de Usuario
Descripción:
La interfaz del panel de supervisión debe emplear esquemas de diferenciación cromática de alto contraste entre el estado de monitoreo normal y el estado de emisión de alerta, evitando que el personal de seguridad confunda o pase por alto una notificación de riesgo activo.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF09
Nombre:
Auto-descriptividad del estado operativo del sistema
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQF08
Característica ISO 25010:
Capacidad de Interacción > Auto-descriptividad
Subsección IEEE 830:
3.1.1 Interfaces de Usuario
Descripción:
El panel de supervisión debe mantener visible un indicador de estado en pantalla que informe de forma permanente al usuario si la transmisión del video de la cámara y el motor de análisis en tiempo real se encuentran activos y funcionando correctamente.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF19
Nombre:
Autenticación de usuario para acceso al panel
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF08
Característica ISO 25010:
Seguridad > Autenticidad
Subsección IEEE 830:
3.1.1 Interfaces de Usuario
Descripción:
El sistema debe requerir la autenticación obligatoria del usuario mediante credenciales (nombre de usuario y contraseña) antes de permitir el acceso y la visualización de la pantalla del panel de supervisión y de las alertas emitidas, restringiendo la interfaz exclusivamente al personal de seguridad y supervisores autorizados.

3.1.2. Interfaces de Hardware y Software
Define las especificaciones de conexión e ingesta física o lógica de datos entre el software y los dispositivos de captura del entorno.

Especificación de Requerimiento Funcional
Detalle
Requerimiento funcional N°:
RQF01
Nombre:
Captura continua de flujo de video
Tipo:
Funcional
Prioridad:
Alta
Característica ISO 25010:
Adecuación Funcional > Completitud Funcional
Subsección IEEE 830:
3.1.2 Interfaces de Hardware/Software
Descripción:
El sistema debe capturar de manera continua la señal de video transmitida por la cámara fija instalada en la escalera del campus UPAO, asegurando la ingesta ininterrumpida de fotogramas para su análisis en tiempo real.


3.2. Funciones
Justificación del Criterio de Organización:
Para organizar esta subsección se adoptó formalmente el criterio de Jerarquía Funcional / Pipeline de Procesamiento de Datos. Esta decisión se fundamenta en que:
El proyecto cuenta con un perfil único de usuario (personal de seguridad/supervisores), lo que vuelve redundante la organización por roles.
El procesamiento visual posee un flujo lineal e interdependiente muy claro: Ingesta de video → Análisis e inferencia → Seguimiento y evaluación de riesgo → Gestión de alertas.
3.2.1. Ingesta y Captura de Video
(Etapa inicial del pipeline orientada a la recepción ininterrumpida de fotogramas de la escalera; la especificación de entrada de la señal física corresponde a RQF01 en la subsección 3.1.2).
3.2.2. Análisis de Visión por Computadora e Inferencia

Especificación de Requerimiento Funcional
Detalle
Requerimiento funcional N°:
RQF02
Nombre:
Detección automatizada de categorías de riesgo
Tipo:
Funcional
Prioridad:
Alta
Característica ISO 25010:
Adecuación Funcional > Completitud Funcional
Subsección IEEE 830:
3.2.2 Análisis de Visión por Computadora e Inferencia
Descripción:
El sistema debe detectar e identificar en tiempo real objetos y conductas pertenecientes a cuatro (4) categorías de riesgo definidas en escaleras: (1) objetos u obstáculos abandonados en escalones, (2) no uso del pasamanos durante el tránsito, (3) distracciones peligrosas (uso de celular/lectura), y (4) caídas activas o pérdida inminente de equilibrio.


Especificación de Requerimiento Funcional
Detalle
Requerimiento funcional N°:
RQF03
Nombre:
Discriminación entre uso seguro y condición de riesgo
Tipo:
Funcional
Prioridad:
Alta
Característica ISO 25010:
Adecuación Funcional > Corrección Funcional
Subsección IEEE 830:
3.2.2 Análisis de Visión por Computadora e Inferencia
Descripción:
El sistema debe diferenciar entre la presencia de peatones en tránsito seguro (postura erguida, sujeción de barandilla) y conductas de riesgo (descenso distraído mirando el teléfono o tránsito sin sujetarse del pasamanos).


Especificación de Requerimiento Funcional
Detalle
Requerimiento funcional N°:
RQF04
Nombre:
Filtrado por umbrales de confianza configurables
Tipo:
Funcional
Prioridad:
Alta
Característica ISO 25010:
Adecuación Funcional > Corrección Funcional
Subsección IEEE 830:
3.2.2 Análisis de Visión por Computadora e Inferencia
Descripción:
El sistema debe aplicar un filtro de decisión basado en umbrales de confianza (confidence threshold) para descartar detecciones con baja certeza matemática, estableciendo un umbral mínimo configurable (por defecto 75% para pre-filtrado de detección y 85% para confirmación de alerta).

3.2.3. Seguimiento Espacio-Temporal y Evaluación de Riesgo
Especificación de Requerimiento Funcional
Detalle
Requerimiento funcional N°:
RQF05
Nombre:
Seguimiento espacial de objetos con tolerancia a oclusión
Tipo:
Funcional
Prioridad:
Alta
Característica ISO 25010:
Adecuación Funcional > Corrección Funcional
Subsección IEEE 830:
3.2.3 Seguimiento Espacio-Temporal y Evaluación de Riesgo
Descripción:
El sistema debe mantener la identidad única (ID de seguimiento) de cada peatón u objeto en la escalera a lo largo del tiempo mediante un algoritmo de seguimiento IoU propio, conservando el rastreo incluso ante oclusiones temporales entre personas de hasta un (1) segundo continuo.


Especificación de Requerimiento Funcional
Detalle
Requerimiento funcional N°:
RQF06
Nombre:
Validación de persistencia temporal del riesgo
Tipo:
Funcional
Prioridad:
Alta
Característica ISO 25010:
Adecuación Funcional > Corrección Funcional
Subsección IEEE 830:
3.2.3 Seguimiento Espacio-Temporal y Evaluación de Riesgo
Descripción:
El sistema debe verificar que una condición de riesgo (como imprudencia sostenida o presencia de obstáculo) permanezca un tiempo mínimo según su nivel de riesgo (ALTO: 1.2 s, MEDIO: 0.6 s, con histéresis de 1.0 s) antes de emitir una alerta confirmada; exceptuando la categoría de caída activa, la cual se alertará de manera inmediata sin esperar dicho umbral.

Nota de implementación (V1): El motor de riesgo aplica umbrales de persistencia por nivel de riesgo — PERSIST_SECONDS = {"ALTO": 1.2, "MEDIO": 0.6} — con histéresis PERSIST_GRACE = 1.0 s y enfriamiento ALERT_COOLDOWN = 15.0 s por identidad y tipo.

3.2.4. Gestión y Notificación de Alertas
Especificación de Requerimiento Funcional
Detalle
Requerimiento funcional N°:
RQF07
Nombre:
Generación y emisión de alertas de seguridad
Tipo:
Funcional
Prioridad:
Alta
Característica ISO 25010:
Adecuación Funcional > Pertinencia Funcional
Subsección IEEE 830:
3.2.4 Gestión y Notificación de Alertas
Descripción:
El sistema debe generar y transmitir una notificación de alerta estructurada una vez confirmada la condición de riesgo o caída en la escalera, incluyendo la ubicación de la cámara, fecha, hora exacta, categoría del riesgo, ID de seguimiento y la captura del fotograma del incidente.

3.3. Requisitos de Rendimiento
Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF01
Nombre:
Tasa de procesamiento de video en tiempo real
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01
Característica ISO 25010:
Eficiencia de Desempeño > Comportamiento Temporal
Subsección IEEE 830:
3.3 Requisitos de Rendimiento
Descripción:
El sistema debe procesar la señal de video de la escalera en tiempo real a una tasa de rendimiento mínima sostenida de treinta (30) fotogramas por segundo (FPS) en la GPU, garantizando la fluidez en el cálculo postural con MediaPipe y la detección con YOLOv8.

Nota de cumplimiento (V1): El motor completo mide 26.5 fps con personas presentes (MediaPipe Pose en CPU es el cuello de botella, ~18 ms por frame). El requisito de 30 FPS no se cumple de forma sostenida en escenas con peatones. Sin modo asíncrono se alcanzan ~20 fps. La latencia de alerta (<500 ms) sí se cumple con holgura (~41 ms en async mode).


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF02
Nombre:
Latencia en la emisión de alertas de seguridad
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF06, RQF07
Característica ISO 25010:
Eficiencia de Desempeño > Comportamiento Temporal
Subsección IEEE 830:
3.3 Requisitos de Rendimiento
Descripción:
El sistema debe transmitir y desplegar la notificación de alerta en el panel de supervisión con una latencia máxima menor a quinientos (500) milisegundos, contados inmediatamente después de que se confirme el cumplimiento de la regla de persistencia temporal del riesgo (tiempo mínimo según nivel: ALTO 1.2 s, MEDIO 0.6 s, con histéresis 1.0 s).


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF04
Nombre:
Capacidad de procesamiento de flujo continuo de entrada
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01
Característica ISO 25010:
Eficiencia de Desempeño > Capacidad
Subsección IEEE 830:
3.3 Requisitos de Rendimiento
Descripción:
El sistema debe garantizar la ingesta y procesamiento ininterrumpido de un (1) flujo continuo de video proveniente de la cámara fija de la escalera, manteniendo la captura completa sin descarte ni pérdida involuntaria de fotogramas por saturación de búfer.

3.4. Restricciones de Diseño
Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF03
Nombre:
Requerimiento de memoria VRAM para procesamiento gráfico
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQF02, RQF05
Característica ISO 25010:
Eficiencia de Desempeño > Utilización de Recursos
Subsección IEEE 830:
3.4 Restricciones de Diseño
Descripción:
El sistema debe ejecutar los algoritmos de inferencia (YOLOv8 para objetos y MediaPipe Pose para estimación esquelética) y seguimiento (algoritmo IoU propio) haciendo uso de aceleración por GPU NVIDIA con VRAM mínima de seis (6) gigabytes (GB).


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF05
Nombre:
Coexistencia con el sistema operativo y entorno de interfaz
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQF08
Característica ISO 25010:
Compatibilidad > Coexistencia
Subsección IEEE 830:
3.4 Restricciones de Diseño
Descripción:
El sistema debe ejecutar sus algoritmos de análisis de video por computadora e inferencia continua en GPU de manera aislada y eficiente, garantizando que el consumo intensivo de recursos gráficos y de procesamiento no cause bloqueos en los controladores de video, congelamiento del sistema operativo anfitrión ni degrada la fluidez de la interfaz del panel de supervisión.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF27
Nombre:
Restricción de actuación física y control pasivo
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF08
Característica ISO 25010:
Protección > Restricción Operativa
Subsección IEEE 830:
3.4 Restricciones de Diseño
Descripción:
El sistema debe operar exclusivamente como una herramienta de supervisión pasiva y apoyo a la decisión mediante la emisión de alertas en pantalla, quedando explícitamente restringido de ejecutar controles automáticos sobre actuadores físicos o bloqueos mecánicos en la escalera que puedan asustar a los transeúntes o provocar tropiezos secundarios.

3.5. Atributos del Sistema
3.5.1. Capacidad de Interacción / Usabilidad
Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF07
Nombre:
Aprendizabilidad de la interfaz de supervisión
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF08
Característica ISO 25010:
Capacidad de Interacción > Aprendizabilidad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
La interfaz del panel de supervisión debe ser legible e intuitiva para el personal de seguridad sin formación en ingeniería ni sistemas informáticos, permitiendo que un operador sea capaz de identificar e interpretar correctamente la información de una alerta de prueba mostrada en pantalla sin necesidad de consultar manuales técnicos ni requerir capacitación especializada.

3.5.2. Fiabilidad
Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF10
Nombre:
Precisión algorítmica de detección
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF02, RQF04
Característica ISO 25010:
Fiabilidad > Madurez
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El modelo del sistema debe alcanzar una precisión promedio media (mAP@0.5) superior al setenta y cinco por ciento (75%) en la identificación de objetos en escalones y la detección de pose/caídas en el entorno de la escalera.

Resultados medidos (V1, modelo runs/yolov8s_zoom/weights/best.pt): mAP@0.5 = 0.718 en validación y 0.703 en test. **RQNF10 no se cumple en ninguna partición** (ambas por debajo del 75% requerido). La clase que más lo frena es `persona` (val 0.70, test 0.45), cuyo ground truth de test es pseudo-etiquetado de detector y no anotación humana. Sobre clases con ground truth humano inequívoco: 0.650 en val y 0.754 en test (esta última justo por encima del 75%). Las clases de riesgo críticas superan el umbral en validación: persona_caido 0.80, persona_desequilibrio 0.86, escalera 0.83.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF11
Nombre:
Disponibilidad en el horario operativo del campus
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01
Característica ISO 25010:
Fiabilidad > Disponibilidad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe garantizar su disponibilidad e ingesta ininterrumpida de video durante el horario operativo de tránsito peatonal en el campus de la UPAO, asegurando el monitoreo continuo de las escaleras sin asumir un funcionamiento de 24 horas los 7 días de la semana.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF12
Nombre:
Resiliencia algorítmica ante variaciones de iluminación
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF02, RQF04
Característica ISO 25010:
Fiabilidad > Tolerancia a Fallos
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El motor de análisis visual debe mantener la estabilidad en la detección de posturas y clasificación de riesgos sin sufrir degradación severa ante variaciones habituales de iluminación en la escalera, como cambios de luz natural o sombras producidas por el paso de transeúntes.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF13
Nombre:
Reconexión y recuperación automática de la señal de video
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01
Característica ISO 25010:
Fiabilidad > Recuperabilidad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
Ante una interrupción o pérdida temporal de la señal de video de la cámara fija de la escalera, el sistema debe ejecutar un mecanismo de reconexión automática que restablezca la ingesta y el análisis en cuanto la transmisión se reanude, sin requerir la intervención del operador.

3.5.3. Seguridad (Security)
Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF14
Nombre:
Confidencialidad y protección de datos personales
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQF08
Característica ISO 25010:
Seguridad > Confidencialidad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe restringir la visualización y acceso a los flujos de video, fotogramas de evidencia e información de eventos exclusivamente al personal autorizados, garantizando la protección de los datos personales de las personas captadas en las escaleras según la Ley N.º 29733 y la norma ISO/IEC 27001.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF15
Nombre:
Integridad de los datos de alertas
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF07, RQF08
Característica ISO 25010:
Seguridad > Integridad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe garantizar que la estructura de datos correspondiente a cada alerta emitida (fecha, hora, categoría de riesgo, ID de seguimiento y captura del fotograma) permanezca inalterada y protegida contra corrupciones o modificaciones no autorizadas durante su transmisión y despliegue en el panel.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF16
Nombre:
Trazabilidad y registro auditable de alertas
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF07, RQF08
Característica ISO 25010:
Seguridad > Responsabilidad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe generar y conservar un registro de eventos (log) inalterable para cada alerta confirmada, asociando la marca de tiempo, cámara de origen, tipo de riesgo e ID de seguimiento, permitiendo la auditoría posterior del comportamiento de las detecciones conforme a los requerimientos de gobernanza de la norma ISO/IEC 42001.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF17
Nombre:
Resistencia de componentes y modelos de IA
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF02, RQF04
Característica ISO 25010:
Seguridad > Resistencia
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe proteger los archivos de configuración, los umbrales de detección definidos y los pesos del modelo de visión por computadora contra alteraciones, escrituras o manipulaciones no autorizadas dentro del entorno local de ejecución.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF18
Nombre:
No repudio de la evidencia de eventos de riesgo
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF07, RQF08
Característica ISO 25010:
Seguridad > No Repudio
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe vincular de manera inalterable la captura del fotograma del incidente en la escalera con su respectiva marca de tiempo, ID de seguimiento y tipo de riesgo, asegurando la validez técnica del registro como evidencia incontestable para la prevención de accidentes.

3.5.4. Mantenibilidad
Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF20
Nombre:
Arquitectura modular de componentes desacoplados
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQF02, RQF05, RQF06, RQF07, RQF08
Característica ISO 25010:
Mantenibilidad > Modularidad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe estar estructurado en componentes independientes y desacoplados (módulo de ingesta de video, motor de inferencia de visión por computadora, algoritmo de seguimiento de objetos, evaluador de persistencia temporal de riesgo e interfaz de usuario), garantizando que las modificaciones o mantenimiento realizados en la lógica interna de un módulo no alteren de forma no deseada el funcionamiento de los demás componentes.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF21
Nombre:
Modificabilidad de parámetros y modelos mediante archivos de configuración
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF02, RQF04, RQF06
Característica ISO 25010:
Mantenibilidad > Capacidad de ser Modificado
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe permitir el ajuste de los umbrales de confianza para detección y alerta, la modificación del tiempo de persistencia temporal del riesgo y la actualización o sustitución de los archivos de pesos del modelo de visión por computadora mediante archivos de configuración externos, sin requerir la modificación ni recompilación del código fuente de la aplicación.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF22
Nombre:
Capacidad de verificación y prueba mediante secuencias de video pregrabadas
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQF02, RQF07
Característica ISO 25010:
Mantenibilidad > Capacidad de ser Probado
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe permitir la ejecución de pruebas de software y validación algorítmica utilizando archivos o secuencias de video pregrabadas como fuente de entrada alternativa a la transmisión en vivo de la cámara física, facilitando la verificación del rendimiento del modelo y la comprobación de la generación de alertas en entornos de desarrollo y pruebas.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF23
Nombre:
Registro de diagnóstico e inspección técnica de fallos
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQF02, RQNF01
Característica ISO 25010:
Mantenibilidad > Analizabilidad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe generar un registro de diagnóstico técnico (log de sistema) que registre errores de ejecución, excepciones en el procesamiento gráfico e interrupciones en la ingesta del flujo de video, permitiendo al equipo de desarrollo e ingeniería analizar e identificar la causa raíz de fallos o degradaciones operativas del software.

3.5.5. Flexibilidad / Portabilidad
Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF24
Nombre:
Adaptabilidad del sistema ante cambios operativos en la zona de escaleras
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQF02
Característica ISO 25010:
Flexibilidad > Adaptabilidad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe permitir adaptarse a variaciones físicas u operativas en la zona de escaleras (como reubicaciones del ángulo de la cámara o cambios en las zonas de detección del pasamanos y peldaños) mediante archivos de configuración externos, sin reingeniería del código fuente.


Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF25
Nombre:
Escalabilidad de la arquitectura para múltiples flujos de video
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQNF04
Característica ISO 25010:
Flexibilidad > Escalabilidad
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
La arquitectura del software debe estar diseñada de forma modular para permitir la incorporación y análisis simultáneo de múltiples flujos continuos de video en fases futuras de expansión, sin requerir una reestructuración de la lógica central del pipeline de procesamiento de datos.

3.5.6. Protección (Safety)
Especificación de Requerimiento No Funcional
Detalle
Requerimiento no funcional N°:
RQNF26
Nombre:
Indicación explícita de estado no operativo ante fallos del sistema
Tipo:
No Funcional
Prioridad:
Alta
Se aplica a partir de:
RQF01, RQNF09, RQNF13
Característica ISO 25010:
Protección > Protección ante Fallos
Subsección IEEE 830:
3.5 Atributos del Sistema
Descripción:
El sistema debe transicionar a un estado explícito de "Monitoreo No Operativo" y desplegar dicha condición de forma visible en el panel de supervisión ante cualquier fallo crítico (como la pérdida de la señal de video o detención del análisis en GPU), evitando generar una falsa sensación de vigilancia activa en las escaleras.

4. APÉNDICES
APÉNDICE A - Glosario Extendido y Definición de Acrónimos
El presente glosario consolida y define de forma precisa la terminología técnica, acrónimos, métricas y marcos normativos empleados a lo largo del documento de especificación de requisitos (Sección 2 y Sección 3), sirviendo como referencia conceptual estandarizada para el sistema de seguridad en escaleras.


Término / Acrónimo
Definición
Algoritmo de seguimiento IoU (V1)
Algoritmo de seguimiento multiobjeto en tiempo real (~30 líneas) que mantiene la identidad de peatones u objetos en la escalera asociando cajas delimitadoras de alta y baja confianza mediante IoU > 0.25 y OCCLUSION_SECONDS = 1.0. En la versión V1 del sistema se implementa de manera propia en lugar del ByteTrack de Ultralytics.
MediaPipe Pose
Estimación de Pose Corporación en Tiempo Real. Framework que detecta keypoints anatómicos (manos, hombros, caderas, pies) para analizar la sujeción del pasamanos y cambios bruscos de postura/caídas en la escalera.
FPS
Frames Per Second (Fotogramas por Segundo). Tasa de rendimiento de procesamiento de video. En el proyecto se especifica un mínimo sostenido de 30 FPS.
ID de seguimiento
Identificador único y persistente asignado a cada transeúnte u objeto detectado en la escalera a lo largo de la secuencia de video.
ISO/IEC 25010
Estándar internacional que define el modelo de calidad para sistemas y software en 9 características principales.
ISO/IEC 27001
Estándar internacional para Sistemas de Gestión de Seguridad de la Información (SGSI).
ISO/IEC 42001
Estándar internacional para Sistemas de Gestión de Inteligencia Artificial (AIMS).
Ley N.º 29733
Ley de Protección de Datos Personales del Perú. Normativa aplicable a la captura y tratamiento de imágenes de transeúntes en las escaleras.
mAP@0.5
Mean Average Precision at IoU 0.5. Métrica de precisión promedio media en visión por computadora. Se exige mAP@0.5 > 75%.
Persistencia temporal
Regla que exige que un comportamiento o riesgo en escalera permanezca un tiempo mínimo según su nivel de riesgo (ALTO: 1.2 s, MEDIO: 0.6 s, con histéresis de 1.0 s) antes de confirmar la alerta (salvo en caídas activas, que se alertan de manera inmediata). En la versión V1 se implementa con PERSIST_SECONDS = {"ALTO": 1.2, "MEDIO": 0.6} y PERSIST_GRACE = 1.0 s.
Tracking
Seguimiento Espacio-Temporal. Proceso que rastrea la trayectoria de personas u objetos sobre los peldaños y tramos de la escalera.
Umbral de confianza
Confidence Threshold. Valor porcentual mínimo de certeza matemática para validar detecciones (75% pre-filtrado / 85% confirmación).
VRAM
Video Random Access Memory. Memoria dedicada de la GPU necesaria para ejecutar YOLOv8 y MediaPipe en paralelo (mínimo 6 GB).
YOLOv8
You Only Look Once (v8). Arquitectura de red neuronal para detección rápida y precisa de objetos u obstáculos en escalones.


APÉNDICE B - Diagramas de Arquitectura y Pipeline de Datos
El pipeline de procesamiento del sistema de percepción computacional para escaleras está estructurado en cuatro etapas consecutivas y desacopladas. La siguiente especificación técnica en texto estructurado define las entradas, transformaciones y salidas de cada módulo, sirviendo de base oficial para la arquitectura del sistema (V1).

Arquitectura general (V1):
- Frame (cámara / imagen / vídeo) → Inferencer (threading persistente, app/server.py:109) → RiskEngine.step() (app/risk_engine.py:971) → Panel de supervisión (Flask, server.py:1035).
- El Inferencer ejecuta en un hilo dedicado con cola; Flask atiende en multihilo (threaded=True). Cada hilo nuevo debe re-enlazar contexto CUDA (~82 ms vs ~29 ms). El modelo se carga dentro del hilo para que CUDA quede ligado a él (`Inferencer._run()`).

Detalles de Entradas, Salidas y Procesamiento por Etapa
1. Etapa 1: Ingesta y Captura de Video
Entrada: Señal de video en vivo transmitida continuamente desde la cámara fija instalada en la escalera del campus UPAO (conforme a RQF01).
Procesamiento: Captura ininterrumpida de fotogramas y almacenamiento en búfer de memoria gráfica sin pérdida de cuadros (conforme a RQNF04). Modos de ingesta: /api/detect (imagen JPEG), /api/video (MP4/AVI), /api/frame (JPEG del navegador), /api/stream (MJPEG del servidor).
Salida: Secuencia continua de fotogramas sin procesar (raw frames).

2. Etapa 2: Análisis de Visión por Computadora e Inferencia
Entrada: Secuencia continua de fotogramas sin procesar (raw frames).
Procesamiento: Inferencia combinada en GPU (VRAM ≥ 6 GB, conforme a RQNF03) del modelo propio YOLOv8 (6 clases: persona_caido, persona_sentado, persona_erguida, escalera, persona, persona_desequilibrio) para detección de objetos en escalones, más MediaPipe Pose (modo tracking, static_image_mode=False) para estimación esquelética. Se identifican las 4 categorías de riesgo (RQF02), se discrimina tránsito seguro de imprudencias (RQF03) y se aplica umbral de confianza del 75% pre-filtrado / 85% confirmación (RQF04). Un detector COCO separado (yolov8s.pt, CONF_OBSTACLE=0.25) detecta obstáculos sobre escalón (mochilas, botellas, libros, etc.).
Salida: Cajas delimitadoras de objetos, keypoints posturales de transeúntes, categorías de riesgo detectadas y puntajes de confianza asociados.

3. Etapa 3: Seguimiento Espacio-Temporal y Evaluación de Riesgo
Entrada: Cajas delimitadoras, categorías de riesgo y puntajes de confianza emitidos en la Etapa 2.
Procesamiento: Ejecución del algoritmo de seguimiento IoU propio (~30 líneas, IoU > 0.25 para asociar detecciones con tracks vivos; OCCLUSION_SECONDS = 1.0) para mantener la identidad única (ID) de cada persona u objeto en la escalera (RQF05). Evaluación de la persistencia temporal de conductas de riesgo con umbrales por nivel — PERSIST_SECONDS = {"ALTO": 1.2, "MEDIO": 0.6}, PERSIST_GRACE = 1.0 s de histéresis, ALERT_COOLDOWN = 15.0 s — con confirmación al 85% (RQF06), o detección inmediata en caso de caída activa. Se incluyeanonimización de rostros mediante MediaPipe Face Detection (pixelación 1/12) antes de guardar evidencia (RQNF14, Ley 29733).
Salida: Estructura de evento de alerta confirmada que vincula el ID de seguimiento, la categoría de riesgo en escalera, marca de tiempo, captura del fotograma como evidencia y (opcional) clip de audio de 0.5 s antes → 0.5 s después (RQNF-AUDIO).

4. Etapa 4: Gestión y Notificación de Alertas
Entrada: Estructura de evento de alerta confirmada.
Procesamiento: Empaquetado de los datos del incidente, registro inalterable en el historial auditable de eventos (alertas.jsonl + fotograma anotado en runs/alertas/<AAAA-MM-DD>/, conforme a RQNF16), extracción de audio sincronizado (RQNF-AUDIO) y transmisión inmediata del paquete de datos hacia la interfaz visual.
Salida: Despliegue visual de la alerta en el panel de supervisión (RQF08) con una latencia total menor a 500 ms (RQNF02). El panel muestra estado operativo (OPERATIVO / DEGRADADO / NO OPERATIVO), refresco cada 3 s, y autenticación por PANEL_TOKEN (RQNF19).
APÉNDICE C - Matriz de Trazabilidad Completa
La siguiente matriz consolida la totalidad de los 35 requisitos especificados para el proyecto (8 Requisitos Funcionales y 27 Requisitos No Funcionales), estableciendo la trazabilidad directa entre la característica/subcaracterística de origen de la norma ISO/IEC 25010 y la subsección correspondiente del estándar IEEE 830:

Código
Nombre del Requisito
Característica / Subcaracterística ISO/IEC 25010
Subsección IEEE 830
RQF01
Captura continua de flujo de video
Adecuación Funcional > Completitud Funcional
3.1.2 Interfaces de H/S
RQF02
Detección automatizada de categorías de riesgo
Adecuación Funcional > Completitud Funcional
3.2.2 Análisis e Inferencia
RQF03
Discriminación entre uso seguro y condición de riesgo
Adecuación Funcional > Corrección Funcional
3.2.2 Análisis e Inferencia
RQF04
Filtrado por umbrales de confianza configurables
Adecuación Funcional > Corrección Funcional
3.2.2 Análisis e Inferencia
RQF05
Seguimiento espacial con tolerancia a oclusión
Adecuación Funcional > Corrección Funcional
3.2.3 Seguimiento Espacio-Temp.
RQF06
Validación de persistencia temporal del riesgo
Adecuación Funcional > Corrección Funcional
3.2.3 Seguimiento Espacio-Temp.
RQF07
Generación y emisión de alertas de seguridad
Adecuación Funcional > Pertinencia Funcional
3.2.4 Gestión de Alertas
RQF08
Despliegue de datos en el panel de supervisión
Adecuación Funcional > Pertinencia Funcional
3.1.1 Interfaces de Usuario
RQNF01
Tasa de procesamiento de video en tiempo real (30 FPS)
Eficiencia de Desempeño > Comportamiento Temporal
3.3 Requisitos de Rendimiento
RQNF02
Latencia en la emisión de alertas de seguridad (< 500 ms)
Eficiencia de Desempeño > Comportamiento Temporal
3.3 Requisitos de Rendimiento
RQNF03
Requerimiento de memoria VRAM (GPU ≥ 6 GB VRAM)
Eficiencia de Desempeño > Utilización de Recursos
3.4 Restricciones de Diseño
RQNF04
Capacidad de procesamiento de flujo continuo de entrada
Eficiencia de Desempeño > Capacidad
3.3 Requisitos de Rendimiento
RQNF05
Coexistencia con el sistema operativo y entorno de interfaz
Compatibilidad > Coexistencia
3.4 Restricciones de Diseño
RQNF06
Reconocibilidad visual de la información de alerta
Capacidad de Interacción > Reconocibilidad de Adecuación
3.1.1 Interfaces de Usuario
RQNF07
Aprendizabilidad de la interfaz de supervisión
Capacidad de Interacción > Aprendizabilidad
3.5 Atributos del Sistema
RQNF08
Protección visual frente a errores de interpretación
Capacidad de Interacción > Protección Errores Usuario
3.1.1 Interfaces de Usuario
RQNF09
Auto-descriptividad del estado operativo del sistema
Capacidad de Interacción > Auto-descriptividad
3.1.1 Interfaces de Usuario
RQNF10
Precisión algorítmica de detección (mAP@0.5 > 75%)
Fiabilidad > Madurez
3.5 Atributos del Sistema
RQNF11
Disponibilidad en el horario operativo del campus
Fiabilidad > Disponibilidad
3.5 Atributos del Sistema
RQNF12
Resiliencia algorítmica ante variaciones de iluminación
Fiabilidad > Tolerancia a Fallos
3.5 Atributos del Sistema
RQNF13
Reconexión y recuperación automática de señal de video
Fiabilidad > Recuperabilidad
3.5 Atributos del Sistema
RQNF14
Confidencialidad y protección de datos (Ley N.º 29733)
Seguridad > Confidencialidad
3.5 Atributos del Sistema
RQNF15
Integridad de los datos de alertas
Seguridad > Integridad
3.5 Atributos del Sistema
RQNF16
Trazabilidad y registro auditable de alertas (ISO 42001)
Seguridad > Responsabilidad
3.5 Atributos del Sistema
RQNF17
Resistencia de componentes y modelos de IA
Seguridad > Resistencia
3.5 Atributos del Sistema
RQNF18
No repudio de la evidencia de eventos de riesgo
Seguridad > No Repudio
3.5 Atributos del Sistema
RQNF19
Autenticación de usuario para acceso al panel
Seguridad > Autenticidad
3.1.1 Interfaces de Usuario
RQNF20
Arquitectura modular de componentes desacoplados
Mantenibilidad > Modularidad
3.5 Atributos del Sistema
RQNF21
Modificabilidad mediante archivos de configuración
Mantenibilidad > Capacidad de ser Modificado
3.5 Atributos del Sistema
RQNF22
Capacidad de verificación con videos pregrabados
Mantenibilidad > Capacidad de ser Probado
3.5 Atributos del Sistema
RQNF23
Registro de diagnóstico e inspección técnica de fallos
Mantenibilidad > Analizabilidad
3.5 Atributos del Sistema
RQNF24
Adaptabilidad del sistema ante cambios operativos
Flexibilidad > Adaptabilidad
3.5 Atributos del Sistema
RQNF25
Escalabilidad para múltiples flujos de video
Flexibilidad > Escalabilidad
3.5 Atributos del Sistema
RQNF26
Indicación explícita de estado no operativo ante fallos
Protección > Protección ante Fallos
3.5 Atributos del Sistema
RQNF27
Restricción de actuación física y control pasivo
Protección > Restricción Operativa
3.4 Restricciones de Diseño


