#!/usr/bin/env python3
"""Genera la Especificación de Requisitos de Software (ERS, IEEE 830) del
sistema, con la estructura de la plantilla «Percepción v4 B.docx» y el
contenido real del proyecto.

    python docs/informe/generar_ers.py

Formato del curso: carátula, índice de contenido, índice de imágenes, Times
New Roman 12, interlineado doble, espaciado 0 entre párrafos, número de página
en el pie, encabezado con logo y apellidos y nombres, citas y referencias APA 7
(normas ISO y legislación). Las fichas de requisitos siguen el diseño de la
plantilla (dos columnas, cabecera gris) a espacio sencillo.

Cada cambio de la etapa de avance aparece UNA sola vez, dentro del requisito
al que pertenece; el Apéndice D solo lo referencia (historial de versiones).
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
FIG = HERE / "figuras"
OUT = HERE / "ERS IEEE 830 - Sistema de prevención de caídas en escaleras.docx"

ESTUDIANTES = [
    "Hermenegildo Rumiche, Andy Humberto",
    "Mendoza Avila, Jorge Luis",
    "Lopez Gonzalez, Jorge",
    "Coronel Ruiz, Brayan",
    "Cortez Acon, Jonaiker Mirosevic",
    "Sifuentes Berrocal, Edy Anthony",
    "Pereda Obando, Luis Go",
]
DOCENTE = "Mendoza Corpus, Carlos Alfredo"
TITULO = ("Sistema Inteligente para Prevención de Caídas y Riesgos en Escaleras "
          "Basado en Video con YOLOv8, YOLOv8-Pose y MediaPipe")
FONT = "Times New Roman"
J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT


# =========================================================================== #
# Utilidades de formato
# =========================================================================== #
def set_font(run, size=12, bold=None, italic=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(k), FONT)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    return run


def fmt(par, *, align=None, indent=True, single=False, keep_next=False):
    pf = par.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE if single else WD_LINE_SPACING.DOUBLE
    pf.first_line_indent = Cm(1.27) if indent else Cm(0)
    if align is not None:
        par.alignment = align
    if keep_next:
        pf.keep_with_next = True
    return par


def runs(par, parts, size=12):
    """`parts`: str o lista de (texto, 'b'|'i'|'bi'|'')."""
    if isinstance(parts, str):
        parts = [(parts, "")]
    for item in parts:
        t, st = (item, "") if isinstance(item, str) else item
        set_font(par.add_run(t), size=size, bold="b" in st or None,
                 italic="i" in st or None)
    return par


def add_field(par, instr: str, placeholder: str = ""):
    def fld(kind):
        r = OxmlElement("w:r")
        c = OxmlElement("w:fldChar")
        c.set(qn("w:fldCharType"), kind)
        if kind == "begin":
            c.set(qn("w:dirty"), "true")
        r.append(c)
        return r
    p = par._p
    p.append(fld("begin"))
    r = OxmlElement("w:r")
    t = OxmlElement("w:instrText")
    t.set(qn("xml:space"), "preserve")
    t.text = f" {instr} "
    r.append(t)
    p.append(r)
    p.append(fld("separate"))
    run = set_font(par.add_run(placeholder))
    p.append(fld("end"))
    return run


def P(doc, parts, *, align=J, indent=True):
    return runs(fmt(doc.add_paragraph(), align=align, indent=indent), parts)


def H(doc, title, level):
    """Títulos APA 7 (1 centrado negrita; 2 izquierda negrita; 3 izquierda
    negrita cursiva) con los estilos Heading para el índice."""
    par = doc.add_heading(level=level)
    fmt(par, indent=False, keep_next=True, align=C if level == 1 else L)
    set_font(par.add_run(title), bold=True, italic=(level == 3))
    return par


def bullets(doc, items, style="List Bullet"):
    for it in items:
        par = fmt(doc.add_paragraph(style=style), align=J, indent=False)
        runs(par, it)


def page_break(doc):
    fmt(doc.add_paragraph(), indent=False).add_run().add_break(WD_BREAK.PAGE)


_FIG = [0]
_TAB = [0]


def figure(doc, path, title, note, width_cm=15.5):
    _FIG[0] += 1
    cap = fmt(doc.add_paragraph(style="Caption"), indent=False, keep_next=True, align=L)
    set_font(cap.add_run("Figura "), bold=True, italic=False)
    add_field(cap, "SEQ Figura \\* ARABIC", str(_FIG[0])).bold = True
    br = cap.add_run()
    br.add_break(WD_BREAK.LINE)
    set_font(cap.add_run(title), italic=True, bold=False)
    for r in cap.runs:
        r.font.color.rgb = RGBColor(0, 0, 0)
    pic = fmt(doc.add_paragraph(), indent=False, keep_next=True, align=C, single=True)
    pic.add_run().add_picture(str(path), width=Cm(width_cm))
    n = fmt(doc.add_paragraph(), indent=False, align=L)
    runs(n, [("Nota. ", "i"), (note, "")])


def _borders(tbl, full=True):
    tblPr = tbl._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        on = full or edge in ("top", "bottom")
        e.set(qn("w:val"), "single" if on else "nil")
        if on:
            e.set(qn("w:sz"), "8")
            e.set(qn("w:color"), "000000")
        b.append(e)
    tblPr.append(b)


def _shade(cell, fill="F3F3F3"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _widths(tbl, widths_cm):
    tbl.autofit = False
    tblPr = tbl._tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout"):
        old = tblPr.find(qn(tag))
        if old is not None:
            tblPr.remove(old)
    tw = OxmlElement("w:tblW")
    tw.set(qn("w:w"), str(int(Cm(sum(widths_cm)).twips)))
    tw.set(qn("w:type"), "dxa")
    tblPr.append(tw)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    grid = tbl._tbl.tblGrid.findall(qn("w:gridCol"))
    for gc, w in zip(grid, widths_cm):
        gc.set(qn("w:w"), str(int(Cm(w).twips)))
    for row in tbl.rows:
        for cell, w in zip(row.cells, widths_cm):
            cell.width = Cm(w)


def _keep_together(tbl):
    """Que la ficha no se parta entre páginas (si cabe en una)."""
    rows = tbl.rows
    for i, row in enumerate(rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        if i < len(rows) - 1:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True


def _repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))


def _cell(cell, parts, size=12, bold=False):
    par = fmt(cell.paragraphs[0], indent=False, single=True, align=L)
    if isinstance(parts, str):
        parts = [(parts, "b" if bold else "")]
    runs(par, parts, size=size)


def req(doc, r: dict):
    """Ficha de requisito con el diseño de la plantilla."""
    funcional = r["code"].startswith("RQF")
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    rows = [("Especificación de Requerimiento " + ("Funcional" if funcional else "No Funcional"), "Detalle"),
            ("Requerimiento funcional N°:" if funcional else "Requerimiento no funcional N°:", r["code"]),
            ("Nombre:", r["name"]),
            ("Tipo:", "Funcional" if funcional else "No Funcional"),
            ("Prioridad:", r.get("prio", "Alta"))]
    if r.get("aplica"):
        rows.append(("Se aplica a partir de:", r["aplica"]))
    rows += [("Característica ISO/IEC 25010:", r["iso"]),
             ("Subsección IEEE 830:", r["sub"]),
             ("Descripción:", r["desc"])]
    if r.get("impl"):
        rows.append(("Implementación:", r["impl"]))
    rows.append(("Estado y verificación:", [(r["estado"] + ". ", "b"), (r["evid"], "")]))
    for i, (k, v) in enumerate(rows):
        cells = t.add_row().cells
        _cell(cells[0], k, bold=True)
        _cell(cells[1], v, bold=(i == 0))
        if i == 0:
            _shade(cells[0])
            _shade(cells[1])
            _repeat_header(t.rows[-1])
    _widths(t, [5.0, 10.9])
    _borders(t)
    _keep_together(t)
    fmt(doc.add_paragraph(), indent=False)


def data_table(doc, title, header, rows, widths, size=12, note=None):
    """Tabla de datos APA 7: «Tabla N» en negrita, título en cursiva."""
    _TAB[0] += 1
    runs(fmt(doc.add_paragraph(), indent=False, keep_next=True), [(f"Tabla {_TAB[0]}", "b")])
    runs(fmt(doc.add_paragraph(), indent=False, keep_next=True), [(title, "i")])
    t = doc.add_table(rows=0, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate([header] + rows):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            _cell(cells[j], v, size=size, bold=(i == 0))
            if i == 0:
                _shade(cells[j])
        if i == 0:
            _repeat_header(t.rows[-1])
    _widths(t, widths)
    _borders(t)
    if note:
        runs(fmt(doc.add_paragraph(), indent=False), [("Nota. ", "i"), (note, "")])
    else:
        fmt(doc.add_paragraph(), indent=False)


def reference(doc, parts):
    par = fmt(doc.add_paragraph(), indent=False, align=L)
    par.paragraph_format.left_indent = Cm(1.27)
    par.paragraph_format.first_line_indent = Cm(-1.27)
    runs(par, parts)


# =========================================================================== #
# Documento: estilos, encabezado y pie
# =========================================================================== #
def setup(doc):
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.54))
    sec.header_distance = Cm(1.0)
    sec.footer_distance = Cm(1.0)
    for name in ("Normal", "Caption", "List Bullet", "List Number", "Heading 1",
                 "Heading 2", "Heading 3", "Title"):
        st = doc.styles[name]
        st.font.name = FONT
        st.font.size = Pt(12)
        st.font.color.rgb = RGBColor(0, 0, 0)
        rpr = st.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.insert(0, fonts)
        for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            fonts.set(qn(k), FONT)
        for k in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if fonts.get(qn(k)) is not None:
                del fonts.attrib[qn(k)]
        pf = st.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    for name in ("Heading 1", "Heading 2", "Heading 3", "Title"):
        doc.styles[name].font.bold = True
    title = doc.styles["Title"]
    title.paragraph_format.alignment = C
    pPr = title.element.get_or_add_pPr()
    for b in pPr.findall(qn("w:pBdr")):
        pPr.remove(b)
    for name in ("TOC 1", "TOC 2", "TOC 3", "Table of Figures"):
        try:
            st = doc.styles[name]
        except KeyError:
            st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            st.base_style = doc.styles["Normal"]
        st.font.name = FONT
        st.font.size = Pt(12)
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        st.paragraph_format.left_indent = Cm({"TOC 2": 0.6, "TOC 3": 1.2}.get(name, 0))
    upd = OxmlElement("w:updateFields")
    upd.set(qn("w:val"), "true")
    doc.settings.element.append(upd)

    # Encabezado: logo + apellidos y nombres
    hdr = sec.header
    ht = hdr.add_table(rows=1, cols=2, width=Cm(16))
    ht.alignment = WD_TABLE_ALIGNMENT.CENTER
    _widths(ht, [1.8, 14.2])
    c0, c1 = ht.rows[0].cells
    fmt(c0.paragraphs[0], indent=False, single=True).add_run().add_picture(
        str(FIG / "logo_upao.png"), height=Cm(1.6))
    for i in range(0, len(ESTUDIANTES), 3):
        par = c1.paragraphs[0] if i == 0 else c1.add_paragraph()
        fmt(par, indent=False, single=True, align=WD_ALIGN_PARAGRAPH.RIGHT)
        set_font(par.add_run(" · ".join(ESTUDIANTES[i:i + 3])), size=8)
    hp = fmt(hdr.paragraphs[0], indent=False, single=True)
    pPr = hp._p.get_or_add_pPr()
    bd = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "1F3864")):
        bot.set(qn(k), v)
    bd.append(bot)
    pPr.append(bd)
    add_field(fmt(sec.footer.paragraphs[0], indent=False, single=True, align=C), "PAGE", "1")


def caratula(doc):
    for t in ("UNIVERSIDAD PRIVADA ANTENOR ORREGO", "FACULTAD DE INGENIERÍA",
              "ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS E INTELIGENCIA ARTIFICIAL"):
        P(doc, [(t, "b")], align=C, indent=False)
    fmt(doc.add_paragraph(), align=C, indent=False, single=True).add_run().add_picture(
        str(FIG / "logo_upao.png"), height=Cm(3.4))
    P(doc, [("Especificación de Requisitos de Software (IEEE 830)", "b")], align=C, indent=False)
    P(doc, [(TITULO, "b")], align=C, indent=False)
    P(doc, [("Estudiantes:", "b")], align=C, indent=False)
    for e in ESTUDIANTES:
        runs(fmt(doc.add_paragraph(), align=C, indent=False, single=True), e)
    P(doc, [("Docente:", "b")], align=C, indent=False)
    runs(fmt(doc.add_paragraph(), align=C, indent=False, single=True), DOCENTE)
    P(doc, [("Curso:", "b")], align=C, indent=False)
    runs(fmt(doc.add_paragraph(), align=C, indent=False, single=True), "Percepción Computacional")
    P(doc, "Trujillo – Perú, 2026", align=C, indent=False)
    page_break(doc)


def indices(doc):
    for title, instr in (("Índice de contenido", 'TOC \\o "1-3" \\h \\z \\u'),
                         ("Índice de imágenes", 'TOC \\h \\z \\c "Figura"')):
        p = H(doc, title, 1)
        p.style = doc.styles["Title"]
        add_field(fmt(doc.add_paragraph(), indent=False), instr,
                  "Abra el documento en Word y acepte actualizar los campos (o pulse F9).")
        page_break(doc)


# =========================================================================== #
# Requisitos (contenido real del proyecto)
# =========================================================================== #
ISO_AF = "Adecuación Funcional"
R = {}


def _r(code, **kw):
    R[code] = {"code": code, **kw}


_r("RQF01", name="Captura continua de flujo de video en vivo", iso=f"{ISO_AF} > Completitud Funcional",
   sub="3.1.2 Interfaces de Hardware y Software",
   desc="El sistema debe capturar de manera continua la señal de video en vivo de la cámara fija "
        "de la escalera y entregarla al motor de análisis sin intervención del operador. El video "
        "es exclusivamente en vivo: no se admiten videos ni imágenes subidos al motor de video.",
   impl="Dos fuentes en vivo: la webcam del navegador (getUserMedia → JPEG de 960 px por "
        "petición a /api/frame, hasta 30 fps) y una cámara IP o USB conectada al servidor "
        "(RTSP/índice → /api/stream, MJPEG). En esta etapa se eliminaron las rutas /api/detect "
        "(imagen subida) y /api/video (video subido) y sus pestañas.",
   estado="Cumple", evid="tests/test_api_smoke.py comprueba /api/frame y que las rutas de subida ya no existen.")
_r("RQF02", name="Detección automatizada de categorías de riesgo", iso=f"{ISO_AF} > Completitud Funcional",
   sub="3.2.2 Análisis de Visión por Computadora e Inferencia",
   desc="El sistema debe detectar en tiempo real cuatro categorías de riesgo: (1) obstáculos en "
        "los escalones, (2) no uso del pasamanos, (3) distracción (celular o lectura) y (4) caída "
        "activa o pérdida de equilibrio.",
   impl="(1) YOLOv8s COCO: mochila, bolso, maleta, botella, libro, paraguas → obstáculo ALTO. "
        "(2) MediaPipe Pose: ninguna muñeca a la altura de la barandilla y brazo separado del "
        "tronco → MEDIO. (3) Mano junto a la cara con cabeza inclinada, o celular a menos de 0,14 "
        "del rostro → MEDIO. (4) Modelo propio S7 v4 (persona caída / desequilibrio), señal de "
        "tambaleo y, desde esta etapa, esqueletos YOLOv8s-Pose de todas las personas: ángulo del "
        "tronco, velocidad y aceleración de cadera y cabeza y centro de masa frente a los pies; "
        "alerta «desequilibrio_pose» con ≥ 2 criterios sostenidos 0,25 s (umbrales en "
        "configs/app.yaml → skeleton.balance).",
   estado="Cumple", evid="tests/test_skeleton.py (caída detectada a ~0,33 s; bajar deprisa no alerta); "
        "las reglas de pasamanos y distracción están calibradas con pocas escenas.")
_r("RQF03", name="Discriminación entre uso seguro y condición de riesgo", iso=f"{ISO_AF} > Corrección Funcional",
   sub="3.2.2 Análisis de Visión por Computadora e Inferencia",
   desc="El sistema debe diferenciar el tránsito seguro (postura erguida, sujeción del "
        "pasamanos) de las conductas de riesgo.",
   impl="Cada persona recibe BAJO (erguida), MEDIO (postura no ergonómica, sin pasamanos, "
        "distracción) o ALTO (caída, desequilibrio, obstáculo) y la escena un veredicto global. "
        "La geometría de pose prevalece sobre el recorte (medido: de pie 64 % frente a 31 %).",
   estado="Cumple", evid="Veredicto visible en el panel; tests/test_units.py (geometría de pose).")
_r("RQF04", name="Filtrado por umbrales de confianza configurables", iso=f"{ISO_AF} > Corrección Funcional",
   sub="3.2.2 Análisis de Visión por Computadora e Inferencia",
   desc="El sistema debe descartar detecciones de baja certeza mediante umbrales configurables "
        "(75 % pre-filtrado y 85 % confirmación).",
   impl="thresholds.prefilter = 0,75 y thresholds.confirm = 0,85 en configs/app.yaml clasifican "
        "las detecciones en preliminares y confirmadas; la puerta de disparo de alertas es "
        "thresholds.alert = 0,35.",
   estado="Parcial (desviación consciente)",
   evid="Con 0,75 el detector no emitía ninguna alerta: sus confianzas reales están entre 0,2 y "
        "0,9 (mediana de caída 0,39). El riesgo de falsos positivos se controla con la "
        "persistencia temporal (RQF06).")
_r("RQF05", name="Seguimiento espacial con tolerancia a oclusión", iso=f"{ISO_AF} > Corrección Funcional",
   sub="3.2.3 Seguimiento Espacio-Temporal y Evaluación de Riesgo",
   desc="El sistema debe mantener la identidad única de cada persona u objeto, incluso ante "
        "oclusiones temporales de hasta un segundo.",
   impl="Cajas de persona: seguidor IoU propio (IoU > 0,25, oclusión 1 s). Esqueletos: ByteTrack "
        "de Ultralytics con filtro de Kalman (configs/tracker_esqueletos.yaml, ~3 s de oclusión). "
        "Un pasamanos que parte a una persona en dos detecciones se une en un solo esqueleto, y "
        "los puntos tapados se mantienen 0,6 s.",
   estado="Cumple", evid="tests/test_units.py (identidad estable, olvido tras oclusión) y "
        "tests/test_skeleton.py (fusión de la persona partida por el pasamanos).")
_r("RQF06", name="Validación de persistencia temporal del riesgo", iso=f"{ISO_AF} > Corrección Funcional",
   sub="3.2.3 Seguimiento Espacio-Temporal y Evaluación de Riesgo",
   desc="El sistema debe confirmar que una condición de riesgo persiste un tiempo mínimo antes de "
        "emitir una alerta, para evitar avisos por movimientos pasajeros.",
   impl="ALTO 1,2 s; MEDIO 0,6 s; histéresis 1,0 s; una alerta por persona y tipo cada 15 s. Una "
        "caída vista solo por el recorte del clasificador queda en «REVISAR» hasta 3 fotogramas "
        "seguidos. La pérdida de equilibrio del esqueleto usa su propia persistencia (0,25 s) y "
        "enfriamiento (10 s).",
   estado="Cumple (con desviación del plazo original de 3 s)",
   evid="Una caída dura ~1 s: con 3 s solo se avisaba cuando la persona llevaba rato en el suelo. "
        "tests/test_units.py (alerta tras persistencia).")
_r("RQF07", name="Generación y emisión de alertas de seguridad", iso=f"{ISO_AF} > Pertinencia Funcional",
   sub="3.2.4 Gestión y Notificación de Alertas",
   desc="El sistema debe generar una alerta estructurada con fecha, hora, categoría, identidad "
        "de seguimiento y fotograma del incidente.",
   impl="Fotograma anotado y con rostros pixelados en outputs/alerts/<AAAA-MM-DD>/ y una línea "
        "JSON por alerta en alertas.jsonl. Las alertas acústicas (RQF09) se guardan en "
        "outputs/audio_alerts/ y se muestran junto a las visuales.",
   estado="Cumple", evid="tests/test_units.py (evidencia y auditoría); "
        "modulos/audio/tests/test_integracion_panel.py.")
_r("RQF08", name="Despliegue de datos en el panel de supervisión", iso=f"{ISO_AF} > Pertinencia Funcional",
   sub="3.1.1 Interfaces de Usuario",
   desc="El sistema debe mostrar en el panel la información de cada alerta (fotograma, fecha, "
        "hora, categoría e identidad) para su revisión por el personal de seguridad.",
   impl="Panel web: pestañas «Cámara en vivo», «Audio», «Imagen» y «Clases»; lista de alertas "
        "con miniatura, descarga ZIP y borrado individual, por selección o total; tabla de "
        "esqueletos por persona; refresco cada 3 s.",
   estado="Cumple", evid="Recorrido del panel con cámara simulada (Figuras 2 a 6).")
# --- ampliación de alcance (etapa de avance) ------------------------------
_r("RQF09", name="Detección acústica de riesgos en la escalera", prio="Media",
   iso=f"{ISO_AF} > Completitud Funcional", sub="3.2.2 Análisis de Visión por Computadora e Inferencia",
   desc="Ampliación solicitada por el cliente en la etapa de avance. El sistema debe analizar el "
        "audio del micrófono en vivo para detectar pisadas (caminando o corriendo), impactos "
        "compatibles con una caída y gritos o pedidos de auxilio, en un módulo independiente que "
        "no interfiera con el video.",
   impl="modulos/audio/ (repositorio brigitteve/percepcion; no se migró su versión antigua de "
        "video). AudioWorklet a 16 kHz en bloques de 500 ms; detector por reglas de final.py "
        "(cadencia de onsets, pico RMS y factor de cresta, F0 con pYIN) y, en paralelo, el "
        "clasificador MLP de seis clases del grupo (sin reentrenar). Sensibilidad alta, media o "
        "baja; demostración simulada; análisis de archivos solo de audio. Pestaña «Audio».",
   estado="Cumple (validado con señales sintéticas)",
   evid="26 pruebas en modulos/audio/tests y 5 del AudioWorklet; ~165 ms de CPU por bloque de "
        "500 ms. Sus umbrales no están calibrados en una escalera real.")
_r("RQF10", name="Análisis de imagen de la escena", prio="Media",
   iso=f"{ISO_AF} > Completitud Funcional", sub="3.2.2 Análisis de Visión por Computadora e Inferencia",
   desc="Ampliación solicitada por el cliente. El sistema debe analizar fotografías o fotogramas "
        "de la escena (calidad óptica, personas, celulares y postura) en un módulo independiente "
        "que sustituye a la antigua pestaña de imagen.",
   impl="modulos/imagen/ (repositorio AndreCastAst, rama s7-imagen): calidad óptica, "
        "preprocesamiento adaptativo / CLAHE / gamma, YOLO11n (personas y celulares), YOLO11n-Pose "
        "(17 articulaciones e inclinación del torso), advertencias y vector de características "
        "imagen-0.1.0 del último fotograma en vivo, de solo lectura. Modelos cargados en el primer uso.",
   estado="Cumple", evid="13 pruebas en modulos/imagen/tests; tests/test_api_smoke.py analiza una fotografía.")
_r("RQF11", name="Parada ordenada del sistema", prio="Media",
   iso="Fiabilidad > Recuperabilidad", sub="3.2.5 Operación del Sistema",
   desc="Ampliación solicitada por el cliente. El sistema debe ofrecer una función interna para "
        "detenerse de forma ordenada, liberando cámara, micrófono, modelos y servidor.",
   impl="backend/services/lifecycle.py: botón «Detener sistema», POST /api/system/stop, "
        "python -m backend --stop o Ctrl+C. Deja de aceptar trabajo (HTTP 503), cierra el audio, "
        "libera los modelos de imagen, detiene la mejora en vivo y la inferencia (vacía la memoria "
        "de la GPU) y apaga el servidor. Sin token, solo se acepta desde el propio equipo.",
   estado="Cumple", evid="tests/test_units.py (LifecycleTest) y test_integracion_panel.py (parada con audio activo).")

_r("RQNF01", name="Tasa de procesamiento de video en tiempo real", aplica="RQF01",
   iso="Eficiencia de Desempeño > Comportamiento Temporal", sub="3.3 Requisitos de Rendimiento",
   desc="El sistema debe procesar el video a una tasa mínima sostenida de 30 fotogramas por segundo.",
   impl="Inferencia asíncrona en un único hilo CUDA: el navegador muestra el video a su ritmo y "
        "dibuja las cajas sobre el fotograma que se analizó (frame_id). Etapas con cadencias "
        "coprimas (pose 8, personas 6, escalera 13, obstáculos 11, clasificación 6, esqueletos 2).",
   estado="No cumple con varias personas",
   evid="Medido en RTX 5070 Laptop con 4 personas a 960×540: motor completo 23 fps (p50 41 ms, "
        "p95 69 ms); la vista en vivo se mantiene a 27-28 fps. El cuello de botella es la "
        "clasificación por recorte (22 ms), que se repite por cada persona.")
_r("RQNF02", name="Latencia en la emisión de alertas", aplica="RQF06, RQF07",
   iso="Eficiencia de Desempeño > Comportamiento Temporal", sub="3.3 Requisitos de Rendimiento",
   desc="La alerta debe mostrarse en el panel en menos de 500 ms tras confirmarse su persistencia.",
   impl="La alerta se registra en el mismo fotograma en que se confirma; el panel consulta "
        "/api/alerts cada 3 s y el estado del riesgo llega en cada respuesta de /api/frame.",
   estado="Cumple en el análisis; el refresco de la lista es de 3 s",
   evid="/api/frame responde en 3 ms (p50) y el motor tarda 69 ms (p95). La lista de alertas se "
        "actualiza por consulta periódica, no por envío inmediato.")
_r("RQNF03", name="Memoria de video (VRAM) para el procesamiento", aplica="RQF01, RQF02, RQF05",
   iso="Eficiencia de Desempeño > Utilización de Recursos", sub="3.4 Restricciones de Diseño",
   desc="El sistema debe ejecutar la inferencia con aceleración por GPU NVIDIA (requisito "
        "original: 6 GB de VRAM), con alternativa en CPU.",
   impl="PyTorch 2.11 + CUDA 12.8 (kernels desde sm_75). Alternativa en CPU con --device cpu.",
   estado="Supera", evid="Pico medido con todo activo: 0,36 GB de VRAM (ver RQNF28).")
_r("RQNF04", name="Capacidad de procesamiento de un flujo continuo", aplica="RQF01",
   iso="Eficiencia de Desempeño > Capacidad", sub="3.3 Requisitos de Rendimiento",
   desc="El sistema debe ingerir y procesar de forma ininterrumpida un flujo continuo de video.",
   impl="La ingesta es continua; si el motor no llega al ritmo de la cámara, descarta los "
        "fotogramas en vivo atrasados (cola de 3) para no acumular retraso.",
   estado="Cumple con descarte deliberado", evid="El descarte es intencional: prioriza la "
        "latencia frente a analizar cada fotograma.")
_r("RQNF05", name="Coexistencia con el sistema operativo y la interfaz", aplica="RQF01, RQF08",
   iso="Compatibilidad > Coexistencia", sub="3.4 Restricciones de Diseño",
   desc="El procesamiento intensivo no debe bloquear el sistema operativo ni la interfaz.",
   impl="Un único hilo de inferencia; el audio corre en CPU con BLAS limitado a un hilo; los "
        "módulos de audio e imagen se cargan aislados (si fallan, el video sigue).",
   estado="Cumple", evid="test_integracion_panel.py: el video responde mientras se procesa audio.")
_r("RQNF06", name="Reconocibilidad visual de la información de alerta", aplica="RQF08",
   iso="Capacidad de Interacción > Reconocibilidad de Adecuación", sub="3.1.1 Interfaces de Usuario",
   desc="Cada alerta debe presentarse en un bloque que reúna fecha, hora, categoría, identidad y fotograma.",
   impl="Tarjeta por alerta con miniatura, tipo, severidad, fecha, duración e identidad; las "
        "acústicas se marcan con «audio».", estado="Cumple", evid="Figura 2.")
_r("RQNF07", name="Aprendizabilidad de la interfaz", aplica="RQF08",
   iso="Capacidad de Interacción > Aprendizabilidad", sub="3.5.1 Capacidad de Interacción / Usabilidad",
   desc="Un operador sin formación técnica debe poder interpretar una alerta sin manuales.",
   impl="Textos en español, colores por riesgo, pestaña «Clases y requisitos» con la leyenda.",
   estado="No verificado", evid="No se han hecho pruebas con usuarios finales.")
_r("RQNF08", name="Protección visual frente a errores de interpretación", aplica="RQF08",
   iso="Capacidad de Interacción > Protección frente a Errores de Usuario", sub="3.1.1 Interfaces de Usuario",
   desc="La interfaz debe diferenciar con alto contraste el estado normal del de alerta.",
   impl="Verde (BAJO), naranja (MEDIO) y rojo (ALTO) en cajas, esqueletos y paneles; franja roja "
        "de alerta acústica visible en cualquier pestaña.", estado="Cumple", evid="Figuras 2 a 4.")
_r("RQNF09", name="Auto-descriptividad del estado operativo", aplica="RQF01, RQF08",
   iso="Capacidad de Interacción > Auto-descriptividad", sub="3.1.1 Interfaces de Usuario",
   desc="El panel debe indicar de forma permanente si la cámara y el motor funcionan.",
   impl="Estado real OPERATIVO / DEGRADADO / NO OPERATIVO desde /api/health, estado de los "
        "módulos y de la cámara («conectada» / «sin señal»).", estado="Cumple", evid="Figura 2.")
_r("RQNF10", name="Precisión algorítmica de detección", aplica="RQF02, RQF04",
   iso="Fiabilidad > Madurez", sub="3.5.2 Fiabilidad",
   desc="El modelo debe alcanzar un mAP@0,5 superior al 75 %.",
   impl="Modelo S7 v4 (YOLOv8s, 6 clases, release v1.2.0-escaleras-v4), entregable definitivo "
        "que no se reentrena.",
   estado="Cumple", evid="mAP@0,5 = 0,776 en validación y 0,768 en test (precisión 0,665 / 0,645; "
        "recall 0,786 / 0,650). La clase «persona» baja a 0,383 en test, cuyo ground truth es "
        "pseudo-etiquetado.")
_r("RQNF11", name="Disponibilidad en el horario operativo", aplica="RQF01",
   iso="Fiabilidad > Disponibilidad", sub="3.5.2 Fiabilidad",
   desc="El sistema debe estar disponible durante el horario de tránsito del campus, sin asumir 24/7.",
   impl="Arranque con scripts/run.sh y parada ordenada (RQF11).",
   estado="Depende del despliegue", evid="No se ha medido disponibilidad en un turno real.")
_r("RQNF12", name="Resiliencia ante variaciones de iluminación", aplica="RQF02, RQF04",
   iso="Fiabilidad > Tolerancia a Fallos", sub="3.5.2 Fiabilidad",
   desc="La detección no debe degradarse de forma severa ante cambios habituales de luz.",
   impl="Aumento de datos HSV en el entrenamiento; mejora Real-ESRGAN para visualizar; el módulo "
        "de imagen mide luminosidad y aplica corrección adaptativa.",
   estado="Parcial", evid="No hay una prueba cuantitativa con escenas de iluminación variable.")
_r("RQNF13", name="Reconexión automática de la señal de video", aplica="RQF01",
   iso="Fiabilidad > Recuperabilidad", sub="3.5.2 Fiabilidad",
   desc="Ante una pérdida de señal, el sistema debe reconectarse sin intervención del operador.",
   impl="Navegador: hasta 5 reintentos con espera creciente. Cámara del servidor o RTSP: "
        "open_capture con 5 intentos y espera de hasta 8 s.", estado="Cumple",
   evid="Código en frontend/templates/index.html y backend/services/video_io.py.")
_r("RQNF14", name="Confidencialidad y protección de datos personales", aplica="RQF01, RQF08",
   iso="Seguridad > Confidencialidad", sub="3.5.3 Seguridad",
   desc="El acceso a video, evidencias y eventos debe restringirse al personal autorizado, "
        "protegiendo los datos personales.",
   impl="Rostros pixelados antes de escribir cualquier evidencia; no se graba el video ni el "
        "audio (solo un fotograma por alerta y metadatos acústicos); evidencias fuera del "
        "repositorio; conservación de 7 días o 200 imágenes; panel con token.",
   estado="Cumple", evid="backend/privacy/anonymizer.py; .gitignore excluye outputs/.")
_r("RQNF15", name="Integridad de los datos de alertas", aplica="RQF07, RQF08",
   iso="Seguridad > Integridad", sub="3.5.3 Seguridad",
   desc="Los datos de cada alerta no deben alterarse durante su registro y despliegue.",
   impl="Imagen y línea JSON se escriben juntas y se borran juntas; rutas protegidas contra "
        "path traversal.", estado="Parcial", evid="No hay firma ni hash criptográfico por alerta.")
_r("RQNF16", name="Trazabilidad y registro auditable de alertas", aplica="RQF07, RQF08",
   iso="Seguridad > Responsabilidad", sub="3.5.3 Seguridad",
   desc="Cada alerta debe dejar un registro auditable con marca de tiempo, tipo e identidad.",
   impl="alertas.jsonl de solo añadido (append-only) para video y para audio.",
   estado="Cumple", evid="Registro sin protección criptográfica contra edición manual.")
_r("RQNF17", name="Resistencia de componentes y modelos de IA", aplica="RQF02, RQF04",
   iso="Seguridad > Resistencia", sub="3.5.3 Seguridad",
   desc="La configuración, los umbrales y los pesos deben protegerse contra manipulación.",
   impl="Todos los pesos se verifican con SHA-256 (tools/download_models.py --check).",
   estado="Parcial", evid="La configuración solo está protegida por los permisos del sistema operativo.")
_r("RQNF18", name="No repudio de la evidencia", aplica="RQF07, RQF08",
   iso="Seguridad > No Repudio", sub="3.5.3 Seguridad",
   desc="El fotograma debe quedar vinculado a su marca de tiempo, identidad y tipo de riesgo.",
   impl="La evidencia y su registro comparten identificador y marca de tiempo.",
   estado="Parcial", evid="Sin firma digital no hay no repudio en sentido estricto.")
_r("RQNF19", name="Autenticación para acceder al panel", aplica="RQF08",
   iso="Seguridad > Autenticidad", sub="3.1.1 Interfaces de Usuario",
   desc="El acceso al panel y a las alertas debe exigir autenticación.",
   impl="Token compartido en la variable de entorno PANEL_TOKEN, comparado en tiempo constante; "
        "sin token el panel arranca abierto y lo indica en /api/health.",
   estado="Parcial", evid="No hay usuarios ni contraseñas individuales, ni roles.")
_r("RQNF20", name="Arquitectura modular de componentes desacoplados",
   aplica="RQF01, RQF02, RQF05, RQF06, RQF07, RQF08",
   iso="Mantenibilidad > Modularidad", sub="3.5.4 Mantenibilidad",
   desc="El sistema debe dividirse en componentes independientes.",
   impl="backend/ en capas (api, services, perception, enhancement, privacy, rendering); "
        "esqueletos en perception/skeleton/; módulos aparte en modulos/audio y modulos/imagen, "
        "que no importan el video.", estado="Cumple", evid="Figura 1.")
_r("RQNF21", name="Modificabilidad mediante archivos de configuración", aplica="RQF02, RQF04, RQF06",
   iso="Mantenibilidad > Capacidad de ser Modificado", sub="3.5.4 Mantenibilidad",
   desc="Umbrales, persistencia y modelos deben cambiarse sin tocar el código.",
   impl="configs/app.yaml (umbrales, cadencias, persistencia, modelos, cámara, hardware, "
        "esqueletos, módulos) y python -m backend --set clave=valor.", estado="Cumple",
   evid="tests/test_units.py (valores por defecto y sobrescrituras).")
_r("RQNF22", name="Verificación con secuencias de video pregrabadas", aplica="RQF01, RQF02, RQF07",
   iso="Mantenibilidad > Capacidad de ser Probado", sub="3.5.4 Mantenibilidad",
   desc="El sistema debía admitir videos pregrabados como entrada alternativa para pruebas.",
   impl="Retirado: el cliente pidió que el sistema trabaje solo con video en vivo. Un video "
        "grabado puede servirse como cámara RTSP en /api/stream; las pruebas automáticas usan "
        "esqueletos sintéticos y fotogramas de ejemplo.",
   estado="Retirado por pedido del cliente", evid="Ver Apéndice D.")
_r("RQNF23", name="Registro de diagnóstico de fallos", aplica="RQF01, RQF02, RQNF01",
   iso="Mantenibilidad > Analizabilidad", sub="3.5.4 Mantenibilidad",
   desc="El sistema debe registrar errores e interrupciones para analizar su causa.",
   impl="Contadores de fallos por etapa en /api/health, avisos en consola, tiempos por etapa en "
        "cada respuesta y diagnóstico del entorno (tools/check_env.py).",
   estado="Cumple (salida por consola)", evid="No se escribe un archivo de log propio.")
_r("RQNF24", name="Adaptabilidad ante cambios en la escalera", aplica="RQF01, RQF02",
   iso="Flexibilidad > Adaptabilidad", sub="3.5.5 Flexibilidad / Portabilidad",
   desc="El sistema debe adaptarse a cambios de cámara o de escena mediante configuración.",
   impl="Fuente de cámara, umbrales, punto de fuga de las verticales (vista picada, "
        "skeleton.perspective.nadir) y perfil de hardware configurables.",
   estado="Parcial", evid="La altura de la barandilla no es configurable por zona.")
_r("RQNF25", name="Escalabilidad a múltiples flujos de video", aplica="RQF01, RQNF04",
   iso="Flexibilidad > Escalabilidad", sub="3.5.5 Flexibilidad / Portabilidad",
   desc="La arquitectura debe permitir analizar varias cámaras en fases futuras.",
   impl="Servicios desacoplados; en esta versión, una cámara.", estado="Preparado",
   evid="No implementado (Sección 2.6).")
_r("RQNF26", name="Indicación explícita de estado no operativo", aplica="RQF01, RQNF09, RQNF13",
   iso="Protección > Protección ante Fallos", sub="3.5.6 Protección (Safety)",
   desc="Ante un fallo crítico, el panel debe mostrar que el monitoreo no está operativo.",
   impl="DEGRADADO o NO OPERATIVO según fallos reales; «sin señal» al perder la cámara; "
        "«DETENIENDO» y pantalla «Sistema detenido» durante la parada (RQF11).",
   estado="Cumple", evid="Figura 6.")
_r("RQNF27", name="Restricción de actuación física y control pasivo", aplica="RQF08",
   iso="Protección > Restricción Operativa", sub="3.4 Restricciones de Diseño",
   desc="El sistema solo debe supervisar y alertar; no controla actuadores físicos.",
   impl="Solo emite alertas en pantalla, pitido opcional en el navegador y notificaciones del "
        "sistema operativo (módulo de audio).", estado="Cumple", evid="No existe código de actuación.")
_r("RQNF28", name="Ejecución en una GPU de 4 GB (GTX 1650)", prio="Alta", aplica="RQF01, RQF02, RQNF03",
   iso="Eficiencia de Desempeño > Utilización de Recursos", sub="3.4 Restricciones de Diseño",
   desc="Ampliación solicitada por el cliente. El sistema debe funcionar en una NVIDIA GeForce "
        "GTX 1650 (Turing sm_75, 4 GB, sin Tensor Cores) sin reentrenar modelos.",
   impl="backend/hardware.py activa el perfil «baja_vram» (≤ 4,5 GB o serie GTX 16): "
        "Real-ESRGAN en FP32 (las GTX 16xx fallan con FP16), entrada de la mejora en vivo de "
        "320 px y esqueletos cada 3 fotogramas. Se retiró el modelo RealESRGAN_x4plus (16,7 M "
        "parámetros), que solo usaban las subidas. Los modelos de imagen se cargan al primer uso.",
   estado="Cumple (validado con memoria limitada)",
   evid="Con la GPU limitada a 3,6 GB y el perfil forzado: pico de 0,36 GB y ~26 ms por pasada "
        "de YOLOv8s-Pose. Falta medir en una GTX 1650 física.")


def _resumen_estados() -> str:
    from collections import Counter
    n = Counter(r["estado"].split(" (")[0].split(" con ")[0].split(" en ")[0]
                for r in R.values())
    total = len(R)
    return (f"{total} requisitos. " + " · ".join(f"{k}: {v}" for k, v in n.most_common())
            + ". Detalle y evidencia de cada estado en su ficha (Sección 3).")


def ficha(doc, *codes):
    for c in codes:
        req(doc, R[c])


# =========================================================================== #
# Cuerpo
# =========================================================================== #
def seccion1(doc):
    H(doc, "1. Introducción", 1)
    P(doc, "La presente Especificación de Requisitos de Software (ERS) define los requisitos "
           "funcionales y no funcionales del sistema de percepción computacional para la "
           "prevención de caídas en las escaleras de la Universidad Privada Antenor Orrego "
           "(UPAO), y registra para cada uno cómo se implementó y en qué estado se encuentra. "
           "Sigue la estructura de la práctica recomendada IEEE Std 830-1998, cuyo contenido "
           "recoge hoy la norma ISO/IEC/IEEE 29148 (International Organization for "
           "Standardization [ISO], 2018a).")
    H(doc, "1.1. Propósito", 2)
    P(doc, ["El propósito de este documento es especificar de forma verificable el ",
            (TITULO, "i"), " y servir de referencia a tres audiencias:"])
    bullets(doc, [
        [("Equipo de desarrollo: ", "b"), ("especificación técnica para el diseño, las pruebas "
          "y el despliegue.", "")],
        [("Docente evaluador del curso de Percepción Computacional: ", "b"), ("instrumento de "
          "evaluación de la calidad del producto (ISO, 2023c), de la gestión de la IA (ISO, "
          "2023b) y de la ingeniería de requisitos (ISO, 2018a).", "")],
        [("Personal de seguridad y supervisores del campus: ", "b"), ("referencia del alcance "
          "de la herramienta de monitoreo y de sus alertas.", "")],
    ])
    H(doc, "1.2. Ámbito del Sistema", 2)
    H(doc, "1.2.1. Identificación del Sistema", 3)
    bullets(doc, [
        [("Nombre: ", "b"), (TITULO + ".", "")],
        [("Tipo de producto: ", "b"), ("aplicación web local independiente: un servidor Python "
          "(Flask) ejecuta los modelos y sirve el panel en el navegador. Usa GPU NVIDIA "
          "(CUDA 12.8; mínimo probado: perfil de GTX 1650) y puede funcionar en CPU, más lento.", "")],
        [("Versión: ", "b"), ("rama expo, modelo de detección «S7 v4» (release "
          "v1.2.0-escaleras-v4), con los módulos de audio e imagen y los esqueletos YOLOv8-Pose "
          "incorporados en la etapa de avance (octubre de 2026). Ningún modelo se reentrenó.", "")],
    ])
    H(doc, "1.2.2. Alcance Operativo (Lo que el sistema HARÁ)", 3)
    bullets(doc, [
        [("Captura en vivo ", "b"), ("de la cámara de la escalera (RQF01).", "")],
        [("Detección de las cuatro categorías de riesgo ", "b"), ("con YOLOv8, MediaPipe Pose y "
          "esqueletos YOLOv8-Pose (RQF02), distinguiendo el tránsito seguro (RQF03).", "")],
        [("Seguimiento de identidades ", "b"), ("con tolerancia a oclusión (RQF05) y "
          "confirmación temporal del riesgo (RQF06).", "")],
        [("Alertas con evidencia anonimizada ", "b"), ("(RQF07) y panel de supervisión (RQF08).", "")],
        [("Percepción acústica ", "b"), ("del micrófono en vivo (RQF09) y ", ""),
         ("análisis de imagen ", "b"), ("de la escena (RQF10), como módulos independientes.", "")],
        [("Parada ordenada ", "b"), ("de todo el sistema (RQF11).", "")],
        [("Mejora de calidad del video en vivo ", "b"), ("con Real-ESRGAN x4, solo para visualizar.", "")],
    ])
    H(doc, "1.2.3. Límites y Exclusiones del Sistema (Lo que el sistema NO HARÁ)", 3)
    bullets(doc, [
        [("Sin actuación física: ", "b"), ("no controla barreras, sirenas ni otros actuadores (RQNF27).", "")],
        [("Sin videos ni imágenes subidos al motor de video: ", "b"), ("el análisis de video es "
          "solo en vivo; el análisis de fotografías pertenece al módulo de imagen (RQF10).", "")],
        [("Sin grabación: ", "b"), ("no guarda el video ni el audio; solo un fotograma anonimizado "
          "por alerta y metadatos acústicos (RQNF14).", "")],
        [("Sin reconocimiento de personas: ", "b"), ("no identifica a nadie; las identidades de "
          "seguimiento son temporales.", "")],
        [("Sin operación 24/7 garantizada ", "b"), ("(RQNF11) y ", ""),
         ("una sola cámara ", "b"), ("en esta versión (RQNF25).", "")],
        [("Sin integración externa ", "b"), ("con sistemas de videovigilancia (VMS/NVR) ni bases "
          "de datos institucionales.", "")],
    ])
    H(doc, "1.2.4. Beneficios, Objetivos y Metas", 3)
    bullets(doc, [
        [("Monitoreo automatizado ", "b"), ("de las escaleras en lugar de la vigilancia humana continua.", "")],
        [("Respuesta oportuna: ", "b"), ("la alerta se registra en el mismo fotograma en que se "
          "confirma el riesgo (RQNF02).", "")],
        [("Prevención: ", "b"), ("detección de conductas previas a la caída (desequilibrio, "
          "distracción, falta de pasamanos, obstáculos).", "")],
        [("Bajo costo de hardware: ", "b"), ("funciona en una GPU de gama de entrada de 4 GB (RQNF28).", "")],
    ])
    H(doc, "1.3. Definiciones, Acrónimos y Abreviaturas", 2)
    data_table(doc, "Glosario del sistema", ["Término", "Definición"], [
        ["ByteTrack", "Seguidor multiobjeto con filtro de Kalman (Ultralytics); mantiene la identidad de los esqueletos durante oclusiones."],
        ["Centro de masa (CdM)", "Media ponderada de los segmentos corporales visibles (proporciones de Dempster), comparada con la base de apoyo de los pies."],
        ["ERS", "Especificación de Requisitos de Software (IEEE 830; ISO/IEC/IEEE 29148)."],
        ["FPS", "Fotogramas por segundo."],
        ["Keypoint", "Punto anatómico estimado (17 en el formato COCO de YOLOv8-Pose; 33 en MediaPipe)."],
        ["mAP@0,5", "Precisión media de detección con IoU 0,5."],
        ["MediaPipe Pose", "Estimador de pose de Google (una persona por fotograma); señales de pasamanos, distracción y tambaleo."],
        ["MLP", "Perceptrón multicapa; clasificador acústico de seis clases del módulo de audio."],
        ["Nadir", "Punto de la imagen donde convergen las verticales del mundo en una vista picada."],
        ["One Euro", "Filtro de suavizado que se adapta a la velocidad: suaviza en reposo y casi no retrasa los movimientos bruscos; se aplica a los keypoints."],
        ["Persistencia temporal", "Tiempo mínimo que un riesgo debe sostenerse para emitir alerta (ALTO 1,2 s; MEDIO 0,6 s)."],
        ["pYIN", "Estimador de frecuencia fundamental usado para detectar gritos."],
        ["Real-ESRGAN", "Red de superresolución x4 usada para mejorar la visualización del video en vivo."],
        ["S7 v4", "Modelo de detección propio (YOLOv8s, 6 clases) entregado como versión definitiva."],
        ["Torso (unidad)", "Distancia hombros-caderas de una persona; normaliza velocidades en la vista picada."],
        ["VRAM", "Memoria de la GPU."],
        ["YOLOv8 / YOLOv8-Pose / YOLO11", "Redes de Ultralytics para detección de objetos / estimación de pose / módulo de imagen."],
    ], [4.6, 11.3])
    H(doc, "1.4. Referencias", 2)
    P(doc, "La especificación se apoya en las normas ISO/IEC de requisitos (ISO, 2018a), calidad "
           "del producto (ISO, 2023c), gestión y riesgos de la IA (ISO, 2023a, 2023b), "
           "terminología de IA (ISO, 2022b), seguridad de la información (ISO, 2022a) y "
           "seguridad y salud en el trabajo (ISO, 2018b), y en la legislación peruana de "
           "protección de datos personales (Congreso de la República del Perú, 2011; Ministerio "
           "de Justicia y Derechos Humanos [MINJUSDH], 2024; Dirección General de Transparencia, "
           "Acceso a la Información Pública y Protección de Datos Personales [DGTAIPD], 2020) y "
           "de inteligencia artificial (Congreso de la República del Perú, 2023). Las "
           "referencias completas, en formato APA 7, figuran al final del documento.")
    H(doc, "1.5. Visión General del Documento", 2)
    bullets(doc, [
        [("Sección 1: ", "b"), ("propósito, ámbito, glosario y referencias.", "")],
        [("Sección 2: ", "b"), ("perspectiva, funciones, usuarios, restricciones, suposiciones y "
          "requisitos futuros.", "")],
        [("Sección 3: ", "b"), ("39 requisitos (11 funcionales y 28 no funcionales) en fichas con "
          "su implementación y su estado verificado. RQF09–RQF11 y RQNF28 se incorporaron en la "
          "etapa de avance y RQNF22 se retiró.", "")],
        [("Sección 4: ", "b"), ("apéndices: arquitectura y pipeline (A), matriz de trazabilidad "
          "(B), verificación (C) e historial de versiones (D).", "")],
    ])


def seccion2(doc):
    H(doc, "2. Descripción General", 1)
    P(doc, "El sistema supervisa una escalera del campus con una cámara fija, identifica "
           "situaciones de riesgo y avisa al personal de seguridad con evidencia anonimizada. "
           "Opcionalmente analiza el sonido del entorno y fotografías de la escena.")
    H(doc, "2.1. Perspectiva del Producto", 2)
    P(doc, "Es un producto independiente: no depende de otros sistemas de la universidad. Su "
           "entorno se define por sus entradas y salidas (Figura 1):")
    bullets(doc, [
        [("Entradas: ", "b"), ("video en vivo de la webcam del navegador o de una cámara IP/USB; "
          "audio del micrófono del navegador (módulo de audio); fotografías o fotogramas "
          "(módulo de imagen).", "")],
        [("Procesamiento: ", "b"), ("un motor de riesgo en un único hilo de GPU (detectores, "
          "pose, esqueletos, seguimiento, persistencia y alertas); los módulos de audio e "
          "imagen se ejecutan aparte.", "")],
        [("Salidas: ", "b"), ("panel web de supervisión, alertas con evidencia y registro "
          "auditable.", "")],
    ])
    figure(doc, FIG / "fig_arquitectura.png", "Arquitectura del sistema",
           "Entradas, panel web, API Flask, servicios, módulos aparte (audio e imagen), "
           "núcleo de percepción y salidas. Elaboración propia (tools/render_architecture.py).")
    H(doc, "2.2. Funciones del Producto", 2)
    for t in (
        [("Observación continua del video en vivo ", "b"), ("con inferencia asíncrona y mejora "
          "de calidad opcional.", "")],
        [("Identificación de los cuatro riesgos de la escalera: ", "b"), ("obstáculos, no uso del "
          "pasamanos, distracción y caída o pérdida de equilibrio; esta última también a partir "
          "del esqueleto de cada persona (ángulo del tronco, velocidad y aceleración de cadera y "
          "cabeza, centro de masa frente a los pies).", "")],
        [("Seguimiento y persistencia: ", "b"), ("identidad por persona y confirmación temporal "
          "antes de alertar.", "")],
        [("Alertas: ", "b"), ("fotograma anotado con rostros pixelados, registro JSON, descarga "
          "ZIP y borrado desde el panel.", "")],
        [("Percepción acústica: ", "b"), ("pisadas, carrera, impacto de caída, gritos y "
          "clasificador ML de caídas.", "")],
        [("Análisis de imagen: ", "b"), ("calidad óptica, personas, celulares y postura en "
          "fotografías.", "")],
        [("Operación: ", "b"), ("estado del sistema, autenticación por token y parada ordenada.", "")],
    ):
        par = fmt(doc.add_paragraph(style="List Number"), align=J, indent=False)
        runs(par, t)
    H(doc, "2.3. Características de los Usuarios", 2)
    P(doc, "Los usuarios son vigilantes, supervisores de infraestructura y responsables de "
           "riesgos del campus: conocen bien los protocolos de seguridad, pero no tienen "
           "formación en programación ni en administración de servidores. Por eso el panel usa "
           "colores por riesgo, textos en español y alertas con fotografía, fecha, hora y tipo, "
           "sin configuraciones técnicas en la interfaz (RQNF06–RQNF08).")
    H(doc, "2.4. Restricciones", 2)
    bullets(doc, [
        [("Gobernanza de la IA: ", "b"), ("trazabilidad de modelos e inferencias (ISO, 2023b) "
          "—pesos verificados con SHA-256 y registro de alertas— y gestión de riesgos (ISO, "
          "2023a): las alertas acústicas son «posible caída» que requiere verificación humana y "
          "el módulo de imagen nunca confirma una caída con una sola fotografía. Terminología "
          "según ISO/IEC 22989 (ISO, 2022b); uso responsable de la IA (Congreso de la "
          "República del Perú, 2023).", "")],
        [("Protección de datos: ", "b"), ("la imagen y la voz que identifican a una persona son "
          "datos personales (Congreso de la República del Perú, 2011; MINJUSDH, 2024); el "
          "tratamiento con cámaras sigue la directiva de videovigilancia (DGTAIPD, 2020). Se "
          "pixelan los rostros, no se graba video ni audio y la evidencia se conserva 7 días.", "")],
        [("Seguridad de la información: ", "b"), ("controles de acceso inspirados en ISO/IEC "
          "27001 (ISO, 2022a): token del panel, parada remota solo con token y rutas de "
          "evidencia protegidas.", "")],
        [("Hardware: ", "b"), ("GPU NVIDIA desde Turing (sm_75) con 4 GB; perfil automático para "
          "la GTX 1650 (RQNF28). Una sola cámara por instancia.", "")],
        [("Calidad del producto: ", "b"), ("cada requisito se traza a una característica de "
          "ISO/IEC 25010 (ISO, 2023c) en el Apéndice B.", "")],
        [("Modelos: ", "b"), ("no se reentrena ningún modelo; se usan el S7 v4 y modelos "
          "preentrenados públicos (YOLOv8s COCO, YOLOv8s-Pose, YOLO11n, Real-ESRGAN, MediaPipe) "
          "y el MLP acústico del grupo de audio.", "")],
        [("Seguridad física: ", "b"), ("el sistema reduce el riesgo de caídas pero no sustituye "
          "las medidas de seguridad y salud en el trabajo (ISO, 2018b).", "")],
    ])
    H(doc, "2.5. Suposiciones y Dependencias", 2)
    bullets(doc, [
        [("Cámara fija ", "b"), ("en posición alta y diagonal (vista picada) con iluminación "
          "suficiente; el pasamanos puede tapar parte del cuerpo.", "")],
        [("Red local y energía estables; ", "b"), ("ante cortes de la señal actúa la reconexión "
          "automática (RQNF13).", "")],
        [("Datos de entrenamiento del S7 v4: ", "b"), ("Fall Detection Dataset, Stairs y MOT15 "
          "(Kaggle), Le2i Fall Detection y videos propios de escaleras de la UPAO, unificados en "
          "el dataset andres31416/stairs_fallhuman (Hugging Face). SisFall, FDD y URFD se "
          "descartaron por requerir registro.", "")],
        [("Software: ", "b"), ("Python 3.10–3.12, PyTorch 2.11 + CUDA 12.8 (controlador NVIDIA "
          "≥ 570), Ultralytics 8.4, MediaPipe 0.10.21, NumPy 1.26, OpenCV 4.10, Flask 3.1, lap, "
          "librosa y scikit-learn 1.7 (instalación automática con scripts/setup.sh).", "")],
        [("Personal: ", "b"), ("el panel está atendido durante el horario operativo.", "")],
    ])
    H(doc, "2.6. Requisitos Futuros", 2)
    bullets(doc, [
        [("Varias cámaras ", "b"), ("(RQNF25) e integración con la videovigilancia institucional.", "")],
        [("Calibración con grabaciones reales ", "b"), ("de los umbrales de equilibrio, "
          "pasamanos, distracción y audio, y medición del nadir de la cámara instalada.", "")],
        [("Usuarios y roles ", "b"), ("en lugar del token compartido (RQNF19) y firma de las "
          "evidencias (RQNF15, RQNF18).", "")],
        [("Fusión audio-video ", "b"), ("de las alertas y notificaciones móviles.", "")],
        [("Mejor rendimiento ", "b"), ("con varias personas: clasificación por recorte en lote "
          "para alcanzar 30 fps sostenidos (RQNF01).", "")],
    ])


def seccion3(doc):
    H(doc, "3. Requisitos Específicos", 1)
    P(doc, "Cada ficha conserva el diseño de la especificación original y añade la "
           "implementación y el estado verificado. Estados: Cumple, Parcial, No cumple, No "
           "verificado, Preparado y Retirado.")
    H(doc, "3.1. Interfaces Externas", 2)
    H(doc, "3.1.1. Interfaces de Usuario", 3)
    ficha(doc, "RQF08", "RQNF06", "RQNF08", "RQNF09", "RQNF19")
    figure(doc, FIG / "fig_camara_en_vivo.png", "Pestaña «Cámara en vivo» del panel de supervisión",
           "Cámara simulada del navegador. Botón «Detener sistema», estado de los módulos, "
           "riesgo de la escena, estado del sistema y alertas.")
    H(doc, "3.1.2. Interfaces de Hardware y Software", 3)
    ficha(doc, "RQF01")
    data_table(doc, "Interfaces de software (API HTTP)", ["Ruta", "Uso"], [
        ["POST /api/frame", "Fotograma JPEG en vivo → cajas, esqueletos (17 puntos por persona), riesgo y estado"],
        ["GET /api/stream", "MJPEG de la cámara del servidor o RTSP"],
        ["GET /api/health", "Estado, GPU, perfil de hardware, módulos"],
        ["GET /api/alerts · DELETE /api/alertas · GET /api/alertas.zip", "Alertas de video y audio"],
        ["POST /api/system/stop", "Parada ordenada (RQF11)"],
        ["/api/audio/*", "Sesiones de micrófono, sensibilidad, simulación y archivo de audio (RQF09)"],
        ["/api/imagen/*", "Análisis de fotografía y vector de características (RQF10)"],
    ], [6.4, 9.5])
    H(doc, "3.2. Funciones", 2)
    P(doc, [("Criterio de organización: ", "b"), ("jerarquía funcional del pipeline de datos, "
            "porque hay un único perfil de usuario y el flujo es lineal: ingesta → análisis e "
            "inferencia → seguimiento y evaluación del riesgo → gestión de alertas → operación "
            "del sistema.", "")])
    H(doc, "3.2.1. Ingesta y Captura de Video", 3)
    P(doc, "La entrada de la señal se especifica en RQF01 (Sección 3.1.2); su capacidad, en "
           "RQNF04 (Sección 3.3).")
    H(doc, "3.2.2. Análisis de Visión por Computadora e Inferencia", 3)
    ficha(doc, "RQF02")
    figure(doc, FIG / "fig_esqueletos.png", "Esqueletos YOLOv8-Pose sobre el video en vivo",
           "Video de prueba con personas usado como cámara. Cada persona tiene identidad (P1–P3), "
           "huesos por grupo, centro de masa (cruz) y métricas en la tabla lateral. Las etiquetas "
           "de las cajas («Persona caída») provienen del clasificador por recorte y no del "
           "esqueleto, que evalúa a las tres personas como estables.")
    ficha(doc, "RQF03", "RQF04", "RQF09")
    figure(doc, FIG / "fig_modulo_audio.png", "Pestaña «Audio»: módulo de percepción acústica",
           "Micrófono simulado del navegador y sesión sintética de final.py con sus niveles de riesgo.")
    ficha(doc, "RQF10")
    figure(doc, FIG / "fig_modulo_imagen.png", "Pestaña «Imagen»: YOLO11 y YOLO11-Pose sobre una fotografía",
           "Imagen de ejemplo de Ultralytics (bus.jpg) con calidad óptica, preprocesamiento, "
           "advertencias y JSON del módulo. La franja roja es una alerta acústica de la prueba.")
    H(doc, "3.2.3. Seguimiento Espacio-Temporal y Evaluación de Riesgo", 3)
    ficha(doc, "RQF05", "RQF06")
    H(doc, "3.2.4. Gestión y Notificación de Alertas", 3)
    ficha(doc, "RQF07")
    H(doc, "3.2.5. Operación del Sistema", 3)
    ficha(doc, "RQF11")
    figure(doc, FIG / "fig_sistema_detenido.png", "Panel tras pulsar «Detener sistema»",
           "Modelos liberados y servidor apagado; para volver a usarlo se arranca desde la consola.",
           width_cm=14)
    H(doc, "3.3. Requisitos de Rendimiento", 2)
    ficha(doc, "RQNF01", "RQNF02", "RQNF04")
    H(doc, "3.4. Restricciones de Diseño", 2)
    ficha(doc, "RQNF03", "RQNF28", "RQNF05", "RQNF27")
    H(doc, "3.5. Atributos del Sistema", 2)
    H(doc, "3.5.1. Capacidad de Interacción / Usabilidad", 3)
    ficha(doc, "RQNF07")
    H(doc, "3.5.2. Fiabilidad", 3)
    ficha(doc, "RQNF10", "RQNF11", "RQNF12", "RQNF13")
    H(doc, "3.5.3. Seguridad", 3)
    ficha(doc, "RQNF14", "RQNF15", "RQNF16", "RQNF17", "RQNF18")
    H(doc, "3.5.4. Mantenibilidad", 3)
    ficha(doc, "RQNF20", "RQNF21", "RQNF22", "RQNF23")
    H(doc, "3.5.5. Flexibilidad / Portabilidad", 3)
    ficha(doc, "RQNF24", "RQNF25")
    H(doc, "3.5.6. Protección (Safety)", 3)
    ficha(doc, "RQNF26")


def seccion4(doc):
    H(doc, "4. Apéndices", 1)
    H(doc, "Apéndice A. Arquitectura y Pipeline de Datos", 2)
    P(doc, "La arquitectura se muestra en la Figura 1. El pipeline, por etapa:")
    data_table(doc, "Etapas del pipeline", ["Etapa", "Entrada → procesamiento → salida"], [
        ["1. Ingesta", "Cámara en vivo → JPEG por petición (/api/frame) o MJPEG (/api/stream) → fotograma BGR."],
        ["2. Análisis", "Fotograma → personas y obstáculos (YOLOv8s COCO), escalera y postura (S7 v4), pose (MediaPipe), esqueletos con identidad (YOLOv8s-Pose + ByteTrack, filtro One Euro, cinemática) → detecciones y señales."],
        ["3. Seguimiento y riesgo", "Detecciones → seguidor IoU, confirmación de caídas, pérdida de equilibrio del esqueleto, persistencia → riesgos confirmados."],
        ["4. Alertas", "Riesgo confirmado → fotograma anotado y anonimizado + línea JSON → panel."],
        ["Módulo de audio", "PCM 16 kHz del navegador → reglas de final.py + MLP → alertas acústicas (solo metadatos)."],
        ["Módulo de imagen", "Fotografía o fotograma → calidad, preprocesamiento, YOLO11 + pose → JSON e imagen anotada."],
    ], [3.8, 12.1])
    H(doc, "Apéndice B. Matriz de Trazabilidad", 2)
    order = ["RQF01", "RQF02", "RQF03", "RQF04", "RQF05", "RQF06", "RQF07", "RQF08", "RQF09",
             "RQF10", "RQF11"] + [f"RQNF{i:02d}" for i in range(1, 29)]
    data_table(doc, "Matriz de trazabilidad ISO/IEC 25010 – IEEE 830 y estado",
               ["Código", "Requisito", "Característica ISO/IEC 25010", "Subsección", "Estado"],
               [[c, R[c]["name"], R[c]["iso"], R[c]["sub"].split(" ")[0], R[c]["estado"]]
                for c in order],
               [1.8, 4.4, 4.2, 2.2, 3.3], size=10,
               note=_resumen_estados())
    H(doc, "Apéndice C. Verificación", 2)
    data_table(doc, "Pruebas automatizadas", ["Conjunto", "Qué verifica", "Resultado"], [
        ["tests/test_units.py", "Configuración, seguidor, pose, persistencia, alertas, perfil GTX 1650, parada", "13 de 13"],
        ["tests/test_skeleton.py", "Filtrado, One Euro, oclusión, cinemática, nadir, caída, bajar deprisa, pasamanos, fusión", "14 de 14"],
        ["tests/test_api_smoke.py", "Servidor real con modelos: salud, rutas de subida retiradas, video en vivo con esqueletos, imagen, alertas", "8 de 8"],
        ["modulos/audio/tests", "Sesiones, PCM, modelo ML (hash y predicciones), integración con el panel", "26 de 26"],
        ["modulos/imagen/tests", "Calidad, preprocesamiento, posturas, ROI, clips, características", "13 de 13"],
        ["Worklet de audio (Node)", "Remuestreo a 16 kHz", "5 de 5"],
    ], [4.4, 8.6, 2.9], note="python -m pytest (74 pruebas) y node --test.")
    data_table(doc, "Mediciones de rendimiento", ["Medida", "Resultado"], [
        ["Motor completo, 4 personas, 960×540 (RTX 5070 Laptop)", "23 fps; p50 41 ms; p95 69 ms"],
        ["Respuesta de /api/frame (asíncrona)", "3 ms (p50)"],
        ["YOLOv8s-Pose por pasada", "~26 ms"],
        ["Pico de VRAM, perfil GTX 1650, todo activo", "0,36 GB (GPU limitada a 3,6 GB)"],
        ["Audio por bloque de 500 ms", "~165 ms de CPU"],
        ["mAP@0,5 del S7 v4", "0,776 validación · 0,768 test"],
    ], [9.4, 6.5])
    H(doc, "Apéndice D. Historial de Versiones", 2)
    data_table(doc, "Cambios de la especificación", ["Versión", "Cambio", "Requisitos"], [
        ["V4 (semana 7)", "Modelo S7 v4, motor de riesgo modular, Real-ESRGAN en vivo, anonimización", "RQF01–RQF08, RQNF01–RQNF27"],
        ["Avance, oct. 2026", "Solo video en vivo: se retiran imagen y video subidos", "RQF01, RQNF22 (retirado)"],
        ["Avance, oct. 2026", "Perfil de GPU de 4 GB (GTX 1650)", "RQNF28 (nuevo), RQNF03"],
        ["Avance, oct. 2026", "Módulo de audio (brigitteve/percepcion)", "RQF09 (nuevo)"],
        ["Avance, oct. 2026", "Módulo de imagen (AndreCastAst, s7-imagen)", "RQF10 (nuevo)"],
        ["Avance, oct. 2026", "Parada ordenada del sistema", "RQF11 (nuevo), RQNF26"],
        ["Avance, oct. 2026", "Esqueletos YOLOv8-Pose y pérdida de equilibrio", "RQF02, RQF05, RQF06"],
    ], [3.3, 8.2, 4.4])


def referencias(doc):
    page_break(doc)
    H(doc, "Referencias", 1)
    N, I = "", "i"
    for r in [
        [("Congreso de la República del Perú. (2011, 3 de julio). ", N),
         ("Ley N.° 29733, Ley de Protección de Datos Personales", I), (". Diario Oficial El Peruano.", N)],
        [("Congreso de la República del Perú. (2023, 5 de julio). ", N),
         ("Ley N.° 31814, Ley que promueve el uso de la inteligencia artificial en favor del "
          "desarrollo económico y social del país", I), (". Diario Oficial El Peruano.", N)],
        [("Dirección General de Transparencia, Acceso a la Información Pública y Protección de "
          "Datos Personales. (2020). ", N),
         ("Directiva N.° 01-2020-JUS/DGTAIPD, Tratamiento de datos personales mediante sistemas "
          "de videovigilancia", I), (". Ministerio de Justicia y Derechos Humanos.", N)],
        [("International Organization for Standardization. (2018a). ", N),
         ("Systems and software engineering — Life cycle processes — Requirements engineering", I),
         (" (ISO/IEC/IEEE Standard No. 29148:2018). https://www.iso.org/standard/72089.html", N)],
        [("International Organization for Standardization. (2018b). ", N),
         ("Occupational health and safety management systems — Requirements with guidance for use", I),
         (" (ISO Standard No. 45001:2018). https://www.iso.org/standard/63787.html", N)],
        [("International Organization for Standardization. (2022a). ", N),
         ("Information security, cybersecurity and privacy protection — Information security "
          "management systems — Requirements", I),
         (" (ISO/IEC Standard No. 27001:2022). https://www.iso.org/standard/27001", N)],
        [("International Organization for Standardization. (2022b). ", N),
         ("Information technology — Artificial intelligence — Artificial intelligence concepts "
          "and terminology", I),
         (" (ISO/IEC Standard No. 22989:2022). https://www.iso.org/standard/74296.html", N)],
        [("International Organization for Standardization. (2023a). ", N),
         ("Information technology — Artificial intelligence — Guidance on risk management", I),
         (" (ISO/IEC Standard No. 23894:2023). https://www.iso.org/standard/77304.html", N)],
        [("International Organization for Standardization. (2023b). ", N),
         ("Information technology — Artificial intelligence — Management system", I),
         (" (ISO/IEC Standard No. 42001:2023). https://www.iso.org/standard/81230.html", N)],
        [("International Organization for Standardization. (2023c). ", N),
         ("Systems and software engineering — Systems and software Quality Requirements and "
          "Evaluation (SQuaRE) — Product quality model", I),
         (" (ISO/IEC Standard No. 25010:2023). https://www.iso.org/standard/78176.html", N)],
        [("Ministerio de Justicia y Derechos Humanos. (2024, 30 de noviembre). ", N),
         ("Decreto Supremo N.° 016-2024-JUS, que aprueba el Reglamento de la Ley N.° 29733, Ley "
          "de Protección de Datos Personales", I), (". Diario Oficial El Peruano.", N)],
    ]:
        reference(doc, r)


def main():
    doc = Document()
    setup(doc)
    caratula(doc)
    indices(doc)
    seccion1(doc)
    seccion2(doc)
    seccion3(doc)
    seccion4(doc)
    referencias(doc)
    doc.save(OUT)
    print(f"guardado {OUT}")
    print(_resumen_estados())


if __name__ == "__main__":
    main()
