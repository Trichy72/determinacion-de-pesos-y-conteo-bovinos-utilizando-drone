"""
Generador de PDF a partir del texto/markdown del agente IA.

Toma una respuesta del agente (en formato markdown) y la convierte en un PDF
profesional con la marca HMS, listo para imprimir o enviar al productor.

A diferencia de pdf_report.py (que genera reportes estructurados de lote),
este módulo es genérico: convierte cualquier texto markdown en PDF con
header/footer corporativos.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# Reutilizamos las constantes de marca y el header/footer corporativos
from reportlab.platypus import Image, PageBreak, KeepTogether
from .pdf_report import (
    PRIMARY, ACCENT, WHITE, LIGHT, _draw_header_footer,
    _encontrar_logo_color, _encontrar_logo_blanco,
    EMPRESA_NOMBRE, ASESOR_DEFAULT, TAGLINE,
    CONTACTO_TELEFONO, CONTACTO_EMAIL, CONTACTO_DIRECCION,
    CONTACTO_WEB, CONTACTO_INSTAGRAM,
)


# ---------------------------------------------------------------------
# Conversión markdown → reportlab
# ---------------------------------------------------------------------

def _convert_inline(text: str) -> str:
    """Convierte markdown inline básico (bold, italic, código) a HTML que
    reportlab Paragraph entiende."""
    # Bold: **xxx** o __xxx__
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__([^_]+)__", r"<b>\1</b>", text)
    # Italic: *xxx* o _xxx_  (cuidado de no chocar con bold)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*]+)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    # Código inline: `xxx`
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" backColor="#f0f0f0">\1</font>', text)
    # Links: [texto](url) → solo texto + url entre paréntesis
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Escape básicos para reportlab
    return text


def _parse_markdown_table(lines: list, start: int) -> tuple:
    """Parsea una tabla markdown desde la línea start. Devuelve (table_data, end_idx)."""
    # Línea 0: encabezados | sep | encabezados |
    # Línea 1: |---|---|---|
    # Líneas siguientes: datos
    if start + 1 >= len(lines):
        return None, start

    header_line = lines[start].strip().strip("|")
    sep_line = lines[start + 1].strip()
    if not sep_line.startswith("|") and not all(c in "-: |" for c in sep_line):
        return None, start
    if "---" not in sep_line:
        return None, start

    headers = [h.strip() for h in header_line.split("|")]
    table_data = [headers]
    i = start + 2
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|"):
            break
        row = [c.strip() for c in line.strip("|").split("|")]
        # Pad o truncar para que coincida con headers
        while len(row) < len(headers):
            row.append("")
        table_data.append(row[:len(headers)])
        i += 1
    return table_data, i


# ---------------------------------------------------------------------
# Generación del PDF
# ---------------------------------------------------------------------

def _construir_caratula(
    titulo: str,
    subtitulo: str,
    cliente: str,
    establecimiento: str,
    lote: str,
    raza: str,
    categoria: str,
    cantidad: int,
    peso_kg: float,
    objetivo: str,
    fecha: datetime,
) -> list:
    """Genera los elementos visuales de la carátula del informe."""
    elementos = []

    # Estilos exclusivos de portada
    cover_title = ParagraphStyle(
        "CoverTitle", fontName="Helvetica-Bold", fontSize=26,
        textColor=PRIMARY, alignment=1, leading=32, spaceAfter=8,
    )
    cover_subtitle = ParagraphStyle(
        "CoverSubtitle", fontName="Helvetica", fontSize=14,
        textColor=ACCENT, alignment=1, leading=18, spaceAfter=20,
    )
    cover_label = ParagraphStyle(
        "CoverLabel", fontName="Helvetica-Bold", fontSize=10,
        textColor=PRIMARY,
    )
    cover_value = ParagraphStyle(
        "CoverValue", fontName="Helvetica", fontSize=11,
        textColor=colors.HexColor("#1a1a1a"),
    )
    cover_section = ParagraphStyle(
        "CoverSection", fontName="Helvetica-Bold", fontSize=12,
        textColor=WHITE, alignment=1, leading=18,
    )
    cover_meta = ParagraphStyle(
        "CoverMeta", fontName="Helvetica", fontSize=9,
        textColor=colors.grey, alignment=1,
    )

    # 1) Logo grande centrado
    logo_color = _encontrar_logo_color()
    if logo_color:
        try:
            elementos.append(Spacer(1, 1.0 * cm))
            elementos.append(Image(str(logo_color), width=5 * cm, height=5 * cm,
                                    kind="proportional", hAlign="CENTER"))
            elementos.append(Spacer(1, 0.5 * cm))
        except Exception:
            elementos.append(Spacer(1, 4 * cm))
    else:
        elementos.append(Spacer(1, 3 * cm))

    # 2) Empresa y tagline
    elementos.append(Paragraph(
        f'<font color="#1B3E27"><b>{EMPRESA_NOMBRE}</b></font>',
        ParagraphStyle("EmpresaCover", fontSize=18, alignment=1,
                        spaceAfter=4, leading=22),
    ))
    elementos.append(Paragraph(
        f'<font color="#8BC53F">{TAGLINE}</font>',
        ParagraphStyle("TaglineCover", fontSize=11, alignment=1,
                        spaceAfter=30, leading=14),
    ))

    # 3) Línea divisoria lima
    sep = Table([[" "]], colWidths=[10 * cm])
    sep.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 2, ACCENT),
    ]))
    sep.hAlign = "CENTER"
    elementos.append(sep)
    elementos.append(Spacer(1, 0.6 * cm))

    # 4) Título del informe
    elementos.append(Paragraph(titulo, cover_title))
    if subtitulo:
        elementos.append(Paragraph(subtitulo, cover_subtitle))
    else:
        elementos.append(Spacer(1, 0.5 * cm))

    # 5) Bloque "DATOS DEL CLIENTE"
    elementos.append(Spacer(1, 0.5 * cm))
    titulo_cli = Table(
        [[Paragraph("DATOS DEL CLIENTE", cover_section)]],
        colWidths=[16 * cm],
    )
    titulo_cli.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PRIMARY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elementos.append(titulo_cli)

    datos_cli = []
    if cliente:
        datos_cli.append([
            Paragraph("Cliente:", cover_label),
            Paragraph(cliente, cover_value),
        ])
    if establecimiento:
        datos_cli.append([
            Paragraph("Establecimiento:", cover_label),
            Paragraph(establecimiento, cover_value),
        ])
    datos_cli.append([
        Paragraph("Fecha del informe:", cover_label),
        Paragraph(fecha.strftime("%d de %B de %Y"), cover_value),
    ])
    datos_cli.append([
        Paragraph("Asesor:", cover_label),
        Paragraph(ASESOR_DEFAULT, cover_value),
    ])

    if datos_cli:
        t_cli = Table(datos_cli, colWidths=[5.5 * cm, 10.5 * cm])
        t_cli.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.lightgrey),
        ]))
        elementos.append(t_cli)

    # 6) Bloque "DATOS DEL LOTE"
    elementos.append(Spacer(1, 0.4 * cm))
    titulo_lote = Table(
        [[Paragraph("DATOS DEL LOTE", cover_section)]],
        colWidths=[16 * cm],
    )
    titulo_lote.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PRIMARY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elementos.append(titulo_lote)

    datos_lote = []
    if lote:
        datos_lote.append([
            Paragraph("Identificación:", cover_label),
            Paragraph(lote, cover_value),
        ])
    if categoria:
        datos_lote.append([
            Paragraph("Categoría:", cover_label),
            Paragraph(categoria.title(), cover_value),
        ])
    if raza:
        datos_lote.append([
            Paragraph("Raza:", cover_label),
            Paragraph(raza.title(), cover_value),
        ])
    if cantidad > 0:
        datos_lote.append([
            Paragraph("Cantidad de animales:", cover_label),
            Paragraph(str(cantidad), cover_value),
        ])
    if peso_kg > 0:
        datos_lote.append([
            Paragraph("Peso promedio:", cover_label),
            Paragraph(f"{peso_kg:.0f} kg", cover_value),
        ])
    if objetivo:
        datos_lote.append([
            Paragraph("Objetivo productivo:", cover_label),
            Paragraph(objetivo, cover_value),
        ])

    if datos_lote:
        t_lote = Table(datos_lote, colWidths=[5.5 * cm, 10.5 * cm])
        t_lote.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.lightgrey),
        ]))
        elementos.append(t_lote)
    else:
        elementos.append(Paragraph(
            "<i>(Datos del lote disponibles en el cuerpo del informe)</i>",
            cover_meta,
        ))

    elementos.append(PageBreak())
    return elementos


def generar_pdf_informe_chat(
    output_path: str | Path,
    contenido_markdown: str,
    titulo_default: str = "Informe técnico nutricional",
    cliente: str = "",
    establecimiento: str = "",
    lote: str = "",
    raza: str = "",
    categoria: str = "",
    cantidad: int = 0,
    peso_kg: float = 0,
    objetivo: str = "",
    fecha: Optional[datetime] = None,
    incluir_caratula: bool = True,
) -> Path:
    """Convierte un texto markdown (típicamente respuesta del agente IA) en
    un PDF profesional con marca HMS y carátula de presentación."""
    output_path = Path(output_path)
    fecha = fecha or datetime.now()

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=3.0 * cm, bottomMargin=3.0 * cm,
    )

    # ------------- Estilos -------------
    title_st = ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=18, textColor=PRIMARY,
        spaceAfter=6, alignment=1,
    )
    subtitle_st = ParagraphStyle(
        "Subtitle", fontName="Helvetica", fontSize=10, textColor=colors.grey,
        spaceAfter=18, alignment=1,
    )
    h1_st = ParagraphStyle(
        "H1", fontName="Helvetica-Bold", fontSize=15, textColor=PRIMARY,
        spaceBefore=14, spaceAfter=6,
    )
    h2_st = ParagraphStyle(
        "H2", fontName="Helvetica-Bold", fontSize=12, textColor=PRIMARY,
        spaceBefore=10, spaceAfter=4,
    )
    h3_st = ParagraphStyle(
        "H3", fontName="Helvetica-Bold", fontSize=10, textColor=PRIMARY,
        spaceBefore=8, spaceAfter=3,
    )
    body_st = ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=10, leading=14,
    )
    bullet_st = ParagraphStyle(
        "Bullet", fontName="Helvetica", fontSize=10, leading=14,
        leftIndent=18, bulletIndent=8,
    )
    quote_st = ParagraphStyle(
        "Quote", fontName="Helvetica-Oblique", fontSize=10, leading=14,
        leftIndent=18, textColor=colors.grey,
    )

    story = []

    # El TÍTULO del PDF lo define el usuario en el form (titulo_default).
    # Si el markdown del agente arranca con un "# " grande, lo descartamos
    # para no duplicar el título.
    lines_raw = contenido_markdown.strip().split("\n")
    subtitulo_caratula = ""
    if lines_raw and lines_raw[0].startswith("# "):
        if titulo_default in ("", "Informe técnico nutricional"):
            titulo_default = lines_raw[0][2:].strip()
        lines_raw = lines_raw[1:]

    # Detectar si después del título hay un subtítulo en cursiva o normal
    if lines_raw and lines_raw[0].strip().startswith("*") and not lines_raw[0].strip().startswith("**"):
        subtitulo_caratula = lines_raw[0].strip().strip("*").strip()
        lines_raw = lines_raw[1:]

    # Carátula de presentación
    if incluir_caratula:
        story.extend(_construir_caratula(
            titulo=titulo_default,
            subtitulo=subtitulo_caratula,
            cliente=cliente,
            establecimiento=establecimiento,
            lote=lote,
            raza=raza,
            categoria=categoria,
            cantidad=cantidad,
            peso_kg=peso_kg,
            objetivo=objetivo,
            fecha=fecha,
        ))

    # Header del cuerpo del informe (después de la carátula)
    story.append(Paragraph(titulo_default, title_st))
    sub_partes = []
    if cliente:
        sub_partes.append(cliente)
    if lote:
        sub_partes.append(f"Lote: {lote}")
    sub_partes.append(fecha.strftime("%d/%m/%Y"))
    story.append(Paragraph(" · ".join(sub_partes), subtitle_st))

    # ------------- Convertir el cuerpo -------------
    i = 0
    while i < len(lines_raw):
        line = lines_raw[i].rstrip()
        stripped = line.strip()

        # Línea vacía → spacer
        if not stripped:
            story.append(Spacer(1, 0.15 * cm))
            i += 1
            continue

        # Tabla (empieza con | y la siguiente línea tiene ---)
        if stripped.startswith("|") and i + 1 < len(lines_raw) and "---" in lines_raw[i + 1]:
            table_data, end_i = _parse_markdown_table(lines_raw, i)
            if table_data and len(table_data) > 1:
                # Estilos para celdas de cuerpo y header (necesario para que
                # se interprete <b>, <i>, etc. en las celdas — si pasamos
                # solo el string crudo, reportlab Table NO renderiza markdown)
                cell_st = ParagraphStyle(
                    "Cell", fontName="Helvetica", fontSize=9, leading=11,
                )
                header_cell_st = ParagraphStyle(
                    "HeaderCell", fontName="Helvetica-Bold", fontSize=9,
                    leading=11, textColor=WHITE, alignment=1,
                )
                # Convertir cada celda a Paragraph (convierte inline + permite
                # que el texto envuelva si es muy largo)
                table_data_paras = []
                for row_idx, row in enumerate(table_data):
                    paragraphed_row = []
                    for cell in row:
                        cell_html = _convert_inline(str(cell))
                        st_to_use = header_cell_st if row_idx == 0 else cell_st
                        paragraphed_row.append(Paragraph(cell_html, st_to_use))
                    table_data_paras.append(paragraphed_row)

                t = Table(table_data_paras, colWidths=None, repeatRows=1)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(t)
                story.append(Spacer(1, 0.2 * cm))
                i = end_i
                continue

        # Encabezados
        if stripped.startswith("# "):
            story.append(Paragraph(_convert_inline(stripped[2:]), h1_st))
        elif stripped.startswith("## "):
            story.append(Paragraph(_convert_inline(stripped[3:]), h2_st))
        elif stripped.startswith("### "):
            story.append(Paragraph(_convert_inline(stripped[4:]), h3_st))
        # Listas con bullets
        elif stripped.startswith(("- ", "* ", "• ")):
            story.append(Paragraph(
                _convert_inline(stripped[2:]),
                bullet_st, bulletText="•",
            ))
        # Lista numerada
        elif re.match(r"^\d+\.\s", stripped):
            content = re.sub(r"^\d+\.\s", "", stripped)
            num_match = re.match(r"^(\d+)\.", stripped)
            num = num_match.group(1) if num_match else "•"
            story.append(Paragraph(
                _convert_inline(content),
                bullet_st, bulletText=f"{num}.",
            ))
        # Cita (>)
        elif stripped.startswith("> "):
            story.append(Paragraph(_convert_inline(stripped[2:]), quote_st))
        # Separador horizontal
        elif stripped in ("---", "***", "___"):
            story.append(Spacer(1, 0.1 * cm))
            t = Table([[" "]], colWidths=[16 * cm])
            t.setStyle(TableStyle([
                ("LINEABOVE", (0, 0), (-1, 0), 0.5, ACCENT),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.1 * cm))
        # Párrafo normal
        else:
            story.append(Paragraph(_convert_inline(line), body_st))

        i += 1

    doc.build(story, onFirstPage=_draw_header_footer,
              onLaterPages=_draw_header_footer)
    return output_path
