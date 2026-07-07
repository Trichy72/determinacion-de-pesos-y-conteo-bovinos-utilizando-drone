"""Genera un PDF de "Inspección visual" del lote.

Incluye:
  - Header con datos del lote y cliente
  - Sección de fotos categorizadas (bosta/animales/comedero/...)
  - Comentario de cada foto
  - Fecha de cada foto
  - Pie con datos de HMS Nutrición Animal

Uso típico:
    from src.pdf_lote_inspeccion import generar_pdf_inspeccion_lote
    pdf_bytes = generar_pdf_inspeccion_lote(lote_id=7)
    Path("informe.pdf").write_bytes(pdf_bytes)
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, PageBreak,
)

from src import database as db
from src import fotos_lote as fl


# Colores HMS (verde corporativo)
COLOR_HMS = colors.HexColor("#1e7d36")
COLOR_SOFT = colors.HexColor("#f0f7f0")
COLOR_GRIS = colors.HexColor("#6c757d")


def _logo_path() -> Optional[Path]:
    """Busca el logo color en data/ o assets/."""
    for nombre in [
        "logo_hms_color.png", "logo_color.png", "logo.png",
    ]:
        for base in ["data", "assets", "static"]:
            p = (Path(__file__).resolve().parents[1] / base / nombre)
            if p.exists():
                return p
    return None


def _header_footer(canvas, doc):
    canvas.saveState()
    # Header: logo + título
    logo = _logo_path()
    if logo:
        try:
            canvas.drawImage(
                str(logo), 1.5 * cm, A4[1] - 2.2 * cm,
                width=2.5 * cm, height=1.2 * cm,
                preserveAspectRatio=True, mask="auto",
            )
        except Exception:
            pass
    canvas.setFillColor(COLOR_HMS)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawRightString(
        A4[0] - 1.5 * cm, A4[1] - 1.5 * cm,
        "HMS Nutrición Animal",
    )
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(COLOR_GRIS)
    canvas.drawRightString(
        A4[0] - 1.5 * cm, A4[1] - 1.9 * cm,
        "Mauricio Suárez — Asesor Técnico Nutricional",
    )

    # Footer: página + fecha
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(COLOR_GRIS)
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    canvas.drawString(1.5 * cm, 1 * cm, f"Generado: {fecha}")
    canvas.drawRightString(
        A4[0] - 1.5 * cm, 1 * cm, f"Página {doc.page}",
    )
    canvas.restoreState()


def _estilos():
    base = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle(
            "Titulo",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=COLOR_HMS,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=COLOR_HMS,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.black,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=COLOR_GRIS,
        ),
    }


def _imagen_segura(path: str, max_w: float, max_h: float
                    ) -> Optional[RLImage]:
    """Carga una imagen con tamaño limitado. Devuelve None si falla."""
    try:
        # Verificar que es archivo legible
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        img = RLImage(str(p))
        # Escalar manteniendo proporción
        ratio = min(max_w / img.imageWidth, max_h / img.imageHeight)
        img.drawWidth = img.imageWidth * ratio
        img.drawHeight = img.imageHeight * ratio
        return img
    except Exception:
        return None


def generar_pdf_inspeccion_lote(lote_id: int,
                                 recordatorio_id: Optional[int] = None
                                 ) -> bytes:
    """Genera el PDF de inspección visual del lote.

    Si pasás `recordatorio_id`, solo incluye fotos de esa consulta.
    Si no, incluye todas las del lote.
    """
    # Datos del lote
    lote = db.obtener_lote(lote_id) or {}
    cli = db.obtener_cliente(lote.get("cliente_id")) or {}

    # Fotos categorizadas
    por_tipo = fl.listar_fotos_categorizadas(lote_id, recordatorio_id)
    total = sum(len(v) for v in por_tipo.values())

    # Setup PDF
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=2.8 * cm, bottomMargin=1.6 * cm,
        title=f"Inspección visual — {lote.get('identificador','lote')}",
        author="HMS Nutrición Animal",
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="normal",
    )
    doc.addPageTemplates([
        PageTemplate(
            id="MainTpl", frames=frame, onPage=_header_footer,
        ),
    ])
    estilos = _estilos()
    story = []

    # Título
    story.append(Paragraph(
        f"📸 Inspección visual — {lote.get('identificador','')}",
        estilos["titulo"],
    ))

    # Datos del lote en tabla 2 columnas
    datos = [
        ["Cliente:", cli.get("nombre", "—")],
        ["Establecimiento:",
         cli.get("establecimiento") or cli.get("localidad") or "—"],
        ["Identificador del lote:", lote.get("identificador", "—")],
        ["Categoría:", lote.get("categoria", "—")],
        ["Raza:", lote.get("raza", "—")],
        ["Cabezas iniciales:", str(lote.get("cantidad_inicial", "—"))],
        ["Peso ingreso:",
         f"{lote.get('peso_ingreso_kg', '—')} kg"],
        ["Fecha ingreso:", lote.get("fecha_ingreso", "—") or "—"],
    ]
    tbl = Table(datos, colWidths=[5 * cm, 11 * cm])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_HMS),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_SOFT),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_HMS),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Sección de fotos por categoría
    if total == 0:
        story.append(Paragraph(
            "⚠️ Este lote todavía no tiene fotos de inspección "
            "cargadas. Subí fotos desde la ficha clínica del lote "
            "para que aparezcan acá.",
            estilos["body"],
        ))
    else:
        story.append(Paragraph(
            f"Resumen: {total} foto(s) cargada(s) en "
            f"{len(por_tipo)} categoría(s).",
            estilos["caption"],
        ))
        story.append(Spacer(1, 8))

        # Renderizar cada categoría
        max_img_w = 7 * cm  # 2 por fila en A4 con márgenes
        max_img_h = 6 * cm

        for tipo_key, tipo_meta in db.TIPOS_FOTO_LOTE.items():
            fotos_tipo = por_tipo.get(tipo_key, [])
            if not fotos_tipo:
                continue
            story.append(Paragraph(
                f"{tipo_meta['emoji']} {tipo_meta['label']} "
                f"({len(fotos_tipo)})",
                estilos["h2"],
            ))

            # Armar tabla 2 columnas de imágenes
            filas = []
            fila_actual = []
            for f in fotos_tipo:
                img = _imagen_segura(
                    f.get("archivo_path", ""), max_img_w, max_img_h,
                )
                if img is None:
                    cell = [Paragraph(
                        f"⚠️ Foto #{f['id']} no disponible "
                        "(archivo borrado)",
                        estilos["caption"],
                    )]
                else:
                    cell = [img]
                # Comentario debajo
                com = (f.get("comentario") or "").strip()
                fecha = (f.get("fecha") or "")[:16]
                meta = f"📅 {fecha}"
                if com:
                    meta += f" — {com[:140]}"
                cell.append(Paragraph(meta, estilos["caption"]))
                fila_actual.append(cell)
                if len(fila_actual) == 2:
                    filas.append(fila_actual)
                    fila_actual = []
            if fila_actual:
                # Rellenar con celda vacía
                fila_actual.append([Paragraph("", estilos["caption"])])
                filas.append(fila_actual)

            for fila in filas:
                t = Table(fila, colWidths=[8.5 * cm, 8.5 * cm])
                t.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(t)

    doc.build(story)
    return buf.getvalue()
