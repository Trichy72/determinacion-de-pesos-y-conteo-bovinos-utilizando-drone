"""
Generador de reporte PDF profesional para entregar al productor.

Incluye:
  - Encabezado con datos del establecimiento + fecha + asesor
  - Resumen ejecutivo con métricas clave
  - Análisis de uniformidad con gráfico de distribución
  - Listado individual de pesos
  - Recomendación nutricional (si se incluye)
  - Pie de página con disclaimer

Usa reportlab (incluido en muchos entornos Python por defecto, si no:
    pip install reportlab matplotlib)
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from .nutritional_analysis import (
    DietaRecomendada, PesoProjection, UniformityAnalysis,
)


# =====================================================================
# IDENTIDAD CORPORATIVA HMS NUTRICIÓN ANIMAL
# =====================================================================
PRIMARY = colors.HexColor("#1B3E27")     # Verde oscuro HMS
ACCENT = colors.HexColor("#8BC53F")      # Lima HMS
DARK = colors.HexColor("#1a1a1a")
LIGHT = colors.HexColor("#f4f4f4")
WHITE = colors.HexColor("#FFFFFF")

# Logos de la marca
def _encontrar_logo_color() -> Optional[Path]:
    """Logo verde/color para fondos claros (carátula, cuerpo del informe)."""
    candidatos = [
        Path("assets/logo.png"),
        Path("assets/logo.jpg"),
        Path("assets/logo.jpeg"),
    ]
    for p in candidatos:
        if p.exists():
            return p
    return None


def _encontrar_logo_blanco() -> Optional[Path]:
    """Logo blanco para fondos oscuros (banda verde del header/pie)."""
    candidatos = [
        Path("assets/logo_blanco.png"),
        Path("assets/logo_blanco.jpg"),
        Path("assets/logo_white.png"),
    ]
    for p in candidatos:
        if p.exists():
            return p
    return None


LOGO_PATH = _encontrar_logo_color() or Path("assets/logo.png")
EMPRESA_NOMBRE = "HMS NUTRICIÓN ANIMAL"
TAGLINE = "Asesoramiento técnico nutricional"
ASESOR_DEFAULT = "Mauricio Suárez — Asesor Técnico Nutricional"

# Datos de contacto corporativos
CONTACTO_TELEFONO = "+54 2954 51-7407"
CONTACTO_EMAIL = "mauricio@hmsnutricionanimal.com.ar"
CONTACTO_DIRECCION = "Ruta Nacional 5, km 525 — Catriló, La Pampa"
CONTACTO_WEB = "hmsnutricionanimal.com.ar"
CONTACTO_INSTAGRAM = "@hmsnutricionanimal"


def _title_style():
    return ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=20,
        textColor=PRIMARY, alignment=1, spaceAfter=12,
    )


def _h2_style():
    return ParagraphStyle(
        "H2", fontName="Helvetica-Bold", fontSize=13,
        textColor=PRIMARY, spaceBefore=14, spaceAfter=6,
    )


def _body_style():
    return ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=10,
        textColor=DARK, leading=14,
    )


def _caption_style():
    return ParagraphStyle(
        "Caption", fontName="Helvetica-Oblique", fontSize=8,
        textColor=colors.grey, alignment=1, spaceBefore=4,
    )


def _draw_header_footer(canvas, doc):
    """Encabezado verde HMS + barra lima + logo + pie corporativo."""
    canvas.saveState()

    # Banda superior verde HMS
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, A4[1] - 2.2 * cm, A4[0], 2.2 * cm, fill=1, stroke=0)

    # Línea acento lima debajo
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 2.4 * cm, A4[0], 0.2 * cm, fill=1, stroke=0)

    # Logo: priorizar el BLANCO en fondo oscuro; si no hay, usar el de color
    logo_w, logo_h = 1.6 * cm, 1.6 * cm
    logo_x = 1.2 * cm
    logo_y = A4[1] - 2.0 * cm
    logo_actual = _encontrar_logo_blanco() or _encontrar_logo_color()
    if logo_actual:
        try:
            canvas.drawImage(
                str(logo_actual), logo_x, logo_y, logo_w, logo_h,
                preserveAspectRatio=True, mask="auto",
            )
        except Exception:
            pass

    # Texto del encabezado — genérico, sin mencionar drone
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(logo_x + logo_w + 0.5 * cm, A4[1] - 1.2 * cm,
                       EMPRESA_NOMBRE)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(logo_x + logo_w + 0.5 * cm, A4[1] - 1.7 * cm,
                       TAGLINE)

    # Fecha a la derecha
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 1.2 * cm, A4[1] - 1.2 * cm,
                            datetime.now().strftime("%d/%m/%Y %H:%M"))

    # Banda inferior de contacto (verde + acento lima)
    pie_alto = 2.0 * cm
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, 0, A4[0], pie_alto, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, pie_alto, A4[0], 0.1 * cm, fill=1, stroke=0)

    # Línea 1: nombre de empresa (resaltado, primera línea)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(1.2 * cm, pie_alto - 0.45 * cm, EMPRESA_NOMBRE)

    # Línea 2: asesor (debajo del nombre de empresa)
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(1.2 * cm, pie_alto - 0.80 * cm, ASESOR_DEFAULT)

    # Línea 3: dirección + tel
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(1.2 * cm, pie_alto - 1.20 * cm,
                       f"{CONTACTO_DIRECCION}    Tel: {CONTACTO_TELEFONO}")

    # Línea 4: email + web + instagram
    canvas.drawString(1.2 * cm, pie_alto - 1.55 * cm,
                       f"{CONTACTO_EMAIL}    {CONTACTO_WEB}    "
                       f"Instagram: {CONTACTO_INSTAGRAM}")

    # Página a la derecha del pie
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(A4[0] - 1.2 * cm, pie_alto - 1.55 * cm,
                            f"Página {doc.page}")

    canvas.restoreState()


def _grafico_distribucion(pesos, promedio, mediana) -> io.BytesIO:
    """Histograma + boxplot + líneas de promedio/mediana — colores HMS."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2),
                                    gridspec_kw={"width_ratios": [3, 1]})
    ax1.hist(pesos, bins=12, color="#8BC53F", edgecolor="#1B3E27", alpha=0.85)
    ax1.axvline(promedio, color="#1B3E27", linestyle="--", linewidth=2,
                label=f"Promedio {promedio:.0f} kg")
    ax1.axvline(mediana, color="#cc3300", linestyle=":", linewidth=2,
                label=f"Mediana {mediana:.0f} kg")
    ax1.set_xlabel("Peso (kg)")
    ax1.set_ylabel("Cantidad de animales")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_title("Distribución de pesos del lote")

    ax2.boxplot(pesos, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#8BC53F"),
                medianprops=dict(color="#cc3300", linewidth=2))
    ax2.set_xticks([])
    ax2.set_ylabel("Peso (kg)")
    ax2.grid(alpha=0.3)
    ax2.set_title("Boxplot")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generar_pdf(
    output_path: str | Path,
    *,
    establecimiento: str = "",
    asesor: str = "",
    lote: str = "",
    fecha: Optional[datetime] = None,
    raza: str = "",
    categoria: str = "",
    n_animales: int = 0,
    peso_promedio_kg: float = 0,
    peso_total_kg: float = 0,
    desvio_kg: float = 0,
    animales: Optional[list] = None,
    uniformidad: Optional[UniformityAnalysis] = None,
    proyeccion: Optional[PesoProjection] = None,
    dieta: Optional[DietaRecomendada] = None,
    calidad_pct: float = 100.0,
    notas_extra: str = "",
) -> Path:
    """Genera un PDF profesional con todos los datos del análisis."""
    output_path = Path(output_path)
    fecha = fecha or datetime.now()
    animales = animales or []

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=3.0 * cm, bottomMargin=3.0 * cm,
    )

    title_st = _title_style()
    h2 = _h2_style()
    body = _body_style()
    cap = _caption_style()

    story = []

    # --- BLOQUE 1: cabecera de identificación
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Análisis de pesada del lote", title_st))

    info_data = [
        ["Establecimiento:", establecimiento or "—",
         "Lote:", lote or "—"],
        ["Asesor / Técnico:", asesor or "—",
         "Fecha de pesada:", fecha.strftime("%d/%m/%Y")],
        ["Raza predominante:", raza or "—",
         "Categoría:", categoria or "—"],
        ["Cantidad de animales:", str(n_animales),
         "Calidad de captura:", f"{calidad_pct:.0f}%"],
    ]
    info_t = Table(info_data, colWidths=[3.8 * cm, 5.5 * cm, 3.3 * cm, 4.4 * cm])
    info_t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(info_t)

    # --- BLOQUE 2: métricas resumen
    story.append(Paragraph("Resumen del lote", h2))
    metric_data = [
        ["Peso promedio", f"{peso_promedio_kg:.1f} kg"],
        ["Peso total del lote", f"{peso_total_kg:.0f} kg"],
        ["Desvío estándar", f"{desvio_kg:.1f} kg"],
        ["Coeficiente de variación", f"{(desvio_kg/peso_promedio_kg*100 if peso_promedio_kg else 0):.1f} %"],
    ]
    m_t = Table(metric_data, colWidths=[8 * cm, 9 * cm])
    m_t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (0, -1), ACCENT),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(m_t)

    # --- BLOQUE 3: gráfico distribución
    if animales:
        pesos = [a.get("peso_kg", a.get("Peso (kg)", 0)) for a in animales]
        if uniformidad:
            buf = _grafico_distribucion(pesos, uniformidad.promedio_kg, uniformidad.mediana_kg)
        else:
            buf = _grafico_distribucion(pesos, sum(pesos) / len(pesos),
                                         sorted(pesos)[len(pesos) // 2])
        story.append(Paragraph("Distribución estadística", h2))
        story.append(Image(buf, width=17 * cm, height=6.8 * cm))

    # --- BLOQUE 4: análisis de uniformidad
    if uniformidad and uniformidad.n > 0:
        story.append(Paragraph("Análisis de uniformidad", h2))
        unif_t = Table([
            ["Mínimo", f"{uniformidad.min_kg:.1f} kg",
             "P10 (cabeza-baja)", f"{uniformidad.p10_kg:.1f} kg"],
            ["P25", f"{uniformidad.p25_kg:.1f} kg",
             "Mediana", f"{uniformidad.mediana_kg:.1f} kg"],
            ["P75", f"{uniformidad.p75_kg:.1f} kg",
             "P90 (cabeza-alta)", f"{uniformidad.p90_kg:.1f} kg"],
            ["Rango intercuartil", f"{uniformidad.rango_intercuartil_kg:.1f} kg",
             "Máximo", f"{uniformidad.max_kg:.1f} kg"],
        ], colWidths=[4 * cm, 3.5 * cm, 4.5 * cm, 5 * cm])
        unif_t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(unif_t)
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(f"<b>Diagnóstico:</b> {uniformidad.diagnostico}", body))
        story.append(Paragraph(f"<b>Recomendación:</b> {uniformidad.recomendacion}", body))

    # --- BLOQUE 5: proyección
    if proyeccion:
        story.append(Paragraph("Proyección de peso", h2))
        proj_text = (
            f"Con un ADG actual de <b>{proyeccion.adg_kg_dia:+.3f} kg/día</b>, "
            f"el peso proyectado a <b>{proyeccion.fecha_objetivo_dias} días</b> "
            f"es <b>{proyeccion.peso_proyectado_kg:.1f} kg</b> "
            f"(rango {proyeccion.intervalo_confianza[0]:.0f}–{proyeccion.intervalo_confianza[1]:.0f} kg)."
        )
        story.append(Paragraph(proj_text, body))
        if proyeccion.cumple_objetivo is not None:
            if proyeccion.cumple_objetivo:
                story.append(Paragraph(
                    f"✅ <b>Cumple el objetivo</b> con margen de "
                    f"{proyeccion.diferencia_objetivo_kg:+.1f} kg.", body))
            else:
                story.append(Paragraph(
                    f"⚠️ <b>NO cumple el objetivo</b>. Faltan "
                    f"{abs(proyeccion.diferencia_objetivo_kg):.1f} kg. "
                    f"ADG requerido para llegar: "
                    f"<b>{proyeccion.adg_requerido_para_objetivo:.3f} kg/día</b>.",
                    body))

    # --- BLOQUE 6: dieta recomendada (NASEM 2016)
    if dieta:
        story.append(PageBreak())
        story.append(Paragraph("Recomendación nutricional (NASEM 2016)", h2))
        dieta_t = Table([
            ["Consumo MS / día", f"{dieta.consumo_ms_kg:.1f} kg ({dieta.consumo_ms_pct_pv:.1f}% PV)"],
            ["Proteína Metabolizable (MP)", f"{dieta.mp_requerida_g:.0f} g/día"],
            ["Proteína Bruta equiv.", f"{dieta.pb_pct_ms:.1f}% MS ({dieta.pb_gramos:.0f} g/día)"],
            ["NEm (mantenimiento)", f"{dieta.nem_mcal:.2f} Mcal/día"],
            ["NEg (ganancia)", f"{dieta.neg_mcal:.2f} Mcal/día"],
            ["Energía Metabolizable", f"{dieta.em_mcal:.2f} Mcal/día ({dieta.em_concentracion_mcal_kg:.2f} Mcal/kg MS)"],
            ["FDN mínimo", f"{dieta.fdn_min_pct:.0f}% de MS"],
            ["Calcio", f"{dieta.calcio_g:.0f} g/día"],
            ["Fósforo", f"{dieta.fosforo_g:.0f} g/día"],
            ["Relación Ca:P", f"{dieta.relacion_ca_p:.2f} (objetivo 1.5–2.5)"],
        ], colWidths=[6 * cm, 11 * cm])
        dieta_t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (0, -1), ACCENT),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
            ("BACKGROUND", (1, 0), (1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(dieta_t)

        story.append(Paragraph("Composición de la dieta sugerida", h2))
        comp_data = [["Ingrediente", "% de la mezcla"]]
        for ing, pct in dieta.composicion_sugerida.items():
            comp_data.append([ing, f"{pct:.1f}%"])
        comp_t = Table(comp_data, colWidths=[12 * cm, 5 * cm])
        comp_t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(comp_t)

        if dieta.notas:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph("Notas de manejo:", h2))
            for n in dieta.notas:
                story.append(Paragraph(f"• {n}", body))

    # --- BLOQUE 7: listado individual
    if animales:
        story.append(PageBreak())
        story.append(Paragraph("Listado individual de animales", h2))
        rows = [["Animal", "Peso (kg)"]]
        for a in animales:
            tid = a.get("track_id", a.get("Animal", "—"))
            peso = a.get("peso_kg", a.get("Peso (kg)", 0))
            rows.append([str(tid), f"{peso:.1f}"])
        ind_t = Table(rows, colWidths=[4 * cm, 6 * cm])
        ind_t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ]))
        story.append(ind_t)

    # --- BLOQUE 8: notas extra y disclaimer
    if notas_extra:
        story.append(Paragraph("Notas adicionales", h2))
        story.append(Paragraph(notas_extra, body))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Este informe es generado automáticamente por procesamiento de imagen "
        "de drone con calibración óptica de 1,02 m. La precisión esperada es "
        "de ±2-5% en lotes uniformes con buen seguimiento de protocolo de "
        "captura. Para diagnóstico individual, complementar con balanza física.",
        cap,
    ))

    doc.build(story, onFirstPage=_draw_header_footer,
              onLaterPages=_draw_header_footer)
    return output_path
