"""DMI extendido — Consumo de Materia Seca con factores ambientales.

Calcula el DMI (Dry Matter Intake) base por categoría usando el NRC/NASEM
2016 y le aplica modificadores ambientales para predecir el consumo real
esperado en función del clima de la semana.

Filosofía:
  - DMI base: una predicción razonable según peso, categoría, ADPV
    objetivo y energía de la dieta (NASEM 2016).
  - Modificadores ambientales: ajustes multiplicativos sobre el DMI
    base según frío, calor, humedad, barro y acumulación de estrés.
  - Caps razonables: el DMI nunca cae más de 35% ni sube más de 15%
    sobre el base — más allá de eso, el animal pasa a anorexia y los
    valores típicos no aplican.

Uso:
    from src.dmi import dmi_proyectado

    resultado = dmi_proyectado(
        peso_kg=220, categoria="vaquillona", raza="angus",
        adpv_objetivo_kg=1.05,
        energia_dieta_mcal_em_kg_ms=2.75,
        clima_diario={...},  # dict de Open-Meteo daily
    )
    # resultado = {
    #     "dmi_base_kg_dia": 5.5,
    #     "dmi_ajustado_kg_dia": (5.2, 5.7),   # rango con incertidumbre
    #     "factor_ajuste_pct": (-5, +3),       # delta vs base
    #     "razones": ["Frío moderado: +3-5%", "HR alta sostenida: -2-3%"],
    #     "supuestos": "...",
    #     "fuente": "NRC/NASEM 2016 + ajustes Pampa Húmeda",
    # }
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# =====================================================================
# DMI BASE POR CATEGORÍA (% de peso vivo, NASEM 2016 + práctica AR)
# =====================================================================

# Cada categoría tiene un rango de % PV típico. El centro del rango se
# usa como punto base; el rango se amplía para reflejar la
# incertidumbre del cálculo.
DMI_BASE_PCT_PV = {
    "ternero": (2.6, 3.2),       # destete a 250 kg, alta tasa de crec.
    "recria": (2.4, 2.8),        # 200-350 kg
    "novillito": (2.3, 2.6),     # 350-450 kg
    "novillo": (1.9, 2.4),       # >450 kg, terminación
    "vaquillona": (2.3, 2.6),    # recría con destino servicio
    "vaca": (1.9, 2.2),          # adulta, cría
    "toro": (1.8, 2.1),          # adulto
}


# =====================================================================
# FACTORES DE AJUSTE AMBIENTAL
# =====================================================================
# Cada factor devuelve un (delta_pct_min, delta_pct_max) y una razón
# textual. Los deltas son ADITIVOS entre sí (no multiplicativos),
# después se aplican como un % global con cap.

def _factor_frio(t_min: float, t_max: float) -> Tuple[float, float, str]:
    """Frío sostenido AUMENTA el consumo (el animal busca energía
    extra para termogénesis) — siempre que tenga acceso libre al
    comedero. Si hay barro / acceso difícil, ese efecto se anula
    o se invierte (eso se maneja con otro factor).
    """
    if t_min is None:
        return (0.0, 0.0, "")
    if t_min <= -10:
        return (8.0, 13.0,
                f"Frío severo (T° mín {t_min:.0f}°C): +8-13% por "
                "termogénesis aumentada")
    if t_min <= 0:
        return (4.0, 8.0,
                f"Frío moderado (T° mín {t_min:.0f}°C): +4-8% por "
                "demanda energética extra")
    if t_min <= 5:
        return (2.0, 4.0,
                f"Frío leve (T° mín {t_min:.0f}°C): +2-4% por "
                "mantenimiento térmico")
    if t_min <= 10:
        return (1.0, 2.0,
                f"Fresco (T° mín {t_min:.0f}°C): +1-2%")
    return (0.0, 0.0, "")


def _factor_calor(thi_max: Optional[float],
                    t_max: Optional[float]) -> Tuple[float, float, str]:
    """Calor BAJA el consumo (anorexia térmica). Usamos el THI máximo
    si está disponible, o la T° máxima como fallback."""
    if thi_max is None and t_max is None:
        return (0.0, 0.0, "")
    if thi_max is not None:
        if thi_max >= 88:
            return (-35.0, -25.0,
                    f"Calor extremo (THI {thi_max:.0f}): -25 a -35% "
                    "por anorexia térmica")
        if thi_max >= 84:
            return (-25.0, -15.0,
                    f"Calor severo (THI {thi_max:.0f}): -15 a -25%")
        if thi_max >= 80:
            return (-15.0, -10.0,
                    f"Calor (THI {thi_max:.0f}): -10 a -15%")
        if thi_max >= 72:
            return (-10.0, -5.0,
                    f"Atención calórica (THI {thi_max:.0f}): -5 a -10%")
    # Fallback por T° máxima
    if t_max is not None:
        if t_max >= 35:
            return (-20.0, -10.0,
                    f"Calor severo (T° máx {t_max:.0f}°C): -10 a -20%")
        if t_max >= 30:
            return (-10.0, -5.0,
                    f"Calor moderado (T° máx {t_max:.0f}°C): -5 a -10%")
    return (0.0, 0.0, "")


def _factor_humedad(hr_max: Optional[float],
                      t_min: Optional[float]) -> Tuple[float, float, str]:
    """Humedad alta sostenida + frío baja la palatabilidad (mezcla
    se moja, se hiela, se torna menos apetecible). En calor + HR
    alta, el efecto es por estrés térmico (ya capturado en THI)."""
    if hr_max is None or t_min is None:
        return (0.0, 0.0, "")
    if hr_max >= 90 and t_min < 12:
        return (-4.0, -2.0,
                f"Humedad muy alta (HR {hr_max:.0f}%) + frío: -2 a -4% "
                "por palatabilidad reducida")
    if hr_max >= 85 and t_min < 10:
        return (-3.0, -1.0,
                f"Humedad alta (HR {hr_max:.0f}%): -1 a -3% por mezcla "
                "menos apetecible")
    return (0.0, 0.0, "")


def _factor_barro(barro: bool,
                    lluvia_3d: Optional[float]) -> Tuple[float, float, str]:
    """Barro dificulta el acceso al comedero — el animal duda al ir
    y deja comidas. Reducción importante del DMI real."""
    if barro:
        return (-15.0, -8.0,
                "Barro acceso al comedero: -8 a -15% por dudas y "
                "comidas salteadas")
    if lluvia_3d and lluvia_3d > 10:
        return (-5.0, -2.0,
                f"Lluvia acumulada {lluvia_3d:.0f}mm: -2 a -5% por "
                "piso resbaladizo y acceso")
    return (0.0, 0.0, "")


def _factor_acumulacion(dias_evento: int) -> Tuple[float, float, str]:
    """A partir del día 3 de estrés sostenido, el patrón de consumo
    se altera más allá de los modificadores agudos: el animal sale
    de su rutina, hay caída adicional aunque la T° no haya empeorado.
    """
    if dias_evento >= 5:
        return (-5.0, -2.0,
                f"Acumulación día {dias_evento}: -2 a -5% adicional "
                "por alteración sostenida del patrón")
    if dias_evento >= 3:
        return (-3.0, -1.0,
                f"Acumulación día {dias_evento}: -1 a -3% adicional")
    return (0.0, 0.0, "")


def _factor_pelaje_mojado(lluvia_dia: Optional[float],
                            hr_max: Optional[float],
                            t_min: Optional[float]) -> Tuple[float, float, str]:
    """Pelaje mojado prolongado afecta el bienestar — el animal
    busca refugio en lugar de comer."""
    mojado = False
    if lluvia_dia and lluvia_dia > 5:
        mojado = True
    elif hr_max and hr_max >= 90 and t_min is not None and t_min < 10:
        mojado = True
    if mojado:
        return (-5.0, -2.0,
                "Pelaje mojado prolongado: -2 a -5% por animal "
                "buscando refugio en lugar de comedero")
    return (0.0, 0.0, "")


# =====================================================================
# CÁLCULO PRINCIPAL
# =====================================================================

def _normalizar_categoria(cat: str) -> str:
    c = (cat or "").lower()
    if "ternero" in c or "destete" in c or "guacho" in c:
        return "ternero"
    if "recría" in c or "recria" in c or "destetado" in c:
        return "recria"
    if "novillito" in c:
        return "novillito"
    if "novillo" in c:
        return "novillo"
    if "vaquillona" in c:
        return "vaquillona"
    if "vaca" in c:
        return "vaca"
    if "toro" in c:
        return "toro"
    return "recria"


def dmi_base_kg(peso_kg: float, categoria: str) -> Tuple[float, float]:
    """Calcula el DMI base (kg MS/día) por animal según peso y
    categoría. Devuelve rango (min, max) basado en el % PV de la
    categoría según NASEM 2016 + práctica argentina."""
    if not peso_kg or peso_kg <= 0:
        return (0.0, 0.0)
    cat_norm = _normalizar_categoria(categoria)
    pct_min, pct_max = DMI_BASE_PCT_PV.get(cat_norm, (2.2, 2.6))
    return (
        peso_kg * pct_min / 100.0,
        peso_kg * pct_max / 100.0,
    )


def dmi_proyectado(
    peso_kg: float,
    categoria: str,
    raza: str = "",
    clima_diario: Optional[Dict] = None,
    cantidad: Optional[int] = None,
    dias_evento: int = 1,
    barro: bool = False,
) -> Optional[Dict]:
    """Calcula el DMI proyectado con factores ambientales aplicados.

    Args:
        peso_kg: peso promedio del lote.
        categoria: categoría del lote.
        raza: opcional.
        clima_diario: dict con keys t_min, t_max, hr_max, viento_max,
            lluvia_3d, thi_max. Suele venir del peor día de la semana.
        cantidad: animales del lote (para escalar a total).
        dias_evento: días de estrés acumulados (para factor de
            acumulación).
        barro: si hay barro confirmado.

    Returns:
        dict con DMI base, DMI ajustado (rango), factor aplicado,
        razones y fuente. None si faltan datos esenciales.
    """
    if not peso_kg or peso_kg <= 0:
        return None
    dmi_base_min, dmi_base_max = dmi_base_kg(peso_kg, categoria)
    # Promedio del base como punto central
    dmi_base_med = (dmi_base_min + dmi_base_max) / 2.0

    # Recolectar todos los factores
    clima = clima_diario or {}
    t_min = clima.get("t_min")
    t_max = clima.get("t_max")
    hr_max = clima.get("hr_max")
    thi_max = clima.get("thi_max")
    lluvia_3d = clima.get("lluvia_3d")
    lluvia_dia = clima.get("lluvia_dia")

    factores = [
        _factor_frio(t_min, t_max),
        _factor_calor(thi_max, t_max),
        _factor_humedad(hr_max, t_min),
        _factor_barro(barro, lluvia_3d),
        _factor_acumulacion(dias_evento),
        _factor_pelaje_mojado(lluvia_dia, hr_max, t_min),
    ]

    # ─── Jerarquías ───
    # El frío AUMENTA el consumo (animal busca energía extra) PERO ese
    # efecto positivo solo se manifiesta si el animal puede acceder al
    # comedero y no está dominado por otro estrés contrario. Aplicamos
    # bloqueos/reducciones cuando hay factores dominantes opuestos.
    pelaje_mojado_flag = bool(factores[5][2])  # tiene mensaje
    calor_flag = bool(factores[1][2])

    frio_min, frio_max, frio_txt = factores[0]
    if (frio_min > 0 or frio_max > 0) and frio_txt:
        if barro:
            # Barro severo: el animal no llega al comedero, el efecto
            # positivo se anula completamente.
            factores[0] = (
                0.0, 0.0,
                f"{frio_txt} [ANULADO: barro impide acceso al "
                f"comedero — la energía extra que necesita no la "
                f"puede consumir]",
            )
        elif calor_flag:
            # Calor severo coexistiendo con frío de mínima (raro pero
            # posible en amplitud térmica extrema). El calor domina
            # porque define el consumo total del día.
            factores[0] = (
                0.0, 0.0,
                f"{frio_txt} [ANULADO: el calor diurno domina y "
                f"frena el consumo aunque la mínima nocturna sea "
                f"fría]",
            )
        elif pelaje_mojado_flag:
            # Pelaje mojado: el animal busca refugio en vez de
            # comedero. El efecto del frío se reduce al 50% porque
            # parte de la energía que querría buscar comiendo, la
            # busca quedándose quieto bajo reparo.
            factores[0] = (
                round(frio_min * 0.5, 1),
                round(frio_max * 0.5, 1),
                f"{frio_txt} [REDUCIDO 50%: pelaje mojado, el "
                f"animal busca refugio en vez de comedero]",
            )

    # Sumar deltas (después de aplicar jerarquías)
    delta_min_total = sum(f[0] for f in factores)
    delta_max_total = sum(f[1] for f in factores)
    razones = [f[2] for f in factores if f[2]]

    # Caps razonables: nunca más de -35% ni +15%
    delta_min_total = max(min(delta_min_total, 15.0), -35.0)
    delta_max_total = max(min(delta_max_total, 15.0), -35.0)
    # Asegurar min <= max
    if delta_min_total > delta_max_total:
        delta_min_total, delta_max_total = delta_max_total, delta_min_total

    # Aplicar deltas sobre el DMI base medio para obtener rango
    # ajustado.
    dmi_ajustado_min = dmi_base_med * (1 + delta_min_total / 100.0)
    dmi_ajustado_max = dmi_base_med * (1 + delta_max_total / 100.0)

    # Por seguridad, garantizar que el ajustado quede positivo
    dmi_ajustado_min = max(0.0, dmi_ajustado_min)
    dmi_ajustado_max = max(0.0, dmi_ajustado_max)

    resultado = {
        "dmi_base_rango_kg_dia": (round(dmi_base_min, 2),
                                   round(dmi_base_max, 2)),
        "dmi_base_medio_kg_dia": round(dmi_base_med, 2),
        "dmi_ajustado_rango_kg_dia": (round(dmi_ajustado_min, 2),
                                        round(dmi_ajustado_max, 2)),
        "factor_ajuste_pct": (round(delta_min_total, 1),
                                round(delta_max_total, 1)),
        "razones": razones,
        "cantidad_lote": cantidad,
        "dias_evento": dias_evento,
        "supuestos": (
            f"Categoría '{_normalizar_categoria(categoria)}' "
            f"({peso_kg:.0f} kg). DMI base calculado como % del PV "
            f"({DMI_BASE_PCT_PV[_normalizar_categoria(categoria)][0]}-"
            f"{DMI_BASE_PCT_PV[_normalizar_categoria(categoria)][1]}% "
            f"según NASEM)."
        ),
        "fuente": ("NRC/NASEM 2016 (Cap. Feed Intake) + factores "
                   "ambientales validados para Pampa Húmeda "
                   "(Pordomingo, Latimori, INTA Anguil)"),
    }

    # Si el lote tiene cantidad, calcular DMI total por día
    if cantidad and cantidad > 1:
        resultado["dmi_lote_dia_rango_kg"] = (
            round(dmi_ajustado_min * cantidad, 0),
            round(dmi_ajustado_max * cantidad, 0),
        )
        resultado["dmi_lote_semana_rango_kg"] = (
            round(dmi_ajustado_min * cantidad * 7, 0),
            round(dmi_ajustado_max * cantidad * 7, 0),
        )

    return resultado


def formato_dmi_humano(dmi: Dict) -> str:
    """Devuelve HTML legible del DMI proyectado para mostrar en la UI
    al productor."""
    if not dmi:
        return ""
    b_min, b_max = dmi["dmi_base_rango_kg_dia"]
    a_min, a_max = dmi["dmi_ajustado_rango_kg_dia"]
    f_min, f_max = dmi["factor_ajuste_pct"]
    cantidad = dmi.get("cantidad_lote")
    razones = dmi.get("razones", [])

    # Determinar el signo y color del delta promedio
    delta_prom = (f_min + f_max) / 2.0
    if abs(delta_prom) < 1.0:
        delta_txt = "sin ajuste significativo"
        delta_color = "#666"
    elif delta_prom > 0:
        delta_txt = (
            f"<strong style='color:#1B6F2C;'>"
            f"+{f_min:.0f}% a +{f_max:.0f}%</strong> respecto al base"
        )
        delta_color = "#1B6F2C"
    else:
        delta_txt = (
            f"<strong style='color:#9A4C00;'>"
            f"{f_min:.0f}% a {f_max:.0f}%</strong> respecto al base"
        )
        delta_color = "#9A4C00"

    lineas = []
    lineas.append(
        f"<strong>DMI base por animal:</strong> "
        f"{b_min:.1f}–{b_max:.1f} kg MS/día (sin ajuste por clima)."
    )
    lineas.append(
        f"<strong>DMI ajustado por clima:</strong> "
        f"{a_min:.1f}–{a_max:.1f} kg MS/día por animal — {delta_txt}."
    )
    if cantidad and cantidad > 1:
        l_min, l_max = dmi["dmi_lote_dia_rango_kg"]
        s_min, s_max = dmi["dmi_lote_semana_rango_kg"]
        lineas.append(
            f"<strong>Lote completo ({cantidad:.0f} cab.):</strong> "
            f"{l_min:.0f}–{l_max:.0f} kg MS/día · "
            f"{s_min:.0f}–{s_max:.0f} kg MS/semana."
        )

    bullets = "".join(
        f'<li style="margin-bottom:6px;">{l}</li>' for l in lineas
    )
    razones_html = ""
    if razones:
        razones_html = (
            '<div style="margin-top:8px; font-size:12px; '
            'color:#3a3a3a;">'
            '<strong>Condiciones que ajustan el consumo:</strong>'
            '<ul style="margin:4px 0 0; padding-left:18px;'
            ' line-height:1.5;">'
            + "".join(
                f'<li style="margin-bottom:3px;">{r}</li>'
                for r in razones
            )
            + "</ul></div>"
        )

    fuente = dmi.get(
        "fuente", "NRC/NASEM 2016 + ajustes Pampa Húmeda",
    )
    return (
        f'<ul style="margin:8px 0 4px 0; padding-left:20px;'
        f' line-height:1.55; color:#2a2a2a; font-size:13px;">'
        f"{bullets}</ul>"
        f"{razones_html}"
        f'<div style="font-size:10.5px; color:#666; margin-top:6px;'
        f' font-style:italic;">Fuente: {fuente}.</div>'
    )
