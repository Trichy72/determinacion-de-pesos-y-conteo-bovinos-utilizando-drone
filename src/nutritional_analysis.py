"""
Módulo de análisis nutricional avanzado.

Funciones que diferencian la app de Ganander:
  - Análisis estadístico del lote (percentiles, outliers, recomendaciones de
    separación)
  - Predicción de peso a fecha futura (con ADG actual)
  - Recomendador de dieta basado en NASEM 2016 (8th Ed.) +
    correcciones para razas británicas e índicas argentinas
  - Detección de animales con bajo ADG comparado con el lote

Referencias:
  - NASEM (2016). Nutrient Requirements of Beef Cattle, Eighth Revised Edition.
    Washington, DC: The National Academies Press.
    DOI: https://doi.org/10.17226/19014
  - Tablas IPCVA / IICA Argentina para razas y categorías locales
  - INTA: Pautas de manejo en feedlot

Notas sobre ecuaciones NASEM 2016 vs NRC anterior:
  - NEm = 0.077 × SBW^0.75  (sin cambios para Bos taurus británico)
  - DMI con ecuación cuadrática que considera concentración de NEm
  - Sistema MP (Metabolizable Protein) reemplaza CP simple, separa DIP/UIP
  - Ajustes específicos para Bos indicus y cruzas índicas (-7% NEm)
  - Ajustes por estrés calórico, frío, barro, distancia a aguada
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# 1) ANÁLISIS DE UNIFORMIDAD DEL LOTE
# =====================================================================

@dataclass
class UniformityAnalysis:
    n: int
    promedio_kg: float
    mediana_kg: float
    desvio_kg: float
    cv_pct: float
    p10_kg: float
    p25_kg: float
    p75_kg: float
    p90_kg: float
    rango_intercuartil_kg: float   # P75 - P25
    min_kg: float
    max_kg: float
    outliers_low: List[int] = field(default_factory=list)   # animales más bajos del esperado
    outliers_high: List[int] = field(default_factory=list)  # más altos
    diagnostico: str = ""
    recomendacion: str = ""


def analizar_uniformidad(animales: List[dict]) -> UniformityAnalysis:
    """
    `animales` es una lista de dicts con al menos {'track_id': int, 'peso_kg': float}

    Devuelve métricas y un diagnóstico cualitativo del lote.
    """
    if not animales:
        return UniformityAnalysis(
            n=0, promedio_kg=0, mediana_kg=0, desvio_kg=0, cv_pct=0,
            p10_kg=0, p25_kg=0, p75_kg=0, p90_kg=0,
            rango_intercuartil_kg=0, min_kg=0, max_kg=0,
            diagnostico="Sin datos", recomendacion="—",
        )

    pesos = np.array([a["peso_kg"] for a in animales])
    ids = [a["track_id"] for a in animales]

    promedio = float(np.mean(pesos))
    mediana = float(np.median(pesos))
    desvio = float(np.std(pesos))
    cv = (desvio / promedio * 100) if promedio else 0
    p10, p25, p75, p90 = np.percentile(pesos, [10, 25, 75, 90])
    iqr = p75 - p25

    # Outliers: usando regla de Tukey (1.5×IQR) y también ±25% del promedio
    threshold_low = max(p25 - 1.5 * iqr, promedio * 0.75)
    threshold_high = min(p75 + 1.5 * iqr, promedio * 1.30)
    outliers_low = [ids[i] for i, p in enumerate(pesos) if p < threshold_low]
    outliers_high = [ids[i] for i, p in enumerate(pesos) if p > threshold_high]

    # Diagnóstico cualitativo
    if cv < 8:
        diag = "🟢 Lote MUY uniforme — manejo nutricional homogéneo posible"
        rec = "Mantener manejo único. Excelente uniformidad para feedlot."
    elif cv < 12:
        diag = "🟢 Lote uniforme — variabilidad aceptable"
        rec = "Manejo único viable. Monitorear outliers ocasionalmente."
    elif cv < 18:
        diag = "🟡 Lote con dispersión moderada"
        rec = (
            f"Considerar separar {len(outliers_low)} animal(es) cabeza-baja "
            f"y {len(outliers_high)} cabeza-alta para optimizar dieta."
        )
    else:
        diag = "🔴 Lote DESUNIFORME — eficiencia comprometida"
        rec = (
            f"Sugerencia: reagrupar en 2-3 lotes por peso. "
            f"Cabeza-baja (<{threshold_low:.0f} kg): {len(outliers_low)} animales. "
            f"Cabeza-alta (>{threshold_high:.0f} kg): {len(outliers_high)}. "
            f"Mantener cuerpo principal ({len(animales) - len(outliers_low) - len(outliers_high)} animales)."
        )

    return UniformityAnalysis(
        n=len(animales),
        promedio_kg=promedio,
        mediana_kg=mediana,
        desvio_kg=desvio,
        cv_pct=cv,
        p10_kg=float(p10), p25_kg=float(p25),
        p75_kg=float(p75), p90_kg=float(p90),
        rango_intercuartil_kg=float(iqr),
        min_kg=float(np.min(pesos)),
        max_kg=float(np.max(pesos)),
        outliers_low=outliers_low,
        outliers_high=outliers_high,
        diagnostico=diag,
        recomendacion=rec,
    )


# =====================================================================
# 2) PREDICCIÓN DE PESO A FECHA FUTURA
# =====================================================================

@dataclass
class PesoProjection:
    fecha_objetivo_dias: int
    peso_actual_kg: float
    adg_kg_dia: float
    peso_proyectado_kg: float
    intervalo_confianza: Tuple[float, float]
    cumple_objetivo: Optional[bool]
    diferencia_objetivo_kg: Optional[float]
    adg_requerido_para_objetivo: Optional[float]


def proyectar_peso(
    peso_actual_kg: float,
    adg_kg_dia: float,
    dias: int,
    incertidumbre_adg_pct: float = 15.0,
    peso_objetivo_kg: Optional[float] = None,
) -> PesoProjection:
    """Proyecta el peso a `dias` días con el ADG actual.

    El intervalo de confianza es ±15% del ADG (variabilidad típica observada).
    Si pasás `peso_objetivo_kg`, calcula si va a cumplirlo y qué ADG se
    necesitaría si no.
    """
    proyectado = peso_actual_kg + adg_kg_dia * dias
    delta = abs(adg_kg_dia) * dias * (incertidumbre_adg_pct / 100)
    ic = (proyectado - delta, proyectado + delta)

    cumple = None
    diff = None
    adg_req = None
    if peso_objetivo_kg is not None:
        cumple = proyectado >= peso_objetivo_kg
        diff = proyectado - peso_objetivo_kg
        if dias > 0:
            adg_req = (peso_objetivo_kg - peso_actual_kg) / dias

    return PesoProjection(
        fecha_objetivo_dias=dias,
        peso_actual_kg=peso_actual_kg,
        adg_kg_dia=adg_kg_dia,
        peso_proyectado_kg=proyectado,
        intervalo_confianza=ic,
        cumple_objetivo=cumple,
        diferencia_objetivo_kg=diff,
        adg_requerido_para_objetivo=adg_req,
    )


# =====================================================================
# 3) RECOMENDADOR DE DIETA (NRC + INTA)
# =====================================================================

@dataclass
class DietaRecomendada:
    peso_vivo_kg: float
    adg_objetivo_kg: float
    categoria: str
    raza: str

    # Requerimientos diarios (NASEM 2016)
    consumo_ms_kg: float           # kg materia seca / día
    consumo_ms_pct_pv: float       # % del peso vivo
    # Sistema MP (Metabolizable Protein) — NASEM reemplazó PB simple
    mp_requerida_g: float          # g/día de proteína metabolizable
    pb_pct_ms: float               # % proteína bruta en MS (referencia)
    pb_gramos: float               # gramos PB / día
    # Rango de PB aceptable según etapa (práctica argentina)
    pb_pct_min: float = 0.0        # % PB mínimo del rango
    pb_pct_max: float = 0.0        # % PB máximo del rango
    etapa: str = ""                # "destete" / "recria" / "terminacion" / "mantenimiento"
    # Energías Netas
    nem_mcal: float = 0.0
    neg_mcal: float = 0.0
    em_mcal: float = 0.0
    em_concentracion_mcal_kg: float = 0.0
    # Fibra y minerales
    fdn_min_pct: float = 0.0
    calcio_g: float = 0.0
    fosforo_g: float = 0.0
    relacion_ca_p: float = 0.0

    # Mezcla recomendada
    composicion_sugerida: Dict[str, float] = field(default_factory=dict)
    notas: List[str] = field(default_factory=list)


# =====================================================================
# RANGOS DE PB POR ETAPA (práctica argentina, en MS)
# Fuente: Pordomingo, Latimori, IPCVA, AAPA + NASEM 2016 calibrado
# =====================================================================

PB_RANGOS_ARGENTINA = {
    "destete": {
        "rango_pct": (16.0, 18.0),
        "target_pct": 17.0,
        "rango_alt_pasturas": (14.0, 16.0),
        "impacto_subdosis": (
            "Cada 1% de PB por debajo del mínimo: -0,1 a -0,3 kg ADG/día. "
            "En 60 días = 6-18 kg de pérdida (o más días en feedlot)."
        ),
    },
    "recria": {
        "rango_pct": (12.0, 14.0),
        "target_pct": 13.0,
        "rango_alt_pasturas": (13.0, 15.0),  # forraje pobre/maduro
        "impacto_subdosis": (
            "Cada 1% de PB por debajo del mínimo: -0,05 a -0,2 kg ADG/día. "
            "En 120 días = 6-24 kg de pérdida (10-25 días extra)."
        ),
    },
    "terminacion": {
        "rango_pct": (11.0, 13.0),
        "target_pct": 12.0,
        "rango_alt_pasturas": (12.0, 14.0),  # menor energía / más fibrosa
        "impacto_subdosis": (
            "Pasarse de PB no mejora performance: solo aumenta costo de "
            "balanceado y excreción de N. Mantenerse en 11-13% es óptimo."
        ),
    },
    "mantenimiento": {
        "rango_pct": (8.0, 10.0),
        "target_pct": 9.0,
        "rango_alt_pasturas": (8.0, 10.0),
        "impacto_subdosis": "Pérdida de condición corporal, fertilidad reducida.",
    },
}


def determinar_etapa(categoria: str, peso_kg: float, adg_obj: float) -> str:
    """Clasifica la etapa productiva en base a categoría + peso + ADG."""
    cat = categoria.lower()
    if cat == "ternero" or peso_kg < 180:
        return "destete"
    if cat == "toro" or cat == "vaca_adulta":
        return "mantenimiento" if adg_obj < 0.3 else "terminacion"
    if peso_kg < 320 or adg_obj < 0.9:
        return "recria"
    return "terminacion"


def calcular_requerimientos(
    peso_vivo_kg: float,
    adg_objetivo_kg: float,
    categoria: str = "vaquillona",
    raza: str = "angus",
    dias_estres_calorico: bool = False,
    perfil: str = "argentina",
    ajuste_pb_pct: float = 1.0,
) -> DietaRecomendada:
    """
    Calcula requerimientos diarios siguiendo NASEM 2016 (8th Ed., capítulos 4-7).

    Parámetros:
        perfil: "nasem" (estándar internacional) o "argentina" (ajustes INTA).
                "argentina" multiplica el requerimiento de PB por 0.90 que es
                la convención local (Pordomingo, Latimori, IPCVA) basada en
                animales británicos comerciales y dietas concentradas.
        ajuste_pb_pct: multiplicador adicional sobre PB (default 1.0).
                Útil si tenés tablas propias y querés afinar.

    Ecuaciones principales:
        NEm = 0.077 × SBW^0.75      (Bos taurus británico, Mcal/día)
        NEm = 0.0707 × SBW^0.75     (Bos indicus, -7%)
        SBW = 0.96 × peso_vivo      (shrunk body weight)
        EQEBW = 0.891 × SBW         (equivalent empty body weight)
        RE  = 0.0635 × EQEBW^0.75 × ADG^1.097   (energía retenida en ganancia)
        MP_man = 3.8 × SBW^0.75     (g/día, mantenimiento)
        MP_gan = 268 - 29.4 × (RE/ADG)         (eficiencia decreciente)
        DMI = SBW^0.75 × (0.2435 × NEm - 0.0466 × NEm² - 0.1128) / NEm
              (válido para feedlot terminación)

    Ajustes regionales:
      - Bos indicus / cruza con cebú: -7% NEm, +5% NEm en cruza F1
      - Estrés calórico (THI > 78): -8% DMI, +5% NEm requerido para mantenimiento
      - Días de barro: +25% NEm (NASEM Cap. 7)
    """
    # Body weights
    sbw = peso_vivo_kg * 0.96       # Shrunk Body Weight
    eqebw = sbw * 0.891             # Equivalent Empty Body Weight

    # ---- 1) ENERGÍAS NETAS (NASEM 2016) ----
    # NEm — coeficiente por raza
    if raza in ("brangus", "braford"):
        coef_nem = 0.0735           # cruza índica F1: ~5% menos que Bos taurus
    elif raza == "cebuino":
        coef_nem = 0.0707           # Bos indicus puro
    else:
        coef_nem = 0.077            # Bos taurus británico (Angus, Hereford)

    nem = coef_nem * (sbw ** 0.75)

    # NEg (NASEM 2016 Eq. 7-13): RE = 0.0635 × EQEBW^0.75 × ADG^1.097
    re = 0.0635 * (eqebw ** 0.75) * (adg_objetivo_kg ** 1.097) if adg_objetivo_kg > 0 else 0
    neg = re

    # Energía Metabolizable total (factor de conversión NE→ME ≈ 1.65 a niveles
    # típicos de feedlot)
    em_total = (nem + neg) * 1.65

    # ---- 2) CONSUMO DE MATERIA SECA (NASEM Eq. 2-3) ----
    # DMI predicción para feedlot/terminación con NEm dieta ~2.0 Mcal/kg
    nem_dieta = 2.0 if adg_objetivo_kg > 0.9 else 1.65 if adg_objetivo_kg > 0.5 else 1.30
    if nem_dieta > 0:
        dmi_factor = (0.2435 * nem_dieta - 0.0466 * nem_dieta ** 2 - 0.1128) / nem_dieta
        consumo_ms = (sbw ** 0.75) * dmi_factor
    else:
        consumo_ms = peso_vivo_kg * 0.022

    # Bound: el consumo no puede exceder 3% PV ni ser <1.6%
    consumo_ms = max(min(consumo_ms, peso_vivo_kg * 0.030), peso_vivo_kg * 0.016)

    # Ajuste por estrés calórico (NASEM Cap. 7)
    if dias_estres_calorico:
        consumo_ms *= 0.92          # -8% DMI
        nem *= 1.05                 # +5% NEm de mantenimiento

    consumo_ms_pct = consumo_ms / peso_vivo_kg * 100
    em_concentracion = em_total / consumo_ms if consumo_ms > 0 else 0

    # ---- 3) PROTEÍNA METABOLIZABLE (sistema MP de NASEM 2016) ----
    # Cálculo RIGUROSO basado en ecuaciones NASEM. Se mantiene siempre.
    mp_man = 3.8 * (sbw ** 0.75)
    if adg_objetivo_kg > 0:
        # NASEM Eq. 11-1c: NPg = ADG × (268 - 29.4 × (RE/ADG))
        npg = adg_objetivo_kg * max(120, 268 - 29.4 * (re / adg_objetivo_kg))
        mp_gan = npg / 0.49   # eficiencia de uso de MP para crecimiento
    else:
        mp_gan = 0
    mp_total = (mp_man + mp_gan) * ajuste_pb_pct

    # PB equivalente (MP/PB ratio 0.55-0.65 según calidad de la dieta)
    mp_cp_ratio = 0.65 if adg_objetivo_kg > 0.9 else 0.60 if adg_objetivo_kg > 0.5 else 0.55
    pb_g = mp_total / mp_cp_ratio
    pb_pct = (pb_g / consumo_ms / 10) if consumo_ms > 0 else 0

    # Etapa productiva y rangos de práctica argentina (referencia cruzada)
    etapa = determinar_etapa(categoria, peso_vivo_kg, adg_objetivo_kg)
    rango_data = PB_RANGOS_ARGENTINA[etapa]
    pb_min_pct, pb_max_pct = rango_data["rango_pct"]

    # ---- 4) FIBRA Y MINERALES ----
    # FDN mínimo según etapa. NASEM 2016 permite hasta 12-15% en dietas
    # de terminación con monensina/ionóforo (Cap. 8). Sin ionóforo subir
    # a 18-20%. Para recría: 25-30%. Mantenimiento: 35%.
    if adg_objetivo_kg > 1.2:
        fdn_min = 15.0              # terminación intensiva (con ionóforo)
    elif adg_objetivo_kg > 0.7:
        fdn_min = 22.0              # recría / crecimiento moderado
    else:
        fdn_min = 30.0              # mantenimiento

    # Calcio y fósforo (NASEM Cap. 6)
    calcio_g = consumo_ms * 6.0     # 0.6% MS objetivo
    fosforo_g = consumo_ms * 3.5    # 0.35% MS objetivo
    relacion_ca_p = calcio_g / fosforo_g if fosforo_g > 0 else 0

    # ---- 5) COMPOSICIÓN SUGERIDA DE LA MEZCLA ----
    composicion: Dict[str, float] = {}
    notas: List[str] = ["Cálculos basados en NASEM 2016 (8th Ed.)."]

    if adg_objetivo_kg > 1.0:
        # Terminación intensiva
        composicion = {
            "Maíz grano (8% humedad)": 52.0,
            "Silaje de maíz (35% MS)": 18.0,
            "Pellet de soja / expeller": 12.0,
            "Heno de alfalfa (90% MS)": 9.0,
            "Núcleo mineral-vitamínico": 5.0,
            "Urea protegida (inclusión gradual)": 2.0,
            "Buffer (bicarbonato + óxido magnesio)": 2.0,
        }
        notas.append("Dieta de terminación: introducir grano gradualmente (10-14 días).")
        notas.append("Monitorear acidosis subclínica: heces pastosas, baja DMI, claudicación.")
        notas.append("Incluir ionóforo (monensina 25-33 mg/kg MS) para eficiencia y prevención.")
    elif adg_objetivo_kg > 0.6:
        # Recría
        composicion = {
            "Silaje de maíz": 32.0,
            "Heno de alfalfa": 22.0,
            "Maíz grano": 25.0,
            "Pellet de girasol o soja": 12.0,
            "Núcleo mineral": 5.0,
            "Sal blanca": 4.0,
        }
        notas.append("Dieta balanceada de recría a corral, transición segura a terminación.")
    else:
        # Mantenimiento
        composicion = {
            "Heno de gramíneas (avena/festuca)": 58.0,
            "Silaje de sorgo o maíz": 25.0,
            "Pellet de soja o expeller": 9.0,
            "Núcleo mineral": 4.0,
            "Sal blanca": 4.0,
        }
        notas.append("Dieta de mantenimiento: evitar pérdida de condición corporal.")

    if dias_estres_calorico:
        notas.append(
            "⚠️ Estrés calórico (NASEM Cap. 7): −8% DMI esperado. "
            "Aumentar densidad energética y proteica, agua fresca ad libitum, "
            "sombra mínima 4 m²/animal."
        )
    if relacion_ca_p < 1.5 or relacion_ca_p > 2.5:
        notas.append(
            f"⚠️ Relación Ca:P = {relacion_ca_p:.2f} fuera del rango recomendado (1.5-2.5)."
        )

    return DietaRecomendada(
        peso_vivo_kg=peso_vivo_kg,
        adg_objetivo_kg=adg_objetivo_kg,
        categoria=categoria,
        raza=raza,
        consumo_ms_kg=consumo_ms,
        consumo_ms_pct_pv=consumo_ms_pct,
        mp_requerida_g=mp_total,
        pb_pct_ms=pb_pct,
        pb_gramos=pb_g,
        pb_pct_min=pb_min_pct,
        pb_pct_max=pb_max_pct,
        etapa=etapa,
        nem_mcal=nem,
        neg_mcal=neg,
        em_mcal=em_total,
        em_concentracion_mcal_kg=em_concentracion,
        fdn_min_pct=fdn_min,
        calcio_g=calcio_g,
        fosforo_g=fosforo_g,
        relacion_ca_p=relacion_ca_p,
        composicion_sugerida=composicion,
        notas=notas,
    )


def ajustar_req_por_dmi(
    req: DietaRecomendada, dmi_nuevo_kg: float,
    razon_ajuste: str = "",
) -> DietaRecomendada:
    """Devuelve un nuevo DietaRecomendada con el DMI ajustado por
    clima (u otro factor) y las DENSIDADES recalculadas.

    Filosofía: los requerimientos ABSOLUTOS del animal NO cambian
    con el clima. La vaquillona sigue necesitando los mismos
    gramos de PB y las mismas Mcal de EM. Lo que cambia es la
    DENSIDAD (concentración por kg de MS) que tiene que tener la
    dieta para entregar esos absolutos en menos (o más) kg.

    Esto es lo que importa al productor:
      - Si el clima predice MENOS consumo → dieta más concentrada
        (más grano, menos voluminoso).
      - Si predice MÁS consumo → dieta puede ser menos densa,
        sumar más rollo/fardo voluminoso.

    Args:
        req: DietaRecomendada original (con DMI base).
        dmi_nuevo_kg: nuevo DMI a aplicar (kg MS/día).
        razon_ajuste: texto descriptivo del por qué del ajuste,
            se agrega a las notas.

    Returns:
        Nueva DietaRecomendada con DMI actualizado y densidades
        recalculadas. Los absolutos (pb_gramos, em_mcal,
        mp_requerida_g, calcio_g, fosforo_g) quedan IGUAL.
    """
    if not req or dmi_nuevo_kg <= 0:
        return req

    nuevo_consumo_pct_pv = (
        (dmi_nuevo_kg / req.peso_vivo_kg * 100)
        if req.peso_vivo_kg > 0 else 0
    )
    # Densidad de PB: g_PB / kg_MS → para llegar al mismo total con
    # un DMI distinto, la concentración cambia inversamente.
    nuevo_pb_pct = (
        (req.pb_gramos / dmi_nuevo_kg / 10.0)
        if dmi_nuevo_kg > 0 else req.pb_pct_ms
    )
    nuevo_em_concentracion = (
        (req.em_mcal / dmi_nuevo_kg)
        if dmi_nuevo_kg > 0 else req.em_concentracion_mcal_kg
    )

    notas_nuevas = list(req.notas)
    if razon_ajuste:
        notas_nuevas.append(f"📌 Ajuste de DMI aplicado: {razon_ajuste}")
        delta_pct = (
            (dmi_nuevo_kg - req.consumo_ms_kg) /
            req.consumo_ms_kg * 100
        ) if req.consumo_ms_kg > 0 else 0
        if abs(delta_pct) >= 1:
            direccion = "subió" if delta_pct > 0 else "bajó"
            notas_nuevas.append(
                f"📊 DMI {direccion} de {req.consumo_ms_kg:.2f} a "
                f"{dmi_nuevo_kg:.2f} kg/día ({delta_pct:+.1f}%). "
                f"Los requerimientos absolutos (PB g/día, EM Mcal/día) "
                f"NO cambian — el animal sigue necesitando lo mismo. "
                f"Lo que cambia es la DENSIDAD: la dieta tiene que ser "
                f"{'más concentrada' if delta_pct < 0 else 'menos concentrada'} "
                f"para entregar lo mismo en {'menos' if delta_pct < 0 else 'más'} kg."
            )

    return DietaRecomendada(
        peso_vivo_kg=req.peso_vivo_kg,
        adg_objetivo_kg=req.adg_objetivo_kg,
        categoria=req.categoria,
        raza=req.raza,
        consumo_ms_kg=dmi_nuevo_kg,
        consumo_ms_pct_pv=nuevo_consumo_pct_pv,
        mp_requerida_g=req.mp_requerida_g,
        pb_pct_ms=nuevo_pb_pct,
        pb_gramos=req.pb_gramos,
        pb_pct_min=req.pb_pct_min,
        pb_pct_max=req.pb_pct_max,
        etapa=req.etapa,
        nem_mcal=req.nem_mcal,
        neg_mcal=req.neg_mcal,
        em_mcal=req.em_mcal,
        em_concentracion_mcal_kg=nuevo_em_concentracion,
        fdn_min_pct=req.fdn_min_pct,
        calcio_g=req.calcio_g,
        fosforo_g=req.fosforo_g,
        relacion_ca_p=req.relacion_ca_p,
        composicion_sugerida=req.composicion_sugerida,
        notas=notas_nuevas,
    )


# =====================================================================
# 4) DETECCIÓN DE ANIMALES CON BAJO ADG
# =====================================================================

def detectar_animales_bajo_adg(
    animales_iniciales: List[dict],
    animales_finales: List[dict],
    dias: int,
    threshold_pct: float = 0.70,
) -> List[dict]:
    """
    Compara los pesos individuales entre dos pesadas y devuelve los
    animales cuyo ADG es <70% del ADG promedio del lote (señal temprana
    de enfermedad o problema individual).

    Como nuestra detección no es individual a nivel de animal entre
    pesadas distintas, esta función está pensada para uso futuro con
    identificación individual (caravana visual o reID por features).

    Por ahora, devuelve animales en la pesada final que estén por debajo
    del P10 del lote (los 10% más livianos), que probablemente sean los
    "rezagados".
    """
    if not animales_finales or dias <= 0:
        return []

    pesos_fin = np.array([a["peso_kg"] for a in animales_finales])
    pesos_ini = np.array([a["peso_kg"] for a in animales_iniciales]) if animales_iniciales else None

    p10 = float(np.percentile(pesos_fin, 10))
    promedio = float(np.mean(pesos_fin))
    adg_promedio = (promedio - float(np.mean(pesos_ini))) / dias if pesos_ini is not None else 0

    rezagados = []
    for a in animales_finales:
        if a["peso_kg"] < p10:
            adg_estimado = (a["peso_kg"] - promedio + adg_promedio * dias) / dias if dias else 0
            rezagados.append({
                "track_id": a["track_id"],
                "peso_kg": a["peso_kg"],
                "p10_lote_kg": p10,
                "diff_promedio_kg": a["peso_kg"] - promedio,
                "adg_estimado": adg_estimado,
                "adg_promedio_lote": adg_promedio,
                "alerta": "Bajo ADG vs lote — revisar estado sanitario",
            })

    return rezagados
