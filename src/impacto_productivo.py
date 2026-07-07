"""Estimador de impacto productivo de eventos climáticos.

Calcula el incremento de requerimiento energético y la pérdida potencial
de ganancia de peso (ADPV) durante un evento de frío, basado en fórmulas
NRC/NASEM 2016 + ajustes para Pampa Húmeda argentina (Pordomingo,
Latimori, Pezzola).

Filosofía: dar al productor un valor concreto del costo de NO actuar,
expresado en kg de carne perdidos. Rango honesto, no número falsamente
preciso. Defaults razonables por categoría si no hay datos específicos
del lote.

Uso:
    from src.impacto_productivo import estimar_impacto_frio

    impacto = estimar_impacto_frio(
        peso_kg=280,
        categoria="ternero",
        raza="angus",
        t_min_c=2,
        viento_kmh=18,
        humedad_pct=78,
        barro=True,
        pelaje_mojado=False,
        dias_evento=3,
        cantidad=100,
    )
    # impacto = {
    #     "gasto_extra_pct": (12, 18),
    #     "adpv_perdida_kg_rango": (0.18, 0.32),
    #     "pct_adpv_perdida": (15, 27),
    #     "kg_perdidos_lote_total": (54, 96),
    #     "supuestos": "Defaults NRC para ternero 280 kg, ...",
    # }
"""
from __future__ import annotations

from typing import Optional, Dict, Tuple


# =====================================================================
# DEFAULTS POR CATEGORÍA (basados en práctica argentina + NRC/NASEM)
# Si el productor carga datos específicos del lote, los reemplazan.
# =====================================================================

DEFAULTS_CATEGORIA = {
    "ternero": {
        "adpv_objetivo_kg": 0.8,
        "energia_dieta_mcal_em_kg_ms": 2.6,
        "consumo_ms_pct_pv": 2.8,  # % del peso vivo
        "lct_seco_c": 10,
        "lct_mojado_c": 17,
        "neg_mcal_kg_ganancia": 3.8,
    },
    "recria": {
        "adpv_objetivo_kg": 1.0,
        "energia_dieta_mcal_em_kg_ms": 2.7,
        "consumo_ms_pct_pv": 2.5,
        "lct_seco_c": 5,
        "lct_mojado_c": 12,
        "neg_mcal_kg_ganancia": 4.5,
    },
    "novillito": {
        "adpv_objetivo_kg": 1.1,
        "energia_dieta_mcal_em_kg_ms": 2.7,
        "consumo_ms_pct_pv": 2.4,
        "lct_seco_c": 0,
        "lct_mojado_c": 10,
        "neg_mcal_kg_ganancia": 5.0,
    },
    "novillo": {
        "adpv_objetivo_kg": 1.2,
        "energia_dieta_mcal_em_kg_ms": 2.9,
        "consumo_ms_pct_pv": 2.2,
        "lct_seco_c": -5,
        "lct_mojado_c": 5,
        "neg_mcal_kg_ganancia": 5.5,
    },
    "vaquillona": {
        "adpv_objetivo_kg": 0.9,
        "energia_dieta_mcal_em_kg_ms": 2.6,
        "consumo_ms_pct_pv": 2.4,
        "lct_seco_c": 0,
        "lct_mojado_c": 10,
        "neg_mcal_kg_ganancia": 4.5,
    },
    "vaca": {
        "adpv_objetivo_kg": 0.4,  # vaca de cría, no ganancia activa
        "energia_dieta_mcal_em_kg_ms": 2.4,
        "consumo_ms_pct_pv": 2.0,
        "lct_seco_c": -5,
        "lct_mojado_c": 5,
        "neg_mcal_kg_ganancia": 6.0,
    },
    "toro": {
        "adpv_objetivo_kg": 0.5,
        "energia_dieta_mcal_em_kg_ms": 2.5,
        "consumo_ms_pct_pv": 2.0,
        "lct_seco_c": -5,
        "lct_mojado_c": 5,
        "neg_mcal_kg_ganancia": 6.0,
    },
}


def _normalizar_categoria(categoria: str) -> str:
    """Mapea la categoría real del lote a una de las 7 categorías base."""
    c = (categoria or "").lower()
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
    # Default: tratar como recría (intermedio)
    return "recria"


def _ajuste_raza(raza: str) -> float:
    """Modificador racial sobre tolerancia al frío.

    Razas británicas adaptadas (Angus, Hereford): bajan la LCT ~3°C
    (toleran mejor frío seco).
    Cebuinas/índicas: suben la LCT ~5°C (peor tolerancia al frío).
    Cruzas: sin ajuste.
    """
    r = (raza or "").lower()
    if any(k in r for k in ("angus", "hereford", "británic", "britanic")):
        return -3.0
    if any(k in r for k in ("brangus", "braford", "nelore", "cebu", "índic", "indic")):
        return 5.0
    return 0.0


def _sensacion_termica(
    t_c: float, viento_kmh: float, humedad_pct: float
) -> float:
    """Calcula la temperatura efectiva ajustada por viento (windchill)
    y humedad. Aproximación simple — para sentir el ajuste, no para
    estación meteorológica.

    Windchill (fórmula Steadman simplificada para ganadería):
      Tef = T - (viento/10) × 1.5 si viento > 5 km/h
    Humedad: si HR > 80% y T < 15°C, baja la temperatura efectiva 1-2°C
      adicionales (pelaje moja).
    """
    tef = t_c
    if viento_kmh and viento_kmh > 5:
        tef -= (viento_kmh / 10.0) * 1.5
    if humedad_pct and humedad_pct >= 80 and t_c < 15:
        tef -= 1.5
    if humedad_pct and humedad_pct >= 90 and t_c < 15:
        tef -= 0.5  # 2°C total
    return tef


def estimar_impacto_frio(
    peso_kg: float,
    categoria: str,
    raza: str = "",
    t_min_c: Optional[float] = None,
    viento_kmh: Optional[float] = None,
    humedad_pct: Optional[float] = None,
    barro: bool = False,
    pelaje_mojado: bool = False,
    dias_evento: int = 1,
    cantidad: Optional[int] = None,
    # Datos específicos del lote (override de defaults)
    adpv_objetivo_kg: Optional[float] = None,
    energia_dieta_mcal_em_kg_ms: Optional[float] = None,
) -> Optional[Dict]:
    """Estima impacto productivo de un evento de frío sobre un lote.

    Devuelve dict con rangos honestos o None si los datos son
    insuficientes (no inventa).

    Returns dict:
      {
        "gasto_extra_pct": (min%, max%),   # incremento de mantenimiento
        "adpv_perdida_kg_rango": (min, max),  # kg/día perdidos por animal
        "pct_adpv_perdida": (min%, max%),
        "kg_perdidos_lote_periodo": (min, max),  # total del lote en N días
        "supuestos": str,
        "fuente": "NRC 2016 + ajustes prácticos Pampa Húmeda",
      }
    """
    # Validaciones mínimas — si faltan datos clave, no inventamos
    if peso_kg is None or peso_kg <= 0:
        return None
    if t_min_c is None:
        return None
    if dias_evento <= 0:
        return None

    # Resolver defaults por categoría
    cat_norm = _normalizar_categoria(categoria)
    defaults = DEFAULTS_CATEGORIA[cat_norm]
    adpv_obj = adpv_objetivo_kg or defaults["adpv_objetivo_kg"]
    energia_dieta = (energia_dieta_mcal_em_kg_ms
                     or defaults["energia_dieta_mcal_em_kg_ms"])
    consumo_ms = peso_kg * (defaults["consumo_ms_pct_pv"] / 100.0)
    neg_kg_ganancia = defaults["neg_mcal_kg_ganancia"]

    # Determinar LCT efectiva
    pelaje_real_mojado = pelaje_mojado or (
        humedad_pct is not None and humedad_pct >= 85
        and (t_min_c < 12)
    )
    lct_base = (defaults["lct_mojado_c"] if pelaje_real_mojado
                else defaults["lct_seco_c"])
    lct_efectiva = lct_base + _ajuste_raza(raza)

    # Temperatura efectiva (windchill + humedad)
    t_efectiva = _sensacion_termica(
        t_min_c, viento_kmh or 0, humedad_pct or 0,
    )

    # Si la T° efectiva está por encima de la LCT, NO hay estrés calculable
    if t_efectiva >= lct_efectiva:
        return None

    delta_t = lct_efectiva - t_efectiva  # cuántos grados debajo de LCT

    # Incremento de requerimiento de mantenimiento (NRC: ~1% por grado
    # bajo LCT en condiciones secas, hasta 2% por grado con pelaje
    # mojado o barro severo)
    factor_pct_por_grado_min = 0.8
    factor_pct_por_grado_max = 1.2
    if pelaje_real_mojado:
        factor_pct_por_grado_min = 1.2
        factor_pct_por_grado_max = 1.8
    if barro:
        factor_pct_por_grado_max += 0.3

    gasto_extra_pct_min = delta_t * factor_pct_por_grado_min
    gasto_extra_pct_max = delta_t * factor_pct_por_grado_max

    # Cap razonable: nunca más de 40% (literatura)
    gasto_extra_pct_min = min(gasto_extra_pct_min, 30)
    gasto_extra_pct_max = min(gasto_extra_pct_max, 40)

    # Convertir el gasto extra a Mcal/día
    # NEm para mantenimiento ≈ 0.077 × peso^0.75 Mcal NEm/día (NASEM)
    nem_mantenimiento = 0.077 * (peso_kg ** 0.75)
    gasto_extra_mcal_min = nem_mantenimiento * (gasto_extra_pct_min / 100.0)
    gasto_extra_mcal_max = nem_mantenimiento * (gasto_extra_pct_max / 100.0)

    # Si el animal NO aumenta consumo para compensar (caso típico
    # cuando hay barro, comedero alterado o patrón de consumo
    # desplazado), esa energía sale del fondo de ganancia.
    # ADPV perdida = gasto extra / NEg por kg de ganancia
    adpv_perdida_min = gasto_extra_mcal_min / neg_kg_ganancia
    adpv_perdida_max = gasto_extra_mcal_max / neg_kg_ganancia

    # Cap: la pérdida no puede exceder el ADPV objetivo (el animal
    # como máximo deja de ganar peso, no pierde más allá del objetivo
    # sin movilizar reservas, lo que sería un capítulo aparte).
    adpv_perdida_min = min(adpv_perdida_min, adpv_obj)
    adpv_perdida_max = min(adpv_perdida_max, adpv_obj)

    pct_adpv_min = (adpv_perdida_min / adpv_obj) * 100 if adpv_obj > 0 else 0
    pct_adpv_max = (adpv_perdida_max / adpv_obj) * 100 if adpv_obj > 0 else 0

    # Total del lote en el período del evento
    kg_lote_min = adpv_perdida_min * dias_evento * (cantidad or 1)
    kg_lote_max = adpv_perdida_max * dias_evento * (cantidad or 1)

    supuestos = (
        f"Categoría '{cat_norm}' ({peso_kg:.0f} kg), "
        f"{'pelaje mojado/HR alta' if pelaje_real_mojado else 'piso seco'}, "
        f"LCT efectiva {lct_efectiva:.0f}°C, "
        f"T° efectiva {t_efectiva:.0f}°C (T° {t_min_c:.0f}°C "
        f"+ ajuste viento/humedad). "
        f"Defaults usados: ADPV objetivo {adpv_obj} kg/día, "
        f"dieta {energia_dieta} Mcal EM/kg MS."
    )

    return {
        "gasto_extra_pct": (round(gasto_extra_pct_min, 0),
                            round(gasto_extra_pct_max, 0)),
        "adpv_perdida_kg_rango": (round(adpv_perdida_min, 2),
                                  round(adpv_perdida_max, 2)),
        "pct_adpv_perdida": (round(pct_adpv_min, 0),
                              round(pct_adpv_max, 0)),
        "kg_perdidos_lote_periodo": (round(kg_lote_min, 0),
                                      round(kg_lote_max, 0)),
        "dias_evento": dias_evento,
        "cantidad_lote": cantidad,
        "supuestos": supuestos,
        "fuente": ("NRC/NASEM 2016 (Cap. Environment) + "
                   "ajustes Pampa Húmeda (Pordomingo, Latimori)"),
    }


def formato_impacto_humano(impacto: Dict) -> str:
    """Versión LIMPIA del impacto para mostrar AL PRODUCTOR en la UI.

    Diferencia con formato_impacto_texto():
      - Esta NO incluye las instrucciones internas para el LLM
        (NO aplicar rendimiento de carcasa, citar fuentes, etc.).
      - Devuelve HTML listo para Streamlit con bullets y tipografía
        clara.
      - Apunta a que el productor lea y entienda en 5 segundos.

    Para inyectar al LLM seguí usando formato_impacto_texto().
    """
    if not impacto:
        return ""
    g_min, g_max = impacto["gasto_extra_pct"]
    a_min, a_max = impacto["adpv_perdida_kg_rango"]
    p_min, p_max = impacto["pct_adpv_perdida"]
    cantidad = impacto.get("cantidad_lote") or 1
    dias = impacto.get("dias_evento", 1)
    lote_dia_min = a_min * cantidad
    lote_dia_max = a_max * cantidad
    k_min, k_max = impacto.get("kg_perdidos_lote_periodo", (0, 0))

    lineas = []
    lineas.append(
        f"<strong>Requerimiento de mantenimiento elevado:</strong> "
        f"+{g_min:.0f}-{g_max:.0f}% durante el evento "
        f"(energía solo para sostener temperatura corporal)."
    )
    lineas.append(
        f"<strong>Pérdida por animal por día:</strong> "
        f"{a_min:.2f}-{a_max:.2f} kg/día de peso vivo "
        f"({p_min:.0f}-{p_max:.0f}% del ADPV objetivo)."
    )
    if cantidad > 1:
        _dia_label = "día" if dias == 1 else "días"
        lineas.append(
            f"<strong>Pérdida sobre todo el lote por día "
            f"({cantidad:.0f} cab.):</strong> "
            f"{lote_dia_min:.1f}-{lote_dia_max:.1f} kg/día de peso "
            f"vivo que el lote no suma."
        )
        lineas.append(
            f"<strong>Pérdida total acumulada en el evento "
            f"({cantidad:.0f} cab. × {dias} {_dia_label}):</strong> "
            f"{k_min:.0f}-{k_max:.0f} kg de peso vivo sobre el lote."
        )
    fuente = impacto.get(
        "fuente",
        "NRC/NASEM 2016 + ajustes Pampa Húmeda",
    )
    # Devolvemos lista de bullets HTML
    bullets = "".join(
        f'<li style="margin-bottom:6px;">{l}</li>' for l in lineas
    )
    return (
        f'<ul style="margin:8px 0 4px 0; padding-left:20px;'
        f' line-height:1.55; color:#2a2a2a; font-size:13px;">'
        f"{bullets}</ul>"
        f'<div style="font-size:10.5px; color:#666; margin-top:4px;'
        f' font-style:italic;">Fuente: {fuente}. Cifras en peso vivo '
        f"(lo que se cobra al frigorífico en balanza).</div>"
    )


def formato_impacto_humano_desde_registro(registro: Dict) -> str:
    """Reconstruye el HTML legible del impacto desde un registro
    guardado en la tabla impactos_lote. Útil para mostrar al productor
    el detalle de un impacto pasado consultando el histórico.

    Recibe una fila como la que devuelve db.listar_impactos_lote().
    """
    if not registro:
        return ""
    # Mapear el registro al formato dict que espera formato_impacto_humano
    impacto = {
        "gasto_extra_pct": (
            registro.get("gasto_extra_pct_min", 0) or 0,
            registro.get("gasto_extra_pct_max", 0) or 0,
        ),
        "adpv_perdida_kg_rango": (
            registro.get("adpv_perdida_min_kg", 0) or 0,
            registro.get("adpv_perdida_max_kg", 0) or 0,
        ),
        "pct_adpv_perdida": (
            registro.get("pct_adpv_min", 0) or 0,
            registro.get("pct_adpv_max", 0) or 0,
        ),
        "kg_perdidos_lote_periodo": (
            registro.get("kg_lote_total_min", 0) or 0,
            registro.get("kg_lote_total_max", 0) or 0,
        ),
        "cantidad_lote": registro.get("cantidad_animales") or 1,
        "dias_evento": registro.get("dias_evento") or 1,
        "fuente": "NRC/NASEM 2016 + ajustes Pampa Húmeda",
    }
    return formato_impacto_humano(impacto)


def auditar_texto_llm(texto: str, impacto: Dict) -> str:
    """Audita un texto generado por el LLM y CORRIGE rangos de kg del
    lote total si están mal calculados.

    El LLM tiene un sesgo persistente: cuando ve un total del lote
    cerca de "kg/día por animal", a veces multiplica solo por cabezas
    (sin multiplicar por días) y reporta el resultado como
    "acumulado en el evento". Este auditor detecta esos errores y los
    corrige.

    Estrategia robusta:
      - Recorrer todas las apariciones de "X-Y kg" en el texto.
      - Si una aparición está CERCA (≤90 chars) de palabras clave del
        lote/evento (lote, acumulad, evento, balanza, cabezas...),
        verificar si los números coinciden con el RANGO CORRECTO
        (k_min-k_max).
      - Si NO coinciden, reemplazarlos por el rango correcto.
      - Tolerancia ±1 al matching del rango correcto (para aceptar
        redondeos del LLM "13-19" cuando el correcto es 13-20).

    Devuelve el texto corregido. Es idempotente: si el texto ya está
    bien, lo devuelve sin cambios.
    """
    import re
    if not texto or not impacto:
        return texto
    k_min, k_max = impacto.get("kg_perdidos_lote_periodo", (0, 0))
    if k_min == 0 and k_max == 0:
        return texto
    correcto_min = int(round(k_min))
    correcto_max = int(round(k_max))
    # Si por casualidad el rango por animal × días da algo similar al
    # rango por-lote-por-día (dias_evento=1), no hay nada que auditar.
    dias = impacto.get("dias_evento", 1)
    if dias <= 1:
        return texto

    # Palabras que indican que la cifra es TOTAL ACUMULADA del evento
    # completo. SOLO corregimos si aparece alguna de estas Y NO aparece
    # una palabra que indique "por día".
    palabras_evento_total = (
        "acumulad", "total", "en todo el evento", "durante el evento",
        "en los 2 día", "en los 3 día", "en los 4 día", "en los 5 día",
        "en los 6 día", "en los 7 día", "en los días del evento",
        "no suman", "no sumar", "no recuper",
    )
    # Palabras que indican que la cifra es POR DÍA (NO corregir).
    palabras_por_dia = (
        "por día", "/día", "diari", "al día", "cada día",
    )

    # Patrón: número entero o decimal opcional + separador (-, –, a)
    # + número + opcional "kg". Aceptamos formatos:
    #   "7-10 kg", "7 a 10 kg", "6.5-10 kg", "6,5-10 kg".
    patron = re.compile(
        r"\b(\d{1,4}(?:[.,]\d+)?)\s*[-–]\s*(\d{1,4}(?:[.,]\d+)?)\s*kg\b",
        re.IGNORECASE,
    )
    patron_a = re.compile(
        r"\b(\d{1,4}(?:[.,]\d+)?)\s+a\s+(\d{1,4}(?:[.,]\d+)?)\s*kg\b",
        re.IGNORECASE,
    )

    def _es_correcto(n1: float, n2: float) -> bool:
        """¿El rango n1-n2 coincide (con tolerancia ±1.5) con el
        rango correcto del evento?"""
        return (abs(n1 - correcto_min) <= 1.5
                and abs(n2 - correcto_max) <= 1.5)

    def _es_por_animal(n1: float, n2: float) -> bool:
        """¿El rango n1-n2 es el de POR ANIMAL POR DÍA (no del lote)?
        Si es así, NO tocarlo."""
        a_min, a_max = impacto.get("adpv_perdida_kg_rango", (0, 0))
        return (abs(n1 - a_min) <= 0.05 and abs(n2 - a_max) <= 0.05)

    def _es_total_del_evento(sufijo_inmediato: str,
                              entorno: str) -> bool:
        """¿El número está etiquetado claramente como TOTAL/ACUMULADO
        del evento completo?

        Heurística en dos pasos:
          1. Si el SUFIJO inmediato del número (los próximos 20 chars
             después del "kg") dice "/día", "por día", "diari", etc.,
             el número es DIARIO — NO tocar.
          2. Si no, buscar palabras de evento total en el entorno.
             Si las hay → corregir.
        """
        sufijo = sufijo_inmediato.lower()
        if any(k in sufijo for k in palabras_por_dia):
            return False  # explícitamente diario, no tocar
        t = entorno.lower()
        return any(k in t for k in palabras_evento_total)

    def _procesar(patron_obj, texto_in):
        # Recolectamos todos los matches y procesamos de atrás hacia
        # adelante para no alterar índices.
        matches = list(patron_obj.finditer(texto_in))
        for m in reversed(matches):
            n1 = float(m.group(1).replace(",", "."))
            n2 = float(m.group(2).replace(",", "."))
            # NO tocar rangos por-animal-por-día (ej: "0,13-0,20 kg")
            if _es_por_animal(n1, n2):
                continue
            # NO tocar rangos correctos
            if _es_correcto(n1, n2):
                continue
            # Sólo corregir si el número está etiquetado claramente
            # como TOTAL/ACUMULADO del evento completo.
            #
            # Sufijo inmediato (20 chars después del "kg"): si dice
            # "/día", "por día", etc., el número es diario → no tocar.
            # Entorno corto (acotado por separadores fuertes si los
            # hay): si hay palabras de "evento total/acumulado", corregir.
            sufijo_inmediato = texto_in[m.end():m.end() + 20]
            separadores = "().;—:"
            ini = max(0, m.start() - 80)
            fin = min(len(texto_in), m.end() + 80)
            for k in range(m.start() - 1, ini - 1, -1):
                if texto_in[k] in separadores:
                    ini = k + 1
                    break
            for k in range(m.end(), fin):
                if texto_in[k] in separadores:
                    fin = k
                    break
            entorno = texto_in[ini:fin]
            if not _es_total_del_evento(sufijo_inmediato, entorno):
                continue
            # Reemplazar el rango numérico (mantener "kg" intacto)
            reemplazo = f"{correcto_min}-{correcto_max} kg"
            texto_in = (
                texto_in[:m.start()] + reemplazo + texto_in[m.end():]
            )
        return texto_in

    texto = _procesar(patron, texto)
    texto = _procesar(patron_a, texto)
    return texto


def estimar_impacto_peor_dia_semanal(
    clima: Dict,
    lotes: Optional[list] = None,
) -> Optional[Dict]:
    """Identifica el peor día de frío en la semana proyectada y calcula
    el impacto productivo sobre el lote más sensible.

    Recibe el dict `clima` de Open-Meteo (con sección `daily` y `hourly`)
    y la lista de lotes del cliente (con peso/cantidad/raza/categoría).
    Devuelve el impacto del peor evento o None si la semana es estable.

    Estrategia simple:
      - Peor día = el de menor temperatura mínima.
      - Lote elegido = el más sensible al frío (ternero > vaquillona >
        recría > novillito > novillo > vaca > toro). Si no hay datos
        suficientes, retorna None.
      - Días del evento = contar cuántos días consecutivos alrededor
        del peor día tienen T° mínima <= 5°C.
    """
    if not lotes:
        return None
    daily = (clima or {}).get("daily", {}) or {}
    t_min_list = daily.get("temperature_2m_min", []) or []
    precip = daily.get("precipitation_sum", []) or []
    hum_max = daily.get("relative_humidity_2m_max", []) or []
    viento_max = daily.get("windspeed_10m_max", []) or []
    if not t_min_list:
        return None

    # Filtrar None y encontrar índice del mínimo
    valores = [(i, v) for i, v in enumerate(t_min_list) if v is not None]
    if not valores:
        return None
    idx_peor, t_min_peor = min(valores, key=lambda x: x[1])
    # Si el peor día no llega a 8°C, no hay evento de frío relevante
    if t_min_peor > 8:
        return None

    # Contar días consecutivos con T° <= 5°C alrededor del peor día
    dias_evento = 1
    i = idx_peor + 1
    while i < len(t_min_list) and (t_min_list[i] or 99) <= 5:
        dias_evento += 1
        i += 1
    i = idx_peor - 1
    while i >= 0 and (t_min_list[i] or 99) <= 5:
        dias_evento += 1
        i -= 1

    # Datos del peor día
    viento_peor = (viento_max[idx_peor]
                    if idx_peor < len(viento_max) else None)
    hum_peor = (hum_max[idx_peor]
                 if idx_peor < len(hum_max) else None)
    lluvia_peor = (precip[idx_peor]
                    if idx_peor < len(precip) else 0)
    # Barro: si llovió >20mm acumulado en 3 días alrededor
    idx_3d = list(range(max(0, idx_peor - 1), idx_peor + 2))
    precip_3d = sum((precip[k] or 0) for k in idx_3d if k < len(precip))
    barro = precip_3d > 20

    # Elegir el lote MÁS sensible
    orden_sensibilidad = {
        "ternero": 0, "recria": 1, "vaquillona": 2, "novillito": 3,
        "novillo": 4, "vaca": 5, "toro": 6,
    }
    candidatos = []
    for l in lotes:
        peso = (l.get("peso_promedio_kg") or l.get("ultimo_peso_kg")
                 or l.get("peso_ingreso_kg"))
        if not peso or peso <= 0:
            continue
        cat_norm = _normalizar_categoria(l.get("categoria", ""))
        candidatos.append((orden_sensibilidad.get(cat_norm, 10), l))
    if not candidatos:
        return None
    candidatos.sort(key=lambda x: x[0])
    _, lote_sensible = candidatos[0]
    peso_l = (lote_sensible.get("peso_promedio_kg")
                or lote_sensible.get("ultimo_peso_kg")
                or lote_sensible.get("peso_ingreso_kg"))

    return estimar_impacto_frio(
        peso_kg=peso_l,
        categoria=lote_sensible.get("categoria", ""),
        raza=lote_sensible.get("raza", ""),
        t_min_c=t_min_peor,
        viento_kmh=viento_peor,
        humedad_pct=hum_peor,
        barro=barro,
        pelaje_mojado=(lluvia_peor or 0) > 5,
        dias_evento=dias_evento,
        cantidad=(lote_sensible.get("cantidad_inicial")
                   or lote_sensible.get("cantidad_animales")),
        # Overrides cargados desde la ficha del lote (si están)
        adpv_objetivo_kg=lote_sensible.get("adpv_objetivo_kg"),
        energia_dieta_mcal_em_kg_ms=lote_sensible.get(
            "energia_dieta_mcal_em_kg_ms"
        ),
    )


def formato_impacto_texto(impacto: Dict) -> str:
    """Convierte el dict de impacto en un BLOQUE DE DATOS para inyectar
    al LLM con TODOS los rangos PRE-CALCULADOS, así el LLM nunca
    necesita hacer aritmética y solo elige cuál citar.

    Estrategia:
      - Damos el dato en 3 granularidades distintas (por animal/día,
        por lote/día, total del evento) ya calculados. El LLM puede
        elegir cualquiera, pero NO debe inventar otros números.
      - Especificamos siempre la unidad y el período en cada línea.
      - Aclaramos explícitamente: NO aplicar rendimiento de carcasa
        (siempre es peso vivo, no res en gancho).
    """
    if not impacto:
        return ""
    g_min, g_max = impacto["gasto_extra_pct"]
    a_min, a_max = impacto["adpv_perdida_kg_rango"]
    p_min, p_max = impacto["pct_adpv_perdida"]
    cantidad = impacto.get("cantidad_lote") or 1
    dias = impacto.get("dias_evento", 1)
    # Pre-calculamos por-lote por-día (intermedio) para que el LLM no
    # tenga que multiplicar.
    lote_dia_min = a_min * cantidad
    lote_dia_max = a_max * cantidad
    k_min, k_max = impacto.get("kg_perdidos_lote_periodo", (0, 0))

    partes = []
    partes.append(
        f"REQUERIMIENTO DE MANTENIMIENTO ELEVADO: "
        f"+{g_min:.0f}-{g_max:.0f}% durante el evento "
        f"(energía solo para sostener temperatura corporal)."
    )
    partes.append(
        f"PÉRDIDA POR ANIMAL POR DÍA: si la dieta no aporta esa "
        f"energía extra y el consumo no sube, cada animal deja de "
        f"ganar {a_min:.2f}-{a_max:.2f} kg/día de PESO VIVO "
        f"({p_min:.0f}-{p_max:.0f}% del ADPV objetivo)."
    )
    if cantidad > 1:
        _dia_label = "día" if dias == 1 else "días"
        partes.append(
            f"PÉRDIDA SOBRE TODO EL LOTE POR DÍA "
            f"({cantidad:.0f} cab.): {lote_dia_min:.1f}-{lote_dia_max:.1f} "
            f"kg de PESO VIVO/día que el lote completo no suma."
        )
        partes.append(
            f"PÉRDIDA TOTAL ACUMULADA EN EL EVENTO "
            f"({cantidad:.0f} cab. × {dias} {_dia_label}): "
            f"{k_min:.0f}-{k_max:.0f} "
            f"kg de PESO VIVO sobre el lote en TODO el evento."
        )
    partes.append(
        "IMPORTANTE: TODOS los kg de arriba son de PESO VIVO en "
        "balanza (lo que el productor cobra al frigorífico). NO son "
        "kg de res en gancho. NO aplicar rendimiento de carcasa "
        "(0,50-0,55) — los números ya son los finales que pierde el "
        "productor. USAR ESTOS RANGOS LITERALES, no recalcularlos."
    )
    fuente = impacto.get(
        "fuente",
        "NRC/NASEM 2016 (Cap. Environment) + ajustes Pampa Húmeda "
        "(Pordomingo, Latimori)",
    )
    partes.append(
        f"FUENTE BIBLIOGRÁFICA de estos números: {fuente}. CUANDO "
        f"CITES alguno de los rangos (% de mantenimiento, kg/día, kg "
        f"totales) en tu respuesta, MENCIONÁ brevemente la fuente "
        f"entre paréntesis o al pie. Ejemplos válidos: "
        f"'+21-36% (NRC 2016)', '(cálculo NRC para este lote)', "
        f"'según NRC/NASEM'. Eso le da autoridad técnica al dato y le "
        f"confirma al productor que no se inventa nada — clave para "
        f"que el sistema gane su confianza."
    )
    return " ".join(partes)
