"""
Formulador de dietas de mínimo costo (Least Cost Ration).

Usa programación lineal (scipy.optimize.linprog) para encontrar la mezcla
de ingredientes disponibles que cumple los requerimientos NASEM al menor
costo posible.

Si el problema es infactible (no se pueden cubrir los requerimientos con
los ingredientes disponibles), identifica qué nutriente falta y sugiere
ingredientes correctivos.

Esto es lo que hace NDS, AMTS, MIXIT y los demás softwares profesionales
de formulación. La diferencia: tu base de ingredientes la editás vos con
los precios de TU campo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# LISTA NEGRA: ingredientes PROHIBIDOS en alimentación bovina (Argentina)
# =====================================================================
# Por SENASA Resolución 1/2002 y normativa internacional (OIE/WOAH) están
# prohibidos los subproductos animales en raciones de rumiantes para
# prevenir Encefalopatía Espongiforme Bovina (BSE / "vaca loca").

INGREDIENTES_PROHIBIDOS_BOVINOS = {
    "harina de carne",
    "harina de hueso",
    "harina de carne y hueso",
    "harina de sangre",
    "harina de pluma",
    "harina de pescado",
    "grasa animal",
    "sebo de vacuno",
    "subproducto avícola",
    "harina de víscera",
}


def es_ingrediente_prohibido(nombre: str) -> bool:
    """Verifica si un ingrediente está prohibido para rumiantes."""
    if not nombre:
        return False
    n = nombre.lower().strip()
    return any(p in n for p in INGREDIENTES_PROHIBIDOS_BOVINOS)


# =====================================================================
# 1) BASE DE INGREDIENTES TÍPICA ARGENTINA
# =====================================================================

@dataclass
class Ingrediente:
    """Composición nutricional de un ingrediente.

    Todos los % nutricionales están EN BASE A MATERIA SECA.
    El precio es por kg TAL CUAL (con humedad real).
    """
    nombre: str
    ms_pct: float                     # % materia seca
    pb_pct_ms: float                  # % proteína bruta en MS (incluye NNP convertido)
    em_mcal_kg_ms: float              # Mcal de EM por kg MS
    fdn_pct_ms: float                 # % FDN en MS
    ca_pct_ms: float                  # % calcio en MS
    p_pct_ms: float                   # % fósforo en MS
    precio_kg_tal_cual: float         # $ por kg como llega
    nnp_pct_ms: float = 0.0           # % NNP (urea, biuret) en MS — control toxicidad
    max_inclusion_pct_ms: float = 100.0  # límite máx en la dieta (% de MS)
    min_inclusion_pct_ms: float = 0.0    # límite mín
    disponible: bool = False              # por default no disponible — el usuario tilda los que tiene
    categoria: str = "concentrado"    # concentrado / forraje / suplemento / mineral / balanceado

    @property
    def precio_kg_ms(self) -> float:
        """Precio por kg de MS (más útil para LP)."""
        if self.ms_pct <= 0:
            return float("inf")
        return self.precio_kg_tal_cual * 100 / self.ms_pct


def ingredientes_default(precio_actualizado: bool = True) -> List[Ingrediente]:
    """Lista típica de ingredientes argentinos para feedlot/recría.

    Precios: aproximaciones a precios pizarra Rosario / mercado libre 2025.
    Editá libremente desde la UI.
    """
    return [
        # ---- GRANOS / CONCENTRADOS ----
        # NOTA: usamos MAÍZ GRANO (entero / partido grueso). El maíz
        # MOLIDO FINO no se utiliza en estos esquemas porque acelera la
        # fermentación ruminal y aumenta el riesgo de acidosis. Si se
        # procesa, debe ser molienda gruesa (>3 mm) o partido.
        Ingrediente("Maíz grano", 88.0, 9.0, 3.10, 10.0, 0.03, 0.30,
                    precio_kg_tal_cual=180, max_inclusion_pct_ms=70.0,
                    categoria="concentrado"),
        Ingrediente("Grano de sorgo molido", 88.0, 10.0, 2.90, 12.0, 0.04, 0.32,
                    precio_kg_tal_cual=160, max_inclusion_pct_ms=60.0,
                    categoria="concentrado"),
        Ingrediente("Grano de cebada molida", 88.0, 11.5, 3.00, 18.0, 0.05, 0.36,
                    precio_kg_tal_cual=170, max_inclusion_pct_ms=60.0,
                    categoria="concentrado"),
        Ingrediente("Grano de trigo molido", 88.0, 13.0, 3.10, 13.0, 0.04, 0.39,
                    precio_kg_tal_cual=200, max_inclusion_pct_ms=40.0,
                    categoria="concentrado"),
        Ingrediente("Grano de avena molida", 89.0, 11.0, 2.70, 32.0, 0.07, 0.40,
                    precio_kg_tal_cual=150, max_inclusion_pct_ms=40.0,
                    categoria="concentrado"),

        # ---- FUENTES PROTEICAS ----
        Ingrediente("Pellet de soja (44% PB)", 89.0, 44.0, 3.00, 14.0, 0.30, 0.65,
                    precio_kg_tal_cual=320, max_inclusion_pct_ms=30.0,
                    categoria="concentrado"),
        Ingrediente("Expeller de soja (38% PB)", 92.0, 38.0, 3.40, 17.0, 0.28, 0.62,
                    precio_kg_tal_cual=300, max_inclusion_pct_ms=25.0,
                    categoria="concentrado"),
        Ingrediente("Pellet de girasol (32% PB)", 90.0, 32.0, 2.40, 38.0, 0.40, 1.00,
                    precio_kg_tal_cual=210, max_inclusion_pct_ms=25.0,
                    categoria="concentrado"),
        Ingrediente("Burlanda de maíz (DDGS)", 90.0, 28.0, 3.10, 35.0, 0.10, 0.85,
                    precio_kg_tal_cual=190, max_inclusion_pct_ms=30.0,
                    categoria="concentrado"),
        # NOTA: Harina de carne y hueso ELIMINADA — PROHIBIDA en Argentina
        # (SENASA Res. 1/2002 y posteriores) para alimentación de rumiantes
        # por prevención de Encefalopatía Espongiforme Bovina (BSE).

        # ---- FORRAJES Y SILAJES ----
        Ingrediente("Silaje de maíz (planta entera)", 35.0, 8.0, 2.40, 45.0, 0.25, 0.22,
                    precio_kg_tal_cual=40, max_inclusion_pct_ms=70.0,
                    min_inclusion_pct_ms=10.0, categoria="forraje"),
        Ingrediente("Silaje de sorgo", 30.0, 7.5, 2.10, 50.0, 0.30, 0.22,
                    precio_kg_tal_cual=35, max_inclusion_pct_ms=60.0,
                    categoria="forraje"),
        Ingrediente("Heno de alfalfa", 90.0, 18.0, 2.05, 46.0, 1.50, 0.25,
                    precio_kg_tal_cual=200, max_inclusion_pct_ms=40.0,
                    categoria="forraje"),
        Ingrediente("Heno de moha / pasto llorón", 90.0, 8.0, 1.85, 65.0, 0.35, 0.18,
                    precio_kg_tal_cual=120, max_inclusion_pct_ms=40.0,
                    categoria="forraje"),
        Ingrediente("Rollo de pradera mezcla", 88.0, 11.0, 1.95, 58.0, 0.50, 0.25,
                    precio_kg_tal_cual=140, max_inclusion_pct_ms=50.0,
                    categoria="forraje"),
        Ingrediente("Pastura natural (verdeo)", 22.0, 12.0, 2.20, 52.0, 0.55, 0.30,
                    precio_kg_tal_cual=15, max_inclusion_pct_ms=80.0,
                    categoria="forraje"),

        # ---- BALANCEADOS Y CONCENTRADOS PROTEICOS COMERCIALES ----
        # Fibrogreen Plus (Biofarma): CONCENTRADO PROTEICO (NO es núcleo).
        # Composición real informada por el productor:
        # - PB: 25% (de los cuales 4% es NNP/urea → ~11,5% PB equiv. del NNP)
        # - MS: 88%
        # - Monensina: 240 ppm en el producto (1,2 kg/tn de premix al 20%)
        # - Contiene además: taninos, levaduras, enzimas fibrolíticas.
        # Limitado al 20% MS de la dieta por la concentración de monensina.
        Ingrediente("Fibrogreen Plus", 88.0, 25.0, 2.45, 28.0, 3.50, 1.20,
                    precio_kg_tal_cual=520, nnp_pct_ms=4.0,
                    max_inclusion_pct_ms=20.0, categoria="balanceado"),
        Ingrediente("Balanceado iniciador 18% PB", 89.0, 18.0, 2.85, 18.0, 1.00, 0.55,
                    precio_kg_tal_cual=420, max_inclusion_pct_ms=80.0,
                    categoria="balanceado"),
        Ingrediente("Balanceado terminación 14% PB", 89.0, 14.0, 2.95, 12.0, 0.85, 0.45,
                    precio_kg_tal_cual=380, max_inclusion_pct_ms=80.0,
                    categoria="balanceado"),
        Ingrediente("Núcleo proteico 30% PB (con monensina + NNP)", 90.0, 30.0, 2.70, 15.0, 2.20, 0.90,
                    precio_kg_tal_cual=480, nnp_pct_ms=3.0,
                    max_inclusion_pct_ms=15.0, categoria="balanceado"),
        Ingrediente("Pellet pellet recría 16% PB", 89.0, 16.0, 2.80, 22.0, 0.90, 0.50,
                    precio_kg_tal_cual=400, max_inclusion_pct_ms=70.0,
                    categoria="balanceado"),

        # ---- SUPLEMENTOS / MINERALES ----
        Ingrediente("Urea (NNP puro)", 99.0, 287.0, 0.0, 0.0, 0.0, 0.0,
                    precio_kg_tal_cual=600, nnp_pct_ms=100.0,
                    max_inclusion_pct_ms=1.5, categoria="suplemento"),
        Ingrediente("Monensina sódica (Rumensin)", 95.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    precio_kg_tal_cual=15000, max_inclusion_pct_ms=0.05,
                    categoria="suplemento"),
        Ingrediente("Núcleo mineral-vitamínico", 95.0, 0.0, 0.0, 0.0, 15.0, 8.0,
                    precio_kg_tal_cual=800, max_inclusion_pct_ms=5.0,
                    min_inclusion_pct_ms=0.0, categoria="mineral"),
        Ingrediente("Sal blanca", 99.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    precio_kg_tal_cual=80, max_inclusion_pct_ms=2.0,
                    min_inclusion_pct_ms=0.0, categoria="mineral"),
        Ingrediente("Bicarbonato de sodio", 99.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    precio_kg_tal_cual=350, max_inclusion_pct_ms=1.5,
                    categoria="mineral"),
        Ingrediente("Carbonato de calcio", 99.0, 0.0, 0.0, 0.0, 38.0, 0.0,
                    precio_kg_tal_cual=120, max_inclusion_pct_ms=2.0,
                    categoria="mineral"),
        Ingrediente("Conchilla molida", 99.0, 0.0, 0.0, 0.0, 36.0, 0.0,
                    precio_kg_tal_cual=100, max_inclusion_pct_ms=2.0,
                    categoria="mineral"),
        Ingrediente("Fosfato dicálcico", 96.0, 0.0, 0.0, 0.0, 22.0, 18.5,
                    precio_kg_tal_cual=900, max_inclusion_pct_ms=1.5,
                    categoria="mineral"),
    ]


# =====================================================================
# 2) RESULTADO DE LA OPTIMIZACIÓN
# =====================================================================

@dataclass
class FormulacionResultado:
    factible: bool
    mensaje: str
    costo_total_dia: float = 0.0
    costo_por_kg_ms: float = 0.0
    consumo_ms_kg: float = 0.0
    consumo_tal_cual_kg: float = 0.0

    # Por ingrediente: {nombre: {ms_kg, tal_cual_kg, pct_ms, costo_dia}}
    composicion: List[Dict] = field(default_factory=list)

    # Aportes vs requerimientos
    pb_aportado_g: float = 0.0
    pb_requerido_g: float = 0.0
    em_aportado_mcal: float = 0.0
    em_requerido_mcal: float = 0.0
    fdn_aportado_pct: float = 0.0
    ca_aportado_g: float = 0.0
    p_aportado_g: float = 0.0
    nnp_aportado_pct: float = 0.0           # % NNP en MS de la dieta total
    nnp_pb_equivalente_g: float = 0.0       # g de PB equivalente que aporta el NNP
    pb_verdadera_g: float = 0.0             # PB total - PB equiv. del NNP

    deficiencias: List[Dict] = field(default_factory=list)
    sugerencias: List[str] = field(default_factory=list)
    advertencias: List[str] = field(default_factory=list)


# =====================================================================
# 3) OPTIMIZADOR DE MÍNIMO COSTO
# =====================================================================

def formular_minimo_costo(
    ingredientes: List[Ingrediente],
    consumo_ms_kg: float,
    pb_g_dia: float,
    em_mcal_dia: float,
    fdn_min_pct: float,
    ca_g_dia: float,
    p_g_dia: float,
) -> FormulacionResultado:
    """
    Resuelve el problema de programación lineal:

        minimize:  sum(x_i × precio_ms_i)
        subject to:
            sum(x_i) = DMI                               (consumo total)
            sum(x_i × pb_i)  >= PB_g_dia                 (proteína)
            sum(x_i × em_i)  >= EM_mcal_dia              (energía)
            sum(x_i × fdn_i) >= FDN_min × DMI            (fibra)
            sum(x_i × ca_i)  >= Ca_g_dia                 (calcio)
            sum(x_i × p_i)   >= P_g_dia                  (fósforo)
            x_i >= min_inclusion_i × DMI                 (mín)
            x_i <= max_inclusion_i × DMI                 (máx)
            x_i = 0 si no disponible
    donde x_i = kg de MS del ingrediente i
    """
    # Filtrar prohibidos (subproductos animales) ANTES de procesar
    prohibidos_detectados = [
        i.nombre for i in ingredientes
        if i.disponible and es_ingrediente_prohibido(i.nombre)
    ]
    if prohibidos_detectados:
        return FormulacionResultado(
            factible=False,
            mensaje=(
                f"🚫 PROHIBIDO en bovinos (SENASA Res. 1/2002): "
                f"{', '.join(prohibidos_detectados)}. "
                "Los subproductos de origen animal no pueden usarse en "
                "alimentación de rumiantes por prevención de BSE."
            ),
        )

    disponibles = [i for i in ingredientes if i.disponible
                   and not es_ingrediente_prohibido(i.nombre)]
    if not disponibles:
        return FormulacionResultado(
            factible=False,
            mensaje="No hay ingredientes disponibles. Marcá al menos uno como disponible.",
        )

    # Defensa final: cualquier valor numérico None / NaN se reemplaza por 0
    # (excepto ms_pct que se cae a 88 para evitar división por cero)
    def _clean(v: float, default: float = 0.0) -> float:
        if v is None:
            return default
        try:
            f = float(v)
            return default if f != f else f  # NaN
        except (TypeError, ValueError):
            return default

    for ing in disponibles:
        ing.ms_pct = _clean(ing.ms_pct, 88.0) or 88.0
        if ing.ms_pct <= 0:
            ing.ms_pct = 88.0
        ing.pb_pct_ms = _clean(ing.pb_pct_ms, 0.0)
        ing.em_mcal_kg_ms = _clean(ing.em_mcal_kg_ms, 0.0)
        ing.fdn_pct_ms = _clean(ing.fdn_pct_ms, 0.0)
        ing.ca_pct_ms = _clean(ing.ca_pct_ms, 0.0)
        ing.p_pct_ms = _clean(ing.p_pct_ms, 0.0)
        ing.precio_kg_tal_cual = _clean(ing.precio_kg_tal_cual, 0.0)
        ing.min_inclusion_pct_ms = _clean(ing.min_inclusion_pct_ms, 0.0)
        ing.max_inclusion_pct_ms = _clean(ing.max_inclusion_pct_ms, 100.0) or 100.0

    n = len(disponibles)
    # Función objetivo: precio por kg de MS
    c = np.array([ing.precio_kg_ms for ing in disponibles])

    # Restricciones de igualdad: sum(x_i) = DMI
    A_eq = np.ones((1, n))
    b_eq = np.array([consumo_ms_kg])

    # Restricciones de desigualdad: A_ub x <= b_ub
    # NASEM: PB, EM, FDN, Ca, P son MÍNIMOS, así que -A_ub x <= -b_min
    rows = []
    bs = []

    # PB ≥ pb_g_dia → -sum(pb_i × x_i) ≤ -pb_g_dia
    rows.append([-(ing.pb_pct_ms / 100 * 1000) for ing in disponibles])  # g por kg MS
    bs.append(-pb_g_dia)

    # EM ≥ em_mcal_dia
    rows.append([-ing.em_mcal_kg_ms for ing in disponibles])
    bs.append(-em_mcal_dia)

    # FDN: sum(fdn_i × x_i) / sum(x_i) ≥ fdn_min_pct/100
    # Como sum(x_i) = DMI, se simplifica: sum(fdn_i × x_i) ≥ fdn_min_pct × DMI / 100
    rows.append([-(ing.fdn_pct_ms) for ing in disponibles])  # %FDN
    bs.append(-fdn_min_pct * consumo_ms_kg)

    # Ca y P en gramos: pct_ms × 1000 g/kg = g por kg MS
    rows.append([-(ing.ca_pct_ms / 100 * 1000) for ing in disponibles])
    bs.append(-ca_g_dia)
    rows.append([-(ing.p_pct_ms / 100 * 1000) for ing in disponibles])
    bs.append(-p_g_dia)

    A_ub = np.array(rows)
    b_ub = np.array(bs)

    # Bounds por ingrediente
    bounds = []
    for ing in disponibles:
        lb = ing.min_inclusion_pct_ms / 100 * consumo_ms_kg
        ub = ing.max_inclusion_pct_ms / 100 * consumo_ms_kg
        bounds.append((lb, ub))

    # Import perezoso de scipy (no se requiere para que el módulo se importe)
    from scipy.optimize import linprog
    # Resolver
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if not result.success:
        # Intentar diagnosticar deficiencias relajando una restricción a la vez
        return _diagnosticar_infactible(
            disponibles, consumo_ms_kg, pb_g_dia, em_mcal_dia,
            fdn_min_pct, ca_g_dia, p_g_dia, ingredientes,
        )

    # Solución factible: armar resultado
    x = result.x
    composicion = []
    for ing, kg_ms in zip(disponibles, x):
        if kg_ms < 0.001:
            continue
        kg_tal_cual = kg_ms * 100 / ing.ms_pct
        costo_dia = kg_ms * ing.precio_kg_ms
        composicion.append({
            "nombre": ing.nombre,
            "categoria": ing.categoria,
            "kg_ms": float(kg_ms),
            "kg_tal_cual": float(kg_tal_cual),
            "pct_ms": float(kg_ms / consumo_ms_kg * 100),
            "costo_dia": float(costo_dia),
            "aporte_pb_g": float(kg_ms * ing.pb_pct_ms * 10),
            "aporte_em_mcal": float(kg_ms * ing.em_mcal_kg_ms),
        })

    costo_total = float(np.sum(c * x))
    pb_aportado = float(sum(c["aporte_pb_g"] for c in composicion))
    em_aportado = float(sum(c["aporte_em_mcal"] for c in composicion))
    fdn_pct = float(sum(ing.fdn_pct_ms * kg / consumo_ms_kg
                        for ing, kg in zip(disponibles, x)))
    ca_g = float(sum(ing.ca_pct_ms * kg * 10 for ing, kg in zip(disponibles, x)))
    p_g = float(sum(ing.p_pct_ms * kg * 10 for ing, kg in zip(disponibles, x)))
    consumo_tal_cual = float(sum(c["kg_tal_cual"] for c in composicion))

    return FormulacionResultado(
        factible=True,
        mensaje="✅ Formulación factible — mínimo costo encontrado",
        costo_total_dia=costo_total,
        costo_por_kg_ms=costo_total / consumo_ms_kg if consumo_ms_kg else 0,
        consumo_ms_kg=consumo_ms_kg,
        consumo_tal_cual_kg=consumo_tal_cual,
        composicion=sorted(composicion, key=lambda r: -r["pct_ms"]),
        pb_aportado_g=pb_aportado,
        pb_requerido_g=pb_g_dia,
        em_aportado_mcal=em_aportado,
        em_requerido_mcal=em_mcal_dia,
        fdn_aportado_pct=fdn_pct,
        ca_aportado_g=ca_g,
        p_aportado_g=p_g,
    )


# =====================================================================
# 4) VERIFICAR UNA RECETA EXISTENTE
# =====================================================================

def verificar_receta(
    ingredientes: List[Ingrediente],
    porcentajes: Dict[str, float],
    consumo_ms_kg: float,
    pb_g_dia: float,
    em_mcal_dia: float,
    fdn_min_pct: float,
    ca_g_dia: float,
    p_g_dia: float,
    pb_rango_pct: Optional[Tuple[float, float]] = None,
) -> FormulacionResultado:
    """
    Evalúa una receta DADA por el usuario (sin optimizar).
    `porcentajes` = {nombre_ingrediente: % en la mezcla}.
    Devuelve los aportes nutricionales y compara con requerimientos NASEM.

    Útil para chequear si una mezcla típica del campo (ej. 88% maíz + 12%
    Fibrogreen) cumple los requerimientos de la categoría/peso.
    """
    suma_pct = sum(porcentajes.values())
    if abs(suma_pct - 100) > 0.5:
        return FormulacionResultado(
            factible=False,
            mensaje=f"Los porcentajes de la receta deben sumar 100% (actual: {suma_pct:.1f}%)",
        )

    # Validar prohibidos
    prohibidos = [n for n in porcentajes if es_ingrediente_prohibido(n)]
    if prohibidos:
        return FormulacionResultado(
            factible=False,
            mensaje=f"🚫 PROHIBIDO: {', '.join(prohibidos)} no se puede usar en bovinos.",
        )

    by_name = {i.nombre: i for i in ingredientes}
    composicion = []
    pb_aportada = em_aportada = fdn_aportada = ca_aportado = p_aportado = 0.0
    nnp_aportado_g = 0.0   # gramos totales de NNP / día
    costo_total = 0.0

    for nombre, pct in porcentajes.items():
        ing = by_name.get(nombre)
        if not ing:
            continue
        kg_ms = consumo_ms_kg * pct / 100
        kg_tc = kg_ms * 100 / ing.ms_pct if ing.ms_pct > 0 else 0
        costo = kg_ms * ing.precio_kg_ms
        costo_total += costo
        pb_aportada += kg_ms * ing.pb_pct_ms * 10
        em_aportada += kg_ms * ing.em_mcal_kg_ms
        fdn_aportada += pct * ing.fdn_pct_ms / 100
        ca_aportado += kg_ms * ing.ca_pct_ms * 10
        p_aportado += kg_ms * ing.p_pct_ms * 10
        nnp_aportado_g += kg_ms * ing.nnp_pct_ms * 10   # g NNP / día
        composicion.append({
            "nombre": ing.nombre,
            "categoria": ing.categoria,
            "pct_ms": float(pct),
            "kg_ms": float(kg_ms),
            "kg_tal_cual": float(kg_tc),
            "costo_dia": float(costo),
            "aporte_pb_g": float(kg_ms * ing.pb_pct_ms * 10),
            "aporte_em_mcal": float(kg_ms * ing.em_mcal_kg_ms),
            "aporte_nnp_g": float(kg_ms * ing.nnp_pct_ms * 10),
        })

    # NNP en % de la dieta total (en MS)
    nnp_pct_dieta = (nnp_aportado_g / 1000) / consumo_ms_kg * 100 if consumo_ms_kg > 0 else 0
    # PB equivalente del NNP (cada kg urea aporta 2.87 kg PB equivalente)
    nnp_pb_equiv = nnp_aportado_g * 2.87
    pb_verdadera = max(0, pb_aportada - nnp_pb_equiv)

    deficiencias = []
    sugerencias = []
    advertencias = []

    # ---- CHEQUEO TOXICIDAD NNP ----
    # Límite seguro NASEM 2016: NNP en MS de la dieta total ≤ 1%
    # Crítico (riesgo de intoxicación): > 1.5%
    if nnp_pct_dieta > 1.5:
        advertencias.append(
            f"🔴 PELIGRO: NNP de la dieta = {nnp_pct_dieta:.2f}% (>1.5%). "
            "Riesgo ALTO de intoxicación amoniacal. REDUCIR urea o "
            "ingrediente con NNP."
        )
    elif nnp_pct_dieta > 1.0:
        advertencias.append(
            f"⚠️ NNP de la dieta = {nnp_pct_dieta:.2f}% (>1%). "
            "Por encima del límite seguro NASEM. Adaptación gradual y "
            "monitoreo de signos clínicos."
        )
    elif nnp_pct_dieta > 0.7:
        advertencias.append(
            f"📊 NNP de la dieta = {nnp_pct_dieta:.2f}% (rango 0.7-1%). "
            "Aceptable con buena adaptación e ionóforo."
        )

    # PB: usar rango práctico (Pezzola/Pordomingo) si fue provisto, sino solo NASEM
    pb_aportada_pct = (pb_aportada / consumo_ms_kg / 10) if consumo_ms_kg > 0 else 0
    if pb_rango_pct is not None:
        pb_min_pct, pb_max_pct = pb_rango_pct
        if pb_aportada_pct < pb_min_pct - 0.3:    # 0.3% de tolerancia
            deficiencias.append({
                "nutriente": "Proteína bruta",
                "requerido": pb_min_pct * consumo_ms_kg * 10,
                "max_alcanzable": pb_aportada,
                "deficit_pct": (pb_min_pct - pb_aportada_pct) / pb_min_pct * 100,
                "info": (
                    f"PB receta: {pb_aportada_pct:.1f}% | rango etapa: "
                    f"{pb_min_pct:.1f}-{pb_max_pct:.1f}% | NASEM puro: "
                    f"{pb_g_dia/consumo_ms_kg/10:.1f}%"
                ),
            })
            sugerencias.append(
                f"Subir PB al menos al {pb_min_pct:.1f}% del rango práctico. "
                "Opciones: pellet de soja, expeller, burlanda DDGS, "
                "o urea (máx 1,5% de la dieta)."
            )
        elif pb_aportada_pct > pb_max_pct + 0.3:
            advertencias.append(
                f"📊 PB de la receta {pb_aportada_pct:.1f}% supera el rango "
                f"práctico {pb_min_pct:.1f}-{pb_max_pct:.1f}% para esta etapa. "
                "No mejora performance, solo aumenta costo y excreción de N."
            )
    else:
        # Sin rango: usar criterio NASEM puro (95% de cobertura)
        if pb_aportada < pb_g_dia * 0.95:
            deficiencias.append({
                "nutriente": "Proteína bruta",
                "requerido": pb_g_dia, "max_alcanzable": pb_aportada,
                "deficit_pct": (1 - pb_aportada / pb_g_dia) * 100,
            })
            sugerencias.append(
                "Subir proporción de fuente proteica (Pellet de soja, Expeller, "
                "Burlanda DDGS) o agregar Urea (NNP, máx 1.5%)."
            )
    if em_aportada < em_mcal_dia * 0.95:
        deficiencias.append({
            "nutriente": "Energía Metabolizable",
            "requerido": em_mcal_dia, "max_alcanzable": em_aportada,
            "deficit_pct": (1 - em_aportada / em_mcal_dia) * 100,
        })
        sugerencias.append("Subir proporción de grano (Maíz, Cebada).")
    if fdn_aportada < fdn_min_pct - 1:
        deficiencias.append({
            "nutriente": "FDN (fibra)",
            "requerido": fdn_min_pct, "max_alcanzable": fdn_aportada,
            "deficit_pct": (fdn_min_pct - fdn_aportada),
        })
        sugerencias.append(
            f"⚠️ Riesgo de acidosis: FDN actual {fdn_aportada:.1f}% < mínimo "
            f"{fdn_min_pct}%. Agregar silaje, heno o más Fibrogreen."
        )
    if ca_aportado < ca_g_dia * 0.95:
        deficiencias.append({
            "nutriente": "Calcio",
            "requerido": ca_g_dia, "max_alcanzable": ca_aportado,
            "deficit_pct": (1 - ca_aportado / ca_g_dia) * 100,
        })
        sugerencias.append("Agregar Carbonato de calcio o Conchilla molida.")
    if p_aportado < p_g_dia * 0.95:
        deficiencias.append({
            "nutriente": "Fósforo",
            "requerido": p_g_dia, "max_alcanzable": p_aportado,
            "deficit_pct": (1 - p_aportado / p_g_dia) * 100,
        })
        sugerencias.append("Agregar Fosfato dicálcico o Núcleo mineral con P.")

    factible = len(deficiencias) == 0
    if factible:
        msg = "✅ La receta cumple los requerimientos NASEM 2016."
    else:
        msg = f"⚠️ La receta NO cubre {len(deficiencias)} requerimiento(s)."

    consumo_tc = sum(c["kg_tal_cual"] for c in composicion)
    return FormulacionResultado(
        factible=factible,
        mensaje=msg,
        costo_total_dia=costo_total,
        costo_por_kg_ms=costo_total / consumo_ms_kg if consumo_ms_kg else 0,
        consumo_ms_kg=consumo_ms_kg,
        consumo_tal_cual_kg=consumo_tc,
        composicion=composicion,
        pb_aportado_g=pb_aportada, pb_requerido_g=pb_g_dia,
        em_aportado_mcal=em_aportada, em_requerido_mcal=em_mcal_dia,
        fdn_aportado_pct=fdn_aportada,
        ca_aportado_g=ca_aportado, p_aportado_g=p_aportado,
        nnp_aportado_pct=nnp_pct_dieta,
        nnp_pb_equivalente_g=nnp_pb_equiv,
        pb_verdadera_g=pb_verdadera,
        deficiencias=deficiencias, sugerencias=sugerencias,
        advertencias=advertencias,
    )


# =====================================================================
# 5) DIAGNÓSTICO DE INFACTIBILIDAD
# =====================================================================

def _diagnosticar_infactible(
    disponibles: List[Ingrediente],
    consumo_ms: float,
    pb_g: float,
    em_mcal: float,
    fdn_min_pct: float,
    ca_g: float,
    p_g: float,
    todos_ingredientes: List[Ingrediente],
) -> FormulacionResultado:
    """Si LP es infactible, evaluar qué nutriente NO se puede cubrir y
    sugerir ingredientes que aporten ese nutriente."""

    # Calcular máximos posibles de cada nutriente con los disponibles
    pb_max = sum(ing.pb_pct_ms * 10 * (ing.max_inclusion_pct_ms / 100 * consumo_ms)
                  for ing in disponibles)
    em_max = sum(ing.em_mcal_kg_ms * (ing.max_inclusion_pct_ms / 100 * consumo_ms)
                  for ing in disponibles)
    ca_max = sum(ing.ca_pct_ms * 10 * (ing.max_inclusion_pct_ms / 100 * consumo_ms)
                  for ing in disponibles)
    p_max = sum(ing.p_pct_ms * 10 * (ing.max_inclusion_pct_ms / 100 * consumo_ms)
                 for ing in disponibles)

    deficiencias = []
    sugerencias = []
    nombres_disp = {i.nombre for i in disponibles}

    if pb_max < pb_g:
        deficiencias.append({
            "nutriente": "Proteína bruta",
            "requerido": pb_g, "max_alcanzable": pb_max,
            "deficit_pct": (1 - pb_max / pb_g) * 100,
        })
        # Sugerir ingredientes con mayor PB que NO estén disponibles
        candidatos = sorted(
            [i for i in todos_ingredientes
             if i.nombre not in nombres_disp and i.pb_pct_ms > 25],
            key=lambda x: -x.pb_pct_ms,
        )
        for c in candidatos[:3]:
            sugerencias.append(
                f"Agregar {c.nombre} (PB {c.pb_pct_ms:.0f}% MS) "
                f"para cubrir el faltante de proteína."
            )

    if em_max < em_mcal:
        deficiencias.append({
            "nutriente": "Energía Metabolizable",
            "requerido": em_mcal, "max_alcanzable": em_max,
            "deficit_pct": (1 - em_max / em_mcal) * 100,
        })
        candidatos = sorted(
            [i for i in todos_ingredientes
             if i.nombre not in nombres_disp and i.em_mcal_kg_ms > 2.7],
            key=lambda x: -x.em_mcal_kg_ms,
        )
        for c in candidatos[:3]:
            sugerencias.append(
                f"Agregar {c.nombre} (EM {c.em_mcal_kg_ms:.2f} Mcal/kg MS) "
                f"para subir la densidad energética."
            )

    if ca_max < ca_g:
        deficiencias.append({
            "nutriente": "Calcio",
            "requerido": ca_g, "max_alcanzable": ca_max,
            "deficit_pct": (1 - ca_max / ca_g) * 100,
        })
        sugerencias.append(
            "Agregar Carbonato de calcio (Ca 38%), Conchilla molida o Heno de alfalfa."
        )

    if p_max < p_g:
        deficiencias.append({
            "nutriente": "Fósforo",
            "requerido": p_g, "max_alcanzable": p_max,
            "deficit_pct": (1 - p_max / p_g) * 100,
        })
        sugerencias.append(
            "Agregar Fosfato dicálcico (P 18.5%) o aumentar el Núcleo mineral con Fósforo."
        )

    if not deficiencias:
        # No hay déficit pero LP igual es infactible: probable conflicto entre
        # mínimos y máximos
        return FormulacionResultado(
            factible=False,
            mensaje=(
                "⚠️ Conflicto entre los porcentajes mínimo y máximo de inclusión. "
                "Revisá que la suma de los mínimos no exceda el 100%."
            ),
            sugerencias=["Reducir los % mínimos de inclusión de algún ingrediente."],
        )

    msg = "🔴 No se pueden cubrir los requerimientos con los ingredientes disponibles."
    return FormulacionResultado(
        factible=False, mensaje=msg,
        deficiencias=deficiencias, sugerencias=sugerencias,
    )
