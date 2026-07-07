"""Catálogo de acciones preventivas y de mitigación por tipo de
evento climático + severidad + categoría animal.

Devuelve listas cortas y concretas para mostrar en la UI junto al
impacto productivo proyectado. Son acciones REALISTAS para la Pampa
Húmeda argentina: nada de cubrir comederos con lona, construir techos,
ni acciones nocturnas (criterio alineado con los system prompts del
LLM).

Uso:
    from src.prevencion import acciones_preventivas

    acciones = acciones_preventivas(
        tipo_evento="frio", severidad="critico",
        categoria="vaquillona", barro=True,
    )
    # → lista de dicts con {icono, titulo, detalle}
"""
from __future__ import annotations

from typing import Dict, List, Optional


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


# Acciones BASE por tipo de evento (aplican a todas las categorías,
# se adaptan según severidad y agravantes).
_BASE_FRIO = [
    {
        "icono": "🌳",
        "titulo": "Cortar el viento",
        "detalle": "Si tenés monte, cortina forestal o rollos "
                   "apilados como cortaviento, verificar que "
                   "el lote tenga acceso libre. Si NO tenés reparo "
                   "natural, hay dos palancas: (1) trasladar el lote "
                   "a un potrero con mejor relieve (loma, hondonada, "
                   "alambrado con vegetación) que corte el viento "
                   "dominante, y (2) aumentar densidad del lote para "
                   "que los animales se den calor mutuo agrupándose. "
                   "Como plan a mediano plazo, considerar plantar "
                   "monte en el sector de invernada — el reparo es "
                   "lo que más impacta en frío con viento.",
    },
    {
        "icono": "💧",
        "titulo": "Agua: limpia, sin hielo y accesible",
        "detalle": "En el campo no se calienta el agua, pero sí hay "
                   "que cuidar los <strong>cuatro factores</strong> "
                   "que determinan si el animal toma o no:<br>"
                   "<strong>1. Limpieza:</strong> bebedero sin "
                   "biofilm, algas, restos de mezcla ni hojas — "
                   "agua sucia hace que el animal rechace tomar y "
                   "automáticamente baja el consumo de materia seca."
                   "<br>"
                   "<strong>2. Sin hielo:</strong> romper la costra "
                   "de la superficie con palo o pala apenas amanece. "
                   "Agua helada obliga al cuerpo a gastar energía "
                   "calentándola antes de absorberla."
                   "<br>"
                   "<strong>3. Caudal:</strong> revisar que el "
                   "flotante y la cañería no se hayan congelado; si "
                   "el bebedero no se rellena rápido cuando varios "
                   "animales toman juntos, los subordinados se "
                   "quedan sin agua."
                   "<br>"
                   "<strong>4. Accesibilidad:</strong> sin barro "
                   "profundo alrededor del bebedero ni acumulación "
                   "de dominantes bloqueando el acceso. Un animal "
                   "que duda al acercarse, salta el turno y termina "
                   "tomando menos."
                   "<br><br>"
                   "<strong>Por qué importa:</strong> un animal sin "
                   "agua deja de comer. La relación agua/MS es 3:1 "
                   "a 4:1 — si cae el consumo de agua, cae "
                   "automáticamente el consumo de mezcla, y con eso "
                   "todo el aumento esperado.",
    },
    {
        "icono": "🍽️",
        "titulo": "Adelantar la última comida",
        "detalle": "Si das comidas estructuradas, adelantar la carga "
                   "de la tarde para concentrar consumo antes del pico "
                   "de frío nocturno. Comer antes del frío genera "
                   "fermentación ruminal activa que produce calor "
                   "metabólico interno.",
    },
    {
        "icono": "🌾",
        "titulo": "Fibra efectiva",
        "detalle": "Aumentar disponibilidad de fibra física por 3-4 "
                   "días: subir 1-2 puntos en mezcla, ofrecer rollo a "
                   "discreción o fardo al comedero. La fibra estimula "
                   "rumia y la rumia produce saliva, el principal "
                   "buffer natural del rumen.",
    },
    {
        "icono": "🛏️",
        "titulo": "Superficie seca para descansar",
        "detalle": "Priorizar zonas drenadas para el descanso "
                   "nocturno. La cama mojada o el barro conducen "
                   "calor del animal al suelo de forma directa — "
                   "echado en barro pierde más calor por la panza "
                   "que parado bajo viento.",
    },
]

_BASE_CALOR = [
    {
        "icono": "🌳",
        "titulo": "Sombra disponible",
        "detalle": "Verificar acceso a sombra natural (monte, árboles) "
                   "o existente. Sin sombra el animal acumula carga "
                   "térmica y deja de comer en las horas centrales.",
    },
    {
        "icono": "💦",
        "titulo": "Agua fresca y abundante",
        "detalle": "Asegurar suministro continuo, caudal alto y agua "
                   "lo más fresca posible. Con calor el consumo de "
                   "agua se duplica; un bebedero chico se vacía y el "
                   "animal corta el consumo.",
    },
    {
        "icono": "🌅",
        "titulo": "Manejos en horas frescas",
        "detalle": "Cualquier manejo (vacunación, traslado, mezcla "
                   "nueva) hacerlo antes de las 10 AM. Manejar al "
                   "mediodía suma estrés térmico y dispara el riesgo "
                   "respiratorio.",
    },
    {
        "icono": "🍽️",
        "titulo": "Cargar mezcla más temprano",
        "detalle": "Adelantar la carga del comedero a primera hora y "
                   "última hora del día. El animal va a comer en "
                   "horarios frescos; mezcla del mediodía se calienta "
                   "y baja palatabilidad.",
    },
]


# Acciones EXTRA según severidad (se suman a las base).
_EXTRA_CRITICO_FRIO = [
    {
        "icono": "👀",
        "titulo": "Monitoreo de rumia al amanecer",
        "detalle": "Observar 15-30 minutos al amanecer: idealmente "
                   ">50% del lote rumiando (mascado rítmico con boca "
                   "cerrada). Si baja, hay desorden ruminal en curso "
                   "y conviene anticipar el ajuste.",
    },
    {
        "icono": "⚠️",
        "titulo": "No sumar grano de golpe",
        "detalle": "El instinto es subir concentrado por el gasto "
                   "extra, pero con rumen alterado un salto de almidón "
                   "dispara acidosis subclínica. Sostener nivel actual "
                   "y trabajar sobre fibra.",
    },
]

_EXTRA_OPERATIVO_FRIO = [
    {
        "icono": "🍽️",
        "titulo": "Revisar el comedero en el primer recorrido",
        "detalle": (
            "Al primer recorrido del día (entre 7 y 9 AM, según "
            "rutina), pasá por el comedero y observá <strong>tres "
            "cosas</strong>:<br>"
            "<strong>1.</strong> <em>¿Cuánta mezcla quedó?</em> Si "
            "quedó mucha del día anterior, comieron poco de noche "
            "por el frío — esa es la primera señal de que están "
            "engordando menos por día, antes incluso de verlo en "
            "balanza.<br>"
            "<strong>2.</strong> <em>¿Cómo está la mezcla?</em> Si "
            "tiene costra, está helada arriba o se ve mojada, la "
            "palatabilidad cayó: el animal selecciona, deja la parte "
            "fea y consume menos energía total aunque parezca que "
            "comió.<br>"
            "<strong>3.</strong> <em>¿Dónde está el lote?</em> Si "
            "siguen echados o agrupados en el reparo en lugar de "
            "estar en el comedero, todavía no entraron al ritmo del "
            "día — eso suma horas de rumen vacío y le baja eficiencia "
            "al lote.<br><br>"
            "<strong>Qué hacer según lo que ves:</strong> si quedó "
            "mucha mezcla, NO cargar comida nueva encima — dejar "
            "consumir la que está y atrasar la próxima carga. Si la "
            "mezcla viene perdiendo palatabilidad por la humedad "
            "nocturna, ajustar el horario de la carga de la tarde "
            "para que el lote la consuma antes del rocío. Dos o tres "
            "días seguidos de \"comer mal de noche\" se traducen en "
            "que el lote engorda menos por día aunque la dieta esté "
            "bien armada — el efecto del frío no se ve al mediodía, "
            "se ve al amanecer."
        ),
    },
]


# Acciones EXTRA si hay barro.
_EXTRA_BARRO = [
    {
        "icono": "🛤️",
        "titulo": "Acceso al comedero sin barreras",
        "detalle": "Nivelar zonas con barro profundo en el acceso. "
                   "Cuando el animal duda al ir a comer, salta "
                   "comidas y se altera el patrón de ingesta — los "
                   "horarios de fermentación dejan de ser estables.",
    },
]


# Acciones EXTRA por categoría sensible.
_EXTRA_TERNERO = [
    {
        "icono": "🐂",
        "titulo": "Densidad bajo reparo",
        "detalle": "Verificar que la zona resguardada tenga espacio "
                   "para todo el lote. Los terneros se agrupan con "
                   "frío; si el lugar es chico, los dominantes acaparan "
                   "y los dominados quedan al viento perdiendo "
                   "condición sin que se note al ojo del lote.",
    },
]


def acciones_preventivas(
    tipo_evento: str,
    severidad: str = "operativo",
    categoria: str = "",
    barro: bool = False,
    pelaje_mojado: bool = False,
) -> List[Dict]:
    """Devuelve la lista de acciones preventivas/de mitigación
    aplicables al evento.

    Args:
        tipo_evento: "frio" o "calor".
        severidad: "atencion" | "operativo" | "critico".
        categoria: categoría del lote (para sumar extras específicos).
        barro: si hay barro confirmado, suma acciones extra.
        pelaje_mojado: si HR alta o lluvia, refuerza acciones.

    Returns:
        Lista de dicts con keys: icono, titulo, detalle.
    """
    tipo = (tipo_evento or "frio").lower()
    sev = (severidad or "operativo").lower()
    cat_norm = _normalizar_categoria(categoria)

    acciones: List[Dict] = []
    if tipo == "frio":
        acciones.extend(_BASE_FRIO)
        if sev == "critico":
            acciones.extend(_EXTRA_CRITICO_FRIO)
        elif sev == "operativo":
            acciones.extend(_EXTRA_OPERATIVO_FRIO)
        if barro or pelaje_mojado:
            acciones.extend(_EXTRA_BARRO)
        if cat_norm == "ternero":
            acciones.extend(_EXTRA_TERNERO)
    elif tipo == "calor":
        acciones.extend(_BASE_CALOR)

    # Para nivel "atención", devolver solo las 3 más importantes
    # (no saturar al productor con 7 acciones cuando el evento es leve).
    if sev == "atencion" and len(acciones) > 3:
        acciones = acciones[:3]

    return acciones


def acciones_preventivas_html(
    tipo_evento: str,
    severidad: str = "operativo",
    categoria: str = "",
    barro: bool = False,
    pelaje_mojado: bool = False,
) -> str:
    """Versión HTML lista para inyectar en Streamlit."""
    items = acciones_preventivas(
        tipo_evento, severidad, categoria, barro, pelaje_mojado,
    )
    if not items:
        return ""
    bullets = []
    for a in items:
        bullets.append(
            f'<li style="margin-bottom:8px;">'
            f'<span style="font-weight:600; color:#1B3E27;">'
            f'{a["icono"]} {a["titulo"]}:</span> '
            f'<span style="color:#3a3a3a;">{a["detalle"]}</span>'
            f'</li>'
        )
    return (
        '<ul style="margin:6px 0 4px 0; padding-left:20px;'
        ' line-height:1.55; font-size:12.5px;">'
        + "".join(bullets) +
        "</ul>"
    )
