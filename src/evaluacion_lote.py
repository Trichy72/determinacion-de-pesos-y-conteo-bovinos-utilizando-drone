"""Módulo de evaluación estructurada del lote durante una conversación
con el cliente.

Transforma una llamada/WhatsApp en un diagnóstico técnico:
1. Arma las preguntas relevantes según el cliente (lotes activos,
   tipo de comedero, dieta vigente, etc.).
2. Analiza las respuestas y sugiere acciones concretas.
3. Cruza el stock declarado contra el consumo esperado por la
   dieta — detecta sub-uso, sobre-uso o desbalance de proporciones.

El output se guarda como markdown legible en notas_cierre del
recordatorio, así el asesor lo ve sin parseo y queda en el
historial del cliente.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# =====================================================================
# OPCIONES DE RESPUESTA (se usan en los selectbox del form)
# =====================================================================

OPCIONES_ASPECTO_ANIMALES = [
    "🟢 Brillantes / activos / pelos lustrosos",
    "🟡 Normales — sin destacar",
    "🟠 Apagados / pelos hirsutos",
    "🔴 Muy decaídos / temblando / huesos marcados",
    "❔ No se evaluó",
]

OPCIONES_COMEDERO = [
    "🔴 Vacío (animales esperando)",
    "🟢 Queda algo (consumo normal)",
    "🟡 Sobra cantidad (oferta excesiva)",
    "🟠 Sobra TODO (no están comiendo)",
    "❔ No se evaluó",
]

OPCIONES_HECES = [
    "🟢 Firmes / normales",
    "🟡 Pastosas",
    "🔴 Líquidas / diarrea",
    "🟠 Muy secas (poco agua)",
    "❔ No se evaluó",
]

OPCIONES_AGUA = [
    "🟢 OK — limpia, caudal normal",
    "🧊 Hielo en la mañana",
    "🟫 Sucia / con barro",
    "🔴 Sin agua o caudal bajo",
    "❔ No se evaluó",
]

OPCIONES_CAMA = [
    "🟢 Seca",
    "🟡 Húmeda",
    "🟠 Embarrada",
    "🔴 Sin cama / piso de tierra mojado",
    "❔ No aplica / no tiene cama",
]

OPCIONES_REPAROS = [
    "🟢 OK — protegidos del viento",
    "🟡 Parciales — falta refuerzo",
    "🔴 Insuficientes — animales expuestos",
    "❔ No aplica / no se evaluó",
]

OPCIONES_CAUSA_MUERTE = [
    "🟫 Acidosis",
    "🫁 Neumonía / respiratoria",
    "💨 Timpanismo / meteorismo",
    "🧊 Hipotermia / estrés por frío",
    "🌡️ Estrés calórico",
    "🤕 Politraumatismo / accidente",
    "🦠 Infecciosa / diarrea",
    "❔ Sin determinar / pendiente necropsia",
    "📝 Otra (especificar en detalle)",
]


OPCIONES_SILO_NIVEL = [
    "100% — lleno (acaba de cargarse)",
    "75% — tres cuartos",
    "50% — mitad",
    "25% — un cuarto",
    "10% — casi vacío",
    "0% — vacío / hay que cargar",
    "❔ No se sabe",
]


# =====================================================================
# ESTRUCTURA DE RESPUESTAS
# =====================================================================

@dataclass
class RespuestasEvaluacion:
    """Estructura tipada de las respuestas del cuestionario."""
    # Identificación
    cliente_nombre: str = ""
    lote_id: Optional[int] = None
    lote_identificador: str = ""
    tipo_contacto: str = ""
    atendio: str = ""

    # Estado de los animales
    aspecto_animales: str = ""
    bajas_48hs: int = 0
    causa_muerte: str = ""  # Una de OPCIONES_CAUSA_MUERTE
    animales_enfermos: int = 0
    # Cambios de cantidad del lote desde el último contacto.
    # Se registran AUTOMÁTICAMENTE como movimientos en la
    # ficha del lote al guardar la evaluación, así no tenés
    # que ir a Movimientos de hacienda por separado.
    ventas_48hs: int = 0
    kg_promedio_ventas: float = 0.0
    detalle_movimientos: str = ""

    # Consumo y rumen
    estado_comedero: str = ""
    heces: str = ""

    # Ambiente
    estado_agua: str = ""
    estado_cama: str = ""
    estado_reparos: str = ""

    # Stock declarado
    maiz_kg_disponible: float = 0.0
    fibrogreen_kg_disponible: float = 0.0
    rollos_disponibles: int = 0
    silo_nivel_pct: int = -1  # -1 = no se sabe
    dias_desde_ultima_carga: int = -1
    kg_ultima_carga: float = 0.0

    # Dieta REAL que aplica el cliente (según entrevista).
    # El sistema compara contra la dieta formulada vigente del lote
    # para detectar subdosis, exceso o ingredientes faltantes.
    #   - dieta_real_modo:
    #       "animal_dia" → cada item.kg = kg por animal por día
    #       "total_dia"  → cada item.kg = kg totales del mixer por día
    #                      (el sistema divide por cantidad de cabezas)
    #       ""           → no se cargó (no hacer comparación)
    #   - dieta_real_items: lista de {nombre: str, kg: float}
    dieta_real_modo: str = ""
    dieta_real_items: List[Dict] = field(default_factory=list)

    # Texto libre
    observaciones: str = ""
    acciones_acordadas: str = ""


# =====================================================================
# ANÁLISIS — sugerencias de acción
# =====================================================================

@dataclass
class Sugerencia:
    severidad: str  # "info" | "atencion" | "urgente"
    titulo: str
    detalle: str
    icono: str = ""


def _icono_sev(sev: str) -> str:
    return {"info": "💡", "atencion": "🟡", "urgente": "🔴"}.get(
        sev, "•",
    )


def analizar_evaluacion(
    r: RespuestasEvaluacion,
    contexto_lote: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Genera sugerencias basadas en las respuestas + contexto del lote.

    Args:
        r: respuestas del cuestionario.
        contexto_lote: dict opcional con info del lote desde la DB
            (dieta vigente, consumo esperado, fecha última carga
            registrada, etc.) para cruces más finos.

    Returns:
        dict con:
        - sugerencias: List[Sugerencia]
        - alertas_cruce: List[str] (chequeos stock vs dieta)
        - resumen_estado: str (semáforo global)
    """
    sugerencias: List[Sugerencia] = []
    alertas_cruce: List[str] = []

    # ── 1. Animales ──
    if "Muy decaídos" in r.aspecto_animales:
        sugerencias.append(Sugerencia(
            severidad="urgente",
            titulo="Animales muy decaídos",
            detalle=(
                "Revisar urgente el corral. Posibles causas: estrés "
                "térmico, deshidratación, problema sanitario "
                "(neumonía, acidosis), sub-alimentación. Coordinar "
                "visita a campo o video del lote en 24hs."
            ),
        ))
    elif "Apagados" in r.aspecto_animales:
        sugerencias.append(Sugerencia(
            severidad="atencion",
            titulo="Animales apagados",
            detalle=(
                "Monitorear durante 48 hs. Chequear consumo, agua "
                "y reparos. Si no mejora, pedir foto y revisar "
                "dieta + ambiente."
            ),
        ))

    if r.bajas_48hs > 0:
        _sev = "urgente" if r.bajas_48hs >= 2 else "atencion"
        # Recomendaciones específicas según la causa
        _causa = r.causa_muerte or ""
        if "Acidosis" in _causa:
            _detalle_baja = (
                "ACIDOSIS: revisar URGENTE proporción de Fibrogreen "
                "Plus en la mezcla actual. Verificar consumo "
                "desparejo (animal hambreado que come golpe), "
                "homogeneidad de la mezcla en el comedero y "
                "frecuencia de carga. Considerar subir FG 1-2 "
                "puntos porcentuales por 3-5 días. Chequear si hay "
                "selección de grano fino en el silo."
            )
        elif "Neumonía" in _causa:
            _detalle_baja = (
                "NEUMONÍA: típica de transición con frío + barro + "
                "viento. Revisar reparos, drenaje del corral, cama "
                "seca. Identificar animales tosiendo / con secreción "
                "nasal para tratamiento veterinario. Considerar "
                "preventivo si hay clima adverso sostenido."
            )
        elif "Timpanismo" in _causa:
            _detalle_baja = (
                "TIMPANISMO: típicamente por adaptación al concentrado "
                "demasiado rápida o mezcla con grano fino + poca "
                "fibra efectiva. Revisar la fase actual del plan de "
                "adaptación — puede haber avanzado antes de tiempo. "
                "Sumar rollo a libre disposición SÍ o SÍ por 5 días."
            )
        elif "Hipotermia" in _causa or "frío" in _causa.lower():
            _detalle_baja = (
                "HIPOTERMIA / FRÍO: animal sin reservas o sin reparo. "
                "Confirmar cama seca + reparo viento. Si es categoría "
                "joven (ternero) o animal flaco, mayor riesgo. "
                "Considerar suplementación energética extra (+1% PV "
                "de concentrado) durante el evento de frío."
            )
        elif "calórico" in _causa.lower() or "calor" in _causa.lower():
            _detalle_baja = (
                "ESTRÉS CALÓRICO: revisar acceso a agua FRESCA y "
                "sombra. Considerar cambio de horario de oferta de "
                "concentrado (más temprano + más tarde, evitar "
                "mediodía). Si vienen más días de calor, manejar "
                "anticipadamente."
            )
        elif "Politraumatismo" in _causa:
            _detalle_baja = (
                "POLITRAUMATISMO: revisar instalaciones (mangas, "
                "alambrados, comederos), densidad animal, dominancia "
                "social. Si fue durante manejo, revisar protocolo."
            )
        elif "Infecciosa" in _causa or "diarrea" in _causa.lower():
            _detalle_baja = (
                "INFECCIOSA / DIARREA: identificar afectados y "
                "aislar. Pedir muestreo veterinario para identificar "
                "agente. Revisar agua (contaminación), barro en "
                "corral. Posible necesidad de tratamiento masivo "
                "preventivo."
            )
        elif "Sin determinar" in _causa or not _causa:
            _detalle_baja = (
                "CAUSA SIN DETERMINAR: pedir al cliente que **haga "
                "necropsia** o saque fotos del cuadro general "
                "(corral, cama, comedero, agua). Sin diagnóstico "
                "no se puede prevenir el próximo. Las 3 causas más "
                "frecuentes son acidosis, neumonía y timpanismo — "
                "guiar el diagnóstico por descarte."
            )
        else:
            _detalle_baja = (
                f"Causa reportada: {_causa}. Documentar el cuadro "
                "para análisis. Si se repite el patrón en 7-10 días, "
                "puede indicar un problema estructural del manejo."
            )
        sugerencias.append(Sugerencia(
            severidad=_sev,
            titulo=(
                f"{r.bajas_48hs} muerte(s) en 48 hs"
                + (f" — {_causa}" if _causa else "")
            ),
            detalle=_detalle_baja,
        ))

    if r.animales_enfermos > 0:
        sugerencias.append(Sugerencia(
            severidad="atencion",
            titulo=f"{r.animales_enfermos} animal(es) enfermo(s)",
            detalle=(
                "Aislar si corresponde. Confirmar tratamiento "
                "veterinario en curso. Documentar evolución para "
                "próximo control."
            ),
        ))

    # ── 2. Consumo ──
    if "Vacío" in r.estado_comedero:
        sugerencias.append(Sugerencia(
            severidad="atencion",
            titulo="Comedero vacío — oferta corta",
            detalle=(
                "Subir oferta diaria 5-10% (kg t/c) o sumar una "
                "carga adicional. Animal hambreado tiende a comer "
                "más rápido cuando llega comida → riesgo de "
                "acidosis. Revisar también frecuencia de carga."
            ),
        ))
    elif "Sobra cantidad" in r.estado_comedero:
        sugerencias.append(Sugerencia(
            severidad="atencion",
            titulo="Sobra mezcla en el comedero",
            detalle=(
                "Reducir oferta 5-10%. Posibles causas adicionales: "
                "clima caluroso, mezcla mal mezclada (selección), "
                "agua sucia, fibra fría/mojada. Revisar."
            ),
        ))
    elif "Sobra TODO" in r.estado_comedero:
        sugerencias.append(Sugerencia(
            severidad="urgente",
            titulo="Animal no consume — alerta sanitaria",
            detalle=(
                "Caída brusca del consumo es un síntoma serio. "
                "Causas posibles: acidosis sub-clínica, neumonía, "
                "mezcla en mal estado, agua contaminada, ambiente "
                "muy adverso. Pedir visita o foto inmediata."
            ),
        ))

    if "diarrea" in r.heces.lower() or "Líquidas" in r.heces:
        sugerencias.append(Sugerencia(
            severidad="urgente",
            titulo="Heces líquidas / diarrea",
            detalle=(
                "Posible acidosis o problema digestivo. Revisar: "
                "(1) carga de Fibrogreen Plus, ¿está cumpliendo el "
                "porcentaje de la dieta? (2) ¿está habiendo "
                "selección del grano? (3) frecuencia de carga "
                "del silo. Considerar subir fibra 1-2 puntos "
                "porcentuales por 3 días."
            ),
        ))
    elif "Pastosas" in r.heces:
        sugerencias.append(Sugerencia(
            severidad="atencion",
            titulo="Heces pastosas",
            detalle=(
                "Está en el límite. Monitorear evolución. Chequear "
                "homogeneidad de la mezcla y consumo de rollo."
            ),
        ))
    elif "Muy secas" in r.heces:
        sugerencias.append(Sugerencia(
            severidad="atencion",
            titulo="Heces muy secas",
            detalle=(
                "Indica bajo consumo de agua. Revisar bebederos: "
                "caudal, limpieza, temperatura del agua, distancia "
                "al comedero."
            ),
        ))

    # ── 3. Ambiente ──
    if "Hielo" in r.estado_agua or "Sin agua" in r.estado_agua:
        sugerencias.append(Sugerencia(
            severidad="urgente",
            titulo=f"Problema con el agua: {r.estado_agua}",
            detalle=(
                "Agua es el insumo crítico. Sin agua o con hielo "
                "el consumo de MS cae 20-30%. Asegurar romper "
                "hielo 2x/día y proteger del viento con reparo."
            ),
        ))
    elif "Sucia" in r.estado_agua:
        sugerencias.append(Sugerencia(
            severidad="atencion",
            titulo="Agua sucia / con barro",
            detalle=(
                "Limpieza inmediata del bebedero. Bovinos rechazan "
                "agua sucia → baja consumo de MS."
            ),
        ))

    if "Sin cama" in r.estado_cama or "Embarrada" in r.estado_cama:
        sugerencias.append(Sugerencia(
            severidad="atencion",
            titulo="Zona de descanso comprometida",
            detalle=(
                "Cama embarrada o sin cama = pérdida de calor "
                "por conducción + estrés. Sumar paja seca "
                "(mínimo 10 cm) en zona de descanso. Verificar "
                "drenaje del corral."
            ),
        ))

    if "Insuficientes" in r.estado_reparos:
        sugerencias.append(Sugerencia(
            severidad="atencion",
            titulo="Reparos insuficientes",
            detalle=(
                "Instalar cortavientos en lado oeste/suroeste. "
                "Solución práctica: fardos o rollos apilados en "
                "'L' como reparo temporal. Prioridad alta si "
                "vienen días fríos."
            ),
        ))

    # ── 4. STOCK Y CRUCE CON DIETA ──
    # Este es el corazón del análisis: cruzar lo declarado con
    # lo esperado por la dieta.
    if contexto_lote:
        dieta = contexto_lote.get("dieta_vigente") or {}
        kg_ms_an_dia = float(dieta.get("consumo_ms_kg") or 0)
        cant_an = int(contexto_lote.get("cantidad_inicial") or 0)
        # Aproximación: consumo tal cual ≈ MS / 0.88 (88% MS)
        kg_tc_an_dia = kg_ms_an_dia / 0.88 if kg_ms_an_dia else 0
        consumo_lote_kg_dia = kg_tc_an_dia * cant_an

        # Stock declarado total
        stock_total_kg = (
            r.maiz_kg_disponible + r.fibrogreen_kg_disponible
        )

        # Días que cubre lo declarado
        if consumo_lote_kg_dia > 0 and stock_total_kg > 0:
            dias_cubre = stock_total_kg / consumo_lote_kg_dia
            alertas_cruce.append(
                f"📊 **Autonomía estimada con stock declarado**: "
                f"≈ {dias_cubre:.1f} días "
                f"({stock_total_kg:.0f} kg ÷ "
                f"{consumo_lote_kg_dia:.0f} kg/día del lote)."
            )
            if dias_cubre < 7:
                sugerencias.append(Sugerencia(
                    severidad="urgente",
                    titulo=(
                        f"Stock alcanza solo {dias_cubre:.0f} días"
                    ),
                    detalle=(
                        "Coordinar entrega urgente. Calcular kg a "
                        "reponer para llegar al fin de ciclo o "
                        "próxima carga programada."
                    ),
                ))
            elif dias_cubre < 14:
                sugerencias.append(Sugerencia(
                    severidad="atencion",
                    titulo=(
                        f"Stock para {dias_cubre:.0f} días"
                    ),
                    detalle=(
                        "Programar entrega en los próximos días "
                        "para no quedar sin margen."
                    ),
                ))

        # Cruce de proporciones (mezcla del silo)
        comp = dieta.get("composicion") or []
        if comp and r.maiz_kg_disponible > 0:
            # Buscar maíz y FG en composición
            kg_ms_maiz = 0.0
            kg_ms_fg = 0.0
            for c in comp:
                nom = (c.get("nombre") or "").lower()
                kg_ms = float(c.get("kg_ms") or 0)
                if "maíz" in nom or "maiz" in nom:
                    kg_ms_maiz += kg_ms
                elif "fibrogreen" in nom or "fibroter" in nom:
                    kg_ms_fg += kg_ms
            if kg_ms_maiz > 0:
                # Proporción esperada FG/maíz en la dieta
                ratio_esperado = (
                    kg_ms_fg / kg_ms_maiz if kg_ms_fg else 0
                )
                ratio_declarado = (
                    r.fibrogreen_kg_disponible
                    / r.maiz_kg_disponible
                    if r.maiz_kg_disponible else 0
                )
                if ratio_esperado > 0 and ratio_declarado >= 0:
                    desvio_pct = (
                        (ratio_declarado - ratio_esperado)
                        / ratio_esperado * 100
                    )
                    alertas_cruce.append(
                        f"🧮 **Proporción FG / maíz en stock**: "
                        f"declarado {ratio_declarado*100:.1f}% "
                        f"vs dieta {ratio_esperado*100:.1f}% "
                        f"(desvío {desvio_pct:+.0f}%)."
                    )
                    if abs(desvio_pct) > 30:
                        if desvio_pct < 0:
                            sugerencias.append(Sugerencia(
                                severidad="urgente",
                                titulo=(
                                    "FALTA Fibrogreen vs proporción "
                                    "de la dieta"
                                ),
                                detalle=(
                                    "El stock de FG es bajo respecto "
                                    "al maíz. Si está cargando el "
                                    "silo con esta proporción → "
                                    "DIETA ACIDÓGENA en campo. "
                                    "Coordinar entrega urgente de FG."
                                ),
                            ))
                        else:
                            sugerencias.append(Sugerencia(
                                severidad="atencion",
                                titulo=(
                                    "Sobra Fibrogreen vs maíz"
                                ),
                                detalle=(
                                    "Buen colchón de FG pero falta "
                                    "maíz. Coordinar entrega de "
                                    "maíz para mantener la fórmula."
                                ),
                            ))

        # Cruce consumo real vs esperado
        if (r.dias_desde_ultima_carga > 0
                and r.kg_ultima_carga > 0
                and r.silo_nivel_pct >= 0
                and consumo_lote_kg_dia > 0):
            # Cuánto debería quedar según consumo esperado
            consumido_esperado = (
                consumo_lote_kg_dia * r.dias_desde_ultima_carga
            )
            queda_esperado = max(
                0, r.kg_ultima_carga - consumido_esperado
            )
            # Cuánto declara el cliente
            queda_declarado = (
                r.kg_ultima_carga * (r.silo_nivel_pct / 100)
            )
            if r.kg_ultima_carga > 0:
                desvio_kg = queda_declarado - queda_esperado
                desvio_pct_silo = (
                    desvio_kg / r.kg_ultima_carga * 100
                )
                alertas_cruce.append(
                    f"🛢️ **Consumo real vs esperado** "
                    f"(últimos {r.dias_desde_ultima_carga}d): "
                    f"queda en silo ≈ {queda_declarado:.0f} kg "
                    f"(declarado {r.silo_nivel_pct}%) · esperado "
                    f"{queda_esperado:.0f} kg · "
                    f"desvío {desvio_kg:+.0f} kg ({desvio_pct_silo:+.0f}%)."
                )
                if desvio_pct_silo < -15:
                    sugerencias.append(Sugerencia(
                        severidad="atencion",
                        titulo=(
                            "Consumo mayor al esperado por dieta"
                        ),
                        detalle=(
                            "Los animales están comiendo más de lo "
                            "que la fórmula proyecta. Posibles "
                            "causas: PV real mayor al estimado, "
                            "frío sostenido (mayor mantenimiento), "
                            "encargado cargando con sobra. Revisar."
                        ),
                    ))
                elif desvio_pct_silo > 15:
                    sugerencias.append(Sugerencia(
                        severidad="atencion",
                        titulo="Consumo menor al esperado",
                        detalle=(
                            "Comen menos de lo proyectado. Causas: "
                            "estrés térmico, problema sanitario "
                            "subclínico, mezcla en mal estado. "
                            "Cruzar con estado del comedero y heces."
                        ),
                    ))

    # ── 5. CRUCE DIETA REAL (lo que tira el cliente) vs DIETA
    #       FORMULADA HMS — sección 5b del cuestionario ──
    comparacion_dieta_real: List[Dict[str, Any]] = []
    if (r.dieta_real_items and contexto_lote):
        dieta_vig = contexto_lote.get("dieta_vigente") or {}
        comp_form = dieta_vig.get("composicion") or []
        cant_an = int(contexto_lote.get("cantidad_inicial") or 0)
        if comp_form and cant_an > 0:
            # Normalizar ambos lados a kg/animal/día
            divisor_real = (
                cant_an if r.dieta_real_modo == "total_dia" else 1
            )
            # Sumar real por nombre normalizado (tolera repetidos)
            real_norm: Dict[str, Dict[str, Any]] = {}
            for it in r.dieta_real_items:
                nm = (it.get("nombre") or "").strip()
                if not nm:
                    continue
                kg = float(it.get("kg") or 0) / divisor_real
                key = nm.lower().strip()
                real_norm.setdefault(
                    key, {"nombre": nm, "kg_an_dia": 0.0},
                )
                real_norm[key]["kg_an_dia"] += kg

            # Formulada por animal/día
            form_norm: Dict[str, Dict[str, Any]] = {}
            for c in comp_form:
                nm = (c.get("nombre") or "").strip()
                if not nm:
                    continue
                kg_an = float(c.get("kg_tal_cual") or 0)
                if kg_an <= 0:
                    continue
                key = nm.lower().strip()
                form_norm[key] = {
                    "nombre": nm,
                    "kg_an_dia": kg_an,
                }

            # Recorrer unión
            todas = set(real_norm.keys()) | set(form_norm.keys())
            for k in sorted(todas):
                f = form_norm.get(k)
                rl = real_norm.get(k)
                kg_form = f["kg_an_dia"] if f else 0
                kg_real = rl["kg_an_dia"] if rl else 0
                nombre = (f or rl or {}).get("nombre", k)
                desvio_kg = kg_real - kg_form
                desvio_pct = (
                    (desvio_kg / kg_form * 100) if kg_form > 0
                    else (100.0 if kg_real > 0 else 0.0)
                )
                # Semáforo
                ad = abs(desvio_pct)
                if kg_form == 0 and kg_real > 0:
                    sem = "atencion"  # ingrediente NO formulado
                    motivo = (
                        f"'{nombre}': el cliente tira "
                        f"{kg_real:.2f} kg/an/día pero NO está en la "
                        "dieta formulada vigente. ¿Confirmar con el "
                        "asesor si conviene sumarlo o sacarlo?"
                    )
                    sugerencias.append(Sugerencia(
                        severidad="atencion",
                        titulo=f"Ingrediente extra: {nombre}",
                        detalle=motivo,
                    ))
                elif kg_real == 0 and kg_form > 0:
                    sem = "urgente"  # falta un ingrediente formulado
                    motivo = (
                        f"'{nombre}': la dieta pide {kg_form:.2f} "
                        "kg/an/día pero el cliente NO lo está usando. "
                        "Riesgo nutricional importante (déficit "
                        "energético, proteico o de minerales según "
                        "qué ingrediente sea)."
                    )
                    sugerencias.append(Sugerencia(
                        severidad="urgente",
                        titulo=(
                            f"Falta ingrediente: {nombre}"
                        ),
                        detalle=motivo,
                    ))
                elif ad <= 10:
                    sem = "verde"
                elif ad <= 20:
                    sem = "atencion"
                else:
                    sem = "urgente"
                    sugerencias.append(Sugerencia(
                        severidad=(
                            "urgente" if ad > 30 else "atencion"
                        ),
                        titulo=(
                            f"{nombre}: desvío {desvio_pct:+.0f}%"
                        ),
                        detalle=(
                            f"Formulado {kg_form:.2f} kg/an/día · "
                            f"Real {kg_real:.2f} kg/an/día · "
                            f"diferencia {desvio_kg:+.2f} kg/an/día. "
                            "Revisar si conviene ajustar la fórmula "
                            "o reentrenar al encargado."
                        ),
                    ))
                comparacion_dieta_real.append({
                    "nombre": nombre,
                    "kg_formulado_animal_dia": round(kg_form, 3),
                    "kg_real_animal_dia": round(kg_real, 3),
                    "desvio_kg": round(desvio_kg, 3),
                    "desvio_pct": round(desvio_pct, 1),
                    "semaforo": sem,
                })

            # Línea-resumen para el bloque alertas_cruce
            if comparacion_dieta_real:
                _ok = sum(
                    1 for c in comparacion_dieta_real
                    if c["semaforo"] == "verde"
                )
                _total = len(comparacion_dieta_real)
                alertas_cruce.append(
                    f"🌾 **Dieta REAL vs formulada**: "
                    f"{_ok}/{_total} ingredientes alineados "
                    f"(±10%). Detalle por ingrediente en bloque "
                    "del análisis."
                )

    # Resumen global de estado
    n_urg = sum(1 for s in sugerencias if s.severidad == "urgente")
    n_at = sum(1 for s in sugerencias if s.severidad == "atencion")
    if n_urg:
        resumen = (
            f"🔴 **{n_urg} tema(s) urgente(s)** + "
            f"{n_at} para atención"
        )
    elif n_at:
        resumen = f"🟡 **{n_at} tema(s) para atención**"
    else:
        resumen = "🟢 **Sin temas críticos detectados**"

    return {
        "sugerencias": sugerencias,
        "alertas_cruce": alertas_cruce,
        "comparacion_dieta_real": comparacion_dieta_real,
        "resumen_estado": resumen,
        "n_urgente": n_urg,
        "n_atencion": n_at,
    }


def analizar_con_agente_llm(
    r: RespuestasEvaluacion,
    contexto_lote: Optional[Dict[str, Any]] = None,
    analisis_reglas: Optional[Dict[str, Any]] = None,
    historial_resumen: str = "",
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """Pide al asesor IA (Claude Haiku por default) un análisis
    técnico de la evaluación.

    A diferencia del motor de reglas (analizar_evaluacion), este
    diagnóstico es **contextual**: considera la dieta vigente, la
    historia del lote, la fase del ciclo, y entrega cuantificaciones
    específicas a esa fórmula en lugar de reglas genéricas.

    Returns:
        dict con:
        - exito: bool
        - analisis_md: str (markdown del diagnóstico LLM)
        - error: str (si exito=False)
    """
    out = {"exito": False, "analisis_md": "", "error": ""}
    try:
        from anthropic import Anthropic
    except ImportError:
        out["error"] = "Falta paquete 'anthropic'."
        return out

    import os
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        out["error"] = "Sin ANTHROPIC_API_KEY configurada."
        return out

    # Armar bloque de contexto para el modelo
    ctx_lote_lines = []
    if contexto_lote:
        ctx_lote_lines.append(
            f"Lote: {contexto_lote.get('identificador','—')}"
        )
        ctx_lote_lines.append(
            f"Categoría: {contexto_lote.get('categoria','—')} · "
            f"Raza: {contexto_lote.get('raza','—')}"
        )
        ctx_lote_lines.append(
            f"Cantidad: {contexto_lote.get('cantidad_inicial', 0)} "
            f"animales"
        )
        ctx_lote_lines.append(
            f"PV ingreso: "
            f"{contexto_lote.get('peso_ingreso_kg','?')} kg · "
            f"ADG objetivo: "
            f"{contexto_lote.get('adpv_objetivo_kg', '?')} kg/día"
        )
        ctx_lote_lines.append(
            f"Fecha ingreso: "
            f"{contexto_lote.get('fecha_ingreso','—')} · "
            f"Objetivo fecha: "
            f"{contexto_lote.get('objetivo_fecha','—')}"
        )
        d = contexto_lote.get("dieta_vigente") or {}
        if d:
            ctx_lote_lines.append("")
            ctx_lote_lines.append("=== DIETA VIGENTE ===")
            ctx_lote_lines.append(
                f"Formulada: {d.get('fecha','—')[:10]}"
            )
            ctx_lote_lines.append(
                f"PB: {d.get('pb_pct',0):.1f}% MS · "
                f"NNP: {d.get('nnp_pct',0):.2f}% MS · "
                f"DMI: {d.get('consumo_ms_kg',0):.2f} kg/día · "
                f"EM: {d.get('em_mcal_dia',0):.1f} Mcal/día"
            )
            obs = d.get("observaciones") or ""
            if obs:
                ctx_lote_lines.append(
                    f"Observaciones formulación: {obs[:200]}"
                )
            comp = d.get("composicion") or []
            if comp:
                ctx_lote_lines.append("Composición:")
                for c in comp:
                    ctx_lote_lines.append(
                        f"  - {c.get('nombre','?')}: "
                        f"{c.get('pct_ms',0):.1f}% MS · "
                        f"{c.get('kg_tal_cual',0):.2f} kg t/c/día"
                    )
    ctx_lote_str = "\n".join(ctx_lote_lines)

    # Respuestas del cuestionario
    respuestas_str = formatear_evaluacion_md(r, {
        "resumen_estado": "(análisis en curso)",
        "sugerencias": [],
        "alertas_cruce": [],
    })

    # Lo que dictó el motor de reglas — para no repetir lo obvio
    reglas_str = ""
    if analisis_reglas:
        reglas_str = (
            "\n\n=== ALERTAS GENERADAS POR EL MOTOR DE REGLAS ===\n"
        )
        for s in analisis_reglas.get("sugerencias", []):
            reglas_str += (
                f"- [{s.severidad.upper()}] {s.titulo}: "
                f"{s.detalle[:120]}\n"
            )
        for a in analisis_reglas.get("alertas_cruce", []):
            reglas_str += f"- {a}\n"

    # ─── Composición del system prompt ───
    # Filosofía única HMS + perfil "evaluacion_cuestionario"
    from . import perfiles_llm as _perfiles_llm
    system_prompt = _perfiles_llm.armar_system_prompt(
        "evaluacion_cuestionario",
    )

    # Bloque viejo deshabilitado (referencia histórica)
    _system_prompt_viejo = (
        "Sos el asesor nutricional senior de HMS Nutrición Animal "
        "(Mauricio Suárez). Trabajás con clientes en La Pampa, "
        "Buenos Aires y Córdoba. Hace 20 años que estás a campo "
        "con ganado en encierre.\n\n"
        "Tu rol acá: el asesor acaba de hacer una llamada con un "
        "cliente y registró un cuestionario estructurado sobre el "
        "estado del lote. Tu tarea es darle un DIAGNÓSTICO TÉCNICO "
        "BREVE y PRÁCTICO, NO un informe largo.\n\n"
        "REGLAS:\n"
        "1. NO repitas lo que ya dijo el motor de reglas — sumá "
        "valor con interpretación contextual, hipótesis, ajustes "
        "específicos a la dieta vigente.\n"
        "2. Sé CUANTITATIVO cuando corresponda (% de cambio, "
        "kg/día, días de monitoreo).\n"
        "3. Si algo está NORMAL no lo menciones — solo lo que "
        "requiere atención o cambio.\n"
        "4. Si no hay nada serio, decilo en una línea y listo.\n"
        "5. NO inventes datos que no estén en el contexto "
        "(reparos, sombra, instalaciones).\n"
        "6. Hablá como Mauricio: prosa criolla técnica, sin pavadas "
        "marketineras, sin 'es importante destacar'.\n"
        "7. Si proponés cambio de fórmula, sé específico: "
        "'subir Fibrogreen Plus de 10% a 12% durante 3 días, "
        "luego volver'.\n\n"
        "ESTRUCTURA DE TU RESPUESTA (markdown, máximo 200 palabras):\n"
        "**Diagnóstico:** 1-2 oraciones de qué está pasando "
        "(hipótesis más probable).\n"
        "**Acciones:** lista bullet de 2-4 acciones concretas con "
        "cuantificación.\n"
        "**A monitorear los próximos días:** 1-2 cosas que querés "
        "ver en la próxima llamada.\n\n"
        "Si la situación es NORMAL/sin novedades, una sola línea: "
        "'Sin temas que requieran ajuste. Próximo control de rutina "
        "en X días.'"
    )

    user_msg = (
        "=== CONTEXTO DEL LOTE ===\n"
        f"{ctx_lote_str}\n\n"
        "=== EVALUACIÓN DEL CUESTIONARIO ===\n"
        f"{respuestas_str}\n"
        f"{reglas_str}\n"
    )
    if historial_resumen:
        user_msg += (
            f"\n=== ÚLTIMOS CONTACTOS PREVIOS ===\n"
            f"{historial_resumen}\n"
        )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        # Extraer texto
        partes = []
        for block in resp.content:
            if hasattr(block, "text"):
                partes.append(block.text)
        out["analisis_md"] = "\n".join(partes).strip()
        out["exito"] = True
    except Exception as e:
        out["error"] = f"LLM falló: {e}"
    return out


def formatear_evaluacion_md(
    r: RespuestasEvaluacion,
    analisis: Dict[str, Any],
    fecha_hora: Optional[str] = None,
) -> str:
    """Formatea respuestas + análisis como markdown legible."""
    if fecha_hora is None:
        fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M")

    bloques: List[str] = []
    bloques.append(f"**📅 Registrado:** {fecha_hora}")
    bloques.append(f"**🤝 Tipo:** {r.tipo_contacto}")
    bloques.append(f"**👤 Atendió:** {r.atendio or '—'}")
    if r.lote_identificador:
        bloques.append(f"**🐂 Lote evaluado:** {r.lote_identificador}")

    bloques.append(f"\n#### {analisis['resumen_estado']}\n")

    # Estado animales
    bloques.append("**Estado de los animales:**")
    bloques.append(
        f"- Aspecto: {r.aspecto_animales or '—'}"
    )
    bloques.append(
        f"- Muertes (mortandad) 48hs: {r.bajas_48hs}"
        + (f" — _Causa: {r.causa_muerte}_" if r.causa_muerte else "")
    )
    bloques.append(f"- Animales enfermos: {r.animales_enfermos}")
    if r.ventas_48hs > 0:
        _txt_v = f"- Ventas / salidas: {r.ventas_48hs} animales"
        if r.kg_promedio_ventas > 0:
            _txt_v += (
                f" (≈ {r.kg_promedio_ventas:.0f} kg/cab → "
                f"{r.ventas_48hs * r.kg_promedio_ventas:.0f} "
                f"kg total)"
            )
        bloques.append(_txt_v)
    if r.detalle_movimientos:
        bloques.append(
            f"- Detalle movimientos: _{r.detalle_movimientos}_"
        )
    if r.bajas_48hs > 0 or r.ventas_48hs > 0:
        bloques.append(
            "  _(registrados automáticamente como movimientos "
            "en el lote)_"
        )
    bloques.append("")

    # Consumo y rumen
    bloques.append("**Consumo y rumen:**")
    bloques.append(
        f"- Comedero: {r.estado_comedero or '—'}"
    )
    bloques.append(f"- Heces: {r.heces or '—'}")
    bloques.append("")

    # Ambiente
    bloques.append("**Ambiente y manejo:**")
    bloques.append(f"- Agua: {r.estado_agua or '—'}")
    bloques.append(f"- Cama: {r.estado_cama or '—'}")
    bloques.append(f"- Reparos: {r.estado_reparos or '—'}")
    bloques.append("")

    # Stock declarado
    bloques.append("**Stock de mercadería declarado:**")
    bloques.append(
        f"- Maíz disponible: {r.maiz_kg_disponible:.0f} kg"
    )
    bloques.append(
        f"- Fibrogreen Plus: {r.fibrogreen_kg_disponible:.0f} kg"
    )
    bloques.append(f"- Rollos disponibles: {r.rollos_disponibles}")
    if r.silo_nivel_pct >= 0:
        bloques.append(
            f"- Silo aproximadamente al {r.silo_nivel_pct}%"
        )
    if r.dias_desde_ultima_carga >= 0:
        bloques.append(
            f"- Última carga al silo: "
            f"hace {r.dias_desde_ultima_carga} días "
            f"({r.kg_ultima_carga:.0f} kg)"
        )
    bloques.append("")

    # Cruces stock vs dieta
    if analisis.get("alertas_cruce"):
        bloques.append("**🧮 Cruces stock vs dieta:**")
        for a in analisis["alertas_cruce"]:
            bloques.append(f"- {a}")
        bloques.append("")

    # Sugerencias automáticas
    if analisis.get("sugerencias"):
        bloques.append("**🎯 Acciones sugeridas por el sistema:**")
        for s in analisis["sugerencias"]:
            ico = _icono_sev(s.severidad)
            bloques.append(f"\n{ico} **{s.titulo}**")
            bloques.append(f"   {s.detalle}")
        bloques.append("")

    # Observaciones libres
    if r.observaciones.strip():
        bloques.append("**📝 Observaciones del asesor:**")
        bloques.append(r.observaciones)
        bloques.append("")

    # Acciones acordadas
    if r.acciones_acordadas.strip():
        bloques.append("**✅ Acciones acordadas con el cliente:**")
        bloques.append(r.acciones_acordadas)

    return "\n".join(bloques)
