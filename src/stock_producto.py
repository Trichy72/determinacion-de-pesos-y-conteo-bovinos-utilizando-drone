"""Cálculo de consumo y stock de producto por lote.

Conecta la dieta formulada del lote (en la tabla `dietas`) con las
entregas de producto registradas (`entregas_producto`) para calcular:

  - Cuánto producto consume el lote por día.
  - Cuántos kg quedan en stock hoy.
  - Cuántos días faltan para que se agote.
  - Si el consumo real difiere del esperado (sub-uso o sobre-uso).

Fuentes:
  - % inclusión: composicion_json de la última dieta del lote.
  - DMI: o bien dietas.consumo_ms_kg, o bien dmi_proyectado del lote.
  - Cantidad de animales: cantidad_vigente_lote(lote, fecha) — respeta
    los movimientos (muertes, ventas, traslados, ingresos) registrados
    en la tabla `movimientos`. Para cada día del cálculo retroactivo se
    usa la cantidad que había ESE día.

Filosofía HMS: el productor que entiende cuánto producto necesita por
día y cuánto le queda, deja de pedir a "ojo" — el sistema le da
visibilidad. Y HMS (asesor) coordina logística con tiempo en lugar de
hacer entregas urgentes.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


def _normalizar_nombre(s: str) -> str:
    """Normaliza nombres de productos para comparación (lower, sin
    espacios extras)."""
    if not s:
        return ""
    return " ".join(s.lower().strip().split())


def _palabras_clave(s: str) -> set:
    """Extrae las palabras 'fuertes' de un nombre de producto, ignorando
    descriptores entre paréntesis y palabras genéricas. Útil para
    cruzar nombres tipo 'Fibroter' vs 'Fibroter (BALCOOP Destete
    Precoz)' vs 'Fibroter granel'."""
    import re
    if not s:
        return set()
    s = s.lower()
    # Sacar contenido entre paréntesis (descriptores comerciales)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    # Tokenizar
    tokens = re.findall(r"[a-záéíóúñ]+", s)
    # Descartar palabras genéricas
    stop = {
        "de", "del", "la", "el", "los", "las", "y", "o", "con",
        "sin", "bolsa", "bolsas", "granel", "kg", "tn", "ton",
        "media", "alta", "baja", "calidad",
    }
    return {t for t in tokens if len(t) >= 3 and t not in stop}


def _mismo_producto(a: str, b: str) -> bool:
    """¿Estos dos nombres se refieren al mismo producto? Tolera
    descriptores extra entre paréntesis y formato granel/bolsa.
    Match si:
      1. Son idénticos (normalizados), o
      2. Una palabra clave fuerte de un lado coincide con la del otro
         (ej. 'Fibroter' vs 'Fibroter (BALCOOP Destete Precoz)')."""
    if _normalizar_nombre(a) == _normalizar_nombre(b):
        return True
    ka = _palabras_clave(a)
    kb = _palabras_clave(b)
    if not ka or not kb:
        return False
    # Si una de las dos es subconjunto de la otra, son el mismo producto
    if ka.issubset(kb) or kb.issubset(ka):
        return True
    # O si comparten al menos una palabra clave significativa y NO hay
    # palabras conflictivas obvias — para nombres comerciales típicos
    # esto es suficiente.
    return bool(ka & kb)


def obtener_pct_inclusion_lote(
    lote_id: int, producto: str,
    fecha_referencia: Optional[str] = None,
) -> Optional[float]:
    """Busca el producto en la dieta vigente del lote y devuelve su
    porcentaje de inclusión en MS. Si no hay dieta o el producto no
    está en ella, devuelve None.

    Args:
        lote_id: id del lote.
        producto: nombre del producto (Fibrogreen, Fibroter, etc.).
            La comparación es case-insensitive y tolera espacios.
        fecha_referencia: ISO YYYY-MM-DD. Si se pasa, usa la dieta
            vigente en esa fecha (útil para planes de adaptación con
            varias dietas con distinta fecha de inicio). Si no se pasa,
            usa la última dieta cargada.

    Returns:
        Float con el % de inclusión (ej. 12.0 si el producto está al
        12% en la mezcla), o None si no se encuentra.
    """
    from . import database as db
    dietas = db.listar_dietas(lote_id)
    if not dietas:
        return None
    # Si se pasa fecha_referencia, usar la última dieta cuya fecha sea
    # <= referencia (es decir, la dieta VIGENTE ese día). Esto es lo
    # que permite manejar planes de adaptación: varias dietas con
    # distinta fecha de inicio, cada una vigente desde su fecha hasta
    # que arranca la siguiente.
    dieta = _dieta_vigente(dietas, fecha_referencia)
    if not dieta:
        # Fallback: si todas son futuras a la referencia, usar la
        # primera por fecha (la más cercana) — escenario típico al
        # cargar el plan antes de que arranque.
        dieta = dietas[-1]
    composicion = dieta.get("composicion") or []
    for item in composicion:
        nombre_item = item.get("nombre") or item.get("ingrediente") or ""
        if _mismo_producto(nombre_item, producto):
            return float(item.get("pct_ms") or 0)
    return None


def _dieta_vigente(dietas: List[Dict], fecha_referencia: Optional[str]):
    """De una lista de dietas (ya ordenadas por fecha DESC), devuelve
    la que estaba vigente en la fecha_referencia. None si todas son
    posteriores."""
    if not dietas:
        return None
    if not fecha_referencia:
        return dietas[0]
    for d in dietas:
        if (d.get("fecha") or "") <= fecha_referencia:
            return d
    return None


def calcular_consumo_diario_kg(
    lote_id: int, producto: str,
    dmi_kg_dia_override: Optional[float] = None,
    fecha_referencia: Optional[str] = None,
) -> Optional[Dict]:
    """Calcula cuántos kg/día del producto consume el LOTE COMPLETO
    en la fecha de referencia indicada.

    Fórmula:
        consumo = DMI_animal × cantidad × pct_inclusion / 100

    Si el lote tiene plan de adaptación (varias dietas con distintas
    fechas de inicio), el % de inclusión usado corresponde a la dieta
    vigente en `fecha_referencia` (default hoy).

    Args:
        lote_id: id del lote.
        producto: nombre del producto.
        dmi_kg_dia_override: opcional, sobreescribe el DMI por animal
            tomado de la dieta vigente. Útil si querés usar DMI
            ajustado por clima.
        fecha_referencia: ISO date (YYYY-MM-DD) — la dieta vigente se
            calcula para ese día. Default: hoy.

    Returns:
        Dict con kg_dia, pct_inclusion, dmi_kg, cantidad_animales,
        fuente_dmi, fase_vigente. None si faltan datos.
    """
    from . import database as db
    from datetime import datetime as _dt
    if not fecha_referencia:
        fecha_referencia = _dt.now().strftime("%Y-%m-%d")
    lote = db.obtener_lote(lote_id)
    if not lote:
        return None
    # Cantidad vigente a la fecha de referencia: cantidad_inicial
    # más/menos los movimientos (muertes, ventas, traslados, ingresos)
    # ocurridos hasta esa fecha. Día por día puede cambiar — clave para
    # que el cálculo retroactivo de consumo respete bajas previas y
    # para que la proyección hacia adelante use la cantidad actual.
    cantidad = db.cantidad_vigente_lote(lote_id, fecha_referencia)
    if cantidad <= 0:
        return None

    pct = obtener_pct_inclusion_lote(
        lote_id, producto, fecha_referencia=fecha_referencia,
    )
    if pct is None or pct <= 0:
        return None

    # DMI por animal: si hay override, usarlo; si no, sacarlo de la
    # dieta VIGENTE en la fecha de referencia.
    dmi_animal = dmi_kg_dia_override
    fuente_dmi = "override"
    fase_vigente = None
    if dmi_animal is None:
        dietas = db.listar_dietas(lote_id)
        dieta_v = _dieta_vigente(dietas, fecha_referencia) or (
            dietas[-1] if dietas else None
        )
        if dieta_v:
            dmi_animal = dieta_v.get("consumo_ms_kg")
            fuente_dmi = f"dieta del {dieta_v.get('fecha')}"
            fase_vigente = dieta_v.get("observaciones") or ""
    if dmi_animal is None or dmi_animal <= 0:
        return None

    consumo_kg_dia = dmi_animal * cantidad * pct / 100.0
    return {
        "kg_dia": round(consumo_kg_dia, 2),
        "pct_inclusion": round(pct, 1),
        "dmi_kg_animal": round(dmi_animal, 2),
        "cantidad_animales": cantidad,
        "fuente_dmi": fuente_dmi,
        "fase_vigente": fase_vigente,
        "fecha_referencia": fecha_referencia,
    }


def _consumo_acumulado_kg(
    lote_id: int, producto: str, fecha_desde: str, fecha_hasta: str,
    dmi_kg_dia_override: Optional[float] = None,
) -> float:
    """Suma el consumo del producto día por día entre `fecha_desde` y
    `fecha_hasta` (inclusive), usando la dieta vigente cada día. Esto
    permite que un plan de adaptación de 4 fases sume correctamente:
    los primeros días con bajo % de inclusión, los últimos con %
    completo.

    Args:
        lote_id: id del lote.
        producto: nombre del producto a sumar.
        fecha_desde: ISO YYYY-MM-DD inclusivo.
        fecha_hasta: ISO YYYY-MM-DD inclusivo.
        dmi_kg_dia_override: si querés usar DMI ajustado por clima.

    Returns:
        Suma total de kg consumidos en ese rango.
    """
    from datetime import datetime as _dt, timedelta as _td
    try:
        d_desde = _dt.strptime(fecha_desde, "%Y-%m-%d").date()
        d_hasta = _dt.strptime(fecha_hasta, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0.0
    if d_hasta < d_desde:
        return 0.0
    total = 0.0
    dia = d_desde
    while dia <= d_hasta:
        info = calcular_consumo_diario_kg(
            lote_id, producto,
            dmi_kg_dia_override=dmi_kg_dia_override,
            fecha_referencia=dia.isoformat(),
        )
        if info:
            total += info["kg_dia"]
        dia += _td(days=1)
    return total


def calcular_stock_actual(
    cliente_id: int, lote_id: int, producto: str,
    fecha_referencia: Optional[str] = None,
    dmi_kg_dia_override: Optional[float] = None,
) -> Optional[Dict]:
    """Calcula stock actual del producto para un lote del cliente.

    Toma todas las entregas asociadas al cliente + producto desde la
    primera entrega del lote en adelante, y resta el consumo
    acumulado al ritmo del consumo diario calculado.

    Args:
        cliente_id: id del cliente.
        lote_id: id del lote (para asociar dieta).
        producto: nombre del producto.
        fecha_referencia: ISO date desde la que calcular (default hoy).
        dmi_kg_dia_override: si querés usar DMI ajustado por clima.

    Returns:
        Dict con kg_entregados_total, consumo_total_a_fecha,
        kg_restantes_hoy, consumo_diario_kg, fecha_agotamiento,
        dias_restantes, diagnostico_uso. None si faltan datos.
    """
    from . import database as db
    if not fecha_referencia:
        fecha_referencia = datetime.now().strftime("%Y-%m-%d")
    fecha_ref = datetime.strptime(fecha_referencia, "%Y-%m-%d").date()

    # Consumo diario HOY (usando dieta vigente en la fecha de referencia)
    consumo_info = calcular_consumo_diario_kg(
        lote_id, producto, dmi_kg_dia_override,
        fecha_referencia=fecha_referencia,
    )
    if not consumo_info:
        return None
    consumo_dia = consumo_info["kg_dia"]
    if consumo_dia <= 0:
        return None

    # Entregas del cliente para este producto, filtradas al lote
    # (si la entrega tiene lote_id, solo cuentan las de este lote;
    # si está en None, suponemos que aplica a todo el cliente).
    todas = db.listar_entregas_cliente(cliente_id, limit=500)
    entregas_filtradas = []
    for e in todas:
        if not _mismo_producto(e.get("producto_nombre", ""), producto):
            continue
        # Filtrar por lote: si la entrega tiene lote_id != None,
        # solo aplica si coincide; si está en None, aplica.
        if e.get("lote_id") and e["lote_id"] != lote_id:
            continue
        entregas_filtradas.append(e)

    if not entregas_filtradas:
        return {
            "kg_entregados_total": 0,
            "kg_restantes_hoy": 0,
            "consumo_diario_kg": consumo_dia,
            "consumo_total_a_fecha": 0,
            "fecha_agotamiento": None,
            "dias_restantes": 0,
            "diagnostico_uso": "sin_entregas",
            "consumo_info": consumo_info,
            "entregas": [],
        }

    kg_entregados = sum(e.get("kg_total") or 0 for e in entregas_filtradas)
    # Fecha de la primera entrega → desde ahí empieza el consumo
    fechas = [
        datetime.strptime(e["fecha_entrega"][:10], "%Y-%m-%d").date()
        for e in entregas_filtradas
    ]
    primera = min(fechas)
    # Consumo acumulado: sumar día por día respetando el plan de
    # adaptación. En el primer día del lote la dieta puede tener
    # bajo % de Fibroter (fase 1), y subir gradualmente hasta el
    # día 22+ (fase 4 a plena dosis). Acumular plano no servía.
    if primera <= fecha_ref:
        consumo_acumulado = _consumo_acumulado_kg(
            lote_id, producto,
            fecha_desde=primera.isoformat(),
            fecha_hasta=(fecha_ref - timedelta(days=1)).isoformat(),
            dmi_kg_dia_override=dmi_kg_dia_override,
        )
    else:
        consumo_acumulado = 0.0
    kg_restantes = max(0.0, kg_entregados - consumo_acumulado)

    # Días que faltan para agotar, proyectando hacia adelante con la
    # dieta vigente día por día (la fase puede aún cambiar de 3→4
    # antes del agotamiento).
    dia_proy = fecha_ref
    kg_restantes_proy = kg_restantes
    dias_restantes = 0
    max_iter = 365  # techo de seguridad
    while kg_restantes_proy > 0 and dias_restantes < max_iter:
        info_dia = calcular_consumo_diario_kg(
            lote_id, producto,
            dmi_kg_dia_override=dmi_kg_dia_override,
            fecha_referencia=dia_proy.isoformat(),
        )
        if not info_dia or info_dia["kg_dia"] <= 0:
            break
        kg_restantes_proy -= info_dia["kg_dia"]
        dias_restantes += 1
        dia_proy += timedelta(days=1)
    fecha_agot = fecha_ref + timedelta(days=int(dias_restantes))

    # Diagnóstico de uso real (compara fechas de entregas)
    # Si el cliente recibió 2+ entregas, podemos inferir cuánto
    # consumió REAL entre ellas y compararlo con el teórico.
    diagnostico = "normal"
    detalle_diag = ""
    if len(entregas_filtradas) >= 2:
        ordenadas = sorted(
            entregas_filtradas,
            key=lambda x: x["fecha_entrega"],
        )
        f1 = datetime.strptime(
            ordenadas[0]["fecha_entrega"][:10], "%Y-%m-%d"
        ).date()
        f2 = datetime.strptime(
            ordenadas[1]["fecha_entrega"][:10], "%Y-%m-%d"
        ).date()
        kg1 = ordenadas[0].get("kg_total") or 0
        dias_entre = (f2 - f1).days
        if dias_entre > 0 and consumo_dia > 0:
            dias_teoricos = kg1 / consumo_dia
            ratio = dias_entre / dias_teoricos if dias_teoricos > 0 else 1
            if ratio < 0.85:
                diagnostico = "sobre_uso"
                detalle_diag = (
                    f"La entrega anterior de {kg1:.0f} kg duró "
                    f"{dias_entre} días en lugar de los "
                    f"{dias_teoricos:.0f} esperados. El lote está "
                    f"consumiendo MÁS que lo formulado "
                    f"({(1/ratio):.1%} sobre lo esperado)."
                )
            elif ratio > 1.15:
                diagnostico = "sub_uso"
                detalle_diag = (
                    f"La entrega anterior de {kg1:.0f} kg duró "
                    f"{dias_entre} días en lugar de los "
                    f"{dias_teoricos:.0f} esperados. El lote está "
                    f"consumiendo MENOS que lo formulado "
                    f"({ratio:.1%} de lo esperado)."
                )
            else:
                diagnostico = "normal"
                detalle_diag = (
                    f"Consumo real coincide con lo formulado: la "
                    f"entrega de {kg1:.0f} kg duró {dias_entre} "
                    f"días (esperado: {dias_teoricos:.0f})."
                )

    return {
        "kg_entregados_total": round(kg_entregados, 1),
        "kg_restantes_hoy": round(kg_restantes, 1),
        "consumo_diario_kg": consumo_dia,
        "consumo_total_a_fecha": round(consumo_acumulado, 1),
        "fecha_agotamiento": fecha_agot.isoformat(),
        "dias_restantes": round(dias_restantes, 1),
        "diagnostico_uso": diagnostico,
        "detalle_diagnostico": detalle_diag,
        "consumo_info": consumo_info,
        "entregas": entregas_filtradas,
        "fecha_primera_entrega": primera.isoformat(),
    }


def listar_productos_lote(lote_id: int) -> List[str]:
    """Devuelve nombres únicos de productos que aparecen en la última
    dieta del lote — útil para mostrar qué productos manejar."""
    from . import database as db
    dietas = db.listar_dietas(lote_id)
    if not dietas:
        return []
    ultima = dietas[0]
    composicion = ultima.get("composicion") or []
    nombres = []
    for item in composicion:
        n = item.get("nombre") or item.get("ingrediente") or ""
        if n and n not in nombres:
            nombres.append(n)
    return nombres


def listar_productos_hms_lote(
    cliente_id: int, lote_id: int,
) -> List[str]:
    """Productos de la dieta que HMS efectivamente entregó al cliente.

    Filosofía: el control de stock solo tiene sentido para los productos
    que vos vendés (núcleos, concentrados, premezclas). El maíz, los
    rollos, el silaje los compra el productor por su lado — no son
    parte de tu logística, así que no tienen por qué aparecer en la
    tabla de stock.

    El criterio: un producto se considera "tuyo" si tiene al menos
    UNA entrega registrada en `entregas_producto` para este cliente
    (sea para este lote o genérica del cliente).

    Args:
        cliente_id: id del cliente.
        lote_id: id del lote (para sacar la dieta).

    Returns:
        Lista de nombres tal como aparecen en la dieta del lote, pero
        solo los que matchean con alguna entrega de HMS para el
        cliente. Si no hay ninguna entrega, devuelve [] (no hay nada
        que trackear de stock todavía).
    """
    from . import database as db
    productos_dieta = listar_productos_lote(lote_id)
    if not productos_dieta:
        return []
    entregas = db.listar_entregas_cliente(cliente_id, limit=500)
    if not entregas:
        return []
    productos_entregados = {
        e.get("producto_nombre", "") for e in entregas
        if e.get("producto_nombre")
    }
    if not productos_entregados:
        return []
    # Para cada producto de la dieta, ver si alguna entrega lo matchea
    # (usando el matcher tolerante a paréntesis/descriptores).
    resultado = []
    for nombre_dieta in productos_dieta:
        for nombre_entrega in productos_entregados:
            if _mismo_producto(nombre_dieta, nombre_entrega):
                resultado.append(nombre_dieta)
                break
    return resultado


# =====================================================================
# ESTIMACIÓN DE PESO VIVO + AJUSTE DINÁMICO DEL CONSUMO POR ADG
# =====================================================================

def estimar_pv_lineal_simple(
    lote: Dict, fecha_ref: Optional[str] = None,
) -> float:
    """PV proyectado lineal SIN pesadas — solo PV_ingreso + ADG × días.

    Se diferencia de estimar_peso_vivo_lote en que ESTE NO mira las
    pesadas. Sirve para mostrar la proyección 'teórica' al lado de
    la real (pesada) y la ajustada por clima.
    """
    from datetime import datetime as _dt
    if not fecha_ref:
        fecha_ref = _dt.now().strftime("%Y-%m-%d")
    peso_ingreso = float(lote.get("peso_ingreso_kg") or 0)
    adpv = float(lote.get("adpv_objetivo_kg") or 0)
    fi_raw = (lote.get("fecha_ingreso") or "")[:10]
    if peso_ingreso > 0 and adpv > 0 and fi_raw:
        try:
            fi = _dt.strptime(fi_raw, "%Y-%m-%d").date()
            fr = _dt.strptime(fecha_ref, "%Y-%m-%d").date()
            dias = max(0, (fr - fi).days)
            return peso_ingreso + adpv * dias
        except Exception:
            pass
    if peso_ingreso > 0:
        return peso_ingreso
    return 0.0


def estimar_pv_balanza(
    lote: Dict, fecha_ref: Optional[str] = None,
) -> Tuple[float, Optional[str]]:
    """PV REAL desde la última pesada registrada (balanza, manga,
    estimación visual).

    Si hay pesadas, devuelve la última con fecha. Si no hay,
    devuelve (0, None).

    Cuando hay una pesada de hace varios días y se quiere "proyectar"
    a hoy, sumamos ADG objetivo × días desde esa pesada (linealmente).

    Returns:
        (pv_kg, fecha_pesada_iso). Si no hay pesadas, (0, None).
    """
    from . import database as db
    from datetime import datetime as _dt
    if not fecha_ref:
        fecha_ref = _dt.now().strftime("%Y-%m-%d")
    try:
        pesadas = db.listar_pesadas(lote["id"]) or []
    except Exception:
        return 0.0, None
    if not pesadas:
        return 0.0, None
    try:
        f_ref = _dt.strptime(fecha_ref, "%Y-%m-%d").date()
    except Exception:
        return 0.0, None
    validas = []
    for p in pesadas:
        pp = float(p.get("peso_promedio_kg") or 0)
        if pp <= 0:
            continue
        fp_raw = (p.get("fecha") or "")[:10]
        try:
            fp = _dt.strptime(fp_raw, "%Y-%m-%d").date()
        except Exception:
            continue
        if fp <= f_ref:
            validas.append((fp, pp))
    if not validas:
        return 0.0, None
    validas.sort(key=lambda x: x[0])
    f_ult, pv_ult = validas[-1]
    # Extender linealmente con ADG hasta fecha_ref si hay días
    # transcurridos desde la pesada
    adpv = float(lote.get("adpv_objetivo_kg") or 0)
    dias_desde = max(0, (f_ref - f_ult).days)
    pv_estimado = pv_ult + adpv * dias_desde
    return pv_estimado, f_ult.isoformat()


# Factores de reducción de ADG por severidad climática real
# (basados en NRC 2016 + ajustes Pampa Húmeda — conservadores).
# Cada día con esa severidad multiplica el ADG objetivo por el
# factor correspondiente.
_FACTORES_ADG_SEVERIDAD = {
    "sin_estres": 1.00,   # 🟢 sin descuento
    "atencion":   0.92,   # 🟡 ~8% menos
    "moderado":   0.78,   # 🟠 ~22% menos
    "critico":    0.55,   # 🔴 ~45% menos
}


def _clasificar_sev_real_dia(
    t_min: Optional[float], t_max: Optional[float],
    hr: Optional[float], precip: Optional[float],
    viento: Optional[float], nubes: Optional[float] = None,
) -> str:
    """Replica la lógica de severidad real del dashboard, devuelve
    una clave string: sin_estres / atencion / moderado / critico.

    Se mantiene en sync con src/dashboard.py — mismo cálculo.
    """
    from .clima import calcular_thi
    # Calor (THI ajustado por viento — Mader 2006)
    if t_max is not None and hr is not None:
        thi = calcular_thi(t_max, hr)
        viento_ms = (viento or 0) / 3.6
        thi_adj = thi - 0.5 * viento_ms
        if thi_adj >= 84:
            return "critico"
        if thi_adj >= 79:
            return "moderado"
        if thi_adj >= 72:
            return "atencion"
    # Frío
    sev = "sin_estres"
    if t_min is not None and t_min <= 15:
        # Wind chill bovino
        tmin_sentida = t_min - (((viento or 0) / 10) * 3)
        pelaje_mojado = (precip or 0) > 5
        niebla_densa = (hr or 0) >= 98 and (viento or 0) < 5
        dia_sin_sol = (
            (nubes or 0) >= 75 and (hr or 0) >= 90
        )
        if pelaje_mojado or niebla_densa:
            tmin_sentida -= 4
        elif dia_sin_sol:
            tmin_sentida -= 2
        if tmin_sentida <= -5:
            return "critico"
        if tmin_sentida <= 0:
            sev = "moderado"
        elif tmin_sentida <= 5:
            sev = "atencion"
    if (precip or 0) > 15 and (t_min or 99) < 12:
        if sev == "sin_estres":
            sev = "atencion"
    return sev


def estimar_pv_ajustado_clima(
    lote: Dict, fecha_ref: Optional[str] = None,
    cliente: Optional[Dict] = None,
) -> Dict:
    """PV ajustado por clima — itera día a día desde fecha_ingreso
    y aplica factor de reducción del ADG según severidad real de cada
    día (usando datos climáticos históricos de Open-Meteo Archive).

    Args:
        lote: dict del lote (con peso_ingreso_kg, adpv_objetivo_kg,
              fecha_ingreso, cliente_id).
        fecha_ref: ISO date (default hoy).
        cliente: dict del cliente con localidad/lat/lon. Si no se
                 pasa, se busca por cliente_id.

    Returns:
        dict con:
          - pv_ajustado_kg: float
          - dias_total: int
          - dias_adversos: int (atencion+moderado+critico)
          - dias_criticos: int
          - delta_vs_lineal_kg: float (negativo = pérdida)
          - origen: "ok" | "sin_pesada" | "sin_clima" | "sin_datos"
    """
    from datetime import datetime as _dt, timedelta as _td
    from .clima import (
        obtener_clima_historico, geocodificar, geocodificar_manual,
    )
    from . import database as db

    if not fecha_ref:
        fecha_ref = _dt.now().strftime("%Y-%m-%d")

    out = {
        "pv_ajustado_kg": 0.0,
        "dias_total": 0,
        "dias_adversos": 0,
        "dias_criticos": 0,
        "delta_vs_lineal_kg": 0.0,
        "origen": "sin_datos",
    }

    peso_ingreso = float(lote.get("peso_ingreso_kg") or 0)
    adpv = float(lote.get("adpv_objetivo_kg") or 0)
    fi_raw = (lote.get("fecha_ingreso") or "")[:10]
    if peso_ingreso <= 0 or adpv <= 0 or not fi_raw:
        return out
    try:
        f_ing = _dt.strptime(fi_raw, "%Y-%m-%d").date()
        f_ref = _dt.strptime(fecha_ref, "%Y-%m-%d").date()
    except Exception:
        return out
    dias_total = max(0, (f_ref - f_ing).days)
    out["dias_total"] = dias_total
    if dias_total <= 0:
        out["pv_ajustado_kg"] = peso_ingreso
        out["origen"] = "ok"
        return out

    # Cliente y coordenadas
    if not cliente:
        try:
            cliente = db.obtener_cliente(lote.get("cliente_id")) or {}
        except Exception:
            cliente = {}
    geo = None
    try:
        if cliente.get("lat") is not None \
                and cliente.get("lon") is not None:
            geo = geocodificar_manual(
                float(cliente["lat"]),
                float(cliente["lon"]),
                cliente.get("localidad", ""),
            )
        elif cliente.get("localidad"):
            geo = geocodificar(cliente["localidad"])
    except Exception:
        geo = None
    if not geo:
        # Sin coordenadas no podemos consultar clima histórico
        out["pv_ajustado_kg"] = peso_ingreso + adpv * dias_total
        out["origen"] = "sin_clima"
        return out

    # Traer clima histórico del período (de fi a fr).
    # OJO: Open-Meteo Archive tiene ~5 días de delay — los datos del
    # día de ayer no suelen estar disponibles. Pedimos hasta MIN(f_ref,
    # hoy-5d). Para los días recientes sin datos archive, asumimos
    # ADG completo (factor 1.0) — somos conservadores con la pérdida.
    _hoy_real = _dt.now().date()
    f_archive_hasta = min(f_ref, _hoy_real - _td(days=5))
    if f_archive_hasta < f_ing:
        # El período completo está dentro del delay del archive —
        # no podemos ajustar. Lineal con mensaje claro.
        out["pv_ajustado_kg"] = peso_ingreso + adpv * dias_total
        out["origen"] = "sin_clima_reciente"
        return out

    try:
        clima = obtener_clima_historico(
            geo["lat"], geo["lon"],
            fecha_desde=f_ing.isoformat(),
            fecha_hasta=f_archive_hasta.isoformat(),
        )
    except Exception:
        clima = None
    if not clima:
        out["pv_ajustado_kg"] = peso_ingreso + adpv * dias_total
        out["origen"] = "sin_clima"
        return out

    daily = (clima.get("daily") or {})
    fechas = daily.get("time") or []
    tmin_arr = daily.get("temperature_2m_min") or []
    tmax_arr = daily.get("temperature_2m_max") or []
    hr_arr = (
        daily.get("relative_humidity_2m_mean")
        or daily.get("relative_humidity_2m_max") or []
    )
    precip_arr = daily.get("precipitation_sum") or []
    viento_arr = (
        daily.get("wind_speed_10m_max")
        or daily.get("windspeed_10m_max") or []
    )
    nubes_arr = daily.get("cloud_cover_mean") or []

    # Iterar día a día sobre los días con datos del archive
    pv = peso_ingreso
    n_adv = 0
    n_crit = 0
    dias_con_datos = 0
    for i, _f_str in enumerate(fechas):
        _tmin = tmin_arr[i] if i < len(tmin_arr) else None
        _tmax = tmax_arr[i] if i < len(tmax_arr) else None
        _hr = hr_arr[i] if i < len(hr_arr) else None
        _prec = precip_arr[i] if i < len(precip_arr) else None
        _vto = viento_arr[i] if i < len(viento_arr) else None
        _nub = nubes_arr[i] if i < len(nubes_arr) else None
        if _tmin is None and _tmax is None:
            # Día sin datos, asumir sin estrés
            pv += adpv
            continue
        sev = _clasificar_sev_real_dia(
            _tmin, _tmax, _hr, _prec, _vto, _nub,
        )
        factor = _FACTORES_ADG_SEVERIDAD.get(sev, 1.0)
        pv += adpv * factor
        dias_con_datos += 1
        if sev != "sin_estres":
            n_adv += 1
        if sev == "critico":
            n_crit += 1

    # Días recientes sin datos del archive (delay ~5 días) — asumir
    # ADG completo. Es conservador: si los últimos días fueron malos,
    # estamos sobreestimando el PV; cuando los datos se publiquen el
    # cálculo se ajusta automáticamente.
    dias_sin_archive = max(
        0, dias_total - dias_con_datos,
    )
    if dias_sin_archive > 0:
        pv += adpv * dias_sin_archive

    pv_lineal = peso_ingreso + adpv * dias_total
    out["pv_ajustado_kg"] = round(pv, 1)
    out["dias_adversos"] = n_adv
    out["dias_criticos"] = n_crit
    out["dias_con_archive"] = dias_con_datos
    out["dias_sin_archive"] = dias_sin_archive
    out["delta_vs_lineal_kg"] = round(pv - pv_lineal, 1)
    out["origen"] = "ok"
    return out


def estimar_peso_vivo_lote(
    lote: Dict, fecha_ref: Optional[str] = None,
) -> float:
    """Estima el peso vivo promedio del lote a `fecha_ref`.

    Jerarquía:
      1) Última pesada registrada ≤ fecha_ref (peso_promedio_kg).
      2) peso_ingreso_kg + ADPV objetivo × días desde fecha_ingreso.
      3) peso_ingreso_kg si no hay ADPV ni fecha_ingreso.
      4) 0 si no hay nada.

    Args:
        lote: dict con peso_ingreso_kg, adpv_objetivo_kg, fecha_ingreso,
              id.
        fecha_ref: ISO date (default hoy).

    Returns:
        kg/cabeza estimados, o 0 si no se puede calcular.
    """
    from . import database as db
    from datetime import datetime as _dt
    if not fecha_ref:
        fecha_ref = _dt.now().strftime("%Y-%m-%d")
    # 1) Pesadas
    try:
        pesadas = db.listar_pesadas(lote["id"])
    except Exception:
        pesadas = []
    if pesadas:
        try:
            f_ref = _dt.strptime(fecha_ref, "%Y-%m-%d").date()
        except Exception:
            f_ref = None
        pesadas_validas = []
        for p in pesadas:
            pp = float(p.get("peso_promedio_kg") or 0)
            if pp <= 0:
                continue
            fp_raw = (p.get("fecha") or "")[:10]
            try:
                fp = _dt.strptime(fp_raw, "%Y-%m-%d").date()
            except Exception:
                fp = None
            if f_ref is None or fp is None or fp <= f_ref:
                pesadas_validas.append((fp, pp))
        if pesadas_validas:
            pesadas_validas.sort(key=lambda x: x[0] or _dt.min.date())
            return pesadas_validas[-1][1]

    # 2) Peso ingreso + ADPV × días
    peso_ingreso = float(lote.get("peso_ingreso_kg") or 0)
    adpv = float(lote.get("adpv_objetivo_kg") or 0)
    fi_raw = (lote.get("fecha_ingreso") or "")[:10]
    if peso_ingreso > 0 and adpv > 0 and fi_raw:
        try:
            fi = _dt.strptime(fi_raw, "%Y-%m-%d").date()
            fr = _dt.strptime(fecha_ref, "%Y-%m-%d").date()
            dias = max(0, (fr - fi).days)
            return peso_ingreso + adpv * dias
        except Exception:
            pass

    # 3) Sólo peso ingreso (sin ADPV)
    if peso_ingreso > 0:
        return peso_ingreso

    return 0.0


# Límites del factor de escala para evitar que un peso estimado erróneo
# (o un ADPV mal cargado) infle/desinfle el consumo más de lo razonable.
_FACTOR_ESCALA_MIN = 0.85   # los animales no consumen <85% de lo
                            # formulado, aunque hayan perdido peso.
_FACTOR_ESCALA_MAX = 1.40   # tope: una dieta vieja escalada a +40%
                            # ya debería haber sido reformulada.


def factor_escala_consumo_pv(
    lote: Dict, dieta: Dict, fecha_referencia: Optional[str] = None,
) -> Tuple[float, Dict]:
    """Factor para escalar kg_tal_cual de una dieta a la fecha de hoy
    según el peso vivo proyectado.

    Premisa: el consumo escala linealmente con el peso vivo (% PV
    constante). Si la dieta se formuló cuando el animal pesaba 320 kg
    y hoy pesa 380 kg, el consumo aumentó ~18% (380/320).

    Args:
        lote: dict con peso_ingreso_kg, adpv_objetivo_kg, fecha_ingreso.
        dieta: dict con campo 'fecha' (fecha en que se formuló la dieta).
        fecha_referencia: ISO date (default hoy).

    Returns:
        (factor, info) donde:
          - factor: float entre _FACTOR_ESCALA_MIN y _FACTOR_ESCALA_MAX.
            1.0 si no se puede calcular (sin ADPV o sin fecha de dieta).
          - info: dict con peso_referencia, peso_actual, dias_desde_dieta,
            origen ('adg'|'estatico'|'sin_adg'|'sin_fecha_dieta').
    """
    from datetime import datetime as _dt
    if not fecha_referencia:
        fecha_referencia = _dt.now().strftime("%Y-%m-%d")

    # Siempre intentamos calcular el peso actual/ref, aunque no podamos
    # aplicar factor — el caller (gráfico, KPIs) los muestra igual.
    peso_act_pre = estimar_peso_vivo_lote(lote, fecha_referencia)
    info = {
        "peso_referencia_kg": round(peso_act_pre, 1),
        "peso_actual_kg": round(peso_act_pre, 1),
        "dias_desde_dieta": 0,
        "origen": "estatico",
        "factor_aplicado": 1.0,
    }

    adpv = float(lote.get("adpv_objetivo_kg") or 0)
    if adpv <= 0:
        info["origen"] = "sin_adg"
        return 1.0, info

    fecha_dieta_raw = (dieta.get("fecha") or "")[:10]
    if not fecha_dieta_raw:
        info["origen"] = "sin_fecha_dieta"
        return 1.0, info

    # Peso estimado a la fecha de formulación de la dieta.
    peso_ref = estimar_peso_vivo_lote(lote, fecha_dieta_raw)
    if peso_ref <= 0:
        info["origen"] = "sin_peso_referencia"
        return 1.0, info
    info["peso_referencia_kg"] = round(peso_ref, 1)

    # Peso estimado a la fecha de referencia (hoy o cualquier otra).
    peso_act = estimar_peso_vivo_lote(lote, fecha_referencia)
    if peso_act <= 0:
        info["origen"] = "sin_peso_actual"
        return 1.0, info
    info["peso_actual_kg"] = round(peso_act, 1)

    try:
        fd = _dt.strptime(fecha_dieta_raw, "%Y-%m-%d").date()
        fr = _dt.strptime(fecha_referencia, "%Y-%m-%d").date()
        info["dias_desde_dieta"] = max(0, (fr - fd).days)
    except Exception:
        pass

    factor_bruto = peso_act / peso_ref
    factor = max(_FACTOR_ESCALA_MIN, min(_FACTOR_ESCALA_MAX, factor_bruto))

    info.update({
        "peso_referencia_kg": round(peso_ref, 1),
        "peso_actual_kg": round(peso_act, 1),
        "factor_bruto": round(factor_bruto, 4),
        "factor_aplicado": round(factor, 4),
        "origen": "adg",
    })
    return factor, info


# =====================================================================
# CARGAS DEL SILOCOMEDERO — proyección de fin de carga
# =====================================================================

def _consumo_diario_mezcla_kg(
    lote_id: int, fecha_referencia: Optional[str] = None,
) -> float:
    """Kg de mezcla TAL CUAL (no MS) consumidos por día por el lote
    completo, según la dieta vigente.

    Suma kg_tal_cual de todos los ingredientes de la composición de la
    dieta vigente × cantidad de animales vigente, escalado por el peso
    vivo proyectado a `fecha_referencia` (vs el peso a la fecha en que
    se formuló la dieta). Si no hay ADPV cargado o no hay fecha de
    dieta, el factor queda en 1.0 (cálculo estático original).

    Returns:
        Float kg/día. 0.0 si no hay dieta o no se puede calcular.
    """
    from . import database as db
    from datetime import datetime as _dt
    if not fecha_referencia:
        fecha_referencia = _dt.now().strftime("%Y-%m-%d")

    dietas = db.listar_dietas(lote_id)
    if not dietas:
        return 0.0
    dieta = _dieta_vigente(dietas, fecha_referencia) or dietas[-1]
    if not dieta:
        return 0.0
    composicion = dieta.get("composicion") or []

    # Si el lote tiene modalidad "aparte" (rollo en otro corral),
    # excluir del cálculo los ingredientes a libre disposición. La
    # carga del silo NO incluye esos forrajes. Si modalidad es
    # "mezclado" o no está definida, sumamos todo.
    try:
        lote = db.obtener_lote(lote_id)
    except Exception:
        lote = None
    _modal = (
        (lote.get("forraje_modalidad") if lote else None)
        or "mezclado"
    ).lower()

    if _modal == "aparte":
        kg_tc_por_animal = sum(
            float(c.get("kg_tal_cual") or 0)
            for c in composicion
            if not _es_a_discrecion(c.get("nombre", ""))
        )
    else:
        kg_tc_por_animal = sum(
            float(c.get("kg_tal_cual") or 0) for c in composicion
        )

    if kg_tc_por_animal <= 0:
        return 0.0
    cantidad = db.cantidad_vigente_lote(lote_id, fecha_referencia)
    if cantidad <= 0:
        return 0.0

    # Escala por peso vivo proyectado (ADG): si hoy el animal pesa más
    # que cuando se formuló la dieta, consume proporcionalmente más.
    factor = 1.0
    if lote:
        factor, _ = factor_escala_consumo_pv(
            lote, dieta, fecha_referencia,
        )
    return round(kg_tc_por_animal * cantidad * factor, 2)


def desglose_carga_silocomedero(
    lote_id: int, dias_carga: int,
    fecha_referencia: Optional[str] = None,
    forraje_modalidad: Optional[str] = None,
) -> Optional[Dict]:
    """Desglose de la carga que hay que preparar para el silocomedero
    en función de los días que tiene que durar la carga.

    Toma la dieta vigente del lote, multiplica el kg/animal/día de cada
    ingrediente por la cantidad de animales vigente y por los días de
    duración. Si `forraje_modalidad="aparte"`, los forrajes
    (rollo/silo/pastura) se separan del cálculo de mezcla y se
    devuelven en `forrajes_aparte` para que el productor los considere
    en su corral separado.

    Args:
        lote_id: id del lote.
        dias_carga: cuántos días tiene que durar la carga (ej. 4-5).
        fecha_referencia: ISO date. Default hoy.
        forraje_modalidad: 'mezclado' (todo va al silocomedero),
            'aparte' (forrajes separados), o None (mezclado por
            default).

    Returns:
        Dict con:
          - dias_carga, cantidad_animales, fecha_dieta
          - kg_total_mezcla (sólo los que SÍ van al silocomedero)
          - kg_total_por_animal (= kg_total_mezcla / cantidad_animales)
          - ingredientes: lista de {nombre, kg_tal_cual_por_animal_dia,
              kg_total, pct_mezcla} — los que VAN al silocomedero
          - forrajes_aparte: lista con la misma estructura — los que
              quedan a libre disposición en corral aparte
          - observaciones_dieta
        None si no hay dieta cargada o no hay cantidad de animales.
    """
    from . import database as db
    from datetime import datetime as _dt
    if not fecha_referencia:
        fecha_referencia = _dt.now().strftime("%Y-%m-%d")
    if dias_carga <= 0:
        return None

    dietas = db.listar_dietas(lote_id)
    if not dietas:
        return None
    dieta = _dieta_vigente(dietas, fecha_referencia) or dietas[-1]
    if not dieta:
        return None

    composicion = dieta.get("composicion") or []
    if not composicion:
        return None

    cantidad = db.cantidad_vigente_lote(lote_id, fecha_referencia) or 0
    if cantidad <= 0:
        return None

    _modal = (forraje_modalidad or "mezclado").lower()

    # Escala por peso vivo proyectado (ADG): la dieta original se
    # multiplica por (peso_actual / peso_a_la_fecha_de_dieta).
    try:
        lote_obj = db.obtener_lote(lote_id)
    except Exception:
        lote_obj = None
    factor_pv = 1.0
    info_pv = {}
    if lote_obj:
        factor_pv, info_pv = factor_escala_consumo_pv(
            lote_obj, dieta, fecha_referencia,
        )

    # Particionar ingredientes en "van al silo" vs "aparte"
    # Aplicar factor de escala al kg/animal/día.
    en_silo, aparte = [], []
    for c in composicion:
        kg_an_dia_raw = float(c.get("kg_tal_cual") or 0)
        if kg_an_dia_raw <= 0:
            continue
        kg_an_dia = kg_an_dia_raw * factor_pv
        es_forr = _es_a_discrecion(c.get("nombre", ""))
        if es_forr and _modal == "aparte":
            aparte.append((c, kg_an_dia))
        else:
            en_silo.append((c, kg_an_dia))

    kg_tc_total_animal = sum(kg for _, kg in en_silo)
    if kg_tc_total_animal <= 0:
        return None

    kg_total_mezcla = round(
        kg_tc_total_animal * cantidad * dias_carga, 1
    )

    def _formatear(items):
        out = []
        total = sum(kg for _, kg in items) or 1
        for c, kg in items:
            kg_total = round(kg * cantidad * dias_carga, 1)
            out.append({
                "nombre": c.get("nombre", "—"),
                "kg_tal_cual_por_animal_dia": round(kg, 3),
                "kg_total": kg_total,
                "pct_mezcla": round(kg / total * 100, 1),
            })
        out.sort(key=lambda x: -x["kg_total"])
        return out

    return {
        "dias_carga": int(dias_carga),
        "cantidad_animales": int(cantidad),
        "fecha_dieta": dieta.get("fecha", "—"),
        "kg_total_mezcla": kg_total_mezcla,
        "kg_total_por_animal": round(
            kg_tc_total_animal * dias_carga, 2
        ),
        "ingredientes": _formatear(en_silo),
        "forrajes_aparte": _formatear(aparte),
        "observaciones_dieta": dieta.get("observaciones", ""),
        "forraje_modalidad": _modal,
        "escala_pv": info_pv,
    }


def proyectar_fin_carga_silocomedero(
    lote_id: int, fecha_referencia: Optional[str] = None,
) -> Optional[Dict]:
    """Calcula cuándo se va a agotar la carga actual del silocomedero.

    Fórmula:
        kg_consumidos = consumo_diario_mezcla × días desde la carga
        kg_restantes = kg_cargados - kg_consumidos
        días_restantes = kg_restantes / consumo_diario_mezcla
        fecha_agotamiento = fecha_referencia + días_restantes

    Args:
        lote_id: id del lote (debe tener tipo_comedero='silocomedero').
        fecha_referencia: ISO date. Default hoy.

    Returns:
        Dict con:
          - kg_cargados, fecha_carga
          - consumo_diario_kg
          - kg_consumidos_acumulados (desde la carga hasta hoy)
          - kg_restantes
          - dias_restantes (int, redondeado para abajo)
          - fecha_agotamiento (ISO)
        None si no hay carga registrada o no se puede calcular.
    """
    from . import database as db
    if not fecha_referencia:
        fecha_referencia = datetime.now().strftime("%Y-%m-%d")
    fecha_ref = datetime.strptime(fecha_referencia, "%Y-%m-%d").date()

    # Tomar la última carga, pero SUMAR todas las cargas del MISMO DÍA
    # (el productor puede haber cargado en 2 viajes: 6660 + 451). Si
    # fueran cargas independientes de días distintos, asumimos que la
    # carga vieja ya se consumió y solo cuenta la última (semántica
    # original del silocomedero — cada carga reemplaza la anterior).
    ultima = db.ultima_carga_silocomedero(lote_id)
    if not ultima:
        return None

    try:
        fecha_carga = datetime.strptime(
            ultima["fecha_carga"][:10], "%Y-%m-%d"
        ).date()
    except (ValueError, TypeError):
        return None

    # Filtrar cargas del MISMO día (excluyendo cargas de rollo)
    try:
        todas = db.listar_cargas_silocomedero(lote_id) or []
    except Exception:
        todas = []
    fecha_carga_iso = fecha_carga.isoformat()
    kg_cargados = 0.0
    for c in todas:
        if (c.get("fecha_carga") or "")[:10] != fecha_carga_iso:
            continue
        if (c.get("tipo_carga") or "").lower() == "rollo_libre":
            continue
        kg_cargados += float(c.get("kg_cargados") or 0)
    # Fallback si por alguna razón no encontró nada
    if kg_cargados <= 0:
        kg_cargados = float(ultima["kg_cargados"] or 0)
    if kg_cargados <= 0:
        return None

    # Consumo diario hoy (dieta vigente × cantidad vigente)
    consumo_dia = _consumo_diario_mezcla_kg(lote_id, fecha_referencia)
    if consumo_dia <= 0:
        return None

    # Cuántos días pasaron desde la carga (mínimo 0)
    dias_pasados = max(0, (fecha_ref - fecha_carga).days)
    kg_consumidos = min(kg_cargados, consumo_dia * dias_pasados)
    kg_restantes = max(0.0, kg_cargados - kg_consumidos)

    if kg_restantes <= 0:
        # Ya se agotó la carga
        dias_restantes = 0
        fecha_agot = fecha_ref.isoformat()
    else:
        dias_restantes_f = kg_restantes / consumo_dia
        dias_restantes = int(dias_restantes_f)  # truncado hacia abajo
        fecha_agot = (
            fecha_ref + timedelta(days=dias_restantes)
        ).isoformat()

    return {
        "kg_cargados": round(kg_cargados, 1),
        "fecha_carga": fecha_carga.isoformat(),
        "consumo_diario_kg": round(consumo_dia, 1),
        "kg_consumidos_acumulados": round(kg_consumidos, 1),
        "kg_restantes": round(kg_restantes, 1),
        "dias_restantes": dias_restantes,
        "fecha_agotamiento": fecha_agot,
    }


def lotes_silocomedero_proximos_agotamiento(
    umbral_dias: int = 1,
) -> List[Dict]:
    """Lotes activos con silocomedero cuya carga actual se agota en
    `umbral_dias` o menos. Para alimentar el cron diario.

    Args:
        umbral_dias: avisar si quedan ≤ esta cantidad de días. Default 1.

    Returns:
        Lista de dicts agrupados por cliente:
        {
            "cliente": {dict},
            "lotes": [
                {
                    "lote_id", "lote_ident", "categoria",
                    "kg_cargados", "kg_restantes",
                    "consumo_diario_kg", "dias_restantes",
                    "fecha_agotamiento", "fecha_carga"
                },
                ...
            ]
        }
    """
    from . import database as db
    out = []
    for c in db.listar_clientes():
        if (c.get("estado") or "activo") != "activo":
            continue
        lotes_alerta = []
        lotes = db.listar_lotes(cliente_id=c["id"], estado="activo")
        for l in lotes:
            # Solo lotes con silocomedero — los lineales/autoconsumo
            # no usan este flujo de alerta.
            if (l.get("tipo_comedero_concentrado") or "") != "silocomedero":
                continue
            try:
                proy = proyectar_fin_carga_silocomedero(l["id"])
            except Exception:
                continue
            if not proy:
                continue
            if proy["dias_restantes"] > umbral_dias:
                continue
            lotes_alerta.append({
                "lote_id": l["id"],
                "lote_ident": l.get("identificador") or "",
                "categoria": l.get("categoria") or "",
                **proy,
            })
        if lotes_alerta:
            lotes_alerta.sort(key=lambda x: x["dias_restantes"])
            out.append({"cliente": c, "lotes": lotes_alerta})
    out.sort(
        key=lambda x: x["lotes"][0]["dias_restantes"]
        if x["lotes"] else 999,
    )
    return out


# =====================================================================
# SERIE TEMPORAL DE CONSUMO DE MS (para gráficos)
# =====================================================================

def serie_consumo_ms_lote(
    lote_id: int,
    fecha_desde: Optional[str] = None,
    fecha_hasta: Optional[str] = None,
    paso_dias: int = 1,
) -> List[Dict]:
    """Serie temporal del consumo de materia seca (DMI) del lote.

    Para cada día entre fecha_desde y fecha_hasta calcula:
      - peso vivo proyectado del animal
      - dieta vigente
      - kg MS/animal/día (escalado por PV proyectado vs PV en fecha
        de formulación de la dieta — el mismo factor que usa
        _consumo_diario_mezcla_kg)
      - kg MS/lote/día (× cantidad de animales vigente)

    Útil para graficar la evolución del consumo a lo largo del
    engorde y visualizar cómo aumenta con el peso vivo.

    Args:
        lote_id: id del lote.
        fecha_desde: ISO date. Default = fecha_ingreso del lote.
        fecha_hasta: ISO date. Default = objetivo_fecha del lote, o
            hoy si no hay objetivo cargado.
        paso_dias: muestrear cada N días. Default 1 (uno por día).
            Para engordes largos podés pasar 7 para alivianar.

    Returns:
        Lista de dicts ordenada por fecha. Cada dict:
        {
          "fecha": "YYYY-MM-DD",
          "dias_desde_ingreso": int,
          "peso_vivo_kg": float,
          "kg_ms_animal_dia": float,
          "kg_ms_lote_dia": float,
          "cantidad_animales": int,
          "factor_escala": float,
          "fecha_dieta_vigente": str,
          "es_proyeccion": bool,  # True si fecha > hoy
        }
        Lista vacía si no se puede calcular (sin dieta, sin fechas).
    """
    from . import database as db
    from datetime import datetime as _dt, timedelta

    lote = db.obtener_lote(lote_id)
    if not lote:
        return []

    dietas = db.listar_dietas(lote_id)
    if not dietas:
        return []

    fi_raw = (lote.get("fecha_ingreso") or "")[:10]
    hoy = _dt.now().strftime("%Y-%m-%d")
    obj_raw = (lote.get("objetivo_fecha") or "")[:10]

    fd = fecha_desde or fi_raw or hoy
    fh = fecha_hasta or obj_raw or hoy
    try:
        d_ini = _dt.strptime(fd, "%Y-%m-%d").date()
        d_fin = _dt.strptime(fh, "%Y-%m-%d").date()
        d_hoy = _dt.strptime(hoy, "%Y-%m-%d").date()
        try:
            d_ing = _dt.strptime(fi_raw, "%Y-%m-%d").date()
        except Exception:
            d_ing = d_ini
    except Exception:
        return []

    if d_fin < d_ini:
        return []

    paso = max(1, int(paso_dias))
    serie = []
    d = d_ini
    while d <= d_fin:
        fecha_str = d.isoformat()
        dieta = _dieta_vigente(dietas, fecha_str) or dietas[-1]
        if not dieta:
            d += timedelta(days=paso)
            continue

        # Consumo MS TOTAL de la dieta (silo + rollo + cualquier otra
        # cosa a libre disposición). Esto representa el DMI total del
        # animal — lo que efectivamente come, sumando todo. Los
        # puntos reales en el gráfico están separados (silo azul,
        # rollo naranja) para que se vea cuánto aporta cada uno.
        comp = dieta.get("composicion") or []
        kg_ms_base = float(dieta.get("consumo_ms_kg") or 0)
        if kg_ms_base <= 0:
            kg_ms_base = sum(
                float(c.get("kg_ms") or 0) for c in comp
            )

        # Además calculamos cuánto de ese total es SOLO silo (mezcla)
        # para poder comparar con los puntos azules de las cargas
        # reales del silocomedero — el productor ve la suma silo+rollo
        # pero el agente puede separar si necesita.
        modalidad_forr = (
            lote.get("forraje_modalidad") or "mezclado"
        ).lower()
        if modalidad_forr == "aparte" and comp:
            kg_ms_solo_silo = sum(
                float(c.get("kg_ms") or 0) for c in comp
                if not _es_a_discrecion(c.get("nombre", ""))
            )
        else:
            kg_ms_solo_silo = kg_ms_base

        # Factor de escala por peso vivo proyectado
        factor, info = factor_escala_consumo_pv(
            lote, dieta, fecha_str,
        )
        peso_dia = info.get("peso_actual_kg") or 0
        kg_ms_animal = round(kg_ms_base * factor, 2)
        kg_ms_animal_silo = round(kg_ms_solo_silo * factor, 2)

        # Cantidad de animales vigente a esa fecha
        try:
            cantidad = db.cantidad_vigente_lote(lote_id, fecha_str) or 0
        except Exception:
            cantidad = lote.get("cantidad_inicial") or 0

        kg_ms_lote = round(kg_ms_animal * cantidad, 1)

        serie.append({
            "fecha": fecha_str,
            "dias_desde_ingreso": max(0, (d - d_ing).days),
            "peso_vivo_kg": round(peso_dia, 1) if peso_dia else 0,
            "kg_ms_animal_dia": kg_ms_animal,
            "kg_ms_animal_dia_solo_silo": kg_ms_animal_silo,
            "kg_ms_lote_dia": kg_ms_lote,
            "cantidad_animales": int(cantidad),
            "factor_escala": round(factor, 4),
            "fecha_dieta_vigente": dieta.get("fecha", "—"),
            "es_proyeccion": d > d_hoy,
        })
        d += timedelta(days=paso)

    return serie


# =====================================================================
# ROLLO A LIBRE DISPOSICIÓN — fórmula cilíndrica + densidades
# =====================================================================

# Densidades de referencia (kg / m³ tal cual) para rollos redondos
# argentinos típicos. Valores promedio INTA Anguil / Manfredi.
# El campo puede variar ±15% según humedad, prensa de la enrolladora,
# tiempo de almacenaje, etc. — se puede ajustar la densidad manualmente
# al cargar la entrega si el productor tiene mejor referencia.
DENSIDAD_ROLLO_KG_M3 = {
    "alfalfa": 170,
    "pastura_consociada": 155,
    "avena": 150,
    "moha": 140,
    "sorgo_diferido": 135,
    "pastura_natural": 130,
    "cebadilla": 155,
    "trigo": 150,
    "mezcla": 145,  # default cuando no se sabe
}

# % de materia seca típico por tipo de forraje (rollo seco bien
# conservado). Si el rollo está mal henificado o tiene mucha humedad
# residual, el % MS baja → menos kg MS aprovechables.
PCT_MS_ROLLO = {
    "alfalfa": 90,
    "pastura_consociada": 88,
    "avena": 88,
    "moha": 87,
    "sorgo_diferido": 89,
    "pastura_natural": 85,
    "cebadilla": 88,
    "trigo": 88,
    "mezcla": 88,
}

# % de desperdicio típico según el sistema de oferta del rollo.
# Es lo que se pisotea, defeca encima, descompone o queda sin comer.
DESPERDICIO_ROLLO_DEFAULT = {
    "sin_parrilla": 25,
    "parrillon_circular": 10,
    "comedero_con_barrera": 7,
}


def calcular_peso_rollo(
    diametro_m: float,
    ancho_m: float,
    tipo_forraje: str = "mezcla",
    densidad_kg_m3: Optional[float] = None,
    pct_ms: Optional[float] = None,
) -> Dict:
    """Calcula el peso estimado de un rollo cilíndrico.

    Fórmula:
        Vol  = π × (D/2)² × ancho
        TC   = Vol × densidad
        MS   = TC × (%MS / 100)

    Args:
        diametro_m: diámetro de la cara del rollo (1.0-2.0 m típico).
        ancho_m: largo del cilindro (1.0-2.0 m típico).
        tipo_forraje: clave de DENSIDAD_ROLLO_KG_M3.
        densidad_kg_m3: override manual si el productor tiene un
            valor mejor (sino se usa la tabla).
        pct_ms: override manual del % MS.

    Returns:
        {
          "volumen_m3", "densidad_kg_m3", "pct_ms",
          "peso_tal_cual_kg", "peso_ms_kg",
          "tipo_forraje", "rango_tc_kg" (±15% para incertidumbre)
        }
    """
    import math
    if diametro_m <= 0 or ancho_m <= 0:
        return {
            "error": "Diámetro y ancho deben ser positivos.",
        }
    tipo = (tipo_forraje or "mezcla").lower().strip()
    dens = (
        densidad_kg_m3
        if densidad_kg_m3 and densidad_kg_m3 > 0
        else DENSIDAD_ROLLO_KG_M3.get(tipo, 145)
    )
    pct = (
        pct_ms
        if pct_ms and pct_ms > 0
        else PCT_MS_ROLLO.get(tipo, 88)
    )

    vol = math.pi * (diametro_m / 2) ** 2 * ancho_m
    peso_tc = vol * dens
    peso_ms = peso_tc * (pct / 100)

    return {
        "volumen_m3": round(vol, 3),
        "densidad_kg_m3": round(dens, 1),
        "pct_ms": round(pct, 1),
        "peso_tal_cual_kg": round(peso_tc, 1),
        "peso_ms_kg": round(peso_ms, 1),
        "tipo_forraje": tipo,
        "rango_tc_kg_min": round(peso_tc * 0.85, 1),
        "rango_tc_kg_max": round(peso_tc * 1.15, 1),
    }


def serie_cargas_reales_ms(
    lote_id: int, incluir_rollo: bool = False,
) -> List[Dict]:
    """Serie de cargas REALES del lote convertidas a kg MS/animal/día.

    Para cada día con carga registrada (lineal o silocomedero), busca
    la dieta vigente, calcula el ratio MS/tal cual de esa dieta y
    convierte los kg cargados a equivalente en materia seca. Después
    divide por la cantidad de animales y los días que cubre la carga.

    Esto permite superponer la curva de "consumo real" sobre la curva
    de "consumo proyectado" en la misma escala (kg MS/animal/día).

    Args:
        lote_id: id del lote.
        incluir_rollo: si True, incluye también cargas de tipo
            'rollo_libre'. Si False (default), solo silo/lineal —
            para mantener comparabilidad con la curva proyectada
            cuando modalidad_forraje='aparte'.

    Args:
        lote_id: id del lote.

    Returns:
        Lista de dicts ordenada por fecha, una entrada por DÍA (si hay
        varias cargas el mismo día se suman). Cada dict:
        {
          "fecha": "YYYY-MM-DD",
          "kg_cargados_tal_cual": float,
          "kg_ms_animal_dia_real": float,
          "kg_ms_lote_dia_real": float,
          "cantidad_animales": int,
          "dias_cubiertos": float,
          "n_cargas_dia": int,
          "tipo_carga": str,
        }
        Lista vacía si no hay cargas registradas.
    """
    from . import database as db

    try:
        cargas = db.listar_cargas_silocomedero(lote_id) or []
    except Exception:
        cargas = []
    if not cargas:
        return []

    # Filtrar las cargas de rollo a libre disposición — esas se
    # trackean por separado con serie_cargas_rollo_lote().
    if not incluir_rollo:
        cargas = [
            c for c in cargas
            if (c.get("tipo_carga") or "").lower() != "rollo_libre"
        ]
    if not cargas:
        return []

    # Agrupar por día (puede haber varias cargas en un mismo día,
    # típico en comedero lineal con 2 comidas)
    cargas_dia = agrupar_cargas_por_dia(cargas)
    if not cargas_dia:
        return []

    dietas = db.listar_dietas(lote_id) or []
    if not dietas:
        return []

    serie = []
    for cdia in cargas_dia:
        fecha = (cdia.get("fecha_carga") or "")[:10]
        if not fecha:
            continue

        kg_tc = float(cdia.get("kg_cargados") or 0)
        if kg_tc <= 0:
            continue

        dias_cub = float(cdia.get("dias_cubiertos") or 1) or 1.0

        # Dieta vigente en esa fecha para sacar el ratio MS/tal cual.
        dieta = _dieta_vigente(dietas, fecha) or dietas[-1]
        if not dieta:
            continue

        composicion = dieta.get("composicion") or []
        # Solo ingredientes de mezcla (no libre disposición), porque
        # las cargas reales son lo que se cargó al silo/comedero.
        kg_ms_dieta = 0.0
        kg_tc_dieta = 0.0
        for c in composicion:
            if _es_a_discrecion(c.get("nombre", "")):
                continue
            kg_ms_dieta += float(c.get("kg_ms") or 0)
            kg_tc_dieta += float(c.get("kg_tal_cual") or 0)
        ratio_ms = (
            kg_ms_dieta / kg_tc_dieta if kg_tc_dieta > 0 else 0.88
        )

        kg_ms_total = kg_tc * ratio_ms

        # Cantidad vigente en la fecha de la carga
        try:
            cantidad = db.cantidad_vigente_lote(lote_id, fecha) or 0
        except Exception:
            cantidad = 0
        if cantidad <= 0:
            continue

        kg_ms_animal_dia = round(
            kg_ms_total / cantidad / dias_cub, 2
        )
        kg_ms_lote_dia = round(kg_ms_total / dias_cub, 1)

        serie.append({
            "fecha": fecha,
            "kg_cargados_tal_cual": round(kg_tc, 1),
            "kg_ms_animal_dia_real": kg_ms_animal_dia,
            "kg_ms_lote_dia_real": kg_ms_lote_dia,
            "cantidad_animales": int(cantidad),
            "dias_cubiertos": dias_cub,
            "n_cargas_dia": int(cdia.get("n_subcargas") or 1),
            "tipo_carga": cdia.get("tipo_carga") or "—",
            "ratio_ms_aplicado": round(ratio_ms, 3),
        })

    serie.sort(key=lambda x: x["fecha"])
    return serie


def serie_cargas_rollo_lote(lote_id: int) -> List[Dict]:
    """Serie de entregas de ROLLO a libre disposición convertidas a
    kg MS APROVECHADO/animal/día (descuenta desperdicio).

    Las cargas de rollo se distinguen del silo por tipo_carga =
    'rollo_libre'. Los metadatos del rollo (cantidad, dimensiones,
    tipo forraje, % desperdicio) se guardan en
    desglose_ingredientes como una sola entrada con campos
    extendidos (cantidad_rollos, diametro_m, ancho_m, tipo_forraje,
    pct_ms, desperdicio_pct, peso_unitario_kg).

    El cálculo:
        kg_ms_aprovechado = kg_cargados × (%MS/100) × (1 − desperdicio/100)
        kg_ms_animal_dia  = kg_ms_aprovechado / cabezas / días_hasta_próxima

    Returns:
        Lista de dicts por día de entrega, ordenada cronológicamente:
        {
          "fecha": "YYYY-MM-DD",
          "tipo_forraje": "alfalfa",
          "cantidad_rollos": 4,
          "kg_cargados_tal_cual": 1880.0,
          "kg_ms_aprovechado": 1410.0,
          "kg_ms_animal_dia_real": 1.65,
          "kg_ms_lote_dia_real": 62.7,
          "cantidad_animales": 38,
          "dias_cubiertos": 14.0,
          "pct_ms_aplicado": 90,
          "desperdicio_pct": 25,
        }
    """
    from . import database as db

    try:
        cargas = db.listar_cargas_silocomedero(lote_id) or []
    except Exception:
        cargas = []
    # Solo cargas de rollo
    cargas_rollo = [
        c for c in cargas
        if (c.get("tipo_carga") or "").lower() == "rollo_libre"
    ]
    if not cargas_rollo:
        return []

    serie = []
    for c in cargas_rollo:
        fecha = (c.get("fecha_carga") or "")[:10]
        if not fecha:
            continue
        kg_tc = float(c.get("kg_cargados") or 0)
        if kg_tc <= 0:
            continue
        dias_cub = float(c.get("dias_cubiertos") or 1) or 1.0

        # Sacar metadata del desglose
        desglose = c.get("desglose_ingredientes") or []
        meta = desglose[0] if desglose else {}
        tipo_forr = (meta.get("tipo_forraje") or "mezcla").lower()
        pct_ms = float(
            meta.get("pct_ms") or PCT_MS_ROLLO.get(tipo_forr, 88)
        )
        despe = float(meta.get("desperdicio_pct") or 25)
        cantidad_rollos = int(meta.get("cantidad_rollos") or 0)

        kg_ms_total = kg_tc * (pct_ms / 100)
        kg_ms_aprov = kg_ms_total * (1 - despe / 100)

        try:
            cabezas = db.cantidad_vigente_lote(lote_id, fecha) or 0
        except Exception:
            cabezas = 0
        if cabezas <= 0:
            continue

        kg_ms_animal_dia = round(
            kg_ms_aprov / cabezas / dias_cub, 2
        )
        kg_ms_lote_dia = round(kg_ms_aprov / dias_cub, 1)

        serie.append({
            "fecha": fecha,
            "tipo_forraje": tipo_forr,
            "cantidad_rollos": cantidad_rollos,
            "kg_cargados_tal_cual": round(kg_tc, 1),
            "kg_ms_aprovechado": round(kg_ms_aprov, 1),
            "kg_ms_animal_dia_real": kg_ms_animal_dia,
            "kg_ms_lote_dia_real": kg_ms_lote_dia,
            "cantidad_animales": int(cabezas),
            "dias_cubiertos": dias_cub,
            "pct_ms_aplicado": round(pct_ms, 1),
            "desperdicio_pct": round(despe, 1),
        })

    serie.sort(key=lambda x: x["fecha"])
    return serie


# =====================================================================
# DEMANDA CONSOLIDADA POR CLIENTE
# =====================================================================

def demanda_insumos_cliente(
    cliente_id: int, fecha_referencia: Optional[str] = None,
) -> Dict:
    """Calcula la demanda diaria de cada insumo para un cliente,
    abriéndola por lote/corral y consolidando un total cliente.

    Para cada lote activo del cliente:
      - Toma la dieta VIGENTE en la fecha de referencia (respeta plan
        de adaptación de varias fases).
      - Toma la cantidad de animales VIGENTE en esa fecha (respeta
        movimientos de hacienda).
      - Por ingrediente: kg_animal_dia × cantidad = kg_lote_dia.
      - Marca si es producto HMS (tiene entregas registradas para
        este cliente) y si es a libre disposición.

    Args:
        cliente_id: id del cliente.
        fecha_referencia: ISO date (default hoy).

    Returns:
        {
          "cliente_id": int,
          "fecha_referencia": ISO,
          "lotes": [
            {
              "lote_id", "lote_ident", "categoria",
              "cantidad_animales": int,
              "fase_vigente": str (observaciones de la dieta),
              "fecha_dieta": str,
              "ingredientes": [
                {"nombre", "kg_animal_dia", "kg_lote_dia",
                 "kg_lote_semana", "es_hms": bool,
                 "es_libre_disposicion": bool},
                ...
              ],
              "mezcla_total_kg_dia": float,  # excluye libre disposición
            },
            ...
          ],
          "total_cliente": {
            "ingredientes": [
              {"nombre", "kg_dia", "kg_semana", "es_hms",
               "es_libre_disposicion", "lotes_que_lo_usan": int},
              ...
            ],
            "mezcla_total_kg_dia": float,
            "cantidad_animales_total": int,
          }
        }
    """
    from . import database as db
    from datetime import datetime as _dt
    if not fecha_referencia:
        fecha_referencia = _dt.now().strftime("%Y-%m-%d")

    # Productos que HMS le vendió alguna vez al cliente → marcamos
    # esos como "es_hms" en la tabla, para que Mauricio sepa cuáles
    # tiene que coordinar entregar (los otros los compra el productor
    # por su lado).
    entregas = db.listar_entregas_cliente(cliente_id, limit=500)
    productos_hms = {
        (e.get("producto_nombre") or "").strip().lower()
        for e in entregas if e.get("producto_nombre")
    }

    def _es_hms(nombre: str) -> bool:
        if not nombre:
            return False
        nm_lower = nombre.strip().lower()
        # Match exacto + tolerante con _mismo_producto
        for hms in productos_hms:
            if _mismo_producto(nombre, hms):
                return True
        return False

    # Wrapper local que delega en el helper módulo-level
    # estimar_peso_vivo_lote (definido arriba en este mismo archivo).
    def _estimar_peso_vivo_lote(lote: Dict, fecha_ref: str) -> float:
        return estimar_peso_vivo_lote(lote, fecha_ref)

    lotes = db.listar_lotes(cliente_id=cliente_id, estado="activo")
    lotes_out = []
    # Acumulador global por nombre normalizado de ingrediente
    total_acum = {}  # nombre_lower → {nombre, kg_dia, es_hms, ...}

    for l in lotes:
        dietas = db.listar_dietas(l["id"])
        if not dietas:
            continue
        dieta = (
            _dieta_vigente(dietas, fecha_referencia)
            or dietas[-1]
        )
        if not dieta:
            continue
        cantidad = db.cantidad_vigente_lote(l["id"], fecha_referencia)
        if cantidad <= 0:
            continue
        composicion = dieta.get("composicion") or []

        # Factor de escala por peso vivo proyectado (ADG). Si la dieta
        # se formuló con peso X y hoy pesan Y, el consumo escala Y/X.
        factor_pv, _info_pv = factor_escala_consumo_pv(
            l, dieta, fecha_referencia,
        )

        ingredientes_lote = []
        mezcla_total_dia = 0.0
        for c in composicion:
            nombre = (
                c.get("nombre") or c.get("ingrediente") or "?"
            ).strip()
            if not nombre:
                continue
            kg_animal = float(c.get("kg_tal_cual") or 0) * factor_pv
            kg_lote_dia = kg_animal * cantidad
            es_libre = _es_a_discrecion(nombre)
            es_hms = _es_hms(nombre)
            if not es_libre:
                mezcla_total_dia += kg_lote_dia
            ingredientes_lote.append({
                "nombre": nombre,
                "kg_animal_dia": round(kg_animal, 2),
                "kg_lote_dia": round(kg_lote_dia, 1),
                "kg_lote_semana": round(kg_lote_dia * 7, 1),
                "es_hms": es_hms,
                "es_libre_disposicion": es_libre,
                # pct_mezcla se calcula en el segundo paso, cuando ya
                # conocemos la suma total de la mezcla del lote.
                "pct_mezcla": 0.0,
            })

            # Acumular en total cliente. Para los a libre disposición
            # solo sumamos kg si los tenemos (orientativo). Para
            # productos HMS la suma sí es operativa.
            key = nombre.lower()
            if key not in total_acum:
                total_acum[key] = {
                    "nombre": nombre,
                    "kg_dia": 0.0,
                    "es_hms": es_hms,
                    "es_libre_disposicion": es_libre,
                    "lotes_que_lo_usan": 0,
                }
            total_acum[key]["kg_dia"] += kg_lote_dia
            total_acum[key]["lotes_que_lo_usan"] += 1
            # Si en algún lote es HMS, queda marcado HMS (los productos
            # HMS no cambian de cliente a cliente).
            if es_hms:
                total_acum[key]["es_hms"] = True

        # Segundo paso: calcular % de inclusión sobre la mezcla del lote
        # (excluyendo los ingredientes a libre disposición). Esto es lo
        # que el productor entiende como "% en la receta".
        if mezcla_total_dia > 0:
            for ing in ingredientes_lote:
                if ing["es_libre_disposicion"]:
                    continue
                ing["pct_mezcla"] = round(
                    ing["kg_lote_dia"] / mezcla_total_dia * 100, 1
                )

        # KPIs por animal: kg de mezcla por cabeza y % del peso vivo.
        # kg_mezcla_animal = suma de kg_animal_dia de ingredientes que NO
        # son a libre disposición (lo que se prepara como mezcla).
        kg_mezcla_animal = sum(
            (ing["kg_animal_dia"] or 0)
            for ing in ingredientes_lote
            if not ing["es_libre_disposicion"]
        )
        peso_vivo = _estimar_peso_vivo_lote(l, fecha_referencia)
        pct_pv_mezcla = (
            round(kg_mezcla_animal / peso_vivo * 100, 2)
            if peso_vivo > 0 else 0.0
        )

        lotes_out.append({
            "lote_id": l["id"],
            "lote_ident": l.get("identificador") or "",
            "categoria": l.get("categoria") or "",
            "cantidad_animales": cantidad,
            "fase_vigente": dieta.get("observaciones") or "",
            "fecha_dieta": dieta.get("fecha") or "",
            "ingredientes": ingredientes_lote,
            "mezcla_total_kg_dia": round(mezcla_total_dia, 1),
            "kg_mezcla_animal_dia": round(kg_mezcla_animal, 2),
            "peso_vivo_estimado_kg": round(peso_vivo, 0),
            "pct_pv_mezcla": pct_pv_mezcla,
        })

    # Armar total cliente ordenado: primero HMS (lo que él coordina),
    # después el resto. Dentro de cada grupo, por kg_dia desc.
    total_lista = []
    for key, item in total_acum.items():
        total_lista.append({
            "nombre": item["nombre"],
            "kg_dia": round(item["kg_dia"], 1),
            "kg_semana": round(item["kg_dia"] * 7, 1),
            "kg_mes": round(item["kg_dia"] * 30, 1),
            "es_hms": item["es_hms"],
            "es_libre_disposicion": item["es_libre_disposicion"],
            "lotes_que_lo_usan": item["lotes_que_lo_usan"],
        })
    total_lista.sort(
        key=lambda x: (
            not x["es_hms"],  # HMS primero
            x["es_libre_disposicion"],  # medibles antes que libres
            -x["kg_dia"],  # más volumen primero
        )
    )

    mezcla_total_cliente = sum(
        x["kg_dia"] for x in total_lista
        if not x["es_libre_disposicion"]
    )
    animales_total = sum(l["cantidad_animales"] for l in lotes_out)

    return {
        "cliente_id": cliente_id,
        "fecha_referencia": fecha_referencia,
        "lotes": lotes_out,
        "total_cliente": {
            "ingredientes": total_lista,
            "mezcla_total_kg_dia": round(mezcla_total_cliente, 1),
            "cantidad_animales_total": animales_total,
        },
    }


# =====================================================================
# CAMBIO DE FASE DEL PLAN DE ADAPTACIÓN
# =====================================================================

# Palabras clave que identifican un ingrediente como forraje a libre
# disposición. En la práctica argentina, estos no se preparan en kg
# precisos — se dejan en el corral y el animal elige cuánto consume.
# La estimación que viene en la dieta es orientativa, no operativa.
_KEYWORDS_LIBRE_DISPOSICION = [
    "rollo", "rolo", "fardo", "silaje", "silo de", "henolaje",
    "pastura", "pasto", "verdeo", "alfalfa rollos",
]


def _es_a_discrecion(nombre_ingrediente: str) -> bool:
    """True si el ingrediente típicamente se da a libre disposición
    (forrajes groseros donde el animal regula el consumo)."""
    if not nombre_ingrediente:
        return False
    nm = nombre_ingrediente.lower().strip()
    return any(kw in nm for kw in _KEYWORDS_LIBRE_DISPOSICION)


def _diff_composiciones(
    comp_actual: List[Dict], comp_nueva: List[Dict],
) -> List[Dict]:
    """Compara dos composiciones (listas de ingredientes con kg_tal_cual)
    y devuelve los ingredientes que cambian.

    Para cada ingrediente devuelve {ingrediente, kg_actual, kg_nueva,
    delta_kg, delta_pct_ms}. Incluye los que aparecen sólo en una de las
    dos (con valor 0 en la otra).
    """
    def _idx(comp):
        out = {}
        for c in comp or []:
            nombre = (c.get("nombre") or c.get("ingrediente") or "").strip()
            if nombre:
                out[nombre.lower()] = c
        return out

    idx_a = _idx(comp_actual)
    idx_n = _idx(comp_nueva)
    nombres = []
    seen = set()
    # Mantener orden: primero los de la composición nueva, después los
    # que sólo estaban en la actual.
    for c in comp_nueva or []:
        nm = (c.get("nombre") or c.get("ingrediente") or "").strip()
        if nm and nm.lower() not in seen:
            nombres.append(nm)
            seen.add(nm.lower())
    for c in comp_actual or []:
        nm = (c.get("nombre") or c.get("ingrediente") or "").strip()
        if nm and nm.lower() not in seen:
            nombres.append(nm)
            seen.add(nm.lower())

    diff = []
    for nombre in nombres:
        a = idx_a.get(nombre.lower(), {})
        n = idx_n.get(nombre.lower(), {})
        kg_a = float(a.get("kg_tal_cual") or 0)
        kg_n = float(n.get("kg_tal_cual") or 0)
        pct_a = float(a.get("pct_ms") or 0)
        pct_n = float(n.get("pct_ms") or 0)
        # Saltamos los que no cambian
        if abs(kg_a - kg_n) < 0.05 and abs(pct_a - pct_n) < 0.5:
            continue
        diff.append({
            "ingrediente": nombre,
            "kg_actual": round(kg_a, 2),
            "kg_nueva": round(kg_n, 2),
            "delta_kg": round(kg_n - kg_a, 2),
            "pct_actual": round(pct_a, 1),
            "pct_nueva": round(pct_n, 1),
        })
    return diff


def lotes_con_cambio_fase_proximo(
    dias_anticipo: int = 1,
    fecha_referencia: Optional[str] = None,
) -> List[Dict]:
    """Detecta lotes donde mañana (o en `dias_anticipo` días) arranca
    una nueva fase del plan de adaptación — o sea, hay una dieta cuya
    fecha de inicio es exactamente `fecha_referencia + dias_anticipo`.

    Esto sólo dispara una alerta el día previo al cambio (no todos
    los días entre fases), porque el dato que importa al cliente es:
    "mañana cambia la receta, prepará la mezcla nueva".

    Args:
        dias_anticipo: avisar N días antes del cambio. Default 1.
        fecha_referencia: hoy si None.

    Returns:
        Lista por cliente con cambios pendientes:
        {
            "cliente": {dict},
            "cambios": [
                {
                    "lote_id", "lote_ident", "categoria",
                    "fecha_cambio": ISO,
                    "dias_para_cambio": int,
                    "fase_actual": {"fecha", "observaciones",
                                    "composicion", "costo_dia"},
                    "fase_nueva": {...},
                    "diff": [...],
                },
                ...
            ]
        }
    """
    from . import database as db
    from datetime import datetime as _dt, timedelta as _td
    if not fecha_referencia:
        fecha_referencia = _dt.now().strftime("%Y-%m-%d")
    fecha_ref = _dt.strptime(fecha_referencia, "%Y-%m-%d").date()
    fecha_objetivo = (fecha_ref + _td(days=dias_anticipo)).isoformat()

    out = []
    for c in db.listar_clientes():
        if (c.get("estado") or "activo") != "activo":
            continue
        cambios = []
        lotes = db.listar_lotes(cliente_id=c["id"], estado="activo")
        for l in lotes:
            dietas = db.listar_dietas(l["id"])
            if len(dietas) < 2:
                # Sin plan de adaptación (cero o una sola dieta) no hay
                # cambio de fase que avisar.
                continue
            # Dietas ordenadas por fecha ASC para detectar duración
            # de la fase nueva (hasta cuándo va).
            dietas_ord = sorted(
                dietas, key=lambda d: (d.get("fecha") or "")[:10],
            )
            # Buscar dieta que arranca en fecha_objetivo
            fase_nueva = None
            idx_nueva = -1
            for i, d in enumerate(dietas_ord):
                if (d.get("fecha") or "")[:10] == fecha_objetivo:
                    fase_nueva = d
                    idx_nueva = i
                    break
            if not fase_nueva:
                continue
            # Fase actual: la última dieta vigente HOY
            fase_actual = _dieta_vigente(dietas, fecha_referencia)
            if not fase_actual:
                # Si ni siquiera arrancó la fase 1 todavía, no hay
                # comparación útil — saltamos.
                continue
            diff = _diff_composiciones(
                fase_actual.get("composicion") or [],
                fase_nueva.get("composicion") or [],
            )
            # Cantidad de animales vigente HOY — sirve para calcular
            # cuánto hay que preparar de cada ingrediente para todo el
            # lote (kg/animal × cantidad).
            cantidad_animales = db.cantidad_vigente_lote(
                l["id"], fecha_referencia,
            )
            # Duración de la fase nueva:
            #   - Si hay una dieta posterior → va hasta esa fecha - 1
            #   - Si es la última fase → usa lote.objetivo_fecha
            #     (la fecha de salida planificada del lote)
            #   - Si tampoco hay objetivo → queda en None y se muestra
            #     un aviso para que el productor / Mauricio la cargue
            fecha_fin_nueva = None
            duracion_dias = None
            es_ultima = (idx_nueva + 1 >= len(dietas_ord))
            try:
                d_ini = _dt.strptime(
                    fecha_objetivo, "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                d_ini = None

            if not es_ultima and d_ini is not None:
                fecha_inicio_sig = (
                    dietas_ord[idx_nueva + 1].get("fecha") or ""
                )[:10]
                try:
                    d_sig = _dt.strptime(
                        fecha_inicio_sig, "%Y-%m-%d"
                    ).date()
                    if d_sig > d_ini:
                        fecha_fin_nueva = (
                            d_sig - _td(days=1)
                        ).isoformat()
                        duracion_dias = (d_sig - d_ini).days
                except (ValueError, TypeError):
                    pass
            elif es_ultima and d_ini is not None:
                # Última fase: fin = objetivo_fecha del lote
                fecha_obj = (l.get("objetivo_fecha") or "")[:10]
                if fecha_obj:
                    try:
                        d_obj = _dt.strptime(
                            fecha_obj, "%Y-%m-%d"
                        ).date()
                        if d_obj >= d_ini:
                            fecha_fin_nueva = d_obj.isoformat()
                            duracion_dias = (d_obj - d_ini).days + 1
                    except (ValueError, TypeError):
                        pass
            cambios.append({
                "lote_id": l["id"],
                "lote_ident": l.get("identificador") or "",
                "categoria": l.get("categoria") or "",
                "cantidad_animales": cantidad_animales,
                "fecha_cambio": fecha_objetivo,
                "dias_para_cambio": dias_anticipo,
                "fase_actual": {
                    "fecha": fase_actual.get("fecha"),
                    "observaciones": fase_actual.get(
                        "observaciones") or "",
                    "composicion": fase_actual.get(
                        "composicion") or [],
                    "costo_dia": fase_actual.get("costo_dia") or 0,
                },
                "fase_nueva": {
                    "fecha": fase_nueva.get("fecha"),
                    "observaciones": fase_nueva.get(
                        "observaciones") or "",
                    "composicion": fase_nueva.get(
                        "composicion") or [],
                    "costo_dia": fase_nueva.get("costo_dia") or 0,
                    # Hasta cuándo va y cuántos días dura. None si es
                    # la última fase del plan (no se sabe).
                    "fecha_fin": fecha_fin_nueva,
                    "duracion_dias": duracion_dias,
                },
                "diff": diff,
            })
        if cambios:
            out.append({"cliente": c, "cambios": cambios})
    return out


def clientes_con_stock_bajo(
    umbral_dias: int = 14,
) -> List[Dict]:
    """Recorre TODOS los clientes activos y devuelve los que tienen
    al menos un producto HMS con stock por debajo del umbral de días
    de autonomía.

    Args:
        umbral_dias: avisar si quedan ≤ esta cantidad de días. Default 14.

    Returns:
        Lista de dicts, uno por cliente afectado:
        {
            "cliente": {dict del cliente},
            "productos": [  # uno por lote × producto con stock bajo
                {
                    "lote_id": int,
                    "lote_ident": str,
                    "producto": str,
                    "kg_restantes": float,
                    "consumo_kg_dia": float,
                    "dias_restantes": int,
                    "fecha_agotamiento": str (ISO),
                },
                ...
            ],
        }
    """
    from . import database as db
    out = []
    for c in db.listar_clientes():
        if (c.get("estado") or "activo") != "activo":
            continue
        productos_bajos = []
        lotes = db.listar_lotes(
            cliente_id=c["id"], estado="activo",
        )
        for l in lotes:
            productos = listar_productos_hms_lote(c["id"], l["id"])
            for prod in productos:
                try:
                    stk = calcular_stock_actual(
                        c["id"], l["id"], prod,
                    )
                except Exception:
                    continue
                if not stk:
                    continue
                if stk.get(
                    "diagnostico_uso") == "sin_entregas":
                    continue
                dias = stk.get("dias_restantes", 0) or 0
                if dias > umbral_dias:
                    continue
                if (stk.get("kg_restantes_hoy") or 0) <= 0:
                    # Ya agotado — incluir igual
                    pass
                productos_bajos.append({
                    "lote_id": l["id"],
                    "lote_ident": l.get("identificador") or "",
                    "producto": prod,
                    "kg_restantes": round(
                        stk.get("kg_restantes_hoy", 0) or 0, 1),
                    "consumo_kg_dia": round(
                        stk.get("consumo_diario_kg", 0) or 0, 1),
                    "dias_restantes": int(dias),
                    "fecha_agotamiento":
                        stk.get("fecha_agotamiento") or "—",
                })
        if productos_bajos:
            # Ordenar por urgencia (menos días primero)
            productos_bajos.sort(
                key=lambda x: x["dias_restantes"],
            )
            out.append({"cliente": c, "productos": productos_bajos})
    # Ordenar clientes por urgencia (el producto más urgente de cada uno)
    out.sort(
        key=lambda x: x["productos"][0]["dias_restantes"]
        if x["productos"] else 999,
    )
    return out


# =====================================================================
# COMPARACIÓN CARGA REAL vs DIETA FORMULADA
# =====================================================================

def agrupar_cargas_por_dia(cargas: List[Dict]) -> List[Dict]:
    """Agrupa cargas del mismo lote por día calendario.

    Modo flexible: el encargado puede registrar 1, 2 o más cargas el
    mismo día (ej. 2 comidas en comedero lineal). Esta función las
    junta en una sola "carga sintética" por día para comparar contra
    la dieta — la dieta es diaria, así que tiene sentido sumar lo del
    día entero.

    Args:
        cargas: lista de dicts tal como salen de
            db.listar_cargas_silocomedero(). Cada carga tiene
            'fecha_carga', 'kg_cargados', 'hora_carga',
            'desglose_ingredientes', 'dias_cubiertos', etc.

    Returns:
        Lista de dicts uno por día, ordenados de más nuevo a más viejo.
        Cada dict trae:
          - lote_id
          - fecha_carga (sin hora)
          - tipo_carga (el del primer item del día)
          - dias_cubiertos (max del día, usualmente 1)
          - kg_cargados (suma del día)
          - detalles (concatenado de las obs.)
          - desglose_ingredientes (suma por ingrediente)
          - subcargas: lista [{id, hora, kg, obs}, ...] de cada carga
            individual del día, en orden cronológico por hora.
          - n_subcargas: int
    """
    from collections import defaultdict
    grupos = defaultdict(list)
    for c in cargas:
        # Las cargas de rollo a libre disposición NO se agrupan acá:
        # se trackean en una serie aparte (serie_cargas_rollo_lote).
        # Si las mezcláramos, el historial mostraría "real" = silo +
        # rollo y el "esperado" sería solo del silo → desvíos
        # falsos.
        if (c.get("tipo_carga") or "").lower() == "rollo_libre":
            continue
        fecha = (c.get("fecha_carga") or "")[:10]
        grupos[fecha].append(c)

    out = []
    for fecha, items in grupos.items():
        # Ordenar por hora ascendente para subcargas
        items_ord = sorted(
            items,
            key=lambda x: (x.get("hora_carga") or "00:00"),
        )
        # Tomar tipo_carga / lote_id / dias_cubiertos del primer item
        primer = items_ord[0]
        kg_total = sum(
            float(x.get("kg_cargados") or 0) for x in items_ord
        )
        # Sumar desglose por ingrediente
        desglose_acum: Dict[str, float] = {}
        for x in items_ord:
            for ing in (x.get("desglose_ingredientes") or []):
                nm = (ing.get("nombre") or "").strip()
                if not nm:
                    continue
                desglose_acum[nm] = (
                    desglose_acum.get(nm, 0.0) + float(ing.get("kg") or 0)
                )
        desglose_lista = [
            {"nombre": k, "kg": round(v, 1)}
            for k, v in desglose_acum.items()
        ]
        # dias_cubiertos: max (en silocomedero típicamente N>1, en
        # lineal siempre 1).
        dias_cub = max(
            float(x.get("dias_cubiertos") or 1) for x in items_ord
        )
        # Concatenar detalles no vacíos
        obs_items = [
            (x.get("detalles") or "").strip() for x in items_ord
        ]
        obs_items = [o for o in obs_items if o]
        out.append({
            "lote_id": primer.get("lote_id"),
            "fecha_carga": fecha,
            "tipo_carga": primer.get("tipo_carga") or "silo_carga",
            "dias_cubiertos": dias_cub,
            "kg_cargados": round(kg_total, 1),
            "detalles": " · ".join(obs_items),
            "desglose_ingredientes": desglose_lista,
            "subcargas": [
                {
                    "id": x["id"],
                    "hora": x.get("hora_carga") or "—",
                    "kg": round(float(x.get("kg_cargados") or 0), 1),
                    "obs": (x.get("detalles") or ""),
                }
                for x in items_ord
            ],
            "n_subcargas": len(items_ord),
        })
    # Ordenar grupos por fecha desc
    out.sort(key=lambda x: x["fecha_carga"], reverse=True)
    return out


def comparar_carga_vs_dieta(carga: Dict) -> Dict:
    """Compara una carga REAL (lo que el productor metió al comedero)
    contra la dieta VIGENTE en la fecha de la carga.

    Filosofía HMS: que el asesor vea de forma objetiva si la entrega
    diaria está alineada con la receta formulada. Si está 15% arriba o
    abajo del plan, algo pasa — ajustar consumo, revisar pesos, o
    actualizar la dieta.

    Args:
        carga: dict tal como sale de db.listar_cargas_silocomedero():
               - lote_id, fecha_carga, kg_cargados, tipo_carga,
                 dias_cubiertos, desglose_ingredientes (lista opcional).

    Returns:
        Dict con:
          - lote_id, fecha_carga, dias_cubiertos
          - cantidad_animales: vigente en la fecha de la carga
          - esperado_total_kg: kg de mezcla recomendados en la carga
          - real_total_kg: kg cargados realmente
          - desvio_kg, desvio_pct: real - esperado
          - semaforo: 'verde' (±5%) / 'amarillo' (±10%) / 'rojo' (>10%)
          - mensaje: explicación corta legible
          - por_ingrediente: lista de comparaciones por ingrediente si
            hay desglose, sino [].
          - dieta_vigente_fecha: fecha de la dieta usada como referencia
          - sin_dieta: True si no hay dieta vigente (no se puede comparar).
    """
    from . import database as db

    lote_id = carga.get("lote_id")
    fecha_carga = (carga.get("fecha_carga") or "")[:10]
    kg_cargados = float(carga.get("kg_cargados") or 0)
    dias_cubiertos = float(carga.get("dias_cubiertos") or 1) or 1.0
    desglose_real = carga.get("desglose_ingredientes") or []

    # Cantidad de animales VIGENTE en la fecha de la carga (respeta
    # movimientos: muertes, ventas, ingresos posteriores no cuentan).
    cantidad = db.cantidad_vigente_lote(lote_id, fecha_carga)

    # Dieta vigente en esa fecha
    dietas = db.listar_dietas(lote_id) if lote_id else []
    dieta = _dieta_vigente(dietas, fecha_carga) if dietas else None
    if not dieta:
        return {
            "lote_id": lote_id,
            "fecha_carga": fecha_carga,
            "dias_cubiertos": dias_cubiertos,
            "cantidad_animales": cantidad,
            "real_total_kg": round(kg_cargados, 1),
            "esperado_total_kg": 0,
            "desvio_kg": 0,
            "desvio_pct": 0,
            "semaforo": "gris",
            "mensaje": (
                "No hay dieta formulada vigente en la fecha de la "
                "carga. Cargá una dieta para que el sistema pueda "
                "comparar."
            ),
            "por_ingrediente": [],
            "dieta_vigente_fecha": None,
            "sin_dieta": True,
        }

    composicion = dieta.get("composicion") or []

    # Factor de escala por ADG: comparar la carga real contra lo que
    # los animales DEBERÍAN consumir HOY (no contra lo que consumían
    # cuando se formuló la dieta hace 30+ días).
    try:
        lote_obj = db.obtener_lote(lote_id)
    except Exception:
        lote_obj = None
    factor_pv = 1.0
    info_pv = {}
    if lote_obj:
        factor_pv, info_pv = factor_escala_consumo_pv(
            lote_obj, dieta, fecha_carga,
        )

    # kg/animal/día de cada ingrediente NO libre disposición, escalado
    # por peso vivo proyectado.
    mezcla_kg_animal_dia = 0.0
    esperado_por_ing = {}  # nombre_lower → {nombre, kg_animal_dia}
    for c in composicion:
        nombre = (c.get("nombre") or "").strip()
        if not nombre:
            continue
        if _es_a_discrecion(nombre):
            continue
        kg_a = float(c.get("kg_tal_cual") or 0) * factor_pv
        mezcla_kg_animal_dia += kg_a
        esperado_por_ing[nombre.lower()] = {
            "nombre": nombre,
            "kg_animal_dia": kg_a,
        }

    # ─── Detección de carga multi-día (comedero lineal con mixer) ───
    # Si el productor cargó MUCHO más de lo que el lote come en 1 día,
    # asumimos que es una carga del mixer que va a durar varios días.
    # En vez de gritar "+1544% desvío" (lo cual es un falso positivo
    # porque la mezcla NO se consume toda hoy), calculamos cuántos
    # días cubre la carga y comparamos por PROPORCIONES de ingredientes.
    consumo_dia_lote_kg = mezcla_kg_animal_dia * cantidad
    dias_cubiertos_auto = (
        kg_cargados / consumo_dia_lote_kg
        if consumo_dia_lote_kg > 0 else 1.0
    )
    # Threshold: si dias_cubiertos viene en 1 (default) y la carga
    # cubre realmente >2 días, asumir que es carga multi-día.
    es_multi_dia = (
        abs(dias_cubiertos - 1.0) < 0.01
        and dias_cubiertos_auto > 2.0
    )
    if es_multi_dia:
        # Recalcular usando los días que efectivamente cubre.
        dias_cubiertos = round(dias_cubiertos_auto, 1)

    esperado_total = mezcla_kg_animal_dia * cantidad * dias_cubiertos
    desvio_kg = kg_cargados - esperado_total
    desvio_pct = (
        round((desvio_kg / esperado_total) * 100, 1)
        if esperado_total > 0 else 0
    )
    abs_dev = abs(desvio_pct)
    if es_multi_dia:
        # Mensaje principal: cuántos días cubre. El % desvío se mantiene
        # cerca de 0 porque ahora esperado ≈ real (escalado por días).
        # Las proporciones de ingredientes se validan en por_ingrediente.
        semaforo = "verde"
        mensaje = (
            f"✅ Esta carga del mixer cubre {dias_cubiertos:.1f} días de "
            f"dieta. Comparación por proporciones de ingredientes "
            f"abajo (las cantidades totales no se comparan contra UN "
            f"día porque la carga es multi-día)."
        )
    elif abs_dev <= 5:
        semaforo = "verde"
        mensaje = "Carga alineada con la dieta (±5%)."
    elif abs_dev <= 10:
        semaforo = "amarillo"
        if desvio_kg > 0:
            mensaje = (
                f"Se cargó {abs_dev:.1f}% por encima de lo planificado. "
                "Revisar si los animales lo están consumiendo todo."
            )
        else:
            mensaje = (
                f"Se cargó {abs_dev:.1f}% por debajo de lo planificado. "
                "Verificar si quedó comedero vacío o sobró del día anterior."
            )
    else:
        semaforo = "rojo"
        if desvio_kg > 0:
            mensaje = (
                f"⚠️ Se cargó {abs_dev:.1f}% MÁS que la dieta. "
                "Riesgo de desperdicio o acidosis si se consume todo. "
                "Revisar plan o cantidad real de animales."
            )
        else:
            mensaje = (
                f"⚠️ Se cargó {abs_dev:.1f}% MENOS que la dieta. "
                "Riesgo de pérdida de ADG y baja en consumo. "
                "Revisar plan o causa del recorte."
            )

    # Comparación por ingrediente (sólo si hay desglose real)
    por_ingrediente = []
    if desglose_real:
        # Sumar todos los ingredientes reales por nombre normalizado
        # para tolerar repetidos.
        real_por_ing = {}
        for d in desglose_real:
            nm = (d.get("nombre") or "").strip()
            if not nm:
                continue
            kg = float(d.get("kg") or 0)
            key = nm.lower()
            real_por_ing.setdefault(key, {"nombre": nm, "kg": 0.0})
            real_por_ing[key]["kg"] += kg

        # Calcular totales para % de proporción
        total_real_kg = sum(v["kg"] for v in real_por_ing.values())
        total_esp_kg_an_dia = sum(
            v["kg_animal_dia"] for v in esperado_por_ing.values()
        )

        # Recorrer la unión de ingredientes (dieta ∪ real)
        all_keys = set(esperado_por_ing.keys()) | set(real_por_ing.keys())
        for k in all_keys:
            esp = esperado_por_ing.get(k)
            rea = real_por_ing.get(k)
            esperado_kg = (
                (esp["kg_animal_dia"] * cantidad * dias_cubiertos)
                if esp else 0
            )
            real_kg = rea["kg"] if rea else 0
            nombre = (
                (esp or rea or {}).get("nombre", k)
            )
            dev_kg = real_kg - esperado_kg
            dev_pct = (
                round((dev_kg / esperado_kg) * 100, 1)
                if esperado_kg > 0 else
                (100.0 if real_kg > 0 else 0.0)
            )

            # Para carga multi-día, lo que importa es la PROPORCIÓN
            # dentro de la mezcla (no el total absoluto). Calculamos
            # proporción esperada vs real y usamos ESA como semáforo.
            prop_esp = (
                (esp["kg_animal_dia"] / total_esp_kg_an_dia * 100)
                if (esp and total_esp_kg_an_dia > 0) else 0
            )
            prop_real = (
                (real_kg / total_real_kg * 100)
                if (rea and total_real_kg > 0) else 0
            )
            desvio_prop = round(prop_real - prop_esp, 2)
            abs_desvio_prop = abs(desvio_prop)

            ad = abs(dev_pct)
            if es_multi_dia:
                # Semáforo basado en desvío de PROPORCIÓN (puntos %),
                # no en cantidad absoluta. Un cambio de ±2 puntos en
                # la proporción ya es relevante (ej: 88% maíz vs 85%).
                if esp is None and rea is not None and real_kg > 0:
                    sem = "rojo"  # ingrediente no formulado
                elif rea is None and esp is not None and prop_esp > 1:
                    sem = "rojo"  # ingrediente faltante (>1% de la mezcla)
                elif abs_desvio_prop <= 2:
                    sem = "verde"
                elif abs_desvio_prop <= 5:
                    sem = "amarillo"
                else:
                    sem = "rojo"
            else:
                if esperado_kg == 0 and real_kg > 0:
                    sem = "rojo"
                elif ad <= 5:
                    sem = "verde"
                elif ad <= 10:
                    sem = "amarillo"
                else:
                    sem = "rojo"
            por_ingrediente.append({
                "nombre": nombre,
                "esperado_kg": round(esperado_kg, 1),
                "real_kg": round(real_kg, 1),
                "desvio_kg": round(dev_kg, 1),
                "desvio_pct": dev_pct,
                "proporcion_esperada_pct": round(prop_esp, 1),
                "proporcion_real_pct": round(prop_real, 1),
                "desvio_proporcion_pp": desvio_prop,
                "semaforo": sem,
            })
        # Ordenar: rojos primero (los problemas), luego por kg desc.
        sem_order = {"rojo": 0, "amarillo": 1, "verde": 2, "gris": 3}
        por_ingrediente.sort(
            key=lambda x: (sem_order.get(x["semaforo"], 9),
                           -x["real_kg"])
        )

    return {
        "lote_id": lote_id,
        "fecha_carga": fecha_carga,
        "dias_cubiertos": dias_cubiertos,
        "es_multi_dia": es_multi_dia,
        "dias_cubiertos_auto": round(dias_cubiertos_auto, 1),
        "consumo_dia_lote_kg": round(consumo_dia_lote_kg, 1),
        "cantidad_animales": cantidad,
        "real_total_kg": round(kg_cargados, 1),
        "esperado_total_kg": round(esperado_total, 1),
        "desvio_kg": round(desvio_kg, 1),
        "desvio_pct": desvio_pct,
        "semaforo": semaforo,
        "mensaje": mensaje,
        "por_ingrediente": por_ingrediente,
        "dieta_vigente_fecha": dieta.get("fecha"),
        "sin_dieta": False,
        "escala_pv": info_pv,
    }
