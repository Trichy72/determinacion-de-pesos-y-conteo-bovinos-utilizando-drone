"""
Dashboard de inicio — KPIs del rodeo bajo seguimiento.

Calcula y devuelve métricas agregadas sobre todos los lotes activos:
- cantidad de clientes
- cantidad de lotes
- total de animales bajo seguimiento
- última pesada (cuándo, qué cliente, qué peso)
- ADG promedio últimos 30 días
- alertas climáticas activas
- pesadas y dietas del último mes
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from . import database as db


def calcular_kpis() -> Dict:
    """Calcula los KPIs principales del sistema."""
    clientes = db.listar_clientes()
    lotes = db.listar_lotes(estado="activo")

    n_clientes = len(clientes)
    n_lotes = len(lotes)
    n_animales_total = sum(l.get("cantidad_inicial", 0) or 0 for l in lotes)

    # Última pesada de todo el sistema
    ultima_pesada = None
    todas_pesadas = []
    for l in lotes:
        ps = db.listar_pesadas(l["id"])
        for p in ps:
            p["_lote_id"] = l["id"]
            p["_cliente"] = l.get("cliente_nombre", "")
            p["_lote_nombre"] = l.get("identificador", "")
            todas_pesadas.append(p)

    if todas_pesadas:
        todas_pesadas.sort(key=lambda x: x.get("fecha", ""), reverse=True)
        ultima_pesada = todas_pesadas[0]

    # Pesadas del último mes
    hoy = datetime.now().date()
    mes_atras = hoy - timedelta(days=30)
    pesadas_mes = [
        p for p in todas_pesadas
        if p.get("fecha") and
        datetime.strptime(p["fecha"], "%Y-%m-%d").date() >= mes_atras
    ]

    # Dietas del último mes
    todas_dietas = []
    for l in lotes:
        ds = db.listar_dietas(l["id"])
        for d in ds:
            d["_lote_id"] = l["id"]
            d["_cliente"] = l.get("cliente_nombre", "")
            todas_dietas.append(d)
    dietas_mes = [
        d for d in todas_dietas
        if d.get("fecha") and
        datetime.strptime(d["fecha"], "%Y-%m-%d").date() >= mes_atras
    ]

    # ADG promedio del rodeo (lotes con 2+ pesadas)
    adgs = []
    for l in lotes:
        evol = db.calcular_evolucion_lote(l["id"])
        if evol["adg_total"] != 0 and evol["n_pesadas"] >= 2:
            adgs.append(evol["adg_total"])
    adg_promedio = sum(adgs) / len(adgs) if adgs else 0

    # Lotes que necesitan próxima pesada (>30 días sin pesar)
    lotes_a_pesar = []
    for l in lotes:
        ult = l.get("ultima_pesada")
        if not ult:
            lotes_a_pesar.append({
                **l,
                "dias_sin_pesar": None,
                "razon": "Sin pesadas registradas",
            })
        else:
            try:
                fecha_ult = datetime.strptime(ult, "%Y-%m-%d").date()
                dias = (hoy - fecha_ult).days
                if dias >= 30:
                    lotes_a_pesar.append({
                        **l,
                        "dias_sin_pesar": dias,
                        "razon": f"{dias} días desde última pesada",
                    })
            except (ValueError, TypeError):
                continue

    # Lotes cerca del objetivo de peso
    lotes_objetivo = []
    for l in lotes:
        if not l.get("objetivo_peso_kg") or not l.get("ultimo_peso_kg"):
            continue
        ratio = l["ultimo_peso_kg"] / l["objetivo_peso_kg"]
        if ratio >= 0.95:
            lotes_objetivo.append({
                **l,
                "ratio_objetivo": ratio,
                "dif_kg": l["objetivo_peso_kg"] - l["ultimo_peso_kg"],
            })

    return {
        "n_clientes": n_clientes,
        "n_lotes": n_lotes,
        "n_animales_total": n_animales_total,
        "n_pesadas_mes": len(pesadas_mes),
        "n_dietas_mes": len(dietas_mes),
        "ultima_pesada": ultima_pesada,
        "adg_promedio": adg_promedio,
        "lotes_a_pesar": lotes_a_pesar,
        "lotes_cerca_objetivo": lotes_objetivo,
        "clientes": clientes,
        "lotes": lotes,
    }


def obtener_alertas_clima_globales() -> Dict:
    """Consulta el clima de TODOS los clientes con localidad cargada.
    Devuelve info detallada de cada localidad: temperatura actual, alertas
    si las hay, y problemas (si no se pudo consultar).

    Estructura del resultado:
    {
      "consultadas": [
        {
          "cliente": "guinter",
          "localidad": "La Carlota",
          "estado": "ok" | "sin_geocodificar" | "sin_clima" | "error",
          "temp_c": 22.5,                   (solo si estado=ok)
          "humedad_pct": 65,
          "thi": 70,
          "thi_estado": "🟢 Sin estrés",
          "alertas_lotes": [                 (lista vacía si no hay alertas)
            {"lote": "x", "categoria": "y", "alertas": [...]},
          ],
          "n_alertas_criticas": 0,
          "n_alertas_warning": 0,
        },
        ...
      ],
      "sin_localidad": ["Cliente A", "Cliente B"],   (no tienen localidad cargada)
      "n_total_clientes": 5,
      "n_con_alertas": 1,
    }
    """
    from .clima import (
        geocodificar, obtener_clima, generar_alertas_predictivas,
        calcular_thi, clasificar_thi,
    )

    consultadas = []
    sin_localidad = []
    clientes = db.listar_clientes()
    return _procesar_clima_clientes(clientes)


def obtener_clima_para_cliente(cliente: dict) -> dict:
    """Versión single-cliente para usar en la ficha del lote.

    Devuelve el mismo `info` dict por cliente que arma
    `obtener_alertas_clima_globales`, pero solo para uno.
    Si no se puede obtener nada, devuelve un dict mínimo.
    """
    if not cliente:
        return {"estado": "sin_cliente"}
    res = _procesar_clima_clientes([cliente])
    consultadas = res.get("consultadas") or []
    if consultadas:
        return consultadas[0]
    return {
        "cliente": cliente.get("nombre", ""),
        "estado": "sin_localidad",
    }


def armar_contexto_clinico_lote(lote_id: int, db_mod) -> str:
    """Recopila todo el contexto histórico del lote para el LLM.

    Junta:
    - Fase del plan de adaptación (de las observaciones de la
      dieta vigente)
    - Movimientos recientes (muertes con causa, ventas)
    - Pesadas reales + ADG real vs objetivo
    - Historial de cargas del silo (sub-consumo / sobre-consumo)
    - Historial clínico (últimas evaluaciones, patrones,
      diagnósticos activos, mortandad por causa)

    Devuelve un string ya formateado para inyectar en el prompt.
    Si no hay datos, devuelve string vacío para que el bloque
    quede limpio.
    """
    from datetime import datetime as _dt, timedelta as _td
    lineas: List[str] = []
    hoy = _dt.now().date()

    try:
        lote = db_mod.obtener_lote(lote_id) or {}
    except Exception:
        return ""

    # ── 1. Fase del plan de adaptación ──
    try:
        dietas = db_mod.listar_dietas(lote_id) or []
        dieta_vig = None
        for d in dietas:
            try:
                f_d = _dt.strptime(
                    str(d.get("fecha", ""))[:10], "%Y-%m-%d"
                ).date()
                if f_d <= hoy:
                    dieta_vig = d
                    break  # ordenadas DESC
            except Exception:
                continue
        if dieta_vig:
            obs = (dieta_vig.get("observaciones") or "")
            if obs:
                # Detectar fase si está formato "FASE N: día X al Y"
                import re as _re
                m = _re.search(
                    r"FASE\s*(\d+).*?d[íi]a\s*(\d+)(?:\s*al\s*(\d+))?",
                    obs, _re.IGNORECASE,
                )
                if m:
                    fase_n = m.group(1)
                    dia_ini = int(m.group(2) or 0)
                    dia_fin = (
                        int(m.group(3)) if m.group(3) else None
                    )
                    # Calcular en qué día del plan está el lote HOY
                    try:
                        f_ing = _dt.strptime(
                            str(lote.get("fecha_ingreso",""))[:10],
                            "%Y-%m-%d",
                        ).date()
                        dia_actual = (hoy - f_ing).days + 1
                    except Exception:
                        dia_actual = None
                    lineas.append(
                        f"📋 FASE DEL PLAN DE ADAPTACIÓN: "
                        f"FASE {fase_n} ("
                        f"día {dia_ini}"
                        + (f" al {dia_fin}" if dia_fin else "")
                        + ")"
                        + (
                            f" — el lote HOY está en día "
                            f"{dia_actual} del ciclo"
                            if dia_actual is not None else ""
                        )
                    )
                else:
                    lineas.append(
                        f"📋 OBSERVACIONES DE LA DIETA: {obs[:200]}"
                    )
    except Exception:
        pass

    # ── 2. Movimientos recientes (últimos 30 días) ──
    try:
        movs = db_mod.listar_movimientos_lote(lote_id) or []
        movs_recientes = []
        for m in movs:
            try:
                f_m = _dt.strptime(
                    str(m.get("fecha", ""))[:10], "%Y-%m-%d"
                ).date()
                if (hoy - f_m).days <= 30:
                    movs_recientes.append(m)
            except Exception:
                continue
        if movs_recientes:
            lineas.append("\n💀🐄 MOVIMIENTOS DE LOS ÚLTIMOS 30 DÍAS:")
            for m in movs_recientes[:8]:
                tipo = m.get("tipo", "—")
                cant = m.get("cantidad", 0)
                det = (m.get("detalles", "") or "")[:120]
                lineas.append(
                    f"  - {m.get('fecha','—')[:10]}: "
                    f"{tipo} x {cant}"
                    + (f" — {det}" if det else "")
                )
    except Exception:
        pass

    # ── 3. ADG real vs objetivo (de pesadas) ──
    try:
        pesadas = db_mod.listar_pesadas(lote_id) or []
        if len(pesadas) >= 2:
            # Ordenar por fecha ascendente
            pes_ord = sorted(
                pesadas,
                key=lambda p: str(p.get("fecha", "")),
            )
            p_ini = pes_ord[0]
            p_fin = pes_ord[-1]
            try:
                f_p_ini = _dt.strptime(
                    str(p_ini.get("fecha", ""))[:10],
                    "%Y-%m-%d",
                ).date()
                f_p_fin = _dt.strptime(
                    str(p_fin.get("fecha", ""))[:10],
                    "%Y-%m-%d",
                ).date()
                dias_p = (f_p_fin - f_p_ini).days
                if dias_p > 0:
                    pv_ini = float(
                        p_ini.get("peso_promedio_kg") or 0
                    )
                    pv_fin = float(
                        p_fin.get("peso_promedio_kg") or 0
                    )
                    if pv_ini > 0 and pv_fin > 0:
                        adg_real = (pv_fin - pv_ini) / dias_p
                        adg_obj = float(
                            lote.get("adpv_objetivo_kg") or 0
                        )
                        cumple = (
                            f" vs OBJETIVO {adg_obj:.2f} "
                            f"({(adg_real/adg_obj*100):.0f}% del "
                            "objetivo)"
                            if adg_obj > 0 else ""
                        )
                        lineas.append(
                            f"\n📊 ADG REAL (medido entre pesadas): "
                            f"{adg_real:.3f} kg/día{cumple}"
                        )
            except Exception:
                pass
    except Exception:
        pass

    # ── 4. Historial de cargas del silo (sub/sobre consumo) ──
    try:
        from .stock_producto import comparar_carga_vs_dieta
        # Solo si hay silocomedero
        if (lote.get("tipo_comedero_concentrado") or "").lower() \
                == "silocomedero":
            cargas = db_mod.listar_cargas_silocomedero(lote_id) or []
            # Última carga COMPLETA (que tenga una siguiente
            # para medir el real)
            if len(cargas) >= 2:
                # Comparar último período cerrado
                try:
                    comp = comparar_carga_vs_dieta(
                        lote_id, cargas[1]["id"],
                    )
                    if comp:
                        real = comp.get("real_kg_ms_an_dia")
                        proy = comp.get(
                            "proyectado_kg_ms_an_dia",
                        )
                        if real and proy and proy > 0:
                            desvio_pct = (
                                (real - proy) / proy * 100
                            )
                            etiq = (
                                "SUB-CONSUMO" if desvio_pct < -5
                                else "SOBRE-CONSUMO"
                                if desvio_pct > 5
                                else "alineado"
                            )
                            lineas.append(
                                f"\n🛢️ ÚLTIMA CARGA CERRADA "
                                f"(consumo medido): "
                                f"real {real:.2f} vs proyectado "
                                f"{proy:.2f} kg MS/an/día "
                                f"→ {etiq} ({desvio_pct:+.1f}%)"
                            )
                except Exception:
                    pass
    except Exception:
        pass

    # ── 5. Historial clínico (ficha_clinica) ──
    try:
        from . import ficha_clinica as _fc
        ficha = _fc.armar_ficha_clinica_lote(lote_id, db_mod)
        # Mortandad acumulada
        tally = ficha.get("tally_mortandad") or {}
        if tally:
            partes = ", ".join(
                f"{k}: {v}" for k, v in tally.items()
            )
            lineas.append(
                f"\n💀 MORTANDAD ACUMULADA del lote (por causa): "
                f"{partes}"
            )
        # Patrones recurrentes
        patrones = ficha.get("patrones") or []
        if patrones:
            lineas.append("\n🔁 PATRONES RECURRENTES DETECTADOS:")
            for p in patrones:
                lineas.append(
                    f"  - {p['label']} "
                    f"({p['frecuencia']})"
                )
        # Diagnósticos activos
        dxs = ficha.get("diagnosticos_activos") or []
        if dxs:
            lineas.append("\n📌 DIAGNÓSTICOS ABIERTOS:")
            for d in dxs:
                lineas.append(
                    f"  - {d['label']} ({d['estado']}, "
                    f"detectado {d.get('fecha_deteccion','—')})"
                )
        # Últimas 2 consultas (cuestionario)
        evals = ficha.get("evaluaciones") or []
        if evals:
            lineas.append(
                f"\n📅 ÚLTIMAS CONSULTAS REGISTRADAS "
                f"({len(evals)} total, mostrando 2):"
            )
            for e in evals[:2]:
                lineas.append(
                    f"  - {e.fecha[:10]} "
                    f"{e.resumen_semaforo}: "
                    + (f"comedero {e.comedero}, " if e.comedero
                        else "")
                    + (f"heces {e.heces}, " if e.heces else "")
                    + (
                        f"{e.bajas} muerte(s)" if e.bajas
                        else "sin muertes"
                    )
                )
    except Exception:
        pass

    if lineas:
        return (
            "\n=== HISTORIAL DEL LOTE — contexto del paciente ===\n"
            + "\n".join(lineas)
        )
    return ""


def analizar_clima_lote_llm(
    lote: dict,
    info_clima: dict,
    dieta_vigente: Optional[dict] = None,
    historial_clinico: str = "",
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    """Pide a Claude un análisis climático narrativo del lote
    que cubra pasado / presente / futuro + medidas concretas.

    Returns:
        dict con:
        - exito: bool
        - analisis_md: str (markdown estructurado en 4 secciones)
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

    # Datos del lote
    cat = lote.get("categoria", "") or "—"
    raza = lote.get("raza", "") or "—"
    cant = lote.get("cantidad_inicial", 0)
    pv_ing = lote.get("peso_ingreso_kg", 0)
    adg = lote.get("adpv_objetivo_kg", 0)
    f_ing = lote.get("fecha_ingreso", "—")
    f_obj = lote.get("objetivo_fecha", "—")
    # Tipo de comedero — CRÍTICO para no recomendar cambios
    # de dosis diaria a un cliente que tiene silocomedero
    # (la mezcla está cargada y se mantiene 20-30 días sin tocar).
    tipo_com = (
        lote.get("tipo_comedero_concentrado") or ""
    ).lower()
    es_silo = tipo_com == "silocomedero"
    contexto_comedero = (
        "SILOCOMEDERO de autoconsumo (cargado cada 20-30 días "
        "con mezcla fija, NO se puede cambiar la dosis diaria ni "
        "modificar la fórmula al día siguiente — la mezcla actual "
        "se mantiene hasta la próxima carga)"
        if es_silo else
        f"Comedero {tipo_com or 'no especificado'} (reparto "
        "diario controlado, sí se puede ajustar dosis o frecuencia)"
    )

    # Pronóstico — separar por tramo
    pron = info_clima.get("pronostico_7d") or []
    hist_lines = []
    hoy_line = ""
    futuro_lines = []
    for d in pron:
        _tramo = d.get("tramo", "")
        _tmax = d.get("t_max")
        _tmin = d.get("t_min")
        _hr = d.get("hr_media")
        _prec = d.get("precipitacion_mm")
        _viento = d.get("viento_max_kmh")
        _nubes = d.get("nubes_pct")
        _sev_real = d.get("severidad_real", "—")
        linea = (
            f"  {d.get('fecha','—')}: "
            f"T° {_tmin:.0f}-{_tmax:.0f}°C, " if _tmin is not None
            and _tmax is not None else
            f"  {d.get('fecha','—')}: "
        )
        if _hr is not None:
            linea += f"HR {_hr:.0f}%, "
        if _prec is not None:
            linea += f"lluvia {_prec:.1f}mm, "
        if _viento is not None:
            linea += f"viento {_viento:.0f}km/h"
        if _nubes is not None:
            linea += f", nubes {_nubes:.0f}%"
        linea += f" → {_sev_real}"

        if "Histórico" in _tramo:
            hist_lines.append(linea)
        elif "HOY" in _tramo:
            hoy_line = linea
        else:
            futuro_lines.append(linea)

    hist_block = (
        "\n".join(hist_lines) if hist_lines
        else "(sin datos históricos)"
    )
    futuro_block = (
        "\n".join(futuro_lines) if futuro_lines
        else "(sin pronóstico)"
    )

    # Dieta vigente resumida
    dieta_str = ""
    if dieta_vigente:
        dieta_str = (
            f"\n=== DIETA VIGENTE ===\n"
            f"DMI {dieta_vigente.get('consumo_ms_kg',0):.1f} kg "
            f"MS/día · PB {dieta_vigente.get('pb_pct',0):.1f}% · "
            f"EM {dieta_vigente.get('em_mcal_dia',0):.1f} Mcal/día"
        )
        obs = (dieta_vigente.get("observaciones") or "")[:200]
        if obs:
            dieta_str += f"\nObservaciones: {obs}"

    # ─── Bloque IMPACTO PRODUCTIVO CALCULADO ───
    # Mismo cálculo que usa el email diario (src/impacto_productivo.py).
    # Le pasamos al LLM los rangos pre-calculados con etiquetas claras
    # (por animal/día, por lote/día, total del evento) para que cite
    # esos números exactos en lugar de recalcular por su cuenta. Esto
    # garantiza alineación entre el análisis del lote y el email.
    impacto_str = ""
    try:
        from .impacto_productivo import estimar_impacto_frio
        # Tomar los datos del peor día de frío proyectado
        peor_tmin = None
        peor_viento = None
        peor_hr = None
        peor_prec = None
        n_dias_frio = 0
        for d in pron:
            _tramo = d.get("tramo", "")
            if "Histórico" in _tramo:
                continue
            _tmin_d = d.get("t_min")
            if _tmin_d is None or _tmin_d > 10:
                continue
            n_dias_frio += 1
            if peor_tmin is None or _tmin_d < peor_tmin:
                peor_tmin = _tmin_d
                peor_viento = d.get("viento_max_kmh")
                peor_hr = d.get("hr_media")
                peor_prec = d.get("precipitacion_mm")

        if (peor_tmin is not None and n_dias_frio > 0
                and pv_ing and cat):
            # PV proyectado HOY (estimación lineal con ADG)
            try:
                from datetime import datetime as _dt
                f_ing_d = _dt.strptime(
                    str(f_ing)[:10], "%Y-%m-%d"
                ).date()
                dias_enc = (_dt.now().date() - f_ing_d).days
                pv_hoy = (
                    float(pv_ing) + float(adg or 0) * max(0, dias_enc)
                )
            except Exception:
                pv_hoy = float(pv_ing)

            impacto = estimar_impacto_frio(
                peso_kg=pv_hoy,
                categoria=cat,
                raza=raza,
                t_min_c=peor_tmin,
                viento_kmh=peor_viento,
                humedad_pct=peor_hr,
                barro=(peor_prec or 0) > 10,
                pelaje_mojado=(peor_hr or 0) >= 85,
                dias_evento=n_dias_frio,
                cantidad=cant,
                adpv_objetivo_kg=adg if adg else None,
                energia_dieta_mcal_em_kg_ms=(
                    (
                        dieta_vigente.get("em_mcal_dia", 0)
                        / dieta_vigente.get("consumo_ms_kg", 1)
                    )
                    if (dieta_vigente
                        and dieta_vigente.get("consumo_ms_kg"))
                    else None
                ),
            )
            if impacto:
                adpv_perdida = impacto.get(
                    "adpv_perdida_kg_rango",
                )
                kg_lote_total = impacto.get(
                    "kg_perdidos_lote_periodo",
                )
                pct_adpv = impacto.get("pct_adpv_perdida")
                impacto_str = (
                    "\n=== IMPACTO PRODUCTIVO CALCULADO "
                    "(NRC 2016 + Pampa Húmeda) ===\n"
                    f"Evento de frío proyectado: {n_dias_frio} día(s) "
                    f"con T° mín ≤ 10°C (peor día "
                    f"{peor_tmin:.0f}°C, viento "
                    f"{peor_viento or 0:.0f} km/h, "
                    f"HR {peor_hr or 0:.0f}%).\n"
                    f"PV proyectado HOY: {pv_hoy:.0f} kg/animal\n"
                )
                if adpv_perdida:
                    impacto_str += (
                        f"• Pérdida ADG: "
                        f"{adpv_perdida[0]:.2f}-{adpv_perdida[1]:.2f} "
                        f"kg POR ANIMAL POR DÍA "
                    )
                    if pct_adpv:
                        impacto_str += (
                            f"({pct_adpv[0]:.0f}-{pct_adpv[1]:.0f}% "
                            f"del ADG objetivo)"
                        )
                    impacto_str += "\n"
                if kg_lote_total:
                    impacto_str += (
                        f"• Pérdida acumulada para el LOTE en los "
                        f"{n_dias_frio} día(s) del evento: "
                        f"{kg_lote_total[0]:.0f}-"
                        f"{kg_lote_total[1]:.0f} kg DE PESO VIVO "
                        f"TOTAL (NO multiplicar ni convertir).\n"
                    )
                if impacto.get("supuestos"):
                    impacto_str += (
                        f"Supuestos: {impacto['supuestos']}\n"
                    )
                impacto_str += (
                    "\n⚠️ INSTRUCCIÓN: usá ESTOS NÚMEROS EXACTOS "
                    "en la sección '🎯 Medidas a tomar' si querés "
                    "cuantificar el impacto. NO los recalcules. "
                    "NO los multipliques por rendimiento de carcasa. "
                    "Etiquetá siempre 'por animal/día' o 'total del "
                    "evento' según corresponda."
                )
    except Exception:
        impacto_str = ""

    # ─── Composición del system prompt ───
    # Filosofía única HMS + perfil específico de este modo
    # (ver src/filosofia_hms.py y src/perfiles_llm.py). El prompt
    # largo que vivía inline acá se movió a esos módulos para que
    # los 4 agentes (chat, análisis climático, cuestionario,
    # resumen clínico) compartan UNA sola fuente de verdad.
    from . import perfiles_llm as _perfiles_llm
    system_prompt = _perfiles_llm.armar_system_prompt(
        "analisis_clima_lote",
    )

    # Bloque viejo deshabilitado (queda como referencia histórica
    # por si hay que comparar — se puede borrar en un próximo PR).
    _system_prompt_viejo = (
        "Sos Mauricio Suárez de HMS Nutrición Animal, asesor "
        "nutricional con 20 años de campo en feedlot en La Pampa "
        "y Pampa Húmeda.\n\n"
        "Tu tarea: análisis climático CONTEXTUAL que conecte el "
        "clima con ESE LOTE específico (categoría, peso, dieta, "
        "comedero) y proponga acciones que SE HAGAN en feedlot "
        "argentino.\n\n"
        "🎬 ESTILO: contá lo que pasa con el animal como si "
        "estuvieras narrando un documental de naturaleza — "
        "voz educativa, descriptiva, con cadenas causales "
        "completas. NO uses bullets en las secciones de análisis "
        "(solo en 'Medidas a tomar'). Escribí en PROSA, conectando "
        "una idea con la siguiente. Que el lector vea el cuadro "
        "biológico en su cabeza.\n\n"
        "FORMATO OBLIGATORIO (markdown, 4 secciones):\n\n"
        "**🕒 Lo que pasó (últimos 7 días):** PÁRRAFO de 4-6 "
        "oraciones que cuente la historia biológica. Explicá la "
        "CADENA COMPLETA de mecanismos QUE ESTÁN EN LA "
        "LITERATURA — sin inventar cifras específicas:\n"
        "  1. Qué condición climática enfrentó el animal "
        "(frío, viento, pelaje húmedo, cielo cubierto sostenido).\n"
        "  2. Cómo se DEFENDIÓ fisiológicamente — el bovino "
        "activa termogénesis para mantener T° corporal cuando "
        "la T° del ambiente cae por debajo de su LCT (NRC 2016). "
        "Aumenta tasa metabólica basal y fermentación ruminal. "
        "Si querés un número, usá los del bloque IMPACTO "
        "PRODUCTIVO CALCULADO que te paso — esos están "
        "calculados con NRC. NO inventes porcentajes de "
        "aumento metabólico sin respaldo.\n"
        "  3. Qué hizo CON SU COMPORTAMIENTO — bibliografía "
        "(NRC 2016; Mader & Davis 2004) describe agrupamiento, "
        "búsqueda de reparo, reducción de actividad. Si "
        "mencionás cambios en frecuencia de visitas al comedero, "
        "describilo cualitativamente ('reduce visitas y "
        "concentra el consumo en franjas cálidas'), NO inventes "
        "el número exacto de visitas/día.\n"
        "  4. La PARADOJA del frío: para sostener calor el "
        "animal necesita rumiar (la fermentación genera calor "
        "metabólico — bien documentado), pero exponerse al "
        "viento en el comedero le hace perder calor por "
        "convección. Esto se traduce en compromiso del consumo "
        "voluntario, sobre todo concentrado (digestión más "
        "rápida → ventana de generación de calor más corta "
        "que con fibra).\n"
        "  5. Resultado cuantificado: usá las cifras del "
        "bloque IMPACTO PRODUCTIVO CALCULADO (kg/día perdidos, "
        "% ADG, kg total del evento). Esas son las únicas "
        "cifras 'seguras' — vienen de NRC + ajustes Pampa "
        "Húmeda. Si NO te pasé ese bloque, NO cuantifiques "
        "la pérdida — decí 'caída esperable de ADG, magnitud "
        "depende de cuántos días dure el evento'.\n"
        "  Si no hubo nada destacable, decilo en una línea.\n\n"
        "**⭐ Lo que pasa HOY:** 2-3 oraciones. Estado actual y "
        "qué se ve EN EL CORRAL hoy mismo (animales agrupados? "
        "consumo restablecido? heces más firmes que ayer?).\n\n"
        "**🔮 Lo que viene (próximos 7 días):** 3-4 oraciones que "
        "narren el evento próximo. No solo decir 'hay frío el "
        "X' — explicar QUÉ va a sentir el animal, QUÉ va a "
        "cambiar en su comportamiento, QUÉ tenés que ver en el "
        "corral cuando el evento llegue. Si hay un día crítico, "
        "mencionalo con fecha + por qué es el peor.\n\n"
        "**🎯 Medidas a tomar:** lista bullet 4-6 acciones "
        "CONCRETAS Y CUANTIFICADAS para ESTE lote. Cada bullet "
        "puede tener una sub-explicación de POR QUÉ esa medida "
        "ayuda (1 línea extra), pero la acción principal "
        "tiene que ser CLARA Y EJECUTABLE.\n\n"
        "Si HR ≥ 90% sostenida y/o lluvia, INCLUÍ "
        "OBLIGATORIAMENTE una medida que aborde el deterioro "
        "del alimento en el comedero:\n"
        "   - Revisar mezcla del silocomedero: ¿está "
        "apelmazada, hinchada, con olor agrio, fermentada en "
        "la superficie? Si sí: retirar lo de arriba, mover/"
        "remover para airear, evitar dejar mezcla expuesta "
        "muchos días.\n"
        "   - En silos con ventana fija, revisar que el "
        "animal pueda llegar a mezcla seca (la de abajo) — "
        "ajustar la ventana / apertura si está limitada.\n"
        "   - Si el cliente carga rollo aparte, asegurar que "
        "el rollo esté protegido de la lluvia.\n\n"
        "REGLAS DURAS — leelas antes de escribir las acciones:\n\n"
        "1) **RESPETÁ EL TIPO DE COMEDERO**:\n"
        "   - Si es **SILOCOMEDERO de autoconsumo**: la mezcla "
        "actual está cargada y se mantiene hasta la próxima "
        "carga (20-30 días). NO sugieras 'cambiar la dosis "
        "diaria', 'aumentar maíz X kg/día', 'modificar la "
        "fórmula', 'subir Fibrogreen 2 puntos'. NADA de eso "
        "se puede hacer hasta la próxima carga del silo.\n"
        "   - Si querés tocar la dieta en silocomedero, lo "
        "que SÍ podés sugerir: 'en la PRÓXIMA carga del silo "
        "ajustar X', revisar las ventanas / apertura del silo "
        "para ajustar oferta efectiva, sumar rollo aparte (libre "
        "disposición) como complemento energético/fibroso.\n"
        "   - Si es comedero lineal/diario: SÍ podés ajustar "
        "dosis o frecuencia día a día.\n\n"
        "2) **NO INVENTES PRÁCTICAS NO COMUNES EN PAMPA HÚMEDA**. "
        "PROHIBIDAS en este contexto:\n"
        "   - Entibiar / calentar / regar / climatizar el agua "
        "de los bebederos (NO se hace en feedlot argentino)\n"
        "   - Riego nocturno de bebederos\n"
        "   - Cobertizos, galpones cerrados, calentadores\n"
        "   - Suplementación intravenosa, electrolitos en agua "
        "(salvo veterinario)\n"
        "   - Cualquier infraestructura que no esté ya en el "
        "campo argentino estándar\n"
        "   Lo que SÍ es común: reparo de fardos/rollos en L, "
        "cama de paja seca, romper hielo del bebedero a mano "
        "a la mañana, mover el bebedero si hay barro, cortinas "
        "vegetales / forestales, monitorear consumo del "
        "comedero, ajustar horarios de pasaje del lote.\n\n"
        "3) **EXPLICÁ EL MECANISMO de la baja de consumo**, no "
        "solo digas que baja. Por ej:\n"
        "   - 'el ternero deriva 15-20% de su energía a "
        "termorregulación → ese costo sale del fondo para "
        "ganancia, ADG cae'\n"
        "   - 'con pelaje húmedo + viento, la pérdida de calor "
        "se duplica y el animal corta consumo voluntariamente "
        "para reducir digestión (proceso que genera calor pero "
        "lo expone) — patrón de selección'\n"
        "   - 'el barro en el comedero hace que evite acercarse → "
        "consume en picos cuando finalmente come = riesgo acidosis'\n\n"
        "🧬 USÁ EL HISTORIAL DEL LOTE (contexto del paciente):\n\n"
        "Si en el contexto te paso un bloque 'HISTORIAL DEL "
        "LOTE — contexto del paciente', es información CRÍTICA "
        "que tenés que integrar al análisis. NO la copies tal "
        "cual — interpretala como un asesor que conoce al "
        "lote:\n\n"
        "- **Fase del plan de adaptación**: la vulnerabilidad "
        "al clima cambia según la fase. En adaptación temprana "
        "(día 1-15) el animal está estresado por cambio "
        "alimenticio + categoría liviana = más vulnerable a "
        "frío. En terminación (último tercio) ya está con "
        "reservas de grasa = más tolerante. Mencionalo si "
        "corresponde.\n\n"
        "- **Movimientos recientes (muertes con causa)**: si "
        "hubo muertes recientes y la causa fue acidosis, "
        "neumonía, timpanismo, etc., HOY no sos neutral — sos "
        "más conservador. Mencioná la continuidad: 'con el "
        "antecedente de X bajas por acidosis hace Y días, "
        "tenemos que ser más prudentes con ...'\n\n"
        "- **ADG REAL vs OBJETIVO**: si el lote viene por "
        "debajo del objetivo, este evento de frío lo va a "
        "agravar. Si viene en línea u arriba, hay margen. "
        "Citá el dato real (no el objetivo) cuando hables "
        "del impacto.\n\n"
        "- **Última carga del silo (sub/sobre consumo "
        "medido)**: si la última carga mostró SUB-consumo, "
        "el lote YA viene comiendo menos de lo proyectado. "
        "Este evento climático no es 'el inicio' del problema, "
        "es 'la continuación'. Citá ese dato.\n\n"
        "- **Patrones recurrentes detectados**: si la ficha "
        "clínica marca 'comedero vacío sostenido' o 'heces "
        "alteradas sostenidas', son señales DE BASE — no "
        "podés ignorarlas en el análisis climático.\n\n"
        "- **Diagnósticos abiertos**: si hay diagnósticos "
        "activos de consultas anteriores, retomalos. El "
        "asesor no empieza de cero cada vez.\n\n"
        "- **Últimas consultas**: si las últimas evaluaciones "
        "mostraron un problema, hoy esa info es contexto. Si "
        "antes había heces pastosas y hoy viene HR alta, eso "
        "AGRAVA el cuadro.\n\n"
        "🚫 LÓGICAS PROHIBIDAS — son alucinaciones comunes:\n\n"
        "1. **NO digas** que en frío el animal 'reduce consumo "
        "para evitar sobrecalentamiento'. Eso es contradictorio "
        "y FALSO: en frío el animal busca CONSERVAR calor, no "
        "perderlo. La caída de consumo en frío + HR alta NO es "
        "por sobrecalentamiento — es por (a) comportamiento de "
        "reparo, (b) deterioro físico del alimento, (c) menor "
        "apetencia general.\n\n"
        "2. **NO uses términos inventados** tipo 'franja de "
        "calentamiento matinal' o cosas raras. Usá lenguaje "
        "que un encargado de feedlot entienda: 'mañana "
        "temprano', 'mediodía', 'tarde', 'noche'.\n\n"
        "3. **NO escribas cadenas causales contradictorias**. "
        "Si decís 'se agrupan en X horario', el animal está "
        "echado/parado en grupo, NO está comiendo al mismo "
        "tiempo. Si decís 'concentran visitas al comedero en "
        "Y horario', es OTRO momento o son OTROS animales. "
        "Cada acción biológica tiene que ser COHERENTE con la "
        "anterior — releé tu propio párrafo antes de escribir "
        "el siguiente.\n\n"
        "4. **CAUSAS REALES** de caída de consumo bajo frío "
        "sostenido con HR alta y/o lluvia — usá ESTAS (no "
        "inventes otras):\n"
        "   (a) **Comportamiento de reparo**: el animal "
        "prefiere quedarse echado en grupo (conserva calor "
        "corporal) y reduce frecuencia y duración de visitas "
        "al comedero porque cada acercamiento expone al viento "
        "y le saca calor. Resultado: come en menos visitas pero "
        "más concentradas (picos) → riesgo de fluctuación de "
        "pH ruminal.\n"
        "   (b) **Deterioro físico de la mezcla por humedad** "
        "(¡importante en silocomedero!): con HR 90-100% "
        "sostenida la mezcla absorbe humedad ambiente. El "
        "concentrado (maíz partido, núcleo) se HINCHA, se "
        "APELMAZA en el comedero, los granos finos se "
        "compactan, los aceites del grano se oxidan, hay "
        "fermentación secundaria que cambia aroma y "
        "palatabilidad. El animal RECHAZA o SELECCIONA "
        "(saca lo de afuera, deja lo apelmazado abajo). En "
        "silocomedero con ventana de oferta limitada, esto "
        "es CRÍTICO porque queda mezcla vieja arriba. SIEMPRE "
        "mencionalo si hay HR ≥ 90% sostenida en silocomedero.\n"
        "   (c) **Barrera física del barro alrededor del "
        "comedero y bebedero** (¡muy importante en Pampa "
        "Húmeda!): con lluvia acumulada + pisoteo en zona de "
        "acceso al comedero y bebedero, se forma un cinturón "
        "de barro profundo que el animal EVITA. Caminar en "
        "barro consume más energía, ensucia las patas, "
        "compromete el aplomo. El animal espera, va menos "
        "veces, o directamente se restringe a los bebederos/"
        "comederos accesibles. Cuando el barro rodea todo el "
        "comedero, baja DRÁSTICAMENTE el consumo y el "
        "consumo de agua. SIEMPRE mencionalo si hay lluvia "
        "acumulada >10 mm o si hubo barro previo no drenado.\n"
        "   (d) **Pelaje húmedo permanente**: cuando se acerca "
        "al comedero, el viento + humedad le sacan calor por "
        "evaporación de la humedad superficial — costo "
        "energético adicional que el animal 'percibe' y "
        "compensa reduciendo exposiciones.\n"
        "   (e) **Apetencia general reducida** por estrés "
        "crónico — bien documentado, no requiere fisiología "
        "compleja para explicarlo.\n\n"
        "🔬 REGLAS DE EVIDENCIA CIENTÍFICA — prioritarias:\n\n"
        "1. **NO INVENTES CIFRAS sin respaldo**. Los únicos "
        "números 'seguros' para citar son:\n"
        "   - Los que vienen en el contexto que te paso "
        "(datos climáticos, datos del lote, dieta vigente)\n"
        "   - Los del bloque IMPACTO PRODUCTIVO CALCULADO "
        "(si te lo paso) — vienen de NRC 2016 + ajustes "
        "Pampa Húmeda con cálculos honestos en rangos\n"
        "   - Umbrales bien establecidos: LCT bovino "
        "(zona termoneutral), THI clásico (NRC 2016), "
        "wind chill bovino (NRC), temperatura corporal "
        "normal (38-39°C). Estos los podés citar.\n"
        "2. **NO INVENTES NÚMEROS sobre comportamiento o "
        "fisiología** (cantidad de visitas al comedero, "
        "% de aumento metabólico, % de caída de consumo, "
        "horas de rumia). Si no podés citar de una fuente "
        "(NRC 2016, NASEM, INTA Anguil/Manfredi, Mader, "
        "Pordomingo, etc.), describilo CUALITATIVAMENTE.\n"
        "3. **Jerarquía de fuentes** (de mayor a menor "
        "preferencia):\n"
        "   - INTA Anguil/Manfredi (Pampa Húmeda)\n"
        "   - Pordomingo, Bavera, Latimori (autores "
        "argentinos)\n"
        "   - NRC 2016 / NASEM 2016 (estándar internacional)\n"
        "   - Mader & Davis (frío en bovinos)\n"
        "   - INRA, CSIRO (último recurso)\n"
        "4. **Si vas a citar un autor**, hacelo de manera "
        "natural ('según NRC 2016', 'siguiendo a Mader'). "
        "NO inventes citas que no conocés.\n"
        "5. **Si dudás entre cuantificar e inventar, NO "
        "cuantifiques**. Mejor 'caída esperable de ADG' "
        "que '15% de caída' inventado.\n"
        "6. **Mecanismos fisiológicos cualitativos** que "
        "SÍ podés usar libremente (sin inventar números):\n"
        "   - Termogénesis obligatoria bajo LCT\n"
        "   - Movilización de reservas grasas en frío "
        "sostenido\n"
        "   - Fermentación ruminal como fuente de calor "
        "metabólico\n"
        "   - Pérdida de calor por convección (viento), "
        "conducción (cama mojada), evaporación (pelaje "
        "húmedo)\n"
        "   - Patrón de consumo: agrupamiento de visitas en "
        "franjas cálidas\n"
        "   - Acidosis por consumo desparejo (picos en "
        "lugar de distribuido)\n\n"
        "OTRAS REGLAS:\n"
        "- No inventes datos no provistos (no asumas que hay "
        "reparos, sombra, instalaciones específicas)\n"
        "- Hablá criollo técnico-narrativo, sin frases "
        "marketineras pero CON vuelo descriptivo (Nat Geo, "
        "no folleto)\n"
        "- Si hay un evento serio próximo, decilo CLARO y en "
        "qué día\n"
        "- Si todo está normal y no hay nada que hacer, decilo: "
        "'Sin eventos relevantes; mantener manejo de rutina.'\n"
        "- LONGITUD: el análisis tiene que ser SUSTANCIAL, "
        "tiene que enseñar al lector qué le pasa al animal. "
        "Apuntá a 400-500 palabras totales. Si es menos, "
        "estás dejando contexto biológico afuera. PERO no "
        "rellenes con cifras inventadas para llegar al "
        "largo — mejor menos palabras y todas verdaderas.\n"
        "- Tono: mostrá conocimiento, no resúmas. El asesor "
        "vino a entender, no a leer un titular.\n\n"
        "REGLA CRÍTICA DE ALINEACIÓN CON EMAIL:\n"
        "- Si en el contexto te paso un bloque 'IMPACTO PRODUCTIVO "
        "CALCULADO', USÁ ESOS NÚMEROS EXACTOS. NO los recalcules. "
        "NO los multipliques por rendimiento de carcasa. NO los "
        "conviertas a 'carne'. Son kg de PESO VIVO directos.\n"
        "- Etiquetá SIEMPRE inequívocamente: 'por animal/día' "
        "vs 'total del evento'. El asesor tiene que leer la "
        "cifra y saber sin duda qué representa.\n"
        "- Si te paso ese bloque, los mismos números van a "
        "aparecer en el email diario del cliente — usá los "
        "mismos rangos para que el cliente y el asesor vean "
        "la misma cuantificación."
    )

    user_msg = (
        f"=== LOTE ===\n"
        f"Identificador: {lote.get('identificador','—')}\n"
        f"Categoría: {cat} · Raza: {raza} · {cant} animales\n"
        f"PV ingreso: {pv_ing} kg · ADG objetivo: {adg} kg/día\n"
        f"Ingresó: {f_ing} · Salida proyectada: {f_obj}\n"
        f"🛢️ Sistema de alimentación: {contexto_comedero}\n"
        f"{dieta_str}\n\n"
        f"=== CLIMA RECIENTE Y PROYECTADO ===\n"
        f"📍 {info_clima.get('nombre_geocode', 'desconocido')}\n"
        f"Estado HOY: "
        f"{info_clima.get('temp_c','?')}°C · "
        f"HR {info_clima.get('humedad_pct','?')}% · "
        f"THI {info_clima.get('thi','?')} "
        f"({info_clima.get('thi_estado','—')})\n\n"
        f"--- Últimos 7 días (datos reales) ---\n{hist_block}\n\n"
        f"--- HOY ---\n{hoy_line or '(sin dato)'}\n\n"
        f"--- Próximos 7 días (pronóstico) ---\n{futuro_block}\n\n"
        f"Severidad real máxima próximos 7 días: "
        f"{info_clima.get('severidad_real_max','—')} "
        f"(fecha: "
        f"{info_clima.get('severidad_real_max_fecha','—')})"
        f"{impacto_str}"
        f"{historial_clinico}"
    )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        partes = []
        for block in resp.content:
            if hasattr(block, "text"):
                partes.append(block.text)
        out["analisis_md"] = "\n".join(partes).strip()
        out["exito"] = True
    except Exception as e:
        out["error"] = f"LLM falló: {e}"
    return out


def _procesar_clima_clientes(clientes: list) -> dict:
    """Procesa una lista de clientes y devuelve el clima de cada uno.

    Esta función contiene la lógica que originalmente estaba inline
    en obtener_alertas_clima_globales — la separamos para poder
    invocarla con una lista de 1 cliente desde la ficha del lote.
    """
    from .clima import (
        geocodificar, obtener_clima, generar_alertas_predictivas,
        calcular_thi, clasificar_thi,
    )
    consultadas = []
    sin_localidad = []

    for c in clientes:
        if not c.get("localidad"):
            sin_localidad.append(c["nombre"])
            continue

        info = {
            "cliente": c["nombre"],
            "localidad": c["localidad"],
            "estado": "ok",
            "temp_c": None,
            "humedad_pct": None,
            "thi": None,
            "thi_estado": None,
            "alertas_lotes": [],
            "n_alertas_criticas": 0,
            "n_alertas_warning": 0,
        }

        try:
            # Si el cliente tiene lat/lon manuales cargadas, usar esas
            if c.get("lat") is not None and c.get("lon") is not None:
                from .clima import geocodificar_manual
                geo = geocodificar_manual(
                    float(c["lat"]), float(c["lon"]), c["localidad"],
                )
            else:
                geo = geocodificar(c["localidad"])
            if not geo:
                info["estado"] = "sin_geocodificar"
                consultadas.append(info)
                continue

            info["lat"] = geo["lat"]
            info["lon"] = geo["lon"]
            info["nombre_geocode"] = f"{geo['nombre']}, {geo.get('admin1','')}"

            clima = obtener_clima(geo["lat"], geo["lon"])
            if not clima:
                info["estado"] = "sin_clima"
                consultadas.append(info)
                continue

            actual = clima.get("current", {}) or {}
            t = actual.get("temperature_2m")
            hr = actual.get("relative_humidity_2m")
            if t is not None and hr is not None:
                thi = calcular_thi(t, hr)
                info["temp_c"] = t
                info["humedad_pct"] = hr
                info["thi"] = thi
                info["thi_estado"] = clasificar_thi(thi)

            # Pronóstico de 7 días — el daily de Open-Meteo trae
            # 14 días (7 pasados + 7 futuros). Filtramos para que
            # SOLO mostremos desde HOY en adelante (lo que importa
            # operativamente).
            daily = clima.get("daily", {}) or {}
            fechas = daily.get("time") or []
            tmin_arr = daily.get("temperature_2m_min") or []
            tmax_arr = daily.get("temperature_2m_max") or []
            hr_arr = (
                daily.get("relative_humidity_2m_mean")
                or daily.get("relative_humidity_2m_max")
                or []
            )
            precip_arr = daily.get("precipitation_sum") or []
            viento_arr = (
                daily.get("wind_speed_10m_max")
                or daily.get("windspeed_10m_max")
                or []
            )
            nubes_arr = daily.get("cloud_cover_mean") or []
            rad_arr = daily.get("shortwave_radiation_sum") or []
            hoy_iso_d = datetime.now().strftime("%Y-%m-%d")
            pronostico_7d = []
            for i, f in enumerate(fechas):
                # Mostrar TODO el rango (7 pasados + 7 futuros = 14
                # días) para que se pueda analizar tendencia + ver
                # qué viene. La columna "Tramo" en la UI distingue
                # histórico vs hoy vs futuro.
                _tmax = (
                    tmax_arr[i] if i < len(tmax_arr) else None
                )
                _tmin = (
                    tmin_arr[i] if i < len(tmin_arr) else None
                )
                _hr = hr_arr[i] if i < len(hr_arr) else None
                _prec = (
                    precip_arr[i]
                    if i < len(precip_arr) else None
                )
                _viento = (
                    viento_arr[i] if i < len(viento_arr) else None
                )
                _nubes = (
                    nubes_arr[i] if i < len(nubes_arr) else None
                )
                _rad = (
                    rad_arr[i] if i < len(rad_arr) else None
                )
                if (_tmax is not None and _hr is not None):
                    _thi_d = calcular_thi(_tmax, _hr)
                    _thi_estado_d = clasificar_thi(_thi_d)
                else:
                    _thi_d = None
                    _thi_estado_d = "—"

                # ── Severidad REAL ajustada para el bovino ──
                # Considera THI + viento (Mader 2006) + frío con
                # viento (NRC wind chill bovino) + lluvia (corral
                # con barro) + frío nocturno.
                _sev = "🟢 Sin estrés"
                # CALOR: THI ajustado por viento
                if _thi_d is not None:
                    _viento_ms = (
                        (_viento or 0) / 3.6  # km/h → m/s
                    )
                    # Mader 2006: THI_adj ≈ THI − 0.5 × viento_ms
                    _thi_adj = _thi_d - 0.5 * _viento_ms
                    if _thi_adj >= 84:
                        _sev = "🔴 Crítico calor"
                    elif _thi_adj >= 79:
                        _sev = "🟠 Moderado calor"
                    elif _thi_adj >= 72:
                        _sev = "🟡 Atención calor"
                # FRÍO: T° min nocturna + viento + lluvia + HR alta.
                # Umbral más amplio (≤ 15°C) porque la sensación
                # del bovino con combinación adversa puede ser
                # mucho menor que la T° real del aire.
                if _tmin is not None and _tmin <= 15:
                    # Wind chill bovino: cada 10 km/h de viento
                    # ≈ 3°C menos sentidos (NRC, aproximado).
                    _tmin_sentida = (
                        _tmin - (((_viento or 0) / 10) * 3)
                    )
                    # ── Pelaje mojado / falta de secado ──
                    # Distintos mecanismos:
                    #   A) Lluvia real (>5 mm) → empapa al
                    #      animal → pierde calor de verdad.
                    #   B) Niebla densa persistente (HR ≥ 98%
                    #      + viento < 5 km/h) → no levanta.
                    #   C) Cielo cubierto + HR alta TODO el día
                    #      (nubosidad ≥ 75% + HR ≥ 90%) →
                    #      el rocío matinal NO evapora porque
                    #      no entra sol → pelaje queda húmedo
                    #      sostenido. Penalización menor que
                    #      la lluvia (no empapa, solo no seca).
                    _pelaje_mojado = (_prec or 0) > 5
                    _niebla_densa = (
                        (_hr or 0) >= 98 and (_viento or 0) < 5
                    )
                    _dia_sin_sol = (
                        (_nubes or 0) >= 75
                        and (_hr or 0) >= 90
                    )
                    if _pelaje_mojado or _niebla_densa:
                        # Mojado real: −4°C sentidos
                        _tmin_sentida -= 4
                    elif _dia_sin_sol:
                        # Pelaje no termina de secar: −2°C
                        _tmin_sentida -= 2
                    # Lluvia fuerte sostenida = barro mañana
                    _barro_proba = (_prec or 0) > 10

                    if _tmin_sentida <= -5:
                        _sev = "🔴 Crítico frío"
                    elif _tmin_sentida <= 0:
                        if "Sin estrés" in _sev:
                            _sev = "🟠 Moderado frío"
                    elif _tmin_sentida <= 5:
                        if "Sin estrés" in _sev:
                            _sev = "🟡 Atención frío"
                    # Caso especial: T° "normal" (>10°C) pero
                    # lluvia fuerte + HR alta + viento = ambiente
                    # adverso aunque la sensación no sea crítica.
                    if (
                        "Sin estrés" in _sev
                        and _barro_proba
                        and (_hr or 0) >= 85
                        and (_viento or 0) >= 15
                    ):
                        _sev = "🟡 Atención lluvia+viento"
                # BARRO acumulado: lluvia >15 mm + T° baja
                if (_prec or 0) > 15 and (_tmin or 99) < 12:
                    if "Sin estrés" in _sev:
                        _sev = "🟡 Atención barro+frío"

                # Tramo: pasado / hoy / futuro
                if f < hoy_iso_d:
                    _tramo = "🕒 Histórico"
                elif f == hoy_iso_d:
                    _tramo = "⭐ HOY"
                else:
                    _tramo = "🔮 Pronóstico"

                pronostico_7d.append({
                    "fecha": f,
                    "tramo": _tramo,
                    "t_min": _tmin,
                    "t_max": _tmax,
                    "hr_media": _hr,
                    "precipitacion_mm": _prec,
                    "viento_max_kmh": _viento,
                    "nubes_pct": _nubes,
                    "radiacion_mj": _rad,
                    "thi": _thi_d,
                    "thi_estado": _thi_estado_d,
                    "severidad_real": _sev,
                })
            info["pronostico_7d"] = pronostico_7d

            # ── Severidad REAL máxima (HOY + futuro) ──
            # El título del expander debe reflejar la peor severidad
            # esperada desde hoy hacia adelante, NO el THI clásico
            # (que no considera viento, lluvia, HR ni acumulación).
            # Ranking: Crítico > Moderado > Atención > Sin estrés.
            _sev_rank = {
                "🔴": 4,  # Crítico
                "🟠": 3,  # Moderado
                "🟡": 2,  # Atención
                "🟢": 1,  # Sin estrés
            }
            _max_sev = "🟢 Sin estrés"
            _max_sev_rank = 1
            _max_sev_fecha = None
            _max_sev_tramo = None
            _sev_hoy = "🟢 Sin estrés"
            for _d in pronostico_7d:
                # Solo considerar HOY + futuro (no histórico)
                if _d.get("tramo") == "🕒 Histórico":
                    continue
                _s = _d.get("severidad_real", "🟢 Sin estrés")
                _emoji = (_s[0] if _s else "🟢")
                _r = _sev_rank.get(_emoji, 1)
                if _d.get("tramo") == "⭐ HOY":
                    _sev_hoy = _s
                if _r > _max_sev_rank:
                    _max_sev_rank = _r
                    _max_sev = _s
                    _max_sev_fecha = _d.get("fecha")
                    _max_sev_tramo = _d.get("tramo")
            info["severidad_real_max"] = _max_sev
            info["severidad_real_max_rank"] = _max_sev_rank
            info["severidad_real_max_fecha"] = _max_sev_fecha
            info["severidad_real_max_tramo"] = _max_sev_tramo
            info["severidad_real_hoy"] = _sev_hoy

            # Para cada lote del cliente, calcular alertas
            lotes_cli = db.listar_lotes(cliente_id=c["id"], estado="activo")
            for l in lotes_cli:
                alertas = generar_alertas_predictivas(
                    clima, categoria=l.get("categoria", ""),
                )
                relevantes = [
                    a for a in alertas
                    if a.get("severidad") in ("warning", "critica")
                ]
                if relevantes:
                    info["alertas_lotes"].append({
                        "lote": l["identificador"],
                        "categoria": l.get("categoria", ""),
                        "alertas": relevantes,
                    })
                    info["n_alertas_criticas"] += sum(
                        1 for a in relevantes if a["severidad"] == "critica"
                    )
                    info["n_alertas_warning"] += sum(
                        1 for a in relevantes if a["severidad"] == "warning"
                    )
        except Exception as e:
            info["estado"] = "error"
            info["error"] = str(e)

        consultadas.append(info)

    n_con_alertas = sum(
        1 for i in consultadas
        if i["n_alertas_criticas"] + i["n_alertas_warning"] > 0
    )

    return {
        "consultadas": consultadas,
        "sin_localidad": sin_localidad,
        "n_total_clientes": len(clientes),
        "n_con_alertas": n_con_alertas,
    }
