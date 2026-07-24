"""Cálculo pesado del bloque de logística del dashboard, extraído
del loop dentro de `app.py` para que también lo pueda correr un
cron externo (`scripts/precompute_dashboard.py`) y dejar el
resultado precomputado en la tabla `dashboard_cache`.

Filosofía: función PURA — no toca `st.session_state`, no llama a
Streamlit, no hace monkey-patch de nada. Solo lee la DB, corre el
cálculo, devuelve un dict. Fácil de testear, fácil de cachear.

Estructura del resultado (identical al que `app.py` guardaba en
`st.session_state["_dash_stock"]`):

    {
        "filas_log":                lista de alertas de stock,
        "stock_total_kg":           kg totales en campo (todos los lotes),
        "proxima_entrega_fecha":    ISO date de la primera reposición,
        "proxima_entrega_cliente":  nombre del cliente asociado,
        "autonomia":                barras por cliente-lote-producto,
        "entregas_sin_dieta":       entregas registradas sin dieta cargada,
    }
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def calcular_dashboard_logistica(
    hoy_iso: Optional[str] = None,
) -> Dict:
    """Corre el loop pesado que popula el bloque de logística.

    Args:
        hoy_iso: fecha de referencia YYYY-MM-DD. Default: hoy.
                 Útil para testing / reproducibilidad.

    Returns:
        Dict con las claves listadas arriba.
    """
    # Imports lazy — así el módulo se puede importar sin arrastrar
    # todo el árbol de dependencias si algún consumidor solo quiere
    # el shape del resultado.
    from src import database as db
    from src.stock_producto import (
        calcular_stock_actual,
        listar_productos_hms_lote,
        proyectar_fin_carga_silocomedero,
        _dieta_vigente,
        _mismo_producto,
        _es_a_discrecion,
    )

    if hoy_iso:
        _hoy_dt = datetime.strptime(hoy_iso, "%Y-%m-%d").date()
    else:
        _hoy_dt = datetime.now().date()
    _hoy_ref = _hoy_dt.isoformat()

    _filas_log: List[Dict] = []
    _stock_total_kg: float = 0.0
    _proxima_entrega_fecha: Optional[str] = None
    _proxima_entrega_cliente: str = ""
    _autonomia_por_cliente_lote: List[Dict] = []
    _entregas_sin_dieta: List[Dict] = []

    for _cli_log in db.listar_clientes():
        if _cli_log.get("estado", "activo") != "activo":
            continue
        _lotes_log = db.listar_lotes(
            cliente_id=_cli_log["id"], estado="activo",
        )
        for _l_log in _lotes_log:
            # Solo productos vendidos por HMS — no maíz/rollos/silaje
            # que el productor compra por su lado.
            _productos_log = listar_productos_hms_lote(
                _cli_log["id"], _l_log["id"]
            )
            if not _productos_log:
                # No hay match dieta ∩ entregas. Pero si hay
                # entregas registradas al lote, las mostramos
                # como "pendientes de dieta".
                try:
                    _ent_lote = db.listar_entregas_lote(_l_log["id"])
                except Exception:
                    _ent_lote = []
                if _ent_lote:
                    _por_prod_lote: Dict = defaultdict(
                        lambda: {"kg": 0, "fechas": []}
                    )
                    for _e in _ent_lote:
                        _por_prod_lote[
                            _e.get("producto_nombre", "?")
                        ]["kg"] += _e.get("kg_total") or 0
                        _por_prod_lote[
                            _e.get("producto_nombre", "?")
                        ]["fechas"].append(_e.get("fecha_entrega"))
                    for _p, _info in _por_prod_lote.items():
                        _entregas_sin_dieta.append({
                            "cliente": _cli_log["nombre"],
                            "lote": _l_log["identificador"],
                            "lote_id": _l_log["id"],
                            "producto": _p,
                            "kg_total": _info["kg"],
                            "ultima_fecha": (
                                max(_info["fechas"])
                                if _info["fechas"] else "—"
                            ),
                            "n_entregas": len(_info["fechas"]),
                        })
                continue
            for _prod_log in _productos_log:
                try:
                    _stock_log = calcular_stock_actual(
                        _cli_log["id"], _l_log["id"], _prod_log,
                    )
                except Exception:
                    continue
                if not _stock_log:
                    continue
                _dias = _stock_log.get("dias_restantes", 0)
                _kg_rest = _stock_log.get("kg_restantes_hoy", 0)
                _kg_dia = _stock_log.get("consumo_diario_kg", 0)
                _kg_entreg = _stock_log.get("kg_entregados_total", 0)
                # Necesitamos entregas registradas para tener algo
                # que mostrar.
                if _stock_log.get(
                    "diagnostico_uso") == "sin_entregas":
                    continue
                # Sumamos al stock total y registramos para las
                # barras de autonomía (todos los activos, no solo
                # los en alerta).
                _stock_total_kg += _kg_rest
                _fecha_agot = _stock_log.get("fecha_agotamiento")
                if _fecha_agot and _kg_rest > 0:
                    if (_proxima_entrega_fecha is None
                            or _fecha_agot < _proxima_entrega_fecha):
                        _proxima_entrega_fecha = _fecha_agot
                        _proxima_entrega_cliente = _cli_log["nombre"]
                # ¿Es silocomedero? Si sí, calculamos cuántas
                # cargas más del silo cubre el stock actual del
                # producto, en lugar de asumir consumo continuo
                # (que es lo que devuelve calcular_stock_actual).
                _es_silo_lt = (
                    (_l_log.get("tipo_comedero_concentrado")
                     or "").lower() == "silocomedero"
                )
                _cargas_rest = None
                _kg_prod_por_carga = None
                _fecha_prox_recarga = None
                _ultima_silo = None
                if _es_silo_lt:
                    try:
                        # % del producto HMS en la mezcla de la
                        # dieta vigente. Lo usamos para estimar
                        # cuánto producto se va por cada carga.
                        _dietas_lt = db.listar_dietas(_l_log["id"])
                        _dieta_lt = (
                            _dieta_vigente(_dietas_lt, _hoy_ref)
                            if _dietas_lt else None
                        )
                        _pct_prod_mezcla = 0.0
                        if _dieta_lt:
                            _kg_tc_prod = 0.0
                            _kg_tc_total = 0.0
                            for _c_d in (
                                _dieta_lt.get("composicion") or []
                            ):
                                _nom_d = (
                                    _c_d.get("nombre") or ""
                                ).strip()
                                _kg_tc = float(
                                    _c_d.get("kg_tal_cual") or 0
                                )
                                # Excluir libre disposición de
                                # la mezcla (rollo a voluntad, etc.)
                                try:
                                    if _es_a_discrecion(_nom_d):
                                        continue
                                except Exception:
                                    pass
                                _kg_tc_total += _kg_tc
                                if _mismo_producto(
                                    _nom_d, _prod_log,
                                ):
                                    _kg_tc_prod = _kg_tc
                            if _kg_tc_total > 0:
                                _pct_prod_mezcla = (
                                    _kg_tc_prod / _kg_tc_total
                                )
                        # kg cargados en la última vez al silo
                        _ultima_silo = db.ultima_carga_silocomedero(
                            _l_log["id"]
                        )
                        _kg_carga_silo = (
                            float(_ultima_silo["kg_cargados"])
                            if _ultima_silo
                            and _ultima_silo.get("kg_cargados")
                            else 0.0
                        )
                        # Si la última carga tiene desglose por
                        # ingrediente, usar el kg REAL del producto
                        # (más preciso que estimar con %).
                        _kg_prod_carga_real = 0.0
                        if _ultima_silo:
                            try:
                                _desg = json.loads(
                                    _ultima_silo.get(
                                        "desglose_ingredientes_json"
                                    ) or "[]"
                                )
                                for _d_ in _desg:
                                    if _mismo_producto(
                                        _d_.get("nombre", ""),
                                        _prod_log,
                                    ):
                                        _kg_prod_carga_real = (
                                            float(_d_.get("kg") or 0)
                                        )
                                        break
                            except Exception:
                                pass
                        if _kg_prod_carga_real > 0:
                            _kg_prod_por_carga = round(
                                _kg_prod_carga_real, 1,
                            )
                        elif (_pct_prod_mezcla > 0
                                and _kg_carga_silo > 0):
                            _kg_prod_por_carga = round(
                                _kg_carga_silo * _pct_prod_mezcla,
                                1,
                            )
                        if (_kg_prod_por_carga
                                and _kg_prod_por_carga > 0):
                            _cargas_rest = round(
                                _kg_rest / _kg_prod_por_carga, 1,
                            )
                        # Fecha en la que se agota la carga
                        # actual del silo (próxima recarga).
                        try:
                            _proy_silo = (
                                proyectar_fin_carga_silocomedero(
                                    _l_log["id"]
                                )
                            )
                            if _proy_silo:
                                _fecha_prox_recarga = (
                                    _proy_silo.get(
                                        "fecha_agotamiento"
                                    )
                                )
                        except Exception:
                            pass
                    except Exception:
                        pass
                # Para silocomedero: sobreescribir días/fecha con
                # la lógica de cargas (coherente con el bloque del
                # silo de abajo). Sin esto la barra muestra
                # consumo continuo, que NO se cumple en silo.
                _dias_final = _dias
                _fecha_agot_final = _fecha_agot
                if (_es_silo_lt and _cargas_rest is not None
                        and _fecha_prox_recarga):
                    try:
                        _f_prox = datetime.strptime(
                            _fecha_prox_recarga, "%Y-%m-%d"
                        ).date()
                        _dias_hasta_prox = max(
                            0, (_f_prox - _hoy_dt).days,
                        )
                        # Última fecha en la que TODAVÍA hay
                        # producto para cargar el silo. Si
                        # cargas_restantes < 1, ya no llega a la
                        # próxima recarga → la fecha crítica es
                        # el agotamiento del silo actual.
                        # Si >= 1, cada carga adicional dura
                        # aprox los mismos días que la actual.
                        _dias_por_carga = 0
                        try:
                            _dc = float(
                                (_ultima_silo or {}).get(
                                    "dias_cubiertos"
                                ) or 0
                            )
                            _dias_por_carga = (
                                _dc if _dc > 0 else 0
                            )
                        except Exception:
                            pass
                        # Caso especial: la carga actual del silo
                        # ya está agotada (dias_hasta_prox==0)
                        # pero todavía queda producto en stock.
                        # Eso es porque el productor todavía no
                        # cargó el silo de nuevo, pero tiene
                        # bolsas en el galpón. En ese caso el
                        # override daría 0, lo cual es confuso
                        # — caemos al cálculo continuo (que es
                        # lo que mejor refleja "para cuántos
                        # días te alcanza el stock que tenés").
                        if (_dias_hasta_prox == 0 and _kg_rest > 0):
                            _dias_final = _dias
                            _fecha_agot_final = _fecha_agot
                        elif _cargas_rest < 1:
                            _dias_final = _dias_hasta_prox
                            _fecha_agot_final = _fecha_prox_recarga
                        else:
                            _cargas_full = max(0, _cargas_rest - 1)
                            _dias_extra = int(round(
                                _cargas_full * _dias_por_carga
                            ))
                            _dias_final = (
                                _dias_hasta_prox + _dias_extra
                            )
                            _fecha_agot_final = (
                                _f_prox + timedelta(days=_dias_extra)
                            ).isoformat()
                    except Exception:
                        pass
                _autonomia_por_cliente_lote.append({
                    "cliente": _cli_log["nombre"],
                    "lote": _l_log["identificador"],
                    "lote_id": _l_log["id"],
                    "producto": _prod_log,
                    "kg_rest": _kg_rest,
                    "kg_entreg": _kg_entreg,
                    "dias": _dias_final,
                    "fecha_agot": _fecha_agot_final,
                    "es_silocomedero": _es_silo_lt,
                    "cargas_restantes": _cargas_rest,
                    "kg_prod_por_carga": _kg_prod_por_carga,
                    "fecha_prox_recarga": _fecha_prox_recarga,
                })
                # El filtro de alertas (≤14 días) lo seguimos
                # aplicando solo para la tabla detallada.
                if _dias > 14:
                    continue
                # Urgencia visual
                if _kg_rest <= 0:
                    _urg_ico = "🔴"
                    _urg_lbl = "AGOTADO"
                elif _dias <= 3:
                    _urg_ico = "🔴"
                    _urg_lbl = "URGENTE"
                elif _dias <= 7:
                    _urg_ico = "🟠"
                    _urg_lbl = "Esta semana"
                else:
                    _urg_ico = "🟡"
                    _urg_lbl = "Próxima semana"
                _filas_log.append({
                    "_dias_sort": _dias,
                    "Urgencia": f"{_urg_ico} {_urg_lbl}",
                    "Cliente": _cli_log["nombre"],
                    "Lote": _l_log["identificador"],
                    "Producto": _prod_log,
                    "Stock (kg)": f"{_kg_rest:.0f}",
                    "Consumo (kg/día)": f"{_kg_dia:.1f}",
                    "Días rest.": f"{_dias:.0f}",
                    "Se acaba": (
                        _stock_log.get("fecha_agotamiento") or "—"
                    ),
                    "Contacto": (
                        _cli_log.get("whatsapp")
                        or _cli_log.get("contacto") or "—"
                    ),
                })

    return {
        "filas_log": _filas_log,
        "stock_total_kg": _stock_total_kg,
        "proxima_entrega_fecha": _proxima_entrega_fecha,
        "proxima_entrega_cliente": _proxima_entrega_cliente,
        "autonomia": _autonomia_por_cliente_lote,
        "entregas_sin_dieta": _entregas_sin_dieta,
    }
