"""
App web (Streamlit) — Conteo y estimación de peso de bovinos por drone.

Cómo correrla:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# cv2 / OpenCV solo se usa para la pestaña de drone (conteo por video).
# En Streamlit Cloud NO lo instalamos para no pasarnos del límite de RAM.
# Si el import falla, deshabilitamos la pestaña drone pero el resto de la
# app (nutrición, dietas, alertas, clientes) sigue funcionando.
try:
    import cv2  # noqa: F401
    _DRONE_LIBS_OK = True
except Exception as _e_cv2:
    cv2 = None  # placeholder para que las referencias no rompan al parsear
    _DRONE_LIBS_OK = False

import numpy as np
import pandas as pd
import streamlit as st
import yaml


def _hash_bytes(data: bytes, *extra: str) -> str:
    """Hash de los datos + parámetros relevantes (para invalidar cache si
    el usuario cambia raza, categoría, ajuste_fino, etc.)."""
    h = hashlib.sha1(data)
    for e in extra:
        h.update(b"|")
        h.update(str(e).encode())
    return h.hexdigest()


def asdict_ing(ing) -> dict:
    """Convierte un Ingrediente a dict para que st.data_editor lo edite."""
    from dataclasses import asdict
    return asdict(ing)


# =====================================================================
# Persistencia local de la API key (entre sesiones / refreshes)
# =====================================================================

API_KEY_FILE = Path("data/.api_key")


def cargar_api_key_persistida() -> str:
    """Lee la API key del archivo local. Si no existe, intenta env var."""
    if API_KEY_FILE.exists():
        try:
            return API_KEY_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return os.environ.get("ANTHROPIC_API_KEY", "")


def guardar_api_key(key: str) -> None:
    """Guarda la API key en archivo local con permisos restrictivos."""
    API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not key:
        if API_KEY_FILE.exists():
            API_KEY_FILE.unlink()
        return
    API_KEY_FILE.write_text(key, encoding="utf-8")
    try:
        API_KEY_FILE.chmod(0o600)   # solo lectura/escritura del dueño
    except Exception:
        pass


def _safe_float(value, default: float = 0.0) -> float:
    """Convierte un valor a float, devolviendo el default si es None/NaN/inválido."""
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:  # NaN check (NaN != NaN)
            return default
        return f
    except (TypeError, ValueError):
        return default


def _slug(texto: str, max_len: int = 30) -> str:
    """Convierte un texto en una versión apta para nombre de archivo:
    sin tildes, sin caracteres especiales, sin espacios. Vacío si la
    entrada es None/vacía. Útil para armar nombres de PDF descriptivos.
    """
    import re
    import unicodedata
    if not texto:
        return ""
    s = str(texto).strip()
    if not s:
        return ""
    # Sacar tildes: 'á' → 'a', 'ñ' → 'n', etc.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Reemplazar caracteres no alfanuméricos por guion bajo
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = s.strip("_")
    # Limitar longitud
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def armar_nombre_pdf(
    cliente: str = "",
    categoria: str = "",
    objetivo: str = "",
    lote: str = "",
    fecha=None,
    prefijo: str = "HMS",
    sufijo: str = "informe",
) -> str:
    """Construye un nombre de archivo descriptivo para PDFs del sistema.

    Formato: ``{prefijo}_{sufijo}_{cliente}_{categoria}_{objetivo}_{YYYY-MM-DD}.pdf``

    Componentes vacíos se omiten. Todos los componentes se sanitizan
    (sin tildes, espacios → '_', truncado). El usuario sabe encontrar
    los PDFs por cliente y por fecha sin abrirlos.

    Ejemplos:
      armar_nombre_pdf("Ezequiel Pezzola", "ternero",
                         "ADG 0.8 kg/día", fecha=date(2026, 5, 19))
      → 'HMS_informe_Pezzola_ternero_ADG_0_8_kg_dia_2026-05-19.pdf'
    """
    from datetime import datetime as _dt, date as _date
    if fecha is None:
        fecha = _dt.now().date()
    elif isinstance(fecha, _dt):
        fecha = fecha.date()
    elif isinstance(fecha, str):
        try:
            fecha = _dt.strptime(fecha[:10], "%Y-%m-%d").date()
        except ValueError:
            fecha = _dt.now().date()
    partes = [prefijo, sufijo]
    # Para el cliente, usar solo el apellido si tiene varias palabras
    # (más legible y más corto). "Ezequiel Pezzola" → "Pezzola".
    if cliente:
        cliente_str = str(cliente).strip()
        if " " in cliente_str:
            cliente_str = cliente_str.split()[-1]
        partes.append(_slug(cliente_str, max_len=20))
    if lote:
        partes.append(_slug(lote, max_len=15))
    if categoria:
        partes.append(_slug(categoria, max_len=15))
    if objetivo:
        partes.append(_slug(objetivo, max_len=25))
    partes.append(fecha.isoformat())
    partes = [p for p in partes if p]
    return "_".join(partes) + ".pdf"

# Módulos que dependen de cv2/ultralytics — solo importables si _DRONE_LIBS_OK.
# En Streamlit Cloud estos NO se instalan (ver requirements-cloud.txt) y la
# pestaña drone queda deshabilitada. En local funciona normal.
#
# Cuando NO están disponibles, en vez de None usamos "stubs" — clases y
# funciones que aceptan cualquier argumento y devuelven None. Así, todo
# código que hace `WeightModel.from_config(...)` o `CattleDetector(...)`
# devuelve None en lugar de tirar AttributeError/TypeError.
class _DroneStub:
    """Clase dummy que devuelve None para cualquier método/atributo/llamada."""
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return None
    def __getattr__(self, name):
        return _DroneStub()

def _drone_stub_fn(*args, **kwargs):
    """Función dummy que siempre devuelve None."""
    return None

if _DRONE_LIBS_OK:
    try:
        from src.calibration import calibrate, calibrate_from_altitude
        from src.detector import CattleDetector
        from src.processor import (
            export_results_csv,
            process_image,
            process_video,
        )
        from src.weight_estimator import WeightModel
    except Exception:
        _DRONE_LIBS_OK = False
        calibrate = calibrate_from_altitude = _drone_stub_fn
        CattleDetector = _DroneStub
        export_results_csv = process_image = process_video = _drone_stub_fn
        WeightModel = _DroneStub
else:
    calibrate = calibrate_from_altitude = _drone_stub_fn
    CattleDetector = _DroneStub
    export_results_csv = process_image = process_video = _drone_stub_fn
    WeightModel = _DroneStub
from src.nutritional_analysis import (
    analizar_uniformidad, calcular_requerimientos, proyectar_peso,
    ajustar_req_por_dmi,
)
from src.pdf_report import generar_pdf
from src.feed_optimizer import (
    Ingrediente, formular_minimo_costo, ingredientes_default,
    verificar_receta,
)
from src.ai_agent import (
    SYSTEM_PROMPT, chat_streaming, construir_contexto_lote,
    construir_contexto_ingredientes,
)
from src.pdf_informe import generar_pdf_informe_chat
from src.clima import (
    resumen_clima_para_ia, geocodificar, obtener_clima,
    calcular_thi, clasificar_thi, generar_alertas_predictivas,
)
from src.clima_lapampa import (
    obtener_datos_estacion, estacion_mas_cercana, resumen_estacion_oficial,
    guardar_datos_manuales, obtener_datos_manuales, URL_BASE as URL_LA_PAMPA,
)
from src.mapa_widget import render_mapa_seleccion
from src import agent_memory as memoria
from src import database as db
from src import dashboard
# training_helper importa cv2 → solo cargarlo si _DRONE_LIBS_OK.
# En Streamlit Cloud lite queda deshabilitado y la pestaña de
# entrenamiento del drone no está disponible. Usamos _DroneStub()
# (no None) para que llamadas tipo `training.generar_xxx()` no
# lancen AttributeError — devuelven None silenciosamente.
if _DRONE_LIBS_OK:
    try:
        from src import training_helper as training
    except Exception:
        training = _DroneStub()
        _DRONE_LIBS_OK = False
else:
    training = _DroneStub()
db.init_db()


# =====================================================================
# Bloque: 📸 FOTOS DE INSPECCIÓN DEL LOTE
# Permite cargar fotos del corral/bosta/comedero/animales que el
# productor manda durante una entrevista. Quedan asociadas al lote
# (y opcionalmente a una consulta puntual) y aparecen después en el
# PDF de informe del lote.
# Vive FUERA del st.form() del cuestionario porque st.form no soporta
# file_uploader de forma limpia.
# =====================================================================
def _render_bloque_fotos_lote(*, lote_id, lote):
    """Bloque "Fotos de inspección" en la ficha clínica del lote."""
    import streamlit as _st
    from pathlib import Path as _Path
    try:
        from src import fotos_lote as _fl
    except Exception as _e:
        _st.warning(f"No se pudo cargar módulo de fotos: {_e}")
        return

    _st.markdown("##### 📸 Fotos de inspección")
    _st.caption(
        "Subí las fotos que mandó el productor/operario (corrales, "
        "bosta, animales, comederos). Quedan asociadas al lote y se "
        "incluyen en el informe PDF."
    )

    # Consultas recientes para asociar la foto.
    # Tomamos los recordatorios COMPLETADOS del cliente que sean del
    # mismo lote y los mostramos como dropdown opcional.
    try:
        _cli_id_lote = lote.get("cliente_id")
        if _cli_id_lote:
            _ult_recs = db.listar_recordatorios_cliente(
                cliente_id=_cli_id_lote,
                incluir_completados=True,
            ) or []
        else:
            _ult_recs = []
    except Exception:
        _ult_recs = []
    _ult_recs = [
        r for r in _ult_recs
        if (r.get("lote_id") or 0) == lote_id
        and r.get("estado") == "completado"
    ][:5]
    _ops_consulta = {"— Sin asociar a consulta puntual —": None}
    for _r in _ult_recs:
        _f = (_r.get("completado_en") or _r.get("creado_en") or "")[:10]
        _ops_consulta[
            f"{_f} · consulta #{_r['id']}"
        ] = _r["id"]

    # Form de subida (form porque queremos un solo botón "Guardar todas")
    with _st.form(f"form_fotos_lote_{lote_id}", clear_on_submit=True):
        _archivos = _st.file_uploader(
            "Arrastrá o seleccioná fotos (podés subir varias a la vez)",
            type=["jpg", "jpeg", "png", "heic", "webp"],
            accept_multiple_files=True,
            key=f"upl_fotos_{lote_id}",
        )
        _c1, _c2 = _st.columns(2)
        _tipo_label = _c1.selectbox(
            "Categoría aplicada a TODAS las fotos del lote",
            options=list(db.TIPOS_FOTO_LOTE.keys()),
            format_func=lambda k: (
                f"{db.TIPOS_FOTO_LOTE[k]['emoji']} "
                f"{db.TIPOS_FOTO_LOTE[k]['label']}"
            ),
            key=f"tipo_fotos_{lote_id}",
        )
        _consulta_label = _c2.selectbox(
            "Asociar a consulta (opcional)",
            options=list(_ops_consulta.keys()),
            key=f"cons_fotos_{lote_id}",
            help=(
                "Si la foto corresponde a una llamada/visita ya "
                "registrada, asocialá. Si no, dejá 'Sin asociar'."
            ),
        )
        _comentario = _st.text_input(
            "Comentario común (opcional — se aplica a todas)",
            placeholder=(
                "Ej: bosta verde clara, exceso PB · comedero con "
                "costra al fondo · barro en línea de comer"
            ),
            key=f"com_fotos_{lote_id}",
        )
        _submit = _st.form_submit_button(
            "📸 Subir fotos al lote", type="primary",
        )

    if _submit and _archivos:
        _rec_id = _ops_consulta.get(_consulta_label)
        _exitos = 0
        _errores = []
        for _archivo in _archivos:
            try:
                _fl.guardar_archivo_subido(
                    uploaded_file=_archivo,
                    lote_id=lote_id,
                    recordatorio_id=_rec_id,
                    tipo=_tipo_label,
                    comentario=_comentario or "",
                )
                _exitos += 1
            except Exception as _e:
                _errores.append(f"{_archivo.name}: {_e}")
        if _exitos:
            _st.success(
                f"✅ {_exitos} foto(s) subida(s) al lote."
            )
        if _errores:
            _st.error("Algunas fotos no se pudieron guardar:\n" +
                       "\n".join(f"- {x}" for x in _errores))
        _st.rerun()

    # ── Galería existente del lote ──
    _por_tipo = _fl.listar_fotos_categorizadas(lote_id)
    _total = sum(len(v) for v in _por_tipo.values())
    if _total == 0:
        _st.info(
            "📷 Todavía no hay fotos cargadas para este lote."
        )
        return

    # Botón de descarga del PDF
    _c_pdf, _ = _st.columns([1, 3])
    if _c_pdf.button(
        "📄 Generar PDF de inspección",
        key=f"btn_pdf_inspeccion_{lote_id}",
        help=(
            "Descarga un PDF con todas las fotos del lote "
            "agrupadas por categoría, con comentarios y fechas."
        ),
    ):
        try:
            from src.pdf_lote_inspeccion import (
                generar_pdf_inspeccion_lote as _gen,
            )
            _pdf_bytes = _gen(lote_id=lote_id)
            _nombre = (
                f"inspeccion_{lote.get('cliente_nombre','cliente')}_"
                f"{lote.get('identificador','lote')}_"
                f"{datetime.now().strftime('%Y%m%d')}.pdf"
            ).replace(" ", "_")
            _st.session_state[f"pdf_insp_{lote_id}"] = (
                _pdf_bytes, _nombre,
            )
        except Exception as _e:
            _st.error(f"No se pudo generar el PDF: {_e}")
    _pdf_state = _st.session_state.get(f"pdf_insp_{lote_id}")
    if _pdf_state:
        _bytes, _nombre_pdf = _pdf_state
        _st.download_button(
            "⬇️ Descargar PDF",
            data=_bytes,
            file_name=_nombre_pdf,
            mime="application/pdf",
            key=f"dl_pdf_insp_{lote_id}",
        )

    _st.markdown(f"**Galería ({_total} foto(s))**")
    # Renderizar agrupado por tipo, con thumbnails en columnas de 4
    for _tipo_key, _tipo_meta in db.TIPOS_FOTO_LOTE.items():
        _fotos_tipo = _por_tipo.get(_tipo_key, [])
        if not _fotos_tipo:
            continue
        _st.markdown(
            f"**{_tipo_meta['emoji']} {_tipo_meta['label']}** "
            f"({len(_fotos_tipo)})"
        )
        _cols = _st.columns(4)
        for _i, _f in enumerate(_fotos_tipo):
            _col = _cols[_i % 4]
            with _col:
                if _f.get("existe"):
                    try:
                        _col.image(
                            _f["archivo_path"],
                            caption=(
                                (_f.get("comentario") or "—")[:80]
                            ),
                            width="stretch",
                        )
                    except Exception:
                        _col.warning(
                            f"⚠️ {_Path(_f['archivo_path']).name}"
                        )
                else:
                    _col.warning(
                        "🗑️ Archivo borrado del disco "
                        "(referencia DB colgando)"
                    )
                _col.caption(
                    f"📅 {(_f.get('fecha') or '')[:16]} · #{_f['id']}"
                )
                if _col.button(
                    "🗑️ Borrar",
                    key=f"del_foto_{_f['id']}",
                    help="Borra la foto y su archivo en disco",
                ):
                    db.eliminar_foto_lote(_f["id"], borrar_archivo=True)
                    _st.rerun()


# =====================================================================
# Visualización: comparación de DIETA REAL vs FORMULADA de una consulta.
# Se muestra dentro del expander de cada consulta en la ficha clínica.
# Lee el evaluacion_json del recordatorio + la dieta vigente del lote
# y renderiza una tabla con semáforo por ingrediente.
# =====================================================================
def _render_comparacion_dieta_real_consulta(*, recordatorio_id, lote_id):
    """Renderiza la tabla de comparación dieta REAL vs formulada.

    Args:
        recordatorio_id: id del recordatorio (consulta) a analizar.
        lote_id: id del lote para traer la dieta vigente.
    """
    import streamlit as _st
    import json as _json

    try:
        with db.get_conn() as _conn:
            _row = _conn.execute(
                "SELECT evaluacion_json FROM recordatorios_llamada "
                "WHERE id = ?",
                (recordatorio_id,),
            ).fetchone()
        if not _row or not _row[0]:
            return
        _ev = _json.loads(_row[0])
    except Exception:
        return
    if not isinstance(_ev, dict):
        return

    _items = _ev.get("dieta_real_items") or []
    if not _items:
        return  # nada que mostrar
    _modo = _ev.get("dieta_real_modo") or "animal_dia"

    # Traer dieta vigente del lote para comparar
    try:
        _dietas = db.listar_dietas(lote_id) or []
        _dietas.sort(key=lambda d: (d.get("fecha") or ""), reverse=True)
        _dieta_vig = _dietas[0] if _dietas else None
    except Exception:
        _dieta_vig = None
    if not _dieta_vig or not _dieta_vig.get("composicion"):
        return

    # Traer cantidad de animales del lote (para normalizar si total_dia)
    try:
        _lote = db.obtener_lote(lote_id) or {}
        _cant_an = int(_lote.get("cantidad_inicial") or 0)
    except Exception:
        _cant_an = 0

    # Llamar a la comparación del módulo evaluacion_lote
    try:
        from src import evaluacion_lote as _ev_mod
        # Re-armar RespuestasEvaluacion mínimo (solo lo que la
        # comparación usa)
        _r = _ev_mod.RespuestasEvaluacion(
            cliente_nombre="",
            lote_id=lote_id,
            lote_identificador="",
            dieta_real_modo=_modo,
            dieta_real_items=_items,
        )
        _ctx = {
            "cantidad_inicial": _cant_an,
            "dieta_vigente": {
                "composicion": _dieta_vig.get("composicion", []),
            },
        }
        _resultado = _ev_mod.analizar_evaluacion(
            _r, contexto_lote=_ctx,
        )
        _comparacion = _resultado.get("comparacion_dieta_real", [])
    except Exception:
        return

    if not _comparacion:
        return

    _st.divider()
    _st.markdown("**🌾 Comparación: dieta REAL del cliente vs formulada HMS**")
    _modo_lbl = (
        "kg/animal/día" if _modo == "animal_dia"
        else "kg totales/día (normalizado)"
    )
    _st.caption(
        f"Modo de carga: **{_modo_lbl}** · "
        f"Dieta formulada referencia: **{_dieta_vig.get('fecha', '—')}**"
    )

    # Render tabla con semáforos
    _emoji = {
        "verde": "🟢", "atencion": "🟡",
        "urgente": "🔴",
    }
    import pandas as _pd
    _df = _pd.DataFrame([
        {
            " ": _emoji.get(c["semaforo"], "⚪"),
            "Ingrediente": c["nombre"],
            "Real": f"{c['kg_real_animal_dia']:.2f}",
            "Formulado": f"{c['kg_formulado_animal_dia']:.2f}",
            "Δ kg/an/día": f"{c['desvio_kg']:+.2f}",
            "Δ %": f"{c['desvio_pct']:+.0f}%",
        }
        for c in _comparacion
    ])
    _st.dataframe(_df, hide_index=True, width="stretch")

    # Resumen contadores
    _n_ok = sum(1 for c in _comparacion if c["semaforo"] == "verde")
    _n_at = sum(1 for c in _comparacion if c["semaforo"] == "atencion")
    _n_ur = sum(1 for c in _comparacion if c["semaforo"] == "urgente")
    _total = len(_comparacion)
    _cap = (
        f"**{_n_ok}/{_total}** alineados (±10%) · "
        f"{_n_at} en atención · "
        f"{_n_ur} con desvío grande."
    )
    if _n_ur > 0:
        _st.error(_cap)
    elif _n_at > 0:
        _st.warning(_cap)
    else:
        _st.success(_cap)


# =====================================================================
# Form de EDICIÓN RÁPIDA de consulta existente
# Permite corregir los campos que más se cargan mal (stock, silo, etc.)
# sin tener que rehacer toda la consulta. Hace UPDATE del JSON.
# =====================================================================
def _render_form_edicion_rapida(*, recordatorio_id, datos_actuales,
                                  on_close_state):
    """Mini-form para editar campos críticos de una consulta existente.

    Args:
        recordatorio_id: id del recordatorio_llamada a editar.
        datos_actuales: EvaluacionRegistrada con los valores actuales.
        on_close_state: clave de session_state para cerrar el form.
    """
    import streamlit as _st
    import json as _json

    _form_key = f"form_edit_consulta_{recordatorio_id}"
    with _st.form(_form_key):
        _st.markdown("##### ✏️ Editar consulta — campos críticos")
        _st.caption(
            "Corregí solo los campos que están mal. El resto de la "
            "consulta (rumia, observaciones, acciones) queda intacto. "
            "Si querés rehacerla entera, mejor creá una nueva."
        )

        # ── Stock disponible (los que más confunden) ──
        _st.markdown("**📦 Stock disponible — STOCK TOTAL almacenado**")
        _st.caption(
            "⚠️ Es lo que el cliente tiene almacenado HOY en el "
            "campo, NO lo que se tira al comedero por día."
        )
        _c1, _c2, _c3 = _st.columns(3)
        _maiz_kg = _c1.number_input(
            "Maíz — STOCK TOTAL (kg)",
            min_value=0.0,
            value=float(datos_actuales.maiz_kg or 0),
            step=100.0,
            key=f"{_form_key}_maiz",
            help=(
                "Toneladas almacenadas hoy. Ej: 3 toneladas → 3000."
            ),
        )
        _fg_kg = _c2.number_input(
            "Fibrogreen — STOCK TOTAL (kg)",
            min_value=0.0,
            value=float(datos_actuales.fg_kg or 0),
            step=25.0,
            key=f"{_form_key}_fg",
            help="Ej: 8 bolsas de 30 kg → 240.",
        )
        # Silo nivel (0-100% o -1 = sin silo)
        _opciones_silo = [
            "—  Sin silo / no aplica",
            "0% — vacío",
            "25% — bajo",
            "50% — medio",
            "75% — alto",
            "100% — lleno",
        ]
        _silo_actual_idx = 0
        if datos_actuales.silo_pct is not None:
            _mapa = {-1: 0, 0: 1, 25: 2, 50: 3, 75: 4, 100: 5}
            _silo_actual_idx = _mapa.get(
                int(datos_actuales.silo_pct), 0,
            )
        _silo_label = _c3.selectbox(
            "Silocomedero — nivel actual",
            options=_opciones_silo,
            index=_silo_actual_idx,
            key=f"{_form_key}_silo",
        )

        # ── Movimientos del lote ──
        _st.markdown("**🐄 Movimientos del lote**")
        _c4, _c5 = _st.columns(2)
        _bajas = _c4.number_input(
            "Bajas (muertes) en últimas 48 hs",
            min_value=0,
            value=int(datos_actuales.bajas or 0),
            step=1,
            key=f"{_form_key}_bajas",
        )
        _enfermos = _c5.number_input(
            "Animales enfermos / tratamiento",
            min_value=0,
            value=int(datos_actuales.enfermos or 0),
            step=1,
            key=f"{_form_key}_enf",
        )

        _c6, _c7 = _st.columns(2)
        _save = _c6.form_submit_button(
            "💾 Guardar cambios", type="primary",
        )
        _cancel = _c7.form_submit_button("❌ Cancelar")

    if _cancel:
        _st.session_state[on_close_state] = False
        _st.rerun()

    if _save:
        # Aplicar los cambios al JSON existente
        try:
            with db.get_conn() as _conn:
                _row = _conn.execute(
                    "SELECT evaluacion_json FROM recordatorios_llamada "
                    "WHERE id = ?",
                    (recordatorio_id,),
                ).fetchone()
                if not _row or not _row[0]:
                    _st.error("No se pudo leer la consulta original.")
                    return
                _ev = _json.loads(_row[0])
                _ev["maiz_kg_disponible"] = float(_maiz_kg)
                _ev["fibrogreen_kg_disponible"] = float(_fg_kg)
                _ev["bajas_48hs"] = int(_bajas)
                _ev["animales_enfermos"] = int(_enfermos)
                # Parsear silo (extraer % o -1)
                _silo_pct = -1
                if "%" in _silo_label:
                    try:
                        _silo_pct = int(
                            _silo_label.split("%")[0].strip()
                        )
                    except Exception:
                        _silo_pct = -1
                _ev["silo_nivel_pct"] = _silo_pct
                # Marca de edición para auditoría
                _ev["_editado_en"] = datetime.now().isoformat(
                    timespec="seconds",
                )
                _conn.execute(
                    "UPDATE recordatorios_llamada "
                    "SET evaluacion_json = ? WHERE id = ?",
                    (_json.dumps(_ev, ensure_ascii=False),
                     recordatorio_id),
                )
            _st.success(
                "✅ Consulta actualizada. Recargá la página para "
                "ver los cambios reflejados."
            )
            _st.session_state[on_close_state] = False
            _st.rerun()
        except Exception as _e:
            _st.error(f"Error al guardar: {_e}")


# =====================================================================
# Form reutilizable: EVALUACIÓN ESTRUCTURADA DE LOTE durante llamada
# Lo usan tanto los recordatorios pendientes (al "registrar lo
# conversado") como "Registrar conversación ahora" en el dashboard.
# Hace TODO en un solo lugar: cuestionario + análisis + guardado.
# =====================================================================
def _renderizar_form_evaluacion(
    *,
    recordatorio_id,
    cliente_id: int,
    cliente_nombre: str,
    ev_mod,
    on_close_state: str,
):
    """Renderiza el cuestionario estructurado para evaluar el lote.

    Args:
        recordatorio_id: id del recordatorio EXISTENTE a cerrar, o
            None si es una conversación espontánea (se crea uno
            nuevo al guardar).
        cliente_id: id del cliente que estamos evaluando.
        cliente_nombre: para mostrar en el header.
        ev_mod: módulo src.evaluacion_lote (pasado para no
            re-importar en cada llamada).
        on_close_state: clave de st.session_state que controla la
            visibilidad del form (la limpiamos al guardar/cancelar).
    """
    import streamlit as _st  # alias para no chocar con outer
    # Cargar lotes del cliente para que el asesor elija cuál evalúa
    _lotes_cli = db.listar_lotes(
        cliente_id=cliente_id, estado="activo",
    ) or []
    _opciones_lotes = {
        f"{l['identificador']} · "
        f"{l.get('categoria','—')} · "
        f"{l.get('cantidad_inicial', 0)} cab": l["id"]
        for l in _lotes_cli
    }
    _form_key = f"form_eval_{recordatorio_id or 'nuevo'}_{cliente_id}"
    with _st.form(_form_key):
        _st.markdown(
            f"#### 📋 Evaluación del lote — {cliente_nombre}"
        )
        _st.caption(
            "Cuestionario estructurado para diagnosticar el estado "
            "del lote y el stock. Cuando guardes, el sistema "
            "cruza tus respuestas con la dieta vigente y te sugiere "
            "acciones concretas."
        )

        # ── Sección 1: contexto ──
        _st.markdown("##### 1. Contacto")
        _c1, _c2 = _st.columns(2)
        _tipo = _c1.selectbox(
            "Tipo de contacto",
            [
                "📞 Llamada telefónica",
                "💬 WhatsApp",
                "📧 Email",
                "🤝 Visita personal",
                "🚫 No atendió / sin respuesta",
            ],
            key=f"{_form_key}_tipo",
        )
        _atendio = _c2.text_input(
            "¿Quién atendió / con quién hablaste?",
            placeholder="Ej: Pedro - encargado",
            key=f"{_form_key}_atendio",
        )

        # Si tiene varios lotes, dejá elegir; si tiene uno solo, default
        _lote_label = None
        _lote_id_sel = None
        if _opciones_lotes:
            _lote_label = _st.selectbox(
                "Lote evaluado",
                options=list(_opciones_lotes.keys()),
                key=f"{_form_key}_lote",
            )
            _lote_id_sel = _opciones_lotes.get(_lote_label)
        else:
            _st.info(
                "ℹ️ Este cliente no tiene lotes activos. "
                "Igual podés registrar la conversación, "
                "pero el análisis cruzado va a ser limitado."
            )

        # ── Sección 2: estado de los animales ──
        _st.markdown("##### 2. Estado de los animales")
        _aspecto = _st.selectbox(
            "Aspecto general — ¿cómo se ven?",
            ev_mod.OPCIONES_ASPECTO_ANIMALES,
            key=f"{_form_key}_aspecto",
        )
        _c2a, _c2b = _st.columns(2)
        _bajas = _c2a.number_input(
            "💀 Muertes (mortandad) en últimas 48 hs",
            min_value=0, value=0, step=1,
            key=f"{_form_key}_bajas",
            help=(
                "Solo animales muertos por problema sanitario. "
                "Al guardar, se registra automáticamente como "
                "una 'baja' en los Movimientos de hacienda del "
                "lote, así las proyecciones de consumo y stock "
                "quedan ajustadas."
            ),
        )
        _enfermos = _c2b.number_input(
            "🤒 Animales enfermos / aislados",
            min_value=0, value=0, step=1,
            key=f"{_form_key}_enfermos",
            help=(
                "Animales aislados en sanitario, con tratamiento "
                "veterinario en curso, o que el encargado "
                "identifica como 'no andan bien'. "
                "(No descuenta cantidad del lote.)"
            ),
        )

        # ─── Causa de muerte (clave para diagnóstico) ───
        # Solo aparece si reportaron muertes. La causa es CRÍTICA
        # porque las acciones de prevención son completamente
        # distintas entre acidosis / neumonía / timpanismo / etc.
        _causa_muerte = ""
        if int(_bajas) > 0:
            _causa_muerte = _st.selectbox(
                "🔬 Causa de muerte (lo que reportó el cliente / "
                "necropsia)",
                ev_mod.OPCIONES_CAUSA_MUERTE,
                key=f"{_form_key}_causa_muerte",
                help=(
                    "Cada causa tiene un manejo preventivo "
                    "completamente distinto. Si no se determinó "
                    "todavía, marcá 'Sin determinar' — el sistema "
                    "va a sugerir hacer necropsia / fotos. "
                    "Si fueron varias causas, registrá la "
                    "más frecuente y aclará en el detalle."
                ),
            )

        # ─── Movimientos del lote (ventas/salidas) ───
        # Se registran automáticamente al guardar la evaluación,
        # así el asesor no tiene que ir a 'Movimientos de hacienda'
        # del lote después.
        _st.markdown(
            "**🐄 ¿Hubo ventas o salidas desde el último contacto?** "
            "_Se registran solas como movimiento en el lote._"
        )
        _c_mv1, _c_mv2 = _st.columns(2)
        _ventas = _c_mv1.number_input(
            "Animales vendidos / salidos",
            min_value=0, value=0, step=1,
            key=f"{_form_key}_ventas",
            help=(
                "Animales que salieron del lote (venta de gordos, "
                "descarte, traslado). Si fueron a otro lote, "
                "después podés ajustar en Movimientos de hacienda."
            ),
        )
        _kg_prom_ventas = _c_mv2.number_input(
            "Peso promedio venta (kg)",
            min_value=0.0, value=0.0, step=10.0,
            key=f"{_form_key}_kg_prom_ventas",
            help=(
                "Opcional. Útil cuando son ventas de gordos para "
                "que el sistema calcule kilos vendidos."
            ),
            disabled=(int(_ventas) == 0),
        )
        _detalle_mov = ""
        if int(_bajas) > 0 or int(_ventas) > 0:
            _detalle_mov = _st.text_input(
                "Detalle del movimiento (opcional)",
                placeholder=(
                    "Ej: 'venta a frigorífico X' / "
                    "'baja por neumonía'"
                ),
                key=f"{_form_key}_detalle_mov",
            )

        # ── Sección 3: consumo y rumen ──
        _st.markdown("##### 3. Consumo y rumen")
        _comedero = _st.selectbox(
            "Estado del comedero al pasar",
            ev_mod.OPCIONES_COMEDERO,
            key=f"{_form_key}_comedero",
        )
        _heces = _st.selectbox(
            "Heces observadas",
            ev_mod.OPCIONES_HECES,
            key=f"{_form_key}_heces",
        )

        # ── Sección 4: ambiente ──
        _st.markdown("##### 4. Ambiente y manejo")
        _c4a, _c4b, _c4c = _st.columns(3)
        _agua = _c4a.selectbox(
            "Agua / bebederos",
            ev_mod.OPCIONES_AGUA,
            key=f"{_form_key}_agua",
        )
        _cama = _c4b.selectbox(
            "Zona de descanso",
            ev_mod.OPCIONES_CAMA,
            key=f"{_form_key}_cama",
        )
        _reparos = _c4c.selectbox(
            "Reparos del viento",
            ev_mod.OPCIONES_REPAROS,
            key=f"{_form_key}_reparos",
        )

        # ── Sección 5: stock declarado (lo que tiene en campo) ──
        _st.markdown(
            "##### 5. Stock de mercadería en campo HOY"
        )
        _st.caption(
            "📦 **STOCK TOTAL almacenado** en el campo (silo, "
            "galpón, depósito) — NO lo que se tira al comedero "
            "por día. Ejemplo: si el cliente recibió 5 toneladas "
            "de maíz hace 3 semanas y le quedan ~3 toneladas, "
            "cargá `3000` kg. El sistema cruza esto con el "
            "consumo diario de la dieta para calcular cuántos "
            "días de autonomía quedan."
        )
        _c5a, _c5b, _c5c = _st.columns(3)
        _maiz_kg = _c5a.number_input(
            "Maíz — STOCK TOTAL (kg)",
            min_value=0.0, value=0.0, step=100.0,
            key=f"{_form_key}_maiz_kg",
            help=(
                "Toneladas/kg almacenados HOY en el campo. "
                "Ejemplo: 'tiene 3 toneladas en el silo' → 3000. "
                "NO cargar acá los kg/día que se mueven al "
                "comedero — eso se calcula automáticamente "
                "desde la dieta."
            ),
        )
        _fg_kg = _c5b.number_input(
            "Fibrogreen Plus — STOCK TOTAL (kg)",
            min_value=0.0, value=0.0, step=25.0,
            key=f"{_form_key}_fg_kg",
            help=(
                "Bolsas de Fibrogreen × 30 kg que tiene "
                "almacenadas el cliente HOY. Ejemplo: 'le "
                "quedan 8 bolsas' → 240 kg."
            ),
        )
        _rollos = _c5c.number_input(
            "Rollos disponibles (unidades)",
            min_value=0, value=0, step=1,
            key=f"{_form_key}_rollos",
            help=(
                "Cantidad de rollos que tiene el cliente en "
                "el campo (no kg). Sirve como buffer ante "
                "frente frío o demora en entrega."
            ),
        )

        _st.markdown(
            "**Silocomedero** "
            "_(si corresponde — sirve para chequear consumo real "
            "vs proyectado)_"
        )
        _c6a, _c6b, _c6c = _st.columns(3)
        _silo_nivel = _c6a.selectbox(
            "Nivel aproximado HOY",
            ev_mod.OPCIONES_SILO_NIVEL,
            key=f"{_form_key}_silo_nivel",
        )
        _dias_carga = _c6b.number_input(
            "Hace cuántos días fue la última carga",
            min_value=-1, value=-1, step=1,
            help="Dejá en -1 si no se sabe / no aplica",
            key=f"{_form_key}_dias_carga",
        )
        _kg_carga = _c6c.number_input(
            "Cuántos kg cargó esa última vez",
            min_value=0.0, value=0.0, step=100.0,
            key=f"{_form_key}_kg_carga",
        )

        # ── Sección 5b: DIETA REAL que aplica el cliente ──
        # Tabla con autocompletar del catálogo + pre-carga con la
        # dieta formulada vigente del lote (solo te falta poner los
        # kg reales). Compara contra dieta formulada para detectar
        # subdosis, exceso o ingredientes faltantes.
        _st.markdown("##### 5b. 🌾 Dieta REAL que aplica el cliente")
        _st.caption(
            "Lo que el productor te dijo en la entrevista que "
            "está REALMENTE tirando al comedero. La tabla viene "
            "**pre-cargada** con los ingredientes de la dieta "
            "formulada del lote — solo poné los kg reales en cada "
            "fila (o ajustá si el cliente usa algo distinto)."
        )
        _modo_dieta_real = _st.radio(
            "¿Cómo te lo pasó el cliente?",
            options=[
                "kg por animal por día",
                "kg totales del mixer por día",
            ],
            horizontal=True,
            key=f"{_form_key}_modo_dieta_real",
            help=(
                "Elegí según cómo te lo dijo el productor. El "
                "sistema normaliza internamente para comparar."
            ),
        )

        # Pre-cargar tabla con ingredientes de la dieta formulada
        # vigente del lote + lista del catálogo de feed_optimizer
        # para autocompletar nombres comunes.
        _import_pd = __import__("pandas")
        _ingredientes_precargados = []
        _opciones_catalogo: list = []
        try:
            from src.feed_optimizer import ingredientes_default as _ing_def
            _opciones_catalogo = sorted({
                ing.nombre for ing in _ing_def(precio_actualizado=False)
            })
        except Exception:
            _opciones_catalogo = []
        # Sumar los ingredientes típicos HMS (Fibrogreen, Cubre rollo,
        # núcleos comerciales, etc.) si no están ya
        for _extra in [
            "Fibrogreen Plus", "Cubre rollo", "Núcleo proteico",
            "Rollo de alfalfa", "Rollo de avena", "Sal mineral",
        ]:
            if _extra not in _opciones_catalogo:
                _opciones_catalogo.append(_extra)

        # Traer la dieta vigente del lote para pre-cargar la tabla
        try:
            if _lote_id_sel:
                _dietas_lote = db.listar_dietas(_lote_id_sel) or []
                # Más reciente primero
                _dietas_lote.sort(
                    key=lambda d: (d.get("fecha") or ""), reverse=True,
                )
                _dieta_vig = _dietas_lote[0] if _dietas_lote else None
                if _dieta_vig and _dieta_vig.get("composicion"):
                    for _c in _dieta_vig["composicion"]:
                        _nm = (_c.get("nombre") or "").strip()
                        if not _nm:
                            continue
                        _ingredientes_precargados.append({
                            "Ingrediente": _nm,
                            "Kg": 0.0,
                        })
                        if _nm not in _opciones_catalogo:
                            _opciones_catalogo.append(_nm)
        except Exception:
            pass
        if not _ingredientes_precargados:
            _ingredientes_precargados = [
                {"Ingrediente": "", "Kg": 0.0},
            ]

        _df_dieta_real = _import_pd.DataFrame(_ingredientes_precargados)
        _unidad_label = (
            "Kg / animal / día"
            if "animal" in _modo_dieta_real
            else "Kg totales / día"
        )
        _dieta_real_edit = _st.data_editor(
            _df_dieta_real,
            num_rows="dynamic",
            column_config={
                "Ingrediente": _st.column_config.SelectboxColumn(
                    "Ingrediente",
                    options=sorted(_opciones_catalogo),
                    help=(
                        "Elegí del catálogo o tipeá uno nuevo. La "
                        "lista incluye los ingredientes típicos "
                        "argentinos + los HMS (Fibrogreen, etc.) + "
                        "los que ya tiene la dieta del lote."
                    ),
                    required=False,
                ),
                "Kg": _st.column_config.NumberColumn(
                    _unidad_label,
                    min_value=0.0, step=0.1, format="%.2f",
                ),
            },
            hide_index=True,
            key=f"{_form_key}_dieta_real_editor",
            width="stretch",
        )
        if _ingredientes_precargados and _ingredientes_precargados[0]["Ingrediente"]:
            _st.caption(
                "_💡 La tabla está pre-cargada con los ingredientes "
                "de la dieta formulada del lote. Poné los kg que el "
                "cliente realmente está usando. Si está usando algo "
                "distinto, agregá filas o cambiá el ingrediente._"
            )
        else:
            _st.caption(
                "_Dejá vacío si el cliente no te pasó este dato — "
                "no es obligatorio. Si lo cargás, el análisis IA del "
                "lote va a sumar una sección de comparación._"
            )

        # ── Sección 6: notas libres + acciones ──
        _st.markdown(
            "##### 6. Observaciones y acciones acordadas"
        )
        _obs = _st.text_area(
            "Observaciones del asesor (texto libre)",
            placeholder=(
                "Cualquier dato adicional que no entró en el "
                "formulario: comentarios del cliente, cosas que "
                "viste, etc."
            ),
            key=f"{_form_key}_obs",
            height=80,
        )
        _acciones = _st.text_area(
            "Acciones acordadas con el cliente",
            placeholder=(
                "• HMS: ej. enviar nueva fórmula / coordinar "
                "entrega\n"
                "• Cliente: ej. mandar foto del rollo / revisar "
                "agua mañana"
            ),
            key=f"{_form_key}_acciones",
            height=80,
        )

        # ── Próximo contacto ──
        _cprx1, _cprx2 = _st.columns([1, 2])
        _prog_prox = _cprx1.checkbox(
            "Programar próximo contacto",
            key=f"{_form_key}_prog_prox",
        )
        from datetime import timedelta as _td_form
        _prox_default = (
            datetime.now().date() + _td_form(days=7)
        )
        _prox_fecha = _cprx2.date_input(
            "Fecha próximo contacto",
            value=_prox_default,
            key=f"{_form_key}_prox_fecha",
            disabled=not _prog_prox,
        )

        # ── Botones ──
        _b1, _b2 = _st.columns(2)
        _ok = _b1.form_submit_button(
            "✅ Guardar evaluación y ver acciones sugeridas",
            type="primary", width="stretch",
        )
        _cancel = _b2.form_submit_button(
            "✖ Cancelar", width="stretch",
        )

        if _cancel:
            st.session_state[on_close_state] = False
            st.rerun()

        if _ok:
            # Armar el objeto de respuestas
            _silo_pct = -1
            try:
                # Extraer número de "100% — lleno"
                _silo_pct = int(
                    _silo_nivel.split("%")[0].strip()
                )
            except Exception:
                _silo_pct = -1

            _resp = ev_mod.RespuestasEvaluacion(
                cliente_nombre=cliente_nombre,
                lote_id=_lote_id_sel,
                lote_identificador=(
                    _lote_label.split(" · ")[0]
                    if _lote_label else ""
                ),
                tipo_contacto=_tipo,
                atendio=_atendio or "",
                aspecto_animales=_aspecto,
                bajas_48hs=int(_bajas),
                causa_muerte=_causa_muerte or "",
                animales_enfermos=int(_enfermos),
                ventas_48hs=int(_ventas),
                kg_promedio_ventas=float(_kg_prom_ventas),
                detalle_movimientos=_detalle_mov or "",
                estado_comedero=_comedero,
                heces=_heces,
                estado_agua=_agua,
                estado_cama=_cama,
                estado_reparos=_reparos,
                maiz_kg_disponible=float(_maiz_kg),
                fibrogreen_kg_disponible=float(_fg_kg),
                rollos_disponibles=int(_rollos),
                silo_nivel_pct=_silo_pct,
                dias_desde_ultima_carga=int(_dias_carga),
                kg_ultima_carga=float(_kg_carga),
                # Dieta REAL que aplica el cliente (sección 5b)
                dieta_real_modo=(
                    "animal_dia"
                    if "animal" in _modo_dieta_real
                    else "total_dia"
                ),
                dieta_real_items=[
                    {
                        "nombre": str(_row["Ingrediente"]).strip(),
                        "kg": float(_row["Kg"] or 0),
                    }
                    for _, _row in _dieta_real_edit.iterrows()
                    if (
                        str(_row["Ingrediente"]).strip()
                        and float(_row["Kg"] or 0) > 0
                    )
                ],
                observaciones=_obs or "",
                acciones_acordadas=_acciones or "",
            )

            # ─── Registrar movimientos automáticamente ───
            # Si hubo muertes o ventas declaradas, las grabamos
            # ya como movimiento del lote, así las proyecciones
            # de stock, silo y dieta quedan ajustadas sin que el
            # asesor tenga que ir a 'Movimientos de hacienda'.
            _mov_msgs = []
            if _lote_id_sel and (int(_bajas) > 0
                                  or int(_ventas) > 0):
                _hoy_mov = datetime.now().date().isoformat()
                _det_base = (_detalle_mov or "").strip()
                _det_sufijo = (
                    f" (registrado desde evaluación de llamada "
                    f"con {cliente_nombre})"
                )
                try:
                    if int(_bajas) > 0:
                        # Sumar causa al detalle si la hay,
                        # así queda registrada en el lote.
                        _det_causa = (
                            f"Causa: {_causa_muerte}. "
                            if _causa_muerte else ""
                        )
                        db.crear_movimiento_lote(
                            lote_id=_lote_id_sel,
                            fecha=_hoy_mov,
                            tipo="muerte",
                            cantidad=int(_bajas),
                            detalles=(
                                _det_causa
                                + (_det_base or "Mortandad")
                                + _det_sufijo
                            ),
                        )
                        _mov_msgs.append(
                            f"💀 {int(_bajas)} baja(s) "
                            + (
                                f"({_causa_muerte}) "
                                if _causa_muerte else ""
                            )
                            + "registrada(s) en el lote"
                        )
                    if int(_ventas) > 0:
                        db.crear_movimiento_lote(
                            lote_id=_lote_id_sel,
                            fecha=_hoy_mov,
                            tipo="venta",
                            cantidad=int(_ventas),
                            kg_promedio_animal=(
                                float(_kg_prom_ventas)
                                if _kg_prom_ventas else None
                            ),
                            detalles=(
                                _det_base + _det_sufijo
                                if _det_base else
                                f"Venta / salida"
                                f"{_det_sufijo}"
                            ),
                        )
                        _mov_msgs.append(
                            f"🐄 {int(_ventas)} venta(s) "
                            "registrada(s) en el lote"
                        )
                except Exception as _e_mov:
                    st.warning(
                        f"⚠️ No pude registrar los movimientos "
                        f"automáticamente: {_e_mov}. Acordate de "
                        f"hacerlo desde 'Movimientos de hacienda' "
                        f"del lote."
                    )

            # Cargar contexto del lote para análisis cruzado
            _ctx = None
            if _lote_id_sel:
                _lote_full = db.obtener_lote(_lote_id_sel) or {}
                _dietas_lote = db.listar_dietas(_lote_id_sel) or []
                _dieta_vig = _dietas_lote[0] if _dietas_lote else None
                _ctx = dict(_lote_full)
                _ctx["dieta_vigente"] = _dieta_vig

            # Analizar (reglas determinísticas)
            _analisis = ev_mod.analizar_evaluacion(_resp, _ctx)

            # ─── Análisis ADICIONAL con agente IA (Haiku) ───
            # El motor de reglas detecta lo obvio; el agente IA
            # contextualiza con la dieta vigente y el histórico.
            # Spinner visible para feedback durante los 5-10s
            # que tarda la llamada.
            _analisis_llm = None
            try:
                # Armar resumen de últimos 2 contactos para contexto
                _ultimos = db.listar_recordatorios_cliente(
                    cliente_id, incluir_completados=True,
                )
                _ultimos_hechos = [
                    x for x in _ultimos
                    if x.get("estado") == "hecho"
                ][:2]
                _hist_str = ""
                for _h in _ultimos_hechos:
                    _hist_str += (
                        f"\n• {(_h.get('completado_en') or '')[:10]}:\n"
                        f"{(_h.get('notas_cierre','') or '')[:400]}\n"
                    )
                # Sumar contexto clínico completo (unificado con
                # análisis climático y chat conversacional)
                try:
                    if _lote_id_sel:
                        _ctx_clinico_ev = (
                            dashboard.armar_contexto_clinico_lote(
                                _lote_id_sel, db,
                            )
                        )
                        if _ctx_clinico_ev:
                            _hist_str += (
                                "\n\n" + _ctx_clinico_ev
                            )
                except Exception:
                    pass

                with st.spinner(
                    "🤖 Pidiéndole al agente IA un diagnóstico "
                    "técnico..."
                ):
                    _analisis_llm = ev_mod.analizar_con_agente_llm(
                        _resp,
                        contexto_lote=_ctx,
                        analisis_reglas=_analisis,
                        historial_resumen=_hist_str,
                        api_key=st.session_state.get(
                            "anthropic_api_key", ""
                        ),
                    )
            except Exception as _e_llm:
                _analisis_llm = {
                    "exito": False,
                    "error": str(_e_llm),
                    "analisis_md": "",
                }

            # Componer markdown final incluyendo análisis LLM
            _notas_md = ev_mod.formatear_evaluacion_md(
                _resp, _analisis,
            )
            if (_analisis_llm and _analisis_llm.get("exito")
                    and _analisis_llm.get("analisis_md")):
                _notas_md += (
                    "\n\n---\n\n**🤖 Análisis del Asesor IA:**\n\n"
                    + _analisis_llm["analisis_md"]
                )

            # Crear o cerrar el recordatorio
            _rid = recordatorio_id
            if _rid is None:
                # Conversación espontánea: crear + completar
                _rid = db.crear_recordatorio_llamada(
                    cliente_id=cliente_id,
                    fecha_objetivo=(
                        datetime.now().date().isoformat()
                    ),
                    motivo=(
                        f"Evaluación de lote "
                        f"({_tipo})"
                    ),
                    origen="manual",
                )
            # Armar JSON estructurado para la ficha clínica
            import json as _json_ev
            try:
                from dataclasses import asdict
                _ev_dict = asdict(_resp)
                _ev_dict["resumen_semaforo"] = (
                    _analisis.get("resumen_estado", "🟢")[:2]
                    if _analisis else "🟢"
                )
                _ev_dict["n_sugerencias_urgentes"] = (
                    _analisis.get("n_urgente", 0)
                    if _analisis else 0
                )
                _ev_dict["n_sugerencias_atencion"] = (
                    _analisis.get("n_atencion", 0)
                    if _analisis else 0
                )
                _ev_json = _json_ev.dumps(_ev_dict, default=str)
            except Exception:
                _ev_json = ""
            db.marcar_recordatorio_hecho(
                _rid,
                notas=_notas_md,
                evaluacion_json=_ev_json,
                lote_id=_lote_id_sel,
            )

            # Próximo contacto
            if _prog_prox:
                try:
                    db.crear_recordatorio_llamada(
                        cliente_id=cliente_id,
                        fecha_objetivo=_prox_fecha.isoformat(),
                        motivo=(
                            f"Seguimiento de evaluación del "
                            f"{datetime.now().strftime('%d/%m/%Y')}."
                            + (
                                "\nAcciones pendientes:\n"
                                + (_acciones or "")
                                if (_acciones or "").strip()
                                else ""
                            )
                        ),
                        origen="manual",
                    )
                except Exception:
                    pass

            # Guardar análisis en session_state para mostrar
            # las sugerencias al asesor recién salido del form
            st.session_state["ultimo_analisis_evaluacion"] = {
                "cliente": cliente_nombre,
                "resumen": _analisis["resumen_estado"],
                "sugerencias": [
                    {
                        "icono": ev_mod._icono_sev(s.severidad),
                        "titulo": s.titulo,
                        "detalle": s.detalle,
                        "severidad": s.severidad,
                    }
                    for s in _analisis["sugerencias"]
                ],
                "alertas_cruce": _analisis.get(
                    "alertas_cruce", [],
                ),
                "analisis_llm_md": (
                    _analisis_llm.get("analisis_md", "")
                    if (_analisis_llm
                        and _analisis_llm.get("exito"))
                    else ""
                ),
                "analisis_llm_error": (
                    _analisis_llm.get("error", "")
                    if (_analisis_llm
                        and not _analisis_llm.get("exito"))
                    else ""
                ),
            }

            st.session_state[on_close_state] = False
            _ok_msg = (
                f"✅ Evaluación guardada — "
                f"{_analisis['resumen_estado']}"
            )
            if _mov_msgs:
                _ok_msg += " · " + " · ".join(_mov_msgs)
            st.success(_ok_msg)
            st.rerun()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="HMS Nutrición Animal — Sistema integrado",
    page_icon="🐄",
    layout="wide",
)

# ── Persistencia de navegación en URL (sobrevive a F5/Cmd+R) ──
# Si la URL tiene ?lote_id=X, restaurar el modo drill-down al lote
# antes de renderizar la app. Cuando el usuario navega entre lotes,
# actualizamos los query params también — así un reload mantiene
# el contexto en vez de mandarte al inicio.
try:
    _qp = st.query_params
    _qp_lote = _qp.get("lote_id")
    if _qp_lote:
        try:
            _qp_lote_int = int(_qp_lote)
            # Solo setear si el lote sigue existiendo en la DB
            if db.obtener_lote(_qp_lote_int):
                if (st.session_state.get("lote_detalle_id")
                        != _qp_lote_int):
                    st.session_state["lote_detalle_id"] = (
                        _qp_lote_int
                    )
        except (ValueError, TypeError):
            pass
except Exception:
    # Streamlit viejo o entorno sin query params — fallback
    # silencioso, no rompe la app.
    pass

# CSS corporativo HMS
st.markdown("""
<style>
    /* Header con paleta HMS */
    h1 { color: #1B3E27; }
    h2, h3 { color: #1B3E27; }
    /* Botón primario lima */
    .stButton > button[kind="primary"] {
        background-color: #1B3E27;
        color: white;
        border: 2px solid #8BC53F;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #8BC53F;
        color: #1B3E27;
        border: 2px solid #1B3E27;
    }
    /* Tabs activos en lima */
    .stTabs [aria-selected="true"] {
        color: #1B3E27 !important;
        border-bottom: 3px solid #8BC53F !important;
    }
    /* Métricas */
    [data-testid="stMetricValue"] { color: #1B3E27; }
</style>
""", unsafe_allow_html=True)

# Header con logo si existe
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if Path("assets/logo.png").exists():
        st.image("assets/logo.png", width=120)
with col_title:
    st.markdown(
        "<h1 style='color:#1B3E27; margin-bottom:0;'>HMS Nutrición Animal</h1>"
        "<p style='color:#8BC53F; font-size:1.2em; margin-top:0; font-weight:600;'>"
        "Sistema integrado: análisis por drone + nutrición NASEM 2016 + asesor IA</p>",
        unsafe_allow_html=True,
    )

st.caption(
    "**Mauricio Suárez — Asesor Técnico Nutricional**  ·  "
    "📍 Ruta Nacional 5, km 525, Catriló, La Pampa  ·  "
    "☎ +54 2954 51-7407  ·  "
    "✉ mauricio@hmsnutricionanimal.com.ar  ·  "
    "🌐 [hmsnutricionanimal.com.ar](https://hmsnutricionanimal.com.ar)  ·  "
    "📷 [@hmsnutricionanimal](https://instagram.com/hmsnutricionanimal)"
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@st.cache_resource
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Cargando modelo YOLO…")
def load_detector(model_path: str, cow_class_id: int, conf: float, iou: float,
                  imgsz: int, modo_tropa_densa: bool = False):
    # En Streamlit Cloud (lite) no está CattleDetector porque cv2/YOLO
    # no se instalan. Devolvemos None y las pestañas del drone quedan
    # deshabilitadas con un mensaje.
    if not _DRONE_LIBS_OK or CattleDetector is None:
        return None
    return CattleDetector(
        model_path=model_path,
        cow_class_id=cow_class_id,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        modo_tropa_densa=modo_tropa_densa,
    )


cfg = load_config()

# ---------------------------------------------------------------------
# Sidebar — controles
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption(
        "Lo principal — marca, API IA y parámetros del drone solo cuando "
        "estés en las pestañas de Imagen / Video."
    )

    # ---------------------------------------------------------------
    # Configuración del módulo DRONE — solo aparece si estás procesando
    # ---------------------------------------------------------------
    with st.expander("🐄 Parámetros módulo Drone (Imagen/Video)", expanded=False):
        st.caption(
            "Estos controles se aplican cuando proceses imagen o video. "
            "Para uso normal podés dejarlos como están."
        )

        st.markdown("**Captura**")
        altura = st.number_input(
            "Altura de vuelo (m)",
            min_value=2.0, max_value=50.0,
            value=float(cfg["captura"]["altura_vuelo_m"]), step=0.5,
        )
        cfg["captura"]["altura_vuelo_m"] = altura

        st.markdown("**Referencia en piso**")
        metodo = st.selectbox(
            "Método de detección",
            ["aruco", "color_square"],
            index=0 if cfg["referencia"]["metodo"] == "aruco" else 1,
        )
        cfg["referencia"]["metodo"] = metodo
        lado = st.number_input(
            "Lado del cuadrado (m)", min_value=0.3, max_value=3.0,
            value=float(cfg["referencia"]["lado_m"]), step=0.01, format="%.2f",
        )
        cfg["referencia"]["lado_m"] = lado

        st.markdown("**Detección YOLO**")
        modo_tropa_densa = st.toggle(
            "🐄🐄🐄 Modo tropa densa",
            value=False,
            help="Activalo cuando filmes lotes apretados (animales pegados, "
                 "embudo, manga). Aumenta resolución, baja confianza, baja IoU NMS.",
        )
        if modo_tropa_densa:
            modelo_path = st.selectbox(
                "Modelo YOLO",
                ["yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
                index=1,
            )
            conf = 0.05
            iou = 0.35
            imgsz = 1920
            st.caption("Auto: conf=0.05, IoU=0.35, imgsz=1920")
        else:
            modelo_path = st.selectbox(
                "Modelo YOLO",
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt",
                 "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt"],
                index=2,
            )
            conf = st.slider("Confianza mínima", 0.05, 0.9, 0.10, 0.05)
            iou = st.slider("IoU NMS", 0.1, 0.9,
                            float(cfg["deteccion"]["iou_threshold"]), 0.05)
            imgsz = st.select_slider(
                "Tamaño inferencia",
                [640, 960, 1280, 1600, 1920], 1280,
            )

        st.markdown("**Estimación de peso**")
        raza = st.selectbox(
            "Raza predominante",
            ["angus", "hereford", "brangus", "braford", "cruza", "desconocido"],
            index=0,
        )
        try:
            _cats_drone = db.nombres_categorias()
        except Exception:
            _cats_drone = []
        if not _cats_drone:
            _cats_drone = ["ternero", "vaquillona", "novillo",
                           "vaca_adulta", "toro"]
        _cats_drone = list(_cats_drone) + ["desconocido"]
        _default_drone = ("vaquillona"
                          if "vaquillona" in _cats_drone else _cats_drone[0])
        categoria = st.selectbox(
            "Categoría / edad",
            _cats_drone,
            index=_cats_drone.index(_default_drone),
        )
        ajuste_fino = st.slider(
            "Ajuste fino de peso", 0.70, 1.30, 1.00, 0.01,
            help="Multiplicador final. Calibrá una vez con balanza real.",
        )

        with st.expander("🧮 Calculadora de ajuste fino"):
            st.caption(
                "Cargá peso real de balanza vs peso de la app y "
                "te dice dónde poner el slider."
            )
            peso_real = st.number_input(
                "Peso REAL balanza (kg)",
                min_value=50.0, max_value=1200.0, value=260.0, step=1.0,
            )
            peso_app = st.number_input(
                "Peso APP (con ajuste = 1.00, kg)",
                min_value=50.0, max_value=1200.0, value=260.0, step=1.0,
            )
            if peso_app > 0:
                sugerido = peso_real / peso_app
                error_actual = (peso_app - peso_real) / peso_real * 100
                st.metric(
                    "Ajuste sugerido", f"{sugerido:.2f}",
                    f"{error_actual:+.1f}% error",
                )
                if abs(sugerido - 1.0) < 0.03:
                    st.success("✅ Ya estás calibrado (<3% error)")
                elif 0.70 <= sugerido <= 1.30:
                    st.info(f"👉 Mové el slider a **{sugerido:.2f}**")
                else:
                    st.warning("⚠️ Fuera del rango — revisá calibración")

        use_custom_model = st.toggle("Usar modelo peso calibrado (JSON)",
                                      value=False)
        weight_json = None
        if use_custom_model:
            wm_file = st.file_uploader("Subir weight_model.json",
                                        type=["json"])
            if wm_file:
                weight_json = wm_file.read().decode("utf-8")
                st.success("Modelo cargado.")

    st.divider()
    st.subheader("🎨 Identidad HMS")
    st.caption("Marca HMS · Verde #1B3E27 · Lima #8BC53F")

    def _buscar_archivo(prefijos: list) -> Optional[Path]:
        for prefijo in prefijos:
            for ext in [".png", ".jpg", ".jpeg"]:
                p = Path(f"assets/{prefijo}{ext}")
                if p.exists():
                    return p
        return None

    # ---- Logo COLOR (para fondos claros, carátula) ----
    st.markdown("**Logo color** (fondos blancos)")
    logo_color = _buscar_archivo(["logo"])
    cols_lc = st.columns([2, 1])
    with cols_lc[0]:
        logo_color_upload = st.file_uploader(
            "Subir", type=["png", "jpg", "jpeg"],
            label_visibility="collapsed", key="logo_color_upload",
        )
    with cols_lc[1]:
        if logo_color:
            st.image(str(logo_color), width=80)
        else:
            st.caption("(sin cargar)")

    if logo_color_upload is not None:
        Path("assets").mkdir(exist_ok=True)
        ext = Path(logo_color_upload.name).suffix.lower()
        for old_ext in [".png", ".jpg", ".jpeg"]:
            old = Path(f"assets/logo{old_ext}")
            if old.exists():
                old.unlink()
        save_path = Path(f"assets/logo{ext}")
        with open(save_path, "wb") as f:
            f.write(logo_color_upload.getvalue())
        st.success(f"✅ Logo color: {save_path.name}")
        st.rerun()

    # ---- Logo BLANCO (para banda verde del header) ----
    st.markdown("**Logo blanco** (fondos oscuros / header verde)")
    logo_blanco = _buscar_archivo(["logo_blanco", "logo_white"])
    cols_lb = st.columns([2, 1])
    with cols_lb[0]:
        logo_blanco_upload = st.file_uploader(
            "Subir", type=["png", "jpg", "jpeg"],
            label_visibility="collapsed", key="logo_blanco_upload",
        )
    with cols_lb[1]:
        if logo_blanco:
            # Cuadradito con fondo verde para previsualizar logo blanco
            st.markdown(
                f'<div style="background:#1B3E27;padding:6px;border-radius:6px;'
                f'width:90px;text-align:center;">'
                f'<img src="data:image/png;base64,{__import__("base64").b64encode(open(logo_blanco,"rb").read()).decode()}" '
                f'style="max-width:80px;max-height:80px;"/>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("(sin cargar)")

    if logo_blanco_upload is not None:
        Path("assets").mkdir(exist_ok=True)
        ext = Path(logo_blanco_upload.name).suffix.lower()
        for old_ext in [".png", ".jpg", ".jpeg"]:
            old = Path(f"assets/logo_blanco{old_ext}")
            if old.exists():
                old.unlink()
        save_path = Path(f"assets/logo_blanco{ext}")
        with open(save_path, "wb") as f:
            f.write(logo_blanco_upload.getvalue())
        st.success(f"✅ Logo blanco: {save_path.name}")
        st.rerun()

    if not logo_color and not logo_blanco:
        st.warning(
            "⚠️ Sin logos cargados — los PDFs salen sin imagen. "
            "Subí al menos uno (preferentemente el blanco para el header)."
        )

    st.divider()
    st.subheader("🤖 Asesor IA")
    # La primera vez en esta sesión, cargar key persistida del disco
    if "api_key_input" not in st.session_state:
        st.session_state["api_key_input"] = cargar_api_key_persistida()

    api_key = st.text_input(
        "Claude API Key",
        type="password",
        key="api_key_input",
        help="Se guarda en data/.api_key (local, permiso 600). "
             "Persiste entre refreshes y reinicios. Para borrarla, "
             "tocá el botón.",
    )

    col_api1, col_api2 = st.columns(2)
    if col_api1.button("💾 Guardar key", width="stretch",
                        disabled=not api_key):
        guardar_api_key(api_key)
        st.success("✅ Guardada en disco")
    if col_api2.button("🗑️ Borrar key", width="stretch"):
        guardar_api_key("")
        st.session_state["api_key_input"] = ""
        st.rerun()

    if api_key:
        # Verificación visual del formato
        if api_key.startswith("sk-ant-") and len(api_key) > 80:
            st.caption(f"✅ Key cargada ({len(api_key)} chars)")
        else:
            st.warning(
                f"⚠️ Formato sospechoso ({len(api_key)} chars). "
                "Una key válida empieza con `sk-ant-` y tiene ~108 chars."
            )

    # Versión "estable" para uso en otros tabs
    st.session_state["anthropic_api_key"] = api_key

# ---------------------------------------------------------------------
# Carga de modelo
# ---------------------------------------------------------------------
detector = load_detector(
    modelo_path, cfg["deteccion"]["clase_cow_id"], conf, iou, imgsz,
    modo_tropa_densa=modo_tropa_densa,
)

if not _DRONE_LIBS_OK or WeightModel is None:
    # Modo lite (Streamlit Cloud): sin YOLO no hay estimación de peso.
    # Las pestañas del drone van a mostrar mensaje de "no disponible".
    weight_model = None
elif weight_json:
    import json
    weight_model = WeightModel(**json.loads(weight_json))
else:
    weight_model = WeightModel.from_config(cfg)

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
(tab_inicio, tab_clientes, tab_img, tab_vid, tab_evo, tab_avanzado,
 tab_ia, tab_historial, tab_config, tab_train, tab_help) = st.tabs([
    "🏠 Inicio",
    "🏢 Clientes/Lotes",
    "📷 Imagen 🐄",
    "🎞️ Video 🐄",
    "📈 Evolución 🐄",
    "🔬 Análisis 🍽️",
    "🤖 Asesor IA 🍽️",
    "📚 Historial",
    "⚙️ Configuración",
    "🎓 Entrenamiento 🐄",
    "ℹ️ Ayuda",
])

# Leyenda de los íconos en cada pestaña
st.markdown(
    "<div style='font-size:0.85em;color:#666;margin-top:-15px;'>"
    "<b>🐄 = Módulo Drone</b> (conteo y peso) &nbsp;·&nbsp; "
    "<b>🍽️ = Módulo Asesor Nutricional</b> (NASEM 2016 + IA + alertas)"
    "</div>",
    unsafe_allow_html=True,
)

# ----------------------------- INICIO ---------------------------------
with tab_inicio:
    kpis = dashboard.calcular_kpis()

    # Cabecera de bienvenida
    st.markdown(
        f"<h2 style='color:#1B3E27;margin-bottom:0;'>"
        f"Bienvenido, Mauricio 👋</h2>"
        f"<p style='color:#8BC53F;font-size:1.1em;margin-top:0;'>"
        f"Sistema integrado HMS — drone + asesor nutricional</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # KPIs principales
    st.markdown("### 📊 Resumen del rodeo bajo seguimiento")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Clientes", kpis["n_clientes"])
    k2.metric("🐄 Lotes activos", kpis["n_lotes"])
    k3.metric("🔢 Animales total", f"{kpis['n_animales_total']:,}")
    if kpis["adg_promedio"]:
        k4.metric("📈 ADG prom rodeo",
                  f"{kpis['adg_promedio']:.3f} kg/d",
                  f"de {len(kpis['lotes'])} lote(s)")
    else:
        k4.metric("📈 ADG prom rodeo", "—",
                  "Faltan pesadas comparables")

    k5, k6, k7 = st.columns(3)
    k5.metric("📅 Pesadas último mes", kpis["n_pesadas_mes"])
    k6.metric("🍽️ Dietas registradas mes", kpis["n_dietas_mes"])
    if kpis["ultima_pesada"]:
        up = kpis["ultima_pesada"]
        k7.metric(
            "🕒 Última pesada",
            up.get("fecha", ""),
            f"{up.get('_cliente','')} · {up.get('peso_promedio_kg',0):.0f} kg",
        )
    else:
        k7.metric("🕒 Última pesada", "—")

    st.divider()

    # =================================================================
    # Avisos enviados (auditoría rápida)
    # =================================================================
    # Lista cronológica de los últimos avisos que mandó el sistema a
    # cualquier cliente: alertas climáticas, alertas de stock, WhatsApp
    # críticos, informes semanales, etc. Útil para verificar al toque
    # qué se envió hoy / esta semana sin tener que ir cliente por
    # cliente.
    with st.expander(
        "📨 Avisos enviados últimos 7 días",
        expanded=False,
    ):
        st.caption(
            "Auditoría rápida de todo lo que el sistema mandó por "
            "email o WhatsApp a los clientes en la última semana. "
            "Para ver el detalle por cliente, entrá a su ficha en "
            "🏢 Clientes/Lotes."
        )
        _avisos_dash = db.listar_avisos_enviados(
            cliente_id=None, dias=7, limit=100,
        )
        if not _avisos_dash:
            st.info(
                "Sin avisos enviados en los últimos 7 días."
            )
        else:
            # Resumen arriba
            _n_email = sum(
                1 for a in _avisos_dash if a["canal"] == "email"
            )
            _n_wa = sum(
                1 for a in _avisos_dash if a["canal"] == "whatsapp"
            )
            _clientes_unicos = len({
                a.get("cliente_id") for a in _avisos_dash
                if a.get("cliente_id")
            })
            _k1, _k2, _k3 = st.columns(3)
            _k1.metric("📧 Emails", _n_email)
            _k2.metric("📱 WhatsApp", _n_wa)
            _k3.metric("👥 Clientes alcanzados", _clientes_unicos)
            # Tabla
            _ICONOS_CANAL = {"email": "📧", "whatsapp": "📱"}
            _filas_dash = []
            for av in _avisos_dash[:50]:
                _ico = _ICONOS_CANAL.get(av["canal"], "•")
                _fc = (av.get("fecha_creacion") or "")[:16]
                _est = av.get("estado") or "—"
                _est_ico = (
                    "✅" if str(_est).lower()
                    in ("enviada", "ok", "sent")
                    else ("❌" if av.get("error") else "⏳")
                )
                _filas_dash.append({
                    "Cuándo": _fc,
                    "Cliente": (av.get("cliente_nombre")
                                 or "(admin)")[:30],
                    "Canal": f"{_ico} {av['canal']}",
                    "Asunto / motivo":
                        (av.get("asunto") or "")[:70],
                    "Estado": f"{_est_ico} {_est}",
                })
            st.dataframe(
                pd.DataFrame(_filas_dash),
                hide_index=True, width="stretch",
            )
            if len(_avisos_dash) > 50:
                st.caption(
                    f"_Mostrando 50 de {len(_avisos_dash)} avisos._"
                )

    st.divider()

    # ─── Mostrar acciones sugeridas de la ÚLTIMA evaluación ───
    # Si recién guardó una evaluación, mostrar el resumen acá
    # arriba para que el asesor vea las sugerencias del sistema.
    _ultimo_eval = st.session_state.get(
        "ultimo_analisis_evaluacion"
    )
    if _ultimo_eval:
        st.markdown("### 🎯 Acciones sugeridas — última evaluación")
        st.markdown(
            f"**Cliente:** {_ultimo_eval['cliente']}  ·  "
            f"{_ultimo_eval['resumen']}"
        )

        # ── Análisis del agente IA (arriba, prominente) ──
        _llm_md = _ultimo_eval.get("analisis_llm_md", "")
        _llm_err = _ultimo_eval.get("analisis_llm_error", "")
        if _llm_md:
            st.markdown(
                "<div style='background:#f0f7ff;"
                "border-left:4px solid #2c7be5;"
                "padding:12px 16px;border-radius:6px;"
                "margin:8px 0;'>"
                "<b>🤖 Diagnóstico del Asesor IA:</b>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(_llm_md)
        elif _llm_err:
            st.warning(
                f"🤖 No se pudo obtener análisis del agente IA: "
                f"{_llm_err}"
            )

        # ── Cruces stock vs dieta ──
        if _ultimo_eval.get("alertas_cruce"):
            st.markdown("**🧮 Cruces stock vs dieta:**")
            for _alerta in _ultimo_eval["alertas_cruce"]:
                st.info(_alerta)

        # ── Sugerencias del motor de reglas ──
        if _ultimo_eval.get("sugerencias"):
            st.markdown(
                "**⚡ Alertas rápidas (motor de reglas):**"
            )
            for _sug in _ultimo_eval["sugerencias"]:
                _box_func = (
                    st.error if _sug["severidad"] == "urgente"
                    else st.warning
                    if _sug["severidad"] == "atencion"
                    else st.info
                )
                _box_func(
                    f"{_sug['icono']} **{_sug['titulo']}**  \n"
                    f"{_sug['detalle']}"
                )

        _cb1, _cb2 = st.columns([1, 4])
        if _cb1.button(
            "✖ Ocultar sugerencias",
            key="ocultar_ultimo_eval",
        ):
            del st.session_state["ultimo_analisis_evaluacion"]
            st.rerun()
        st.caption(
            "_Diagnóstico IA + alertas + cruces quedaron también "
            "guardados en el historial de llamadas del cliente._"
        )
        st.divider()

    # ─── 📞 Llamados pendientes a clientes ───
    # Bloque que junta los recordatorios manuales + sugerencias
    # automáticas del sistema (lote nuevo, sin contacto hace 30d).
    # Visible arriba para que sea lo primero que ves al abrir la
    # app — es el "ojo del negocio": chequear el lote del cliente
    # con una llamada antes de que algo se complique.
    st.markdown("### 📞 Llamados pendientes a clientes")
    try:
        # Generar sugerencias automáticas al cargar el dashboard.
        # La función ya tiene dedup interno (ventana 14-21 días),
        # así que no spamea aunque se llame en cada reload.
        _n_nuevos = db.generar_sugerencias_recordatorios()
        if _n_nuevos > 0:
            st.toast(
                f"📞 {_n_nuevos} llamado(s) sugerido(s) por el "
                "sistema. Revisalos abajo.",
                icon="📞",
            )
    except Exception:
        pass

    _recos = []
    try:
        _recos = db.listar_recordatorios_pendientes(
            dias_hasta=14, incluir_atrasados=True,
        )
    except Exception as _e:
        st.warning(f"No pude leer recordatorios: {_e}")

    # ─── Botón único: programar futuro llamado ───
    # El registro/evaluación detallada ya no se hace acá — se hace
    # desde la ficha clínica del lote (botón 📝 Registrar nueva
    # consulta). El dashboard es resumen.
    _col_btn_reco, _col_info_reco = st.columns([1, 3])
    with _col_btn_reco:
        if st.button(
            "➕ Programar futuro llamado",
            key="btn_nuevo_recordatorio",
            width="stretch",
            help=(
                "Anotar un llamado a hacer en una fecha futura. "
                "Te va a aparecer en este bloque ese día."
            ),
        ):
            st.session_state["mostrar_form_recordatorio"] = True
    with _col_info_reco:
        if _recos:
            _hoy_iso_r = datetime.now().date().isoformat()
            _atrasados = sum(
                1 for r in _recos
                if r["fecha_objetivo"] < _hoy_iso_r
            )
            _hoy_n = sum(
                1 for r in _recos
                if r["fecha_objetivo"] == _hoy_iso_r
            )
            _futuros = len(_recos) - _atrasados - _hoy_n
            _msg_partes = []
            if _atrasados:
                _msg_partes.append(
                    f"🔴 **{_atrasados}** atrasado(s)"
                )
            if _hoy_n:
                _msg_partes.append(f"🟡 **{_hoy_n}** para HOY")
            if _futuros:
                _msg_partes.append(
                    f"🟢 **{_futuros}** próximos"
                )
            st.markdown(" · ".join(_msg_partes))
        else:
            st.caption("✅ No tenés llamados pendientes.")

    # ─── Form de nuevo recordatorio ───
    if st.session_state.get("mostrar_form_recordatorio"):
        with st.form("form_recordatorio_nuevo"):
            st.markdown("##### Nuevo recordatorio de llamado")
            _clientes_reco = db.listar_clientes()
            _opciones_cli = {
                f"{c['nombre']} ({c.get('localidad','')})": c["id"]
                for c in _clientes_reco
                if (c.get("estado") or "activo") == "activo"
            }
            _col_r1, _col_r2 = st.columns([2, 1])
            _cli_sel = _col_r1.selectbox(
                "Cliente",
                options=list(_opciones_cli.keys()),
                key="reco_cliente_nuevo",
            )
            _fecha_reco = _col_r2.date_input(
                "Cuándo llamarlo",
                value=datetime.now().date(),
                key="reco_fecha_nueva",
            )
            _motivo_reco = st.text_area(
                "Motivo / qué chequear",
                placeholder=(
                    "Ej: chequear consumo del silo, ver cómo "
                    "anda la adaptación de los terneros, "
                    "coordinar próxima entrega..."
                ),
                key="reco_motivo_nuevo",
                height=80,
            )
            _c_btn1, _c_btn2 = st.columns(2)
            _ok_nuevo = _c_btn1.form_submit_button(
                "✅ Programar", type="primary", width="stretch",
            )
            _cancel_nuevo = _c_btn2.form_submit_button(
                "✖ Cancelar", width="stretch",
            )
            if _ok_nuevo and _cli_sel:
                try:
                    db.crear_recordatorio_llamada(
                        cliente_id=_opciones_cli[_cli_sel],
                        fecha_objetivo=_fecha_reco.isoformat(),
                        motivo=_motivo_reco or "",
                        origen="manual",
                    )
                    st.success(
                        f"✅ Llamado programado para "
                        f"{_fecha_reco.strftime('%d/%m/%Y')}"
                    )
                    st.session_state["mostrar_form_recordatorio"] = False
                    st.rerun()
                except Exception as _e:
                    st.error(f"Error: {_e}")
            elif _cancel_nuevo:
                st.session_state["mostrar_form_recordatorio"] = False
                st.rerun()

    # _form_registrar_ya: removido del dashboard.
    # Las conversaciones espontáneas ahora se registran desde la
    # ficha clínica del lote (botón "📝 Registrar nueva consulta").
    if False:
        pass

    # ─── Lista de recordatorios pendientes ───
    if _recos:
        _hoy_d = datetime.now().date()
        for _r in _recos:
            try:
                _fobj = datetime.strptime(
                    _r["fecha_objetivo"], "%Y-%m-%d"
                ).date()
            except Exception:
                _fobj = _hoy_d
            _delta_d = (_fobj - _hoy_d).days
            if _delta_d < 0:
                _ico_r = "🔴"
                _txt_d = f"Atrasado {abs(_delta_d)} día(s)"
            elif _delta_d == 0:
                _ico_r = "🟡"
                _txt_d = "HOY"
            elif _delta_d == 1:
                _ico_r = "🟢"
                _txt_d = "Mañana"
            else:
                _ico_r = "🟢"
                _txt_d = f"En {_delta_d} días"

            # Etiqueta de origen
            _orig = _r.get("origen", "manual")
            _ico_orig = (
                "✋ Manual"
                if _orig == "manual"
                else "🤖 Sugerido"
            )

            # ─── Lista compacta ─── un renglón por recordatorio
            # con cliente + fecha + motivo + atajos rápidos. El
            # trabajo pesado (ficha técnica, evaluación, ver historial
            # clínico) se hace desde la ficha del lote.
            _cols_r = st.columns([3, 1, 1, 1])
            with _cols_r[0]:
                _motivo_short = (
                    (_r.get("motivo", "") or "").strip()
                )
                if len(_motivo_short) > 140:
                    _motivo_short = _motivo_short[:137] + "..."
                st.markdown(
                    f"{_ico_r} **{_r['cliente_nombre']}** · "
                    f"{_fobj.strftime('%d/%m/%Y')} "
                    f"_({_txt_d})_ · {_ico_orig}"
                )
                if _motivo_short:
                    st.caption(f"📝 {_motivo_short}")
                if _r.get("cliente_localidad"):
                    st.caption(
                        f"📍 {_r['cliente_localidad']}"
                    )

            # Atajo: ir directo a la ficha del lote más reciente
            # del cliente (donde está la ficha clínica completa)
            with _cols_r[1]:
                _lotes_cli_r = []
                try:
                    _lotes_cli_r = db.listar_lotes(
                        cliente_id=_r["cliente_id"], estado="activo",
                    ) or []
                except Exception:
                    pass
                if _lotes_cli_r:
                    _lt_pri = _lotes_cli_r[0]
                    if st.button(
                        "🩺 Ir al lote",
                        key=f"go_lote_{_r['id']}",
                        width="stretch",
                        help=(
                            f"Abrir la ficha clínica del lote "
                            f"'{_lt_pri.get('identificador','')}' "
                            "donde está toda la historia + el "
                            "botón para registrar una nueva consulta."
                        ),
                    ):
                        st.session_state["lote_detalle_id"] = (
                            _lt_pri["id"]
                        )
                        st.query_params["lote_id"] = str(
                            _lt_pri["id"]
                        )
                        st.rerun()
                else:
                    st.caption("_Sin lotes_")
            with _cols_r[2]:
                if st.button(
                    "📅 +7d",
                    key=f"reco_reprog7_{_r['id']}",
                    width="stretch",
                    help="Postergar 7 días",
                ):
                    from datetime import timedelta as _td_r
                    _nf = (_fobj + _td_r(days=7)).isoformat()
                    db.reprogramar_recordatorio(_r["id"], _nf)
                    st.rerun()
            with _cols_r[3]:
                if st.button(
                    "✖ Cerrar",
                    key=f"reco_cancel_{_r['id']}",
                    width="stretch",
                    help="Cancelar este recordatorio",
                ):
                    db.cancelar_recordatorio(_r["id"])
                    st.rerun()
            st.divider()
            # Saltar el bloque expandido viejo
            if False:

                # ─── RESUMEN DEL CONTACTO ANTERIOR ───
                # Justo arriba de la ficha técnica, mostramos el
                # último llamado HECHO del mismo cliente — qué se
                # habló, qué se acordó, qué quedó pendiente. Si
                # tiene JSON estructurado lo resumimos en bullets;
                # si no, mostramos el markdown libre.
                try:
                    _prev_recos = db.listar_recordatorios_cliente(
                        _r["cliente_id"],
                        incluir_completados=True,
                    )
                    _prev_hecho = next(
                        (x for x in _prev_recos
                         if x.get("estado") == "hecho"
                         and x.get("id") != _r["id"]),
                        None,
                    )
                except Exception:
                    _prev_hecho = None

                if _prev_hecho:
                    _f_prev = (
                        (_prev_hecho.get("completado_en") or "")[:10]
                        or _prev_hecho.get("fecha_objetivo", "—")
                    )
                    # Calcular cuántos días pasaron
                    try:
                        _f_prev_d = datetime.strptime(
                            _f_prev, "%Y-%m-%d"
                        ).date()
                        _dias_desde = (
                            datetime.now().date() - _f_prev_d
                        ).days
                        _txt_dias = (
                            f" · hace **{_dias_desde}** día(s)"
                            if _dias_desde > 0
                            else " · hoy mismo"
                        )
                    except Exception:
                        _txt_dias = ""

                    # Resumen visual destacado
                    st.markdown(
                        f"<div style='background:#fff8e6;"
                        f"border-left:4px solid #f0ad4e;"
                        f"padding:10px 14px;border-radius:6px;"
                        f"margin:8px 0;'>"
                        f"<b>📋 Último contacto: {_f_prev}</b>"
                        f"{_txt_dias}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Si hay JSON estructurado, sacamos bullets
                    # útiles. Si no, mostramos el markdown crudo.
                    _ev_json_raw = (
                        _prev_hecho.get("evaluacion_json") or ""
                    )
                    _ev_struct = {}
                    if _ev_json_raw:
                        try:
                            import json as _json_prev
                            _ev_struct = _json_prev.loads(
                                _ev_json_raw
                            )
                        except Exception:
                            _ev_struct = {}

                    if _ev_struct:
                        # Resumen estructurado
                        _bullets = []
                        if _ev_struct.get("tipo_contacto"):
                            _bullets.append(
                                f"**Tipo:** "
                                f"{_ev_struct['tipo_contacto']}"
                                + (
                                    f" con "
                                    f"_{_ev_struct.get('atendio','')}_"
                                    if _ev_struct.get('atendio')
                                    else ""
                                )
                            )
                        if _ev_struct.get("resumen_semaforo"):
                            _bullets.append(
                                f"**Estado clínico de esa "
                                f"consulta:** "
                                f"{_ev_struct['resumen_semaforo']}"
                            )
                        if int(_ev_struct.get("bajas_48hs") or 0) > 0:
                            _bullets.append(
                                f"💀 **{_ev_struct['bajas_48hs']} "
                                "muerte(s)** registrada(s)"
                                + (
                                    f" — "
                                    f"{_ev_struct.get('causa_muerte','')}"
                                    if _ev_struct.get('causa_muerte')
                                    else ""
                                )
                            )
                        if int(_ev_struct.get("animales_enfermos")
                               or 0) > 0:
                            _bullets.append(
                                f"🤒 **"
                                f"{_ev_struct['animales_enfermos']}"
                                " enfermo(s)** en seguimiento"
                            )
                        if (
                            _ev_struct.get("acciones_acordadas")
                            and _ev_struct["acciones_acordadas"]
                            .strip()
                        ):
                            _bullets.append(
                                "✅ **Acciones acordadas la "
                                "vez pasada:**\n"
                                + _ev_struct["acciones_acordadas"]
                            )
                        if (_ev_struct.get("observaciones")
                                and _ev_struct["observaciones"]
                                .strip()):
                            _obs_prev = _ev_struct[
                                "observaciones"
                            ][:300]
                            _bullets.append(
                                f"📝 **Observaciones:** {_obs_prev}"
                            )
                        if _bullets:
                            for _b in _bullets:
                                st.markdown(f"- {_b}")
                        else:
                            st.caption(
                                "_(Sin datos relevantes para "
                                "destacar.)_"
                            )

                        # Botón para ver TODO el detalle si lo
                        # necesita
                        with st.expander(
                            "📄 Ver detalle completo del contacto "
                            "anterior",
                        ):
                            st.markdown(
                                _prev_hecho.get(
                                    "notas_cierre", "—",
                                ) or "—"
                            )
                    else:
                        # Markdown crudo del cierre
                        _notas_prev = (
                            _prev_hecho.get("notas_cierre", "")
                            or ""
                        )
                        if _notas_prev.strip():
                            st.markdown(_notas_prev[:1500])
                            if len(_notas_prev) > 1500:
                                st.caption("_(truncado)_")
                        else:
                            st.caption(
                                "_(El contacto anterior no quedó "
                                "registrado con detalle.)_"
                            )
                else:
                    st.caption(
                        "🆕 Es la primera vez que llamás a este "
                        "cliente — sin contactos previos registrados."
                    )

                # ─── FICHA TÉCNICA DE REVISIÓN ───
                # Snapshot del cliente armado en tiempo real para
                # que tengas todo a mano antes de marcar.
                with st.expander(
                    "📋 Ficha técnica para la llamada — "
                    "lotes, dieta, stock, alertas",
                    expanded=False,
                ):
                    try:
                        _ficha = db.armar_ficha_revision_cliente(
                            _r["cliente_id"]
                        )
                    except Exception as _eficha:
                        _ficha = None
                        st.warning(
                            f"No pude armar la ficha: {_eficha}"
                        )

                    if _ficha:
                        # Cabecera cliente
                        _cli_f = _ficha.get("cliente", {})
                        _cts_f = _cli_f.get("contactos", []) or []
                        st.markdown(
                            f"**{_cli_f.get('nombre','')}** · "
                            f"📍 {_cli_f.get('localidad','—')}"
                        )
                        if _cts_f:
                            _lineas_cts = []
                            for _ct in _cts_f[:3]:
                                _nm = _ct.get('nombre') or '—'
                                _wa = _ct.get('whatsapp') or ''
                                _em = _ct.get('email') or ''
                                _parts = [f"👤 {_nm}"]
                                if _wa:
                                    _parts.append(f"📱 {_wa}")
                                if _em:
                                    _parts.append(f"✉️ {_em}")
                                _lineas_cts.append(
                                    " · ".join(_parts)
                                )
                            st.markdown(
                                "<br>".join(_lineas_cts),
                                unsafe_allow_html=True,
                            )

                        # Último contacto
                        _uc = _ficha.get("ultimo_contacto")
                        if _uc:
                            _f_uc = (
                                (_uc.get('completado_en') or '')[:10]
                                or _uc.get('fecha_objetivo', '—')
                            )
                            st.caption(
                                f"📅 Último llamado completado: "
                                f"**{_f_uc}**"
                            )
                            _notas_prev = _uc.get(
                                "notas_cierre", ""
                            ) or ""
                            if _notas_prev:
                                with st.expander(
                                    "Ver notas del último llamado"
                                ):
                                    st.markdown(_notas_prev)
                        else:
                            st.caption(
                                "📅 Sin llamados previos registrados"
                            )

                        # Lotes activos
                        _lotes_f = _ficha.get("lotes", []) or []
                        if _lotes_f:
                            st.markdown("##### 🐂 Lotes activos")
                            for _lt_f in _lotes_f:
                                _bullet = (
                                    f"**{_lt_f['identificador']}** · "
                                    f"{_lt_f.get('categoria','—')} "
                                    f"{_lt_f.get('raza','')} · "
                                    f"{_lt_f.get('cantidad','—')} cab · "
                                    f"{_lt_f['dias']}d en sistema · "
                                    f"PV ingreso "
                                    f"{_lt_f['pv_ingreso_kg']:.0f} kg "
                                    f"→ HOY "
                                    f"**{_lt_f['pv_hoy_kg']:.0f} kg** "
                                    f"(ADG {_lt_f['adg_obj']:.2f})"
                                )
                                st.markdown(f"- {_bullet}")
                                _d_v = _lt_f.get("dieta_vigente")
                                if _d_v:
                                    _obs_v = (
                                        _d_v.get('observaciones', '')
                                        or ''
                                    )
                                    # KPIs nutricionales
                                    st.markdown(
                                        f"  - 🍽️ **Dieta vigente** "
                                        f"({_d_v.get('fecha','—')[:10]}): "
                                        f"PB {_d_v.get('pb_pct',0):.1f}% · "
                                        f"DMI "
                                        f"{_d_v.get('consumo_ms_kg',0):.2f} "
                                        f"kg MS/día · "
                                        f"EM {_d_v.get('em_mcal_dia',0):.1f} "
                                        f"Mcal/día"
                                    )
                                    if obs := _obs_v.strip():
                                        st.markdown(
                                            f"  - _{obs[:200]}_"
                                        )
                                    # Composición ingrediente por ingrediente
                                    _comp_v = _d_v.get(
                                        "composicion", []
                                    ) or []
                                    if _comp_v:
                                        _cant_lote_ff = int(
                                            _lt_f.get("cantidad", 0)
                                            or 0
                                        )
                                        _filas_form = []
                                        _total_kg_an = 0.0
                                        _total_pct = 0.0
                                        for _ing in _comp_v:
                                            _kg_tc_ing = float(
                                                _ing.get(
                                                    "kg_tal_cual", 0,
                                                ) or 0
                                            )
                                            _pct_r_ing = float(
                                                _ing.get(
                                                    "pct_racion", 0,
                                                ) or 0
                                            )
                                            _total_kg_an += _kg_tc_ing
                                            _total_pct += _pct_r_ing
                                            _filas_form.append({
                                                "Ingrediente": (
                                                    _ing.get(
                                                        "nombre", "?",
                                                    )
                                                ),
                                                "% ración": (
                                                    f"{_pct_r_ing:.1f}%"
                                                ),
                                                "kg/an/día": round(
                                                    _kg_tc_ing, 2,
                                                ),
                                                "kg/lote/día": round(
                                                    _kg_tc_ing
                                                    * _cant_lote_ff, 1,
                                                ),
                                            })
                                        # Fila total
                                        _filas_form.append({
                                            "Ingrediente": "TOTAL",
                                            "% ración": (
                                                f"{_total_pct:.0f}%"
                                            ),
                                            "kg/an/día": round(
                                                _total_kg_an, 2,
                                            ),
                                            "kg/lote/día": round(
                                                _total_kg_an
                                                * _cant_lote_ff, 1,
                                            ),
                                        })
                                        _comp_from = (
                                            _d_v.get(
                                                "composicion_origen_fecha",
                                                ""
                                            ) or ""
                                        )
                                        if _comp_from:
                                            st.markdown(
                                                f"  - **📋 Fórmula** "
                                                f"({_cant_lote_ff} "
                                                f"animales) — "
                                                f"_composición tomada "
                                                f"de la dieta del "
                                                f"{_comp_from[:10]} "
                                                f"(la vigente solo "
                                                f"tenía KPIs)_:"
                                            )
                                        else:
                                            st.markdown(
                                                f"  - **📋 Fórmula vigente** "
                                                f"({_cant_lote_ff} animales):"
                                            )
                                        st.dataframe(
                                            pd.DataFrame(_filas_form),
                                            hide_index=True,
                                            width="stretch",
                                        )
                                    else:
                                        st.markdown(
                                            "  - ⚠️ Esta dieta no "
                                            "tiene composición "
                                            "detallada cargada — "
                                            "solo se guardaron los "
                                            "KPIs globales. Para ver "
                                            "ingredientes, hay que "
                                            "re-formularla con el "
                                            "Asesor IA."
                                        )
                                else:
                                    st.markdown(
                                        "  - ⚠️ Sin dieta cargada "
                                        "en el historial del lote"
                                    )

                        # Alertas recientes
                        _als_f = _ficha.get(
                            "alertas_recientes", []
                        ) or []
                        if _als_f:
                            st.markdown("##### 🌦️ Alertas recientes")
                            for _a_f in _als_f:
                                st.markdown(
                                    f"- {_a_f.get('fecha','—')[:10]} "
                                    f"· {_a_f.get('tipo','—')} "
                                    f"· {(_a_f.get('asunto','') or '')[:60]}"
                                )

                        # Puntos a chequear sugeridos
                        _pts_f = _ficha.get("puntos_chequeo", []) or []
                        if _pts_f:
                            st.markdown(
                                "##### ✅ Puntos sugeridos a chequear"
                            )
                            for _p in _pts_f:
                                st.markdown(f"- {_p}")

                # Botones de acción
                _bcols = st.columns([2, 1, 1, 1])
                if _bcols[0].button(
                    "📝 Registrar lo conversado",
                    key=f"reco_hecho_{_r['id']}",
                    width="stretch",
                    type="primary",
                    help=(
                        "Anotar el resultado de la llamada / "
                        "WhatsApp con el cliente. Marca este "
                        "recordatorio como completado."
                    ),
                ):
                    st.session_state[
                        f"reco_cerrar_{_r['id']}"
                    ] = True
                    st.rerun()

                if _bcols[1].button(
                    "📅 +7 días",
                    key=f"reco_reprog7_{_r['id']}",
                    width="stretch",
                    help=(
                        "Postergar el recordatorio 7 días "
                        "(útil si quedaste sin tiempo de "
                        "llamarlo)"
                    ),
                ):
                    from datetime import timedelta as _td_r
                    _nf = (_fobj + _td_r(days=7)).isoformat()
                    db.reprogramar_recordatorio(_r["id"], _nf)
                    st.rerun()

                if _bcols[2].button(
                    "✖ Cancelar",
                    key=f"reco_cancel_{_r['id']}",
                    width="stretch",
                    help=(
                        "Descartar este recordatorio "
                        "(no es necesario llamar)"
                    ),
                ):
                    db.cancelar_recordatorio(_r["id"])
                    st.rerun()

                # ─── Form de EVALUACIÓN ESTRUCTURADA DEL LOTE ───
                # Ya no es solo "resumen libre" — es un cuestionario
                # técnico que el asesor llena durante/después de la
                # conversación. Al guardar, el sistema CRUZA las
                # respuestas con la dieta vigente del lote y SUGIERE
                # acciones concretas. Todo queda en notas_cierre como
                # markdown estructurado.
                if st.session_state.get(
                    f"reco_cerrar_{_r['id']}"
                ):
                    from src import evaluacion_lote as ev
                    _renderizar_form_evaluacion(
                        recordatorio_id=_r["id"],
                        cliente_id=_r["cliente_id"],
                        cliente_nombre=_r.get(
                            "cliente_nombre", ""
                        ),
                        ev_mod=ev,
                        on_close_state=f"reco_cerrar_{_r['id']}",
                    )

    st.divider()

    # Alertas climáticas globales — detalladas por localidad
    st.markdown("### 🌦️ Clima y alertas por localidad")
    with st.spinner("Consultando clima de tus clientes..."):
        try:
            datos_clima = dashboard.obtener_alertas_clima_globales()
        except Exception as e:
            datos_clima = {"consultadas": [], "sin_localidad": [],
                           "n_total_clientes": 0, "n_con_alertas": 0,
                           "error": str(e)}

    n_consultadas = len(datos_clima["consultadas"])
    n_sin_loc = len(datos_clima["sin_localidad"])
    n_con_alertas = datos_clima["n_con_alertas"]

    if n_consultadas == 0 and n_sin_loc == 0:
        st.info(
            "No hay clientes cargados aún. Cargá clientes en "
            "**🏢 Clientes/Lotes** y poneles localidad para que el sistema "
            "consulte el clima automáticamente."
        )
    else:
        # Resumen rápido
        st.caption(
            f"Consulté **{n_consultadas} localidad(es)** · "
            f"**{n_con_alertas} con alertas** · "
            f"**{n_sin_loc} cliente(s) sin localidad cargada**"
        )

        if n_con_alertas == 0 and n_consultadas > 0:
            st.success(
                f"✅ Sin alertas climáticas críticas en las "
                f"{n_consultadas} localidad(es) consultada(s) "
                f"en los próximos 7 días."
            )

        # Detalle por localidad
        for info in datos_clima["consultadas"]:
            estado = info["estado"]
            cliente = info["cliente"]
            loc = info["localidad"]

            if estado == "sin_geocodificar":
                with st.warning(
                    f"⚠️ **{cliente}** — '{loc}': no pude geocodificar la "
                    "localidad. Probá con un nombre más específico "
                    "(ej. 'La Carlota, Córdoba')."
                ):
                    pass
                continue

            if estado in ("sin_clima", "error"):
                msj = info.get("error", "Sin datos disponibles")
                st.warning(
                    f"⚠️ **{cliente}** — {loc}: {msj}"
                )
                continue

            # Estado OK — mostrar info climática
            n_crit = info["n_alertas_criticas"]
            n_warn = info["n_alertas_warning"]

            # Severidad REAL máxima (HOY + futuro), considerando
            # viento + lluvia + HR + acumulación. Esto puede subir el
            # nivel del semáforo más allá de lo que dice solo el THI
            # clásico — por ej. mañana fría con viento y lluvia que
            # el THI marca como "sin estrés" pero el bovino siente.
            _sev_real_max = info.get(
                "severidad_real_max", "🟢 Sin estrés"
            )
            _sev_rank_real = info.get("severidad_real_max_rank", 1)

            # Combinar: el color/ícono del título es el peor entre
            # las alertas predictivas y la severidad real.
            if n_crit > 0 or _sev_rank_real >= 4:
                titulo_color = "🔴"
                box_func = st.error
            elif n_warn > 0 or _sev_rank_real == 3:
                titulo_color = "🟠"
                box_func = st.warning
            elif _sev_rank_real == 2:
                titulo_color = "🟡"
                box_func = st.warning
            else:
                titulo_color = "🟢"
                box_func = st.success

            # Etiqueta del título: mostrar HOY explícito y, si el
            # peor día de la semana es PEOR que hoy, agregar
            # "→ peor DD/MM: ..." para que se entienda por qué el
            # semáforo del título no coincide con el de hoy.
            _sev_real_hoy = info.get(
                "severidad_real_hoy", "🟢 Sin estrés"
            )
            _sev_rank_hoy = _sev_rank_real if _sev_real_hoy == _sev_real_max else None
            _fecha_peor = info.get("severidad_real_max_fecha")
            _tramo_peor = info.get("severidad_real_max_tramo")

            # Helper rápido para sacar el rank de la sev de hoy
            _rk = {"🔴": 4, "🟠": 3, "🟡": 2, "🟢": 1}
            _rank_hoy = _rk.get(
                (_sev_real_hoy[0] if _sev_real_hoy else "🟢"), 1
            )

            _label_estado = f"HOY {_sev_real_hoy}"
            # Si el peor de la semana es distinto/peor que hoy,
            # explicitar el día.
            if _sev_rank_real > _rank_hoy and _fecha_peor:
                # Formato dd/mm para que se lea rápido
                try:
                    _fp = datetime.strptime(
                        _fecha_peor, "%Y-%m-%d"
                    ).strftime("%d/%m")
                except Exception:
                    _fp = _fecha_peor
                _label_estado += (
                    f" → peor {_fp}: {_sev_real_max}"
                )

            # Si el COLOR del título es más severo que la
            # severidad climática real (es decir, lo dispararon
            # las alertas predictivas por categoría/lotes), agregar
            # al título un sufijo que aclare la causa. Así no
            # parece arbitrario que el círculo sea naranja cuando
            # el clima de HOY dice atención (amarillo).
            _color_title_rank = _rk.get(titulo_color, 1)
            _sufijo_lotes = ""
            if _color_title_rank > _sev_rank_real and (
                n_crit > 0 or n_warn > 0
            ):
                # Detectar categorías afectadas para mostrar
                _cats_afectadas = []
                for _g in info.get("alertas_lotes", []) or []:
                    _c = _g.get("categoria", "") or ""
                    if _c and _c not in _cats_afectadas:
                        _cats_afectadas.append(_c)
                _cats_str = (
                    ", ".join(_cats_afectadas[:2])
                    if _cats_afectadas else "lotes vulnerables"
                )
                _total_alertas = n_crit + n_warn
                _sufijo_lotes = (
                    f" · 🚨 {_total_alertas} "
                    f"alerta(s) por {_cats_str}"
                )

            with st.expander(
                f"{titulo_color} **{cliente}** — {loc} · "
                f"{info['temp_c']:.0f}°C · "
                f"THI {info['thi']:.0f} · {_label_estado}"
                f"{_sufijo_lotes}",
                expanded=False,
            ):
                # Línea compacta de datos climáticos + ubicación, en
                # vez de 4 métricas grandes que ocupan mucho vertical.
                _ubic_html = ""
                if "nombre_geocode" in info:
                    _ubic_html = (
                        f" · 📍 {info['nombre_geocode']} "
                        f"<span style='color:#999;font-size:0.85em;'>"
                        f"({info['lat']:.2f}, {info['lon']:.2f})</span>"
                    )
                st.markdown(
                    f"<div style='font-size:0.95em;color:#444;"
                    f"margin-bottom:8px;'>"
                    f"🌡️ <b>{info['temp_c']:.0f}°C</b> · "
                    f"💧 <b>{info['humedad_pct']:.0f}%</b> · "
                    f"📊 THI <b>{info['thi']:.0f}</b> "
                    f"({info['thi_estado']})"
                    f"{_ubic_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Pronóstico de 7 días — siempre visible para tener
                # el clima por delante en cada cliente.
                _pron = info.get("pronostico_7d") or []
                if _pron:
                    _filas_pron = []
                    for d in _pron:
                        _thi_d = d.get("thi")
                        _estado_d = d.get("thi_estado", "—") or "—"
                        _filas_pron.append({
                            "Tramo": d.get("tramo", "—"),
                            "Fecha": d.get("fecha", "—"),
                            "T° min": (
                                f"{d['t_min']:.0f}°C"
                                if d.get("t_min") is not None
                                else "—"
                            ),
                            "T° máx": (
                                f"{d['t_max']:.0f}°C"
                                if d.get("t_max") is not None
                                else "—"
                            ),
                            "HR": (
                                f"{d['hr_media']:.0f}%"
                                if d.get("hr_media") is not None
                                else "—"
                            ),
                            "Lluvia": (
                                f"{d['precipitacion_mm']:.1f} mm"
                                if d.get("precipitacion_mm") not in
                                (None, 0) else (
                                    "—"
                                    if d.get("precipitacion_mm") is None
                                    else "0 mm"
                                )
                            ),
                            "Viento máx": (
                                f"{d['viento_max_kmh']:.0f} km/h"
                                if d.get("viento_max_kmh") is not None
                                else "—"
                            ),
                            "Cielo": (
                                # Convertir % nubes a un icono
                                # rápido de leer
                                (
                                    "☀️ Despejado"
                                    if (d.get("nubes_pct") or 0) < 30
                                    else (
                                        "⛅ Parcial"
                                        if (d.get("nubes_pct") or 0) < 70
                                        else "☁️ Cubierto"
                                    )
                                ) + f" ({d['nubes_pct']:.0f}%)"
                                if d.get("nubes_pct") is not None
                                else "—"
                            ),
                            "THI": (
                                f"{_thi_d:.0f}"
                                if _thi_d is not None else "—"
                            ),
                            "Estado (THI)": _estado_d,
                            "Severidad real": d.get(
                                "severidad_real", "—",
                            ),
                        })
                    st.markdown(
                        "<div style='font-size:0.85em;color:#666;"
                        "margin-top:6px;margin-bottom:4px;'>"
                        "<b>Clima 14 días</b> (7 pasados + HOY "
                        "+ 7 futuros) · "
                        "<i>THI = índice clásico (T°máx + HR media). "
                        "Severidad real = THI ajustado por viento "
                        "(Mader 2006) + frío con wind chill bovino "
                        "+ barro por lluvia + falta de secado por "
                        "cielo cubierto sostenido.</i>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        pd.DataFrame(_filas_pron),
                        hide_index=True, width="stretch",
                    )

                if not info["alertas_lotes"]:
                    st.caption(
                        f"✅ Sin alertas para los lotes activos en "
                        f"los próximos 7 días."
                    )
                else:
                    for grupo in info["alertas_lotes"]:
                        # Una sola fila por lote con todas las alertas
                        # concatenadas. Mucho más compacto que un
                        # st.error/warning grande por cada alerta.
                        _items = []
                        _sev_max = "warning"
                        for a in grupo["alertas"]:
                            if a.get("severidad") == "critica":
                                _sev_max = "critica"
                            _items.append(
                                f"{a['icono']} <b>{a['titulo']}</b> "
                                f"<span style='color:#666;'>"
                                f"({a['cuando']})</span>"
                            )
                        _bg = ("#FBE9E9" if _sev_max == "critica"
                               else "#FFF7E0")
                        _bd = ("#E24B4A" if _sev_max == "critica"
                               else "#EF9F27")
                        st.markdown(
                            f"<div style='background:{_bg};"
                            f"border-left:3px solid {_bd};"
                            f"padding:6px 10px;margin-bottom:4px;"
                            f"font-size:0.9em;border-radius:4px;'>"
                            f"<b>{grupo['lote']}</b> "
                            f"<span style='color:#888;'>"
                            f"({grupo['categoria']})</span> — "
                            + " · ".join(_items)
                            + "</div>",
                            unsafe_allow_html=True,
                        )

        if datos_clima["sin_localidad"]:
            st.caption(
                f"💡 Clientes sin localidad cargada (no se consulta clima): "
                f"{', '.join(datos_clima['sin_localidad'])}. "
                "Cargá la localidad en **🏢 Clientes/Lotes** para que el "
                "sistema consulte el clima de ellos también."
            )

    # Lotes que necesitan atención
    col_la, col_lb = st.columns(2)
    with col_la:
        st.markdown("### ⚠️ Lotes a pesar")
        if not kpis["lotes_a_pesar"]:
            st.success("Todos los lotes tienen pesada reciente ✅")
        else:
            for l in kpis["lotes_a_pesar"][:8]:
                st.warning(
                    f"**{l['cliente_nombre']}** — {l['identificador']}  \n"
                    f"_{l['razon']}_"
                )

    with col_lb:
        st.markdown("### 🎯 Cerca del objetivo")
        if not kpis["lotes_cerca_objetivo"]:
            st.info("Sin lotes próximos al peso objetivo")
        else:
            for l in kpis["lotes_cerca_objetivo"][:8]:
                st.success(
                    f"**{l['cliente_nombre']}** — {l['identificador']}  \n"
                    f"_{l['ultimo_peso_kg']:.0f} / "
                    f"{l['objetivo_peso_kg']:.0f} kg "
                    f"({l['ratio_objetivo']*100:.0f}%) — "
                    f"faltan {l['dif_kg']:.0f} kg_"
                )

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # LOGÍSTICA — entregas próximas a vencer y stock por agotar
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 📦 Logística — stock y próximas entregas")

    # Atajo: registrar entrega directamente desde el dashboard
    with st.expander(
        "➕ Registrar entrega rápida (atajo desde dashboard)",
        expanded=False,
    ):
        st.caption(
            "Cargá una entrega de producto sin tener que ir a la "
            "ficha del cliente. El stock se actualiza automáticamente "
            "en la tabla de abajo."
        )
        _clis_dash = [
            c for c in db.listar_clientes()
            if c.get("estado", "activo") == "activo"
        ]
        if not _clis_dash:
            st.info("No hay clientes activos cargados todavía.")
        else:
            # IMPORTANTE: sacamos esto de st.form porque dentro de un
            # form los widgets NO son reactivos — al cambiar el cliente,
            # el dropdown de lotes no se actualiza hasta hacer submit
            # (y entonces ya intentó crear la entrega con el lote
            # equivocado). Con widgets sueltos, Streamlit rerendea con
            # cada cambio y filtra los lotes del cliente seleccionado.
            _opc_clis = {
                c["nombre"]: c["id"] for c in _clis_dash
            }
            _col_q1, _col_q2 = st.columns(2)
            with _col_q1:
                _cli_q = st.selectbox(
                    "Cliente",
                    list(_opc_clis.keys()),
                    key="dash_ent_cli",
                )
                _cli_id_q = _opc_clis[_cli_q]
                _lotes_q = db.listar_lotes(
                    cliente_id=_cli_id_q, estado="activo",
                )
                _opc_lotes_q = {
                    f"{l['identificador']} ({l.get('categoria','')})":
                    l["id"]
                    for l in _lotes_q
                }
                if not _opc_lotes_q:
                    st.warning(
                        "Este cliente no tiene lotes activos."
                    )
                    _lote_id_q = None
                    _lote_sel_q = "—"
                else:
                    # Forzamos un key dependiente del cliente, así
                    # cuando cambiás de cliente el selectbox se resetea
                    # al primer lote del cliente nuevo (no queda el
                    # índice del cliente anterior).
                    _lote_sel_q = st.selectbox(
                        "Lote", list(_opc_lotes_q.keys()),
                        key=f"dash_ent_lote_{_cli_id_q}",
                    )
                    _lote_id_q = _opc_lotes_q[_lote_sel_q]
            with _col_q2:
                # Si el lote tiene dieta, sugerir productos
                _sug_productos = []
                if _lote_id_q:
                    try:
                        from src.stock_producto import (
                            listar_productos_lote as _list_prod_q,
                        )
                        _sug_productos = _list_prod_q(_lote_id_q)
                    except Exception:
                        _sug_productos = []
                _producto_q = st.text_input(
                    "Producto",
                    placeholder=(
                        "Fibroter / Fibrogreen / otro"
                        if not _sug_productos
                        else (
                            "Productos en dieta: "
                            + ", ".join(_sug_productos[:3])
                        )
                    ),
                    key="dash_ent_prod",
                )
                _fecha_q = st.date_input(
                    "Fecha de entrega",
                    value=datetime.now().date(),
                    key="dash_ent_fecha",
                )

            _col_q3, _col_q4, _col_q5 = st.columns(3)
            with _col_q3:
                _formato_q = st.selectbox(
                    "Formato", ["bolsa", "granel"],
                    key="dash_ent_fmt",
                )
            with _col_q4:
                if _formato_q == "bolsa":
                    _bolsas_q = st.number_input(
                        "Cantidad de bolsas",
                        min_value=0.0, step=1.0, value=0.0,
                        key="dash_ent_bolsas",
                    )
                    _kg_bolsa_q = st.number_input(
                        "Kg por bolsa", min_value=0.0,
                        step=1.0, value=30.0,
                        key="dash_ent_kgb",
                    )
                    _kg_total_q = _bolsas_q * _kg_bolsa_q
                else:
                    _kg_total_q = st.number_input(
                        "Kg granel", min_value=0.0,
                        step=10.0, value=0.0,
                        key="dash_ent_kgg",
                    )
                    _bolsas_q = 0
                    _kg_bolsa_q = 0
            with _col_q5:
                # Cuando es bolsa, pedimos PRECIO POR BOLSA (es lo
                # que el asesor maneja en la cabeza). Cuando es
                # granel, precio por kg.
                if _formato_q == "bolsa" and _kg_bolsa_q > 0:
                    _precio_bolsa_q = st.number_input(
                        f"Precio por bolsa de {_kg_bolsa_q:.0f} kg",
                        min_value=0.0, step=100.0, value=0.0,
                        key="dash_ent_precio_bolsa",
                        help=(
                            "Lo que pagás por cada bolsa. El "
                            "sistema calcula el precio por kg "
                            "automáticamente."
                        ),
                    )
                    _precio_q = (
                        _precio_bolsa_q / _kg_bolsa_q
                        if _kg_bolsa_q > 0 else 0
                    )
                else:
                    _precio_q = st.number_input(
                        "Precio $/kg",
                        min_value=0.0, step=10.0, value=0.0,
                        key="dash_ent_precio",
                    )

            if _kg_total_q > 0:
                _txt_unidad = (
                    f"= ${_precio_q:,.0f}/kg"
                    if _formato_q == "bolsa" and _precio_q > 0
                    else ""
                )
                st.caption(
                    f"📦 Total: **{_kg_total_q:.0f} kg** "
                    f"({_formato_q}){_txt_unidad}"
                    + (f" · **${_precio_q * _kg_total_q:,.0f}**"
                       if _precio_q > 0 else "")
                )

            _notas_q = st.text_input(
                "Notas (opcional)",
                placeholder="Ej: remito 1234",
                key="dash_ent_notas",
            )

            if st.button(
                "📦 Registrar entrega",
                type="primary",
                key="dash_ent_submit",
            ):
                    if not _producto_q:
                        st.error("Falta el nombre del producto.")
                    elif _kg_total_q <= 0:
                        st.error("La cantidad debe ser mayor a 0.")
                    elif not _lote_id_q:
                        st.error(
                            "El cliente no tiene lotes activos. "
                            "Cargá uno primero."
                        )
                    else:
                        try:
                            db.crear_entrega(
                                cliente_id=_cli_id_q,
                                lote_id=_lote_id_q,
                                producto_nombre=_producto_q,
                                kg_total=_kg_total_q,
                                fecha_entrega=_fecha_q.isoformat(),
                                formato=_formato_q,
                                cantidad_bolsas=_bolsas_q,
                                kg_por_bolsa=_kg_bolsa_q,
                                precio_kg=_precio_q,
                                precio_total=_precio_q * _kg_total_q,
                                notas=_notas_q,
                            )
                            st.success(
                                f"✅ Entrega registrada para "
                                f"**{_cli_q}** — {_kg_total_q:.0f} kg "
                                f"de {_producto_q}. La tabla de stock "
                                f"de abajo se va a actualizar."
                            )
                            st.rerun()
                        except Exception as _e_q:
                            st.error(f"Error: {_e_q}")

    # ─────── KPIs del mes ───────
    try:
        from datetime import timedelta as _td_kpi
        _hoy_kpi = datetime.now().date()
        _inicio_mes = _hoy_kpi.replace(day=1).isoformat()
        # Mes anterior para comparación
        _ultimo_dia_mes_ant = _hoy_kpi.replace(day=1) - _td_kpi(days=1)
        _inicio_mes_ant = _ultimo_dia_mes_ant.replace(day=1).isoformat()
        _fin_mes_ant = _ultimo_dia_mes_ant.isoformat()

        _entregas_mes = db.listar_entregas_periodo(
            _inicio_mes, _hoy_kpi.isoformat(),
        )
        _entregas_mes_ant = db.listar_entregas_periodo(
            _inicio_mes_ant, _fin_mes_ant,
        )
        _kg_mes = sum(e.get("kg_total") or 0 for e in _entregas_mes)
        _fact_mes = sum(e.get("precio_total") or 0 for e in _entregas_mes)
        _kg_mes_ant = sum(
            e.get("kg_total") or 0 for e in _entregas_mes_ant
        )
        _fact_mes_ant = sum(
            e.get("precio_total") or 0 for e in _entregas_mes_ant
        )
        _clis_unicos_mes = len(
            {e.get("cliente_id") for e in _entregas_mes}
        )
        _delta_fact = ""
        if _fact_mes_ant > 0:
            _delta_pct = (_fact_mes - _fact_mes_ant) / _fact_mes_ant * 100
            _signo = "+" if _delta_pct >= 0 else ""
            _delta_fact = f"{_signo}{_delta_pct:.0f}% vs mes anterior"
    except Exception:
        _entregas_mes = []
        _kg_mes = _fact_mes = _kg_mes_ant = _fact_mes_ant = 0
        _clis_unicos_mes = 0
        _delta_fact = ""

    try:
        from src.stock_producto import (
            calcular_stock_actual, listar_productos_hms_lote,
        )
        _hoy_log = datetime.now().date()

        # ── Fast path: cachear queries DB durante el render ──
        # Sin esto, calcular_stock_actual + calcular_consumo_diario_kg
        # llaman a listar_dietas / listar_entregas_cliente cientos de
        # veces (loop día por día hasta agotamiento). Con cache de
        # request-scope, cortamos ~90% de las queries y bajamos el
        # tiempo de 30-60s a 2-5s.
        import src.database as _db_mod_ct
        import functools as _ft_ct
        if not hasattr(_db_mod_ct, "_ORIG_DASH_CACHE"):
            _db_mod_ct._ORIG_DASH_CACHE = {
                "listar_dietas": _db_mod_ct.listar_dietas,
                "listar_entregas_cliente": _db_mod_ct.listar_entregas_cliente,
                "listar_entregas_lote": _db_mod_ct.listar_entregas_lote,
                "listar_lotes": _db_mod_ct.listar_lotes,
                "listar_clientes": _db_mod_ct.listar_clientes,
                "ultima_carga_silocomedero": _db_mod_ct.ultima_carga_silocomedero,
            }
        _orig_ct = _db_mod_ct._ORIG_DASH_CACHE
        # Cache de la vida del render (se pierde al terminar).
        _rq_cache = {}

        def _mk_cached(nombre, fn):
            def _wrapper(*args, **kwargs):
                key = (nombre, args, tuple(sorted(kwargs.items())))
                if key not in _rq_cache:
                    _rq_cache[key] = fn(*args, **kwargs)
                return _rq_cache[key]
            _wrapper.__wrapped__ = fn
            return _wrapper

        # Monkey-patch: usar versión cacheada durante el bloque
        for _nom, _fn in _orig_ct.items():
            setattr(_db_mod_ct, _nom, _mk_cached(_nom, _fn))

        # ── Cache del cálculo pesado (5 min TTL) ──
        # El loop hace ~50 queries a Postgres remoto que suman 15-60s
        # por render (peor con más clientes). Cacheamos en session_state
        # con TTL amplio + botón manual para refrescar.
        import time as _t_c
        _TTL_CACHE_STOCK = 300  # 5 minutos
        _btn_col1, _btn_col2 = st.columns([6, 1])
        with _btn_col2:
            if st.button(
                "🔄 Actualizar",
                key="_btn_refresh_stock",
                help="Recalcular stock actual (por defecto se refresca cada 5 min)",
                width="stretch",
            ):
                st.session_state.pop("_dash_stock", None)
                st.session_state.pop("_dash_stock_ts", None)
                st.rerun()
        _cache_valida = (
            "_dash_stock" in st.session_state
            and _t_c.time() - st.session_state.get(
                "_dash_stock_ts", 0
            ) < _TTL_CACHE_STOCK
        )
        if not _cache_valida:
            _spinner_msg = (
                f"⏳ Calculando stock ({len(db.listar_clientes())} clientes)... "
                "Puede tardar 15-60 seg la primera vez. "
                "Después queda cacheado 5 min."
            )
            _spinner_ph = st.empty()
            _spinner_ph.info(_spinner_msg)
        else:
            _spinner_ph = None
        _c_dash = st.session_state.get("_dash_stock", {}) or {}

        _filas_log = (
            list(_c_dash.get("filas_log", []))
            if _cache_valida else []
        )
        # Para los KPIs/visuales sumamos TODO el stock vigente y la
        # próxima fecha de agotamiento, no solo lo que está en alerta.
        _stock_total_kg = (
            _c_dash.get("stock_total_kg", 0) if _cache_valida else 0
        )
        _proxima_entrega_fecha = (
            _c_dash.get("proxima_entrega_fecha")
            if _cache_valida else None
        )
        _proxima_entrega_cliente = (
            _c_dash.get("proxima_entrega_cliente", "")
            if _cache_valida else ""
        )
        _autonomia_por_cliente_lote = (
            list(_c_dash.get("autonomia", []))
            if _cache_valida else []
        )  # para las barras visuales
        # Entregas registradas a lotes que NO tienen dieta cargada —
        # no podemos calcular stock pero hay que mostrar igual que la
        # carga se hizo, para que el asesor entienda qué falta.
        _entregas_sin_dieta = (
            list(_c_dash.get("entregas_sin_dieta", []))
            if _cache_valida else []
        )

        # Si el cache está fresco, saltamos el loop entero. Cuando no,
        # iteramos y al final guardamos el resultado en session_state.
        _clientes_a_iterar = (
            [] if _cache_valida else db.listar_clientes()
        )
        for _cli_log in _clientes_a_iterar:
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
                        # Agrupar por producto
                        from collections import defaultdict as _dd
                        _por_prod_lote = _dd(
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
                    _kg_entreg = _stock_log.get(
                        "kg_entregados_total", 0)
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
                    if _es_silo_lt:
                        try:
                            from src.stock_producto import (
                                proyectar_fin_carga_silocomedero,
                                _dieta_vigente,
                                _mismo_producto,
                            )
                            # % del producto HMS en la mezcla de la
                            # dieta vigente. Lo usamos para estimar
                            # cuánto producto se va por cada carga.
                            _dietas_lt = db.listar_dietas(
                                _l_log["id"]
                            )
                            _hoy_ref = datetime.now().strftime(
                                "%Y-%m-%d"
                            )
                            _dieta_lt = (
                                _dieta_vigente(_dietas_lt, _hoy_ref)
                                if _dietas_lt else None
                            )
                            _pct_prod_mezcla = 0.0
                            if _dieta_lt:
                                _kg_tc_prod = 0.0
                                _kg_tc_total = 0.0
                                for _c_d in (
                                    _dieta_lt.get("composicion")
                                    or []
                                ):
                                    _nom_d = (
                                        _c_d.get("nombre") or ""
                                    ).strip()
                                    _kg_tc = float(
                                        _c_d.get("kg_tal_cual") or 0
                                    )
                                    # Excluir libre disposición de
                                    # la mezcla (rollo a voluntad,
                                    # etc.).
                                    try:
                                        from src.stock_producto import (
                                            _es_a_discrecion,
                                        )
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
                            _ultima_silo = (
                                db.ultima_carga_silocomedero(
                                    _l_log["id"]
                                )
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
                                import json as _json_d
                                try:
                                    _desg = _json_d.loads(
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
                                                float(_d_.get("kg")
                                                       or 0)
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
                                    _kg_rest
                                    / _kg_prod_por_carga,
                                    1,
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
                            from datetime import datetime as _dt_x
                            _f_prox = _dt_x.strptime(
                                _fecha_prox_recarga, "%Y-%m-%d"
                            ).date()
                            _hoy_dt = _dt_x.now().date()
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
                            _ult_silo_local = _ultima_silo
                            _dias_por_carga = 0
                            try:
                                _dc = float(
                                    (_ult_silo_local or {}).get(
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
                            if (_dias_hasta_prox == 0
                                    and _kg_rest > 0):
                                _dias_final = _dias
                                _fecha_agot_final = _fecha_agot
                            elif _cargas_rest < 1:
                                _dias_final = _dias_hasta_prox
                                _fecha_agot_final = (
                                    _fecha_prox_recarga
                                )
                            else:
                                _cargas_full = max(
                                    0, _cargas_rest - 1
                                )
                                _dias_extra = int(round(
                                    _cargas_full * _dias_por_carga
                                ))
                                from datetime import timedelta as _td_x
                                _dias_final = (
                                    _dias_hasta_prox + _dias_extra
                                )
                                _fecha_agot_final = (
                                    _f_prox + _td_x(days=_dias_extra)
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
                    # aplicando solo para la tabla detallada
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
                        "Se acaba":
                            _stock_log.get(
                                "fecha_agotamiento") or "—",
                        "Contacto":
                            _cli_log.get("whatsapp")
                            or _cli_log.get("contacto") or "—",
                    })

        # Guardar el resultado en session_state si acabamos de
        # calcularlo (no venía del cache).
        if not _cache_valida:
            st.session_state["_dash_stock"] = {
                "filas_log": _filas_log,
                "stock_total_kg": _stock_total_kg,
                "proxima_entrega_fecha": _proxima_entrega_fecha,
                "proxima_entrega_cliente": _proxima_entrega_cliente,
                "autonomia": _autonomia_por_cliente_lote,
                "entregas_sin_dieta": _entregas_sin_dieta,
            }
            st.session_state["_dash_stock_ts"] = _t_c.time()
            if _spinner_ph is not None:
                _spinner_ph.empty()

        # ═══════════════ BLOQUE VISUAL ═══════════════
        # KPIs del mes + barras de autonomía + cronograma.
        # Siempre se muestra (haya o no alertas) para dar contexto
        # al asesor sobre la salud general de la logística.

        # ── Fila 1: 4 KPIs grandes ──
        _kpic1, _kpic2, _kpic3, _kpic4 = st.columns(4)
        with _kpic1:
            st.metric(
                "Entregado este mes",
                f"{_kg_mes:.0f} kg",
                help=f"{len(_entregas_mes)} entregas a "
                     f"{_clis_unicos_mes} cliente(s)",
            )
        with _kpic2:
            _fact_str = (
                f"$ {_fact_mes/1_000_000:.1f}M"
                if _fact_mes >= 1_000_000 else f"$ {_fact_mes:,.0f}"
            )
            st.metric(
                "Facturado este mes",
                _fact_str,
                delta=_delta_fact if _delta_fact else None,
                delta_color="normal",
            )
        with _kpic3:
            st.metric(
                "Stock total en campo",
                f"{_stock_total_kg:.0f} kg",
                help=(
                    f"{len(_autonomia_por_cliente_lote)} "
                    f"lote(s)/producto(s) con stock vigente"
                ),
            )
        with _kpic4:
            if _proxima_entrega_fecha:
                try:
                    _f_agot = datetime.strptime(
                        _proxima_entrega_fecha, "%Y-%m-%d"
                    ).date()
                    _dias_a_prox = (_f_agot - _hoy_log).days
                    st.metric(
                        "Próxima entrega",
                        (f"en {_dias_a_prox} días"
                         if _dias_a_prox > 0 else "HOY"),
                        help=(
                            f"{_proxima_entrega_cliente} · "
                            f"{_f_agot.strftime('%d/%m')}"
                        ),
                    )
                except Exception:
                    st.metric("Próxima entrega", "—")
            else:
                st.metric(
                    "Próxima entrega", "—",
                    help="Sin entregas registradas todavía",
                )

        # ── Entregas registradas pero SIN dieta cargada ──
        # Caso típico: el asesor cargó la entrega pero todavía no
        # formuló la dieta del lote. Sin dieta no podemos calcular
        # consumo, así que la entrega "no se ve" en las barras. La
        # mostramos acá explícitamente para que el asesor sepa qué
        # falta cargar.
        if _entregas_sin_dieta:
            st.markdown(
                "##### ⚠️ Entregas registradas — falta dieta del lote"
            )
            st.caption(
                "Estas entregas están guardadas correctamente, pero "
                "el lote no tiene dieta cargada todavía. Sin dieta, "
                "el sistema no puede calcular el consumo diario ni "
                "estimar la autonomía. Formulá la dieta con el "
                "Asesor IA o desde la pestaña Análisis."
            )
            for _esd in _entregas_sin_dieta:
                _col_e1, _col_e2 = st.columns([3, 1])
                with _col_e1:
                    st.markdown(
                        f"<div style='background:rgba(239,159,39,0.08);"
                        f"border-left:3px solid #BA7517;"
                        f"padding:8px 12px; margin-bottom:6px;"
                        f"font-size:13px;'>"
                        f"<strong>{_esd['cliente']}</strong> · "
                        f"{_esd['lote']} · {_esd['producto']}<br>"
                        f"<span style='color:#5F5E5A;'>"
                        f"{_esd['kg_total']:.0f} kg entregados "
                        f"({_esd['n_entregas']} entrega"
                        f"{'s' if _esd['n_entregas'] > 1 else ''}, "
                        f"última {_esd['ultima_fecha']})"
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )
                with _col_e2:
                    st.caption(
                        "👉 Cargá la dieta del lote en Análisis o "
                        "con el agente IA."
                    )

        # ── Barras de autonomía por cliente/lote/producto ──
        if _autonomia_por_cliente_lote:
            _autonomia_por_cliente_lote.sort(
                key=lambda x: x["dias"]
            )
            with st.container():
                st.markdown(
                    "##### 📊 Autonomía de cada cliente"
                )
                # Leyenda explicativa de los umbrales y colores
                st.caption(
                    "Escala fija **0 → 60 días**. "
                    "🔴 urgente (≤7d) · 🟠 esta semana (8-14d) · "
                    "🟡 próximas 2 semanas (15-30d) · "
                    "🟢 tranquilo (>30d). Las marcas grises sobre la "
                    "barra son los umbrales de 7, 14 y 30 días."
                )

                # Escala fija para todas las barras (días)
                _ESCALA_MAX_DIAS = 60
                for _a in _autonomia_por_cliente_lote:
                    _dias_a = _a["dias"]
                    # % de la barra (capado a 100% si pasa los 60d)
                    _pct = min(100.0, max(
                        0.0, _dias_a / _ESCALA_MAX_DIAS * 100,
                    ))
                    # Color de la barra según urgencia
                    if _dias_a <= 7:
                        _emoji = "🔴"
                        _color_bar = "#E13B3B"     # rojo
                        _color_txt = "#A32D2D"
                    elif _dias_a <= 14:
                        _emoji = "🟠"
                        _color_bar = "#E89938"     # naranja
                        _color_txt = "#854F0B"
                    elif _dias_a <= 30:
                        _emoji = "🟡"
                        _color_bar = "#D9C84C"     # amarillo
                        _color_txt = "#7A6A0F"
                    else:
                        _emoji = "🟢"
                        _color_bar = "#5BAE7D"     # verde
                        _color_txt = "#0F6E56"

                    _col_lbl, _col_bar, _col_dat = st.columns(
                        [3, 4, 2]
                    )
                    with _col_lbl:
                        st.markdown(
                            f"**{_a['cliente']}** · "
                            f"{_a['lote']}"
                        )
                        st.caption(
                            f"{_emoji} {_a['producto']}"
                        )
                    with _col_bar:
                        # Barra HTML custom: marcas verticales en 7, 14, 30
                        # días y relleno con el color de urgencia.
                        # Posición % de cada marca sobre la escala 0-60.
                        _m7 = 7 / _ESCALA_MAX_DIAS * 100
                        _m14 = 14 / _ESCALA_MAX_DIAS * 100
                        _m30 = 30 / _ESCALA_MAX_DIAS * 100
                        # Label dentro o al lado de la barra según
                        # cuánto avance tenga (si <12%, fuera; si no,
                        # dentro)
                        _label_in = (
                            f"{_dias_a:.0f}d"
                            if _pct >= 18 else ""
                        )
                        _label_out = (
                            f"{_dias_a:.0f}d"
                            if _pct < 18 else ""
                        )
                        # IMPORTANTE: HTML en una sola línea, sin
                        # indentación ni comentarios. Streamlit con
                        # unsafe_allow_html=True parsea el bloque por
                        # markdown primero y la indentación >4 espacios
                        # hace que se renderice literal en barras chicas.
                        _barra_html = (
                            f'<div style="display:flex;align-items:center;'
                            f'gap:6px;margin-top:10px;">'
                            f'<div style="position:relative;flex:1;'
                            f'background:#F0F0F0;border-radius:6px;'
                            f'height:22px;overflow:hidden;">'
                            f'<div style="position:absolute;left:{_m7}%;'
                            f'top:0;bottom:0;width:1px;'
                            f'background:rgba(0,0,0,0.18);z-index:2;"></div>'
                            f'<div style="position:absolute;left:{_m14}%;'
                            f'top:0;bottom:0;width:1px;'
                            f'background:rgba(0,0,0,0.18);z-index:2;"></div>'
                            f'<div style="position:absolute;left:{_m30}%;'
                            f'top:0;bottom:0;width:1px;'
                            f'background:rgba(0,0,0,0.18);z-index:2;"></div>'
                            f'<div style="background:{_color_bar};'
                            f'height:100%;width:{_pct}%;'
                            f'border-radius:6px 0 0 6px;z-index:1;'
                            f'position:relative;display:flex;'
                            f'align-items:center;padding-left:8px;'
                            f'color:white;font-size:12px;font-weight:600;'
                            f'white-space:nowrap;">{_label_in}</div>'
                            f'</div>'
                            f'<span style="color:{_color_txt};'
                            f'font-size:12px;font-weight:600;'
                            f'min-width:30px;">{_label_out}</span>'
                            f'</div>'
                            f'<div style="display:flex;'
                            f'justify-content:space-between;'
                            f'font-size:10px;color:#999;margin-top:2px;'
                            f'padding:0 2px;">'
                            f'<span>0</span>'
                            f'<span style="margin-left:-8px;">7d</span>'
                            f'<span style="margin-left:8px;">14d</span>'
                            f'<span>30d</span>'
                            f'<span>60d+</span>'
                            f'</div>'
                        )
                        st.markdown(
                            _barra_html, unsafe_allow_html=True,
                        )
                    with _col_dat:
                        _fecha_agot_raw = _a.get("fecha_agot")
                        try:
                            _fecha_agot_show = (
                                datetime.strptime(
                                    _fecha_agot_raw, "%Y-%m-%d",
                                ).strftime("%d/%m/%y")
                                if _fecha_agot_raw else "—"
                            )
                        except Exception:
                            _fecha_agot_show = _fecha_agot_raw or "—"
                        # Render unificado: en silocomedero o lineal,
                        # mostramos "se acaba DD/MM/YY". En silo
                        # agregamos abajo un detalle gris con kg/carga
                        # y aviso si la próxima carga no llega.
                        _detalle_extra = ""
                        if (_a.get("es_silocomedero")
                                and _a.get("cargas_restantes")
                                is not None):
                            _cr = _a["cargas_restantes"]
                            _kgpc = _a.get(
                                "kg_prod_por_carga"
                            ) or 0
                            _aviso = ""
                            if _cr < 1:
                                _aviso = " · ⚠️ no alcanza la próxima"
                            _detalle_extra = (
                                f"<br><span style='color:#999;"
                                f"font-size:11px;'>"
                                f"~{_kgpc:.0f} kg/carga del silo"
                                f"{_aviso}"
                                f"</span>"
                            )
                        st.markdown(
                            f"<div style='text-align:right;"
                            f"margin-top:10px;'>"
                            f"<strong>{_a['kg_rest']:.0f} kg"
                            f"</strong>"
                            f"<br><span style='color:{_color_txt};"
                            f"font-size:13px;'>"
                            f"se acaba {_fecha_agot_show}"
                            f"</span>"
                            f"{_detalle_extra}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # ── Autonomía del silocomedero por lote ──
            # Para los lotes que están en silocomedero (mezcla cargada
            # en silo que dura varios días), proyectar cuándo se va a
            # agotar la carga ACTUAL. Es distinto del stock comercial
            # del cliente (Fibrogreen, etc.): acá nos importa cuándo
            # tiene que recargar el silo, no cuándo HMS le entrega
            # producto.
            from src.stock_producto import (
                proyectar_fin_carga_silocomedero,
            )
            _silos_dash = []
            _lotes_silos = [
                l for l in db.listar_lotes(estado="activo")
                if (l.get("tipo_comedero_concentrado") or "").lower()
                == "silocomedero"
            ]
            _clientes_idx = {
                c["id"]: c for c in db.listar_clientes()
            }
            for _lt_s in _lotes_silos:
                try:
                    _proy = proyectar_fin_carga_silocomedero(
                        _lt_s["id"]
                    )
                except Exception:
                    _proy = None
                if not _proy:
                    continue
                _cli_s = _clientes_idx.get(_lt_s["cliente_id"]) or {}
                _silos_dash.append({
                    "cliente": _cli_s.get("nombre", "?"),
                    "lote": _lt_s.get("identificador", "?"),
                    "kg_cargados": _proy["kg_cargados"],
                    "fecha_carga": _proy["fecha_carga"],
                    "consumo_dia": _proy["consumo_diario_kg"],
                    "kg_restantes": _proy["kg_restantes"],
                    "dias_restantes": _proy["dias_restantes"],
                    "fecha_agot": _proy["fecha_agotamiento"],
                })
            if _silos_dash:
                # Ordenar por urgencia (menos días primero)
                _silos_dash.sort(key=lambda x: x["dias_restantes"])
                st.markdown(
                    "##### 🛢️ Autonomía del silocomedero por lote"
                )
                st.caption(
                    "Cuándo se va a agotar la carga actual del "
                    "silocomedero, según consumo diario de la dieta "
                    "vigente × cantidad de animales. Sirve para "
                    "planificar la próxima carga. "
                    "Escala fija **0 → 60 días**. "
                    "🔴 urgente (≤2d) · 🟠 esta semana (3-5d) · "
                    "🟡 próximas 2 semanas (6-14d) · "
                    "🟢 tranquilo (>14d)."
                )

                from datetime import datetime as _dt_sil
                _ESCALA_MAX_SILO = 60
                for _s in _silos_dash:
                    _d = _s["dias_restantes"]
                    _pct_s = min(100.0, max(
                        0.0, _d / _ESCALA_MAX_SILO * 100,
                    ))
                    # Color según urgencia (silocomedero suele tener
                    # ventanas más cortas que el stock de producto)
                    if _d <= 2:
                        _emoji_s = "🔴"
                        _color_bar_s = "#E13B3B"
                        _color_txt_s = "#A32D2D"
                    elif _d <= 5:
                        _emoji_s = "🟠"
                        _color_bar_s = "#E89938"
                        _color_txt_s = "#854F0B"
                    elif _d <= 14:
                        _emoji_s = "🟡"
                        _color_bar_s = "#D9C84C"
                        _color_txt_s = "#7A6A0F"
                    else:
                        _emoji_s = "🟢"
                        _color_bar_s = "#5BAE7D"
                        _color_txt_s = "#0F6E56"

                    try:
                        _fa_show = _dt_sil.strptime(
                            _s["fecha_agot"], "%Y-%m-%d"
                        ).strftime("%d/%m/%y")
                    except Exception:
                        _fa_show = _s["fecha_agot"]
                    try:
                        _fc_show = _dt_sil.strptime(
                            _s["fecha_carga"], "%Y-%m-%d"
                        ).strftime("%d/%m/%y")
                    except Exception:
                        _fc_show = _s["fecha_carga"]

                    _col_lbl_s, _col_bar_s, _col_dat_s = st.columns(
                        [3, 4, 2]
                    )
                    with _col_lbl_s:
                        st.markdown(
                            f"**{_s['cliente']}** · {_s['lote']}"
                        )
                        st.caption(
                            f"{_emoji_s} cargado {_fc_show}: "
                            f"{_s['kg_cargados']:.0f} kg · "
                            f"consumo {_s['consumo_dia']:.0f} kg/día"
                        )
                    with _col_bar_s:
                        # Marcas verticales en 7, 14, 30 días sobre
                        # escala 0-60.
                        _m7s = 7 / _ESCALA_MAX_SILO * 100
                        _m14s = 14 / _ESCALA_MAX_SILO * 100
                        _m30s = 30 / _ESCALA_MAX_SILO * 100
                        _lab_in_s = (
                            f"{_d:.0f}d" if _pct_s >= 18 else ""
                        )
                        _lab_out_s = (
                            f"{_d:.0f}d" if _pct_s < 18 else ""
                        )
                        _barra_silo = (
                            f'<div style="display:flex;'
                            f'align-items:center;gap:6px;'
                            f'margin-top:10px;">'
                            f'<div style="position:relative;flex:1;'
                            f'background:#F0F0F0;border-radius:6px;'
                            f'height:22px;overflow:hidden;">'
                            f'<div style="position:absolute;'
                            f'left:{_m7s}%;top:0;bottom:0;width:1px;'
                            f'background:rgba(0,0,0,0.18);'
                            f'z-index:2;"></div>'
                            f'<div style="position:absolute;'
                            f'left:{_m14s}%;top:0;bottom:0;width:1px;'
                            f'background:rgba(0,0,0,0.18);'
                            f'z-index:2;"></div>'
                            f'<div style="position:absolute;'
                            f'left:{_m30s}%;top:0;bottom:0;width:1px;'
                            f'background:rgba(0,0,0,0.18);'
                            f'z-index:2;"></div>'
                            f'<div style="background:{_color_bar_s};'
                            f'height:100%;width:{_pct_s}%;'
                            f'border-radius:6px 0 0 6px;z-index:1;'
                            f'position:relative;display:flex;'
                            f'align-items:center;padding-left:8px;'
                            f'color:white;font-size:12px;'
                            f'font-weight:600;white-space:nowrap;">'
                            f'{_lab_in_s}</div>'
                            f'</div>'
                            f'<span style="color:{_color_txt_s};'
                            f'font-size:12px;font-weight:600;'
                            f'min-width:30px;">{_lab_out_s}</span>'
                            f'</div>'
                            f'<div style="display:flex;'
                            f'justify-content:space-between;'
                            f'font-size:10px;color:#999;'
                            f'margin-top:2px;padding:0 2px;">'
                            f'<span>0</span>'
                            f'<span style="margin-left:-8px;">7d</span>'
                            f'<span style="margin-left:8px;">14d</span>'
                            f'<span>30d</span>'
                            f'<span>60d+</span>'
                            f'</div>'
                        )
                        st.markdown(
                            _barra_silo, unsafe_allow_html=True,
                        )
                    with _col_dat_s:
                        st.markdown(
                            f"<div style='text-align:right;"
                            f"margin-top:10px;'>"
                            f"<strong>{_s['kg_restantes']:.0f} kg"
                            f"</strong>"
                            f"<br><span style='color:{_color_txt_s};"
                            f"font-size:13px;'>"
                            f"se agota {_fa_show}"
                            f"</span></div>",
                            unsafe_allow_html=True,
                        )
                st.divider()

            # ── Cronograma de próximas entregas + productos top ──
            _col_crono, _col_prods = st.columns(2)

            with _col_crono:
                st.markdown(
                    "##### 📅 Próximas entregas estimadas"
                )
                _items_crono = [
                    a for a in _autonomia_por_cliente_lote
                    if a.get("fecha_agot") and a["kg_rest"] > 0
                ]
                _items_crono.sort(
                    key=lambda x: x["fecha_agot"] or "9999"
                )
                if not _items_crono:
                    st.caption(
                        "Sin proyecciones de agotamiento "
                        "todavía."
                    )
                else:
                    _esta_sem = []
                    _prox_sem = []
                    _mas = []
                    for it in _items_crono[:8]:
                        try:
                            _f = datetime.strptime(
                                it["fecha_agot"], "%Y-%m-%d",
                            ).date()
                            _d = (_f - _hoy_log).days
                            if _d <= 7:
                                _esta_sem.append((it, _f, _d))
                            elif _d <= 14:
                                _prox_sem.append((it, _f, _d))
                            else:
                                _mas.append((it, _f, _d))
                        except Exception:
                            continue
                    if _esta_sem:
                        st.caption("Esta semana")
                        for it, _f, _d in _esta_sem:
                            st.markdown(
                                f"<div style='border-left:3px solid "
                                f"#E24B4A; padding:6px 10px; "
                                f"margin-bottom:6px; "
                                f"background:rgba(226,75,74,0.06);"
                                f"font-size:13px;'>"
                                f"<strong>{it['cliente']}</strong>"
                                f" · {it['producto']}"
                                f" · {_f.strftime('%a %d/%m')}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    if _prox_sem:
                        st.caption("Próxima semana")
                        for it, _f, _d in _prox_sem:
                            st.markdown(
                                f"<div style='border-left:3px solid "
                                f"#EF9F27; padding:6px 10px; "
                                f"margin-bottom:6px; "
                                f"background:rgba(239,159,39,0.06);"
                                f"font-size:13px;'>"
                                f"<strong>{it['cliente']}</strong>"
                                f" · {it['producto']}"
                                f" · {_f.strftime('%a %d/%m')}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    if _mas:
                        st.caption("Más adelante")
                        for it, _f, _d in _mas[:3]:
                            st.markdown(
                                f"<div style='border-left:3px solid "
                                f"#1D9E75; padding:6px 10px; "
                                f"margin-bottom:6px; "
                                f"background:rgba(29,158,117,0.06);"
                                f"font-size:13px;'>"
                                f"<strong>{it['cliente']}</strong>"
                                f" · {it['producto']}"
                                f" · {_f.strftime('%d/%m')}"
                                f" ({_d}d)"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

            with _col_prods:
                st.markdown(
                    "##### 📦 Producto entregado · últimos 30 días"
                )
                from collections import defaultdict
                _por_prod = defaultdict(float)
                from datetime import timedelta as _td30
                _hace30 = (_hoy_log - _td30(days=30)).isoformat()
                try:
                    _ent_30d = db.listar_entregas_periodo(
                        _hace30, _hoy_log.isoformat(),
                    )
                except Exception:
                    _ent_30d = []
                for e in _ent_30d:
                    _por_prod[
                        e.get("producto_nombre", "")
                    ] += e.get("kg_total") or 0
                if not _por_prod:
                    st.caption(
                        "Sin entregas registradas en los últimos "
                        "30 días."
                    )
                else:
                    _max_prod = max(_por_prod.values()) or 1
                    _items_prod = sorted(
                        _por_prod.items(), key=lambda x: -x[1],
                    )
                    for nom, kg in _items_prod[:6]:
                        _pct_p = kg / _max_prod * 100
                        st.markdown(
                            f"<div style='font-size:13px; "
                            f"margin-bottom:4px;'>"
                            f"<div style='display:flex; "
                            f"justify-content:space-between;'>"
                            f"<span>{nom}</span>"
                            f"<span style='color:#5F5E5A;'>"
                            f"{kg:.0f} kg</span></div>"
                            f"<div style='background:#F1EFE8; "
                            f"height:6px; border-radius:3px;"
                            f"margin-top:3px;'>"
                            f"<div style='background:#0F6E56; "
                            f"height:100%; width:{_pct_p:.0f}%;"
                            f"border-radius:3px;'></div></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        st.divider()

        # ═══════════════ BLOQUE DE ALERTAS (original) ═══════════════
        if not _filas_log:
            st.success(
                "✅ Sin alertas de stock — todos los clientes con "
                "entregas registradas tienen más de 14 días de "
                "autonomía. Volvé a chequear durante la semana."
            )
        else:
            # Ordenar por urgencia (días ascendente)
            _filas_log.sort(key=lambda x: x["_dias_sort"])
            _n_urg = sum(
                1 for f in _filas_log
                if "🔴" in f["Urgencia"]
            )
            _n_sem = sum(
                1 for f in _filas_log
                if "🟠" in f["Urgencia"]
            )
            _n_prox = sum(
                1 for f in _filas_log
                if "🟡" in f["Urgencia"]
            )
            _logc1, _logc2, _logc3 = st.columns(3)
            _logc1.metric(
                "🔴 Urgentes / agotados", _n_urg,
                help="Stock para 3 días o menos, o ya agotado",
            )
            _logc2.metric(
                "🟠 Esta semana", _n_sem,
                help="Stock para 4-7 días",
            )
            _logc3.metric(
                "🟡 Próxima semana", _n_prox,
                help="Stock para 8-14 días",
            )

            if _n_urg > 0:
                st.error(
                    f"🔴 **{_n_urg} cliente(s) con stock urgente.** "
                    f"Contactar HOY para coordinar entrega."
                )
            elif _n_sem > 0:
                st.warning(
                    f"🟠 **{_n_sem} cliente(s) necesita reposición "
                    f"esta semana.** Anticipar logística."
                )

            # Tabla detallada
            import pandas as _pd_log
            _df_log = _pd_log.DataFrame(_filas_log).drop(
                columns=["_dias_sort"],
            )
            st.dataframe(
                _df_log, hide_index=True, width="stretch",
            )

    except Exception as _e_log:
        import traceback as _tb_log
        st.error(
            f"⚠️ Error en bloque logística: "
            f"{type(_e_log).__name__}: {_e_log}"
        )
        with st.expander("Ver traceback completo"):
            st.code(_tb_log.format_exc(), language="python")
    finally:
        # Restaurar funciones DB originales (revertir monkey-patch)
        try:
            import src.database as _db_mod_ct_rst
            if hasattr(_db_mod_ct_rst, "_ORIG_DASH_CACHE"):
                for _n, _fn in _db_mod_ct_rst._ORIG_DASH_CACHE.items():
                    setattr(_db_mod_ct_rst, _n, _fn)
        except Exception:
            pass

    st.divider()

    # Accesos rápidos — 4 acciones DISTINTAS, no redundantes
    st.markdown("### 🚀 ¿Qué necesitás hacer hoy?")
    st.caption(
        "Click en cada tarjeta para que te diga a qué pestaña ir. "
        "Cada una sirve para algo diferente — leelas:"
    )

    qa1, qa2, qa3, qa4 = st.columns(4)

    with qa1:
        st.markdown(
            "<div style='background:#1B3E27;color:white;border-radius:8px;"
            "padding:14px;height:160px;'>"
            "<h4 style='color:white;margin-top:0;'>🐄 Pesar con drone</h4>"
            "<p style='font-size:0.85em;color:#d8e8d6;'>"
            "Tenés un video del drone y querés saber cuántos animales "
            "hay y cuánto pesan</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("👉 Ir a Video drone", key="qa_drone",
                      width="stretch"):
            st.info(
                "Hacé click en la pestaña **🎞️ Video 🐄** de arriba. "
                "Ahí subís el video del drone y te procesa conteo + peso."
            )

    with qa2:
        st.markdown(
            "<div style='background:#1B3E27;color:white;border-radius:8px;"
            "padding:14px;height:160px;'>"
            "<h4 style='color:white;margin-top:0;'>✏️ Pesada manual</h4>"
            "<p style='font-size:0.85em;color:#d8e8d6;'>"
            "Pesaste con balanza, manga o estimación. Cargás el dato "
            "directo, sin video</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("👉 Ir a Clientes/Lotes", key="qa_manual",
                      width="stretch"):
            st.info(
                "Hacé click en **🏢 Clientes/Lotes** → tab Lotes → "
                "seleccionar el lote → expander **'✏️ Cargar pesada manual'**."
            )

    with qa3:
        st.markdown(
            "<div style='background:#8BC53F;color:#1B3E27;border-radius:8px;"
            "padding:14px;height:160px;'>"
            "<h4 style='color:#1B3E27;margin-top:0;'>🤖 Asesor IA</h4>"
            "<p style='font-size:0.85em;'>"
            "Consultar al asesor experto: dieta, diagnóstico, manejo, "
            "informes para el productor</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("👉 Ir a Asesor IA", key="qa_ia",
                      width="stretch"):
            st.info(
                "Hacé click en **🤖 Asesor IA 🍽️** de arriba. "
                "El agente formula dietas, diagnostica problemas y "
                "te genera informes en PDF."
            )

    with qa4:
        st.markdown(
            "<div style='background:#1B3E27;color:white;border-radius:8px;"
            "padding:14px;height:160px;'>"
            "<h4 style='color:white;margin-top:0;'>📚 Ver historial</h4>"
            "<p style='font-size:0.85em;color:#d8e8d6;'>"
            "Pesadas anteriores, dietas previas, evolución de ADG por "
            "cliente y por lote</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("👉 Ir a Historial", key="qa_hist",
                      width="stretch"):
            st.info(
                "Hacé click en **📚 Historial** de arriba. "
                "Seleccionás cliente y lote, ves toda la evolución."
            )

    st.caption(
        f"💡 Las pestañas marcadas con 🐄 son del **módulo Drone**, "
        f"las 🍽️ del **módulo Asesor Nutricional**."
    )

    st.divider()
    st.markdown(
        "### 🎯 Modos de uso por cliente\n"
        "El sistema soporta clientes con distintas necesidades — usá lo que aplique:"
    )
    mu1, mu2, mu3 = st.columns(3)
    with mu1:
        st.markdown(
            "<div style='border:1px solid #1B3E27;border-radius:8px;"
            "padding:12px;height:160px;'>"
            "<h4 style='color:#1B3E27;margin-top:0;'>"
            "🐄 Cliente con DRONE</h4>"
            "<ul style='font-size:0.9em;'>"
            "<li>Análisis por video (Imagen/Video)</li>"
            "<li>Conteo y peso automático</li>"
            "<li>Asesoría nutricional</li>"
            "<li>Tracking completo</li>"
            "</ul></div>", unsafe_allow_html=True,
        )
    with mu2:
        st.markdown(
            "<div style='border:1px solid #1B3E27;border-radius:8px;"
            "padding:12px;height:160px;'>"
            "<h4 style='color:#1B3E27;margin-top:0;'>"
            "✏️ Cliente con BALANZA</h4>"
            "<ul style='font-size:0.9em;'>"
            "<li>Carga manual de pesadas</li>"
            "<li>(Clientes/Lotes → Pesada manual)</li>"
            "<li>Asesoría nutricional</li>"
            "<li>Tracking completo</li>"
            "</ul></div>", unsafe_allow_html=True,
        )
    with mu3:
        st.markdown(
            "<div style='border:1px solid #1B3E27;border-radius:8px;"
            "padding:12px;height:160px;'>"
            "<h4 style='color:#1B3E27;margin-top:0;'>"
            "🍽️ Cliente solo NUTRICIÓN</h4>"
            "<ul style='font-size:0.9em;'>"
            "<li>Sin pesadas necesarias</li>"
            "<li>Asesor IA + dietas + clima</li>"
            "<li>Cargás solo cliente y lote</li>"
            "<li>Recomendaciones generales</li>"
            "</ul></div>", unsafe_allow_html=True,
        )


# ----------------------- CLIENTES Y LOTES -----------------------------
def _render_seguimiento_completo_lote(lote_id_sel: int) -> None:
    """Ficha completa de seguimiento del lote: peso, dietas,
    movimientos, cargas del comedero y gráfico comparativo.

    Se renderiza tanto en la pestaña Clientes/Lotes (dentro
    de la ficha del lote) como en la pestaña Historial.
    """
    lote = db.obtener_lote(lote_id_sel)
    pesadas = db.listar_pesadas(lote_id_sel)
    evolucion = db.calcular_evolucion_lote(lote_id_sel)
    dietas = db.listar_dietas(lote_id_sel)

    # ---- Cabecera del lote ----
    st.markdown(f"#### 🐄 {lote['identificador']}")
    cab1, cab2, cab3, cab4 = st.columns(4)
    cab1.metric("Cliente", lote["cliente_nombre"])
    cab2.metric("Corral", lote.get("corral", "—") or "—")
    cab3.metric("Categoría", lote.get("categoria", "—"))
    cab4.metric("Estado", lote.get("estado", "").upper())

    cab5, cab6, cab7, cab8 = st.columns(4)
    # Cantidad actual = inicial + Σ movimientos. Mostramos el
    # delta vs inicial para que se vea de un vistazo si hubo
    # bajas o ingresos. Si nunca hubo movimientos, delta=0 y
    # Streamlit no muestra la flechita.
    _cant_inicial = int(lote.get("cantidad_inicial", 0) or 0)
    _cant_actual = db.cantidad_vigente_lote(lote_id_sel)
    _delta_cant = _cant_actual - _cant_inicial
    cab5.metric(
        "Cantidad actual",
        f"{_cant_actual} cab.",
        f"{_delta_cant:+d} vs {_cant_inicial} inicial"
        if _delta_cant != 0 else None,
    )
    # Peso ingreso + fecha de ingreso como subtexto.
    _f_ing_raw = (lote.get("fecha_ingreso") or "")[:10]
    _f_ing_disp = ""
    if _f_ing_raw:
        try:
            _f_ing_disp = datetime.strptime(
                _f_ing_raw, "%Y-%m-%d"
            ).strftime("%d/%m/%Y")
        except Exception:
            _f_ing_disp = _f_ing_raw
    cab6.metric(
        "Peso ingreso",
        f"{lote.get('peso_ingreso_kg', 0):.0f} kg",
        f"📅 ingresó {_f_ing_disp}" if _f_ing_disp else None,
    )
    cab7.metric("Pesadas", evolucion["n_pesadas"])
    if evolucion["adg_total"]:
        cab8.metric(
            "ADG total", f"{evolucion['adg_total']:.3f} kg/día",
            f"{evolucion['ganancia_total_kg']:+.1f} kg en {evolucion['dias_totales']} d",
        )

    # ---- Encierre / Salida proyectada ----
    # Si tenemos peso ingreso + peso objetivo + ADPV, calculamos
    # los días de encierre y la fecha estimada de salida. Si la
    # fecha objetivo no está cargada en el lote, ofrecemos un
    # botón para guardarla. Esto es clave para el cron de
    # cambio de fase (la última fase usa objetivo_fecha como fin).
    _proy_lote = db.calcular_fecha_objetivo_estimada(
        fecha_ingreso=lote.get("fecha_ingreso") or "",
        peso_ingreso_kg=float(lote.get("peso_ingreso_kg") or 0),
        peso_objetivo_kg=float(lote.get("objetivo_peso_kg") or 0),
        adpv_kg_dia=lote.get("adpv_objetivo_kg"),
        categoria=lote.get("categoria") or "",
    )
    obj_fecha_actual = (lote.get("objetivo_fecha") or "")[:10]
    if _proy_lote or obj_fecha_actual:
        # Header de 3 columnas. El "Peso HOY (proyectado)" que
        # antes vivía acá se movió al bloque dedicado de abajo
        # ("📊 Peso HOY — proyecciones") que muestra las 3
        # visiones (lineal, ajustado por clima, real balanza)
        # de manera más clara.
        c_obj1, c_obj2, c_obj3 = st.columns(3)
        c_obj1.metric(
            "Peso objetivo",
            f"{float(lote.get('objetivo_peso_kg') or 0):.0f} kg"
            if lote.get("objetivo_peso_kg") else "—",
        )

        if obj_fecha_actual:
            # Días entre hoy y la fecha objetivo guardada
            try:
                from datetime import date as _date2
                d_obj_g = datetime.strptime(
                    obj_fecha_actual, "%Y-%m-%d"
                ).date()
                dias_restantes = (d_obj_g - _date2.today()).days
                c_obj2.metric(
                    "Salida proyectada",
                    obj_fecha_actual,
                    f"{dias_restantes:+d} días desde hoy"
                    if dias_restantes != 0 else None,
                )
            except (ValueError, TypeError):
                c_obj2.metric(
                    "Salida proyectada", obj_fecha_actual,
                )
        elif _proy_lote:
            c_obj2.metric(
                "Salida estimada",
                _proy_lote["fecha_objetivo"],
                "calculada — no guardada",
            )
        else:
            c_obj2.metric("Salida proyectada", "—")

        if _proy_lote:
            c_obj3.metric(
                "Días de encierre",
                f"{_proy_lote['dias_encierre']} días",
                f"ADPV {_proy_lote['adpv_usado']:.2f} "
                f"({_proy_lote['fuente_adpv']})",
            )
        else:
            c_obj3.metric("Días de encierre", "—")

        # ─── 📊 Peso HOY — 3 proyecciones ───
        # (1) Lineal: PV_ingreso + ADG × días (teórico, asume
        #     cumplimiento perfecto del ADG)
        # (2) Ajustado por clima: ADG con descuento día a día según
        #     severidad climática real (más realista en otoño/invierno)
        # (3) Real (balanza): la última pesada interpolada con ADG
        #     hasta hoy. Si no hay pesadas, "—".
        # Cacheamos en session_state — la versión ajustada por clima
        # llama a Open-Meteo Archive y puede tardar 1-2 segundos.
        try:
            from src.stock_producto import (
                estimar_pv_lineal_simple as _epv_lin,
                estimar_pv_balanza as _epv_bal,
                estimar_pv_ajustado_clima as _epv_aj,
            )
            _hoy_iso_pv = datetime.now().strftime("%Y-%m-%d")
            _pv_lineal = _epv_lin(lote, _hoy_iso_pv)
            _pv_bal, _f_bal = _epv_bal(lote, _hoy_iso_pv)
            # Cache del ajustado por clima
            _cache_pv_key = (
                f"pv_ajustado_clima_{lote_id_sel}_{_hoy_iso_pv}"
            )
            if _cache_pv_key in st.session_state:
                _pv_aj_info = st.session_state[_cache_pv_key]
            else:
                with st.spinner(
                    "🌦️ Calculando peso ajustado por clima..."
                ):
                    _pv_aj_info = _epv_aj(lote, _hoy_iso_pv)
                st.session_state[_cache_pv_key] = _pv_aj_info
            _pv_aj = float(_pv_aj_info.get("pv_ajustado_kg") or 0)
            _delta_aj = float(
                _pv_aj_info.get("delta_vs_lineal_kg") or 0
            )

            st.markdown("##### 📊 Peso HOY — proyecciones")
            _pv_c1, _pv_c2, _pv_c3 = st.columns(3)
            # (1) Lineal
            _pv_c1.metric(
                "📐 Lineal (teórico)",
                f"{_pv_lineal:.0f} kg" if _pv_lineal > 0 else "—",
                help=(
                    "PV de ingreso + ADG objetivo × días "
                    "transcurridos. Asume que se está cumpliendo "
                    "el ADG sin descuentos por clima."
                ),
            )
            # (2) Ajustado por clima
            _origen_aj = _pv_aj_info.get("origen", "sin_datos")
            if _pv_aj > 0 and _origen_aj == "ok":
                _txt_aj_delta = None
                _dias_arch = _pv_aj_info.get("dias_con_archive", 0)
                _dias_sin = _pv_aj_info.get("dias_sin_archive", 0)
                if abs(_delta_aj) >= 0.5:
                    _txt_aj_delta = (
                        f"{_delta_aj:+.1f} kg vs lineal · "
                        f"{_pv_aj_info.get('dias_adversos',0)}d "
                        "adversos"
                    )
                _pv_c2.metric(
                    "🌦️ Ajustado por clima",
                    f"{_pv_aj:.0f} kg",
                    _txt_aj_delta,
                    help=(
                        "Itera día a día desde el ingreso aplicando "
                        "descuento del ADG según severidad real "
                        "(frío sostenido, lluvia, viento, HR alta). "
                        f"Calculado con datos archive de "
                        f"{_dias_arch} días. "
                        + (
                            f"Últimos {_dias_sin} días sin datos "
                            "del archive (delay ~5d) → asume ADG "
                            "completo (conservador)."
                            if _dias_sin > 0 else ""
                        )
                    ),
                )
            elif _origen_aj == "sin_clima_reciente":
                # Período entero dentro del delay del archive
                _pv_c2.metric(
                    "🌦️ Ajustado por clima",
                    "—",
                    help=(
                        "Open-Meteo Archive tiene ~5 días de "
                        "delay. Este lote ingresó hace menos de "
                        "5 días, no hay datos confirmados todavía. "
                        "El valor aparecerá cuando se publiquen "
                        "los datos archive."
                    ),
                )
            elif _origen_aj == "sin_clima":
                _pv_c2.metric(
                    "🌦️ Ajustado por clima",
                    "—",
                    help=(
                        "Open-Meteo no respondió. Puede ser un "
                        "problema temporal de la API o que falten "
                        "coordenadas del cliente. Reintentar en "
                        "unos minutos."
                    ),
                )
            else:
                _pv_c2.metric(
                    "🌦️ Ajustado por clima",
                    "—",
                    help=(
                        "Faltan datos del lote para calcular: "
                        "ADG objetivo, fecha de ingreso, o "
                        "peso de ingreso."
                    ),
                )
            # (3) Real (balanza)
            if _pv_bal > 0 and _f_bal:
                from datetime import datetime as _dt_b
                try:
                    _dias_desde_pesada = (
                        datetime.now().date()
                        - _dt_b.strptime(_f_bal, "%Y-%m-%d").date()
                    ).days
                except Exception:
                    _dias_desde_pesada = 0
                _delta_vs_lin = _pv_bal - _pv_lineal
                _pv_c3.metric(
                    "⚖️ Real (balanza)",
                    f"{_pv_bal:.0f} kg",
                    (
                        f"{_delta_vs_lin:+.1f} kg vs lineal · "
                        f"última pesada hace {_dias_desde_pesada}d"
                    ),
                    help=(
                        f"Última pesada registrada: {_f_bal}. "
                        "Si la pesada no es de HOY, se proyecta con "
                        "ADG objetivo hasta hoy."
                    ),
                )
            else:
                _pv_c3.metric(
                    "⚖️ Real (balanza)",
                    "—",
                    help=(
                        "Este lote no tiene pesadas registradas. "
                        "Procesá un video desde la pestaña Video o "
                        "cargá una pesada manual."
                    ),
                )
        except Exception as _e_3pv:
            st.caption(f"_(No pude calcular las 3 proyecciones: {_e_3pv})_")

        # Si tenemos cálculo pero el lote no tiene fecha
        # objetivo guardada, ofrecer guardarla con un click.
        if _proy_lote and not obj_fecha_actual:
            st.info(
                f"💡 El lote no tiene **fecha objetivo guardada** — "
                f"el sistema la estima en "
                f"**{_proy_lote['fecha_objetivo']}** "
                f"({_proy_lote['dias_encierre']} días desde el "
                f"ingreso). Guardala para que el cron de cambio "
                f"de fase pueda calcular bien la duración de la "
                f"última fase del plan de adaptación."
            )
            if st.button(
                f"💾 Guardar fecha objetivo "
                f"({_proy_lote['fecha_objetivo']})",
                key=f"guardar_obj_fecha_{lote_id_sel}",
            ):
                try:
                    db.actualizar_lote(
                        lote_id_sel,
                        objetivo_fecha=_proy_lote[
                            "fecha_objetivo"],
                    )
                    st.success(
                        f"✅ Fecha objetivo guardada: "
                        f"{_proy_lote['fecha_objetivo']}"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

    # ---- Línea de evolución ----
    if pesadas:
        st.markdown("##### 📈 Evolución de peso promedio")
        df_pes = pd.DataFrame([
            {
                "Fecha": p["fecha"],
                "Peso prom (kg)": p["peso_promedio_kg"],
                "CV (%)": p.get("cv_pct", 0),
                "Animales": p["cantidad_animales"],
            }
            for p in pesadas
        ])
        try:
            import matplotlib.pyplot as plt
            fechas = [datetime.strptime(p["fecha"], "%Y-%m-%d") for p in pesadas]
            pesos = [p["peso_promedio_kg"] for p in pesadas]
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.plot(fechas, pesos, marker="o", color="#2c5f2d", linewidth=2,
                    markersize=8)
            if lote.get("objetivo_peso_kg"):
                ax.axhline(lote["objetivo_peso_kg"], color="red",
                            linestyle="--", alpha=0.7,
                            label=f"Objetivo {lote['objetivo_peso_kg']:.0f} kg")
                ax.legend()
            ax.set_ylabel("Peso promedio (kg)")
            ax.set_xlabel("Fecha")
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"No pude graficar: {e}")

        st.markdown("##### 📋 Pesadas")
        st.dataframe(df_pes, hide_index=True, width="stretch")

        # Tabla de tendencia (ADG entre cada par)
        if evolucion["tendencia"]:
            st.markdown("##### 🔄 ADG entre pesadas consecutivas")
            tend_df = pd.DataFrame([
                {
                    "Desde": t["desde"], "Hasta": t["hasta"],
                    "Días": t["dias"],
                    "Ganancia (kg)": round(t["ganancia_kg"], 1),
                    "ADG (kg/día)": round(t["adg"], 3),
                }
                for t in evolucion["tendencia"]
            ])
            st.dataframe(tend_df, hide_index=True, width="stretch")

    else:
        st.info(
            "Este lote no tiene pesadas registradas todavía. "
            "Procesá un video en **🎞️ Video** y guardalo al historial."
        )

    # ---- Evolución del consumo de MS ----
    # Gráfico que muestra cómo evoluciona el consumo de materia seca
    # (DMI) del lote a medida que los animales ganan peso. El sistema
    # escala la dieta vigente por el peso vivo proyectado (ADG), así
    # que esta línea sube con el tiempo conforme los animales engordan.
    # Útil para anticipar cuánto va a aumentar la demanda de insumos.
    if dietas and lote.get("fecha_ingreso"):
        try:
            from src.stock_producto import serie_consumo_ms_lote
            _serie = serie_consumo_ms_lote(
                lote_id_sel, paso_dias=7,
            )
        except Exception as _e_serie:
            _serie = []
            st.caption(
                f"_No pude calcular la serie de consumo: {_e_serie}_"
            )
        if _serie:
            st.markdown("---")
            st.markdown(
                "##### 🌾 Evolución del consumo de materia seca"
            )

            # Si el lote no tiene ADPV cargado, el factor de escala se
            # queda fijo en 1.0 → la línea sale plana. Avisar para que
            # el usuario lo cargue.
            _adpv_lote = lote.get("adpv_objetivo_kg")
            _sin_adg = not _adpv_lote or float(_adpv_lote) <= 0
            if _sin_adg:
                st.warning(
                    "⚠️ Este lote no tiene **ADPV objetivo** cargado, "
                    "así que el sistema no puede proyectar cómo crece "
                    "el consumo a medida que los animales engordan. "
                    "El gráfico va a salir plano (siempre la dieta "
                    "original). Cargá el ADPV objetivo en el form "
                    "del lote (arriba en esta misma pantalla) para "
                    "que aparezca la curva real."
                )

            _hoy_iso = datetime.now().strftime("%Y-%m-%d")
            # Métricas resumen: hoy vs ingreso, % aumento.
            _df_serie = pd.DataFrame(_serie)
            _hist = _df_serie[_df_serie["fecha"] <= _hoy_iso]
            _proy = _df_serie[_df_serie["fecha"] > _hoy_iso]

            if not _hist.empty:
                _ini = _hist.iloc[0]
                _ult = _hist.iloc[-1]
                _delta_an = (
                    _ult["kg_ms_animal_dia"] - _ini["kg_ms_animal_dia"]
                )
                _pct_an = (
                    (_delta_an / _ini["kg_ms_animal_dia"] * 100)
                    if _ini["kg_ms_animal_dia"] > 0 else 0
                )
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(
                    "MS / animal HOY",
                    f"{_ult['kg_ms_animal_dia']:.2f} kg",
                    f"{_delta_an:+.2f} kg vs ingreso "
                    f"({_pct_an:+.1f}%)"
                    if _delta_an != 0 else None,
                )
                k2.metric(
                    "MS / lote HOY",
                    f"{_ult['kg_ms_lote_dia']:.0f} kg/día",
                )
                k3.metric(
                    "PV proyectado HOY",
                    f"{_ult['peso_vivo_kg']:.0f} kg",
                )
                if not _proy.empty:
                    _fin = _proy.iloc[-1]
                    _pct_pv = (
                        (_fin["kg_ms_animal_dia"] /
                         _fin["peso_vivo_kg"] * 100)
                        if _fin["peso_vivo_kg"] > 0 else 0
                    )
                    k4.metric(
                        f"MS / animal al {_fin['fecha']}",
                        f"{_fin['kg_ms_animal_dia']:.2f} kg",
                        f"≈ {_pct_pv:.1f}% PV",
                    )

            # Obtener cargas reales para superponer la curva real
            try:
                from src.stock_producto import (
                    serie_cargas_reales_ms,
                    serie_cargas_rollo_lote,
                )
                _serie_real = serie_cargas_reales_ms(lote_id_sel)
                _serie_rollo = serie_cargas_rollo_lote(lote_id_sel)
            except Exception:
                _serie_real = []
                _serie_rollo = []

            # Gráfico UNIFICADO con tres series superpuestas:
            #   - Línea sólida verde: histórico proyectado
            #   - Línea punteada gris: proyección futura
            #   - Puntos azules: cargas reales registradas
            # Para que el histórico y la proyección se vean continuos
            # (sin gap visual) en altair, duplicamos el punto de "hoy"
            # en ambos tramos.
            try:
                import altair as alt

                _df_serie = _df_serie.rename(columns={
                    "fecha": "Fecha",
                })

                _df_hist = _df_serie[_df_serie["es_proyeccion"] == False].copy()
                _df_proy = _df_serie[_df_serie["es_proyeccion"] == True].copy()

                # Conectar visualmente: el primer punto de proyección se
                # solapa con el último del histórico (mismo X, mismo Y)
                if not _df_hist.empty and not _df_proy.empty:
                    _ult_hist = _df_hist.iloc[-1:].copy()
                    _df_proy = pd.concat([_ult_hist, _df_proy],
                                         ignore_index=True)

                _df_real_chart = pd.DataFrame(_serie_real)
                if not _df_real_chart.empty:
                    _df_real_chart["Fecha"] = pd.to_datetime(
                        _df_real_chart["fecha"]
                    )

                _df_rollo_chart = pd.DataFrame(_serie_rollo)
                if not _df_rollo_chart.empty:
                    _df_rollo_chart["Fecha"] = pd.to_datetime(
                        _df_rollo_chart["fecha"]
                    )

                _t1, _t2 = st.tabs(["Por animal", "Por lote"])

                def _build_chart(col_y_proy: str, col_y_real: str,
                                 y_title: str, extra_tt: list):
                    """Arma el gráfico stack: hist+proy+real."""
                    base_y_min = float("inf")
                    base_y_max = float("-inf")
                    # Incluir todas las series para que ningún punto
                    # quede fuera del rango visible.
                    for _d in (_df_hist, _df_proy):
                        if not _d.empty:
                            base_y_min = min(
                                base_y_min, _d[col_y_proy].min()
                            )
                            base_y_max = max(
                                base_y_max, _d[col_y_proy].max()
                            )
                    if not _df_real_chart.empty:
                        base_y_min = min(
                            base_y_min,
                            _df_real_chart[col_y_real].min(),
                        )
                        base_y_max = max(
                            base_y_max,
                            _df_real_chart[col_y_real].max(),
                        )
                    if not _df_rollo_chart.empty:
                        base_y_min = min(
                            base_y_min,
                            _df_rollo_chart[col_y_real].min(),
                        )
                        base_y_max = max(
                            base_y_max,
                            _df_rollo_chart[col_y_real].max(),
                        )
                    if base_y_min == float("inf"):
                        base_y_min, base_y_max = 0, 10
                    # Padding amplio: 15% del rango + asegurar
                    # mínimo 1.0 kg para que la curva no se vea
                    # apretada.
                    _rng = max(1.0, base_y_max - base_y_min)
                    pad = _rng * 0.15
                    y_dom = [
                        max(0, base_y_min - pad),
                        base_y_max + pad,
                    ]

                    capas = []
                    # Capa 1: línea verde histórica + puntos
                    if not _df_hist.empty:
                        _base_hist = alt.Chart(
                            _df_hist,
                        ).encode(
                            x=alt.X("Fecha:T", title="Fecha"),
                            y=alt.Y(
                                f"{col_y_proy}:Q",
                                title=y_title,
                                scale=alt.Scale(
                                    domain=y_dom, zero=False,
                                ),
                            ),
                            tooltip=[
                                "Fecha:T",
                                alt.Tooltip(
                                    f"{col_y_proy}:Q",
                                    title="Proyectado",
                                    format=".2f",
                                ),
                            ] + extra_tt,
                        )
                        capas.append(
                            _base_hist.mark_line(
                                color="#2c5f2d",
                                strokeWidth=2.5,
                            )
                        )
                        capas.append(
                            _base_hist.mark_circle(
                                color="#2c5f2d",
                                size=50,
                            )
                        )
                    # Capa 2: línea gris punteada (proyección)
                    if not _df_proy.empty:
                        _base_proy = alt.Chart(
                            _df_proy,
                        ).encode(
                            x="Fecha:T",
                            y=alt.Y(
                                f"{col_y_proy}:Q",
                                scale=alt.Scale(
                                    domain=y_dom, zero=False,
                                ),
                            ),
                            tooltip=[
                                "Fecha:T",
                                alt.Tooltip(
                                    f"{col_y_proy}:Q",
                                    title="Proyectado",
                                    format=".2f",
                                ),
                            ] + extra_tt,
                        )
                        capas.append(
                            _base_proy.mark_line(
                                color="#9aa9a0",
                                strokeWidth=2,
                                strokeDash=[5, 4],
                            )
                        )
                        capas.append(
                            _base_proy.mark_circle(
                                color="#9aa9a0",
                                size=35,
                            )
                        )
                    # Capa 3: puntos azules del silo real
                    if not _df_real_chart.empty:
                        capas.append(
                            alt.Chart(_df_real_chart).mark_circle(
                                color="#1f6feb",
                                size=140,
                            ).encode(
                                x="Fecha:T",
                                y=alt.Y(
                                    f"{col_y_real}:Q",
                                    scale=alt.Scale(
                                        domain=y_dom, zero=False,
                                    ),
                                ),
                                tooltip=[
                                    "Fecha:T",
                                    alt.Tooltip(
                                        f"{col_y_real}:Q",
                                        title="Silo (real)",
                                        format=".2f",
                                    ),
                                    alt.Tooltip(
                                        "cantidad_animales:Q",
                                        title="Cabezas",
                                    ),
                                    alt.Tooltip(
                                        "kg_cargados_tal_cual:Q",
                                        title="kg cargados (TC)",
                                        format=".0f",
                                    ),
                                ],
                            )
                        )
                    # Capa 4: triángulos naranjas del rollo real
                    if not _df_rollo_chart.empty:
                        capas.append(
                            alt.Chart(_df_rollo_chart).mark_point(
                                color="#d97706",
                                size=180,
                                filled=True,
                                shape="triangle-up",
                            ).encode(
                                x="Fecha:T",
                                y=alt.Y(
                                    f"{col_y_real}:Q",
                                    scale=alt.Scale(
                                        domain=y_dom, zero=False,
                                    ),
                                ),
                                tooltip=[
                                    "Fecha:T",
                                    alt.Tooltip(
                                        f"{col_y_real}:Q",
                                        title="Rollo (real)",
                                        format=".2f",
                                    ),
                                    alt.Tooltip(
                                        "tipo_forraje:N",
                                        title="Forraje",
                                    ),
                                    alt.Tooltip(
                                        "cantidad_rollos:Q",
                                        title="Rollos",
                                    ),
                                    alt.Tooltip(
                                        "kg_cargados_tal_cual:Q",
                                        title="kg TC",
                                        format=".0f",
                                    ),
                                    alt.Tooltip(
                                        "desperdicio_pct:Q",
                                        title="Desperdicio %",
                                    ),
                                ],
                            )
                        )
                    if not capas:
                        return None
                    return alt.layer(*capas).properties(height=340)

                with _t1:
                    _ch = _build_chart(
                        "kg_ms_animal_dia",
                        "kg_ms_animal_dia_real",
                        "kg MS / animal / día",
                        [
                            alt.Tooltip(
                                "peso_vivo_kg:Q",
                                title="PV (kg)", format=".0f",
                            ),
                            alt.Tooltip(
                                "factor_escala:Q",
                                title="Factor", format=".3f",
                            ),
                        ],
                    )
                    if _ch is not None:
                        st.altair_chart(_ch, use_container_width=True)
                with _t2:
                    _ch = _build_chart(
                        "kg_ms_lote_dia",
                        "kg_ms_lote_dia_real",
                        "kg MS / lote / día",
                        [
                            alt.Tooltip(
                                "cantidad_animales:Q",
                                title="Cabezas",
                            ),
                        ],
                    )
                    if _ch is not None:
                        st.altair_chart(_ch, use_container_width=True)

                # Leyenda manual con bullets de color
                st.caption(
                    "🟢 **Línea verde** = histórico proyectado por dieta "
                    "+ ADG · ⚪ **Línea punteada** = proyección futura "
                    "hasta fecha objetivo · 🔵 **Puntos azules** = "
                    "silo/mezcla real (kg cargados × %MS dieta) · "
                    "🔺 **Triángulos naranjas** = rollo real "
                    "(kg cargados × %MS × (1−desperdicio))."
                )

                # ── Tabla detallada de la serie del gráfico ──
                # Permite ver los números exactos detrás de la curva
                # y exportarlos. Cada fila es un punto del gráfico.
                with st.expander(
                    "📋 Ver datos del gráfico (tabla detallada)",
                    expanded=False,
                ):
                    if _serie:
                        _df_serie_tabla = pd.DataFrame([
                            {
                                "Fecha": s["fecha"],
                                "Días desde ingreso":
                                    s.get("dias_desde_ingreso", 0),
                                "Peso vivo (kg)":
                                    s.get("peso_vivo_kg", 0),
                                "Cabezas":
                                    s.get("cantidad_animales", 0),
                                "Factor escala":
                                    f"{s.get('factor_escala', 1.0):.4f}",
                                "MS proy total (kg/an/día)":
                                    s.get("kg_ms_animal_dia", 0),
                                "MS proy silo (kg/an/día)":
                                    s.get(
                                        "kg_ms_animal_dia_solo_silo",
                                        0,
                                    ),
                                "MS proy lote (kg/día)":
                                    s.get("kg_ms_lote_dia", 0),
                                "Tramo": (
                                    "Proyección"
                                    if s.get("es_proyeccion")
                                    else "Histórico"
                                ),
                            }
                            for s in _serie
                        ])
                        st.dataframe(
                            _df_serie_tabla,
                            hide_index=True,
                            width="stretch",
                        )
                        st.caption(
                            "**MS proy total** incluye silo + rollo "
                            "(es el DMI total de la dieta). "
                            "**MS proy silo** excluye libre disposición "
                            "(es lo que se carga al silocomedero, "
                            "comparable con cargas reales). "
                            "**Factor escala** = peso vivo proyectado ÷ "
                            "peso vivo cuando se formuló la dieta — "
                            "es cuánto se ajustó el consumo por ADG. "
                            "Para análisis: copiá la tabla con el "
                            "botón de los 3 puntos arriba a la derecha "
                            "y pegala en Excel."
                        )

                        # Bloque mini-stats útiles
                        if not _df_serie_tabla.empty:
                            _hoy_iso2 = datetime.now().strftime(
                                "%Y-%m-%d"
                            )
                            _hist_only = [
                                s for s in _serie
                                if not s.get("es_proyeccion")
                            ]
                            _proy_only = [
                                s for s in _serie
                                if s.get("es_proyeccion")
                            ]
                            _ms1, _ms2, _ms3 = st.columns(3)
                            if _hist_only:
                                _delta_hist = (
                                    _hist_only[-1].get(
                                        "kg_ms_animal_dia", 0,
                                    )
                                    - _hist_only[0].get(
                                        "kg_ms_animal_dia", 0,
                                    )
                                )
                                _ms1.metric(
                                    "Δ consumo histórico",
                                    f"{_delta_hist:+.2f} kg",
                                    f"{len(_hist_only)} puntos",
                                )
                            if _proy_only:
                                _delta_proy = (
                                    _proy_only[-1].get(
                                        "kg_ms_animal_dia", 0,
                                    )
                                    - _proy_only[0].get(
                                        "kg_ms_animal_dia", 0,
                                    )
                                )
                                _ms2.metric(
                                    "Δ consumo proyectado",
                                    f"{_delta_proy:+.2f} kg",
                                    f"{len(_proy_only)} puntos",
                                )
                            _ms3.metric(
                                "Rango factor escala",
                                f"{min(s.get('factor_escala', 1.0) for s in _serie):.3f} – "
                                f"{max(s.get('factor_escala', 1.0) for s in _serie):.3f}",
                            )
                    else:
                        st.info(
                            "Sin datos en la serie todavía. "
                            "Cargá una dieta y un ADPV objetivo "
                            "para ver puntos."
                        )

            except Exception as _e_chart:
                st.warning(
                    f"No pude renderizar el gráfico: {_e_chart}"
                )

            # Tabla comparativa proyectado vs real
            if _serie_real or _serie_rollo:
                st.markdown(
                    "**📊 Comparativa proyectado vs real "
                    "(por período de carga)**"
                )
                _rows_tabla = []

                # Helper: promedio del proyectado durante un
                # período [desde, hasta]. El animal define el
                # consumo en ese período — comparamos promedios.
                from datetime import timedelta as _td_h

                def _proy_promedio_periodo(desde_iso, dias):
                    try:
                        d_ini = datetime.strptime(
                            desde_iso, "%Y-%m-%d"
                        ).date()
                    except Exception:
                        return None
                    d_fin = d_ini + _td_h(days=int(dias))
                    # Tomar los puntos del proyectado dentro del
                    # rango [d_ini, d_fin]. Si no hay ninguno en
                    # el rango exacto, usar el más cercano al
                    # punto medio.
                    vals = []
                    for p in _serie:
                        try:
                            pf = datetime.strptime(
                                p["fecha"], "%Y-%m-%d"
                            ).date()
                            if d_ini <= pf <= d_fin:
                                vals.append(
                                    float(
                                        p.get(
                                            "kg_ms_animal_dia"
                                        ) or 0
                                    )
                                )
                        except Exception:
                            pass
                    if vals:
                        return sum(vals) / len(vals)
                    # Fallback: punto medio + más cercano
                    d_mid = d_ini + _td_h(days=int(dias / 2))
                    mejor, mejor_d = None, 999
                    for p in _serie:
                        try:
                            pf = datetime.strptime(
                                p["fecha"], "%Y-%m-%d"
                            ).date()
                            dif = abs((pf - d_mid).days)
                            if dif < mejor_d:
                                mejor_d = dif
                                mejor = float(
                                    p.get("kg_ms_animal_dia") or 0
                                )
                        except Exception:
                            pass
                    return mejor

                # Combinar entregas de silo y de rollo. Cada una
                # es UN evento con su propio período. Para saber
                # cuánto duró REALMENTE necesitamos la fecha de
                # la SIGUIENTE carga del mismo tipo — sino está
                # "en curso" y solo tenemos la estimación.
                _silos_ord = sorted(
                    _serie_real, key=lambda x: x["fecha"],
                )
                _rollos_ord = sorted(
                    _serie_rollo, key=lambda x: x["fecha"],
                )

                _eventos = []
                _hoy_fecha = datetime.now().date()
                for i, r in enumerate(_silos_ord):
                    # Días reales = fecha siguiente carga - fecha
                    # actual. Si es la última, sigue "en curso".
                    if i + 1 < len(_silos_ord):
                        try:
                            f_act = datetime.strptime(
                                r["fecha"], "%Y-%m-%d"
                            ).date()
                            f_sig = datetime.strptime(
                                _silos_ord[i+1]["fecha"],
                                "%Y-%m-%d",
                            ).date()
                            dias_real = (f_sig - f_act).days
                            en_curso = False
                        except Exception:
                            dias_real = None
                            en_curso = True
                    else:
                        dias_real = None
                        en_curso = True
                    _eventos.append({
                        "fecha": r["fecha"],
                        "tipo": "silo",
                        "dias_estimados":
                            r.get("dias_cubiertos") or 1,
                        "dias_real": dias_real,
                        "en_curso": en_curso,
                        "kg_cargados":
                            r.get("kg_cargados_tal_cual") or 0,
                        "cabezas":
                            r.get("cantidad_animales") or 0,
                        "ratio_ms": r.get(
                            "ratio_ms_aplicado", 0.88,
                        ),
                    })
                for i, r in enumerate(_rollos_ord):
                    if i + 1 < len(_rollos_ord):
                        try:
                            f_act = datetime.strptime(
                                r["fecha"], "%Y-%m-%d"
                            ).date()
                            f_sig = datetime.strptime(
                                _rollos_ord[i+1]["fecha"],
                                "%Y-%m-%d",
                            ).date()
                            dias_real = (f_sig - f_act).days
                            en_curso = False
                        except Exception:
                            dias_real = None
                            en_curso = True
                    else:
                        dias_real = None
                        en_curso = True
                    _eventos.append({
                        "fecha": r["fecha"],
                        "tipo": "rollo",
                        "dias_estimados":
                            r.get("dias_cubiertos") or 1,
                        "dias_real": dias_real,
                        "en_curso": en_curso,
                        "kg_cargados":
                            r.get("kg_cargados_tal_cual") or 0,
                        "cabezas":
                            r.get("cantidad_animales") or 0,
                        "pct_ms": r.get("pct_ms_aplicado", 88),
                        "desperdicio_pct":
                            r.get("desperdicio_pct", 25),
                    })
                _eventos.sort(key=lambda x: x["fecha"], reverse=True)

                for ev in _eventos:
                    _dias_est = float(ev["dias_estimados"])
                    _dias_r = ev.get("dias_real")
                    _en_curso = ev.get("en_curso", False)

                    # Para promediar el proyectado: usar días
                    # reales si está cerrado, sino los estimados
                    # (referencia provisional).
                    _dias_para_proy = (
                        float(_dias_r)
                        if _dias_r is not None and _dias_r > 0
                        else _dias_est
                    )
                    _proy_prom = _proy_promedio_periodo(
                        ev["fecha"], _dias_para_proy,
                    ) or 0

                    if ev["tipo"] == "silo":
                        # Proyectado de la parte que va al silo
                        # (excluye libre disposición).
                        _proy_silo = 0
                        try:
                            d_ini = datetime.strptime(
                                ev["fecha"], "%Y-%m-%d"
                            ).date()
                            d_fin = d_ini + _td_h(
                                days=int(_dias_para_proy)
                            )
                            vals_silo = []
                            for p in _serie:
                                try:
                                    pf = datetime.strptime(
                                        p["fecha"], "%Y-%m-%d"
                                    ).date()
                                    if d_ini <= pf <= d_fin:
                                        vals_silo.append(
                                            float(
                                                p.get(
                                                    "kg_ms_animal_dia_solo_silo"
                                                ) or 0
                                            )
                                        )
                                except Exception:
                                    pass
                            if vals_silo:
                                _proy_silo = (
                                    sum(vals_silo)
                                    / len(vals_silo)
                                )
                        except Exception:
                            _proy_silo = 0
                        _proy_ref = _proy_silo or _proy_prom
                        _label_tipo = "🛢️ Silo"
                    else:
                        # Proyectado de rollo = total - silo
                        try:
                            d_ini = datetime.strptime(
                                ev["fecha"], "%Y-%m-%d"
                            ).date()
                            d_fin = d_ini + _td_h(
                                days=int(_dias_para_proy)
                            )
                            vs = []
                            for p in _serie:
                                try:
                                    pf = datetime.strptime(
                                        p["fecha"], "%Y-%m-%d"
                                    ).date()
                                    if d_ini <= pf <= d_fin:
                                        _t = float(
                                            p.get(
                                                "kg_ms_animal_dia"
                                            ) or 0
                                        )
                                        _s = float(
                                            p.get(
                                                "kg_ms_animal_dia_solo_silo"
                                            ) or 0
                                        )
                                        vs.append(_t - _s)
                                except Exception:
                                    pass
                            _proy_rollo = (
                                sum(vs) / len(vs) if vs else 0
                            )
                        except Exception:
                            _proy_rollo = 0
                        _proy_ref = (
                            _proy_rollo if _proy_rollo > 0
                            else _proy_prom
                        )
                        _label_tipo = "🌾 Rollo"

                    # Calcular real promedio SOLO si se sabe
                    # cuántos días duró realmente (hay carga
                    # siguiente). Si está en curso, no se sabe.
                    _cabezas = ev.get("cabezas") or 1
                    if _dias_r is not None and _dias_r > 0:
                        # Convertir kg cargados a kg MS aprovechado
                        # según tipo
                        if ev["tipo"] == "silo":
                            _kg_ms_total = (
                                ev["kg_cargados"]
                                * float(ev.get("ratio_ms", 0.88))
                            )
                        else:  # rollo
                            _pct_ms = float(
                                ev.get("pct_ms", 88)
                            ) / 100
                            _despe = float(
                                ev.get("desperdicio_pct", 25)
                            ) / 100
                            _kg_ms_total = (
                                ev["kg_cargados"]
                                * _pct_ms
                                * (1 - _despe)
                            )
                        real_an = (
                            _kg_ms_total
                            / _cabezas
                            / _dias_r
                        )
                        desvio = real_an - _proy_ref
                        desvio_pct = (
                            (desvio / _proy_ref * 100)
                            if _proy_ref > 0 else 0
                        )
                        ad = abs(desvio_pct)
                        if ad <= 5:
                            sem = "🟢"
                        elif ad <= 10:
                            sem = "🟡"
                        else:
                            sem = "🔴"
                        _real_str = f"{real_an:.2f}"
                        _desvio_kg_str = f"{desvio:+.2f}"
                        _desvio_pct_str = f"{desvio_pct:+.1f}%"
                        _dias_real_str = f"{_dias_r}"
                    else:
                        # Carga en curso — no se sabe el real
                        # todavía. Mostrar días transcurridos
                        # y días restantes estimados (según
                        # los días_estimados al cargar).
                        try:
                            d_ini = datetime.strptime(
                                ev["fecha"], "%Y-%m-%d"
                            ).date()
                            _dias_transcurridos = (
                                _hoy_fecha - d_ini
                            ).days
                        except Exception:
                            _dias_transcurridos = 0
                        # Días restantes según la estimación
                        # inicial. Puede ser negativo si la
                        # carga ya superó su duración estimada
                        # (señal de subconsumo o que falta
                        # registrar la siguiente carga).
                        _dias_restantes = (
                            _dias_est - _dias_transcurridos
                        )
                        # Fecha estimada de agotamiento
                        try:
                            _fecha_fin_est = (
                                d_ini + _td_h(days=int(_dias_est))
                            ).strftime("%d/%m")
                        except Exception:
                            _fecha_fin_est = "—"

                        sem = "⏳"
                        if _dias_restantes < 0:
                            _real_str = (
                                f"— (en curso, "
                                f"{_dias_transcurridos}d transc. · "
                                f"vencida hace {abs(_dias_restantes):.0f}d)"
                            )
                            _dias_real_str = (
                                f"⏳ {_dias_transcurridos}d · "
                                f"⚠️ vencida hace "
                                f"{abs(_dias_restantes):.0f}d"
                            )
                        else:
                            _real_str = (
                                f"— (en curso, "
                                f"{_dias_transcurridos}d transc. · "
                                f"~{_dias_restantes:.0f}d restantes)"
                            )
                            _dias_real_str = (
                                f"⏳ {_dias_transcurridos}d · "
                                f"faltan ~{_dias_restantes:.0f}d "
                                f"(hasta {_fecha_fin_est})"
                            )
                        _desvio_kg_str = "—"
                        _desvio_pct_str = "—"

                    # Período legible: usar días reales si está
                    # cerrado, sino los estimados con (?).
                    try:
                        d_ini = datetime.strptime(
                            ev["fecha"], "%Y-%m-%d"
                        ).date()
                        _dias_show = (
                            float(_dias_r)
                            if _dias_r is not None
                            else _dias_est
                        )
                        d_fin = d_ini + _td_h(
                            days=int(_dias_show)
                        )
                        _periodo = (
                            f"{d_ini.strftime('%d/%m')} → "
                            f"{d_fin.strftime('%d/%m')}"
                            + (" (est.)" if _en_curso else "")
                        )
                    except Exception:
                        _periodo = ev["fecha"]

                    _rows_tabla.append({
                        "Período": _periodo,
                        "Tipo": _label_tipo,
                        "Días estimados": f"{_dias_est:.1f}",
                        "Días reales": _dias_real_str,
                        "kg cargados (TC)":
                            f"{ev['kg_cargados']:.0f}",
                        "Real prom (kg MS/an/día)":
                            _real_str,
                        "Proy prom (kg MS/an/día)":
                            f"{_proy_ref:.2f}",
                        "Desvío (kg)": _desvio_kg_str,
                        "Desvío (%)": _desvio_pct_str,
                        "": sem,
                    })

                if _rows_tabla:
                    _df_tabla = pd.DataFrame(_rows_tabla)
                    st.dataframe(
                        _df_tabla, hide_index=True,
                        width="stretch",
                    )
                    st.caption(
                        "📝 **Cómo se lee esta tabla**: cada fila "
                        "es un evento de carga. El **Real prom** "
                        "solo se conoce cuando hay una **siguiente "
                        "carga del mismo tipo** — recién ahí "
                        "sabemos cuántos días duró realmente y "
                        "podemos calcular el consumo diario "
                        "promedio (kg ÷ cabezas ÷ días reales). "
                        "Si una carga está en curso (⏳), todavía "
                        "no se puede medir el real. El **Proy "
                        "prom** es el promedio del consumo "
                        "proyectado durante esos mismos días. "
                        "🟢 ±5% (alineado) · 🟡 ±10% (revisar) · "
                        "🔴 >10% (alerta) · ⏳ en curso. "
                        "Cuando el cliente avise una nueva carga, "
                        "la fila pasa de ⏳ a un semáforo según "
                        "el desvío real."
                    )

    # ---- Dietas asociadas ----
    if dietas:
        st.markdown(f"##### 🍽️ Dietas registradas ({len(dietas)})")
        for d in dietas:
            fecha_d = d.get("fecha", "")
            costo_d = d.get("costo_dia", 0) or 0
            pb_d = d.get("pb_pct", 0) or 0
            obs_d = d.get("observaciones") or ""
            label = (
                f"📅 {fecha_d}  ·  "
                f"💰 ${costo_d:.0f}/día  ·  "
                f"PB {pb_d:.1f}%"
                + (f"  ·  {obs_d[:30]}" if obs_d else "")
            )
            with st.expander(label):
                # ─── Selector de "fecha de evaluación" ───
                # La dieta guardada es estática (kg/an/día al momento
                # de formularla). Pero como el animal mete kilos, su
                # consumo SUBE proporcionalmente al PV (premisa: %PV
                # constante ≈ 2.5-3% PV). Acá dejamos elegir a qué
                # PV se quiere ver la dieta: inicio / HOY / fin de
                # ciclo. Las alertas/silocomedero/curva DMI ya usan
                # este escalado por debajo — esto solo lo HACE VISIBLE
                # en el bloque "Dietas registradas".
                from src.stock_producto import (
                    factor_escala_consumo_pv as _f_esc,
                    estimar_peso_vivo_lote as _est_pv,
                )
                from datetime import datetime as _dt_loc
                _adpv = float(lote.get("adpv_objetivo_kg") or 0)
                _fecha_fin = (
                    lote.get("objetivo_fecha")
                    or lote.get("fecha_salida_objetivo")
                    or ""
                )
                _hoy_iso = _dt_loc.now().strftime("%Y-%m-%d")
                _opts_fecha = [
                    ("Al inicio (fecha de la dieta)", fecha_d[:10]),
                    (f"HOY ({_hoy_iso})", _hoy_iso),
                ]
                if _fecha_fin and _fecha_fin > _hoy_iso:
                    _opts_fecha.append(
                        (f"Fin de ciclo ({_fecha_fin})", _fecha_fin),
                    )
                if _adpv > 0:
                    _label_eval = st.radio(
                        "📅 Ver dieta escalada a la fecha de:",
                        options=[o[0] for o in _opts_fecha],
                        index=1,  # default: HOY
                        horizontal=True,
                        key=f"dieta_eval_fecha_{d['id']}",
                        help=(
                            "El consumo escala con el peso vivo (~2.5-3% "
                            "del PV). A medida que el animal mete kilos, "
                            "kg/an/día sube proporcionalmente. La "
                            "composición % se mantiene."
                        ),
                    )
                    _fecha_eval = next(
                        o[1] for o in _opts_fecha if o[0] == _label_eval
                    )
                else:
                    st.info(
                        "ℹ️ Esta dieta no se está escalando por PV "
                        "porque el lote no tiene **ADG objetivo (kg/día)** "
                        "cargado. Editá el lote y poné el ADG para que "
                        "veas la dieta proyectada HOY y al fin de ciclo."
                    )
                    _fecha_eval = fecha_d[:10]

                # Calcular factor y peso a la fecha elegida
                _factor, _info_f = _f_esc(lote, d, _fecha_eval)
                _peso_eval = _info_f.get("peso_actual_kg", 0)
                _peso_ref = _info_f.get("peso_referencia_kg", 0)

                # Banner contextual
                if _factor != 1.0:
                    _delta_pct = (_factor - 1) * 100
                    _signo = "+" if _delta_pct >= 0 else ""
                    st.caption(
                        f"⚖️ PV proyectado a la fecha elegida: "
                        f"**{_peso_eval:.0f} kg** "
                        f"(vs **{_peso_ref:.0f} kg** al formularla) "
                        f"→ escalado **{_signo}{_delta_pct:.1f}%** "
                        f"sobre el consumo original."
                    )

                # Métricas escaladas
                _consumo_orig = float(d.get("consumo_ms_kg", 0) or 0)
                _em_orig = float(d.get("em_mcal_dia", 0) or 0)
                _costo_orig = float(costo_d or 0)
                _consumo_esc = _consumo_orig * _factor
                _em_esc = _em_orig * _factor
                _costo_esc = _costo_orig * _factor

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Costo / animal / día",
                    f"${_costo_esc:.2f}",
                    delta=(
                        f"{_costo_esc - _costo_orig:+.2f} vs inicial"
                        if _factor != 1.0 else None
                    ),
                )
                m2.metric(
                    "Consumo MS",
                    f"{_consumo_esc:.2f} kg",
                    delta=(
                        f"{_consumo_esc - _consumo_orig:+.2f} kg "
                        "vs inicial"
                        if _factor != 1.0 else None
                    ),
                )
                m3.metric(
                    "Energía Metab.",
                    f"{_em_esc:.1f} Mcal/día",
                    delta=(
                        f"{_em_esc - _em_orig:+.1f} Mcal vs inicial"
                        if _factor != 1.0 else None
                    ),
                )

                m4, m5 = st.columns(2)
                m4.metric("PB", f"{pb_d:.1f}% MS")
                m5.metric("NNP", f"{d.get('nnp_pct', 0):.2f}% MS")

                if obs_d:
                    st.caption(f"📝 {obs_d}")

                # Composición
                comp = d.get("composicion") or []
                if comp:
                    # Totales para calcular % sobre la ración (incluye
                    # rollos a discreción) y % sobre mezcla (excluye
                    # rollos — es lo que se carga al silo).
                    from src.stock_producto import (
                        _es_a_discrecion as _esd,
                    )
                    _tot_ms_racion = sum(
                        float(c.get("kg_ms") or 0) for c in comp
                    ) or 1.0
                    _tot_ms_mezcla = sum(
                        float(c.get("kg_ms") or 0) for c in comp
                        if not _esd(c.get("nombre", ""))
                    ) or 1.0
                    _rows_comp = []
                    for c in comp:
                        _kg_ms_c = float(c.get("kg_ms") or 0)
                        _kg_tc_c = float(c.get("kg_tal_cual") or 0)
                        _costo_c = float(c.get("costo_dia") or 0)
                        _es_disc = _esd(c.get("nombre", ""))
                        _pct_racion = _kg_ms_c / _tot_ms_racion * 100
                        _pct_mezcla = (
                            0 if _es_disc
                            else _kg_ms_c / _tot_ms_mezcla * 100
                        )
                        # Escalado por PV proyectado
                        _kg_ms_esc = _kg_ms_c * _factor
                        _kg_tc_esc = _kg_tc_c * _factor
                        _costo_esc_ing = _costo_c * _factor
                        _rows_comp.append({
                            "Ingrediente": c.get("nombre", "?"),
                            "% ración (sobre total)": (
                                f"{_pct_racion:.1f}%"
                            ),
                            "% mezcla (silo)": (
                                f"{_pct_mezcla:.1f}%"
                                if not _es_disc else "—"
                            ),
                            "kg MS/día": round(_kg_ms_esc, 2),
                            "kg t/c/día": round(_kg_tc_esc, 2),
                            "Costo $/día": round(_costo_esc_ing, 2),
                        })
                    df_comp = pd.DataFrame(_rows_comp)
                    st.dataframe(df_comp, hide_index=True,
                                 width="stretch")
                    st.caption(
                        "**% ración** = participación del ingrediente "
                        "en el DMI total (suma 100%). "
                        "**% mezcla** = participación sobre la mezcla "
                        "del silo (excluye forrajes a libre disposición "
                        "como rollo). "
                        + (
                            f"**kg/día** = escalados a PV "
                            f"**{_peso_eval:.0f} kg** "
                            f"(factor ×{_factor:.3f})."
                            if _factor != 1.0 else
                            "**kg/día** = al PV de formulación."
                        )
                    )

                # ─── Cuadro de evolución semanal ───
                # Muestra cómo cambia el consumo de cada ingrediente
                # (y el total) a lo largo del ciclo, a medida que el
                # animal mete kilos. Solo si hay ADG y fecha objetivo.
                _cant_anim = int(lote.get("cantidad_inicial") or 0)
                if _adpv > 0 and _fecha_fin and _cant_anim > 0:
                    with st.expander(
                        "📈 Evolución semanal de cantidades — "
                        "por animal y para todo el lote",
                        expanded=False,
                    ):
                        from datetime import timedelta as _td_loc
                        try:
                            _f_ini_loc = _dt_loc.strptime(
                                fecha_d[:10], "%Y-%m-%d"
                            ).date()
                            _f_fin_loc = _dt_loc.strptime(
                                _fecha_fin, "%Y-%m-%d"
                            ).date()
                        except Exception:
                            _f_ini_loc = None
                            _f_fin_loc = None

                        if _f_ini_loc and _f_fin_loc and _f_fin_loc > _f_ini_loc:
                            # Generar muestras semanales (incluyendo
                            # extremos: inicio + cada 7d + fin)
                            _muestras = []
                            _f_cur = _f_ini_loc
                            while _f_cur < _f_fin_loc:
                                _muestras.append(_f_cur)
                                _f_cur = _f_cur + _td_loc(days=7)
                            _muestras.append(_f_fin_loc)

                            # Por cada fecha, calcular factor y kg
                            _nombres_ing = [
                                c.get("nombre", "?") for c in comp
                            ]
                            _kg_tc_base = [
                                float(c.get("kg_tal_cual") or 0)
                                for c in comp
                            ]
                            _es_disc_ing = [
                                _esd(n) for n in _nombres_ing
                            ]

                            _rows_evo = []
                            for _fm in _muestras:
                                _fm_iso = _fm.isoformat()
                                _fact_m, _info_m = _f_esc(
                                    lote, d, _fm_iso,
                                )
                                _pv_m = _info_m.get(
                                    "peso_actual_kg", 0,
                                )
                                _dias_m = (_fm - _f_ini_loc).days
                                _fila = {
                                    "Fecha": _fm.strftime(
                                        "%d/%m/%y"
                                    ),
                                    "Días": _dias_m,
                                    "PV (kg)": f"{_pv_m:.0f}",
                                }
                                # Por ingrediente: kg t/c por animal
                                _total_an_tc = 0.0
                                _total_an_mezcla = 0.0
                                for _i, (_n, _b, _disc) in enumerate(
                                    zip(_nombres_ing, _kg_tc_base,
                                        _es_disc_ing)
                                ):
                                    _kg_an = _b * _fact_m
                                    _fila[f"{_n} (kg/an)"] = round(
                                        _kg_an, 2,
                                    )
                                    _total_an_tc += _kg_an
                                    if not _disc:
                                        _total_an_mezcla += _kg_an
                                # Totales por animal y por lote
                                _fila["Total ración (kg/an)"] = round(
                                    _total_an_tc, 2,
                                )
                                _fila["Mezcla silo (kg/an)"] = round(
                                    _total_an_mezcla, 2,
                                )
                                _fila["Mezcla LOTE (kg/día)"] = round(
                                    _total_an_mezcla * _cant_anim, 1,
                                )
                                _rows_evo.append(_fila)
                            _df_evo = pd.DataFrame(_rows_evo)
                            st.dataframe(
                                _df_evo, hide_index=True,
                                width="stretch",
                            )
                            st.caption(
                                f"📋 Muestras cada 7 días desde "
                                f"{_f_ini_loc.strftime('%d/%m/%y')} "
                                f"hasta {_f_fin_loc.strftime('%d/%m/%y')} "
                                f"({(_f_fin_loc - _f_ini_loc).days} "
                                f"días totales). "
                                f"PV escalado con ADG **{_adpv} kg/día**. "
                                f"Lote: **{_cant_anim} animales**. "
                                "**kg/an** = por animal por día · "
                                "**Total ración** = todos los "
                                "ingredientes (incluye libre "
                                "disposición) · "
                                "**Mezcla silo** = solo lo que va al "
                                "comedero (sin rollo) · "
                                "**Mezcla LOTE** = lo que tenés que "
                                "preparar por día para todo el lote."
                            )

                            # ─── Totales acumulados al fin de ciclo ───
                            _dias_ciclo = (
                                _f_fin_loc - _f_ini_loc
                            ).days
                            # Aproximación: integral del consumo
                            # (factor lineal entre fechas) ≈ promedio
                            # de factores × días.
                            _factores_ms = [
                                _f_esc(
                                    lote, d,
                                    (_f_ini_loc + _td_loc(days=k))
                                    .isoformat(),
                                )[0]
                                for k in range(0, _dias_ciclo + 1, 7)
                            ]
                            _factor_prom = (
                                sum(_factores_ms) / len(_factores_ms)
                                if _factores_ms else 1.0
                            )
                            st.markdown(
                                "##### 📦 Demanda acumulada del ciclo"
                            )
                            _rows_tot = []
                            for _n, _b, _disc in zip(
                                _nombres_ing, _kg_tc_base, _es_disc_ing
                            ):
                                _kg_total_lote = (
                                    _b * _factor_prom
                                    * _dias_ciclo * _cant_anim
                                )
                                _rows_tot.append({
                                    "Ingrediente": _n,
                                    "Tipo": (
                                        "Rollo / libre"
                                        if _disc else "Mezcla silo"
                                    ),
                                    "kg t/c TOTAL ciclo": round(
                                        _kg_total_lote, 0,
                                    ),
                                    "Toneladas": (
                                        f"{_kg_total_lote/1000:.2f} t"
                                    ),
                                })
                            _df_tot = pd.DataFrame(_rows_tot)
                            st.dataframe(
                                _df_tot, hide_index=True,
                                width="stretch",
                            )
                            st.caption(
                                f"💡 Estimación de **demanda total** "
                                f"para los **{_dias_ciclo} días** "
                                f"del ciclo, considerando el aumento "
                                f"progresivo del consumo por ganancia "
                                f"de peso. Te sirve para planificar "
                                f"compras."
                            )
                        else:
                            st.warning(
                                "⚠️ No puedo calcular evolución: "
                                "faltan fechas o son incoherentes."
                            )

                # Acciones
                col_d1, col_d2, col_d3 = st.columns(3)
                if col_d1.button("🔄 Retomar esta dieta",
                                  key=f"retomar_{d['id']}",
                                  help="Carga esta receta en la "
                                       "pestaña Dieta para modificarla"):
                    # Guardamos la receta en session_state para que
                    # la pestaña Dieta la cargue al abrir
                    st.session_state["receta_porcentajes"] = {
                        c.get("nombre", ""): c.get("pct_ms", 0)
                        for c in comp
                    }
                    st.success(
                        "✅ Receta cargada. Andá a **🍽️ Dieta** → "
                        "expander 'Verificar mi receta' para verla."
                    )

                if col_d2.button("📋 Copiar como texto",
                                  key=f"copy_{d['id']}"):
                    txt = f"Dieta del {fecha_d}\n"
                    txt += f"Costo: ${costo_d:.2f}/animal/día\n"
                    txt += f"PB: {pb_d:.1f}%  EM: {d.get('em_mcal_dia',0):.1f}\n\n"
                    for c in comp:
                        txt += (
                            f"  • {c.get('nombre','?')}: "
                            f"{c.get('pct_ms',0):.1f}% "
                            f"({c.get('kg_tal_cual',0):.2f} kg t/c/día)\n"
                        )
                    st.code(txt, language="text")

    # ---- Carga del comedero (silo o lineal) + comparación ----
    # Se muestra para CUALQUIER tipo de comedero. Para silocomedero
    # se mantiene la proyección de fin de carga. Para lineal_diario
    # cada entrada cubre 1 día. En ambos casos el sistema compara
    # lo cargado contra lo que la dieta formulada pide (alerta si
    # hay sub-uso o sobre-uso).
    _tipo_com = (
        lote.get("tipo_comedero_concentrado") or ""
    ).lower()
    if _tipo_com:
        from src import stock_producto as sp
        _es_silo = (_tipo_com == "silocomedero")
        _titulo_blk = (
            "🛢️ Cargas del silocomedero"
            if _es_silo
            else "🍽️ Cargas del comedero lineal"
        )
        st.markdown(f"##### {_titulo_blk}")
        if _es_silo:
            st.caption(
                "Registrá cada vez que cargues mezcla en el "
                "silo. El sistema proyecta cuándo se agota "
                "usando el consumo diario de la dieta vigente "
                "× cantidad de animales. Te avisa por email/"
                "WhatsApp 1 día antes de que se termine. "
                "Además compara lo cargado con lo que la dieta "
                "pide para detectar sub-uso o sobre-uso."
            )
        else:
            st.caption(
                "Registrá cada día lo que cargás al comedero. "
                "El sistema compara lo cargado contra lo que "
                "la dieta formulada pide y te marca si estás "
                "por encima o por debajo del plan."
            )

        # KPIs de la carga actual (solo silocomedero)
        if _es_silo:
            proy = sp.proyectar_fin_carga_silocomedero(
                lote_id_sel
            )
            if proy:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(
                    "Última carga",
                    f"{proy['kg_cargados']:.0f} kg",
                    proy["fecha_carga"],
                )
                k2.metric(
                    "Consumo / día",
                    f"{proy['consumo_diario_kg']:.0f} kg",
                )
                k3.metric(
                    "Restan en silo",
                    f"{proy['kg_restantes']:.0f} kg",
                )
                dr = proy["dias_restantes"]
                k4.metric(
                    "Se agota",
                    f"{dr} día{'s' if dr != 1 else ''}",
                    proy["fecha_agotamiento"],
                )
                if dr <= 0:
                    st.error(
                        "🔴 La carga ya se agotó — hay que "
                        "preparar mezcla nueva hoy."
                    )
                elif dr == 1:
                    st.warning(
                        "🟠 Mañana se termina la carga — "
                        "preparar la próxima mezcla."
                    )
            else:
                st.info(
                    "No hay carga registrada todavía. Cargá "
                    "la primera abajo y el sistema empieza a "
                    "proyectar el fin de carga."
                )
        if not db.listar_dietas(lote_id_sel):
            st.caption(
                "⚠️ No hay dieta cargada para el lote — sin "
                "dieta no se puede comparar la carga real "
                "contra el plan."
            )

        # ---------- FORM: registrar carga ----------
        # Obtener ingredientes de la dieta vigente (para
        # el editor por ingrediente).
        _dietas_lote = db.listar_dietas(lote_id_sel)
        from src.stock_producto import _dieta_vigente
        _hoy_iso = datetime.now().strftime("%Y-%m-%d")
        _dieta_act = (
            _dieta_vigente(_dietas_lote, _hoy_iso)
            if _dietas_lote else None
        )
        _comp_act = (
            _dieta_act.get("composicion") or []
        ) if _dieta_act else []

        # Selector del modo FUERA del form para que se actualice
        # al cambiar (con st.form los widgets internos no
        # disparan rerun hasta el submit).
        _modo_carga = st.radio(
            "¿Cómo querés cargar?",
            [
                "Total (un solo kg)",
                "Por ingrediente (desglose)"
            ],
            horizontal=True,
            key=f"modo_carga_{lote_id_sel}",
            help=(
                "Total = ingresás los kg de mezcla totales. "
                "Por ingrediente = ingresás cuántos kg de cada "
                "ingrediente (mejor para auditar dónde está el "
                "desvío)."
            ),
        )
        _modo_total = _modo_carga.startswith("Total")

        # Consumo diario teórico (kg/día tal cual, excluyendo libre
        # disposición) según la dieta vigente × cantidad vigente del
        # lote. Se usa para calcular automáticamente "días que cubre"
        # una carga del silocomedero cuando es modo por ingrediente.
        _consumo_diario_teorico = 0.0
        if _comp_act:
            _cant_anim_act = (
                db.cantidad_vigente_lote(lote_id_sel, _hoy_iso) or 0
            )
            for _ing in _comp_act:
                _nom_ing = (_ing.get("nombre") or "").strip()
                if not _nom_ing or sp._es_a_discrecion(_nom_ing):
                    continue
                _consumo_diario_teorico += (
                    float(_ing.get("kg_tal_cual") or 0)
                    * _cant_anim_act
                )

        # ¿Calculamos los días automáticamente?
        # Sí: silocomedero + hay dieta vigente con consumo > 0.
        _dias_auto = bool(
            _es_silo and _consumo_diario_teorico > 0
        )

        with st.expander(
            "📝 Registrar nueva carga al silo",
            expanded=False,
        ):
            with st.form(
                f"nueva_carga_silo_{lote_id_sel}",
                clear_on_submit=False,
            ):
                _cfa, _cfb, _cfc = st.columns([2, 1, 1])
                with _cfa:
                    carga_fecha = st.date_input(
                        "Fecha de carga",
                        value=datetime.now().date(),
                        key=f"carga_fecha_{lote_id_sel}",
                    )
                with _cfb:
                    from datetime import time as _time_cls
                    _hora_def = datetime.now().time().replace(
                        second=0, microsecond=0
                    )
                    carga_hora = st.time_input(
                        "Hora",
                        value=_hora_def,
                        key=f"carga_hora_{lote_id_sel}",
                        help=(
                            "Hora exacta de la carga. Si fueron 2 "
                            "comidas en el día, registrá cada una "
                            "con su hora."
                        ),
                    )
                with _cfc:
                    if _es_silo and not _dias_auto:
                        # Solo manual si NO hay dieta para calcular.
                        carga_dias = st.number_input(
                            "Días que cubre",
                            min_value=1.0, max_value=30.0,
                            value=5.0, step=1.0,
                            key=f"carga_dias_{lote_id_sel}",
                            help=(
                                "Cargá una dieta vigente y los días se "
                                "calculan solos a partir del consumo."
                            ),
                        )
                    elif _es_silo and _dias_auto:
                        # Calculado automáticamente del consumo diario.
                        carga_dias = None   # se resuelve en el submit
                        st.markdown(
                            "**Días que cubre**"
                        )
                        st.caption(
                            f"_Cálculo automático según consumo de "
                            f"**{_consumo_diario_teorico:.0f} kg/día** "
                            f"de la dieta vigente._"
                        )
                    else:
                        carga_dias = 1.0
                        st.caption(
                            "_Lineal: 1 comida_"
                        )

                desglose_para_guardar = None
                if _modo_total:
                    carga_kg = st.number_input(
                        "Kg de mezcla cargados (total)",
                        min_value=1.0, max_value=100000.0,
                        value=100.0, step=10.0,
                        key=f"carga_kg_{lote_id_sel}",
                    )
                    # Si hay dieta vigente y es silocomedero, mostrar
                    # también el cálculo de días estimados acá (igual que
                    # en modo Por ingrediente).
                    if (
                        _dias_auto and carga_kg > 0
                        and _consumo_diario_teorico > 0
                    ):
                        _dias_est_total = (
                            carga_kg / _consumo_diario_teorico
                        )
                        st.caption(
                            f"**{carga_kg:.0f} kg** cubren "
                            f"**~{_dias_est_total:.1f} días** según la "
                            f"dieta vigente "
                            f"({_consumo_diario_teorico:.0f} kg/día)."
                        )
                else:
                    if not _comp_act:
                        st.warning(
                            "No hay dieta vigente — no puedo armar "
                            "el editor por ingrediente. Cargá una "
                            "dieta primero o usá modo Total."
                        )
                        carga_kg = 0
                    else:
                        st.caption(
                            "Ingresá cuántos kg de **cada "
                            "ingrediente** cargaste hoy:"
                        )
                        _filas_ed = []
                        for _ing in _comp_act:
                            _nom = (_ing.get("nombre") or "").strip()
                            if not _nom:
                                continue
                            _filas_ed.append({
                                "Ingrediente": _nom,
                                "kg cargados": 0.0,
                            })
                        _df_ed = pd.DataFrame(_filas_ed)
                        _df_edit = st.data_editor(
                            _df_ed,
                            hide_index=True,
                            width="stretch",
                            disabled=["Ingrediente"],
                            key=f"editor_desglose_{lote_id_sel}",
                        )
                        desglose_para_guardar = [
                            {"nombre": r["Ingrediente"],
                             "kg": float(r["kg cargados"] or 0)}
                            for _, r in _df_edit.iterrows()
                            if float(r["kg cargados"] or 0) > 0
                        ]
                        carga_kg = sum(
                            d["kg"] for d in desglose_para_guardar
                        )
                        # Línea de info con total + días estimados si es
                        # silocomedero y hay dieta vigente para calcular.
                        if (
                            _dias_auto and carga_kg > 0
                            and _consumo_diario_teorico > 0
                        ):
                            _dias_est = (
                                carga_kg / _consumo_diario_teorico
                            )
                            st.caption(
                                f"Total ingresado: "
                                f"**{carga_kg:.1f} kg** → cubre "
                                f"**~{_dias_est:.1f} días** según la "
                                f"dieta vigente "
                                f"({_consumo_diario_teorico:.0f} kg/día)."
                            )
                        else:
                            st.caption(
                                f"Total ingresado: "
                                f"**{carga_kg:.1f} kg**"
                            )

                carga_obs = st.text_input(
                    "Observaciones (opcional)",
                    key=f"carga_obs_{lote_id_sel}",
                )
                btn_carga = st.form_submit_button(
                    "💾 Registrar carga", type="primary",
                )
                if btn_carga:
                    try:
                        if carga_kg <= 0:
                            st.error(
                                "❌ Tenés que cargar al menos 1 kg."
                            )
                        else:
                            _hora_str = carga_hora.strftime("%H:%M")
                            # Si carga_dias quedó en None (modo auto del
                            # silocomedero), calcular ahora con la carga
                            # real ingresada.
                            if carga_dias is None:
                                if _consumo_diario_teorico > 0:
                                    _dias_final = round(
                                        carga_kg
                                        / _consumo_diario_teorico,
                                        2,
                                    )
                                    # Mínimo 1 día por sanidad de la DB.
                                    _dias_final = max(_dias_final, 1.0)
                                else:
                                    _dias_final = 1.0
                            else:
                                _dias_final = float(carga_dias)
                            db.crear_carga_silocomedero(
                                lote_id=lote_id_sel,
                                fecha_carga=(
                                    carga_fecha.isoformat()
                                ),
                                kg_cargados=float(carga_kg),
                                detalles=carga_obs or "",
                                tipo_carga=(
                                    "silo_carga" if _es_silo
                                    else "lineal_diario"
                                ),
                                desglose_ingredientes=(
                                    desglose_para_guardar
                                ),
                                dias_cubiertos=_dias_final,
                                hora_carga=_hora_str,
                            )
                            st.success(
                                f"✅ Carga de {carga_kg:.0f} kg "
                                f"registrada el "
                                f"{carga_fecha.isoformat()}."
                            )
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # ---------- FORM ENTREGA DE ROLLO (libre disposición) ---------
        # Si el lote tiene modalidad de forraje "aparte", el rollo se
        # entrega a libre disposición y se trackea por separado.
        # Calculadora cilíndrica + densidades INTA + % desperdicio.
        if (l_data.get("forraje_modalidad") or "mezclado") == "aparte":
            # Título visible — el sub-expander solo envuelve el form
            st.markdown(
                "##### 🌾 Rollo a libre disposición"
            )
            st.caption(
                "Registrá cada vez que repongas rollos. El sistema "
                "proyecta cuándo se agota usando el consumo "
                "diario de rollo de la dieta vigente × cantidad "
                "de animales. Te avisa cuando se queda corto."
            )

            # ── KPIs equivalentes a los del silocomedero ──
            # Calculamos la última entrega de rollo y su autonomía
            # estimada, para mostrar 4 KPIs en línea como en el silo.
            from src.stock_producto import (
                serie_cargas_rollo_lote as _serie_rollo_kpi,
                _es_a_discrecion as _esd_kpi,
                _dieta_vigente as _div_kpi,
            )
            _entregas_rollo = _serie_rollo_kpi(lote_id_sel)
            _ult_rollo = (
                _entregas_rollo[-1] if _entregas_rollo else None
            )
            # Consumo proyectado de rollo del lote (kg MS/día) =
            # suma de kg_ms de ingredientes a discreción de la
            # dieta vigente × cabezas.
            _dietas_kpi = db.listar_dietas(lote_id_sel) or []
            _d_vig_kpi = (
                _div_kpi(_dietas_kpi, datetime.now().strftime("%Y-%m-%d"))
                if _dietas_kpi else None
            ) or (_dietas_kpi[-1] if _dietas_kpi else None)
            _cons_rollo_an_dia = 0.0
            if _d_vig_kpi:
                for _c in (_d_vig_kpi.get("composicion") or []):
                    if _esd_kpi(_c.get("nombre", "")):
                        _cons_rollo_an_dia += float(
                            _c.get("kg_ms") or 0
                        )
            _cabezas_kpi = (
                db.cantidad_vigente_lote(lote_id_sel) or 0
            )
            _cons_rollo_lote_dia = (
                _cons_rollo_an_dia * _cabezas_kpi
            )

            kr1, kr2, kr3, kr4 = st.columns(4)
            if _ult_rollo:
                kr1.metric(
                    "Última entrega",
                    f"{_ult_rollo['kg_cargados_tal_cual']:.0f} kg",
                    f"{_ult_rollo['fecha']}",
                )
                # Consumo MS por lote por día (en kg MS, como
                # mostramos el silo en kg TC del concentrado)
                kr2.metric(
                    "Consumo / día",
                    f"{_cons_rollo_lote_dia:.0f} kg MS",
                    f"{_cons_rollo_an_dia:.2f} kg MS/an"
                    if _cons_rollo_an_dia > 0 else None,
                )
                # Cuánto queda hoy del rollo (MS aprovechado restante)
                try:
                    _f_ult = datetime.strptime(
                        _ult_rollo['fecha'], "%Y-%m-%d"
                    ).date()
                    _dias_transc_kpi = max(
                        0, (datetime.now().date() - _f_ult).days
                    )
                except Exception:
                    _dias_transc_kpi = 0
                _kg_ms_total_inicial = float(
                    _ult_rollo.get("kg_ms_aprovechado") or 0
                )
                _kg_ms_consumido = (
                    _cons_rollo_lote_dia * _dias_transc_kpi
                )
                _kg_ms_restante = max(
                    0, _kg_ms_total_inicial - _kg_ms_consumido,
                )
                kr3.metric(
                    "Restan en rollo",
                    f"{_kg_ms_restante:.0f} kg MS",
                    f"{_dias_transc_kpi}d transcurridos",
                )
                # Días hasta agotamiento
                if (_cons_rollo_lote_dia > 0
                        and _kg_ms_restante > 0):
                    _dias_restantes_kpi = (
                        _kg_ms_restante / _cons_rollo_lote_dia
                    )
                    from datetime import timedelta as _td_kpi
                    _fecha_agot_kpi = (
                        datetime.now().date()
                        + _td_kpi(days=int(_dias_restantes_kpi))
                    ).strftime("%Y-%m-%d")
                    kr4.metric(
                        "Se agota",
                        f"{_dias_restantes_kpi:.1f} días",
                        f"{_fecha_agot_kpi}",
                    )
                else:
                    kr4.metric(
                        "Se agota",
                        "0 días",
                        "⚠️ hay que reponer",
                    )
            else:
                kr1.metric("Última entrega", "—")
                kr2.metric(
                    "Consumo / día",
                    f"{_cons_rollo_lote_dia:.0f} kg MS"
                    if _cons_rollo_lote_dia > 0 else "—",
                )
                kr3.metric("Restan en rollo", "—")
                kr4.metric("Se agota", "—")
                st.info(
                    "Todavía no hay entregas de rollo registradas. "
                    "Cargá la primera con el botón de abajo."
                )

            with st.expander(
                "📝 Registrar nueva entrega de rollo",
                expanded=False,
            ):
                from src.stock_producto import (
                    calcular_peso_rollo,
                    DENSIDAD_ROLLO_KG_M3,
                    PCT_MS_ROLLO,
                    DESPERDICIO_ROLLO_DEFAULT,
                )

                _tipos_disp = list(DENSIDAD_ROLLO_KG_M3.keys())
                _tipos_label = {
                    "alfalfa": "Alfalfa pura",
                    "pastura_consociada": "Pastura consociada",
                    "avena": "Avena",
                    "moha": "Moha",
                    "sorgo_diferido": "Sorgo diferido",
                    "pastura_natural": "Pastura natural",
                    "cebadilla": "Cebadilla",
                    "trigo": "Trigo / cebada heno",
                    "mezcla": "Mezcla (default)",
                }

                # Calcular consumo proyectado de rollo del lote, sacado
                # de los ingredientes "a libre disposición" de la dieta
                # vigente. Eso nos permite estimar AUTOMÁTICAMENTE
                # cuántos días va a durar la entrega.
                from src.stock_producto import (
                    _es_a_discrecion as _esd,
                    _dieta_vigente,
                )
                _dietas_lote = db.listar_dietas(lote_id_sel) or []
                _hoy_iso = datetime.now().strftime("%Y-%m-%d")
                _d_vig = (
                    _dieta_vigente(_dietas_lote, _hoy_iso)
                    if _dietas_lote else None
                ) or (_dietas_lote[-1] if _dietas_lote else None)
                _cons_rollo_proy_kg_ms_an = 0.0
                if _d_vig:
                    for _c in (_d_vig.get("composicion") or []):
                        if _esd(_c.get("nombre", "")):
                            _cons_rollo_proy_kg_ms_an += float(
                                _c.get("kg_ms") or 0
                            )

                # Inputs estáticos del rollo (tipo, dimensiones,
                # sistema de oferta) en sub-expander porque
                # casi nunca cambian. La fecha + observaciones
                # + KPIs + botón quedan visibles abajo.
                with st.expander(
                    "🔧 Datos del rollo (tipo, dimensiones, sistema)",
                    expanded=False,
                ):
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        rollo_tipo = st.selectbox(
                            "Tipo de forraje",
                            _tipos_disp,
                            format_func=lambda x: _tipos_label.get(x, x),
                            index=0,
                            key=f"rollo_tipo_{lote_id_sel}",
                        )
                        rollo_cant = st.number_input(
                            "Cantidad de rollos",
                            min_value=1, max_value=100, value=1, step=1,
                            key=f"rollo_cant_{lote_id_sel}",
                        )
                    with rc2:
                        rollo_diam = st.number_input(
                            "Diámetro (m)",
                            min_value=0.8, max_value=2.5, value=1.5,
                            step=0.05,
                            key=f"rollo_diam_{lote_id_sel}",
                        )
                        rollo_ancho = st.number_input(
                            "Ancho (m)",
                            min_value=0.8, max_value=2.5, value=1.5,
                            step=0.05,
                            key=f"rollo_ancho_{lote_id_sel}",
                        )
                    with rc3:
                        _dens_def = DENSIDAD_ROLLO_KG_M3.get(
                            rollo_tipo, 145
                        )
                        rollo_dens = st.number_input(
                            "Densidad (kg/m³)",
                            min_value=80, max_value=300,
                            value=int(_dens_def), step=5,
                            key=f"rollo_dens_{lote_id_sel}",
                            help=(
                                "Ajustá si tu zona/enrolladora suele dar "
                                "rollos más densos o más sueltos."
                            ),
                        )
                        rollo_modo = st.selectbox(
                            "Sistema de oferta",
                            ["parrillon_circular", "sin_parrilla",
                             "comedero_con_barrera"],
                            format_func=lambda x: {
                                "sin_parrilla":
                                    "Sin cubre rollo / piso",
                                "parrillon_circular":
                                    "Cubre rollo",
                                "comedero_con_barrera":
                                    "Comedero con barrera",
                            }.get(x, x),
                            index=0,  # Default: cubre rollo (lo común
                                      # en zona de Pampa Húmeda)
                            key=f"rollo_modo_{lote_id_sel}",
                        )

                    # % MS y % desperdicio quedan AUTOMÁTICOS por default.
                    # Si el usuario quiere ajustar (raro), abre avanzado.
                    _ms_std = float(PCT_MS_ROLLO.get(rollo_tipo, 88))
                    _despe_std = float(
                        DESPERDICIO_ROLLO_DEFAULT.get(rollo_modo, 25)
                    )

                    with st.expander(
                        "🔧 Ajustes avanzados (opcional)",
                        expanded=False,
                    ):
                        st.caption(
                            f"Los valores típicos para "
                            f"{_tipos_label.get(rollo_tipo, rollo_tipo)} "
                            f"+ {rollo_modo.replace('_', ' ')} son "
                            f"**{_ms_std:.0f}% MS** y "
                            f"**{_despe_std:.0f}% desperdicio**. "
                            f"Solo ajustá si tenés análisis específico "
                            f"o sabés que tu situación difiere."
                        )
                        rcA1, rcA2 = st.columns(2)
                        with rcA1:
                            # Key incluye tipo para forzar recreación
                            # del widget cuando cambia el tipo de forraje
                            # — sino Streamlit retiene el valor viejo.
                            rollo_ms = st.number_input(
                                "% Materia seca",
                                min_value=60.0, max_value=95.0,
                                value=_ms_std, step=0.5,
                                key=(
                                    f"rollo_ms_{lote_id_sel}_"
                                    f"{rollo_tipo}"
                                ),
                            )
                        with rcA2:
                            # Key incluye modo para que al cambiar el
                            # sistema de oferta, el default de
                            # desperdicio se actualice (sino queda
                            # cacheado en el valor anterior).
                            rollo_despe = st.number_input(
                                "% Desperdicio",
                                min_value=0.0, max_value=50.0,
                                value=_despe_std, step=1.0,
                                key=(
                                    f"rollo_despe_{lote_id_sel}_"
                                    f"{rollo_modo}"
                                ),
                            )

                # Cálculo del peso del rollo y MS aprovechable
                _peso = calcular_peso_rollo(
                    diametro_m=float(rollo_diam),
                    ancho_m=float(rollo_ancho),
                    tipo_forraje=rollo_tipo,
                    densidad_kg_m3=float(rollo_dens),
                    pct_ms=float(rollo_ms),
                )
                _kg_tc_unit = _peso.get("peso_tal_cual_kg", 0)
                _kg_tc_total = _kg_tc_unit * rollo_cant
                _kg_ms_total = _kg_tc_total * (rollo_ms / 100)
                _kg_ms_aprov = _kg_ms_total * (1 - rollo_despe / 100)
                _cabezas = db.cantidad_vigente_lote(lote_id_sel)

                # DÍAS AUTOMÁTICOS:
                # dias = kg_ms_aprovechado / (cabezas × consumo_proy_rollo)
                # Si no hay consumo proyectado de rollo en la dieta
                # vigente, asumimos 1.5 kg MS/animal/día como fallback
                # razonable para feedlot estándar.
                _cons_rollo_eff = (
                    _cons_rollo_proy_kg_ms_an
                    if _cons_rollo_proy_kg_ms_an > 0 else 1.5
                )
                if _cabezas > 0 and _cons_rollo_eff > 0:
                    _dias_auto = (
                        _kg_ms_aprov / _cabezas / _cons_rollo_eff
                    )
                else:
                    _dias_auto = 14.0
                _dias_auto = max(1.0, round(_dias_auto, 1))

                rc7, rc8, rc9 = st.columns(3)
                with rc7:
                    rollo_fecha = st.date_input(
                        "Fecha de la entrega",
                        value=datetime.now().date(),
                        key=f"rollo_fecha_{lote_id_sel}",
                    )
                with rc8:
                    rollo_dias_override = st.checkbox(
                        "Forzar duración manual",
                        value=False,
                        key=f"rollo_dias_ovr_{lote_id_sel}",
                        help=(
                            "Por default el sistema calcula los días "
                            "según el consumo proyectado de rollo de "
                            "la dieta vigente del lote."
                        ),
                    )
                with rc9:
                    if rollo_dias_override:
                        rollo_dias_cub = st.number_input(
                            "Días que va a durar",
                            min_value=1.0, max_value=60.0,
                            value=float(_dias_auto), step=1.0,
                            key=f"rollo_dias_{lote_id_sel}",
                        )
                    else:
                        rollo_dias_cub = _dias_auto
                        st.metric(
                            "Días estimados",
                            f"{_dias_auto:.1f} días",
                            help=(
                                f"Consumo proyectado rollo: "
                                f"{_cons_rollo_eff:.2f} kg MS/an/día "
                                f"({'de la dieta' if _cons_rollo_proy_kg_ms_an > 0 else 'default 1.5'})"
                            ),
                        )

                rollo_obs = st.text_input(
                    "Observaciones",
                    value="",
                    key=f"rollo_obs_{lote_id_sel}",
                )

                _ms_an_dia = (
                    _kg_ms_aprov / _cabezas / rollo_dias_cub
                    if _cabezas > 0 and rollo_dias_cub > 0 else 0
                )

                k1, k2, k3, k4 = st.columns(4)
                k1.metric(
                    "Peso por rollo",
                    f"{_kg_tc_unit:.0f} kg",
                    f"rango {_peso.get('rango_tc_kg_min', 0):.0f}–"
                    f"{_peso.get('rango_tc_kg_max', 0):.0f} kg",
                )
                k2.metric(
                    "Total tal cual",
                    f"{_kg_tc_total:.0f} kg",
                    f"{rollo_cant} rollos",
                )
                k3.metric(
                    "MS aprovechado",
                    f"{_kg_ms_aprov:.0f} kg",
                    f"{_ms_std:.0f}% MS · -{rollo_despe:.0f}% desp.",
                )
                k4.metric(
                    "MS / animal / día",
                    f"{_ms_an_dia:.2f} kg",
                    f"{_cabezas} cab. × {rollo_dias_cub:.1f} días",
                )

                if st.button(
                    "💾 Registrar entrega de rollo",
                    type="primary",
                    key=f"btn_rollo_{lote_id_sel}",
                ):
                    try:
                        # Metadata del rollo va en desglose
                        # (reusamos la estructura existente)
                        _desg_rollo = [{
                            "nombre": (
                                f"Rollo {_tipos_label.get(rollo_tipo, rollo_tipo)}"
                            ),
                            "kg": float(_kg_tc_total),
                            "tipo_forraje": rollo_tipo,
                            "cantidad_rollos": int(rollo_cant),
                            "diametro_m": float(rollo_diam),
                            "ancho_m": float(rollo_ancho),
                            "densidad_kg_m3": float(rollo_dens),
                            "pct_ms": float(rollo_ms),
                            "desperdicio_pct": float(rollo_despe),
                            "sistema_oferta": rollo_modo,
                            "peso_unitario_kg": float(_kg_tc_unit),
                        }]
                        db.crear_carga_silocomedero(
                            lote_id=lote_id_sel,
                            fecha_carga=rollo_fecha.isoformat(),
                            kg_cargados=float(_kg_tc_total),
                            detalles=rollo_obs or "",
                            tipo_carga="rollo_libre",
                            desglose_ingredientes=_desg_rollo,
                            dias_cubiertos=float(rollo_dias_cub),
                            hora_carga=None,
                        )
                        st.success(
                            f"✅ Entrega de {rollo_cant} rollos "
                            f"({_kg_tc_total:.0f} kg tal cual / "
                            f"{_kg_ms_aprov:.0f} kg MS aprovechable) "
                            f"registrada el {rollo_fecha.isoformat()}."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

                st.caption(
                    "Fórmula: V = π × (D/2)² × ancho · "
                    "Densidades de referencia INTA Anguil/Manfredi · "
                    "Las densidades son orientativas — pueden variar "
                    "±15% según humedad, prensa de enrolladora y "
                    "estado de conservación. Si conocés un valor "
                    "más preciso de tu zona, ajustalo arriba."
                )

                # ── Historial de entregas de rollo con editar/borrar ──
            _todas_cargas_rollo = (
                db.listar_cargas_silocomedero(
                    lote_id_sel, limit=60,
                ) or []
            )
            _rollos_registrados = [
                c for c in _todas_cargas_rollo
                if (c.get("tipo_carga") or "").lower()
                == "rollo_libre"
            ]
            if _rollos_registrados:
                st.markdown("---")
                st.markdown(
                    "**📋 Entregas de rollo registradas**"
                )
                for _cr in _rollos_registrados:
                    _meta = (
                        (_cr.get("desglose_ingredientes") or [{}])[0]
                    )
                    _tipo_label_show = _tipos_label.get(
                        _meta.get("tipo_forraje", "mezcla"),
                        _meta.get("tipo_forraje", "—"),
                    )
                    _fecha_show = (
                        _cr.get("fecha_carga") or ""
                    )[:10]
                    _cant_show = int(
                        _meta.get("cantidad_rollos") or 0
                    )
                    _kg_show = float(
                        _cr.get("kg_cargados") or 0
                    )
                    _desp_show = float(
                        _meta.get("desperdicio_pct") or 25
                    )
                    _dias_show = float(
                        _cr.get("dias_cubiertos") or 1
                    )
                    _sist_show = _meta.get(
                        "sistema_oferta", "sin_parrilla",
                    )
                    _sist_label = {
                        "sin_parrilla":
                            "Sin cubre rollo / piso",
                        "parrillon_circular":
                            "Cubre rollo",
                        "comedero_con_barrera":
                            "Comedero con barrera",
                    }.get(_sist_show, _sist_show)

                    _rc_a, _rc_b, _rc_c = st.columns([4, 1, 1])
                    with _rc_a:
                        st.write(
                            f"📅 **{_fecha_show}** · "
                            f"{_cant_show} × "
                            f"{_tipo_label_show} = "
                            f"{_kg_show:.0f} kg TC · "
                            f"{_sist_label} "
                            f"({_desp_show:.0f}% desp.) · "
                            f"dura ~{_dias_show:.1f} días"
                        )
                    with _rc_b:
                        if st.button(
                            "✏️ Editar",
                            key=f"edit_rollo_{_cr['id']}",
                            width="stretch",
                        ):
                            st.session_state[
                                f"editar_rollo_open_{_cr['id']}"
                            ] = True
                    with _rc_c:
                        if st.button(
                            "🗑️ Borrar",
                            key=f"del_rollo_{_cr['id']}",
                            width="stretch",
                        ):
                            db.eliminar_carga_silocomedero(
                                _cr["id"]
                            )
                            st.success("Entrega borrada.")
                            st.rerun()

                    # Form de edición (se abre con el botón
                    # Editar — guarda en st.session_state)
                    if st.session_state.get(
                        f"editar_rollo_open_{_cr['id']}", False,
                    ):
                        with st.container(border=True):
                            st.markdown(
                                f"**✏️ Editando entrega "
                                f"del {_fecha_show}**"
                            )
                            _ec1, _ec2, _ec3 = st.columns(3)
                            with _ec1:
                                _e_tipo = st.selectbox(
                                    "Tipo forraje",
                                    _tipos_disp,
                                    format_func=(
                                        lambda x:
                                        _tipos_label.get(x, x)
                                    ),
                                    index=(
                                        _tipos_disp.index(
                                            _meta.get(
                                                "tipo_forraje",
                                                "mezcla",
                                            )
                                        )
                                        if _meta.get(
                                            "tipo_forraje",
                                            "mezcla",
                                        ) in _tipos_disp else 0
                                    ),
                                    key=f"e_tipo_{_cr['id']}",
                                )
                                _e_cant = st.number_input(
                                    "Cantidad",
                                    min_value=1,
                                    max_value=100,
                                    value=max(
                                        1, int(_cant_show),
                                    ),
                                    step=1,
                                    key=f"e_cant_{_cr['id']}",
                                )
                            with _ec2:
                                _e_diam = st.number_input(
                                    "Diámetro (m)",
                                    min_value=0.8,
                                    max_value=2.5,
                                    value=float(
                                        _meta.get(
                                            "diametro_m", 1.5
                                        )
                                    ),
                                    step=0.05,
                                    key=f"e_diam_{_cr['id']}",
                                )
                                _e_ancho = st.number_input(
                                    "Ancho (m)",
                                    min_value=0.8,
                                    max_value=2.5,
                                    value=float(
                                        _meta.get(
                                            "ancho_m", 1.5
                                        )
                                    ),
                                    step=0.05,
                                    key=f"e_ancho_{_cr['id']}",
                                )
                            with _ec3:
                                _e_modo = st.selectbox(
                                    "Sistema de oferta",
                                    ["sin_parrilla",
                                     "parrillon_circular",
                                     "comedero_con_barrera"],
                                    format_func=(
                                        lambda x: {
                                            "sin_parrilla":
                                                "Sin cubre rollo",
                                            "parrillon_circular":
                                                "Cubre rollo",
                                            "comedero_con_barrera":
                                                "Comedero c/barrera",
                                        }.get(x, x)
                                    ),
                                    index=[
                                        "sin_parrilla",
                                        "parrillon_circular",
                                        "comedero_con_barrera",
                                    ].index(_sist_show)
                                    if _sist_show in [
                                        "sin_parrilla",
                                        "parrillon_circular",
                                        "comedero_con_barrera",
                                    ] else 0,
                                    key=f"e_modo_{_cr['id']}",
                                )
                                # Key con modo para refrescar
                                # default al cambiar oferta
                                _e_desp = st.number_input(
                                    "% Desperdicio",
                                    min_value=0.0,
                                    max_value=50.0,
                                    value=float(
                                        DESPERDICIO_ROLLO_DEFAULT
                                        .get(_e_modo, 25)
                                    ),
                                    step=1.0,
                                    key=(
                                        f"e_desp_{_cr['id']}_"
                                        f"{_e_modo}"
                                    ),
                                )

                            _e_fecha = st.date_input(
                                "Fecha de la entrega",
                                value=datetime.strptime(
                                    _fecha_show, "%Y-%m-%d",
                                ).date(),
                                key=f"e_fecha_{_cr['id']}",
                            )

                            # Recalcular kg y MS con datos
                            # nuevos del form de edición
                            _e_peso = calcular_peso_rollo(
                                diametro_m=float(_e_diam),
                                ancho_m=float(_e_ancho),
                                tipo_forraje=_e_tipo,
                                densidad_kg_m3=float(
                                    _meta.get(
                                        "densidad_kg_m3",
                                        DENSIDAD_ROLLO_KG_M3
                                        .get(_e_tipo, 145),
                                    )
                                ),
                                pct_ms=float(
                                    _meta.get(
                                        "pct_ms",
                                        PCT_MS_ROLLO
                                        .get(_e_tipo, 88),
                                    )
                                ),
                            )
                            _e_kg_tc_unit = (
                                _e_peso.get(
                                    "peso_tal_cual_kg", 0,
                                )
                            )
                            _e_kg_tc_total = (
                                _e_kg_tc_unit * _e_cant
                            )
                            _e_pct_ms_use = float(
                                _meta.get(
                                    "pct_ms",
                                    PCT_MS_ROLLO.get(
                                        _e_tipo, 88,
                                    ),
                                )
                            )
                            _e_kg_ms_aprov = (
                                _e_kg_tc_total
                                * (_e_pct_ms_use / 100)
                                * (1 - _e_desp / 100)
                            )
                            _e_cabezas = (
                                db.cantidad_vigente_lote(
                                    lote_id_sel
                                )
                            )
                            # Recalcular días auto con
                            # consumo proyectado del rollo
                            _e_cons_rollo = (
                                _cons_rollo_proy_kg_ms_an
                                if _cons_rollo_proy_kg_ms_an > 0
                                else 1.5
                            )
                            if (_e_cabezas > 0
                                    and _e_cons_rollo > 0):
                                _e_dias_auto = (
                                    _e_kg_ms_aprov
                                    / _e_cabezas
                                    / _e_cons_rollo
                                )
                            else:
                                _e_dias_auto = float(
                                    _dias_show
                                )

                            st.caption(
                                f"Nuevo total TC: "
                                f"**{_e_kg_tc_total:.0f} kg** "
                                f"· MS aprovechado: "
                                f"**{_e_kg_ms_aprov:.0f} kg** "
                                f"· Días estimados: "
                                f"**{_e_dias_auto:.1f}**"
                            )

                            _bg1, _bg2 = st.columns(2)
                            with _bg1:
                                if st.button(
                                    "💾 Guardar cambios",
                                    type="primary",
                                    key=(
                                        f"save_rollo_"
                                        f"{_cr['id']}"
                                    ),
                                    width="stretch",
                                ):
                                    try:
                                        _new_desg = [{
                                            "nombre": (
                                                f"Rollo "
                                                f"{_tipos_label.get(_e_tipo, _e_tipo)}"
                                            ),
                                            "kg": float(
                                                _e_kg_tc_total
                                            ),
                                            "tipo_forraje":
                                                _e_tipo,
                                            "cantidad_rollos":
                                                int(_e_cant),
                                            "diametro_m":
                                                float(_e_diam),
                                            "ancho_m":
                                                float(_e_ancho),
                                            "densidad_kg_m3":
                                                float(
                                                    _meta.get(
                                                        "densidad_kg_m3",
                                                        DENSIDAD_ROLLO_KG_M3.get(
                                                            _e_tipo, 145,
                                                        ),
                                                    )
                                                ),
                                            "pct_ms": float(
                                                _e_pct_ms_use
                                            ),
                                            "desperdicio_pct":
                                                float(_e_desp),
                                            "sistema_oferta":
                                                _e_modo,
                                            "peso_unitario_kg":
                                                float(
                                                    _e_kg_tc_unit
                                                ),
                                        }]
                                        db.actualizar_carga_silocomedero(
                                            carga_id=_cr["id"],
                                            fecha_carga=(
                                                _e_fecha
                                                .isoformat()
                                            ),
                                            kg_cargados=float(
                                                _e_kg_tc_total
                                            ),
                                            desglose_ingredientes=(
                                                _new_desg
                                            ),
                                            dias_cubiertos=float(
                                                _e_dias_auto
                                            ),
                                        )
                                        st.session_state.pop(
                                            f"editar_rollo_open_"
                                            f"{_cr['id']}",
                                            None,
                                        )
                                        st.success(
                                            "✅ Entrega "
                                            "actualizada."
                                        )
                                        st.rerun()
                                    except Exception as _e_upd:
                                        st.error(
                                            f"Error: {_e_upd}"
                                        )
                            with _bg2:
                                if st.button(
                                    "Cancelar",
                                    key=(
                                        f"cancel_rollo_"
                                        f"{_cr['id']}"
                                    ),
                                    width="stretch",
                                ):
                                    st.session_state.pop(
                                        f"editar_rollo_open_"
                                        f"{_cr['id']}",
                                        None,
                                    )
                                    st.rerun()

        # ---------- HISTORIAL con comparación ----------
        # Modo flexible: el encargado puede cargar varias entradas en
        # el mismo día (ej. 2 comidas en lineal). Acá agrupamos por día
        # y comparamos el TOTAL del día contra la dieta.
        st.markdown("**Historial de cargas — vs dieta**")
        cargas = db.listar_cargas_silocomedero(
            lote_id_sel, limit=60,
        )
        if not cargas:
            st.caption("_Sin cargas registradas._")
        else:
            _COLORES_SEM = {
                "verde": "#1B7A3E",
                "amarillo": "#B07D0E",
                "rojo": "#A32D2D",
                "gris": "#666666",
            }
            _ICONOS_SEM = {
                "verde": "🟢", "amarillo": "🟠",
                "rojo": "🔴", "gris": "⚪",
            }
            # Agrupar por día calendario
            grupos_dia = sp.agrupar_cargas_por_dia(cargas)
            filas_carga = []
            comps_por_fecha = {}
            for g in grupos_dia:
                comp = sp.comparar_carga_vs_dieta(g)
                comps_por_fecha[g["fecha_carga"]] = comp
                _sem = comp["semaforo"]
                _ico = _ICONOS_SEM.get(_sem, "⚪")
                _esp = (
                    f"{comp['esperado_total_kg']:.0f}"
                    if not comp.get("sin_dieta") else "—"
                )
                if comp.get("sin_dieta"):
                    _desv_kg = "—"
                    _desv_pct = "—"
                else:
                    _desv_kg = f"{comp['desvio_kg']:+.0f}"
                    _desv_pct = f"{comp['desvio_pct']:+.1f}%"
                # Tipo: "1 comida 08:30", "2 comidas (08:30, 16:00)",
                # "carga 5d" para silocomedero, etc.
                _n = g["n_subcargas"]
                if g["tipo_carga"] == "silo_carga":
                    _tipo_str = f"silo {g['dias_cubiertos']:.0f}d"
                else:
                    _horas = [
                        s["hora"] for s in g["subcargas"]
                        if s["hora"] and s["hora"] != "—"
                    ]
                    if _n == 1:
                        _tipo_str = (
                            f"1 comida {_horas[0]}"
                            if _horas else "1 comida"
                        )
                    else:
                        _tipo_str = (
                            f"{_n} comidas ({', '.join(_horas)})"
                            if _horas else f"{_n} comidas"
                        )
                filas_carga.append({
                    "Fecha": g["fecha_carga"],
                    "Tipo": _tipo_str,
                    "Real (kg)": f"{g['kg_cargados']:.0f}",
                    "Esperado (kg)": _esp,
                    "Desvío kg": _desv_kg,
                    "Desvío %": f"{_ico} {_desv_pct}",
                    "Obs.": (g.get("detalles") or "")[:30],
                })
            df_carga = pd.DataFrame(filas_carga)
            st.dataframe(
                df_carga, hide_index=True, width="stretch",
            )

            # ---------- Gráfico comparativo ----------
            # Línea de tiempo con dos series: lo cargado vs lo
            # que la fórmula pide. Usa los grupos por día.
            _chart_rows = []
            for g in grupos_dia[::-1]:  # cronológico ascendente
                _comp = comps_por_fecha.get(g["fecha_carga"])
                if not _comp or _comp.get("sin_dieta"):
                    continue
                _chart_rows.append({
                    "Fecha": g["fecha_carga"],
                    "Cargado real": _comp["real_total_kg"],
                    "Esperado por fórmula":
                        _comp["esperado_total_kg"],
                })
            if len(_chart_rows) >= 2:
                st.markdown(
                    "**📊 Cargado real vs esperado por fórmula**"
                )
                _df_chart = pd.DataFrame(_chart_rows)
                _df_chart = _df_chart.set_index("Fecha")
                # Gráfico de líneas: dos curvas (real y esperado).
                # Más fácil leer tendencia que barras agrupadas cuando
                # hay muchos días.
                st.line_chart(
                    _df_chart,
                    height=280,
                    color=["#5BAE7D", "#E89938"],
                )
                # Mini estadística: promedio del desvío en %
                _comps_validas = [
                    c for c in comps_por_fecha.values()
                    if not c.get("sin_dieta", True)
                ]
                _desv_prom = (
                    sum(c["desvio_pct"] for c in _comps_validas)
                    / max(1, len(_comps_validas))
                )
                _signo = "por encima" if _desv_prom > 0 else (
                    "por debajo" if _desv_prom < 0 else "en línea"
                )
                st.caption(
                    f"En promedio las cargas vienen "
                    f"**{abs(_desv_prom):.1f}% {_signo}** de "
                    f"lo que pide la fórmula. Verde = lo "
                    f"cargado, naranja = lo que la dieta pide."
                )
            elif len(_chart_rows) == 1:
                st.caption(
                    "_Cargá al menos 2 días para ver el "
                    "gráfico comparativo._"
                )

            # Detalle del último día agrupado (con subcargas por hora +
            # desglose por ingrediente si lo hay).
            _ult_grupo = grupos_dia[0]
            _ult_comp = comps_por_fecha[_ult_grupo["fecha_carga"]]
            with st.expander(
                f"🔎 Detalle del último día "
                f"({_ult_grupo['fecha_carga']} · "
                f"{_ult_grupo['n_subcargas']} carga(s))",
                expanded=False,
            ):
                if _ult_comp.get("sin_dieta"):
                    st.warning(_ult_comp["mensaje"])
                else:
                    _sem = _ult_comp["semaforo"]
                    _col_sem = _COLORES_SEM.get(_sem, "#666")
                    st.markdown(
                        f"<div style='padding:10px;"
                        f"border-left:4px solid {_col_sem};"
                        f"background:#FAFAFA;"
                        f"border-radius:4px;'>"
                        f"{_ICONOS_SEM[_sem]} "
                        f"<strong>{_ult_comp['mensaje']}</strong>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Dieta de referencia: "
                        f"{_ult_comp.get('dieta_vigente_fecha')}"
                        f" · {_ult_comp['cantidad_animales']} cab. "
                        f"× {_ult_comp['dias_cubiertos']:.0f} día(s)"
                    )
                    # Subcargas (comidas) del día
                    if _ult_grupo["n_subcargas"] >= 1:
                        st.markdown(
                            "**Comidas del día (por hora)**"
                        )
                        _filas_sub = []
                        for _s in _ult_grupo["subcargas"]:
                            _filas_sub.append({
                                "Hora": _s["hora"],
                                "kg cargados":
                                    f"{_s['kg']:.0f}",
                                "Observaciones":
                                    (_s.get("obs") or "")[:50],
                            })
                        st.dataframe(
                            pd.DataFrame(_filas_sub),
                            hide_index=True, width="stretch",
                        )
                    if _ult_comp["por_ingrediente"]:
                        st.markdown(
                            "**Comparación por ingrediente (total del día)**"
                        )
                        _filas_ing = []
                        for _ing in _ult_comp["por_ingrediente"]:
                            _ico = _ICONOS_SEM.get(
                                _ing["semaforo"], "⚪"
                            )
                            _filas_ing.append({
                                "Ingrediente": _ing["nombre"],
                                "Esperado (kg)":
                                    f"{_ing['esperado_kg']:.1f}",
                                "Real (kg)":
                                    f"{_ing['real_kg']:.1f}",
                                "Desvío kg":
                                    f"{_ing['desvio_kg']:+.1f}",
                                "Desvío %":
                                    f"{_ico} "
                                    f"{_ing['desvio_pct']:+.1f}%",
                            })
                        st.dataframe(
                            pd.DataFrame(_filas_ing),
                            hide_index=True, width="stretch",
                        )
                    else:
                        st.caption(
                            "_Carga registrada como total. "
                            "Para auditar dónde está el desvío "
                            "por ingrediente, registrá la "
                            "próxima carga en modo 'Por "
                            "ingrediente'._"
                        )

            with st.expander(
                "🗑️ Borrar una carga", expanded=False
            ):
                opciones_c = {
                    c["id"]: (
                        f"{c['fecha_carga']}"
                        f"{' ' + c['hora_carga'] if c.get('hora_carga') else ''}"
                        f" · {c['kg_cargados']:.0f} kg"
                    )
                    for c in cargas
                }
                carga_a_borrar = st.selectbox(
                    "Cuál borrar",
                    list(opciones_c.keys()),
                    format_func=lambda k: opciones_c[k],
                    key=f"borrar_carga_sel_{lote_id_sel}",
                )
                if st.button(
                    "Borrar carga",
                    key=f"borrar_carga_btn_{lote_id_sel}",
                ):
                    db.eliminar_carga_silocomedero(
                        carga_a_borrar
                    )
                    st.success("Carga eliminada.")
                    st.rerun()

    # ---- 🌦️ Clima del establecimiento ----
    # Mismo bloque que mostraba el dashboard pero acá enfocado al
    # lote: clima actual, pronóstico 14 días (7 pasados + HOY + 7
    # futuros) y semáforo de severidad real para los próximos días.
    # Usa las coordenadas del cliente vinculado al lote.
    # ABIERTO por default porque es info operativa importante —
    # querés verlo cada vez que entrás al lote.
    st.divider()
    st.markdown(
        "### 🌦️ Clima del establecimiento "
        "— actual + pronóstico"
    )
    # Cargar cliente del lote
    try:
        _cli_lote = db.obtener_cliente(lote.get("cliente_id"))
    except Exception:
        _cli_lote = None

    if not _cli_lote or not (_cli_lote.get("localidad")
                              or _cli_lote.get("lat")):
        st.info(
            "⚠️ Para ver el clima del establecimiento, cargá "
            "la **localidad** o las **coordenadas (lat, lon)** "
            "en la ficha del cliente."
        )
    else:
        if True:  # placeholder para mantener identación del bloque
            with st.spinner(
                "Consultando clima del establecimiento..."
            ):
                _info_clima = dashboard.obtener_clima_para_cliente(
                    _cli_lote,
                )

            _estado_cl = _info_clima.get("estado", "")
            if _estado_cl == "sin_geocodificar":
                st.warning(
                    f"No pude geocodificar "
                    f"'{_cli_lote.get('localidad','')}'. "
                    "Probá con un nombre más específico o cargá "
                    "lat/lon manuales."
                )
            elif _estado_cl in ("sin_clima", "error"):
                st.warning(
                    f"No pude obtener el clima: "
                    f"{_info_clima.get('error', 'sin datos')}"
                )
            elif (_info_clima.get("temp_c") is None):
                st.info("Sin datos climáticos disponibles ahora.")
            else:
                # ── Cabecera con KPIs actuales ──
                _k1, _k2, _k3, _k4 = st.columns(4)
                _k1.metric(
                    "🌡️ Temperatura HOY",
                    f"{_info_clima['temp_c']:.0f}°C",
                )
                _k2.metric(
                    "💧 Humedad",
                    f"{_info_clima['humedad_pct']:.0f}%",
                )
                _k3.metric(
                    "📊 THI",
                    f"{_info_clima['thi']:.0f}",
                    help=_info_clima.get('thi_estado', ''),
                )
                _sev_max_l = _info_clima.get(
                    "severidad_real_max", "🟢 Sin estrés",
                )
                _k4.metric(
                    "Próximos 7d",
                    _sev_max_l,
                    help=(
                        "Peor severidad real proyectada "
                        "(HOY + próximos 7 días)"
                    ),
                )

                # Si la peor severidad es 🟠 o 🔴, destacar
                _rank_l = _info_clima.get(
                    "severidad_real_max_rank", 1,
                )
                _fecha_peor_l = _info_clima.get(
                    "severidad_real_max_fecha", "",
                )
                if _rank_l >= 3 and _fecha_peor_l:
                    try:
                        _fp_disp = datetime.strptime(
                            _fecha_peor_l, "%Y-%m-%d"
                        ).strftime("%d/%m")
                    except Exception:
                        _fp_disp = _fecha_peor_l
                    st.warning(
                        f"⚠️ **Atención**: el {_fp_disp} se "
                        f"proyecta **{_sev_max_l}**. "
                        f"Coordinar manejo preventivo "
                        f"(reparos, cama, agua) antes del evento."
                    )

                # ─── 🤖 Análisis climático narrativo con IA ───
                # Pasado / presente / futuro + medidas concretas
                # específicas para este lote. Se cachea en
                # session_state para no llamar al LLM cada reload.
                st.markdown("---")
                _cache_clima_key = (
                    f"analisis_clima_lote_{lote_id_sel}_"
                    f"{_fecha_peor_l or 'na'}_"
                    f"{_info_clima.get('temp_c','na')}"
                )
                _col_an1, _col_an2 = st.columns([1, 3])
                if _col_an1.button(
                    "🤖 Generar análisis IA",
                    key=f"btn_analisis_clima_{lote_id_sel}",
                    type="primary",
                    width="stretch",
                    help=(
                        "Pide al agente IA un análisis narrativo: "
                        "qué pasó / qué pasa / qué viene + "
                        "medidas concretas para este lote."
                    ),
                ):
                    # Cargar dieta vigente para contexto del LLM
                    try:
                        _dietas_ctx = db.listar_dietas(
                            lote_id_sel,
                        ) or []
                        _dv_ctx = (
                            _dietas_ctx[0] if _dietas_ctx else None
                        )
                    except Exception:
                        _dv_ctx = None
                    # Armar contexto clínico completo (historial,
                    # cargas, pesadas, movimientos, fase del plan)
                    # para que el LLM sea un asesor con memoria,
                    # no solo un intérprete del clima de hoy.
                    try:
                        _hist_ctx = (
                            dashboard.armar_contexto_clinico_lote(
                                lote_id_sel, db,
                            )
                        )
                    except Exception as _e_hist:
                        _hist_ctx = ""
                    with st.spinner(
                        "🤖 Pidiendo análisis al asesor IA..."
                    ):
                        _res_an = (
                            dashboard.analizar_clima_lote_llm(
                                lote, _info_clima, _dv_ctx,
                                historial_clinico=_hist_ctx,
                                api_key=st.session_state.get(
                                    "anthropic_api_key", ""
                                ),
                            )
                        )
                    if _res_an.get("exito"):
                        st.session_state[_cache_clima_key] = (
                            _res_an["analisis_md"]
                        )
                    else:
                        _col_an2.warning(
                            f"⚠️ {_res_an.get('error','LLM falló')}"
                        )
                with _col_an2:
                    if _cache_clima_key not in st.session_state:
                        st.caption(
                            "_Hacé clic para que el asesor IA "
                            "interprete el clima y proponga "
                            "acciones concretas para este lote._"
                        )

                # Mostrar análisis si está cacheado
                if _cache_clima_key in st.session_state:
                    st.markdown(
                        "<div style='background:#f0f7ff;"
                        "border-left:4px solid #2c7be5;"
                        "padding:12px 16px;border-radius:6px;"
                        "margin:8px 0;'>"
                        "<b>🤖 Análisis del Asesor IA</b>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        st.session_state[_cache_clima_key]
                    )

                # ── Tabla del pronóstico (mismo formato que dash) ──
                _pron_l = _info_clima.get("pronostico_7d") or []
                if _pron_l:
                    _filas_l = []
                    for _d_l in _pron_l:
                        _thi_d = _d_l.get("thi")
                        _filas_l.append({
                            "Tramo": _d_l.get("tramo", "—"),
                            "Fecha": _d_l.get("fecha", "—"),
                            "T° min": (
                                f"{_d_l['t_min']:.0f}°C"
                                if _d_l.get("t_min") is not None
                                else "—"
                            ),
                            "T° máx": (
                                f"{_d_l['t_max']:.0f}°C"
                                if _d_l.get("t_max") is not None
                                else "—"
                            ),
                            "HR": (
                                f"{_d_l['hr_media']:.0f}%"
                                if _d_l.get("hr_media") is not None
                                else "—"
                            ),
                            "Lluvia": (
                                f"{_d_l['precipitacion_mm']:.1f} mm"
                                if _d_l.get("precipitacion_mm")
                                not in (None, 0)
                                else "—"
                                if _d_l.get("precipitacion_mm")
                                is None else "0 mm"
                            ),
                            "Viento": (
                                f"{_d_l['viento_max_kmh']:.0f} km/h"
                                if _d_l.get("viento_max_kmh")
                                is not None else "—"
                            ),
                            "Cielo": (
                                (
                                    "☀️ Despejado"
                                    if (_d_l.get("nubes_pct") or 0) < 30
                                    else "⛅ Parcial"
                                    if (_d_l.get("nubes_pct") or 0) < 70
                                    else "☁️ Cubierto"
                                ) + f" ({_d_l['nubes_pct']:.0f}%)"
                                if _d_l.get("nubes_pct") is not None
                                else "—"
                            ),
                            "THI": (
                                f"{_thi_d:.0f}"
                                if _thi_d is not None else "—"
                            ),
                            "Severidad real": _d_l.get(
                                "severidad_real", "—",
                            ),
                        })
                    st.markdown(
                        "##### Pronóstico 14 días "
                        "(7 pasados · HOY · 7 futuros)"
                    )
                    st.dataframe(
                        pd.DataFrame(_filas_l),
                        hide_index=True, width="stretch",
                    )
                    st.caption(
                        "_Severidad real = THI + viento (Mader "
                        "2006) + wind chill bovino + barro por "
                        "lluvia + falta de secado por cielo "
                        "cubierto sostenido._"
                    )

    # ---- Ficha clínica del lote (estilo historia médica) ----
    # Acumula las evaluaciones de cada llamada/visita como si fueran
    # consultas médicas. Detecta patrones recurrentes, tally de
    # mortandad por causa, diagnósticos activos y un resumen
    # generado con IA.
    with st.expander(
        "🩺 Ficha clínica del lote — historia médica del paciente",
        expanded=False,
    ):
        from src import ficha_clinica as fc
        from src import evaluacion_lote as ev
        _ficha_cl = fc.armar_ficha_clinica_lote(lote_id_sel, db)
        _n_ev = _ficha_cl.get("n_evaluaciones", 0)

        # ─── Botón para registrar nueva consulta ───
        # Es el "ABM" estilo médico: desde la ficha del paciente,
        # cargás una nueva consulta. Crea + completa un recordatorio
        # asociado a este lote en un solo paso.
        _key_form_consulta = f"form_consulta_lote_{lote_id_sel}"
        _cb_consulta_col1, _cb_consulta_col2 = st.columns([1, 3])
        if _cb_consulta_col1.button(
            "📝 Registrar nueva consulta",
            key=f"btn_nueva_consulta_{lote_id_sel}",
            type="primary",
            width="stretch",
            help=(
                "Abre el cuestionario de evaluación con este lote "
                "y cliente pre-cargados. Al guardar, se suma a la "
                "historia clínica de abajo."
            ),
        ):
            st.session_state[_key_form_consulta] = True

        with _cb_consulta_col2:
            _lote_cli_nombre = (
                lote.get("cliente_nombre", "")
                or "este cliente"
            )
            st.caption(
                f"_Carga aquí lo que conversaste / observaste sobre "
                f"el lote **{lote.get('identificador','')}** con "
                f"{_lote_cli_nombre}. Queda registrado abajo en la "
                f"historia._"
            )

        # Renderizar el form si está activo
        if st.session_state.get(_key_form_consulta):
            _renderizar_form_evaluacion(
                recordatorio_id=None,
                cliente_id=lote.get("cliente_id"),
                cliente_nombre=(
                    lote.get("cliente_nombre", "")
                    or "(sin nombre)"
                ),
                ev_mod=ev,
                on_close_state=_key_form_consulta,
            )
            st.divider()

        # ── 📸 Fotos de inspección del lote ──
        # Permite cargar fotos que mandó el productor/operario por
        # WhatsApp durante la entrevista. Cada foto se categoriza
        # (bosta/animales/comedero/corral/bebedero/otros) y opcional-
        # mente se asocia a la consulta más reciente del lote. Las
        # fotos quedan disponibles para el informe PDF del lote.
        _render_bloque_fotos_lote(
            lote_id=lote_id_sel,
            lote=lote,
        )
        st.divider()

        if _n_ev == 0:
            st.info(
                "📋 Este lote todavía no tiene consultas "
                "registradas. Usá el botón **📝 Registrar nueva "
                "consulta** de arriba para empezar la historia "
                "clínica."
            )
        else:
            # ── Header: cuántas consultas / muertes / ventas ──
            _kpi1, _kpi2, _kpi3, _kpi4 = st.columns(4)
            _kpi1.metric(
                "📅 Consultas",
                _n_ev,
                help="Evaluaciones registradas del lote",
            )
            _kpi2.metric(
                "💀 Muertes acumuladas",
                _ficha_cl.get("total_muertes", 0),
            )
            _kpi3.metric(
                "🐄 Ventas/salidas",
                _ficha_cl.get("total_ventas", 0),
            )
            _kpi4.metric(
                "📌 Dx activos",
                len(_ficha_cl.get("diagnosticos_activos", [])),
            )

            # ── Resumen clínico IA ──
            st.markdown("##### 🩺 Resumen clínico (IA)")
            _btn_gen_col, _btn_msg_col = st.columns([1, 3])
            _cache_key = (
                f"resumen_clinico_lote_{lote_id_sel}_"
                f"{_n_ev}"
            )
            if _btn_gen_col.button(
                "🔄 Generar / actualizar resumen",
                key=f"gen_resumen_cl_{lote_id_sel}",
            ):
                # Sumar al ficha_cl el contexto clínico unificado
                # (ADG real, sub-consumo, fase, etc.) para que el
                # resumen también lo considere
                try:
                    _ctx_unif = (
                        dashboard.armar_contexto_clinico_lote(
                            lote_id_sel, db,
                        )
                    )
                    if _ctx_unif:
                        _ficha_cl["contexto_unificado"] = _ctx_unif
                except Exception:
                    pass
                with st.spinner(
                    "Pidiéndole al asesor IA el resumen clínico..."
                ):
                    _resum = fc.generar_resumen_clinico_llm(
                        _ficha_cl,
                        api_key=st.session_state.get(
                            "anthropic_api_key", ""
                        ),
                    )
                if _resum.get("exito"):
                    st.session_state[_cache_key] = (
                        _resum["resumen_md"]
                    )
                else:
                    _btn_msg_col.warning(
                        f"⚠️ {_resum.get('error','LLM falló')}"
                    )

            if _cache_key in st.session_state:
                st.markdown(
                    "<div style='background:#f0f7ff;"
                    "border-left:4px solid #2c7be5;"
                    "padding:12px 16px;border-radius:6px;"
                    "margin:8px 0;'>"
                    + st.session_state[_cache_key].replace(
                        "\n", "<br>"
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    "_Hacé clic en 'Generar' para que el "
                    "agente IA te dé un resumen del estado "
                    "clínico del lote._"
                )

            # ── Diagnósticos activos HOY ──
            _dxs = _ficha_cl.get("diagnosticos_activos", [])
            if _dxs:
                st.markdown(
                    "##### 📌 Diagnósticos activos (última consulta)"
                )
                for _dx in _dxs:
                    st.markdown(
                        f"- **{_dx['label']}** · "
                        f"_{_dx['estado']}_ "
                        f"(detectado {_dx['fecha_deteccion']})"
                    )

            # ── Patrones recurrentes ──
            _pts = _ficha_cl.get("patrones", [])
            if _pts:
                st.markdown(
                    "##### 🔁 Patrones recurrentes detectados"
                )
                st.caption(
                    "Síntomas que aparecen en >=50% de las "
                    "últimas evaluaciones. Indican un problema "
                    "estructural, no puntual."
                )
                for _p in _pts:
                    st.warning(
                        f"**{_p['label']}** · "
                        f"_{_p['frecuencia']}_  \n"
                        f"{_p['sugerencia']}"
                    )

            # ── Tally de mortandad por causa ──
            _tally = _ficha_cl.get("tally_mortandad", {})
            if _tally:
                st.markdown(
                    "##### 💀 Mortandad acumulada por causa"
                )
                _filas_tally = [
                    {"Causa": k, "Total muertes": v}
                    for k, v in sorted(
                        _tally.items(),
                        key=lambda kv: -kv[1],
                    )
                ]
                st.dataframe(
                    pd.DataFrame(_filas_tally),
                    hide_index=True, width="stretch",
                )

            # ── Línea de tiempo (evaluaciones) ──
            st.markdown(
                "##### 📜 Línea de tiempo — consultas anteriores"
            )
            _evals = _ficha_cl.get("evaluaciones", [])
            for _ev in _evals[:15]:
                _hdr = (
                    f"{_ev.resumen_semaforo} "
                    f"**{_ev.fecha[:10]}** · "
                    f"{_ev.tipo_contacto}"
                )
                if _ev.atendio:
                    _hdr += f" con {_ev.atendio}"
                if _ev.bajas > 0:
                    _hdr += f" · 💀 {_ev.bajas} muerte(s)"
                if _ev.ventas > 0:
                    _hdr += f" · 🐄 {_ev.ventas} venta(s)"
                with st.expander(_hdr):
                    st.markdown(_ev.notas_md)
                    # Bloque visual: comparación dieta REAL vs formulada
                    # de esta consulta puntual (si el asesor cargó la
                    # dieta real del cliente en sección 5b del form).
                    _render_comparacion_dieta_real_consulta(
                        recordatorio_id=_ev.rid,
                        lote_id=lote_id_sel,
                    )
                    # Botón de edición rápida — abre form que permite
                    # corregir los datos más comunes que se cargan mal
                    # (stock, silo, rumia, agua, etc.) sin tener que
                    # rehacer toda la consulta.
                    st.divider()
                    _edit_key = f"edit_consulta_{_ev.rid}"
                    if st.button(
                        "✏️ Editar esta consulta",
                        key=f"btn_{_edit_key}",
                        help=(
                            "Corregí campos mal cargados sin rehacer "
                            "toda la consulta. Útil cuando confundís "
                            "stock total con consumo diario, etc."
                        ),
                    ):
                        st.session_state[_edit_key] = True

                    if st.session_state.get(_edit_key):
                        _render_form_edicion_rapida(
                            recordatorio_id=_ev.rid,
                            datos_actuales=_ev,
                            on_close_state=_edit_key,
                        )
            if len(_evals) > 15:
                st.caption(
                    f"_Mostrando 15 de {len(_evals)} consultas._"
                )

    # ---- Movimientos de hacienda (al final, en expander) ----
    # Registro de bajas/ingresos del lote. Lo dejamos al fondo
    # como menú desplegable porque es información operativa
    # menos frecuente que el seguimiento principal (KPIs,
    # dietas, cargas, evolución de consumo).
    with st.expander(
        "🔄 Movimientos de hacienda (bajas, ventas, ingresos, traslados)",
        expanded=False,
    ):
        # ---- Movimientos de hacienda ----
        # Registro de cambios en la cantidad del lote: muertes,
        # ventas, traslados e ingresos. La cantidad vigente que se
        # muestra arriba sale de acá. Afecta directamente el cálculo
        # de consumo de producto y la autonomía de stock.
        st.markdown("##### 🔄 Movimientos de hacienda")
        st.caption(
            "Registrá bajas (muerte, venta), ingresos (compra, "
            "nacimiento) o traslados entre lotes. El sistema usa la "
            "cantidad vigente cada día para calcular consumo de "
            "producto y autonomía de stock."
        )

        mov_col_form, mov_col_hist = st.columns([1, 1])

        with mov_col_form:
            with st.form(f"nuevo_mov_{lote_id_sel}",
                         clear_on_submit=True):
                st.markdown("**Registrar movimiento**")
                mov_motivo = st.selectbox(
                    "Motivo",
                    list(db.MOVIMIENTO_LABELS.keys()),
                    format_func=lambda k: db.MOVIMIENTO_LABELS[k],
                    key=f"mov_motivo_{lote_id_sel}",
                )
                mov_fecha = st.date_input(
                    "Fecha", value=datetime.now().date(),
                    key=f"mov_fecha_{lote_id_sel}",
                )
                mov_cant = st.number_input(
                    "Cantidad de animales",
                    min_value=1, max_value=10000, value=1, step=1,
                    key=f"mov_cant_{lote_id_sel}",
                )
                # Peso promedio: relevante en venta (calcular kg
                # facturados) y en muerte (a qué peso se perdió).
                # Para traslados/ingresos también ayuda pero es
                # opcional.
                mov_kg = st.number_input(
                    "Peso promedio por animal (kg) — opcional",
                    min_value=0.0, max_value=1500.0, value=0.0,
                    step=10.0,
                    key=f"mov_kg_{lote_id_sel}",
                    help="Útil en ventas (kg facturados) y en "
                         "muertes (peso del animal perdido).",
                )
                # Destino: solo para traslado_egreso. Listamos los
                # otros lotes activos del mismo cliente.
                destino_id = None
                if mov_motivo == "traslado_egreso":
                    otros_lotes = [
                        l for l in db.listar_lotes(
                            cliente_id=lote["cliente_id"],
                            estado="activo",
                        )
                        if l["id"] != lote_id_sel
                    ]
                    if otros_lotes:
                        opciones = {
                            l["id"]: f"{l['identificador']} "
                                      f"({l.get('categoria','—')})"
                            for l in otros_lotes
                        }
                        destino_id = st.selectbox(
                            "Lote destino",
                            list(opciones.keys()),
                            format_func=lambda k: opciones[k],
                            key=f"mov_dest_{lote_id_sel}",
                        )
                    else:
                        st.warning(
                            "No hay otros lotes activos del mismo "
                            "cliente. Para traslados, primero creá "
                            "el lote destino."
                        )
                mov_obs = st.text_input(
                    "Observaciones (opcional)",
                    key=f"mov_obs_{lote_id_sel}",
                )
                btn_mov = st.form_submit_button(
                    "💾 Registrar movimiento",
                    type="primary",
                )
                if btn_mov:
                    # Validaciones previas: no permitir bajar de 0.
                    signo = db.MOVIMIENTO_TIPOS[mov_motivo]
                    if signo < 0:
                        cant_actual = db.cantidad_vigente_lote(
                            lote_id_sel,
                            fecha=mov_fecha.isoformat(),
                        )
                        if mov_cant > cant_actual:
                            st.error(
                                f"❌ No podés sacar {mov_cant} "
                                f"animales — el lote tiene "
                                f"{cant_actual} a esa fecha."
                            )
                            st.stop()
                    if (mov_motivo == "traslado_egreso"
                            and not destino_id):
                        st.error(
                            "❌ Elegí el lote destino para el "
                            "traslado."
                        )
                        st.stop()
                    try:
                        db.crear_movimiento_lote(
                            lote_id=lote_id_sel,
                            fecha=mov_fecha.isoformat(),
                            tipo=mov_motivo,
                            cantidad=int(mov_cant),
                            kg_promedio_animal=(
                                float(mov_kg) if mov_kg > 0
                                else None
                            ),
                            destino_lote_id=destino_id,
                            detalles=mov_obs or "",
                        )
                        st.success(
                            f"✅ {db.MOVIMIENTO_LABELS[mov_motivo]}"
                            f" — {mov_cant} cab. registradas."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        with mov_col_hist:
            st.markdown("**Historial**")
            movs = db.listar_movimientos_lote(lote_id_sel)
            if not movs:
                st.caption("_Sin movimientos registrados._")
            else:
                # Tabla con los movimientos. El signo del impacto
                # se infiere del tipo, así el usuario ve si suma
                # o resta sin pensarlo.
                filas_mov = []
                for m in movs:
                    signo = db.MOVIMIENTO_TIPOS.get(m["tipo"], 0)
                    delta = signo * int(m.get("cantidad") or 0)
                    kg_unit = m.get("kg_promedio_animal") or 0
                    kg_total = (
                        kg_unit * (m.get("cantidad") or 0)
                        if kg_unit else 0
                    )
                    filas_mov.append({
                        "Fecha": m["fecha"],
                        "Motivo": db.MOVIMIENTO_LABELS.get(
                            m["tipo"], m["tipo"]
                        ),
                        "Δ animales": f"{delta:+d}",
                        "Kg/cab": (
                            f"{kg_unit:.0f}" if kg_unit else "—"
                        ),
                        "Kg total": (
                            f"{kg_total:,.0f}"
                            if kg_total else "—"
                        ),
                        "Obs.": (m.get("detalles") or "")[:40],
                        "_id": m["id"],
                    })
                df_mov = pd.DataFrame(filas_mov)
                st.dataframe(
                    df_mov.drop(columns=["_id"]),
                    hide_index=True, width="stretch",
                )

                # Permitir borrar un movimiento (selector + botón).
                # Útil si se cargó por error. Si era un traslado,
                # también se elimina el espejo en el lote destino.
                with st.expander(
                    "🗑️ Borrar un movimiento", expanded=False
                ):
                    opciones_mov = {
                        m["id"]: (
                            f"{m['fecha']} · "
                            f"{db.MOVIMIENTO_LABELS.get(m['tipo'], m['tipo'])}"
                            f" · {m['cantidad']} cab."
                        )
                        for m in movs
                    }
                    mov_a_borrar = st.selectbox(
                        "Cuál borrar",
                        list(opciones_mov.keys()),
                        format_func=lambda k: opciones_mov[k],
                        key=f"borrar_mov_sel_{lote_id_sel}",
                    )
                    if st.button(
                        "Borrar movimiento",
                        key=f"borrar_mov_btn_{lote_id_sel}",
                    ):
                        db.eliminar_movimiento_lote(mov_a_borrar)
                        st.success("Movimiento eliminado.")
                        st.rerun()


with tab_clientes:
    st.markdown(
        "### 🏢 Gestión de Clientes y Lotes\n"
        "Registrá tus clientes (establecimientos) y los lotes activos para "
        "seguimiento longitudinal. Cada análisis por drone se guarda asociado a "
        "un lote, lo que te permite ver evolución a lo largo del tiempo."
    )

    sub1, sub2 = st.tabs(["👥 Clientes", "🐄 Lotes"])

    # ---- Clientes ----
    with sub1:
        col_lista, col_form = st.columns([2, 1])

        with col_form:
            st.markdown("##### ➕ Nuevo cliente")

            # Si el último submit fue exitoso, limpiar los campos ANTES
            # de instanciar los widgets (no podemos modificar
            # session_state después de que un widget con esa key se
            # creó — Streamlit tira error).
            if st.session_state.get("ncl_clear_pending"):
                for _k in [
                    "ncl_nombre", "ncl_estab",
                    "ncl_loc", "ncl_cont", "ncl_email",
                    "ncl_whatsapp", "ncl_notas",
                ]:
                    st.session_state.pop(_k, None)
                st.session_state["ncl_clear_pending"] = False

            # Mostrar mensaje persistente del último intento (sobrevive el
            # rerun que hace Streamlit al enviar el form)
            if st.session_state.get("ncl_msg"):
                _msg_kind, _msg_text = st.session_state["ncl_msg"]
                if _msg_kind == "success":
                    st.success(_msg_text)
                elif _msg_kind == "error":
                    st.error(_msg_text)
                else:
                    st.info(_msg_text)
                # No la limpiamos automáticamente — se limpia al próximo
                # submit o al cambiar de pestaña

            # Datos básicos en el form
            with st.form("nuevo_cliente", clear_on_submit=False):
                c_nombre = st.text_input("Nombre / razón social *",
                                          key="ncl_nombre")
                c_estab = st.text_input("Establecimiento", key="ncl_estab")
                c_localidad = st.text_input(
                    "Localidad / provincia", key="ncl_loc",
                    help="Ej: 'Catriló, La Pampa', 'Realicó', "
                         "'Colonia La Carlota, La Pampa'",
                )
                c_contacto = st.text_input("Contacto (tel/email)",
                                            key="ncl_cont")
                c_email = st.text_input(
                    "Email para alertas climáticas (opcional)",
                    key="ncl_email",
                    placeholder="cliente@dominio.com",
                    help="Si cargás un email, el cliente recibe sus alertas "
                          "climáticas todos los días a las 08:00.",
                )
                c_alertas_on = st.checkbox(
                    "Recibir alertas climáticas por email",
                    value=True, key="ncl_alertas_on",
                )
                c_whatsapp = st.text_input(
                    "WhatsApp del cliente (con código país)",
                    key="ncl_whatsapp",
                    placeholder="ej: +54 9 2954 51-7407",
                    help="Formato libre, el sistema lo normaliza. Para "
                          "Argentina, móvil con +549.",
                )
                c_wa_on = st.checkbox(
                    "Recibir alertas críticas por WhatsApp",
                    value=True, key="ncl_wa_on",
                )
                c_notas = st.text_area("Notas", height=70, key="ncl_notas")

                # Mostrar coordenadas si las hay seleccionadas
                lat_sel = st.session_state.get("ncl_lat_sel", 0.0)
                lon_sel = st.session_state.get("ncl_lon_sel", 0.0)
                if lat_sel != 0 or lon_sel != 0:
                    st.success(
                        f"📍 Punto del campo: **{lat_sel:.4f}, {lon_sel:.4f}** "
                        f"(seleccionado en el mapa)"
                    )
                else:
                    st.info(
                        "📍 Marcá el campo en el mapa de abajo (opcional). "
                        "Si no marcás, el sistema busca por la localidad."
                    )

                guardar_cli = st.form_submit_button(
                    "💾 Guardar cliente", type="primary",
                )
                if guardar_cli:
                    if not c_nombre or not c_nombre.strip():
                        st.session_state["ncl_msg"] = (
                            "error",
                            "❌ El nombre es obligatorio.",
                        )
                        st.rerun()
                    else:
                        # Chequear UNIQUE constraint antes de intentar
                        # crear, para dar un error más claro. La columna
                        # nombre tiene UNIQUE en la DB y listar_clientes
                        # devuelve todos (activos + dados de baja).
                        _existentes = db.listar_clientes()
                        _nombres_norm = {
                            (c.get("nombre") or "").strip().lower()
                            for c in _existentes
                        }
                        if c_nombre.strip().lower() in _nombres_norm:
                            st.session_state["ncl_msg"] = (
                                "error",
                                f"❌ Ya existe un cliente con el nombre "
                                f"'{c_nombre}'. Si fue dado de baja, "
                                f"activá 'Mostrar dados de baja' y "
                                f"reactivalo en lugar de crear uno "
                                f"nuevo. Si es realmente distinto, "
                                f"agregale algo (ej. 'Miguel Bergondi - "
                                f"La Pampa').",
                            )
                            st.rerun()
                        else:
                            try:
                                cid = db.crear_cliente(
                                    c_nombre.strip(), c_contacto,
                                    c_estab, c_localidad, c_notas,
                                    email=c_email,
                                    alertas_email_activas=int(
                                        c_alertas_on),
                                    whatsapp=c_whatsapp,
                                    alertas_whatsapp_activas=int(
                                        c_wa_on),
                                )
                                if lat_sel != 0 or lon_sel != 0:
                                    db.actualizar_cliente(
                                        cid, lat=lat_sel, lon=lon_sel,
                                    )
                                # Limpiar selección de coordenadas
                                st.session_state["ncl_lat_sel"] = 0.0
                                st.session_state["ncl_lon_sel"] = 0.0
                                # Flag para limpiar los campos en el
                                # próximo rerun, ANTES de que los
                                # widgets se instancien.
                                st.session_state["ncl_clear_pending"] = True
                                st.session_state["ncl_msg"] = (
                                    "success",
                                    f"✅ Cliente '{c_nombre}' creado "
                                    f"(id #{cid}). Ya aparece en el "
                                    f"listado de la izquierda.",
                                )
                                st.rerun()
                            except Exception as e:
                                import traceback
                                tb = traceback.format_exc()
                                st.session_state["ncl_msg"] = (
                                    "error",
                                    f"❌ Error al guardar: {e}\n\n"
                                    f"```\n{tb}\n```",
                                )
                                st.rerun()

            # Parser de link de WhatsApp / Google Maps (FUERA del form
            # también, porque tiene un botón propio que dispara rerun).
            with st.expander(
                "📍 Pegar link de ubicación (WhatsApp / Google Maps) — más rápido que el mapa",
                expanded=False,
            ):
                st.caption(
                    "Si el cliente te pasó la ubicación por WhatsApp o "
                    "tenés un link de Google Maps, pegalo acá y "
                    "extraemos las coordenadas automáticamente. "
                    "Funciona con varios formatos:\n"
                    "• Link de Google Maps (`maps.google.com/?q=...` "
                    "o `/@lat,lon`)\n"
                    "• Link corto (`maps.app.goo.gl/...`)\n"
                    "• Coordenadas pegadas directo (`-36.42, -63.49`)"
                )
                from src.mapa_widget import parsear_link_ubicacion
                _link_pegado = st.text_input(
                    "Pegá el link o las coordenadas:",
                    key="ncl_link_pegado",
                    placeholder=(
                        "https://maps.app.goo.gl/... o "
                        "-36.42, -63.49"
                    ),
                )
                if st.button(
                    "📌 Extraer coordenadas del link",
                    key="ncl_link_extract",
                    type="secondary",
                ):
                    if not _link_pegado:
                        st.warning("Pegá un link o coordenadas primero.")
                    else:
                        _coords = parsear_link_ubicacion(_link_pegado)
                        if _coords:
                            _lat_p, _lon_p = _coords
                            st.session_state["ncl_lat_sel"] = _lat_p
                            st.session_state["ncl_lon_sel"] = _lon_p
                            st.success(
                                f"✅ Coordenadas extraídas: "
                                f"**{_lat_p:.4f}, {_lon_p:.4f}**. "
                                f"Ahora guardá el cliente con el botón "
                                f"de arriba."
                            )
                        else:
                            st.error(
                                "❌ No pude reconocer ese formato. "
                                "Probá pegar el link completo de "
                                "Google Maps o las coordenadas "
                                "directo (ej. `-36.42, -63.49`)."
                            )

            # Mapa interactivo FUERA del form (st_folium no funciona dentro)
            with st.expander("🗺️ Marcar el campo en el mapa (opcional)",
                              expanded=False):
                st.caption(
                    "Buscá tu campo en el mapa, hacé click sobre el punto "
                    "exacto y las coordenadas se cargan al cliente. "
                    "Cambiá entre vista de calles y satélite con el botón "
                    "arriba a la derecha del mapa."
                )
                lat_clicked, lon_clicked = render_mapa_seleccion(
                    lat_actual=st.session_state.get("ncl_lat_sel", 0.0),
                    lon_actual=st.session_state.get("ncl_lon_sel", 0.0),
                    localidad_busqueda=st.session_state.get("ncl_loc", ""),
                    altura=400,
                    key="mapa_nuevo_cli",
                )
                if lat_clicked is not None and lon_clicked is not None:
                    st.session_state["ncl_lat_sel"] = lat_clicked
                    st.session_state["ncl_lon_sel"] = lon_clicked
                    st.success(
                        f"📍 Punto seleccionado: **{lat_clicked:.4f}, "
                        f"{lon_clicked:.4f}** — guardalo con el botón "
                        f"'Guardar cliente' del formulario de arriba."
                    )

                col_lim1, col_lim2 = st.columns(2)
                if col_lim1.button("🔄 Limpiar punto seleccionado"):
                    st.session_state["ncl_lat_sel"] = 0.0
                    st.session_state["ncl_lon_sel"] = 0.0
                    st.rerun()

        with col_lista:
            st.markdown("##### Listado")
            clientes = db.listar_clientes()
            if not clientes:
                st.info("No hay clientes cargados todavía.")
            else:
                # Filtro: por default no muestra los dados de baja
                col_filtro1, col_filtro2 = st.columns([3, 1])
                mostrar_baja = col_filtro2.checkbox(
                    "Mostrar dados de baja", value=False, key="ver_bajas"
                )

                def _esta_de_baja(c):
                    """True si el cliente fue marcado explícitamente
                    como dado de baja (estado='baja'). Independiente
                    de si tiene o no alertas activadas — un cliente
                    activo puede haber elegido no recibir alertas,
                    y un cliente dado de baja queda archivado aunque
                    todavía tenga datos de contacto cargados.
                    """
                    return (c.get("estado") or "activo") == "baja"

                clientes_visibles = (clientes if mostrar_baja
                                      else [c for c in clientes
                                            if not _esta_de_baja(c)])

                col_filtro1.caption(
                    f"📊 {len(clientes_visibles)} de {len(clientes)} clientes "
                    f"({sum(1 for c in clientes if _esta_de_baja(c))} dados de baja)"
                )

                df_cli = pd.DataFrame([
                    {
                        "ID": c["id"],
                        "Nombre": (f"🔕 {c['nombre']}"
                                   if _esta_de_baja(c) else c["nombre"]),
                        "Establecimiento": c.get("establecimiento", ""),
                        "Localidad": c.get("localidad", ""),
                        "Email": c.get("email", "") or "—",
                        "📧": (
                            "✅" if c.get("alertas_email_activas", 1)
                            and c.get("email") else "—"
                        ),
                        "WhatsApp": c.get("whatsapp", "") or "—",
                        "💬": (
                            "✅" if c.get("alertas_whatsapp_activas", 1)
                            and c.get("whatsapp") else "—"
                        ),
                        "Alta": c.get("fecha_alta", "")[:10],
                    }
                    for c in clientes_visibles
                ])
                st.dataframe(df_cli, hide_index=True, width="stretch")

                cli_para_editar = st.selectbox(
                    "Editar / Eliminar cliente",
                    [None] + [c["id"] for c in clientes],
                    format_func=lambda x: "—" if x is None else
                        next(c["nombre"] for c in clientes if c["id"] == x),
                    key="edit_cli_sel",
                )
                if cli_para_editar:
                    c_data = db.obtener_cliente(cli_para_editar)

                    # Inicializar selección si es la primera vez con este cliente
                    edit_sel_key = f"edit_lat_sel_{cli_para_editar}"
                    if edit_sel_key not in st.session_state:
                        st.session_state[edit_sel_key] = float(c_data.get("lat") or 0)
                        st.session_state[edit_sel_key.replace("lat", "lon")] = float(c_data.get("lon") or 0)

                    edit_lat = st.session_state[edit_sel_key]
                    edit_lon = st.session_state[edit_sel_key.replace("lat", "lon")]

                    # === Estado del cliente: activo / dado de baja ===
                    # Mostrado como banner antes del form de edición.
                    # Los botones de "dar de baja" / "reactivar" están
                    # FUERA del form para que no se confundan con guardar
                    # cambios.
                    _estado_actual = (
                        c_data.get("estado") or "activo"
                    ).lower()
                    if _estado_actual == "baja":
                        _fbaja = c_data.get("fecha_baja") or "—"
                        _mbaja = c_data.get("motivo_baja") or ""
                        st.error(
                            f"🔕 **Cliente dado de baja** "
                            f"(desde {_fbaja})"
                            + (f" — {_mbaja}" if _mbaja else "")
                        )
                        if st.button(
                            "♻️ Reactivar cliente",
                            key=f"reactivar_{cli_para_editar}",
                            type="primary",
                        ):
                            db.reactivar_cliente(cli_para_editar)
                            st.success(
                                f"Cliente '{c_data['nombre']}' "
                                f"reactivado. Revisá si querés volver "
                                f"a prender las alertas por email / "
                                f"WhatsApp."
                            )
                            st.rerun()
                    else:
                        st.success(
                            f"✅ **Cliente activo** — "
                            f"{c_data['nombre']}"
                        )

                    with st.form(f"edit_cli_{cli_para_editar}"):
                        e_nombre = st.text_input("Nombre", c_data["nombre"])
                        e_estab = st.text_input("Establecimiento",
                                                 c_data.get("establecimiento") or "")
                        e_loc = st.text_input("Localidad",
                                               c_data.get("localidad") or "")
                        e_cont = st.text_input("Contacto",
                                                c_data.get("contacto") or "")
                        e_email = st.text_input(
                            "Email para alertas climáticas",
                            c_data.get("email") or "",
                            placeholder="cliente@dominio.com",
                        )
                        e_alertas_on = st.checkbox(
                            "Recibir alertas climáticas por email",
                            value=bool(c_data.get("alertas_email_activas", 1)),
                        )
                        e_whatsapp = st.text_input(
                            "WhatsApp del cliente",
                            c_data.get("whatsapp") or "",
                            placeholder="+54 9 2954 51-7407",
                        )
                        e_wa_on = st.checkbox(
                            "Recibir alertas críticas por WhatsApp",
                            value=bool(c_data.get("alertas_whatsapp_activas", 1)),
                        )
                        e_notas = st.text_area("Notas",
                                                c_data.get("notas") or "")

                        if edit_lat != 0 or edit_lon != 0:
                            st.success(
                                f"📍 Coordenadas: **{edit_lat:.4f}, {edit_lon:.4f}**"
                            )
                        else:
                            st.info(
                                "📍 Sin coordenadas manuales. El sistema buscará "
                                "por la localidad. Marcá un punto en el mapa de abajo "
                                "si querés precisión."
                            )

                        col_b1, col_b2, col_b3 = st.columns([2, 2, 1])
                        if col_b1.form_submit_button(
                            "💾 Guardar cambios", type="primary",
                        ):
                            campos = dict(
                                nombre=e_nombre, establecimiento=e_estab,
                                localidad=e_loc, contacto=e_cont, notas=e_notas,
                                email=e_email,
                                alertas_email_activas=int(e_alertas_on),
                                whatsapp=e_whatsapp,
                                alertas_whatsapp_activas=int(e_wa_on),
                            )
                            if edit_lat != 0 or edit_lon != 0:
                                campos["lat"] = edit_lat
                                campos["lon"] = edit_lon
                            else:
                                campos["lat"] = None
                                campos["lon"] = None
                            db.actualizar_cliente(cli_para_editar, **campos)
                            # Borrar cache de geocoding
                            try:
                                from src.clima import GEOCODE_CACHE
                                if GEOCODE_CACHE.exists():
                                    GEOCODE_CACHE.unlink()
                            except Exception:
                                pass
                            st.success("Actualizado")
                            st.rerun()
                        # Botón "Dar de baja" solo si el cliente está
                        # activo. Si ya está dado de baja, el botón de
                        # reactivar aparece arriba (fuera del form).
                        if _estado_actual != "baja":
                            if col_b2.form_submit_button(
                                "🔕 Dar de baja",
                                help=(
                                    "Archiva al cliente (no aparece "
                                    "en la tabla principal). Se "
                                    "puede reactivar más tarde. "
                                    "Para borrarlo definitivamente, "
                                    "usá el botón rojo."
                                ),
                            ):
                                db.dar_de_baja_cliente(
                                    cli_para_editar,
                                    motivo="Baja manual desde la ficha",
                                    desactivar_alertas=True,
                                )
                                st.warning(
                                    f"Cliente '{c_data['nombre']}' "
                                    f"dado de baja. Las alertas se "
                                    f"desactivaron. Se puede "
                                    f"reactivar después si "
                                    f"vuelve a comprar."
                                )
                                st.rerun()
                        else:
                            col_b2.caption(
                                "Para reactivar usá el botón de arriba"
                            )
                        if col_b3.form_submit_button(
                            "🗑️", help="Borrar definitivamente (irreversible)",
                        ):
                            db.eliminar_cliente(cli_para_editar)
                            st.warning("Eliminado")
                            st.rerun()

                    # Mapa interactivo FUERA del form
                    # Parser de link (más rápido si el cliente lo
                    # pasa por WhatsApp).
                    with st.expander(
                        "📍 Pegar link de ubicación (WhatsApp / Google Maps)",
                        expanded=False,
                    ):
                        st.caption(
                            "Si el cliente te pasó la ubicación por "
                            "WhatsApp o tenés un link de Google Maps, "
                            "pegalo y extraemos las coordenadas al toque."
                        )
                        from src.mapa_widget import (
                            parsear_link_ubicacion as _parse_link_edit,
                        )
                        _link_edit = st.text_input(
                            "Pegá el link o las coordenadas:",
                            key=f"edit_link_pegado_{cli_para_editar}",
                            placeholder=(
                                "https://maps.app.goo.gl/... o "
                                "-36.42, -63.49"
                            ),
                        )
                        if st.button(
                            "📌 Extraer coordenadas",
                            key=f"edit_link_extract_{cli_para_editar}",
                            type="secondary",
                        ):
                            if not _link_edit:
                                st.warning(
                                    "Pegá un link o coordenadas primero."
                                )
                            else:
                                _coords_e = _parse_link_edit(_link_edit)
                                if _coords_e:
                                    _lat_e, _lon_e = _coords_e
                                    st.session_state[edit_sel_key] = _lat_e
                                    st.session_state[
                                        edit_sel_key.replace("lat", "lon")
                                    ] = _lon_e
                                    st.success(
                                        f"✅ Coordenadas: "
                                        f"**{_lat_e:.4f}, {_lon_e:.4f}**. "
                                        f"Guardá los cambios con el botón "
                                        f"de arriba del form."
                                    )
                                    st.rerun()
                                else:
                                    st.error(
                                        "❌ Formato no reconocido. "
                                        "Probá con el link completo de "
                                        "Google Maps o coordenadas "
                                        "directas (ej. `-36.42, -63.49`)."
                                    )

                    with st.expander(
                        "🗺️ Marcar / cambiar punto del campo en el mapa",
                        expanded=False,
                    ):
                        st.caption(
                            "Click sobre el campo en el mapa para fijar las "
                            "coordenadas. Vista satelital disponible (botón "
                            "arriba derecha del mapa)."
                        )
                        lat_clk, lon_clk = render_mapa_seleccion(
                            lat_actual=edit_lat,
                            lon_actual=edit_lon,
                            localidad_busqueda=c_data.get("localidad", ""),
                            altura=400,
                            key=f"mapa_edit_{cli_para_editar}",
                        )
                        if lat_clk is not None and lon_clk is not None:
                            st.session_state[edit_sel_key] = lat_clk
                            st.session_state[edit_sel_key.replace("lat", "lon")] = lon_clk
                            st.success(
                                f"📍 Nuevo punto: **{lat_clk:.4f}, {lon_clk:.4f}** "
                                "— ahora hacé click en 'Guardar cambios' arriba."
                            )

                        if st.button(
                            "🔄 Volver a búsqueda automática (limpiar coordenadas)",
                            key=f"clear_coords_{cli_para_editar}",
                        ):
                            st.session_state[edit_sel_key] = 0.0
                            st.session_state[edit_sel_key.replace("lat", "lon")] = 0.0
                            st.rerun()

                    # ============================================================
                    #     DEMANDA CONSOLIDADA DE INSUMOS POR CLIENTE
                    # ============================================================
                    # Vista por lote/corral + total cliente para planificación
                    # de logística. Distinguimos productos HMS (los que Mauricio
                    # vende y coordina) del resto (maíz, rollo, los compra el
                    # productor por su lado).
                    with st.expander(
                        "📊 Demanda diaria de insumos — por lote y total cliente",
                        expanded=False,
                    ):
                        st.caption(
                            "Vista consolidada de los lotes activos del "
                            "cliente con sus dietas vigentes. Muestra kg "
                            "por animal y kg totales del lote, marcando "
                            "los productos HMS (los que coordinás vos)."
                        )
                        from src.stock_producto import (
                            demanda_insumos_cliente,
                        )
                        _demanda = demanda_insumos_cliente(
                            cli_para_editar,
                        )
                        if not _demanda["lotes"]:
                            st.info(
                                "Este cliente no tiene lotes activos "
                                "con dieta cargada. Cargá un lote y al "
                                "menos una dieta para ver la demanda."
                            )
                        else:
                            # KPIs cabecera
                            _tot = _demanda["total_cliente"]
                            _k1, _k2, _k3 = st.columns(3)
                            _k1.metric(
                                "Lotes activos",
                                len(_demanda["lotes"]),
                            )
                            _k2.metric(
                                "Animales totales",
                                _tot["cantidad_animales_total"],
                            )
                            _k3.metric(
                                "Mezcla total / día",
                                f"{_tot['mezcla_total_kg_dia']:,.0f} kg"
                                .replace(",", "."),
                                help="Suma de todos los lotes, excluye "
                                     "forrajes a libre disposición.",
                            )

                            # Tabla por lote
                            st.markdown("##### 🐄 Por lote / corral")
                            for _lt in _demanda["lotes"]:
                                _label_fase = (
                                    f" · {_lt['fase_vigente']}"
                                    if _lt['fase_vigente'] else ""
                                )
                                st.markdown(
                                    f"**{_lt['lote_ident']}** "
                                    f"— {_lt['categoria']} · "
                                    f"{_lt['cantidad_animales']} cab."
                                    f"{_label_fase}"
                                )
                                _filas_lote = []
                                for _ing in _lt["ingredientes"]:
                                    if _ing["es_libre_disposicion"]:
                                        _pct = "—"
                                        _kg_a = "libre disposición"
                                        _kg_d = "—"
                                        _kg_s = "—"
                                    else:
                                        _pct = (
                                            f"{_ing.get('pct_mezcla', 0):.1f}%"
                                        )
                                        _kg_a = (
                                            f"{_ing['kg_animal_dia']:.2f}"
                                        )
                                        _kg_d = (
                                            f"{_ing['kg_lote_dia']:,.1f}"
                                            .replace(",", ".")
                                        )
                                        _kg_s = (
                                            f"{_ing['kg_lote_semana']:,.0f}"
                                            .replace(",", ".")
                                        )
                                    _filas_lote.append({
                                        "Ingrediente": _ing["nombre"],
                                        "HMS": (
                                            "✓" if _ing["es_hms"]
                                            else ""
                                        ),
                                        "% mezcla": _pct,
                                        "kg / animal · día": _kg_a,
                                        "kg / lote · día": _kg_d,
                                        "kg / lote · semana": _kg_s,
                                    })
                                _df_lote = pd.DataFrame(_filas_lote)
                                st.dataframe(
                                    _df_lote, hide_index=True,
                                    width="stretch",
                                )
                                st.caption(
                                    f"_Mezcla del lote: "
                                    f"{_lt['mezcla_total_kg_dia']:,.0f} "
                                    f"kg/día (sin contar libre "
                                    f"disposición)._".replace(",", ".")
                                )
                                st.markdown("")

                            # Tabla total cliente consolidada
                            st.markdown(
                                "##### 🧮 Total cliente "
                                "(suma de todos los lotes)"
                            )
                            _filas_tot = []
                            for _ing in _tot["ingredientes"]:
                                if _ing["es_libre_disposicion"]:
                                    _kg_d = "libre disposición"
                                    _kg_s = "—"
                                    _kg_m = "—"
                                else:
                                    _kg_d = (
                                        f"{_ing['kg_dia']:,.1f}"
                                        .replace(",", ".")
                                    )
                                    _kg_s = (
                                        f"{_ing['kg_semana']:,.0f}"
                                        .replace(",", ".")
                                    )
                                    _kg_m = (
                                        f"{_ing['kg_mes']:,.0f}"
                                        .replace(",", ".")
                                    )
                                _filas_tot.append({
                                    "Ingrediente": _ing["nombre"],
                                    "HMS": (
                                        "✓" if _ing["es_hms"] else ""
                                    ),
                                    "En N lotes":
                                        _ing["lotes_que_lo_usan"],
                                    "kg / día": _kg_d,
                                    "kg / semana": _kg_s,
                                    "kg / mes": _kg_m,
                                })
                            _df_tot = pd.DataFrame(_filas_tot)
                            st.dataframe(
                                _df_tot, hide_index=True,
                                width="stretch",
                            )
                            st.caption(
                                "_La columna **HMS ✓** marca los "
                                "productos que vos vendés y coordinás. "
                                "Esa es la demanda real de tu logística._"
                            )

                    # ============================================================
                    #     ENTREGAS Y STOCK DE PRODUCTO (concentrados/núcleos)
                    # ============================================================
                    with st.expander(
                        "📦 Entregas y stock de producto — Fibrogreen, Fibroter, otros",
                        expanded=False,
                    ):
                        st.caption(
                            "Registrá las entregas de producto al "
                            "cliente y el sistema te calcula stock "
                            "actual, consumo diario (según última "
                            "dieta del lote × cabezas × % inclusión) "
                            "y días estimados de agotamiento. Si el "
                            "consumo real difiere del teórico te lo "
                            "marca como sub-uso o sobre-uso."
                        )

                        from src.stock_producto import (
                            calcular_stock_actual,
                            listar_productos_lote,
                            listar_productos_hms_lote,
                            calcular_consumo_diario_kg,
                        )

                        # === Form: registrar entrega nueva ===
                        st.markdown("##### ➕ Registrar entrega nueva")
                        _lotes_act_cli = db.listar_lotes(
                            cliente_id=cli_para_editar,
                        )
                        if not _lotes_act_cli:
                            st.info(
                                "Cargá al menos un lote para este "
                                "cliente antes de registrar entregas."
                            )
                        else:
                            with st.form(
                                f"entrega_nueva_{cli_para_editar}",
                                clear_on_submit=True,
                            ):
                                _col_e1, _col_e2, _col_e3 = st.columns(3)
                                with _col_e1:
                                    _opc_lotes = {
                                        f"{l['identificador']} ({l.get('categoria','')})":
                                        l["id"]
                                        for l in _lotes_act_cli
                                    }
                                    _opc_lotes_keys = (
                                        ["— Sin asociar a lote —"]
                                        + list(_opc_lotes.keys())
                                    )
                                    _lote_sel_ent = st.selectbox(
                                        "Lote", _opc_lotes_keys,
                                    )
                                    _lote_id_ent = (
                                        _opc_lotes.get(_lote_sel_ent)
                                        if _lote_sel_ent != "— Sin asociar a lote —"
                                        else None
                                    )
                                with _col_e2:
                                    _producto_ent = st.text_input(
                                        "Producto",
                                        placeholder="Fibrogreen / Fibroter / otro",
                                    )
                                with _col_e3:
                                    _fecha_ent = st.date_input(
                                        "Fecha de entrega",
                                        value=datetime.now().date(),
                                    )

                                _col_e4, _col_e5, _col_e6 = st.columns(3)
                                with _col_e4:
                                    _formato_ent = st.selectbox(
                                        "Formato",
                                        ["bolsa", "granel"],
                                    )
                                with _col_e5:
                                    if _formato_ent == "bolsa":
                                        _bolsas = st.number_input(
                                            "Cantidad de bolsas",
                                            min_value=0.0,
                                            step=1.0, value=0.0,
                                        )
                                        _kg_bolsa = st.number_input(
                                            "Kg por bolsa",
                                            min_value=0.0,
                                            step=1.0, value=30.0,
                                        )
                                        _kg_total_ent = (
                                            _bolsas * _kg_bolsa
                                        )
                                    else:
                                        _kg_total_ent = st.number_input(
                                            "Kg granel",
                                            min_value=0.0,
                                            step=10.0, value=0.0,
                                        )
                                        _bolsas = 0
                                        _kg_bolsa = 0
                                with _col_e6:
                                    _precio_kg = st.number_input(
                                        "Precio $/kg (opcional)",
                                        min_value=0.0, step=10.0,
                                        value=0.0,
                                    )

                                _notas_ent = st.text_input(
                                    "Notas (opcional)",
                                    placeholder="Ej: entregado por transporte propio, remito #1234",
                                )

                                if st.form_submit_button(
                                    "📦 Registrar entrega",
                                    type="primary",
                                ):
                                    if not _producto_ent:
                                        st.error("Falta el nombre del producto.")
                                    elif _kg_total_ent <= 0:
                                        st.error("La cantidad debe ser mayor a 0.")
                                    else:
                                        try:
                                            db.crear_entrega(
                                                cliente_id=cli_para_editar,
                                                lote_id=_lote_id_ent,
                                                producto_nombre=_producto_ent,
                                                kg_total=_kg_total_ent,
                                                fecha_entrega=_fecha_ent.isoformat(),
                                                formato=_formato_ent,
                                                cantidad_bolsas=_bolsas,
                                                kg_por_bolsa=_kg_bolsa,
                                                precio_kg=_precio_kg,
                                                precio_total=(
                                                    _precio_kg * _kg_total_ent
                                                ),
                                                notas=_notas_ent,
                                            )
                                            st.success(
                                                f"✅ Entrega registrada: "
                                                f"{_kg_total_ent:.0f} kg de "
                                                f"{_producto_ent}"
                                            )
                                            st.rerun()
                                        except Exception as e_ent:
                                            st.error(f"Error: {e_ent}")

                        # === Tabla: stock actual por lote × producto ===
                        st.markdown(
                            "##### 📊 Stock actual y proyección por lote"
                        )
                        _filas_stock = []
                        for _l_st in _lotes_act_cli:
                            if _l_st.get("estado") != "activo":
                                continue
                            # Solo productos que HMS efectivamente vendió
                            # (no maíz, rollos, silaje que el productor
                            # compra por su lado).
                            _productos_lote = listar_productos_hms_lote(
                                cli_para_editar, _l_st["id"]
                            )
                            if not _productos_lote:
                                continue
                            for _prod in _productos_lote:
                                _stock = calcular_stock_actual(
                                    cli_para_editar, _l_st["id"], _prod,
                                )
                                if not _stock:
                                    continue
                                _diag = _stock.get(
                                    "diagnostico_uso", "")
                                _ico_diag = {
                                    "sin_entregas": "⚪",
                                    "normal": "🟢",
                                    "sub_uso": "🟡",
                                    "sobre_uso": "🔴",
                                }.get(_diag, "⚪")
                                _alerta = ""
                                if _stock["dias_restantes"] <= 7 and \
                                   _stock["kg_restantes_hoy"] > 0:
                                    _alerta = "⚠️"
                                elif _stock["kg_restantes_hoy"] <= 0:
                                    _alerta = "🔴"
                                _filas_stock.append({
                                    "Lote": _l_st["identificador"],
                                    "Producto": _prod,
                                    "Consumo hoy (kg/día)":
                                        f"{_stock['consumo_diario_kg']:.1f}",
                                    "Entregado (kg)":
                                        f"{_stock['kg_entregados_total']:.0f}",
                                    "Stock (kg)":
                                        f"{_alerta} {_stock['kg_restantes_hoy']:.0f}",
                                    "Días rest.":
                                        f"{_stock['dias_restantes']:.0f}",
                                    "Se acaba":
                                        _stock.get(
                                            "fecha_agotamiento") or "—",
                                })
                        if _filas_stock:
                            import pandas as _pd_stock
                            st.dataframe(
                                _pd_stock.DataFrame(_filas_stock),
                                hide_index=True, width="stretch",
                            )
                            # Alertas visibles arriba si hay stock bajo
                            _alertas_stock = [
                                f for f in _filas_stock
                                if "⚠️" in f["Stock (kg)"]
                                or "🔴" in f["Stock (kg)"]
                            ]
                            if _alertas_stock:
                                st.warning(
                                    f"⚠️ **{len(_alertas_stock)} alerta(s) "
                                    f"de stock bajo o agotado.** "
                                    f"Coordinar próxima entrega cuanto antes."
                                )
                        else:
                            st.caption(
                                "No hay lotes activos con dieta cargada "
                                "+ entregas. Formulá una dieta para el "
                                "lote y registrá una entrega para ver "
                                "el cálculo."
                            )

                        # === Histórico de entregas ===
                        st.markdown("##### 📜 Histórico de entregas")
                        _todas_entregas = db.listar_entregas_cliente(
                            cli_para_editar, limit=50,
                        )
                        if not _todas_entregas:
                            st.caption(
                                "Todavía no hay entregas registradas."
                            )
                        else:
                            import pandas as _pd_e
                            _df_e = _pd_e.DataFrame([
                                {
                                    "Fecha": e.get("fecha_entrega", "")[:10],
                                    "Lote": e.get(
                                        "lote_identificador") or "— Sin asociar —",
                                    "Producto": e.get("producto_nombre"),
                                    "Formato": e.get("formato"),
                                    "Cantidad":
                                        (f"{e.get('cantidad_bolsas') or 0:.0f} bolsas × "
                                         f"{e.get('kg_por_bolsa') or 0:.0f} kg"
                                         if e.get("formato") == "bolsa"
                                         else f"{e.get('kg_total') or 0:.0f} kg granel"),
                                    "Kg total":
                                        f"{e.get('kg_total') or 0:.0f}",
                                    "$ total":
                                        f"${e.get('precio_total') or 0:,.0f}"
                                        if e.get("precio_total")
                                        else "—",
                                    "Notas": e.get("notas") or "",
                                }
                                for e in _todas_entregas
                            ])
                            st.dataframe(
                                _df_e, hide_index=True,
                                width="stretch",
                            )
                            _ent_ids = {
                                f"{e.get('fecha_entrega','')[:10]} · "
                                f"{e.get('producto_nombre')} · "
                                f"{e.get('kg_total',0):.0f} kg": e["id"]
                                for e in _todas_entregas
                            }

                            # ───── Editar entrega existente ─────
                            # Permite corregir precio, cantidad, fecha
                            # o notas sin tener que borrar y recrear.
                            with st.expander(
                                "✏️ Editar entrega existente "
                                "(corregir precio, cantidad, fecha)",
                                expanded=False,
                            ):
                                _ed_lbl = st.selectbox(
                                    "¿Cuál entrega querés editar?",
                                    ["—"] + list(_ent_ids.keys()),
                                    key=(f"ed_ent_sel_"
                                         f"{cli_para_editar}"),
                                )
                                if _ed_lbl and _ed_lbl != "—":
                                    _ent_id_ed = _ent_ids[_ed_lbl]
                                    _ent_ed = db.obtener_entrega(
                                        _ent_id_ed)
                                    if _ent_ed:
                                        with st.form(
                                            f"edit_ent_form_"
                                            f"{_ent_id_ed}",
                                        ):
                                            _ec1, _ec2 = st.columns(2)
                                            with _ec1:
                                                _ed_fecha = (
                                                    st.date_input(
                                                        "Fecha de entrega",
                                                        value=datetime.strptime(
                                                            _ent_ed[
                                                                "fecha_entrega"
                                                            ][:10],
                                                            "%Y-%m-%d",
                                                        ).date(),
                                                        key=(f"ed_fec_"
                                                             f"{_ent_id_ed}"),
                                                    )
                                                )
                                                _ed_producto = (
                                                    st.text_input(
                                                        "Producto",
                                                        value=_ent_ed.get(
                                                            "producto_nombre",
                                                            "") or "",
                                                        key=(f"ed_prod_"
                                                             f"{_ent_id_ed}"),
                                                    )
                                                )
                                                _ed_formato = (
                                                    st.selectbox(
                                                        "Formato",
                                                        ["bolsa", "granel"],
                                                        index=0 if
                                                        (_ent_ed.get(
                                                            "formato") or
                                                         "bolsa") == "bolsa"
                                                        else 1,
                                                        key=(f"ed_fmt_"
                                                             f"{_ent_id_ed}"),
                                                    )
                                                )
                                            with _ec2:
                                                if _ed_formato == "bolsa":
                                                    _ed_bolsas = (
                                                        st.number_input(
                                                            "Cantidad de bolsas",
                                                            min_value=0.0,
                                                            step=1.0,
                                                            value=float(
                                                                _ent_ed.get(
                                                                    "cantidad_bolsas",
                                                                    0,
                                                                ) or 0),
                                                            key=(f"ed_bls_"
                                                                 f"{_ent_id_ed}"),
                                                        )
                                                    )
                                                    _ed_kgb = (
                                                        st.number_input(
                                                            "Kg por bolsa",
                                                            min_value=0.0,
                                                            step=1.0,
                                                            value=float(
                                                                _ent_ed.get(
                                                                    "kg_por_bolsa",
                                                                    30,
                                                                ) or 30),
                                                            key=(f"ed_kgb_"
                                                                 f"{_ent_id_ed}"),
                                                        )
                                                    )
                                                    _ed_kg_total = (
                                                        _ed_bolsas * _ed_kgb
                                                    )
                                                    # Precio actual por bolsa
                                                    _pkg_actual = float(
                                                        _ent_ed.get(
                                                            "precio_kg", 0)
                                                        or 0)
                                                    _pbolsa_actual = (
                                                        _pkg_actual * _ed_kgb
                                                    )
                                                    _ed_precio_bolsa = (
                                                        st.number_input(
                                                            f"Precio por bolsa de {_ed_kgb:.0f} kg",
                                                            min_value=0.0,
                                                            step=100.0,
                                                            value=float(
                                                                _pbolsa_actual),
                                                            key=(f"ed_pbol_"
                                                                 f"{_ent_id_ed}"),
                                                            help=(
                                                                "El sistema "
                                                                "convierte a "
                                                                "$/kg "
                                                                "automáticamente."
                                                            ),
                                                        )
                                                    )
                                                    _ed_pkg = (
                                                        _ed_precio_bolsa /
                                                        _ed_kgb
                                                        if _ed_kgb > 0
                                                        else 0
                                                    )
                                                else:
                                                    _ed_kg_total = (
                                                        st.number_input(
                                                            "Kg granel",
                                                            min_value=0.0,
                                                            step=10.0,
                                                            value=float(
                                                                _ent_ed.get(
                                                                    "kg_total",
                                                                    0,
                                                                ) or 0),
                                                            key=(f"ed_kgg_"
                                                                 f"{_ent_id_ed}"),
                                                        )
                                                    )
                                                    _ed_bolsas = 0
                                                    _ed_kgb = 0
                                                    _ed_pkg = (
                                                        st.number_input(
                                                            "Precio $/kg",
                                                            min_value=0.0,
                                                            step=10.0,
                                                            value=float(
                                                                _ent_ed.get(
                                                                    "precio_kg",
                                                                    0) or 0),
                                                            key=(f"ed_pkg_"
                                                                 f"{_ent_id_ed}"),
                                                        )
                                                    )
                                            _ed_notas = st.text_input(
                                                "Notas",
                                                value=_ent_ed.get(
                                                    "notas", "") or "",
                                                key=(f"ed_not_"
                                                     f"{_ent_id_ed}"),
                                            )
                                            _ed_ptotal = (
                                                _ed_pkg * _ed_kg_total
                                            )
                                            st.caption(
                                                f"📦 Total: "
                                                f"**{_ed_kg_total:.0f} kg** "
                                                f"= **${_ed_pkg:,.0f}/kg** "
                                                f"· **${_ed_ptotal:,.0f}** "
                                                f"total"
                                            )
                                            if st.form_submit_button(
                                                "💾 Guardar cambios",
                                                type="primary",
                                            ):
                                                try:
                                                    db.actualizar_entrega(
                                                        _ent_id_ed,
                                                        fecha_entrega=(
                                                            _ed_fecha
                                                            .isoformat()),
                                                        producto_nombre=(
                                                            _ed_producto
                                                            .strip()),
                                                        formato=_ed_formato,
                                                        cantidad_bolsas=(
                                                            _ed_bolsas),
                                                        kg_por_bolsa=_ed_kgb,
                                                        kg_total=_ed_kg_total,
                                                        precio_kg=_ed_pkg,
                                                        precio_total=(
                                                            _ed_ptotal),
                                                        notas=_ed_notas,
                                                    )
                                                    st.success(
                                                        f"✅ Entrega "
                                                        f"actualizada. "
                                                        f"Nuevo total: "
                                                        f"${_ed_ptotal:,.0f}"
                                                    )
                                                    st.rerun()
                                                except Exception as _e:
                                                    st.error(
                                                        f"Error: {_e}"
                                                    )

                            # ───── Borrar entrega puntual ─────
                            _del_lbl = st.selectbox(
                                "Borrar entrega (opcional):",
                                ["—"] + list(_ent_ids.keys()),
                                key=f"del_ent_sel_{cli_para_editar}",
                            )
                            if _del_lbl and _del_lbl != "—":
                                if st.button(
                                    "🗑️ Borrar entrega seleccionada",
                                    key=f"del_ent_btn_{cli_para_editar}",
                                    type="secondary",
                                ):
                                    db.eliminar_entrega(
                                        _ent_ids[_del_lbl]
                                    )
                                    st.success("Entrega eliminada.")
                                    st.rerun()

                    # ============================================================
                    #     HISTORIAL PRODUCTIVO DEL CLIENTE
                    # ============================================================
                    with st.expander(
                        "📚 Historial productivo del cliente — todos los lotes y animales manejados",
                        expanded=False,
                    ):
                        st.caption(
                            "Vista cronológica de TODOS los lotes que "
                            "pasaron por el campo (activos + cerrados + "
                            "finalizados), con cantidad de animales, "
                            "fechas y categorías. Sirve para ver la "
                            "historia productiva del establecimiento "
                            "más allá del lote actual."
                        )

                        # Traer TODOS los lotes del cliente (sin filtro
                        # de estado para incluir cerrados/finalizados)
                        _todos_hist = db.listar_lotes(
                            cliente_id=cli_para_editar,
                        )
                        if not _todos_hist:
                            st.info(
                                "Todavía no hay lotes cargados para "
                                "este cliente. Cargá uno en la "
                                "pestaña Lotes."
                            )
                        else:
                            from datetime import (
                                datetime as _dt_hist,
                                date as _date_hist,
                            )
                            # ── KPIs anuales ──
                            _hoy_hist = _dt_hist.now().date()
                            _ano = _hoy_hist.year
                            _ini_ano = _date_hist(_ano, 1, 1)
                            _fin_ano = _date_hist(_ano, 12, 31)

                            def _parse_d(s):
                                try:
                                    return _dt_hist.strptime(
                                        s[:10], "%Y-%m-%d"
                                    ).date()
                                except (ValueError, TypeError, AttributeError):
                                    return None

                            _total_anuales = 0
                            _activos = 0
                            _cerrados_ano = 0
                            _en_campo_hoy = 0
                            for _l_h in _todos_hist:
                                _cant_h = _l_h.get("cantidad_inicial") or 0
                                _fi_h = _parse_d(_l_h.get("fecha_ingreso", ""))
                                _fs_h = _parse_d(_l_h.get("fecha_salida", ""))
                                _est_h = _l_h.get("estado", "activo")
                                # Animales que ingresaron este año
                                if _fi_h and _ini_ano <= _fi_h <= _fin_ano:
                                    _total_anuales += _cant_h
                                # Lotes activos hoy
                                if _est_h == "activo":
                                    _activos += 1
                                    _en_campo_hoy += _cant_h
                                # Lotes cerrados este año
                                if _est_h in ("cerrado", "finalizado"):
                                    if _fs_h and _ini_ano <= _fs_h <= _fin_ano:
                                        _cerrados_ano += 1

                            _k1, _k2, _k3, _k4 = st.columns(4)
                            _k1.metric(
                                f"Animales ingresados {_ano}",
                                f"{_total_anuales}",
                                help=(
                                    f"Suma de cabezas de todos los "
                                    f"lotes ingresados entre 01/01/"
                                    f"{_ano} y 31/12/{_ano}."
                                ),
                            )
                            _k2.metric(
                                "En campo HOY",
                                f"{_en_campo_hoy} cab.",
                                help="Suma de cantidad inicial de lotes en estado activo.",
                            )
                            _k3.metric(
                                "Lotes activos",
                                f"{_activos}",
                            )
                            _k4.metric(
                                f"Lotes cerrados {_ano}",
                                f"{_cerrados_ano}",
                            )

                            # ── Tabla cronológica ──
                            st.markdown(
                                "##### Cronología de lotes "
                                "(más reciente primero)"
                            )
                            _filas_h = []
                            for _l_h in sorted(
                                _todos_hist,
                                key=lambda x: x.get(
                                    "fecha_ingreso", "") or "",
                                reverse=True,
                            ):
                                _fi = _parse_d(
                                    _l_h.get("fecha_ingreso", ""))
                                _fs = _parse_d(
                                    _l_h.get("fecha_salida", ""))
                                # Días en el campo: si está activo,
                                # cuenta hasta hoy; si está cerrado,
                                # cuenta entre ingreso y salida.
                                _dias = ""
                                if _fi:
                                    if _fs:
                                        _dias = (_fs - _fi).days
                                    else:
                                        _dias = (_hoy_hist - _fi).days
                                _est = _l_h.get("estado", "activo")
                                _ico_e = {
                                    "activo": "🟢",
                                    "cerrado": "🔒",
                                    "finalizado": "✅",
                                }.get(_est, "⚪")
                                _filas_h.append({
                                    "Lote": _l_h.get(
                                        "identificador", ""),
                                    "Categoría": _l_h.get(
                                        "categoria", "—"),
                                    "Raza": _l_h.get("raza", "—"),
                                    "Cab.": _l_h.get(
                                        "cantidad_inicial") or 0,
                                    "Peso ingreso (kg)":
                                        f"{_l_h.get('peso_ingreso_kg') or 0:.0f}",
                                    "Último peso (kg)":
                                        f"{_l_h.get('ultimo_peso_kg'):.0f}"
                                        if _l_h.get("ultimo_peso_kg")
                                        else "—",
                                    "Pesadas": _l_h.get("n_pesadas") or 0,
                                    "Ingreso": (_fi.strftime("%d/%m/%y")
                                                 if _fi else "—"),
                                    "Salida": (_fs.strftime("%d/%m/%y")
                                                if _fs else "—"),
                                    "Días en campo": _dias,
                                    "Estado": f"{_ico_e} {_est}",
                                })
                            import pandas as _pd_h
                            _df_h = _pd_h.DataFrame(_filas_h)
                            st.dataframe(
                                _df_h, hide_index=True,
                                width="stretch",
                            )

                            # ── Timeline mensual ──
                            with st.expander(
                                "📊 Carga mensual de animales en el campo",
                                expanded=False,
                            ):
                                st.caption(
                                    "Cantidad de cabezas presentes en "
                                    "el campo mes a mes en el año "
                                    f"{_ano}. Sumamos lotes activos "
                                    "en cada mes."
                                )
                                _meses_lbl = [
                                    "Ene", "Feb", "Mar", "Abr",
                                    "May", "Jun", "Jul", "Ago",
                                    "Sep", "Oct", "Nov", "Dic",
                                ]
                                _serie_mensual = []
                                for _m in range(1, 13):
                                    _ref = _date_hist(_ano, _m, 15)
                                    _cab_mes = 0
                                    for _l_h in _todos_hist:
                                        _fi = _parse_d(
                                            _l_h.get("fecha_ingreso", ""))
                                        _fs = _parse_d(
                                            _l_h.get("fecha_salida", ""))
                                        if not _fi:
                                            continue
                                        if _fi <= _ref and (
                                            _fs is None or _fs >= _ref
                                        ):
                                            _cab_mes += (
                                                _l_h.get(
                                                    "cantidad_inicial") or 0
                                            )
                                    _serie_mensual.append(_cab_mes)
                                _df_m = _pd_h.DataFrame({
                                    "Mes": _meses_lbl,
                                    "Cabezas": _serie_mensual,
                                })
                                st.bar_chart(
                                    _df_m, x="Mes", y="Cabezas",
                                    width="stretch",
                                )

                    # ============================================================
                    # ============================================================
                    #     HISTORIAL DE LLAMADAS A ESTE CLIENTE
                    # ============================================================
                    with st.expander(
                        "📞 Historial de llamadas registradas",
                        expanded=False,
                    ):
                        st.caption(
                            "Registro de los contactos con este "
                            "cliente: qué se conversó, qué se acordó "
                            "y próximos pasos. Sirve para retomar el "
                            "hilo en la próxima llamada."
                        )
                        _recos_cli = db.listar_recordatorios_cliente(
                            cli_para_editar,
                            incluir_completados=True,
                        )
                        if not _recos_cli:
                            st.info(
                                "Sin registros de llamadas. "
                                "Programá uno desde el dashboard "
                                "o desde el botón de abajo."
                            )
                        else:
                            # Separar pendientes / hechos /
                            # cancelados para que se vea más claro
                            _pend = [
                                r for r in _recos_cli
                                if r.get("estado") == "pendiente"
                            ]
                            _hechos = [
                                r for r in _recos_cli
                                if r.get("estado") == "hecho"
                            ]
                            _cancel = [
                                r for r in _recos_cli
                                if r.get("estado") == "cancelado"
                            ]

                            if _pend:
                                st.markdown(
                                    "##### 📌 Pendientes"
                                )
                                for _r_h in _pend:
                                    st.markdown(
                                        f"- **{_r_h['fecha_objetivo']}** "
                                        f"· {_r_h.get('motivo','—')[:140]}"
                                    )

                            if _hechos:
                                st.markdown(
                                    f"##### ✅ Llamadas hechas "
                                    f"({len(_hechos)})"
                                )
                                for _r_h in _hechos[:20]:
                                    _f_h = (
                                        (_r_h.get('completado_en')
                                         or '')[:10]
                                        or _r_h['fecha_objetivo']
                                    )
                                    with st.expander(
                                        f"📞 {_f_h} — "
                                        f"{(_r_h.get('motivo','') or '—')[:80]}"
                                    ):
                                        _notas_h = (
                                            _r_h.get('notas_cierre')
                                            or ''
                                        )
                                        if _notas_h:
                                            st.markdown(_notas_h)
                                        else:
                                            st.caption(
                                                "(Sin notas registradas)"
                                            )
                                if len(_hechos) > 20:
                                    st.caption(
                                        f"_Mostrando 20 de "
                                        f"{len(_hechos)}._"
                                    )

                            if _cancel:
                                with st.expander(
                                    f"🗑️ Cancelados ({len(_cancel)})"
                                ):
                                    for _r_c in _cancel[:10]:
                                        st.markdown(
                                            f"- {_r_c['fecha_objetivo']}"
                                            f" · {(_r_c.get('motivo','') or '—')[:100]}"
                                        )

                        st.markdown("---")
                        # Botón para programar uno nuevo desde acá
                        if st.button(
                            "➕ Programar nuevo llamado a este cliente",
                            key=f"prog_reco_cli_{cli_para_editar}",
                        ):
                            st.session_state[
                                f"prog_reco_form_{cli_para_editar}"
                            ] = True
                        if st.session_state.get(
                            f"prog_reco_form_{cli_para_editar}"
                        ):
                            with st.form(
                                f"form_prog_reco_cli_"
                                f"{cli_para_editar}"
                            ):
                                _f_prog = st.date_input(
                                    "Fecha del llamado",
                                    value=datetime.now().date(),
                                    key=(
                                        f"f_prog_reco_"
                                        f"{cli_para_editar}"
                                    ),
                                )
                                _mot_prog = st.text_area(
                                    "Motivo / qué chequear",
                                    placeholder=(
                                        "Ej: revisar consumo del "
                                        "silo, coordinar próxima "
                                        "entrega..."
                                    ),
                                    key=(
                                        f"mot_prog_reco_"
                                        f"{cli_para_editar}"
                                    ),
                                )
                                _col_pr1, _col_pr2 = st.columns(2)
                                _ok_prog = (
                                    _col_pr1.form_submit_button(
                                        "✅ Programar",
                                        type="primary",
                                        width="stretch",
                                    )
                                )
                                _cancel_prog = (
                                    _col_pr2.form_submit_button(
                                        "✖ Cancelar",
                                        width="stretch",
                                    )
                                )
                                if _ok_prog:
                                    db.crear_recordatorio_llamada(
                                        cliente_id=cli_para_editar,
                                        fecha_objetivo=(
                                            _f_prog.isoformat()
                                        ),
                                        motivo=_mot_prog or "",
                                        origen="manual",
                                    )
                                    st.session_state[
                                        f"prog_reco_form_"
                                        f"{cli_para_editar}"
                                    ] = False
                                    st.success(
                                        "✅ Llamado programado"
                                    )
                                    st.rerun()
                                elif _cancel_prog:
                                    st.session_state[
                                        f"prog_reco_form_"
                                        f"{cli_para_editar}"
                                    ] = False
                                    st.rerun()

                    # ============================================================
                    # ============================================================
                    #     HISTORIAL DE AVISOS ENVIADOS A ESTE CLIENTE
                    # ============================================================
                    with st.expander(
                        "📨 Historial de avisos enviados "
                        "(email + WhatsApp)",
                        expanded=False,
                    ):
                        st.caption(
                            "Listado cronológico de todo lo que el "
                            "sistema le envió a este cliente: "
                            "alertas climáticas, WhatsApp, "
                            "informes, alertas de stock, etc."
                        )
                        _av_dias = st.selectbox(
                            "Mostrar últimos",
                            [7, 14, 30, 60, 90],
                            index=2,
                            key=f"av_dias_{cli_para_editar}",
                        )
                        _avisos = db.listar_avisos_enviados(
                            cliente_id=cli_para_editar,
                            dias=int(_av_dias), limit=200,
                        )
                        if not _avisos:
                            st.info(
                                "Sin avisos enviados a este cliente "
                                f"en los últimos {_av_dias} días. "
                                "Si esperabas que se haya mandado "
                                "uno, verificá que la alerta esté "
                                "activada y que el cron haya corrido."
                            )
                        else:
                            _filas_av = []
                            _ICONOS_CANAL = {
                                "email": "📧",
                                "whatsapp": "📱",
                            }
                            for av in _avisos:
                                _ico = _ICONOS_CANAL.get(
                                    av["canal"], "•"
                                )
                                _fc = (
                                    av.get("fecha_creacion") or ""
                                )[:16]
                                _est = av.get("estado") or "—"
                                _est_ico = (
                                    "✅" if str(_est).lower()
                                    in ("enviada", "ok", "sent")
                                    else (
                                        "❌"
                                        if av.get("error")
                                        else "⏳"
                                    )
                                )
                                _filas_av.append({
                                    "Cuándo": _fc,
                                    "Canal": (
                                        f"{_ico} {av['canal']}"
                                    ),
                                    "Asunto / clave":
                                        (av.get("asunto") or "")[:80],
                                    "Estado":
                                        f"{_est_ico} {_est}",
                                    "Destinatario":
                                        (av.get("destinatario")
                                         or "—")[:40],
                                })
                            st.dataframe(
                                pd.DataFrame(_filas_av),
                                hide_index=True, width="stretch",
                            )
                            st.caption(
                                f"_Total: {len(_avisos)} avisos en "
                                f"los últimos {_av_dias} días._"
                            )

                    # ============================================================
                    #     CONTACTOS ADICIONALES (encargado, comedero, etc.)
                    # ============================================================
                    with st.expander(
                        "👥 Contactos adicionales del establecimiento "
                        "(encargado, personal de comedero, etc.)",
                        expanded=False,
                    ):
                        st.caption(
                            "El **contacto principal** está más arriba (productor). "
                            "Acá podés sumar gente del equipo que también necesite "
                            "recibir las alertas — encargado, capataz, personal "
                            "de comedero, nutricionista, etc. Cada uno recibe "
                            "primero un mensaje de bienvenida explicando el sistema."
                        )

                        contactos_extra = db.listar_contactos(cli_para_editar)

                        if contactos_extra:
                            st.markdown("##### Contactos cargados")
                            for cx in contactos_extra:
                                bv_em = "✅" if cx.get(
                                    "bienvenida_email_enviada", 0
                                ) else "⏳"
                                bv_wa = "✅" if cx.get(
                                    "bienvenida_whatsapp_enviada", 0
                                ) else "⏳"
                                em_act = "🔔" if cx.get(
                                    "alertas_email_activas", 1
                                ) else "🔕"
                                wa_act = "🔔" if cx.get(
                                    "alertas_whatsapp_activas", 1
                                ) else "🔕"

                                with st.container(border=True):
                                    col_info, col_btns = st.columns([4, 1])
                                    with col_info:
                                        st.markdown(
                                            f"**{cx.get('nombre','')}** "
                                            f"_{cx.get('rol','') or 'Contacto'}_"
                                        )
                                        if cx.get("email"):
                                            st.caption(
                                                f"📧 {cx['email']} {em_act} "
                                                f"· bienvenida {bv_em}"
                                            )
                                        if cx.get("whatsapp"):
                                            st.caption(
                                                f"📱 {cx['whatsapp']} {wa_act} "
                                                f"· bienvenida {bv_wa}"
                                            )
                                    with col_btns:
                                        if st.button(
                                            "✏️ Editar",
                                            key=f"ed_cnt_{cx['id']}",
                                        ):
                                            st.session_state[
                                                f"editing_contacto_{cli_para_editar}"
                                            ] = cx["id"]
                                            st.rerun()
                                        if st.button(
                                            "🗑️ Borrar",
                                            key=f"rm_cnt_{cx['id']}",
                                        ):
                                            db.eliminar_contacto(cx["id"])
                                            st.rerun()
                        else:
                            st.info("Todavía no hay contactos extra cargados.")

                        # Edición de contacto existente
                        editing_id = st.session_state.get(
                            f"editing_contacto_{cli_para_editar}"
                        )
                        if editing_id:
                            cx_edit = db.obtener_contacto(editing_id)
                            if cx_edit:
                                st.markdown("##### Editar contacto")
                                with st.form(f"frm_edit_cnt_{editing_id}"):
                                    en_nom = st.text_input(
                                        "Nombre",
                                        cx_edit.get("nombre", ""),
                                    )
                                    en_rol = st.selectbox(
                                        "Rol",
                                        ["Encargado", "Capataz",
                                         "Personal de comedero",
                                         "Nutricionista", "Veterinario",
                                         "Otro"],
                                        index=(["Encargado", "Capataz",
                                                "Personal de comedero",
                                                "Nutricionista",
                                                "Veterinario", "Otro"].index(
                                            cx_edit.get("rol", "")
                                        ) if cx_edit.get("rol") in [
                                            "Encargado", "Capataz",
                                            "Personal de comedero",
                                            "Nutricionista", "Veterinario",
                                            "Otro"
                                        ] else 0),
                                    )
                                    en_email = st.text_input(
                                        "Email",
                                        cx_edit.get("email", ""),
                                    )
                                    en_em_on = st.checkbox(
                                        "Recibe alertas por email",
                                        value=bool(cx_edit.get(
                                            "alertas_email_activas", 1
                                        )),
                                    )
                                    en_wa = st.text_input(
                                        "WhatsApp",
                                        cx_edit.get("whatsapp", ""),
                                        placeholder="+54 9 2954 51-7407",
                                    )
                                    en_wa_on = st.checkbox(
                                        "Recibe alertas por WhatsApp",
                                        value=bool(cx_edit.get(
                                            "alertas_whatsapp_activas", 1
                                        )),
                                    )
                                    en_notas = st.text_area(
                                        "Notas",
                                        cx_edit.get("notas", ""),
                                        height=60,
                                    )

                                    col_g, col_c, col_r = st.columns(3)
                                    if col_g.form_submit_button(
                                        "💾 Guardar"
                                    ):
                                        db.actualizar_contacto(
                                            editing_id,
                                            nombre=en_nom, rol=en_rol,
                                            email=en_email, whatsapp=en_wa,
                                            alertas_email_activas=int(en_em_on),
                                            alertas_whatsapp_activas=int(en_wa_on),
                                            notas=en_notas,
                                        )
                                        st.session_state.pop(
                                            f"editing_contacto_{cli_para_editar}",
                                            None,
                                        )
                                        st.success("Actualizado")
                                        st.rerun()
                                    if col_c.form_submit_button(
                                        "❌ Cancelar"
                                    ):
                                        st.session_state.pop(
                                            f"editing_contacto_{cli_para_editar}",
                                            None,
                                        )
                                        st.rerun()
                                    if col_r.form_submit_button(
                                        "🔄 Re-mandar bienvenida"
                                    ):
                                        # Resetea los flags para que la próxima
                                        # alerta vuelva a mandar la bienvenida.
                                        db.actualizar_contacto(
                                            editing_id,
                                            bienvenida_email_enviada=0,
                                            bienvenida_whatsapp_enviada=0,
                                        )
                                        st.success(
                                            "Bienvenida marcada como pendiente. "
                                            "Se reenviará en la próxima alerta."
                                        )
                                        st.rerun()

                        # Alta de contacto nuevo
                        st.markdown("##### ➕ Sumar contacto")
                        with st.form(f"frm_new_cnt_{cli_para_editar}",
                                     clear_on_submit=True):
                            n_nom = st.text_input(
                                "Nombre *",
                                placeholder="ej: Carlos Pérez",
                            )
                            n_rol = st.selectbox(
                                "Rol",
                                ["Encargado", "Capataz",
                                 "Personal de comedero",
                                 "Nutricionista", "Veterinario", "Otro"],
                            )
                            n_email = st.text_input(
                                "Email (opcional)",
                                placeholder="contacto@dominio.com",
                            )
                            n_em_on = st.checkbox(
                                "Recibe alertas por email", value=True,
                            )
                            n_wa = st.text_input(
                                "WhatsApp (opcional)",
                                placeholder="+54 9 2954 51-7407",
                            )
                            n_wa_on = st.checkbox(
                                "Recibe alertas por WhatsApp", value=True,
                            )
                            n_notas = st.text_area("Notas", height=60)
                            if st.form_submit_button(
                                "Sumar contacto", type="primary",
                            ):
                                if not n_nom.strip():
                                    st.error("El nombre es obligatorio.")
                                elif not (n_email.strip() or n_wa.strip()):
                                    st.error(
                                        "Tenés que cargar al menos un "
                                        "email o un WhatsApp."
                                    )
                                else:
                                    db.crear_contacto(
                                        cli_para_editar,
                                        nombre=n_nom.strip(),
                                        rol=n_rol,
                                        email=n_email.strip(),
                                        whatsapp=n_wa.strip(),
                                        alertas_email_activas=int(n_em_on),
                                        alertas_whatsapp_activas=int(n_wa_on),
                                        notas=n_notas,
                                    )
                                    st.success(
                                        f"✅ {n_nom} sumado al equipo. "
                                        "Recibirá un mensaje de bienvenida "
                                        "en la próxima alerta."
                                    )
                                    st.rerun()

    # ---- Lotes ----
    with sub2:
        clientes = db.listar_clientes()
        if not clientes:
            st.warning("⚠️ Cargá al menos un cliente antes de crear lotes.")
            st.stop()

        # Modo navegación: si hay lote_detalle_id en session, mostrar
        # vista dedicada del lote (ancho completo, sin form de crear).
        # Sino, vista listado con tabla de KPIs + botones "Ver".
        _drill_id = st.session_state.get("lote_detalle_id")
        # Validar que el lote todavía exista (puede haberse eliminado)
        if _drill_id:
            _drill_check = db.obtener_lote(_drill_id)
            if not _drill_check:
                st.session_state.pop("lote_detalle_id", None)
                _drill_id = None

        if _drill_id:
            # En modo detalle, el form de crear queda casi invisible
            # — todo el ancho para la ficha.
            col_lote_form, col_lote_list = st.columns([0.001, 1000])
        else:
            col_lote_form, col_lote_list = st.columns([1, 2])

        with col_lote_form:
            if not _drill_id:
                st.markdown("##### ➕ Nuevo lote")
                with st.form("nuevo_lote", clear_on_submit=True):
                    cli_sel = st.selectbox(
                        "Cliente *",
                        [c["id"] for c in clientes],
                        format_func=lambda x: next(c["nombre"] for c in clientes if c["id"] == x),
                    )
                    l_id = st.text_input("Identificador del lote *",
                                          placeholder="ej: Vaquillonas Hereford 2025-A")
                    l_corral = st.text_input("Corral / sector",
                                              placeholder="ej: Corral 5")
                    l_raza = st.selectbox("Raza",
                        ["angus", "hereford", "brangus", "braford", "cruza", "otro"])
                    _cats_disp = db.nombres_categorias()
                    if not _cats_disp:
                        _cats_disp = ["ternero"]
                    l_cat = st.selectbox(
                        "Categoría",
                        _cats_disp,
                        help=(
                            "Gestioná esta lista en "
                            "Configuración → Categorías de animales."
                        ),
                    )
                    l_fecha_in = st.date_input("Fecha de ingreso")
                    l_cant = st.number_input("Cantidad inicial", min_value=0, step=1, value=0)
                    l_peso = st.number_input("Peso de ingreso prom. (kg)",
                                              min_value=0.0, step=1.0, value=0.0)
                    l_obj_peso = st.number_input("Peso objetivo (kg)",
                                                  min_value=0.0, step=1.0, value=0.0)

                    # ADPV / energía van JUSTO después del peso objetivo
                    # para que el siguiente bloque pueda calcular días de
                    # encierre y fecha objetivo automáticamente.
                    col_ip1, col_ip2 = st.columns(2)
                    with col_ip1:
                        l_adpv = st.number_input(
                            "ADPV objetivo (kg/día)",
                            min_value=0.0, max_value=2.5, step=0.05,
                            value=0.0,
                            help=(
                                "Ganancia diaria esperada del lote. Si lo "
                                "dejás en 0, se usa el default por "
                                "categoría: ternero 0.8, recría 1.0, "
                                "novillito 1.1, novillo 1.2, vaquillona "
                                "0.9, vaca 0.4, toro 0.5 kg/día."
                            ),
                        )
                    with col_ip2:
                        l_energia = st.number_input(
                            "Energía dieta (Mcal EM/kg MS)",
                            min_value=0.0, max_value=3.5, step=0.05,
                            value=0.0,
                            help=(
                                "Pasto verde ≈ 2.2-2.4, mezcla recría ≈ "
                                "2.6-2.7, terminación grano-fibra ≈ "
                                "2.8-3.0. Si lo dejás en 0, se usa el "
                                "default por categoría."
                            ),
                        )

                    # Vista previa: días de encierre + fecha objetivo
                    # estimados a partir de los datos cargados arriba. Es
                    # informativo — el cálculo definitivo se hace al
                    # guardar.
                    _proy = db.calcular_fecha_objetivo_estimada(
                        fecha_ingreso=l_fecha_in.isoformat()
                                      if l_fecha_in else "",
                        peso_ingreso_kg=float(l_peso),
                        peso_objetivo_kg=float(l_obj_peso),
                        adpv_kg_dia=float(l_adpv) if l_adpv > 0 else None,
                        categoria=l_cat,
                    )
                    if _proy:
                        st.info(
                            f"📅 **Encierre estimado: "
                            f"{_proy['dias_encierre']} días** · "
                            f"Salida proyectada al "
                            f"**{_proy['fecha_objetivo']}**\n\n"
                            f"_Basado en ADPV de "
                            f"{_proy['adpv_usado']:.2f} kg/día "
                            f"({_proy['fuente_adpv']})._"
                        )
                    else:
                        st.caption(
                            "_Cargá peso ingreso, peso objetivo y fecha de "
                            "ingreso para que el sistema calcule días de "
                            "encierre y fecha objetivo automáticamente._"
                        )

                    # La fecha objetivo es opcional — si la dejás vacía se
                    # usa la calculada arriba.
                    l_obj_fecha = st.date_input(
                        "Fecha objetivo (opcional — sobrescribe el cálculo)",
                        value=None,
                        help=(
                            "Si la dejás vacía, el sistema usa la fecha "
                            "calculada con peso objetivo + ADPV. Cargala "
                            "manualmente solo si tu objetivo es una fecha "
                            "fija (ej. cierre de campaña, fecha de feria)."
                        ),
                    )
                    l_notas = st.text_area("Notas", height=60)

                    if st.form_submit_button("Crear lote", type="primary"):
                        if not l_id:
                            st.error("Identificador obligatorio.")
                        else:
                            # Auto-calcular fecha objetivo si no se cargó
                            # manualmente. Si el productor escribió una,
                            # se respeta tal cual.
                            fecha_obj_final = (
                                l_obj_fecha.isoformat()
                                if l_obj_fecha else ""
                            )
                            info_calc = None
                            if not fecha_obj_final:
                                info_calc = db.calcular_fecha_objetivo_estimada(
                                    fecha_ingreso=l_fecha_in.isoformat(),
                                    peso_ingreso_kg=float(l_peso),
                                    peso_objetivo_kg=float(l_obj_peso),
                                    adpv_kg_dia=(float(l_adpv)
                                                 if l_adpv > 0 else None),
                                    categoria=l_cat,
                                )
                                if info_calc:
                                    fecha_obj_final = info_calc["fecha_objetivo"]

                            lid = db.crear_lote(
                                cli_sel, l_id, l_corral, l_raza, l_cat,
                                fecha_ingreso=l_fecha_in.isoformat(),
                                cantidad_inicial=int(l_cant),
                                peso_ingreso_kg=float(l_peso),
                                objetivo_peso_kg=float(l_obj_peso),
                                objetivo_fecha=fecha_obj_final,
                                notas=l_notas,
                                adpv_objetivo_kg=(float(l_adpv)
                                                   if l_adpv > 0 else None),
                                energia_dieta_mcal_em_kg_ms=(float(l_energia)
                                                   if l_energia > 0 else None),
                            )
                            if info_calc:
                                st.success(
                                    f"✅ Lote creado (id {lid}) · "
                                    f"fecha objetivo calculada al "
                                    f"**{info_calc['fecha_objetivo']}** "
                                    f"({info_calc['dias_encierre']} días de "
                                    f"encierre · ADPV "
                                    f"{info_calc['adpv_usado']:.2f}, "
                                    f"{info_calc['fuente_adpv']})"
                                )
                            else:
                                st.success(f"✅ Lote creado (id {lid})")
                            st.rerun()

        with col_lote_list:
            # ── Header de navegación drill-down ──
            # En modo detalle: botón volver + selector para saltar
            # entre lotes. En modo listado: nada acá (la tabla
            # tiene botones "Ver" en cada fila más abajo).
            if _drill_id:
                _hc1, _hc2 = st.columns([1, 4])
                with _hc1:
                    if st.button(
                        "← Volver al listado",
                        key="back_to_lista_lote",
                        width="stretch",
                    ):
                        st.session_state.pop(
                            "lote_detalle_id", None,
                        )
                        # Limpiar query param para que el reload
                        # no te devuelva al lote.
                        try:
                            if "lote_id" in st.query_params:
                                del st.query_params["lote_id"]
                        except Exception:
                            pass
                        st.rerun()
                with _hc2:
                    _todos_lotes_nav = (
                        db.listar_lotes(estado=None) or []
                    )
                    _opc_nav = {
                        l["id"]: (
                            f"{l['cliente_nombre']} — "
                            f"{l['identificador']} · "
                            f"{l.get('categoria', '')} · "
                            f"{l.get('estado', '').upper()}"
                        )
                        for l in _todos_lotes_nav
                    }
                    _ids_nav = list(_opc_nav.keys())
                    _idx_nav = (
                        _ids_nav.index(_drill_id)
                        if _drill_id in _ids_nav else 0
                    )
                    _nuevo_nav = st.selectbox(
                        "Saltar a otro lote",
                        _ids_nav,
                        format_func=lambda x: _opc_nav.get(
                            x, str(x)
                        ),
                        index=_idx_nav,
                        key="jump_lote_drill",
                    )
                    if _nuevo_nav != _drill_id:
                        st.session_state["lote_detalle_id"] = (
                            _nuevo_nav
                        )
                        try:
                            st.query_params["lote_id"] = str(
                                _nuevo_nav
                            )
                        except Exception:
                            pass
                        st.rerun()
                st.divider()

            st.markdown("##### Lotes registrados"
                        if not _drill_id else "")
            if not _drill_id:
                estado_filtro = st.radio(
                    "Estado", ["activo", "cerrado", "todos"],
                    horizontal=True, index=0,
                )
            else:
                estado_filtro = "todos"
            lotes = db.listar_lotes(
                estado=None if estado_filtro == "todos"
                else estado_filtro,
            )
            if not lotes:
                if not _drill_id:
                    st.info("No hay lotes en este estado.")
            else:
                if not _drill_id:
                    # Tabla con KPIs (solo en modo listado)
                    df_lotes = pd.DataFrame([
                        {
                            "ID": l["id"],
                            "Cliente": l["cliente_nombre"],
                            "Lote": l["identificador"],
                            "Corral": l.get("corral", ""),
                            "Categoría": l.get("categoria", ""),
                            "Cabezas": (
                                db.cantidad_vigente_lote(l["id"])
                                if l.get("estado") == "activo"
                                else l.get("cantidad_inicial", 0)
                            ),
                            "Ingreso (kg)": (
                                l.get("peso_ingreso_kg", 0)
                            ),
                            "Último peso (kg)": (
                                f"{l['ultimo_peso_kg']:.1f}"
                                if l.get("ultimo_peso_kg") else "—"
                            ),
                            "Pesadas": l.get("n_pesadas", 0),
                            "Estado": l.get("estado", ""),
                        }
                        for l in lotes
                    ])
                    st.dataframe(
                        df_lotes, hide_index=True, width="stretch",
                    )

                    st.caption(
                        "Click en **Ver ficha** para abrir el "
                        "análisis individual del lote a ancho "
                        "completo."
                    )
                    _bcols_per_row = 3
                    for _i in range(0, len(lotes), _bcols_per_row):
                        _row = lotes[_i: _i + _bcols_per_row]
                        _bcols = st.columns(_bcols_per_row)
                        for _j, _l in enumerate(_row):
                            with _bcols[_j]:
                                _lbl_cli = _l['cliente_nombre'][:18]
                                _lbl_id = _l['identificador'][:18]
                                if st.button(
                                    f"👁️ {_lbl_cli} / {_lbl_id}",
                                    key=f"ver_lote_btn_{_l['id']}",
                                    width="stretch",
                                ):
                                    st.session_state[
                                        "lote_detalle_id"
                                    ] = _l["id"]
                                    try:
                                        st.query_params[
                                            "lote_id"
                                        ] = str(_l["id"])
                                    except Exception:
                                        pass
                                    st.rerun()

            # Modo detalle: lote_sel = drill_id (sin selectbox).
            # Modo listado: si hay lotes, mostrar selectbox legacy.
            # Ambos terminan en "if lote_sel:" que renderiza la
            # ficha entera del lote a ancho completo.
            if _drill_id or lotes:
                if _drill_id:
                    lote_sel = _drill_id
                else:
                    lote_sel = st.selectbox(
                        "Seleccionar lote (legacy — usá los botones)",
                        [None] + [l["id"] for l in lotes],
                        format_func=lambda x: "—" if x is None else
                            next(f"{l['cliente_nombre']} — {l['identificador']}"
                                 for l in lotes if l["id"] == x),
                        key="edit_lote_sel",
                    )
                if lote_sel:
                    l_data = db.obtener_lote(lote_sel)
                    cole1, cole2, cole3 = st.columns(3)
                    if cole1.button("📌 Marcar como ACTIVO"):
                        db.actualizar_lote(lote_sel, estado="activo")
                        st.rerun()
                    if cole2.button("📦 Cerrar lote"):
                        db.actualizar_lote(
                            lote_sel, estado="cerrado",
                            fecha_salida=datetime.now().strftime("%Y-%m-%d"),
                        )
                        st.success("Cerrado")
                        st.rerun()
                    if cole3.button("🗑️ Eliminar lote", type="secondary"):
                        db.eliminar_lote(lote_sel)
                        st.warning("Eliminado")
                        st.rerun()

                    # === DETALLE DEL LOTE (arriba — lo principal) ===
                    # KPIs, pesadas, evolución, gráficos, dietas,
                    # movimientos, cargas, rollo, historial. Es lo
                    # que el usuario busca primero al entrar al lote.
                    st.divider()
                    try:
                        _render_seguimiento_completo_lote(
                            int(lote_sel)
                        )
                    except Exception as _err_seg:
                        st.error(
                            f"Error renderizando seguimiento: "
                            f"{_err_seg}"
                        )
                    st.divider()
                    st.markdown(
                        "### ⚙️ Configuración y acciones del lote"
                    )
                    st.caption(
                        "Editá datos, configurá el encargado, ajustá "
                        "parámetros de impacto o cargá pesadas manuales."
                    )

                    # ---- Editar datos básicos del lote ----
                    with st.expander(
                        "✏️ Editar datos del lote",
                        expanded=False,
                    ):
                        st.caption(
                            "Editás los datos cargados al crear el "
                            "lote: cantidad, pesos, categoría, raza, "
                            "fechas. Los cambios afectan los cálculos "
                            "de impacto productivo y stock."
                        )

                        # Categorías desde la DB (CRUD-administrable)
                        _cats_edit = db.nombres_categorias()
                        if not _cats_edit:
                            _cats_edit = ["ternero"]
                        # Si la categoría actual del lote ya no está en
                        # la lista activa, la insertamos al principio
                        # para no perderla.
                        _cat_actual = (l_data.get("categoria") or "").lower()
                        if _cat_actual and _cat_actual not in _cats_edit:
                            _cats_edit = [_cat_actual] + _cats_edit

                        _razas = ["angus", "hereford", "brangus",
                                  "braford", "cruza", "otro"]
                        _raza_actual = (l_data.get("raza") or "angus").lower()
                        if _raza_actual not in _razas:
                            _razas = [_raza_actual] + _razas

                        with st.form(f"edit_lote_form_{lote_sel}"):
                            _ec1, _ec2 = st.columns(2)
                            with _ec1:
                                _ed_ident = st.text_input(
                                    "Identificador del lote",
                                    value=l_data.get("identificador",
                                                     ""),
                                    key=f"ed_ident_{lote_sel}",
                                )
                                _ed_corral = st.text_input(
                                    "Corral / sector",
                                    value=l_data.get("corral", "") or "",
                                    key=f"ed_corral_{lote_sel}",
                                )
                                _ed_raza = st.selectbox(
                                    "Raza",
                                    _razas,
                                    index=_razas.index(_raza_actual),
                                    key=f"ed_raza_{lote_sel}",
                                )
                                _ed_cat = st.selectbox(
                                    "Categoría",
                                    _cats_edit,
                                    index=(_cats_edit.index(_cat_actual)
                                           if _cat_actual in _cats_edit
                                           else 0),
                                    key=f"ed_cat_{lote_sel}",
                                )
                                try:
                                    _fi_def = datetime.strptime(
                                        l_data.get("fecha_ingreso")
                                        or datetime.now().strftime("%Y-%m-%d"),
                                        "%Y-%m-%d",
                                    ).date()
                                except Exception:
                                    _fi_def = datetime.now().date()
                                _ed_fecha_in = st.date_input(
                                    "Fecha de ingreso",
                                    value=_fi_def,
                                    key=f"ed_fecha_{lote_sel}",
                                )
                            with _ec2:
                                _ed_cant = st.number_input(
                                    "Cantidad inicial",
                                    min_value=0, step=1,
                                    value=int(
                                        l_data.get("cantidad_inicial")
                                        or 0),
                                    key=f"ed_cant_{lote_sel}",
                                )
                                _ed_peso = st.number_input(
                                    "Peso de ingreso prom. (kg)",
                                    min_value=0.0, step=1.0,
                                    value=float(
                                        l_data.get("peso_ingreso_kg")
                                        or 0.0),
                                    key=f"ed_peso_{lote_sel}",
                                )
                                _ed_obj_peso = st.number_input(
                                    "Peso objetivo (kg)",
                                    min_value=0.0, step=1.0,
                                    value=float(
                                        l_data.get("objetivo_peso_kg")
                                        or 0.0),
                                    key=f"ed_obj_peso_{lote_sel}",
                                )
                                _obj_f_raw = (l_data.get("objetivo_fecha")
                                              or "")
                                try:
                                    _of_def = (datetime.strptime(
                                        _obj_f_raw, "%Y-%m-%d",
                                    ).date()
                                        if _obj_f_raw else None)
                                except Exception:
                                    _of_def = None
                                _ed_obj_fecha = st.date_input(
                                    "Fecha objetivo (opcional)",
                                    value=_of_def,
                                    key=f"ed_obj_fecha_{lote_sel}",
                                )
                            _ed_notas = st.text_area(
                                "Notas",
                                value=l_data.get("notas", "") or "",
                                key=f"ed_notas_{lote_sel}",
                                height=80,
                            )
                            if st.form_submit_button(
                                "💾 Guardar cambios del lote",
                                type="primary",
                            ):
                                db.actualizar_lote(
                                    lote_sel,
                                    identificador=_ed_ident.strip(),
                                    corral=_ed_corral.strip(),
                                    raza=_ed_raza,
                                    categoria=_ed_cat,
                                    fecha_ingreso=(
                                        _ed_fecha_in.isoformat()
                                        if _ed_fecha_in else None
                                    ),
                                    cantidad_inicial=int(_ed_cant),
                                    peso_ingreso_kg=float(_ed_peso),
                                    objetivo_peso_kg=(
                                        float(_ed_obj_peso)
                                        if _ed_obj_peso > 0 else None
                                    ),
                                    objetivo_fecha=(
                                        _ed_obj_fecha.isoformat()
                                        if _ed_obj_fecha else None
                                    ),
                                    notas=_ed_notas,
                                )
                                st.success(
                                    "✅ Datos del lote actualizados."
                                )
                                st.rerun()

                    # ---- Encargado + carga diaria por WhatsApp ----
                    # Permite configurar al operario que carga el
                    # comedero todos los días. Si se activa el toggle,
                    # el cron de las 17:00 le manda un WhatsApp con un
                    # link único para que registre la carga del día.
                    with st.expander(
                        "📱 Encargado del lote — carga diaria por "
                        "WhatsApp",
                        expanded=False,
                    ):
                        st.caption(
                            "Configurá quién carga la mezcla al "
                            "comedero todos los días. Si activás la "
                            "pregunta diaria, el sistema le manda un "
                            "WhatsApp a las 17:00 con un link a un "
                            "formulario web pre-llenado con la "
                            "dieta vigente. El encargado solo confirma "
                            "los kg de cada ingrediente y la carga "
                            "queda registrada al instante."
                        )
                        with st.form(f"enc_form_{lote_sel}"):
                            _ec_a, _ec_b = st.columns(2)
                            with _ec_a:
                                _ed_enc_nom = st.text_input(
                                    "Nombre del encargado",
                                    value=(
                                        l_data.get("encargado_nombre")
                                        or ""
                                    ),
                                    placeholder="Ej: Juan",
                                    key=f"enc_nom_{lote_sel}",
                                )
                            with _ec_b:
                                _ed_enc_wa = st.text_input(
                                    "WhatsApp del encargado",
                                    value=(
                                        l_data.get("encargado_whatsapp")
                                        or ""
                                    ),
                                    placeholder="+54 9 2954 XXX XXX",
                                    key=f"enc_wa_{lote_sel}",
                                    help=(
                                        "Formato internacional. "
                                        "Si usás sandbox de Twilio, el "
                                        "encargado tiene que mandar "
                                        "'join <código>' al número "
                                        "sandbox antes de recibir "
                                        "mensajes."
                                    ),
                                )
                            _activa_def = bool(
                                int(l_data.get("carga_diaria_activa") or 0)
                            )
                            _ed_activa = st.toggle(
                                "Activar pregunta diaria por WhatsApp",
                                value=_activa_def,
                                key=f"enc_activa_{lote_sel}",
                                help=(
                                    "Cuando está activa, el cron envía "
                                    "WhatsApp al encargado en los "
                                    "horarios configurados abajo."
                                ),
                            )

                            # Cantidad de comidas y horarios
                            _cant_def = int(
                                l_data.get("cant_comidas_diarias") or 1
                            )
                            _cant_def = 1 if _cant_def not in (1, 2) else _cant_def
                            _ed_cant_com = st.selectbox(
                                "Cantidad de comidas por día",
                                [1, 2],
                                index=(0 if _cant_def == 1 else 1),
                                key=f"enc_cant_com_{lote_sel}",
                                help=(
                                    "Comedero lineal: 1 o 2 cargas "
                                    "diarias. Para silocomedero usá 1 "
                                    "(la pregunta se manda solo el "
                                    "día que toca recargar el silo)."
                                ),
                            )

                            from datetime import time as _time_cls
                            def _parse_hora_default(s: str, fallback: str) -> _time_cls:
                                try:
                                    s = (s or fallback).strip()
                                    hh, mm = s.split(":")[:2]
                                    return _time_cls(int(hh), int(mm))
                                except Exception:
                                    hh, mm = fallback.split(":")
                                    return _time_cls(int(hh), int(mm))

                            _ec_h1, _ec_h2 = st.columns(2)
                            with _ec_h1:
                                _ed_hora_1 = st.time_input(
                                    "Hora 1ra comida",
                                    value=_parse_hora_default(
                                        l_data.get("hora_comida_1"),
                                        "08:30",
                                    ),
                                    key=f"enc_h1_{lote_sel}",
                                )
                            with _ec_h2:
                                if _ed_cant_com == 2:
                                    _ed_hora_2 = st.time_input(
                                        "Hora 2da comida",
                                        value=_parse_hora_default(
                                            l_data.get("hora_comida_2"),
                                            "16:00",
                                        ),
                                        key=f"enc_h2_{lote_sel}",
                                    )
                                else:
                                    _ed_hora_2 = None
                                    st.caption(
                                        "_Solo 1 comida — no usa la "
                                        "hora 2._"
                                    )

                            st.caption(
                                "ℹ️ El cron corre cada 15 min y manda "
                                "el WhatsApp en los ±10 min de cada "
                                "horario. Si el encargado no carga, "
                                "se reenvía 1 hora después."
                            )

                            if st.form_submit_button(
                                "💾 Guardar encargado",
                                type="primary",
                            ):
                                db.actualizar_lote(
                                    lote_sel,
                                    encargado_nombre=(
                                        _ed_enc_nom.strip() or None
                                    ),
                                    encargado_whatsapp=(
                                        _ed_enc_wa.strip() or None
                                    ),
                                    carga_diaria_activa=(
                                        1 if _ed_activa else 0
                                    ),
                                    cant_comidas_diarias=int(
                                        _ed_cant_com
                                    ),
                                    hora_comida_1=(
                                        _ed_hora_1.strftime("%H:%M")
                                    ),
                                    hora_comida_2=(
                                        _ed_hora_2.strftime("%H:%M")
                                        if _ed_hora_2 else None
                                    ),
                                )
                                st.success(
                                    "✅ Encargado actualizado."
                                )
                                st.rerun()

                        # Preview del link que se mandaría hoy (para
                        # que Mauricio pueda probarlo antes de activar).
                        if (l_data.get("encargado_whatsapp") or "").strip():
                            try:
                                _wa_cfg = wa.cargar_config() or {} if (
                                    'wa' in dir()
                                ) else {}
                            except Exception:
                                _wa_cfg = {}
                            if not _wa_cfg:
                                try:
                                    from src import whatsapp as _wa_mod
                                    _wa_cfg = _wa_mod.cargar_config() or {}
                                except Exception:
                                    _wa_cfg = {}
                            _base = (
                                _wa_cfg.get("carga_base_url")
                                or _wa_cfg.get("base_url")
                                or ""
                            ).strip()
                            if _base:
                                from src import carga_diaria_token as _t
                                _link_prev = _t.url_carga_diaria(
                                    _base, int(lote_sel)
                                )
                                st.markdown(
                                    "**Link de prueba para hoy:**"
                                )
                                st.code(_link_prev)
                                st.caption(
                                    "Abrílo desde tu navegador para "
                                    "ver cómo se ve el form del "
                                    "encargado."
                                )
                            else:
                                st.info(
                                    "ℹ️ Para que el link funcione "
                                    "configurá la **URL pública del "
                                    "túnel** en Configuración → "
                                    "WhatsApp (campo `carga_base_url`)."
                                )

                    # ---- Refinar cálculo de impacto productivo
                    #      (editable en lotes ya creados) ----
                    with st.expander(
                        "🎯 Refinar cálculo de impacto productivo "
                        "(ADPV objetivo + energía de dieta)",
                        expanded=False,
                    ):
                        st.caption(
                            "Estos valores reemplazan los defaults por "
                            "categoría en el cálculo NRC de pérdida de "
                            "ADPV ante eventos de frío. Si los dejás "
                            "vacíos, el sistema usa los promedios "
                            "típicos de la categoría."
                        )
                        with st.form(f"impacto_ovr_{lote_sel}"):
                            col_e1, col_e2 = st.columns(2)
                            with col_e1:
                                _adpv_act = (l_data.get("adpv_objetivo_kg")
                                              or 0.0)
                                edit_adpv = st.number_input(
                                    "ADPV objetivo (kg/día)",
                                    min_value=0.0, max_value=2.5,
                                    step=0.05,
                                    value=float(_adpv_act),
                                    help=(
                                        "Default por categoría si "
                                        "queda en 0: ternero 0.8, "
                                        "recría 1.0, novillito 1.1, "
                                        "novillo 1.2, vaquillona 0.9, "
                                        "vaca 0.4, toro 0.5 kg/día."
                                    ),
                                    key=f"edit_adpv_{lote_sel}",
                                )
                            with col_e2:
                                _en_act = (
                                    l_data.get("energia_dieta_mcal_em_kg_ms")
                                    or 0.0
                                )
                                edit_en = st.number_input(
                                    "Energía dieta (Mcal EM/kg MS)",
                                    min_value=0.0, max_value=3.5,
                                    step=0.05,
                                    value=float(_en_act),
                                    help=(
                                        "Pasto verde ≈ 2.2-2.4, "
                                        "mezcla recría ≈ 2.6-2.7, "
                                        "terminación ≈ 2.8-3.0."
                                    ),
                                    key=f"edit_en_{lote_sel}",
                                )
                            if st.form_submit_button(
                                "💾 Guardar refinamiento",
                                type="primary",
                            ):
                                db.actualizar_lote(
                                    lote_sel,
                                    adpv_objetivo_kg=(float(edit_adpv)
                                                       if edit_adpv > 0
                                                       else None),
                                    energia_dieta_mcal_em_kg_ms=(
                                        float(edit_en)
                                        if edit_en > 0 else None
                                    ),
                                )
                                st.success(
                                    "✅ Refinamiento guardado. Las "
                                    "próximas alertas usarán estos "
                                    "valores para el lote."
                                )
                                st.rerun()

                    # ==========================================
                    # 🍽️ Sistema de alimentación del lote
                    # Define qué tan rápido se puede ajustar la
                    # mezcla ante un evento climático.
                    # ==========================================
                    with st.expander(
                        "🍽️ Sistema de alimentación — cómo se entrega "
                        "la ración",
                        expanded=False,
                    ):
                        st.caption(
                            "Define el sistema de comedero y la "
                            "frecuencia de preparación de mezcla. Esto "
                            "permite al sistema de alertas ajustar sus "
                            "recomendaciones a la realidad operativa: "
                            "no le va a sugerir cambiar la mezcla del "
                            "silocomedero al día siguiente si la "
                            "preparación se hace cada 4 días."
                        )

                        # Mostrar diagnóstico actual antes del form
                        _diag_actual = db.diagnostico_alimentacion_lote(
                            l_data,
                        )
                        if _diag_actual["en_adaptacion"]:
                            from datetime import timedelta as _td_ad
                            _dias_d = _diag_actual["dias_desde_ingreso"]
                            _faltan = 15 - _dias_d
                            try:
                                _fi_par = datetime.strptime(
                                    (l_data.get("fecha_ingreso")
                                     or "")[:10],
                                    "%Y-%m-%d",
                                ).date()
                                _fecha_paso = (
                                    _fi_par + _td_ad(days=16)
                                ).strftime("%d/%m/%y")
                            except Exception:
                                _fecha_paso = "—"
                            st.info(
                                f"🌱 **Lote en adaptación** "
                                f"(día {_dias_d} de 15). "
                                f"El sistema **fuerza comedero lineal "
                                f"diario** durante los primeros 15 días, "
                                f"sin importar lo cargado abajo.\n\n"
                                f"📅 El **{_fecha_paso}** (día 16) el "
                                f"lote sale de adaptación y el sistema "
                                f"empieza a usar el comedero que cargues "
                                f"acá abajo (ej. silocomedero). "
                                f"No tenés que marcar nada manualmente "
                                f"— la transición es automática.\n\n"
                                f"💡 **Recomendado:** cargá ahora el "
                                f"sistema final (silocomedero + duración "
                                f"de carga) así el día 16 ya queda "
                                f"listo. Faltan **{_faltan} día(s)**."
                            )
                        elif _diag_actual[
                                "tipo_comedero_efectivo"] == "desconocido":
                            st.warning(
                                "⚠️ No tenés tipo de comedero "
                                "cargado todavía. Las alertas usarán "
                                "lógica genérica hasta que lo definas."
                            )
                        else:
                            st.success(
                                f"✅ {_diag_actual['descripcion']}"
                            )

                        with st.form(
                            f"sistema_alim_{lote_sel}",
                        ):
                            col_sa1, col_sa2 = st.columns(2)
                            with col_sa1:
                                _opciones_comedero = [
                                    "— sin definir —",
                                    "lineal",
                                    "silocomedero",
                                    "autoconsumo",
                                ]
                                _tipo_act = (
                                    l_data.get(
                                        "tipo_comedero_concentrado")
                                    or ""
                                )
                                _idx_tipo = (
                                    _opciones_comedero.index(_tipo_act)
                                    if _tipo_act in _opciones_comedero
                                    else 0
                                )
                                edit_tipo_com = st.selectbox(
                                    "Tipo de comedero del concentrado",
                                    _opciones_comedero,
                                    index=_idx_tipo,
                                    key=f"edit_tcom_{lote_sel}",
                                    help=(
                                        "lineal: comedero lineal con "
                                        "carga desde mixer. "
                                        "silocomedero: mezcla cargada, "
                                        "dura varios días. "
                                        "autoconsumo: el animal se "
                                        "regula solo."
                                    ),
                                )
                                _opciones_forraje = [
                                    "— sin definir —",
                                    "mezclado",
                                    "aparte",
                                ]
                                _forr_act = (
                                    l_data.get("forraje_modalidad")
                                    or ""
                                )
                                _idx_forr = (
                                    _opciones_forraje.index(_forr_act)
                                    if _forr_act in _opciones_forraje
                                    else 0
                                )
                                edit_forraje = st.selectbox(
                                    "Forraje (rollo/silo)",
                                    _opciones_forraje,
                                    index=_idx_forr,
                                    key=f"edit_forr_{lote_sel}",
                                    help=(
                                        "mezclado: el forraje va en "
                                        "el mixer junto con la "
                                        "ración. aparte: el rollo "
                                        "está en autoconsumo o corral "
                                        "separado, la ración va al "
                                        "comedero principal."
                                    ),
                                )
                            with col_sa2:
                                _frec_act = (
                                    l_data.get("frecuencia_mezcla_dias")
                                    or 0
                                )
                                edit_frec = st.number_input(
                                    "Duración de una carga (días)",
                                    min_value=0, max_value=14,
                                    step=1,
                                    value=int(_frec_act),
                                    help=(
                                        "Cuántos días dura una "
                                        "preparación de mezcla en este "
                                        "lote. Ejemplos: comedero "
                                        "lineal diario = 1; mixer "
                                        "cada 2 días = 2; silocomedero "
                                        "según el tamaño del silo y el "
                                        "consumo del lote (puede ser "
                                        "3, 5, 7 o más días). Cargá el "
                                        "valor real observado, no un "
                                        "promedio teórico. Si lo dejás "
                                        "en 0, el sistema usa lógica "
                                        "conservadora."
                                    ),
                                    key=f"edit_frec_{lote_sel}",
                                )

                            if st.form_submit_button(
                                "💾 Guardar sistema de alimentación",
                                type="primary",
                            ):
                                _tipo_save = (
                                    edit_tipo_com
                                    if edit_tipo_com != "— sin definir —"
                                    else None
                                )
                                _forr_save = (
                                    edit_forraje
                                    if edit_forraje != "— sin definir —"
                                    else None
                                )
                                _frec_save = (
                                    int(edit_frec)
                                    if edit_frec > 0 else None
                                )
                                db.actualizar_lote(
                                    lote_sel,
                                    tipo_comedero_concentrado=_tipo_save,
                                    forraje_modalidad=_forr_save,
                                    frecuencia_mezcla_dias=_frec_save,
                                )
                                st.success(
                                    "✅ Sistema de alimentación "
                                    "guardado. Las próximas alertas "
                                    "van a adaptar las recomendaciones "
                                    "nutricionales a este sistema."
                                )
                                st.rerun()

                        # ── Desglose automático de la carga ──
                        # Si el lote tiene tipo silocomedero + duración
                        # de carga > 0 + dieta vigente, mostramos cuántos
                        # kg de cada ingrediente hay que mezclar para
                        # llenar el silocomedero.
                        _tipo_eff = (
                            l_data.get("tipo_comedero_concentrado") or ""
                        )
                        _frec_eff = int(
                            l_data.get("frecuencia_mezcla_dias") or 0
                        )
                        if (_tipo_eff == "silocomedero"
                                and _frec_eff > 0):
                            try:
                                from src.stock_producto import (
                                    desglose_carga_silocomedero,
                                )
                                _desg = desglose_carga_silocomedero(
                                    lote_sel, _frec_eff,
                                    forraje_modalidad=l_data.get(
                                        "forraje_modalidad") or "mezclado",
                                )
                            except Exception as _e_desg:
                                _desg = None
                                st.caption(
                                    f"_No pude calcular el desglose: "
                                    f"{_e_desg}_"
                                )
                            st.markdown("---")
                            st.markdown(
                                "##### 🧮 Receta para cargar el "
                                "silocomedero"
                            )
                            if not _desg:
                                st.info(
                                    "Cargá una dieta para este lote "
                                    "en el Asesor IA y guardala. "
                                    "Acá vas a ver cuántos kg de cada "
                                    "ingrediente preparar para llenar "
                                    "el silocomedero según la duración "
                                    "configurada."
                                )
                            else:
                                _kpi1, _kpi2, _kpi3 = st.columns(3)
                                _kpi1.metric(
                                    "Duración de la carga",
                                    f"{_desg['dias_carga']} días",
                                )
                                _kpi2.metric(
                                    "Total mezcla a preparar",
                                    f"{_desg['kg_total_mezcla']:.0f} kg",
                                    help=(
                                        f"Sólo lo que va al "
                                        f"silocomedero. "
                                        f"{_desg['cantidad_animales']} "
                                        f"animales × "
                                        f"{_desg['kg_total_por_animal']:.2f} "
                                        f"kg/animal en {_desg['dias_carga']} "
                                        f"días."
                                    ),
                                )
                                _kpi3.metric(
                                    "Por animal en el período",
                                    f"{_desg['kg_total_por_animal']:.1f} kg",
                                )
                                _df_desg = pd.DataFrame([
                                    {
                                        "Ingrediente": _i["nombre"],
                                        "% mezcla":
                                            f"{_i['pct_mezcla']:.1f}%",
                                        "kg/animal/día":
                                            f"{_i['kg_tal_cual_por_animal_dia']:.2f}",
                                        "kg total a preparar":
                                            f"{_i['kg_total']:.0f}",
                                        "Bolsas (30 kg)":
                                            (f"{_i['kg_total']/30:.0f}"
                                             if _i["kg_total"] >= 30
                                             else "—"),
                                    }
                                    for _i in _desg["ingredientes"]
                                ])
                                st.dataframe(
                                    _df_desg, hide_index=True,
                                    width="stretch",
                                )

                                # Si hay forrajes a libre disposición
                                # (porque modalidad = aparte), los
                                # mostramos en un bloque separado
                                # para que no se confunda con la
                                # mezcla del silocomedero.
                                _aparte = _desg.get("forrajes_aparte") or []
                                if _aparte:
                                    st.markdown(
                                        "**🌾 Forraje aparte "
                                        "(corral separado / "
                                        "libre disposición)**"
                                    )
                                    _df_ap = pd.DataFrame([
                                        {
                                            "Forraje": _i["nombre"],
                                            "kg/animal/día":
                                                f"{_i['kg_tal_cual_por_animal_dia']:.2f}",
                                            "kg total estimado":
                                                f"{_i['kg_total']:.0f}",
                                        }
                                        for _i in _aparte
                                    ])
                                    st.dataframe(
                                        _df_ap, hide_index=True,
                                        width="stretch",
                                    )
                                    st.caption(
                                        "Estos forrajes NO se cargan al "
                                        "silocomedero. Van en el corral "
                                        "aparte / autoconsumo. Los kg "
                                        "totales son una referencia "
                                        "estimada según la dieta."
                                    )

                                st.caption(
                                    f"Cálculo en base a la dieta vigente "
                                    f"del lote (fecha "
                                    f"{_desg['fecha_dieta']}). "
                                    f"Si cambia la dieta o la cantidad "
                                    f"de animales, los kg se "
                                    f"recalculan automáticamente."
                                )

                                # Mostrar ajuste por peso vivo (ADG)
                                # cuando esté activo, para que Mauricio
                                # entienda por qué el sistema entrega
                                # un valor distinto al de la fórmula
                                # original.
                                _esc_pv = _desg.get("escala_pv") or {}
                                if (_esc_pv.get("origen") == "adg"
                                        and _esc_pv.get("factor_aplicado")
                                        and _esc_pv["factor_aplicado"] != 1.0):
                                    _f_aj = (
                                        _esc_pv["factor_aplicado"] - 1
                                    ) * 100
                                    _signo = "+" if _f_aj >= 0 else ""
                                    _pref = _esc_pv.get(
                                        "peso_referencia_kg", 0)
                                    _pact = _esc_pv.get(
                                        "peso_actual_kg", 0)
                                    st.caption(
                                        f"⚙️ Ajuste automático por peso "
                                        f"vivo proyectado: "
                                        f"**{_signo}{_f_aj:.1f}%** sobre la "
                                        f"fórmula original "
                                        f"(peso ref. {_pref:.0f} kg → "
                                        f"hoy {_pact:.0f} kg, "
                                        f"ADPV objetivo configurado en "
                                        f"el lote)."
                                    )

                    # ---- Impacto productivo proyectado (NRC) ----
                    # Muestra al productor el impacto climático esperado
                    # sobre ESTE lote en los próximos días, ANTES de que
                    # llegue el mail. Le da una vista anticipada del riesgo.
                    with st.container():
                        st.markdown(
                            "##### 📊 Impacto productivo proyectado "
                            "(próximos días)"
                        )
                        # Buscar coordenadas del cliente del lote
                        _cli_lote = db.obtener_cliente(
                            l_data["cliente_id"]
                        ) if l_data.get("cliente_id") else None
                        _lat = (_cli_lote or {}).get("lat")
                        _lon = (_cli_lote or {}).get("lon")
                        if not (_lat and _lon):
                            st.info(
                                "ℹ️ Cargá las coordenadas del establecimiento "
                                "en la ficha del cliente para ver el "
                                "impacto productivo proyectado."
                            )
                        else:
                            with st.spinner("Consultando clima…"):
                                try:
                                    _clima_lote = obtener_clima(
                                        _lat, _lon,
                                    )
                                except Exception as _e:
                                    _clima_lote = None
                                    st.warning(
                                        f"No pude obtener el clima: {_e}"
                                    )
                            if _clima_lote:
                                from src.impacto_productivo import (
                                    estimar_impacto_peor_dia_semanal,
                                    formato_impacto_humano,
                                )
                                # Refrescar el lote con override para
                                # pasarle los datos a la calculadora.
                                _lote_calc = dict(l_data)
                                # Usar último peso o peso ingreso
                                _lote_calc["peso_promedio_kg"] = (
                                    l_data.get("ultimo_peso_kg")
                                    or l_data.get("peso_ingreso_kg")
                                )
                                _lote_calc["cantidad_animales"] = (
                                    l_data.get("cantidad_inicial")
                                )
                                _imp_proy = estimar_impacto_peor_dia_semanal(
                                    _clima_lote, [_lote_calc],
                                )
                                # Mostrar resumen climático breve
                                _daily = (_clima_lote.get("daily")
                                           or {})
                                _tmin = (_daily.get("temperature_2m_min")
                                          or [])
                                _tmax = (_daily.get("temperature_2m_max")
                                          or [])
                                _hr = (_daily.get(
                                    "relative_humidity_2m_max") or [])
                                _viento = (_daily.get(
                                    "windspeed_10m_max") or [])
                                if _tmin and _tmax:
                                    _col_a, _col_b, _col_c, _col_d = (
                                        st.columns(4)
                                    )
                                    _col_a.metric(
                                        "T° mín próx. 7 días",
                                        f"{min([x for x in _tmin[:7] if x is not None] or [0]):.0f}°C",
                                    )
                                    _col_b.metric(
                                        "T° máx próx. 7 días",
                                        f"{max([x for x in _tmax[:7] if x is not None] or [0]):.0f}°C",
                                    )
                                    if _hr:
                                        _col_c.metric(
                                            "HR máx",
                                            f"{max([x for x in _hr[:7] if x is not None] or [0]):.0f}%",
                                        )
                                    if _viento:
                                        _col_d.metric(
                                            "Viento máx",
                                            f"{max([x for x in _viento[:7] if x is not None] or [0]):.0f} km/h",
                                        )
                                # Card del impacto
                                if _imp_proy:
                                    _texto_imp = formato_impacto_humano(
                                        _imp_proy
                                    )
                                    # Guardar al histórico (con dedup
                                    # por semana). Si ya hay uno de
                                    # esta semana, no hace nada.
                                    try:
                                        # Detectar agravantes
                                        _precip = (_daily.get(
                                            "precipitation_sum") or [])
                                        _precip_3d = sum(
                                            (x or 0)
                                            for x in _precip[:3]
                                        )
                                        _barro_proy = _precip_3d > 20
                                        _hum_max = (max(
                                            [x for x in _hr[:7]
                                             if x is not None] or [0]
                                        ) if _hr else 0)
                                        # Severidad heurística
                                        if _imp_proy["gasto_extra_pct"][1] >= 25:
                                            _sev = "critico"
                                        elif _imp_proy["gasto_extra_pct"][1] >= 12:
                                            _sev = "operativo"
                                        else:
                                            _sev = "atencion"
                                        from datetime import datetime as _dt_imp
                                        # Encontrar fecha del peor día
                                        _times = (_daily.get("time")
                                                   or [])
                                        _idx_peor = 0
                                        _t_peor = 99.0
                                        for _i, _v in enumerate(_tmin[:7]):
                                            if _v is not None and _v < _t_peor:
                                                _t_peor = _v
                                                _idx_peor = _i
                                        _fecha_evt_ini = (
                                            _times[_idx_peor]
                                            if _idx_peor < len(_times)
                                            else None
                                        )
                                        _resumen_clima = {
                                            "t_min": min([
                                                x for x in _tmin[:7]
                                                if x is not None
                                            ] or [0]),
                                            "t_max": max([
                                                x for x in _tmax[:7]
                                                if x is not None
                                            ] or [0]),
                                            "hr_max": _hum_max,
                                            "viento_max": (
                                                max([x for x in
                                                      _viento[:7]
                                                      if x is not None]
                                                     or [0])
                                                if _viento else 0
                                            ),
                                            "lluvia_3d": _precip_3d,
                                            "barro": _barro_proy,
                                        }
                                        _saved = db.guardar_impacto_lote(
                                            lote_id=lote_sel,
                                            impacto=_imp_proy,
                                            tipo_evento="frio",
                                            severidad=_sev,
                                            fecha_inicio_evento=_fecha_evt_ini,
                                            clima_resumen=_resumen_clima,
                                            dedup_semana=True,
                                        )
                                    except Exception:
                                        pass
                                    # Tonalidad según severidad del impacto
                                    _g_min, _g_max = _imp_proy.get(
                                        "gasto_extra_pct", (0, 0)
                                    )
                                    if _g_max >= 25:
                                        _bg = "#FDECEC"
                                        _border = "#C0392B"
                                        _ico = "🔴"
                                        _titulo = "Frío crítico proyectado"
                                    elif _g_max >= 12:
                                        _bg = "#FFF6E5"
                                        _border = "#E67E22"
                                        _ico = "🟠"
                                        _titulo = (
                                            "Frío operativo proyectado"
                                        )
                                    else:
                                        _bg = "#FFFCEC"
                                        _border = "#C9A227"
                                        _ico = "🟡"
                                        _titulo = "Frío leve proyectado"
                                    _peso = (l_data.get("ultimo_peso_kg")
                                              or l_data.get(
                                                  "peso_ingreso_kg"))
                                    _ovr_adpv = l_data.get(
                                        "adpv_objetivo_kg")
                                    _ovr_en = l_data.get(
                                        "energia_dieta_mcal_em_kg_ms")
                                    _ovr_txt = ""
                                    if _ovr_adpv or _ovr_en:
                                        _ovr_txt = (
                                            "<br><span style='font-size:"
                                            "11px; color:#1B3E27;'>"
                                            "📌 Override aplicado: "
                                            f"ADPV {_ovr_adpv or '—'} "
                                            f"kg/día · Energía "
                                            f"{_ovr_en or '—'} Mcal/kg MS"
                                            "</span>"
                                        )
                                    st.markdown(
                                        f"""
                                        <div style="background:{_bg};
                                          border-left:4px solid {_border};
                                          padding:14px 16px;
                                          border-radius:6px;
                                          margin-top:10px;">
                                          <div style="font-size:13px;
                                            font-weight:700;
                                            color:{_border};
                                            letter-spacing:0.3px;
                                            margin-bottom:6px;">
                                            {_ico} {_titulo}
                                          </div>
                                          {_texto_imp}
                                          {_ovr_txt}
                                          <div style="font-size:10.5px;
                                            color:#666; margin-top:6px;
                                            font-style:italic;">
                                            Calculado para {_peso or '—'} kg
                                            · {l_data.get('categoria') or '—'}
                                            · {l_data.get('raza') or '—'}.
                                          </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    # Bloque de PREVENCIÓN Y CUIDADO
                                    from src.prevencion import (
                                        acciones_preventivas_html,
                                    )
                                    _prev_html = acciones_preventivas_html(
                                        tipo_evento="frio",
                                        severidad=_sev,
                                        categoria=l_data.get(
                                            "categoria", ""),
                                        barro=_barro_proy,
                                        pelaje_mojado=_hum_max >= 85,
                                    )
                                    if _prev_html:
                                        st.markdown(
                                            f"""
                                            <div style="background:#F4F8F4;
                                              border-left:4px solid
                                              #1B3E27;
                                              padding:14px 16px;
                                              border-radius:6px;
                                              margin-top:10px;">
                                              <div style="font-size:13px;
                                                font-weight:700;
                                                color:#1B3E27;
                                                letter-spacing:0.3px;
                                                margin-bottom:6px;">
                                                🛡️ Prevención y cuidado
                                              </div>
                                              {_prev_html}
                                              <div style="font-size:10.5px;
                                                color:#666; margin-top:6px;
                                                font-style:italic;">
                                                Medidas adaptadas al tipo
                                                de evento, severidad y
                                                categoría del lote.
                                              </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                else:
                                    # Sin frío relevante en la semana
                                    _peso_chk = (
                                        l_data.get("ultimo_peso_kg")
                                        or l_data.get("peso_ingreso_kg")
                                    )
                                    if not _peso_chk:
                                        st.info(
                                            "ℹ️ Cargá una pesada en el lote "
                                            "(o el peso de ingreso) para que el "
                                            "sistema pueda calcular el "
                                            "impacto productivo."
                                        )
                                    else:
                                        st.success(
                                            "✅ Sin estrés productivo "
                                            "calculable para esta semana: "
                                            "el clima proyectado no supera "
                                            "el umbral de confort del lote. "
                                            "Se mantiene seguimiento "
                                            "preventivo."
                                        )

                    # ---- DMI proyectado (consumo de materia seca) ----
                    # Calcula cuánto MS/día consumirá el lote según peso,
                    # categoría y modificadores climáticos. Se muestra
                    # solo si tenemos peso y clima del lote.
                    _peso_dmi = (l_data.get("ultimo_peso_kg")
                                  or l_data.get("peso_ingreso_kg"))
                    _cant_dmi = l_data.get("cantidad_inicial")
                    if (_peso_dmi and _peso_dmi > 0 and _lat and _lon):
                        try:
                            from src.dmi import (
                                dmi_proyectado, formato_dmi_humano,
                            )
                            # Reutilizar el clima ya consultado arriba
                            if _clima_lote:
                                _daily_dmi = (_clima_lote.get("daily")
                                               or {})
                                _tmin_dmi = [
                                    x for x in (_daily_dmi.get(
                                        "temperature_2m_min") or [])[:7]
                                    if x is not None
                                ]
                                _tmax_dmi = [
                                    x for x in (_daily_dmi.get(
                                        "temperature_2m_max") or [])[:7]
                                    if x is not None
                                ]
                                _hr_dmi = [
                                    x for x in (_daily_dmi.get(
                                        "relative_humidity_2m_max")
                                        or [])[:7]
                                    if x is not None
                                ]
                                _viento_dmi = [
                                    x for x in (_daily_dmi.get(
                                        "windspeed_10m_max") or [])[:7]
                                    if x is not None
                                ]
                                _precip_dmi = (_daily_dmi.get(
                                    "precipitation_sum") or [])[:7]
                                _precip_3d_dmi = sum(
                                    (x or 0) for x in _precip_dmi[:3]
                                )
                                _barro_dmi = _precip_3d_dmi > 20
                                _clima_dmi = {
                                    "t_min": (min(_tmin_dmi)
                                               if _tmin_dmi else None),
                                    "t_max": (max(_tmax_dmi)
                                               if _tmax_dmi else None),
                                    "hr_max": (max(_hr_dmi)
                                                if _hr_dmi else None),
                                    "viento_max": (max(_viento_dmi)
                                                    if _viento_dmi
                                                    else None),
                                    "lluvia_3d": _precip_3d_dmi,
                                    "lluvia_dia": (max(_precip_dmi)
                                                    if _precip_dmi
                                                    else 0),
                                }
                                _dmi_proy = dmi_proyectado(
                                    peso_kg=_peso_dmi,
                                    categoria=l_data.get(
                                        "categoria", ""),
                                    raza=l_data.get("raza", ""),
                                    clima_diario=_clima_dmi,
                                    cantidad=_cant_dmi,
                                    dias_evento=1,
                                    barro=_barro_dmi,
                                )
                                if _dmi_proy:
                                    st.markdown(
                                        "##### 🌾 Consumo de materia "
                                        "seca proyectado (próximos días)"
                                    )
                                    _dmi_html = formato_dmi_humano(
                                        _dmi_proy
                                    )
                                    # Tonalidad según factor de ajuste:
                                    # positivo = consumo sube → verde,
                                    # negativo = consumo baja → naranja
                                    _f_dmi_max = _dmi_proy[
                                        "factor_ajuste_pct"][1]
                                    if _f_dmi_max <= -10:
                                        _bg_dmi = "#FFF6E5"
                                        _bd_dmi = "#E67E22"
                                        _ti_dmi = (
                                            "🟠 Consumo bajo proyectado"
                                        )
                                    elif _f_dmi_max >= 3:
                                        _bg_dmi = "#F0F8E8"
                                        _bd_dmi = "#1B6F2C"
                                        _ti_dmi = (
                                            "🟢 Consumo elevado proyectado "
                                            "(demanda extra por frío)"
                                        )
                                    else:
                                        _bg_dmi = "#F4F8F4"
                                        _bd_dmi = "#1B3E27"
                                        _ti_dmi = (
                                            "🌾 Consumo normal proyectado"
                                        )
                                    st.markdown(
                                        f"""
                                        <div style="background:{_bg_dmi};
                                          border-left:4px solid {_bd_dmi};
                                          padding:14px 16px;
                                          border-radius:6px;
                                          margin-top:10px;">
                                          <div style="font-size:13px;
                                            font-weight:700;
                                            color:{_bd_dmi};
                                            letter-spacing:0.3px;
                                            margin-bottom:6px;">
                                            {_ti_dmi}
                                          </div>
                                          {_dmi_html}
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                        except Exception as _e_dmi:
                            # No bloquear la carga si falla el DMI
                            pass

                    # ---- Auto-confirmar proyecciones pasadas ----
                    # Cada vez que se entra a la ficha, revisamos si
                    # hay proyecciones de semanas anteriores que no
                    # tienen su contraparte "confirmada" (cálculo con
                    # clima REAL ya ocurrido). Para esas, traemos el
                    # clima histórico y guardamos un registro
                    # "confirmado" al lado para que el productor pueda
                    # comparar proyección vs realidad.
                    try:
                        from src.clima import obtener_clima_historico
                        from src.impacto_productivo import (
                            estimar_impacto_peor_dia_semanal,
                        )
                        from datetime import (
                            datetime as _dt_conf, timedelta as _td_conf,
                        )
                        _hoy_conf = _dt_conf.now().date()
                        _impactos_existentes = db.listar_impactos_lote(
                            lote_sel, limit=30,
                        )
                        # Separar por estado
                        _proy_por_semana = {}
                        _conf_por_semana = {}
                        for _imp_e in _impactos_existentes:
                            try:
                                _fc_e = _imp_e.get(
                                    "fecha_inicio_evento"
                                ) or _imp_e.get("fecha_calculo", "")[:10]
                                _f_e = _dt_conf.strptime(
                                    _fc_e[:10], "%Y-%m-%d"
                                )
                                _y_e, _w_e, _ = _f_e.isocalendar()
                                _key_e = f"{_y_e}-W{_w_e:02d}"
                                if _imp_e.get("estado") == "confirmado":
                                    _conf_por_semana[_key_e] = _imp_e
                                else:
                                    _proy_por_semana[_key_e] = _imp_e
                            except (ValueError, TypeError):
                                continue
                        # Para cada proyección de semanas con al menos
                        # 7 días pasados (para que la API archive ya
                        # tenga datos), si no hay confirmado, generar.
                        for _key_p, _proy in _proy_por_semana.items():
                            if _key_p in _conf_por_semana:
                                continue
                            try:
                                _fie = _proy.get(
                                    "fecha_inicio_evento"
                                ) or _proy.get(
                                    "fecha_calculo", "")[:10]
                                _f_evt = _dt_conf.strptime(
                                    _fie[:10], "%Y-%m-%d"
                                ).date()
                                if (_hoy_conf - _f_evt).days < 7:
                                    continue  # muy reciente, esperar
                                # Rango: 7 días desde fecha_inicio
                                _hasta_conf = (
                                    _f_evt + _td_conf(days=6)
                                ).isoformat()
                                _desde_conf = _f_evt.isoformat()
                                _clima_hist = obtener_clima_historico(
                                    _lat, _lon, _desde_conf, _hasta_conf,
                                )
                                if not _clima_hist:
                                    continue
                                _lote_conf = dict(l_data)
                                _lote_conf["peso_promedio_kg"] = (
                                    _proy.get("peso_promedio_kg")
                                    or l_data.get("ultimo_peso_kg")
                                    or l_data.get("peso_ingreso_kg")
                                )
                                _lote_conf["cantidad_animales"] = (
                                    _proy.get("cantidad_animales")
                                    or l_data.get("cantidad_inicial")
                                )
                                _imp_conf = estimar_impacto_peor_dia_semanal(
                                    _clima_hist, [_lote_conf],
                                )
                                # Detectar severidad y clima real
                                if _imp_conf:
                                    _daily_h = (_clima_hist.get("daily")
                                                 or {})
                                    _tmin_h = (_daily_h.get(
                                        "temperature_2m_min") or [])
                                    _tmax_h = (_daily_h.get(
                                        "temperature_2m_max") or [])
                                    _hr_h = (_daily_h.get(
                                        "relative_humidity_2m_max") or [])
                                    _viento_h = (_daily_h.get(
                                        "windspeed_10m_max") or [])
                                    _precip_h = (_daily_h.get(
                                        "precipitation_sum") or [])
                                    _precip_3d_h = sum(
                                        (x or 0) for x in _precip_h[:3]
                                    )
                                    if _imp_conf["gasto_extra_pct"][1] >= 25:
                                        _sev_conf = "critico"
                                    elif _imp_conf["gasto_extra_pct"][1] >= 12:
                                        _sev_conf = "operativo"
                                    else:
                                        _sev_conf = "atencion"
                                    _clima_res_conf = {
                                        "t_min": min([
                                            x for x in _tmin_h
                                            if x is not None
                                        ] or [0]),
                                        "t_max": max([
                                            x for x in _tmax_h
                                            if x is not None
                                        ] or [0]),
                                        "hr_max": max([
                                            x for x in _hr_h
                                            if x is not None
                                        ] or [0]),
                                        "viento_max": max([
                                            x for x in _viento_h
                                            if x is not None
                                        ] or [0]),
                                        "lluvia_3d": _precip_3d_h,
                                        "barro": _precip_3d_h > 20,
                                    }
                                    db.guardar_impacto_lote(
                                        lote_id=lote_sel,
                                        impacto=_imp_conf,
                                        tipo_evento="frio",
                                        severidad=_sev_conf,
                                        fecha_inicio_evento=_desde_conf,
                                        fecha_fin_evento=_hasta_conf,
                                        clima_resumen=_clima_res_conf,
                                        estado="confirmado",
                                        dedup_semana=True,
                                    )
                            except Exception:
                                continue
                    except Exception:
                        pass

                    # ---- Histórico de impactos del lote ----
                    with st.expander(
                        "📅 Histórico de impactos productivos de este lote",
                        expanded=False,
                    ):
                        _impactos = db.listar_impactos_lote(
                            lote_sel, limit=24,
                        )
                        if not _impactos:
                            st.caption(
                                "Todavía no hay impactos registrados. "
                                "Cada vez que veas la ficha del lote, "
                                "el sistema guarda el cálculo de la "
                                "semana (un registro por semana ISO)."
                            )
                        else:
                            import pandas as _pd_hist
                            _filas = []
                            for _imp_h in _impactos:
                                # Severidad → emoji
                                _sev_h = (_imp_h.get("severidad")
                                           or "").lower()
                                _ico_h = {
                                    "critico": "🔴",
                                    "operativo": "🟠",
                                    "atencion": "🟡",
                                }.get(_sev_h, "⚪")
                                # Estado del registro
                                _estado_h = (_imp_h.get("estado")
                                              or "proyectado").lower()
                                _estado_label = {
                                    "proyectado": "📅 Proyectado",
                                    "confirmado": "✅ Confirmado",
                                }.get(_estado_h, _estado_h)
                                _g_min_h = _imp_h.get("gasto_extra_pct_min")
                                _g_max_h = _imp_h.get("gasto_extra_pct_max")
                                _adpv_min_h = _imp_h.get(
                                    "adpv_perdida_min_kg")
                                _adpv_max_h = _imp_h.get(
                                    "adpv_perdida_max_kg")
                                _kg_min_h = _imp_h.get("kg_lote_total_min")
                                _kg_max_h = _imp_h.get("kg_lote_total_max")
                                _dias_h = _imp_h.get("dias_evento") or 1
                                _filas.append({
                                    "Fecha cálculo":
                                        _imp_h.get("fecha_calculo", "")[:10],
                                    "Inicio evento":
                                        _imp_h.get(
                                            "fecha_inicio_evento") or "—",
                                    "Estado": _estado_label,
                                    "Tipo": (
                                        f"{_ico_h} "
                                        f"{(_imp_h.get('tipo_evento') or '').capitalize()} "
                                        f"{_sev_h}"
                                    ),
                                    "Días": _dias_h,
                                    "Gasto extra":
                                        f"+{_g_min_h:.0f}–{_g_max_h:.0f}%"
                                        if _g_min_h is not None
                                        else "—",
                                    "ADPV en riesgo (kg/día)":
                                        f"{_adpv_min_h:.2f}–{_adpv_max_h:.2f}"
                                        if _adpv_min_h is not None
                                        else "—",
                                    "Total lote evento (kg)":
                                        f"{_kg_min_h:.0f}–{_kg_max_h:.0f}"
                                        if _kg_min_h is not None
                                        else "—",
                                })
                            _df_hist = _pd_hist.DataFrame(_filas)
                            st.dataframe(
                                _df_hist, hide_index=True,
                                width="stretch",
                            )
                            _n_proy = sum(
                                1 for _i in _impactos
                                if (_i.get("estado") or
                                    "proyectado") == "proyectado"
                            )
                            _n_conf = sum(
                                1 for _i in _impactos
                                if _i.get("estado") == "confirmado"
                            )
                            st.caption(
                                f"📊 {len(_impactos)} registros guardados "
                                f"({_n_proy} proyecciones + {_n_conf} "
                                f"confirmados con clima real). "
                                "**📅 Proyectado** = cálculo basado en el "
                                "pronóstico de esa semana. "
                                "**✅ Confirmado** = recalculado con clima "
                                "real ocurrido (datos observados de "
                                "Open-Meteo Archive, con ~7 días de delay)."
                            )

                            # ---- Detalle consultable de un registro ----
                            st.markdown("---")
                            st.markdown(
                                "**🔍 Ver detalle de un registro pasado**"
                            )
                            _opciones_hist = {
                                f"{_imp_h.get('fecha_calculo','')[:10]} · "
                                f"{(_imp_h.get('tipo_evento') or '').capitalize()} "
                                f"{(_imp_h.get('severidad') or '').lower()} · "
                                f"{('📅 Proyectado' if (_imp_h.get('estado') or 'proyectado')=='proyectado' else '✅ Confirmado')}":
                                _imp_h["id"]
                                for _imp_h in _impactos
                            }
                            _sel_label = st.selectbox(
                                "Seleccioná un registro para ver el "
                                "detalle completo (impacto, prevención "
                                "y clima de ese momento):",
                                options=["—"] + list(_opciones_hist.keys()),
                                key=f"hist_sel_{lote_sel}",
                            )
                            if _sel_label and _sel_label != "—":
                                _reg_id = _opciones_hist[_sel_label]
                                _reg = next(
                                    (r for r in _impactos
                                     if r["id"] == _reg_id), None,
                                )
                                if _reg:
                                    from src.impacto_productivo import (
                                        formato_impacto_humano_desde_registro,
                                    )
                                    from src.prevencion import (
                                        acciones_preventivas_html,
                                    )
                                    import json as _json_hist

                                    # Severidad y clima del registro
                                    _sev_r = (_reg.get("severidad")
                                               or "operativo")
                                    _tipo_r = (_reg.get("tipo_evento")
                                                or "frio")
                                    _clima_r = {}
                                    if _reg.get("clima_resumen_json"):
                                        try:
                                            _clima_r = _json_hist.loads(
                                                _reg["clima_resumen_json"]
                                            )
                                        except (ValueError, TypeError):
                                            _clima_r = {}

                                    # Tonalidad del card según severidad
                                    _estado_r = (_reg.get("estado")
                                                  or "proyectado")
                                    _estado_badge = (
                                        "📅 Proyección"
                                        if _estado_r == "proyectado"
                                        else "✅ Confirmado (clima real)"
                                    )
                                    _bg_r, _border_r, _ico_r, _tit_r = {
                                        "critico": (
                                            "#FDECEC", "#C0392B", "🔴",
                                            f"Frío crítico · {_estado_badge}"
                                        ),
                                        "operativo": (
                                            "#FFF6E5", "#E67E22", "🟠",
                                            f"Frío operativo · {_estado_badge}"
                                        ),
                                        "atencion": (
                                            "#FFFCEC", "#C9A227", "🟡",
                                            f"Frío leve · {_estado_badge}"
                                        ),
                                    }.get(_sev_r, ("#F4F4F4", "#666", "⚪",
                                                    f"Registro · {_estado_badge}"))

                                    # Card del impacto reconstruido
                                    _texto_r = formato_impacto_humano_desde_registro(_reg)
                                    st.markdown(
                                        f"""
                                        <div style="background:{_bg_r};
                                          border-left:4px solid {_border_r};
                                          padding:14px 16px;
                                          border-radius:6px;
                                          margin-top:10px;">
                                          <div style="font-size:13px;
                                            font-weight:700;
                                            color:{_border_r};
                                            letter-spacing:0.3px;
                                            margin-bottom:6px;">
                                            {_ico_r} {_tit_r}
                                          </div>
                                          {_texto_r}
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    # Resumen climático del registro
                                    if _clima_r:
                                        st.markdown(
                                            "**🌤️ Condiciones climáticas "
                                            "registradas esa semana:**"
                                        )
                                        _cm1, _cm2, _cm3, _cm4 = (
                                            st.columns(4)
                                        )
                                        _cm1.metric(
                                            "T° mínima",
                                            f"{_clima_r.get('t_min', 0):.0f}°C",
                                        )
                                        _cm2.metric(
                                            "T° máxima",
                                            f"{_clima_r.get('t_max', 0):.0f}°C",
                                        )
                                        _cm3.metric(
                                            "HR máx",
                                            f"{_clima_r.get('hr_max', 0):.0f}%",
                                        )
                                        _cm4.metric(
                                            "Viento máx",
                                            f"{_clima_r.get('viento_max', 0):.0f} km/h",
                                        )
                                        if _clima_r.get("barro"):
                                            st.caption(
                                                "🟫 Barro probable durante "
                                                "ese período (>20 mm en "
                                                "3 días)."
                                            )

                                    # Prevención que aplicaba al registro
                                    _prev_html_r = acciones_preventivas_html(
                                        tipo_evento=_tipo_r,
                                        severidad=_sev_r,
                                        categoria=l_data.get(
                                            "categoria", ""),
                                        barro=_clima_r.get("barro", False),
                                        pelaje_mojado=(
                                            _clima_r.get("hr_max", 0) >= 85
                                        ),
                                    )
                                    if _prev_html_r:
                                        st.markdown(
                                            f"""
                                            <div style="background:#F4F8F4;
                                              border-left:4px solid #1B3E27;
                                              padding:14px 16px;
                                              border-radius:6px;
                                              margin-top:10px;">
                                              <div style="font-size:13px;
                                                font-weight:700;
                                                color:#1B3E27;
                                                letter-spacing:0.3px;
                                                margin-bottom:6px;">
                                                🛡️ Prevención y cuidado
                                                que aplicaba a este
                                                evento
                                              </div>
                                              {_prev_html_r}
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                    # Botón para borrar el registro
                                    _del1, _del2 = st.columns([3, 1])
                                    with _del2:
                                        if st.button(
                                            "🗑️ Borrar este registro",
                                            key=f"del_hist_{_reg_id}",
                                            type="secondary",
                                            help="Borra solo este "
                                                 "registro del histórico.",
                                        ):
                                            db.eliminar_impacto_lote(_reg_id)
                                            st.success(
                                                "Registro eliminado."
                                            )
                                            st.rerun()

                    # ---- Carga manual de pesadas (sin drone) ----
                    with st.expander(
                        "✏️ Cargar pesada manual (sin drone — balanza, manga, estimación)",
                        expanded=False,
                    ):
                        st.caption(
                            "Para clientes que no usan drone. Carga directa "
                            "del peso medido en balanza, manga, o por "
                            "estimación visual del asesor."
                        )
                        with st.form(f"pesada_manual_{lote_sel}"):
                            col_pm1, col_pm2 = st.columns(2)
                            with col_pm1:
                                pm_fecha = st.date_input(
                                    "Fecha de pesada",
                                    value=datetime.now().date(),
                                    key=f"pm_fecha_{lote_sel}",
                                )
                                pm_metodo = st.selectbox(
                                    "Método", [
                                        "balanza",
                                        "manga",
                                        "estimacion_visual",
                                        "cinta_torácica",
                                        "promedio_lote",
                                        "otro",
                                    ],
                                    key=f"pm_metodo_{lote_sel}",
                                    help=(
                                        "balanza: jaula individual o de lote. "
                                        "manga: pesaje de paso. "
                                        "estimacion_visual: a ojo del asesor. "
                                        "cinta_torácica: con cinta zoométrica."
                                    ),
                                )
                                pm_cantidad = st.number_input(
                                    "Cantidad de animales pesados",
                                    min_value=1, max_value=10000,
                                    value=l_data.get("cantidad_inicial") or 1,
                                    step=1,
                                    key=f"pm_cant_{lote_sel}",
                                )
                            with col_pm2:
                                pm_peso_prom = st.number_input(
                                    "Peso promedio (kg)",
                                    min_value=10.0, max_value=1500.0,
                                    value=float(l_data.get("ultimo_peso_kg")
                                                or l_data.get("peso_ingreso_kg") or 250),
                                    step=1.0,
                                    key=f"pm_peso_{lote_sel}",
                                )
                                pm_desvio = st.number_input(
                                    "Desvío estándar (kg, opcional)",
                                    min_value=0.0, max_value=200.0,
                                    value=0.0, step=1.0,
                                    key=f"pm_desv_{lote_sel}",
                                    help="Si pesaste todos individualmente. "
                                         "Si es promedio del lote, dejá 0.",
                                )
                                pm_obs = st.text_area(
                                    "Observaciones",
                                    height=80,
                                    placeholder="Ej: pesada de balanza con jaula, "
                                                "tarado el día anterior...",
                                    key=f"pm_obs_{lote_sel}",
                                )

                            # Permitir cargar pesos individuales (CSV opcional)
                            st.markdown("**🔢 Pesos individuales (opcional)**")
                            st.caption(
                                "Si tenés los pesos individuales en lugar del "
                                "promedio, pegalos separados por coma o subí CSV. "
                                "El sistema calcula promedio y desvío automáticamente."
                            )
                            pm_individuales_txt = st.text_area(
                                "Pegá pesos individuales separados por coma",
                                placeholder="Ej: 285, 290, 275, 310, 295, 280, ...",
                                key=f"pm_indiv_{lote_sel}",
                                height=80,
                            )

                            cargar_btn = st.form_submit_button(
                                "💾 Guardar pesada manual",
                                type="primary",
                            )
                            if cargar_btn:
                                try:
                                    pesos_individuales = []
                                    # Si pegó pesos individuales, parsear
                                    if pm_individuales_txt.strip():
                                        try:
                                            pesos_individuales = [
                                                float(x.strip())
                                                for x in pm_individuales_txt
                                                    .replace("\n", ",")
                                                    .replace(";", ",")
                                                    .split(",")
                                                if x.strip()
                                            ]
                                            if pesos_individuales:
                                                # Recalcular promedio y desvío
                                                import numpy as np
                                                pm_peso_prom = float(np.mean(pesos_individuales))
                                                pm_desvio = float(np.std(pesos_individuales))
                                                pm_cantidad = len(pesos_individuales)
                                        except ValueError as e:
                                            st.error(
                                                f"Error parseando pesos individuales: {e}"
                                            )
                                            st.stop()

                                    pid = db.guardar_pesada(
                                        lote_id=lote_sel,
                                        fecha=pm_fecha.isoformat(),
                                        metodo=pm_metodo,
                                        cantidad_animales=int(pm_cantidad),
                                        peso_promedio_kg=float(pm_peso_prom),
                                        peso_total_kg=float(pm_peso_prom * pm_cantidad),
                                        desvio_kg=float(pm_desvio),
                                        pesos_individuales=pesos_individuales,
                                        video_path="",
                                        notas=pm_obs,
                                    )
                                    st.success(
                                        f"✅ Pesada manual guardada (id {pid}). "
                                        f"Vela en **📚 Historial**."
                                    )
                                    if pesos_individuales:
                                        st.info(
                                            f"📊 Calculé automáticamente: "
                                            f"{len(pesos_individuales)} animales, "
                                            f"prom {pm_peso_prom:.1f} kg, "
                                            f"desvío {pm_desvio:.1f} kg"
                                        )
                                except Exception as e:
                                    st.error(f"Error: {e}")


                    # === Seguimiento completo del lote ===
                    # MOVIDO arriba (justo después de los botones de
                    # estado) para que el detalle del lote sea lo
                    # primero que se ve. Lo que queda abajo son los
                    # expanders de configuración secundarios.

# ----------------------------- IMAGEN ---------------------------------
with tab_img:
    file = st.file_uploader(
        "Subí una imagen del lote (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="img_upload",
    )
    if file:
        bytes_data = file.getvalue()
        cache_key = _hash_bytes(
            bytes_data, modelo_path, conf, iou, imgsz, raza, categoria,
            ajuste_fino, cfg["referencia"]["metodo"], cfg["referencia"]["lado_m"],
        )

        if st.session_state.get("img_cache_key") != cache_key:
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                st.error("No se pudo leer la imagen.")
                st.stop()

            with st.spinner("Detectando animales y estimando pesos…"):
                annotated, result = process_image(
                    img, detector, weight_model, cfg, raza, categoria,
                    ajuste_fino=ajuste_fino,
                )

            _, png_buf = cv2.imencode(".png", annotated)
            st.session_state["img_cache_key"] = cache_key
            st.session_state["img_annotated_rgb"] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.session_state["img_png_bytes"] = png_buf.tobytes()
            st.session_state["img_orig_name"] = file.name
            st.session_state["img_n"] = result.n_animales
            st.session_state["img_prom"] = result.peso_promedio_kg
            st.session_state["img_total"] = result.peso_total_kg
            st.session_state["img_desv"] = result.desvio_kg
            st.session_state["img_cal_method"] = (
                result.calibracion.method if result.calibracion else None
            )
            st.session_state["img_cal_ppm"] = (
                result.calibracion.pixels_per_meter if result.calibracion else 0
            )
            st.session_state["img_animales"] = [
                {"Animal": a.track_id, "Peso (kg)": round(a.peso_kg, 1)}
                for a in result.animales
            ]

        # ---- Renderizar desde session_state ----
        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(
                st.session_state["img_annotated_rgb"],
                caption="Resultado",
                use_column_width=True,
            )
        with col2:
            st.metric("Animales detectados", st.session_state["img_n"])
            st.metric("Peso promedio", f"{st.session_state['img_prom']:.1f} kg")
            st.metric("Peso total del lote", f"{st.session_state['img_total']:.0f} kg")
            st.metric("Desvío estándar", f"{st.session_state['img_desv']:.1f} kg")

            if st.session_state.get("img_cal_method"):
                st.caption(
                    f"📏 Calibración: {st.session_state['img_cal_method']} — "
                    f"{st.session_state['img_cal_ppm']:.0f} px/m"
                )

            df = pd.DataFrame(st.session_state["img_animales"])
            st.dataframe(df, hide_index=True, width="stretch")

            base_name = Path(st.session_state["img_orig_name"]).stem
            st.download_button(
                "📥 Descargar imagen anotada (PNG)",
                data=st.session_state["img_png_bytes"],
                file_name=f"{base_name}_anotado.png",
                mime="image/png",
                key="dl_img",
            )
            if not df.empty:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Descargar tabla (CSV)",
                    data=csv_bytes,
                    file_name=f"{base_name}_pesos.csv",
                    mime="text/csv",
                    key="dl_img_csv",
                )

# ----------------------------- VIDEO ----------------------------------
with tab_vid:
    st.info(
        "💡 **¿No usás drone con este cliente?** Podés cargar pesadas "
        "manuales (balanza, manga, estimación) en la pestaña "
        "**🏢 Clientes/Lotes** → seleccionar lote → "
        "*'✏️ Cargar pesada manual'*. Las pesadas manuales se integran "
        "igual con el resto del sistema (historial, ADG, asesor IA, dietas)."
    )
    file = st.file_uploader(
        "Subí un video del lote (MP4/MOV)",
        type=["mp4", "mov", "avi"],
        key="vid_upload",
    )

    if file:
        # Leer bytes una sola vez y calcular hash combinado con parámetros
        bytes_data = file.getvalue()
        cache_key = _hash_bytes(
            bytes_data, modelo_path, conf, iou, imgsz, raza, categoria,
            ajuste_fino, cfg["referencia"]["metodo"], cfg["referencia"]["lado_m"],
        )

        # Si ya procesamos exactamente este video con esta configuración,
        # reusamos. La descarga de archivos NO dispara reprocesamiento.
        if st.session_state.get("vid_cache_key") != cache_key:
            with tempfile.NamedTemporaryFile(
                suffix=Path(file.name).suffix, delete=False
            ) as tmp:
                tmp.write(bytes_data)
                in_path = Path(tmp.name)

            out_path = in_path.with_name(in_path.stem + "_anotado.mp4")
            csv_path = in_path.with_name(in_path.stem + "_pesos.csv")

            progress = st.progress(0.0, text="Procesando video…")

            def cb(p: float):
                progress.progress(min(p, 1.0), text=f"Procesando video… {p*100:.0f}%")

            with st.spinner("Procesando — esto puede tardar varios minutos según duración"):
                result = process_video(
                    in_path, out_path, detector, weight_model, cfg,
                    raza=raza, categoria=categoria,
                    ajuste_fino=ajuste_fino, progress_cb=cb,
                )
                export_results_csv(result, csv_path)

            progress.progress(1.0, text="¡Listo!")

            # Guardar todo en session_state
            st.session_state["vid_cache_key"] = cache_key
            st.session_state["vid_out_path"] = str(out_path)
            st.session_state["vid_csv_path"] = str(csv_path)
            st.session_state["vid_video_bytes"] = out_path.read_bytes()
            st.session_state["vid_csv_bytes"] = csv_path.read_bytes()
            st.session_state["vid_orig_name"] = file.name
            st.session_state["vid_n"] = result.n_animales
            st.session_state["vid_prom"] = result.peso_promedio_kg
            st.session_state["vid_total"] = result.peso_total_kg
            st.session_state["vid_desv"] = result.desvio_kg
            st.session_state["vid_calidad_pct"] = result.calidad_captura_pct
            st.session_state["vid_frames_total"] = result.n_frames_total
            st.session_state["vid_frames_validos"] = result.n_frames_validos
            st.session_state["vid_frames_sin_ref"] = result.n_frames_sin_ref
            st.session_state["vid_frames_tilted"] = result.n_frames_tilted
            st.session_state["vid_animales"] = [
                {"Animal": a.track_id, "Peso (kg)": round(a.peso_kg, 1)}
                for a in result.animales
            ]

        # ---- Renderizar resultado desde session_state ----
        col1, col2 = st.columns([3, 2])
        with col1:
            st.video(st.session_state["vid_out_path"])
        with col2:
            st.metric("Animales únicos", st.session_state["vid_n"])
            st.metric("Peso promedio", f"{st.session_state['vid_prom']:.1f} kg")
            st.metric("Peso total", f"{st.session_state['vid_total']:.0f} kg")
            st.metric("Desvío estándar", f"{st.session_state['vid_desv']:.1f} kg")

            df = pd.DataFrame(st.session_state["vid_animales"])
            st.dataframe(df, hide_index=True, width="stretch")

            # Panel de calidad de captura
            calidad = st.session_state.get("vid_calidad_pct", 100)
            sin_ref = st.session_state.get("vid_frames_sin_ref", 0)
            tilted = st.session_state.get("vid_frames_tilted", 0)
            total = st.session_state.get("vid_frames_total", 1)
            if calidad >= 90:
                st.success(
                    f"✅ Calidad de captura: {calidad:.0f}% — "
                    f"setup ideal, los pesos son confiables."
                )
            elif calidad >= 70:
                st.warning(
                    f"⚠️ Calidad de captura: {calidad:.0f}% — "
                    f"{sin_ref} frames sin lona visible, {tilted} con cámara inclinada. "
                    "Reforzá: lona siempre en cuadro y gimbal estable."
                )
            else:
                st.error(
                    f"🔴 Calidad baja: {calidad:.0f}% válidos. "
                    f"Sin referencia: {sin_ref}/{total}, inclinados: {tilted}/{total}. "
                    "El peso puede ser impreciso. Refilmá con mejor encuadre."
                )

            base_name = Path(st.session_state["vid_orig_name"]).stem
            st.download_button(
                "📥 Descargar video anotado",
                data=st.session_state["vid_video_bytes"],
                file_name=f"{base_name}_anotado.mp4",
                mime="video/mp4",
                key="dl_video",
            )
            st.download_button(
                "📥 Descargar CSV de pesos",
                data=st.session_state["vid_csv_bytes"],
                file_name=f"{base_name}_pesos.csv",
                mime="text/csv",
                key="dl_csv",
            )

        # ----- GUARDAR AL HISTORIAL DEL LOTE -----
        st.divider()
        st.markdown("### 💾 Guardar esta pesada al histórico de un lote")
        lotes_activos = db.listar_lotes(estado="activo")
        if not lotes_activos:
            st.info(
                "No hay lotes activos para asociar. Cargá un cliente y un lote en "
                "la pestaña **🏢 Clientes y Lotes** para guardar pesadas."
            )
        else:
            col_g1, col_g2, col_g3 = st.columns([2, 1, 1])
            with col_g1:
                lote_guardar = st.selectbox(
                    "Lote",
                    [l["id"] for l in lotes_activos],
                    format_func=lambda x: next(
                        f"{l['cliente_nombre']} — {l['identificador']} "
                        f"(corral {l.get('corral','—')})"
                        for l in lotes_activos if l["id"] == x
                    ),
                    key="lote_para_pesada",
                )
            with col_g2:
                fecha_pesada = st.date_input(
                    "Fecha", value=datetime.now().date(), key="fecha_pes",
                )
            with col_g3:
                notas_pesada = st.text_input("Notas (opcional)", key="notas_pes")

            if st.button("💾 Guardar pesada al historial", type="primary"):
                pesos_lista = [a["Peso (kg)"] for a in
                               st.session_state["vid_animales"]]
                try:
                    pid = db.guardar_pesada(
                        lote_id=lote_guardar,
                        fecha=fecha_pesada.isoformat(),
                        metodo="drone",
                        cantidad_animales=st.session_state["vid_n"],
                        peso_promedio_kg=st.session_state["vid_prom"],
                        peso_total_kg=st.session_state["vid_total"],
                        desvio_kg=st.session_state["vid_desv"],
                        pesos_individuales=pesos_lista,
                        video_path=st.session_state.get("vid_orig_name", ""),
                        notas=notas_pesada,
                    )
                    st.success(
                        f"✅ Pesada guardada (id {pid}). Vela en la pestaña "
                        f"**📚 Historial**."
                    )
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

# --------------------------- EVOLUCIÓN --------------------------------
with tab_evo:
    st.markdown(
        "### 📈 Comparación entre dos fechas (ADG y eficiencia)\n"
        "Subí dos videos del mismo lote tomados en distintas fechas para "
        "calcular **ganancia diaria de peso (ADG)**, el indicador clave para "
        "evaluar la respuesta a una dieta."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📅 Pesada inicial")
        fecha_ini = st.date_input("Fecha", key="evo_fecha_ini")
        file_ini = st.file_uploader(
            "Video pesada inicial",
            type=["mp4", "mov", "avi"],
            key="evo_vid_ini",
        )
        kg_ms_dia_ini = st.number_input(
            "kg materia seca / animal / día (dieta inicial)",
            min_value=0.0, max_value=20.0, value=8.0, step=0.1,
            key="evo_kgms_ini",
            help="Para calcular conversión alimenticia",
        )

    with col_b:
        st.subheader("📅 Pesada final")
        fecha_fin = st.date_input("Fecha", key="evo_fecha_fin")
        file_fin = st.file_uploader(
            "Video pesada final",
            type=["mp4", "mov", "avi"],
            key="evo_vid_fin",
        )
        kg_ms_dia_fin = st.number_input(
            "kg materia seca / animal / día (dieta final)",
            min_value=0.0, max_value=20.0, value=8.0, step=0.1,
            key="evo_kgms_fin",
        )

    st.divider()

    if file_ini and file_fin:
        if fecha_fin <= fecha_ini:
            st.error("La fecha final debe ser posterior a la inicial.")
        else:
            dias = (fecha_fin - fecha_ini).days

            # Procesar ambos videos (con cache por hash)
            results = {}
            for label, file_obj in [("ini", file_ini), ("fin", file_fin)]:
                bytes_data = file_obj.getvalue()
                cache_key = _hash_bytes(
                    bytes_data, modelo_path, conf, iou, imgsz, raza, categoria,
                    ajuste_fino, cfg["referencia"]["metodo"], cfg["referencia"]["lado_m"],
                )
                ck_state = f"evo_{label}_cache_key"
                if st.session_state.get(ck_state) != cache_key:
                    with tempfile.NamedTemporaryFile(
                        suffix=Path(file_obj.name).suffix, delete=False
                    ) as tmp:
                        tmp.write(bytes_data)
                        in_p = Path(tmp.name)
                    out_p = in_p.with_name(in_p.stem + f"_evo_{label}.mp4")

                    pbar = st.progress(0.0, text=f"Procesando {label}…")
                    def cb_evo(p, _l=label, _pb=pbar):
                        _pb.progress(min(p, 1.0), text=f"Procesando {_l}… {p*100:.0f}%")

                    with st.spinner(f"Procesando video {label}…"):
                        r = process_video(
                            in_p, out_p, detector, weight_model, cfg,
                            raza=raza, categoria=categoria,
                            ajuste_fino=ajuste_fino, progress_cb=cb_evo,
                        )
                    pbar.progress(1.0, text=f"{label} listo")
                    pesos = [a.peso_kg for a in r.animales]
                    st.session_state[ck_state] = cache_key
                    st.session_state[f"evo_{label}_n"] = r.n_animales
                    st.session_state[f"evo_{label}_prom"] = r.peso_promedio_kg
                    st.session_state[f"evo_{label}_desv"] = r.desvio_kg
                    st.session_state[f"evo_{label}_total"] = r.peso_total_kg
                    st.session_state[f"evo_{label}_pesos"] = pesos

                results[label] = {
                    "n": st.session_state[f"evo_{label}_n"],
                    "prom": st.session_state[f"evo_{label}_prom"],
                    "desv": st.session_state[f"evo_{label}_desv"],
                    "total": st.session_state[f"evo_{label}_total"],
                    "pesos": st.session_state[f"evo_{label}_pesos"],
                }

            r_ini = results["ini"]
            r_fin = results["fin"]
            ganancia_total_kg = r_fin["prom"] - r_ini["prom"]
            adg = ganancia_total_kg / dias if dias else 0
            kg_ms_total_dia = (kg_ms_dia_ini + kg_ms_dia_fin) / 2
            conv_alim = kg_ms_total_dia / adg if adg > 0 else float("inf")
            cv_ini = (r_ini["desv"] / r_ini["prom"] * 100) if r_ini["prom"] else 0
            cv_fin = (r_fin["desv"] / r_fin["prom"] * 100) if r_fin["prom"] else 0

            st.markdown(f"### 📊 Resumen del período ({dias} días)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "ADG (kg/día)", f"{adg:+.3f}",
                "🟢 Bueno" if adg > 0.8 else "🟡 Bajo" if adg > 0.3 else "🔴 Crítico",
            )
            m2.metric("Ganancia total", f"{ganancia_total_kg:+.1f} kg")
            m3.metric(
                "Conversión alim.", f"{conv_alim:.1f}",
                help="kg MS consumida / kg ganado. Vaquillonas óptimo 6-8.",
            )
            m4.metric(
                "Cambio CV", f"{cv_fin - cv_ini:+.1f} pp",
                help="Variabilidad del lote. Bajar = lote más uniforme",
            )

            st.markdown("### Comparativa de pesadas")
            comp = pd.DataFrame({
                "Indicador": [
                    "Cantidad de animales", "Peso promedio (kg)",
                    "Peso total del lote (kg)", "Desvío estándar (kg)",
                    "Coef. variación (%)", "Más liviano (kg)", "Más pesado (kg)",
                ],
                f"Inicial ({fecha_ini})": [
                    r_ini["n"], f"{r_ini['prom']:.1f}", f"{r_ini['total']:.0f}",
                    f"{r_ini['desv']:.1f}", f"{cv_ini:.1f}",
                    f"{min(r_ini['pesos']):.1f}" if r_ini["pesos"] else "—",
                    f"{max(r_ini['pesos']):.1f}" if r_ini["pesos"] else "—",
                ],
                f"Final ({fecha_fin})": [
                    r_fin["n"], f"{r_fin['prom']:.1f}", f"{r_fin['total']:.0f}",
                    f"{r_fin['desv']:.1f}", f"{cv_fin:.1f}",
                    f"{min(r_fin['pesos']):.1f}" if r_fin["pesos"] else "—",
                    f"{max(r_fin['pesos']):.1f}" if r_fin["pesos"] else "—",
                ],
            })
            st.dataframe(comp, hide_index=True, width="stretch")

            # Histograma comparado
            st.markdown("### Distribución de pesos")
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(r_ini["pesos"], bins=10, alpha=0.6, label=f"Inicial ({fecha_ini})", color="#1f77b4")
                ax.hist(r_fin["pesos"], bins=10, alpha=0.6, label=f"Final ({fecha_fin})", color="#2ca02c")
                ax.axvline(r_ini["prom"], color="#1f77b4", linestyle="--", linewidth=2)
                ax.axvline(r_fin["prom"], color="#2ca02c", linestyle="--", linewidth=2)
                ax.set_xlabel("Peso (kg)")
                ax.set_ylabel("Cantidad de animales")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"No pude graficar: {e}")

            # Reporte exportable
            reporte = (
                f"REPORTE DE EVOLUCIÓN DEL LOTE\n"
                f"================================\n\n"
                f"Período: {fecha_ini} → {fecha_fin}  ({dias} días)\n"
                f"Raza: {raza}   Categoría: {categoria}\n\n"
                f"PESADA INICIAL ({fecha_ini})\n"
                f"  Animales: {r_ini['n']}\n"
                f"  Peso promedio: {r_ini['prom']:.1f} kg\n"
                f"  Peso total: {r_ini['total']:.0f} kg\n"
                f"  Desvío estándar: {r_ini['desv']:.1f} kg (CV {cv_ini:.1f}%)\n\n"
                f"PESADA FINAL ({fecha_fin})\n"
                f"  Animales: {r_fin['n']}\n"
                f"  Peso promedio: {r_fin['prom']:.1f} kg\n"
                f"  Peso total: {r_fin['total']:.0f} kg\n"
                f"  Desvío estándar: {r_fin['desv']:.1f} kg (CV {cv_fin:.1f}%)\n\n"
                f"INDICADORES NUTRICIONALES\n"
                f"  Ganancia total: {ganancia_total_kg:+.1f} kg/animal\n"
                f"  ADG: {adg:+.3f} kg/día\n"
                f"  kg MS/día prom.: {kg_ms_total_dia:.1f}\n"
                f"  Conversión alimenticia: {conv_alim:.2f} (kg MS / kg ganado)\n"
                f"  Cambio en CV: {cv_fin - cv_ini:+.1f} pp\n"
            )
            st.download_button(
                "📥 Descargar reporte (TXT)",
                data=reporte.encode("utf-8"),
                file_name=f"reporte_evolucion_{fecha_ini}_{fecha_fin}.txt",
                mime="text/plain",
                key="dl_reporte_evo",
            )

# ----------------------- ANÁLISIS AVANZADO ----------------------------
with tab_avanzado:
    st.markdown(
        "### 🔬 Análisis estadístico y diagnóstico del lote\n"
        "Esta pestaña usa el último video procesado en la pestaña 🎞️ Video. "
        "Muestra percentiles, identifica outliers y diagnostica uniformidad."
    )

    if "vid_animales" not in st.session_state or not st.session_state.get("vid_animales"):
        st.info("Procesá un video en la pestaña 🎞️ Video para ver el análisis.")
    else:
        animales_dict = [
            {"track_id": a["Animal"], "peso_kg": a["Peso (kg)"]}
            for a in st.session_state["vid_animales"]
        ]
        unif = analizar_uniformidad(animales_dict)

        # Cards principales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Animales", unif.n)
        c2.metric("Promedio", f"{unif.promedio_kg:.1f} kg")
        c3.metric("Mediana", f"{unif.mediana_kg:.1f} kg")
        c4.metric("CV", f"{unif.cv_pct:.1f}%",
                  "🟢" if unif.cv_pct < 12 else "🟡" if unif.cv_pct < 18 else "🔴")

        st.markdown(f"#### {unif.diagnostico}")
        st.info(f"💡 {unif.recomendacion}")

        # Tabla de percentiles
        st.markdown("#### Distribución por percentiles")
        perc_df = pd.DataFrame({
            "Percentil": ["Mínimo", "P10 (cabeza-baja)", "P25", "Mediana (P50)",
                          "P75", "P90 (cabeza-alta)", "Máximo"],
            "Peso (kg)": [
                f"{unif.min_kg:.1f}", f"{unif.p10_kg:.1f}",
                f"{unif.p25_kg:.1f}", f"{unif.mediana_kg:.1f}",
                f"{unif.p75_kg:.1f}", f"{unif.p90_kg:.1f}",
                f"{unif.max_kg:.1f}",
            ],
        })
        st.dataframe(perc_df, hide_index=True, width="stretch")

        # Histograma + boxplot
        st.markdown("#### Distribución gráfica")
        try:
            import matplotlib.pyplot as plt
            pesos = [a["peso_kg"] for a in animales_dict]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                                      gridspec_kw={"width_ratios": [3, 1]})
            axes[0].hist(pesos, bins=12, color="#97bc62", edgecolor="#2c5f2d")
            axes[0].axvline(unif.promedio_kg, color="#2c5f2d", linestyle="--",
                            label=f"Promedio {unif.promedio_kg:.0f}")
            axes[0].axvline(unif.mediana_kg, color="#cc3300", linestyle=":",
                            label=f"Mediana {unif.mediana_kg:.0f}")
            axes[0].set_xlabel("Peso (kg)")
            axes[0].set_ylabel("Cantidad")
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            axes[1].boxplot(pesos, patch_artist=True,
                            boxprops=dict(facecolor="#97bc62"))
            axes[1].set_ylabel("Peso (kg)")
            axes[1].grid(alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Gráfico no disponible: {e}")

        # Outliers
        if unif.outliers_low or unif.outliers_high:
            st.markdown("#### ⚠️ Animales fuera del rango uniforme")
            col_low, col_high = st.columns(2)
            with col_low:
                if unif.outliers_low:
                    st.markdown(f"**Cabeza-baja** ({len(unif.outliers_low)}):")
                    st.write(unif.outliers_low)
                else:
                    st.markdown("**Cabeza-baja**: ninguno ✅")
            with col_high:
                if unif.outliers_high:
                    st.markdown(f"**Cabeza-alta** ({len(unif.outliers_high)}):")
                    st.write(unif.outliers_high)
                else:
                    st.markdown("**Cabeza-alta**: ninguno ✅")

        # Proyección a futuro (sin ADG real, asume objetivo)
        st.divider()
        st.markdown("### 📅 Proyección a fecha futura")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            adg_estimado = st.number_input(
                "ADG estimado (kg/día)", min_value=0.0, max_value=2.5,
                value=1.0, step=0.05, key="proj_adg",
                help="Si ya hiciste pesada anterior, usá el de la pestaña 📈 Evolución",
            )
        with col_p2:
            dias_proyeccion = st.number_input(
                "Proyectar a (días)", min_value=7, max_value=365,
                value=60, step=1, key="proj_dias",
            )
        with col_p3:
            objetivo_kg = st.number_input(
                "Peso objetivo (kg, opcional)", min_value=0.0, max_value=1200.0,
                value=400.0, step=5.0, key="proj_obj",
            )

        proj = proyectar_peso(
            unif.promedio_kg, adg_estimado, dias_proyeccion,
            peso_objetivo_kg=objetivo_kg if objetivo_kg > 0 else None,
        )
        cp1, cp2, cp3 = st.columns(3)
        cp1.metric("Peso a la fecha", f"{proj.peso_proyectado_kg:.1f} kg",
                   f"+{proj.peso_proyectado_kg - unif.promedio_kg:.0f} kg")
        cp2.metric("Rango estimado",
                   f"{proj.intervalo_confianza[0]:.0f}–{proj.intervalo_confianza[1]:.0f} kg")
        if proj.cumple_objetivo is not None:
            if proj.cumple_objetivo:
                cp3.metric("Cumple objetivo", "✅ SÍ", f"+{proj.diferencia_objetivo_kg:.0f} kg")
            else:
                cp3.metric("Cumple objetivo", "⚠️ NO",
                           f"Falta {-proj.diferencia_objetivo_kg:.0f} kg")
                st.warning(
                    f"Para llegar a {objetivo_kg:.0f} kg en {dias_proyeccion} días "
                    f"necesitás ADG de **{proj.adg_requerido_para_objetivo:.3f} kg/día** "
                    f"(actual: {adg_estimado:.3f}). Ajustar dieta."
                )

        # Guardar la proyección y uniformidad para el PDF
        st.session_state["last_uniformidad"] = unif
        st.session_state["last_proyeccion"] = proj

        # ---- BOTÓN PDF ----
        st.divider()
        st.markdown("### 📄 Generar reporte PDF profesional")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            cliente_pdf = st.text_input(
                "Cliente",
                key="pdf_cliente_analisis",
                placeholder="Ej: Ezequiel Pezzola",
            )
            establecimiento = st.text_input(
                "Establecimiento", key="pdf_estab",
            )
            asesor = st.text_input(
                "Asesor / Técnico",
                value="Mauricio Suárez — Asesor Técnico Nutricional",
                key="pdf_asesor",
            )
        with col_h2:
            lote = st.text_input("Identificación del lote", key="pdf_lote")
            objetivo_pdf = st.text_input(
                "Objetivo productivo",
                key="pdf_objetivo_analisis",
                placeholder="Ej: ADG 0.8 kg/día — Terminación",
            )
            notas_pdf = st.text_area(
                "Notas adicionales", key="pdf_notas", height=80,
            )

        if st.button("📄 Generar PDF", type="primary"):
            try:
                _nombre_pdf = armar_nombre_pdf(
                    cliente=cliente_pdf,
                    categoria=categoria,
                    objetivo=objetivo_pdf,
                    lote=lote,
                    sufijo="reporte",
                )
                pdf_path = Path(tempfile.mkdtemp()) / _nombre_pdf
                generar_pdf(
                    pdf_path,
                    establecimiento=establecimiento, asesor=asesor, lote=lote,
                    raza=raza, categoria=categoria,
                    n_animales=unif.n,
                    peso_promedio_kg=unif.promedio_kg,
                    peso_total_kg=unif.promedio_kg * unif.n,
                    desvio_kg=unif.desvio_kg,
                    animales=animales_dict,
                    uniformidad=unif,
                    proyeccion=proj if objetivo_kg > 0 or adg_estimado > 0 else None,
                    calidad_pct=st.session_state.get("vid_calidad_pct", 100),
                    notas_extra=notas_pdf,
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Descargar PDF",
                        data=f.read(),
                        file_name=_nombre_pdf,
                        mime="application/pdf",
                        key="dl_pdf_avanzado",
                    )
                st.success("✅ PDF generado")
            except Exception as e:
                st.error(f"Error generando PDF: {e}")
                st.exception(e)


# ------------------------- DIETA RECOMENDADA --------------------------
# NOTA: la pestaña 🍽️ Dieta fue eliminada del menú principal.
# El agente IA ahora hace toda la formulación con su contexto rico
# (NASEM + clima + memoria + alertas).
# Lo que sobrevive es la edición de ingredientes, que se movió a
# 🕲 Configuración. El optimizador LP queda accesible internamente
# para que el agente lo invoque vía tool use.

# Bloque del optimizador de dietas (NASEM + LP) — VISIBLE en la
# pestaña Análisis 🔬 como segunda sección.
# Coexiste con las tools del Asesor IA: la app es centro de comando
# de Mauricio (asesor HMS), no del productor final. Mauricio tiene
# DOS formas de operar: (1) el formulario visual de acá, o (2) el
# chat conversacional del Asesor IA que invoca las mismas funciones.
# Cada forma sirve para distintos momentos del trabajo del asesor.
with tab_avanzado:
    st.divider()
    st.markdown(
        "### 🍽️ Formulación de dieta de mínimo costo\n"
        "Basado en **NASEM 2016 (8th Ed.)** + optimización por programación lineal.\n\n"
        "**Cómo funciona**: definís qué ingredientes tenés disponibles y a qué precio, "
        "ingresás los datos del lote, y la app calcula la mezcla de mínimo costo "
        "que cubre los requerimientos nutricionales. Si no se pueden cubrir, "
        "te dice qué ingredientes faltan."
    )

    # ---- BLOQUE A: parámetros del lote ----
    st.markdown("#### 1️⃣ Parámetros del lote")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        peso_d = st.number_input(
            "Peso vivo promedio (kg)", min_value=50.0, max_value=1200.0,
            value=float(st.session_state.get("vid_prom", 300.0)) if "vid_prom" in st.session_state else 300.0,
            step=5.0, key="dieta_peso",
            help="Por defecto carga el promedio del último análisis de video",
        )
    with col_d2:
        adg_obj = st.number_input(
            "ADG objetivo (kg/día)", min_value=0.0, max_value=2.5,
            value=1.0, step=0.05, key="dieta_adg",
        )
    with col_d3:
        estres_calorico = st.toggle("Estrés calórico", key="dieta_estres")

    try:
        _cats_dieta = db.nombres_categorias()
    except Exception:
        _cats_dieta = []
    if not _cats_dieta:
        _cats_dieta = ["ternero", "vaquillona", "novillo",
                       "vaca_adulta", "toro"]
    _default_dieta = ("vaquillona"
                      if "vaquillona" in _cats_dieta else _cats_dieta[0])
    cat_d = st.selectbox(
        "Categoría",
        _cats_dieta,
        index=_cats_dieta.index(_default_dieta), key="dieta_cat",
    )
    raza_d = st.selectbox(
        "Raza",
        ["angus", "hereford", "brangus", "braford", "cruza", "cebuino"],
        index=0, key="dieta_raza",
    )

    ajuste_pb_extra = st.slider(
        "Ajuste fino PB (multiplicador sobre cálculo NASEM)",
        0.80, 1.20, 1.00, 0.01,
        help="Default 1.00 = NASEM puro. Si tu experiencia indica que el "
             "valor de NASEM está alto/bajo, ajustá manualmente. La app SIEMPRE "
             "te muestra el rango de la práctica argentina como referencia "
             "cruzada para validar.",
    )

    # Calcular requerimientos automáticamente (NASEM riguroso)
    req = calcular_requerimientos(
        peso_vivo_kg=peso_d, adg_objetivo_kg=adg_obj,
        categoria=cat_d, raza=raza_d,
        dias_estres_calorico=estres_calorico,
        ajuste_pb_pct=ajuste_pb_extra,
    )

    with st.expander("📊 Requerimientos calculados — NASEM 2016 vs práctica argentina", expanded=True):
        st.caption(
            f"**Etapa detectada**: `{req.etapa}` (categoría {cat_d}, peso {peso_d:.0f} kg, ADG {adg_obj:.2f})"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Consumo MS", f"{req.consumo_ms_kg:.2f} kg/día",
                  f"{req.consumo_ms_pct_pv:.1f}% PV")
        m2.metric("Prot. Metabolizable (NASEM)", f"{req.mp_requerida_g:.0f} g/día")
        m3.metric("Energía Metab.", f"{req.em_mcal:.1f} Mcal/día")

        # Comparativa PB: NASEM vs rango práctica
        st.markdown("##### 🎯 Proteína bruta — comparativa")
        c1, c2, c3 = st.columns(3)
        c1.metric("NASEM 2016 calcula", f"{req.pb_pct_ms:.1f}% MS",
                  f"{req.pb_gramos:.0f} g/día")
        c2.metric(f"Rango práctica AR ({req.etapa})",
                  f"{req.pb_pct_min:.1f}–{req.pb_pct_max:.1f}% MS",
                  "Pordomingo/Latimori/IPCVA")

        # Diagnóstico: ¿NASEM cae dentro del rango práctico?
        if req.pb_pct_min <= req.pb_pct_ms <= req.pb_pct_max:
            c3.metric("Compatibilidad", "🟢 Coincide",
                      "NASEM dentro del rango práctico")
        elif req.pb_pct_ms > req.pb_pct_max:
            exceso = req.pb_pct_ms - req.pb_pct_max
            c3.metric("Compatibilidad", "🟡 NASEM alto",
                      f"+{exceso:.1f}% sobre rango práctico")
            st.info(
                f"💡 NASEM da **{req.pb_pct_ms:.1f}%** pero la práctica "
                f"argentina sostiene **{req.pb_pct_min:.1f}-{req.pb_pct_max:.1f}%** "
                f"para esta etapa ({req.etapa}). "
                "Si querés alinear con la práctica local, bajá el slider de "
                f"'Ajuste fino PB' a ≈{req.pb_pct_max/req.pb_pct_ms:.2f}."
            )
        else:
            faltante = req.pb_pct_min - req.pb_pct_ms
            c3.metric("Compatibilidad", "🔴 NASEM bajo",
                      f"-{faltante:.1f}% bajo rango")

        st.caption(
            f"📚 Fuentes: NASEM 2016 (8th Ed.) cap. 4-7 (cálculo riguroso por "
            f"ecuaciones biológicas); rango práctico de Pordomingo (INTA Anguil), "
            f"Latimori (INTA Marcos Juárez), IPCVA y AAPA."
        )

        m4, m5, m6, m7 = st.columns(4)
        m4.metric("Energía concentr.", f"{req.em_concentracion_mcal_kg:.2f} Mcal/kgMS")
        m5.metric("FDN mín", f"{req.fdn_min_pct:.0f}%")
        m6.metric("Calcio", f"{req.calcio_g:.0f} g")
        m7.metric("Fósforo", f"{req.fosforo_g:.0f} g")

    # ---- AJUSTE POR CLIMA DEL LOTE (FASE 2C) ----
    # Permite seleccionar un lote real del cliente, consultar el clima
    # esperado de la próxima semana y recalcular el DMI ajustado por
    # condiciones climáticas. Los requerimientos absolutos no cambian,
    # pero las densidades de la dieta se ajustan automáticamente.
    with st.expander(
        "🌡️ Ajustar consumo por clima del lote (FASE 2C)",
        expanded=False,
    ):
        st.caption(
            "El consumo de materia seca real depende del clima — frío "
            "sostenido lo sube (animal busca energía extra), calor o "
            "barro lo bajan. Si seleccionás un lote del cliente con "
            "coordenadas cargadas, el sistema recalcula el DMI para la "
            "semana proyectada y ajusta las densidades de la dieta "
            "(los requerimientos absolutos no cambian — el animal "
            "sigue necesitando los mismos g de PB y Mcal de EM, pero "
            "los repartís en menos o más kg de mezcla)."
        )

        # Lista de lotes activos con peso conocido y coords del cliente
        _todos_lotes = db.listar_lotes(estado="activo")
        _lotes_aptos = []
        for _l in _todos_lotes:
            _cli = db.obtener_cliente(_l["cliente_id"])
            if _cli and _cli.get("lat") and _cli.get("lon"):
                _peso_l = _l.get("ultimo_peso_kg") or _l.get(
                    "peso_ingreso_kg")
                if _peso_l and _peso_l > 0:
                    _lotes_aptos.append((_l, _cli))

        if not _lotes_aptos:
            st.info(
                "ℹ️ No hay lotes con coordenadas + peso cargados. "
                "Cargá uno en Clientes y Lotes para usar el ajuste."
            )
            req_ajustado = req
        else:
            _opciones_lote = {
                f"{_cli['nombre']} — {_l['identificador']} "
                f"({_l.get('categoria','')}, "
                f"{(_l.get('ultimo_peso_kg') or _l.get('peso_ingreso_kg')):.0f} kg)":
                (_l["id"], _l, _cli)
                for _l, _cli in _lotes_aptos
            }
            _sel_lote_dmi = st.selectbox(
                "Lote a usar para ajustar consumo:",
                options=["— No ajustar (usar DMI base) —"] +
                         list(_opciones_lote.keys()),
                key="dieta_lote_clima",
            )
            req_ajustado = req
            if _sel_lote_dmi and _sel_lote_dmi.startswith("— No"):
                pass  # mantener req base
            elif _sel_lote_dmi:
                _, _l_sel, _cli_sel = _opciones_lote[_sel_lote_dmi]
                _peso_lote = (_l_sel.get("ultimo_peso_kg")
                               or _l_sel.get("peso_ingreso_kg"))
                _cant_lote = _l_sel.get("cantidad_inicial") or 1
                with st.spinner("Consultando clima del lote…"):
                    try:
                        _clima_aj = obtener_clima(
                            _cli_sel["lat"], _cli_sel["lon"]
                        )
                    except Exception as _e_aj:
                        _clima_aj = None
                        st.warning(f"No pude obtener el clima: {_e_aj}")
                if _clima_aj:
                    from src.dmi import dmi_proyectado
                    _daily_aj = _clima_aj.get("daily", {}) or {}
                    _tmin_l = [
                        x for x in (_daily_aj.get(
                            "temperature_2m_min") or [])[:7]
                        if x is not None
                    ]
                    _tmax_l = [
                        x for x in (_daily_aj.get(
                            "temperature_2m_max") or [])[:7]
                        if x is not None
                    ]
                    _hr_l = [
                        x for x in (_daily_aj.get(
                            "relative_humidity_2m_max") or [])[:7]
                        if x is not None
                    ]
                    _viento_l = [
                        x for x in (_daily_aj.get(
                            "windspeed_10m_max") or [])[:7]
                        if x is not None
                    ]
                    _precip_l = (_daily_aj.get(
                        "precipitation_sum") or [])[:7]
                    _precip_3d_l = sum(
                        (x or 0) for x in _precip_l[:3]
                    )
                    _barro_l = _precip_3d_l > 20
                    _clima_dmi_l = {
                        "t_min": min(_tmin_l) if _tmin_l else None,
                        "t_max": max(_tmax_l) if _tmax_l else None,
                        "hr_max": max(_hr_l) if _hr_l else None,
                        "viento_max": max(_viento_l) if _viento_l else None,
                        "lluvia_3d": _precip_3d_l,
                        "lluvia_dia": max(_precip_l) if _precip_l else 0,
                    }
                    _dmi_aj_obj = dmi_proyectado(
                        peso_kg=_peso_lote,
                        categoria=_l_sel.get("categoria", ""),
                        raza=_l_sel.get("raza", ""),
                        clima_diario=_clima_dmi_l,
                        cantidad=_cant_lote,
                        dias_evento=1,
                        barro=_barro_l,
                    )
                    if _dmi_aj_obj:
                        # Tomar el punto medio del rango ajustado
                        _a_min, _a_max = _dmi_aj_obj[
                            "dmi_ajustado_rango_kg_dia"]
                        _dmi_nuevo = (_a_min + _a_max) / 2.0
                        _f_min, _f_max = _dmi_aj_obj[
                            "factor_ajuste_pct"]
                        _razon = (
                            f"Clima esperado en {_cli_sel.get('localidad','el campo')}: "
                            f"T° mín {_clima_dmi_l['t_min']:.0f}°C, "
                            f"HR máx {(_clima_dmi_l['hr_max'] or 0):.0f}%, "
                            f"viento máx {(_clima_dmi_l['viento_max'] or 0):.0f} km/h"
                            f"{', barro probable' if _barro_l else ''}. "
                            f"Factor neto: {_f_min:+.0f}% a {_f_max:+.0f}%."
                        )
                        req_ajustado = ajustar_req_por_dmi(
                            req, _dmi_nuevo, razon_ajuste=_razon,
                        )

                        # Comparativa antes/después
                        st.markdown("##### 📊 Comparativa antes/después")
                        _c1, _c2, _c3 = st.columns(3)
                        _delta_dmi = req_ajustado.consumo_ms_kg - req.consumo_ms_kg
                        _delta_dmi_pct = (
                            (_delta_dmi / req.consumo_ms_kg * 100)
                            if req.consumo_ms_kg > 0 else 0
                        )
                        _c1.metric(
                            "DMI (kg MS/día)",
                            f"{req_ajustado.consumo_ms_kg:.2f}",
                            f"{_delta_dmi:+.2f} ({_delta_dmi_pct:+.1f}%)",
                        )
                        _delta_pb = req_ajustado.pb_pct_ms - req.pb_pct_ms
                        _c2.metric(
                            "Densidad PB (% MS)",
                            f"{req_ajustado.pb_pct_ms:.1f}%",
                            f"{_delta_pb:+.1f} pp",
                        )
                        _delta_em = (
                            req_ajustado.em_concentracion_mcal_kg
                            - req.em_concentracion_mcal_kg
                        )
                        _c3.metric(
                            "Densidad EM (Mcal/kg MS)",
                            f"{req_ajustado.em_concentracion_mcal_kg:.2f}",
                            f"{_delta_em:+.2f}",
                        )
                        if _delta_dmi_pct < -1:
                            st.warning(
                                f"📉 **El clima reduce el consumo "
                                f"{abs(_delta_dmi_pct):.0f}%.** La dieta "
                                f"debe ser MÁS densa: subir % de "
                                f"concentrados y/o sumar palatables "
                                f"para que el animal reciba los mismos "
                                f"requerimientos en menos kg."
                            )
                        elif _delta_dmi_pct > 1:
                            st.success(
                                f"📈 **El clima sube el consumo "
                                f"{_delta_dmi_pct:.0f}%.** La dieta "
                                f"puede ser menos densa: hay margen "
                                f"para más rollo o fardo voluminoso "
                                f"sin perjudicar los aportes."
                            )
                        else:
                            st.info(
                                "El clima esperado no modifica "
                                "significativamente el consumo. La "
                                "dieta base aplica sin ajustes."
                            )
                        with st.expander(
                            "Ver factores climáticos que aplican",
                            expanded=False,
                        ):
                            for _r in _dmi_aj_obj.get("razones", []):
                                st.markdown(f"- {_r}")
    # Si el productor seleccionó un lote y se generó un req
    # ajustado, lo usamos en el resto del flujo (cálculo final +
    # optimizador). Si no, mantiene el req base.
    if 'req_ajustado' in dir() and req_ajustado is not None:
        req = req_ajustado

    st.divider()

    # ---- BLOQUE B: ingredientes disponibles ----
    st.markdown("#### 2️⃣ Ingredientes disponibles en el campo")
    st.caption(
        "Marcá cuáles tenés disponibles, editá los precios actuales y los "
        "porcentajes mínimo/máximo de inclusión en la dieta. "
        "El sistema solo usará los marcados como **disponibles**."
    )

    # Init UNA SOLA VEZ — después el editor mantiene su propio estado vía key
    if "ingredientes_df" not in st.session_state:
        st.session_state["ingredientes_df"] = pd.DataFrame(
            [asdict_ing(i) for i in ingredientes_default()]
        )

    edited = st.data_editor(
        st.session_state["ingredientes_df"],
        hide_index=True,
        width="stretch",
        height=420,
        key="dieta_ingredientes_editor",
        column_config={
            "nombre": st.column_config.TextColumn("Ingrediente", width="large"),
            "categoria": st.column_config.SelectboxColumn(
                "Categoría",
                options=["concentrado", "forraje", "suplemento", "mineral", "balanceado"],
            ),
            "ms_pct": st.column_config.NumberColumn("MS %", min_value=0, max_value=100, step=0.5, format="%.1f"),
            "pb_pct_ms": st.column_config.NumberColumn("PB %MS", min_value=0, max_value=300, step=0.5, format="%.1f"),
            "em_mcal_kg_ms": st.column_config.NumberColumn("EM Mcal/kgMS", min_value=0, max_value=5, step=0.05, format="%.2f"),
            "fdn_pct_ms": st.column_config.NumberColumn("FDN %MS", min_value=0, max_value=100, step=1, format="%.0f"),
            "ca_pct_ms": st.column_config.NumberColumn("Ca %MS", min_value=0, max_value=50, step=0.05, format="%.2f"),
            "p_pct_ms": st.column_config.NumberColumn("P %MS", min_value=0, max_value=15, step=0.05, format="%.2f"),
            "precio_kg_tal_cual": st.column_config.NumberColumn("$/kg t/c", min_value=0, step=10, format="%.0f"),
            "min_inclusion_pct_ms": st.column_config.NumberColumn(
                "Mín %", min_value=0, max_value=100, step=1, format="%.0f",
                help="Inclusión mínima obligatoria en la dieta",
            ),
            "max_inclusion_pct_ms": st.column_config.NumberColumn(
                "Máx %", min_value=0, max_value=100, step=1, format="%.0f",
                help="Inclusión máxima permitida (ej. Fibrogreen 20% por monensina)",
            ),
            "disponible": st.column_config.CheckboxColumn(
                "Disponible",
                help="Tildá solo los que tenés en el campo",
            ),
        },
        num_rows="dynamic",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    if col_btn1.button("🔄 Restaurar default"):
        st.session_state["ingredientes_df"] = pd.DataFrame(
            [asdict_ing(i) for i in ingredientes_default()]
        )
        # Limpiar el state del editor para forzar re-init
        if "dieta_ingredientes_editor" in st.session_state:
            del st.session_state["dieta_ingredientes_editor"]
        st.rerun()
    if col_btn2.button("✅ Tildar todos"):
        df_actual = edited.copy()
        df_actual["disponible"] = True
        st.session_state["ingredientes_df"] = df_actual
        if "dieta_ingredientes_editor" in st.session_state:
            del st.session_state["dieta_ingredientes_editor"]
        st.rerun()

    st.divider()

    # ---- BLOQUE B-bis: VERIFICAR mi receta ----
    with st.expander("🧪 Verificar mi receta (sin optimizar)", expanded=False):
        st.caption(
            "Cargá los porcentajes de tu receta actual y la app te dice si "
            "cumple los requerimientos NASEM. Útil para validar mezclas que "
            "ya usás (ej. 88% maíz + 12% Fibrogreen)."
        )

        col_pre1, col_pre2, col_pre3 = st.columns(3)
        if col_pre1.button("📋 Plantilla: 88% maíz + 12% Fibrogreen"):
            st.session_state["receta_porcentajes"] = {
                "Maíz grano": 88.0,
                "Fibrogreen (núcleo + monensina)": 12.0,
            }
        if col_pre2.button("📋 Plantilla: 70% maíz + 20% silaje + 10% pellet soja"):
            st.session_state["receta_porcentajes"] = {
                "Maíz grano": 70.0,
                "Silaje de maíz (planta entera)": 20.0,
                "Pellet de soja (44% PB)": 10.0,
            }
        if col_pre3.button("🗑️ Limpiar receta"):
            st.session_state["receta_porcentajes"] = {}

        receta = st.session_state.get("receta_porcentajes", {})

        # Editor de receta: dataframe con nombre + porcentaje
        nombres_disponibles = sorted(set(
            r.get("nombre", "") for r in edited.to_dict("records")
            if r.get("nombre")
        ))

        if not receta:
            receta_df = pd.DataFrame([{"Ingrediente": "", "% mezcla": 0.0}])
        else:
            receta_df = pd.DataFrame([
                {"Ingrediente": k, "% mezcla": v} for k, v in receta.items()
            ])

        receta_edit = st.data_editor(
            receta_df,
            hide_index=True,
            width="stretch",
            num_rows="dynamic",
            key="receta_editor",
            column_config={
                "Ingrediente": st.column_config.SelectboxColumn(
                    "Ingrediente", options=nombres_disponibles, width="large",
                ),
                "% mezcla": st.column_config.NumberColumn(
                    "% mezcla", min_value=0, max_value=100, step=0.5, format="%.1f",
                ),
            },
        )

        suma = sum(_safe_float(r.get("% mezcla", 0)) for r in receta_edit.to_dict("records"))
        col_s1, col_s2 = st.columns([1, 3])
        col_s1.metric("Suma %", f"{suma:.1f}%",
                      "✅" if abs(suma - 100) < 0.5 else "❌ debe sumar 100")

        if st.button("🧪 Verificar receta vs requerimientos"):
            ingredientes_para_check = []
            for row in edited.to_dict("records"):
                nombre = row.get("nombre")
                if not nombre:
                    continue
                campos = {
                    "nombre": str(nombre),
                    "categoria": row.get("categoria") or "concentrado",
                    "ms_pct": _safe_float(row.get("ms_pct"), 88.0) or 88.0,
                    "pb_pct_ms": _safe_float(row.get("pb_pct_ms"), 0.0),
                    "em_mcal_kg_ms": _safe_float(row.get("em_mcal_kg_ms"), 0.0),
                    "fdn_pct_ms": _safe_float(row.get("fdn_pct_ms"), 0.0),
                    "ca_pct_ms": _safe_float(row.get("ca_pct_ms"), 0.0),
                    "p_pct_ms": _safe_float(row.get("p_pct_ms"), 0.0),
                    "precio_kg_tal_cual": _safe_float(row.get("precio_kg_tal_cual"), 0.0),
                    "min_inclusion_pct_ms": _safe_float(row.get("min_inclusion_pct_ms"), 0.0),
                    "max_inclusion_pct_ms": _safe_float(row.get("max_inclusion_pct_ms"), 100.0) or 100.0,
                    "disponible": True,
                }
                try:
                    ingredientes_para_check.append(Ingrediente(**campos))
                except Exception:
                    pass

            porcentajes = {
                r["Ingrediente"]: _safe_float(r["% mezcla"])
                for r in receta_edit.to_dict("records")
                if r.get("Ingrediente") and _safe_float(r.get("% mezcla", 0)) > 0
            }

            if not porcentajes:
                st.error("Agregá al menos un ingrediente con porcentaje >0.")
            else:
                resultado = verificar_receta(
                    ingredientes_para_check, porcentajes,
                    consumo_ms_kg=req.consumo_ms_kg,
                    pb_g_dia=req.pb_gramos, em_mcal_dia=req.em_mcal,
                    fdn_min_pct=req.fdn_min_pct,
                    ca_g_dia=req.calcio_g, p_g_dia=req.fosforo_g,
                    # Rango práctico Pordomingo/Latimori para tolerar
                    # variaciones aceptadas en Argentina:
                    pb_rango_pct=(req.pb_pct_min, req.pb_pct_max),
                )
                st.session_state["last_verificacion"] = resultado

        if "last_verificacion" in st.session_state:
            r = st.session_state["last_verificacion"]
            if r.factible:
                st.success(r.mensaje)
            else:
                st.warning(r.mensaje)

            # Mostrar advertencias de seguridad (NNP, etc.) primero — son críticas
            if r.advertencias:
                for adv in r.advertencias:
                    if adv.startswith("🔴"):
                        st.error(adv)
                    elif adv.startswith("⚠️"):
                        st.warning(adv)
                    else:
                        st.info(adv)

            v1, v2, v3 = st.columns(3)
            v1.metric(
                "PB total", f"{r.pb_aportado_g:.0f} g",
                f"req {r.pb_requerido_g:.0f} · "
                f"{(r.pb_aportado_g/r.pb_requerido_g-1)*100:+.0f}%" if r.pb_requerido_g else "",
            )
            v2.metric(
                "Energía Metab.", f"{r.em_aportado_mcal:.1f} Mcal",
                f"req {r.em_requerido_mcal:.1f} · "
                f"{(r.em_aportado_mcal/r.em_requerido_mcal-1)*100:+.0f}%" if r.em_requerido_mcal else "",
            )
            v3.metric("FDN", f"{r.fdn_aportado_pct:.1f}% MS",
                       f"mín {req.fdn_min_pct:.0f}%")
            v4, v5, v6 = st.columns(3)
            v4.metric("Calcio", f"{r.ca_aportado_g:.0f} g", f"req {req.calcio_g:.0f}")
            v5.metric("Fósforo", f"{r.p_aportado_g:.0f} g", f"req {req.fosforo_g:.0f}")
            v6.metric("💰 Costo/animal/día", f"${r.costo_total_dia:.2f}")

            # NNP: información detallada
            n1, n2, n3 = st.columns(3)
            nnp_color = "🟢" if r.nnp_aportado_pct < 0.7 else "🟡" if r.nnp_aportado_pct < 1.0 else "🔴"
            n1.metric(f"NNP en dieta {nnp_color}", f"{r.nnp_aportado_pct:.2f}% MS",
                       "Límite seguro: <1%")
            n2.metric("PB del NNP (urea)", f"{r.nnp_pb_equivalente_g:.0f} g",
                       "× 2.87 PB equivalente")
            n3.metric("PB verdadera", f"{r.pb_verdadera_g:.0f} g",
                       "Total - NNP equiv.")

            if r.deficiencias:
                st.markdown("##### ⚠️ Deficiencias detectadas")
                for d in r.deficiencias:
                    st.warning(
                        f"**{d['nutriente']}**: "
                        f"aportado {d['max_alcanzable']:.0f}, "
                        f"requerido {d['requerido']:.0f} "
                        f"(faltan {d['deficit_pct']:.0f}%)"
                    )
            if r.sugerencias:
                st.markdown("##### 💡 Cómo corregir")
                for s in r.sugerencias:
                    st.info(s)

            # ----- Guardar dieta al historial -----
            st.divider()
            st.markdown("##### 💾 Guardar al historial del lote")
            lotes_guardar = db.listar_lotes(estado="activo")
            if not lotes_guardar:
                st.info(
                    "Cargá un lote en **🏢 Clientes y Lotes** para asociar "
                    "esta dieta al historial."
                )
            else:
                col_d1, col_d2, col_d3 = st.columns([2, 1, 1])
                with col_d1:
                    lote_dieta_id = st.selectbox(
                        "Lote",
                        [l["id"] for l in lotes_guardar],
                        format_func=lambda x: next(
                            f"{l['cliente_nombre']} — {l['identificador']}"
                            for l in lotes_guardar if l["id"] == x
                        ),
                        key="lote_para_dieta_check",
                    )
                with col_d2:
                    fecha_dieta = st.date_input(
                        "Fecha", value=datetime.now().date(),
                        key="fecha_dieta_check",
                    )
                with col_d3:
                    obs_dieta = st.text_input(
                        "Obs.", placeholder="opcional", key="obs_dieta_check",
                    )

                if st.button("💾 Guardar dieta verificada al historial",
                              type="primary", key="btn_guardar_dieta_check"):
                    try:
                        db.guardar_dieta(
                            lote_id=lote_dieta_id,
                            fecha=fecha_dieta.isoformat(),
                            composicion=r.composicion,
                            costo_dia=r.costo_total_dia,
                            pb_pct=(r.pb_aportado_g / r.consumo_ms_kg / 10
                                     if r.consumo_ms_kg else 0),
                            em_mcal_dia=r.em_aportado_mcal,
                            consumo_ms_kg=r.consumo_ms_kg,
                            nnp_pct=r.nnp_aportado_pct,
                            observaciones=obs_dieta,
                        )
                        st.success(
                            "✅ Dieta guardada en el historial del lote. "
                            "La vas a ver en la pestaña **📚 Historial**."
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.divider()

    # ---- BLOQUE C: optimizar ----
    st.markdown("#### 3️⃣ Formular mezcla de mínimo costo")
    if st.button("⚡ Calcular dieta de mínimo costo", type="primary"):
        # Leer del editor en vivo (no de session_state, que puede estar
        # desfasado un rerun). Sanitizar: descartar filas vacías y reemplazar
        # None/NaN por defaults antes de armar Ingrediente()
        ingredientes = []
        rows_iter = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
        for row in rows_iter:
            nombre = row.get("nombre")
            if not nombre or (isinstance(nombre, float) and pd.isna(nombre)):
                continue   # fila vacía agregada por error
            campos = {
                "nombre": str(nombre),
                "categoria": row.get("categoria") or "concentrado",
                "ms_pct": _safe_float(row.get("ms_pct"), default=88.0),
                "pb_pct_ms": _safe_float(row.get("pb_pct_ms"), default=0.0),
                "em_mcal_kg_ms": _safe_float(row.get("em_mcal_kg_ms"), default=0.0),
                "fdn_pct_ms": _safe_float(row.get("fdn_pct_ms"), default=0.0),
                "ca_pct_ms": _safe_float(row.get("ca_pct_ms"), default=0.0),
                "p_pct_ms": _safe_float(row.get("p_pct_ms"), default=0.0),
                "precio_kg_tal_cual": _safe_float(row.get("precio_kg_tal_cual"), default=0.0),
                "min_inclusion_pct_ms": _safe_float(row.get("min_inclusion_pct_ms"), default=0.0),
                "max_inclusion_pct_ms": _safe_float(row.get("max_inclusion_pct_ms"), default=100.0),
                "disponible": bool(row.get("disponible")) if row.get("disponible") is not None else False,
            }
            try:
                ingredientes.append(Ingrediente(**campos))
            except (TypeError, ValueError) as e:
                st.warning(f"Ignoro fila con datos inválidos: {nombre} ({e})")

        if not any(i.disponible for i in ingredientes):
            st.error("⚠️ No marcaste ningún ingrediente como **disponible**. Tildá la casilla 'Disponible' de los que tengas en el campo.")
        else:
            try:
                resultado = formular_minimo_costo(
                    ingredientes,
                    consumo_ms_kg=req.consumo_ms_kg,
                    pb_g_dia=req.pb_gramos,
                    em_mcal_dia=req.em_mcal,
                    fdn_min_pct=req.fdn_min_pct,
                    ca_g_dia=req.calcio_g,
                    p_g_dia=req.fosforo_g,
                )
                st.session_state["last_formulacion"] = resultado
            except Exception as e:
                st.error(f"Error en la optimización: {e}")
                st.exception(e)

    if "last_formulacion" in st.session_state:
        r = st.session_state["last_formulacion"]

        if r.factible:
            st.success(r.mensaje)
            f1, f2, f3 = st.columns(3)
            f1.metric("💰 Costo / animal / día", f"${r.costo_total_dia:,.2f}")
            f2.metric("Costo por kg MS", f"${r.costo_por_kg_ms:,.2f}/kg")
            f3.metric("Total tal cual / día", f"{r.consumo_tal_cual_kg:.2f} kg")

            st.markdown("##### 📋 Mezcla óptima")
            df_mezcla = pd.DataFrame([
                {
                    "Ingrediente": c["nombre"],
                    "Categoría": c["categoria"],
                    "% MS dieta": round(c["pct_ms"], 1),
                    "kg MS/día": round(c["kg_ms"], 2),
                    "kg tal cual/día": round(c["kg_tal_cual"], 2),
                    "Costo $/día": round(c["costo_dia"], 2),
                }
                for c in r.composicion
            ])
            st.dataframe(df_mezcla, hide_index=True, width="stretch")

            st.markdown("##### ✅ Verificación de aportes vs requerimientos")
            v1, v2, v3 = st.columns(3)
            v1.metric(
                "PB", f"{r.pb_aportado_g:.0f} g",
                f"req {r.pb_requerido_g:.0f} g · "
                f"{(r.pb_aportado_g/r.pb_requerido_g-1)*100:+.1f}%",
            )
            v2.metric(
                "Energía Metab.", f"{r.em_aportado_mcal:.1f} Mcal",
                f"req {r.em_requerido_mcal:.1f} · "
                f"{(r.em_aportado_mcal/r.em_requerido_mcal-1)*100:+.1f}%",
            )
            v3.metric("FDN", f"{r.fdn_aportado_pct:.1f}% MS",
                       f"mín {req.fdn_min_pct:.0f}%")
            v4, v5 = st.columns(2)
            v4.metric("Calcio", f"{r.ca_aportado_g:.0f} g",
                       f"req {req.calcio_g:.0f} g")
            v5.metric("Fósforo", f"{r.p_aportado_g:.0f} g",
                       f"req {req.fosforo_g:.0f} g")

            # Costo por lote
            n_lote = st.session_state.get("vid_n", 0)
            if n_lote:
                st.info(
                    f"💰 Para tu lote de {n_lote} animales: "
                    f"**${r.costo_total_dia * n_lote:,.0f}/día** = "
                    f"**${r.costo_total_dia * n_lote * 30:,.0f}/mes**"
                )

            # Descarga CSV de la mezcla
            csv_mezcla = df_mezcla.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar mezcla (CSV)",
                data=csv_mezcla,
                file_name=f"mezcla_minimocosto_{cat_d}_{peso_d:.0f}kg.csv",
                mime="text/csv",
                key="dl_mezcla",
            )

            # ----- Guardar dieta optimizada al historial -----
            st.divider()
            st.markdown("##### 💾 Guardar al historial del lote")
            lotes_save_opt = db.listar_lotes(estado="activo")
            if not lotes_save_opt:
                st.info(
                    "Cargá un lote en **🏢 Clientes y Lotes** para asociar "
                    "esta dieta al historial."
                )
            else:
                col_o1, col_o2, col_o3 = st.columns([2, 1, 1])
                with col_o1:
                    lote_opt_id = st.selectbox(
                        "Lote",
                        [l["id"] for l in lotes_save_opt],
                        format_func=lambda x: next(
                            f"{l['cliente_nombre']} — {l['identificador']}"
                            for l in lotes_save_opt if l["id"] == x
                        ),
                        key="lote_para_dieta_opt",
                    )
                with col_o2:
                    fecha_opt = st.date_input(
                        "Fecha", value=datetime.now().date(),
                        key="fecha_dieta_opt",
                    )
                with col_o3:
                    obs_opt = st.text_input(
                        "Obs.", placeholder="opcional", key="obs_dieta_opt",
                    )

                if st.button("💾 Guardar mezcla óptima al historial",
                              type="primary", key="btn_guardar_dieta_opt"):
                    try:
                        db.guardar_dieta(
                            lote_id=lote_opt_id,
                            fecha=fecha_opt.isoformat(),
                            composicion=r.composicion,
                            costo_dia=r.costo_total_dia,
                            pb_pct=(r.pb_aportado_g / r.consumo_ms_kg / 10
                                     if r.consumo_ms_kg else 0),
                            em_mcal_dia=r.em_aportado_mcal,
                            consumo_ms_kg=r.consumo_ms_kg,
                            nnp_pct=r.nnp_aportado_pct,
                            observaciones=f"[Mínimo costo] {obs_opt}".strip(),
                        )
                        st.success(
                            "✅ Mezcla óptima guardada en el historial. "
                            "Vela en **📚 Historial**."
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.error(r.mensaje)
            if r.deficiencias:
                st.markdown("##### ⚠️ Nutrientes que NO se pueden cubrir")
                for d in r.deficiencias:
                    st.warning(
                        f"**{d['nutriente']}**: requerido {d['requerido']:.0f}, "
                        f"máximo alcanzable con tus ingredientes "
                        f"{d['max_alcanzable']:.0f} "
                        f"(déficit {d['deficit_pct']:.0f}%)"
                    )
            if r.sugerencias:
                st.markdown("##### 💡 Ingredientes sugeridos para corregir")
                for s in r.sugerencias:
                    st.info(s)


# ----------------------------- HISTORIAL ------------------------------
with tab_historial:
    st.markdown("### 📚 Historial completo del lote")
    st.info(
        "El historial completo de cada lote (peso, dietas, "
        "movimientos, cargas del comedero y gráfico comparativo) "
        "ahora vive dentro de la ficha del lote.\n\n"
        "👉 Andá a **🏢 Clientes/Lotes** → seleccioná el lote → "
        "scroleá hasta el final."
    )
    st.caption(
        "Esta pestaña queda disponible para futuras vistas "
        "comparativas entre lotes."
    )
# ----------------------------- ASESOR IA ------------------------------
with tab_ia:
    st.markdown(
        "### 🤖 Asesor Nutricional IA\n"
        "Agente formulador de raciones con conocimiento de NASEM 2016 + práctica "
        "argentina (Pordomingo, Latimori, IPCVA). Conoce **automáticamente** "
        "los datos del lote actual + histórico de la base de datos."
    )

    api_key = st.session_state.get("anthropic_api_key", "")
    if not api_key:
        st.warning(
            "⚠️ Falta la **Claude API Key**. Cargala en la sidebar para "
            "activar el asesor.\n\n"
            "**Cómo obtenerla**: entrá a https://console.anthropic.com → "
            "API Keys → Create Key. Tiene tier gratuito; cada respuesta "
            "cuesta centavos."
        )
        st.stop()

    # ====== Memoria del agente (expander) ======
    with st.expander("🧠 Memoria del agente — qué aprendió de vos", expanded=False):
        st.caption(
            "El agente Claude no se reentrena automáticamente, pero sí podés "
            "enseñarle cosas que recordará entre sesiones. Lo que cargues acá "
            "se inyecta automáticamente en cada conversación nueva."
        )

        col_mem_form, col_mem_list = st.columns([1, 2])
        with col_mem_form:
            with st.form("nueva_memoria", clear_on_submit=True):
                m_categoria = st.selectbox(
                    "Categoría",
                    ["correccion", "valor_local", "ingrediente",
                     "manejo", "cliente", "preferencia", "general"],
                    format_func=lambda x: {
                        "correccion": "🔧 Corrección técnica",
                        "valor_local": "📍 Valor típico de la zona",
                        "ingrediente": "🌾 Composición de ingrediente",
                        "manejo": "⚙️ Característica de manejo",
                        "cliente": "👤 Info de cliente",
                        "preferencia": "❤️ Preferencia personal",
                        "general": "📝 Otra",
                    }[x],
                )
                m_texto = st.text_area(
                    "Texto a recordar", height=100,
                    placeholder="Ej: 'En La Pampa el silaje de maíz típico "
                                "tiene 28% MS y 7% PB'",
                )
                m_etiqueta = st.text_input("Etiqueta (opcional)")
                if st.form_submit_button("💾 Guardar memoria", type="primary"):
                    if m_texto:
                        memoria.agregar_memoria(m_texto, m_categoria, m_etiqueta)
                        st.success("✅ Guardada")
                        st.rerun()

        with col_mem_list:
            todas = memoria.listar_memorias(activas_solo=False)
            if not todas:
                st.info("Sin memorias. Cargá la primera con el formulario.")
            else:
                for m in sorted(todas, key=lambda x: x["fecha"], reverse=True):
                    estado = "✅" if m.get("activa", True) else "⏸️"
                    cat_emoji = {
                        "correccion": "🔧", "valor_local": "📍",
                        "ingrediente": "🌾", "manejo": "⚙️",
                        "cliente": "👤", "preferencia": "❤️",
                        "general": "📝",
                    }.get(m["categoria"], "📝")
                    cols_m = st.columns([8, 1, 1])
                    cols_m[0].write(f"{estado} {cat_emoji} {m['texto']}")
                    if cols_m[1].button("⏸️" if m.get("activa", True) else "▶️",
                                         key=f"tog_{m['id']}"):
                        if m.get("activa", True):
                            memoria.desactivar_memoria(m["id"])
                        else:
                            memoria.reactivar_memoria(m["id"])
                        st.rerun()
                    if cols_m[2].button("🗑️", key=f"del_{m['id']}"):
                        memoria.eliminar_memoria(m["id"])
                        st.rerun()

    st.divider()

    # Selector de lote para enriquecer el contexto del agente
    lotes_para_ia = db.listar_lotes()
    if lotes_para_ia:
        lote_para_ia_id = st.selectbox(
            "🔗 Lote contextual (opcional, la IA accede a su histórico completo)",
            [None] + [l["id"] for l in lotes_para_ia],
            format_func=lambda x: "—" if x is None else
                next(f"{l['cliente_nombre']} — {l['identificador']}"
                     for l in lotes_para_ia if l["id"] == x),
            key="ia_lote_ctx",
        )

        # Panel de clima si el cliente tiene localidad cargada
        if lote_para_ia_id:
            lote_ctx = db.obtener_lote(lote_para_ia_id)
            if lote_ctx:
                cli_ctx = db.obtener_cliente(lote_ctx["cliente_id"])
                if cli_ctx and cli_ctx.get("localidad"):
                    with st.expander(
                        f"🌦️ Clima en {cli_ctx['localidad']} (auto-cargado al chat)",
                        expanded=False,
                    ):
                        try:
                            # Usar coordenadas manuales si las hay
                            if cli_ctx.get("lat") and cli_ctx.get("lon"):
                                from src.clima import geocodificar_manual
                                geo = geocodificar_manual(
                                    float(cli_ctx["lat"]),
                                    float(cli_ctx["lon"]),
                                    cli_ctx["localidad"],
                                )
                            else:
                                geo = geocodificar(cli_ctx["localidad"])
                            if geo:
                                clima = obtener_clima(geo["lat"], geo["lon"])
                                if clima and clima.get("current"):
                                    actual = clima["current"]
                                    t = actual.get("temperature_2m") or 0
                                    hr = actual.get("relative_humidity_2m") or 0
                                    viento = actual.get("wind_speed_10m") or 0
                                    thi = calcular_thi(t, hr)
                                    cm1, cm2, cm3, cm4 = st.columns(4)
                                    cm1.metric("Temperatura", f"{t:.0f}°C")
                                    cm2.metric("Humedad", f"{hr:.0f}%")
                                    cm3.metric("Viento", f"{viento:.0f} km/h")
                                    cm4.metric("THI", f"{thi:.0f}",
                                               clasificar_thi(thi))
                                    st.caption(
                                        f"📍 {geo['nombre']}, {geo['admin1']} · "
                                        f"Datos: Open-Meteo (gratis, sin API key)"
                                    )

                                    # ---- Alertas predictivas ----
                                    alertas = generar_alertas_predictivas(
                                        clima,
                                        categoria=lote_ctx.get("categoria", ""),
                                    )
                                    # Panel especial: si está en La Pampa, link
                                    # al sistema oficial + carga manual
                                    if -39 <= geo["lat"] <= -34 and -67 <= geo["lon"] <= -62:
                                        from src.clima_lapampa import (
                                            estacion_mas_cercana as _est_cerc,
                                        )
                                        est_cerc = _est_cerc(geo["lat"], geo["lon"])
                                        st.divider()
                                        st.markdown(
                                            f"##### 📡 Estación oficial: **{est_cerc['nombre']}** "
                                            f"(La Pampa)"
                                        )
                                        st.markdown(
                                            f"🔗 **[Abrir Redes Climáticas — La Pampa]"
                                            f"({URL_LA_PAMPA})** "
                                            f"(login con Google) — datos en tiempo real "
                                            f"de la estación más cercana al campo."
                                        )

                                        # Mostrar datos manuales si los hay
                                        manuales = obtener_datos_manuales(
                                            est_cerc["nombre"], ttl_horas=24,
                                        )
                                        if manuales:
                                            st.success(
                                                f"✅ Datos cargados manualmente "
                                                f"hace {(datetime.now() - datetime.fromisoformat(manuales['fecha_consulta'])).total_seconds() / 3600:.1f} hs"
                                            )
                                            cm_t = manuales.get("temperatura_c")
                                            cm_h = manuales.get("humedad_pct")
                                            cm_l24 = manuales.get("precipitacion_mm_24h")
                                            cm_l7 = manuales.get("precipitacion_mm_7d")
                                            cols_m = st.columns(4)
                                            if cm_t is not None:
                                                cols_m[0].metric("T° estación",
                                                                  f"{float(cm_t):.1f}°C")
                                            if cm_h is not None:
                                                cols_m[1].metric("HR estación",
                                                                  f"{float(cm_h):.0f}%")
                                            if cm_l24 is not None:
                                                cols_m[2].metric("Lluvia 24h",
                                                                  f"{float(cm_l24):.1f} mm")
                                            if cm_l7 is not None:
                                                cols_m[3].metric("Lluvia 7d",
                                                                  f"{float(cm_l7):.1f} mm")

                                        with st.expander(
                                            "📋 Cargar datos manualmente desde Redes Climáticas",
                                            expanded=False,
                                        ):
                                            st.caption(
                                                "Pegá los valores que ves en el sitio oficial. "
                                                "Quedan disponibles 24 hs y los usa el agente IA."
                                            )
                                            with st.form(f"form_lapampa_{lote_para_ia_id}"):
                                                col_lp1, col_lp2 = st.columns(2)
                                                with col_lp1:
                                                    lp_t = st.number_input(
                                                        "Temperatura (°C)",
                                                        min_value=-30.0, max_value=55.0,
                                                        value=0.0, step=0.1,
                                                    )
                                                    lp_h = st.number_input(
                                                        "Humedad (%)",
                                                        min_value=0.0, max_value=100.0,
                                                        value=0.0, step=1.0,
                                                    )
                                                    lp_st = st.number_input(
                                                        "Sensación térmica (°C)",
                                                        min_value=-30.0, max_value=55.0,
                                                        value=0.0, step=0.1,
                                                    )
                                                    lp_v = st.number_input(
                                                        "Viento (km/h)",
                                                        min_value=0.0, max_value=200.0,
                                                        value=0.0, step=1.0,
                                                    )
                                                with col_lp2:
                                                    lp_l24 = st.number_input(
                                                        "Lluvia 24h (mm)",
                                                        min_value=0.0, max_value=500.0,
                                                        value=0.0, step=0.1,
                                                    )
                                                    lp_l7 = st.number_input(
                                                        "Lluvia 7 días (mm)",
                                                        min_value=0.0, max_value=1000.0,
                                                        value=0.0, step=0.5,
                                                    )
                                                    lp_tmax = st.number_input(
                                                        "T° máx 24h (°C)",
                                                        min_value=-30.0, max_value=55.0,
                                                        value=0.0, step=0.1,
                                                    )
                                                    lp_tmin = st.number_input(
                                                        "T° mín 24h (°C)",
                                                        min_value=-30.0, max_value=55.0,
                                                        value=0.0, step=0.1,
                                                    )

                                                if st.form_submit_button(
                                                    "💾 Guardar datos manuales",
                                                    type="primary",
                                                ):
                                                    datos = {}
                                                    if lp_t != 0:
                                                        datos["temperatura_c"] = lp_t
                                                    if lp_h != 0:
                                                        datos["humedad_pct"] = lp_h
                                                    if lp_st != 0:
                                                        datos["sensacion_termica_c"] = lp_st
                                                    if lp_v != 0:
                                                        datos["viento_kmh"] = lp_v
                                                    if lp_l24 != 0:
                                                        datos["precipitacion_mm_24h"] = lp_l24
                                                    if lp_l7 != 0:
                                                        datos["precipitacion_mm_7d"] = lp_l7
                                                    if lp_tmax != 0:
                                                        datos["temp_max_24h"] = lp_tmax
                                                    if lp_tmin != 0:
                                                        datos["temp_min_24h"] = lp_tmin

                                                    if datos:
                                                        guardar_datos_manuales(
                                                            est_cerc["nombre"], datos,
                                                        )
                                                        st.success(
                                                            f"✅ Guardados {len(datos)} valores. "
                                                            "El agente los usa en el próximo chat."
                                                        )
                                                        st.rerun()
                                                    else:
                                                        st.warning("Cargá al menos un valor.")

                                    if alertas:
                                        st.divider()
                                        st.markdown("##### 🚨 Alertas predictivas (próximos 7 días)")
                                        for a in alertas:
                                            sev = a.get("severidad", "info")
                                            box_func = (
                                                st.error if sev == "critica"
                                                else st.warning if sev == "warning"
                                                else st.info
                                            )
                                            with box_func(f"{a['icono']} **{a['titulo']}**"):
                                                pass
                                            with st.container():
                                                st.markdown(
                                                    f"**Cuándo**: {a['cuando']}\n\n"
                                                    f"**Qué se espera**: {a['descripcion']}\n\n"
                                                    f"**Impacto**: {a['impacto']}\n\n"
                                                    f"**Acciones recomendadas:**"
                                                )
                                                for acc in a["acciones"]:
                                                    st.markdown(f"- {acc}")
                                                st.divider()
                                    else:
                                        st.success("✅ Sin alertas climáticas críticas en los próximos 7 días")
                                else:
                                    st.info("Sin datos climáticos disponibles ahora.")
                            else:
                                st.warning(
                                    f"No pude geocodificar '{cli_ctx['localidad']}'. "
                                    "Probá con un nombre más específico."
                                )
                        except Exception as e:
                            st.warning(f"Clima no disponible: {e}")
    else:
        lote_para_ia_id = None

    # Inicializar historial
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Sugerencias de inicio rápido (basadas en datos del lote si existen)
    has_lot_data = bool(st.session_state.get("vid_n"))

    # Modelos disponibles. Sonnet 4.5 para razonamiento profundo
    # (formular dietas, análisis NRC). Haiku 4.5 para tareas simples
    # y consultas rápidas (3x más rápido, ~5x más barato).
    _MODELOS_IA = {
        "🦅 Sonnet 4.5 (potente, lento)":
            "claude-sonnet-4-5-20250929",
        "⚡ Haiku 4.5 (rápido y barato)":
            "claude-haiku-4-5-20251001",
    }
    col_a, col_mod, col_b = st.columns([2, 2, 1])
    with col_mod:
        _modelo_label = st.selectbox(
            "Modelo del agente",
            list(_MODELOS_IA.keys()),
            index=0,
            key="chat_modelo_sel",
            help=(
                "Sonnet 4.5: para formular dietas, análisis técnico "
                "completo, planes de adaptación. Más lento, ~5x más "
                "caro.\n\n"
                "Haiku 4.5: para correcciones puntuales, conversiones, "
                "preguntas de referencia. ~3x más rápido, ~5x más "
                "barato. Para tareas críticas mejor Sonnet."
            ),
        )
        _modelo_ia = _MODELOS_IA[_modelo_label]
    with col_b:
        st.markdown(
            "<div style='height:28px;'></div>",
            unsafe_allow_html=True,
        )  # alinear con el selectbox
        if st.button("🗑️ Nuevo chat", width="stretch"):
            st.session_state["chat_messages"] = []
            st.rerun()

    if not st.session_state["chat_messages"]:
        st.markdown("**💡 Sugerencias para empezar:**")
        sug_cols = st.columns(2)
        suggestions = []

        if has_lot_data:
            suggestions = [
                ("📊 Explicame los resultados de mi lote para charlarlos con el productor",
                 "Explicame los resultados del análisis de mi lote en lenguaje simple, "
                 "como para charlarlo con el dueño del campo. Dame cuántos están parejos, "
                 "cuáles necesitan revisión y qué próximos pasos recomendás."),
                ("🍽️ Recomendame una dieta para este lote",
                 "Necesito recomendación de ración para mi lote actual. Tengo disponible "
                 "maíz, silaje de maíz, fibrogreen, núcleo mineral y sal. Pediime los "
                 "datos que falten."),
                ("⚠️ ¿Por qué pueden estar bajos en peso algunos animales?",
                 "Algunos animales del lote están notablemente más livianos que el "
                 "promedio. ¿Cuáles son las causas más probables y qué chequeos recomendás?"),
                ("📅 Proyectame a faena",
                 "Con el peso actual de mi lote, ¿cuántos días faltan para faena (objetivo "
                 "390 kg) si mantengo el ADG actual? ¿Qué dieta recomendás?"),
            ]
        else:
            suggestions = [
                ("🍽️ Formular dieta de recría",
                 "Quiero armar una dieta de recría. Pediime los datos que necesites."),
                ("📚 ¿Cuánto debe ser el % de PB para terminación?",
                 "¿Qué porcentaje de proteína bruta es óptimo para una dieta de "
                 "terminación según NASEM 2016 y según la práctica argentina?"),
                ("🌡️ Manejo en estrés calórico",
                 "¿Cómo ajusto la dieta cuando hay estrés calórico (>30°C) en feedlot?"),
                ("⚙️ Plan de adaptación a grano",
                 "¿Cómo armo un plan de adaptación de 10 días para entrar terneros "
                 "de pasto a corral con dieta de grano alta?"),
            ]

        for i, (label, prompt) in enumerate(suggestions):
            with sug_cols[i % 2]:
                if st.button(label, key=f"sug_{i}", width="stretch"):
                    st.session_state["chat_messages"].append(
                        {"role": "user", "content": prompt}
                    )
                    st.rerun()

    # Contenedor del chat (mensajes anteriores + nueva respuesta si toca generar)
    chat_container = st.container()
    with chat_container:
        # 1) Render del historial existente con botones de acción
        for i, msg in enumerate(st.session_state["chat_messages"]):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Para respuestas del agente: botones "Recordar" + "Generar PDF"
                if msg["role"] == "assistant":
                    cols_btn = st.columns([5, 1, 1])
                    with cols_btn[1]:
                        if st.button(
                            "💾 Recordar", key=f"rem_btn_{i}",
                            help="Guardar como memoria del agente",
                            width="stretch",
                        ):
                            st.session_state["recordando_msg_idx"] = i
                    with cols_btn[2]:
                        if st.button(
                            "📄 PDF", key=f"pdf_btn_{i}",
                            help="Convertir esta respuesta en informe PDF "
                                 "con marca HMS",
                            width="stretch",
                        ):
                            st.session_state["pdf_chat_idx"] = i
                else:
                    # Mensajes del usuario: solo botón Recordar
                    cols_btn = st.columns([6, 1])
                    with cols_btn[1]:
                        if st.button(
                            "💾 Recordar", key=f"rem_btn_{i}",
                            help="Guardar como memoria del agente",
                            width="stretch",
                        ):
                            st.session_state["recordando_msg_idx"] = i

        # 2) Si el último es del usuario y falta respuesta, GENERAR ACÁ
        if (st.session_state["chat_messages"]
                and st.session_state["chat_messages"][-1]["role"] == "user"):
            contexto = construir_contexto_lote(st.session_state)
            if lote_para_ia_id:
                ctx_historico = db.resumen_lote_para_ia(lote_para_ia_id)
                if ctx_historico:
                    contexto = (contexto + "\n\n" + ctx_historico).strip()
                # ─── Histórico clínico completo (unificado con el
                # análisis climático del lote): mortandad por causa,
                # patrones recurrentes, diagnósticos abiertos, ADG
                # real, sub-consumo medido, fase del plan, últimas
                # consultas. Hace que el chat conversacional vea lo
                # mismo que el botón "🤖 Generar análisis IA".
                try:
                    ctx_clinico = (
                        dashboard.armar_contexto_clinico_lote(
                            lote_para_ia_id, db,
                        )
                    )
                    if ctx_clinico:
                        contexto = (
                            contexto + "\n\n" + ctx_clinico
                        ).strip()
                except Exception:
                    pass
                # Si el cliente del lote tiene localidad, sumar clima + alertas
                lote_data = db.obtener_lote(lote_para_ia_id)
                if lote_data:
                    cli = db.obtener_cliente(lote_data["cliente_id"])
                    if cli and cli.get("localidad"):
                        try:
                            ctx_clima = resumen_clima_para_ia(
                                cli["localidad"],
                                categoria=lote_data.get("categoria", ""),
                            )
                            if ctx_clima:
                                contexto = (contexto + "\n\n" + ctx_clima).strip()
                            # Si el cliente está en La Pampa, sumar
                            # estación oficial (datos reales gobierno)
                            if cli.get("lat") and cli.get("lon"):
                                try:
                                    ctx_oficial = resumen_estacion_oficial(
                                        float(cli["lat"]), float(cli["lon"]),
                                    )
                                    if ctx_oficial:
                                        contexto = (
                                            contexto + "\n\n" + ctx_oficial
                                        ).strip()
                                except Exception:
                                    pass
                        except Exception as e:
                            logging.warning(f"Clima no disponible: {e}")
            ctx_ingredientes = construir_contexto_ingredientes(st.session_state)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                try:
                    # Pasar ingredientes en formato dict para tool use
                    ings_data = None
                    if "ingredientes_df" in st.session_state:
                        try:
                            ings_data = st.session_state["ingredientes_df"].to_dict("records")
                        except Exception:
                            ings_data = None

                    # PDFs adjuntos al último mensaje del usuario
                    _pdfs_envio = None
                    _ultimo_msg = st.session_state["chat_messages"][-1]
                    if _ultimo_msg.get("role") == "user":
                        _pdfs_envio = _ultimo_msg.get("_pdf_attachments")
                        if _pdfs_envio:
                            placeholder.markdown(
                                f"📎 *Procesando "
                                f"{len(_pdfs_envio)} PDF(s)…*"
                            )

                    for chunk in chat_streaming(
                        st.session_state["chat_messages"],
                        contexto_lote=contexto,
                        contexto_ingredientes=ctx_ingredientes,
                        ingredientes_session=ings_data,
                        pdf_attachments=_pdfs_envio,
                        api_key=api_key,
                        model=_modelo_ia,
                    ):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    st.session_state["chat_messages"].append(
                        {"role": "assistant", "content": full_response}
                    )
                    # Limpiar los PDFs después de enviar para no
                    # re-enviarlos en el próximo turno (ya no hace falta
                    # que viajen junto a cada mensaje del cliente).
                    if _ultimo_msg.get("_pdf_attachments"):
                        _ultimo_msg["_pdf_attachments"] = None
                except Exception as e:
                    st.error(f"Error al consultar Claude: {e}")

    # Mini-formulario "Recordar" que aparece cuando se clickeó un botón
    if st.session_state.get("recordando_msg_idx") is not None:
        idx_rec = st.session_state["recordando_msg_idx"]
        if 0 <= idx_rec < len(st.session_state["chat_messages"]):
            msg_target = st.session_state["chat_messages"][idx_rec]
            # Sugerencia: si es respuesta del agente, también incluir contexto
            #   del mensaje user previo para entender qué se está recordando
            sugerencia = msg_target["content"]
            if msg_target["role"] == "assistant" and idx_rec > 0:
                user_prev = st.session_state["chat_messages"][idx_rec - 1]
                sugerencia = (
                    f"En respuesta a: «{user_prev['content'][:200]}»\n\n"
                    f"Apliqué/concluyó: {msg_target['content']}"
                )

            st.divider()
            st.markdown("##### 💾 Guardar como memoria del agente")
            with st.form("guardar_memoria_inline", clear_on_submit=False):
                rec_categoria = st.selectbox(
                    "Categoría",
                    ["correccion", "valor_local", "ingrediente",
                     "manejo", "cliente", "preferencia", "general"],
                    format_func=lambda x: {
                        "correccion": "🔧 Corrección técnica",
                        "valor_local": "📍 Valor típico de la zona",
                        "ingrediente": "🌾 Composición de ingrediente",
                        "manejo": "⚙️ Característica de manejo",
                        "cliente": "👤 Info de cliente",
                        "preferencia": "❤️ Preferencia personal",
                        "general": "📝 Otra",
                    }[x],
                    key="rec_cat",
                )
                rec_texto = st.text_area(
                    "Texto a recordar (editá si querés acortarlo o reformular)",
                    value=sugerencia,
                    height=120,
                    key="rec_txt",
                    help="Escribí lo más conciso posible. "
                         "Pensá: ¿qué regla / valor / criterio quiero "
                         "que el agente recuerde la próxima vez?",
                )
                rec_etiq = st.text_input(
                    "Etiqueta (opcional, p. ej. 'silaje', 'feedlot terminación')",
                    key="rec_etq",
                )
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    if st.form_submit_button("✅ Guardar", type="primary",
                                              width="stretch"):
                        memoria.agregar_memoria(rec_texto, rec_categoria, rec_etiq)
                        st.session_state["recordando_msg_idx"] = None
                        st.success("✅ Memoria guardada — se aplica desde el próximo chat")
                        st.rerun()
                with col_g2:
                    if st.form_submit_button("✖ Cancelar",
                                              width="stretch"):
                        st.session_state["recordando_msg_idx"] = None
                        st.rerun()

    # =========== Mini-form PDF de respuesta del agente ===========
    if st.session_state.get("pdf_chat_idx") is not None:
        idx_pdf = st.session_state["pdf_chat_idx"]
        if 0 <= idx_pdf < len(st.session_state["chat_messages"]):
            msg_pdf = st.session_state["chat_messages"][idx_pdf]
            st.divider()
            st.markdown("##### 📄 Generar PDF profesional con marca HMS")

            # ─── Inferir título por contenido ───
            # Antes el default era siempre "Plan de alimentación" — pero
            # el agente genera muchos otros tipos de informe (manejo
            # ante frío/calor, plan de adaptación, prevención sanitaria,
            # análisis nutricional, etc.). Detectamos por keywords para
            # proponer un título coherente con el cuerpo del mensaje.
            def _inferir_titulo(texto_md: str) -> str:
                t = (texto_md or "").lower()
                # Orden importa: el primer match gana
                reglas = [
                    (
                        ("plan de adaptación", "adaptación de 4 fases",
                         "fase 1", "fase 2", "fase 3", "fase 4",
                         "adaptación al concentrado"),
                        "Plan de adaptación",
                    ),
                    (
                        ("estrés calórico", "estrés térmico calor",
                         "thi alto", "ola de calor"),
                        "Manejo ante estrés calórico",
                    ),
                    (
                        ("estrés por frío", "frío sostenido",
                         "wind chill", "reparo del viento",
                         "cama seca", "termorregulación",
                         "clima frío", "ola de frío"),
                        "Manejo ante frío y clima adverso",
                    ),
                    (
                        ("situación climática", "próximos 7 días",
                         "pronóstico", "clima de los próximos",
                         "lluvia prevista"),
                        "Recomendaciones de manejo climático",
                    ),
                    (
                        ("acidosis", "timpanismo", "diarrea",
                         "neumonía", "sanitaria", "patología"),
                        "Recomendaciones sanitarias",
                    ),
                    (
                        ("optimización de dieta", "optimizador",
                         "mínimo costo", "costo de ración"),
                        "Optimización de dieta",
                    ),
                    (
                        ("análisis nutricional", "balance de raciones",
                         "% pb", "em mcal", "ndt %"),
                        "Análisis nutricional",
                    ),
                    (
                        ("plan de alimentación", "fórmula",
                         "ración", "ingredientes", "fibrogreen",
                         "kg/animal/día", "mezcla concentrada"),
                        "Plan de alimentación",
                    ),
                ]
                for kws, titulo in reglas:
                    for k in kws:
                        if k in t:
                            return titulo
                return "Informe técnico"

            _titulo_sugerido = _inferir_titulo(msg_pdf.get("content", ""))

            # ─── Badge de estado: ¿la dieta del lote está guardada? ───
            # Sin dieta guardada en el historial del lote, las alertas
            # (stock, silo, cambio de fase) y la demanda consolidada NO
            # tienen de dónde leer. Mostramos el estado antes de generar
            # el PDF para que el asesor lo confirme.
            if lote_para_ia_id:
                _dietas_lote = db.listar_dietas(lote_para_ia_id)
                if _dietas_lote:
                    st.success(
                        f"✅ Este lote tiene **{len(_dietas_lote)} "
                        f"dieta(s) guardada(s)** en el historial. "
                        f"Las alertas de stock, silo y cambio de fase "
                        f"van a funcionar con esos datos."
                    )
                else:
                    st.error(
                        "⚠️ **Este lote NO tiene dietas guardadas en el "
                        "historial.** Si generás el PDF así, queda como "
                        "documento pero las alertas (stock bajo, fin "
                        "de carga del silo, cambio de fase) y la vista "
                        "de demanda consolidada **NO** van a poder "
                        "calcular nada para este cliente.\n\n"
                        "Antes de generar el PDF, volvé al chat y "
                        "pedile al agente:\n"
                        "*\"Guardá la dieta (o el plan de adaptación) "
                        "en la ficha del lote\"*."
                    )
            else:
                st.info(
                    "ℹ️ Estás generando un PDF sin lote asociado en el "
                    "contexto. El PDF queda como documento autónomo — "
                    "no se conecta con ningún lote del sistema."
                )

            # Sugerir datos desde el contexto si los hay
            cliente_default = ""
            estab_default = ""
            lote_default = ""
            raza_default = ""
            cat_default = ""
            cant_default = 0
            peso_default = 0.0
            if lote_para_ia_id:
                lote_data = db.obtener_lote(lote_para_ia_id)
                if lote_data:
                    cliente_default = lote_data.get("cliente_nombre", "") or ""
                    estab_default = lote_data.get("establecimiento", "") or ""
                    lote_default = lote_data.get("identificador", "") or ""
                    raza_default = lote_data.get("raza", "") or ""
                    cat_default = lote_data.get("categoria", "") or ""
                    cant_default = int(lote_data.get("cantidad_inicial", 0) or 0)
                    peso_default = float(lote_data.get("peso_ingreso_kg", 0) or 0)

            # Si no hay lote contextual pero sí video procesado, usar esos datos
            if not cant_default and st.session_state.get("vid_n"):
                cant_default = int(st.session_state["vid_n"])
            if not peso_default and st.session_state.get("vid_prom"):
                peso_default = float(st.session_state["vid_prom"])

            with st.form("generar_pdf_chat"):
                st.markdown("##### Datos para la carátula")
                col_pdf1, col_pdf2 = st.columns(2)
                with col_pdf1:
                    pdf_titulo = st.text_input(
                        "Título del informe",
                        value=_titulo_sugerido,
                        help=(
                            "Sugerido a partir del contenido de la "
                            "respuesta. Editalo si querés otro título."
                        ),
                    )
                    pdf_cliente = st.text_input("Cliente", value=cliente_default)
                    pdf_estab = st.text_input("Establecimiento", value=estab_default)
                    pdf_lote = st.text_input("Identificación del lote", value=lote_default)
                with col_pdf2:
                    pdf_fecha = st.date_input("Fecha", value=datetime.now().date())
                    pdf_raza = st.text_input("Raza", value=raza_default)
                    pdf_cat = st.text_input("Categoría", value=cat_default)
                    col_p1, col_p2 = st.columns(2)
                    pdf_cant = col_p1.number_input(
                        "Cantidad", min_value=0, value=cant_default, step=1,
                    )
                    pdf_peso = col_p2.number_input(
                        "Peso prom (kg)", min_value=0.0, value=peso_default, step=1.0,
                    )

                pdf_objetivo = st.text_input(
                    "Objetivo productivo (opcional)",
                    placeholder="Ej: ADG 0.8 kg/día — Servicio a 15 meses",
                )

                pdf_contenido = st.text_area(
                    "Contenido del informe (editable, en markdown)",
                    value=msg_pdf["content"],
                    height=300,
                    help="Podés editar el texto antes de generar el PDF. "
                         "Markdown soportado: # ## ### títulos, listas, tablas, **bold**.",
                )

                col_pdf_btn1, col_pdf_btn2 = st.columns(2)
                generar = col_pdf_btn1.form_submit_button(
                    "📄 Generar PDF", type="primary", width="stretch",
                )
                cancelar = col_pdf_btn2.form_submit_button(
                    "✖ Cancelar", width="stretch",
                )

                if generar:
                    try:
                        _nombre_pdf_chat = armar_nombre_pdf(
                            cliente=pdf_cliente,
                            categoria=pdf_cat,
                            objetivo=pdf_objetivo,
                            lote=pdf_lote,
                            fecha=pdf_fecha,
                            sufijo="informe",
                        )
                        out_path = (Path(tempfile.mkdtemp())
                                    / _nombre_pdf_chat)
                        generar_pdf_informe_chat(
                            out_path,
                            contenido_markdown=pdf_contenido,
                            titulo_default=pdf_titulo,
                            cliente=pdf_cliente,
                            establecimiento=pdf_estab,
                            lote=pdf_lote,
                            raza=pdf_raza,
                            categoria=pdf_cat,
                            cantidad=int(pdf_cant),
                            peso_kg=float(pdf_peso),
                            objetivo=pdf_objetivo,
                            fecha=datetime.combine(pdf_fecha, datetime.min.time()),
                        )
                        with open(out_path, "rb") as fh:
                            st.session_state["pdf_chat_bytes"] = fh.read()
                        st.session_state["pdf_chat_filename"] = out_path.name
                        st.session_state["pdf_chat_idx"] = None
                        st.success("✅ PDF generado")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generando PDF: {e}")
                        st.exception(e)
                if cancelar:
                    st.session_state["pdf_chat_idx"] = None
                    st.rerun()

    # Botón de descarga si recién se generó
    if st.session_state.get("pdf_chat_bytes"):
        st.download_button(
            "⬇️ Descargar PDF generado",
            data=st.session_state["pdf_chat_bytes"],
            file_name=st.session_state.get("pdf_chat_filename", "informe.pdf"),
            mime="application/pdf",
            key="dl_chat_pdf",
        )

    # 3) Info de contexto (opcional, expander)
    with st.expander("🔍 Ver el contexto completo que la IA recibe"):
        partes = []
        ctx_lote = construir_contexto_lote(st.session_state)
        if ctx_lote:
            partes.append(ctx_lote)
        if lote_para_ia_id:
            ctx_h = db.resumen_lote_para_ia(lote_para_ia_id)
            if ctx_h:
                partes.append(ctx_h)
        ctx_ings = construir_contexto_ingredientes(st.session_state)
        if ctx_ings:
            partes.append(ctx_ings)
        bloque_mem = memoria.construir_bloque_memoria()
        if bloque_mem:
            partes.append(bloque_mem)
        ctx_completo = "\n\n".join(partes) if partes else "(sin contexto)"
        st.code(ctx_completo, language="text")

    # 4) Adjuntos PDF — para subir dietas formuladas, análisis de
    # laboratorio o cualquier documento que el agente pueda leer.
    with st.expander(
        "📎 Adjuntar PDF al próximo mensaje "
        "(dietas formuladas, análisis, informes…)",
        expanded=False,
    ):
        st.caption(
            "El agente puede leer PDFs directamente — extrae la dieta, "
            "el análisis o la información del documento y la podés "
            "guardar al lote. Útil para migrar fórmulas de Excel o "
            "papel."
        )
        _pdfs_subidos = st.file_uploader(
            "Subí uno o más PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="chat_pdf_uploader",
        )
        if _pdfs_subidos:
            for _f in _pdfs_subidos:
                st.caption(
                    f"📎 {_f.name} ({_f.size/1024:.0f} KB) — listo "
                    f"para enviar con tu próximo mensaje."
                )

    # 5) Input del usuario AL FINAL — debajo del último mensaje
    user_input = st.chat_input("Hacé tu pregunta o pedido…")
    if user_input:
        # Si hay PDFs adjuntos en el uploader, capturarlos AHORA
        # (bytes + nombre) y mandarlos junto con el mensaje. Después
        # los limpiamos para no re-enviarlos en el siguiente turno.
        _pdfs_para_msg = []
        try:
            _archivos = st.session_state.get("chat_pdf_uploader") or []
            for _f in _archivos:
                _pdfs_para_msg.append({
                    "filename": _f.name,
                    "data": _f.getvalue(),
                })
        except Exception:
            _pdfs_para_msg = []
        st.session_state["chat_messages"].append(
            {
                "role": "user",
                "content": user_input,
                "_pdf_attachments": _pdfs_para_msg or None,
            }
        )
        st.rerun()


# ----------------------- ENTRENAMIENTO AVANZADO -----------------------
# --------------------------- CONFIGURACIÓN ---------------------------
with tab_config:
    st.markdown("### ⚙️ Configuración técnica del sistema")
    st.caption(
        "Acá editás los datos que el agente IA usa internamente para "
        "formular dietas y dar recomendaciones. Vos dejás todo cargado "
        "una vez y el agente se ocupa del resto."
    )

    (sub_ings, sub_cats, sub_mem, sub_marca,
     sub_drone, sub_email, sub_whatsapp) = st.tabs([
        "🌾 Ingredientes",
        "🐄 Categorías de animales",
        "🧠 Memoria del agente",
        "🎨 Marca HMS",
        "📐 Calibración drone",
        "📧 Alertas por email",
        "💬 WhatsApp",
    ])

    # ---- Sub-tab: Ingredientes ----
    with sub_ings:
        st.markdown(
            "Marcá los ingredientes **disponibles en el campo** y editá "
            "los precios actuales y los valores nutricionales según tus "
            "análisis reales (no los del catálogo)."
        )
        st.caption(
            "Cuando le pidas al agente IA que arme una dieta, va a usar "
            "EXACTAMENTE estos valores. Si tu Fibrogreen tiene 32% PB "
            "(no 30 estándar), pon 32 acá y el agente lo respeta."
        )

        if "ingredientes_df" not in st.session_state:
            st.session_state["ingredientes_df"] = pd.DataFrame(
                [asdict_ing(i) for i in ingredientes_default()]
            )

        edited_cfg = st.data_editor(
            st.session_state["ingredientes_df"],
            hide_index=True, width="stretch", height=420,
            key="cfg_ingredientes_editor",
            column_config={
                "nombre": st.column_config.TextColumn("Ingrediente", width="large"),
                "categoria": st.column_config.SelectboxColumn(
                    "Categoría",
                    options=["concentrado", "forraje", "suplemento", "mineral", "balanceado"],
                ),
                "ms_pct": st.column_config.NumberColumn("MS %", min_value=0, max_value=100, step=0.5, format="%.1f"),
                "pb_pct_ms": st.column_config.NumberColumn("PB %MS", min_value=0, max_value=300, step=0.5, format="%.1f"),
                "em_mcal_kg_ms": st.column_config.NumberColumn("EM Mcal/kgMS", min_value=0, max_value=5, step=0.05, format="%.2f"),
                "fdn_pct_ms": st.column_config.NumberColumn("FDN %MS", min_value=0, max_value=100, step=1, format="%.0f"),
                "ca_pct_ms": st.column_config.NumberColumn("Ca %MS", min_value=0, max_value=50, step=0.05, format="%.2f"),
                "p_pct_ms": st.column_config.NumberColumn("P %MS", min_value=0, max_value=15, step=0.05, format="%.2f"),
                "nnp_pct_ms": st.column_config.NumberColumn("NNP %MS", min_value=0, max_value=300, step=0.5, format="%.1f",
                    help="Nitrógeno No Proteico (urea equivalente). Crítico para evitar toxicidad."),
                "precio_kg_tal_cual": st.column_config.NumberColumn("$/kg t/c", min_value=0, step=10, format="%.0f"),
                "min_inclusion_pct_ms": st.column_config.NumberColumn("Mín %", min_value=0, max_value=100, step=1, format="%.0f"),
                "max_inclusion_pct_ms": st.column_config.NumberColumn(
                    "Máx %", min_value=0, max_value=100, step=1, format="%.0f",
                    help="Inclusión máxima permitida (ej. Fibrogreen 20% por monensina).",
                ),
                "disponible": st.column_config.CheckboxColumn(
                    "Disponible",
                    help="Tildá solo los que tenés en el campo",
                ),
            },
            num_rows="dynamic",
        )
        # Actualizar estado base
        st.session_state["ingredientes_df"] = edited_cfg

        col_c1, col_c2, col_c3 = st.columns([1, 1, 4])
        if col_c1.button("🔄 Restaurar default", key="cfg_reset_ings"):
            st.session_state["ingredientes_df"] = pd.DataFrame(
                [asdict_ing(i) for i in ingredientes_default()]
            )
            if "cfg_ingredientes_editor" in st.session_state:
                del st.session_state["cfg_ingredientes_editor"]
            st.rerun()
        if col_c2.button("✅ Tildar todos", key="cfg_tildar_todos"):
            df_actual = edited_cfg.copy()
            df_actual["disponible"] = True
            st.session_state["ingredientes_df"] = df_actual
            if "cfg_ingredientes_editor" in st.session_state:
                del st.session_state["cfg_ingredientes_editor"]
            st.rerun()

        # Mostrar resumen de qué tiene cargado
        n_disp = sum(1 for r in edited_cfg.to_dict("records")
                     if r.get("disponible"))
        st.success(f"✅ {n_disp} ingredientes marcados como disponibles "
                    "para que el agente los use al formular dietas.")

    # ---- Sub-tab: Categorías de animales ----
    with sub_cats:
        st.markdown(
            "Administrá la lista de categorías que aparecen en los "
            "dropdowns de **Nuevo Lote**, **Drone** y **Dieta del "
            "Asesor IA**. Podés agregar, renombrar, ajustar el ADPV "
            "default o desactivar las que no usás."
        )
        st.caption(
            "📌 El **ADPV default** se usa cuando un lote no tiene "
            "ganancia diaria propia cargada. Si renombrás una "
            "categoría que ya está en uso por algún lote, ese lote "
            "va a seguir mostrando el nombre viejo hasta que lo "
            "edites. Por eso, en general, conviene **desactivar** "
            "antes que borrar."
        )

        _cats_admin = db.listar_categorias(solo_activas=False)
        if _cats_admin:
            _df_cats = pd.DataFrame([
                {
                    "id": c["id"],
                    "nombre": c["nombre"],
                    "adpv_default_kg_dia": c["adpv_default_kg_dia"],
                    "orden": c["orden"],
                    "activo": bool(c["activo"]),
                    "notas": c.get("notas") or "",
                }
                for c in _cats_admin
            ])
            _df_cats_ed = st.data_editor(
                _df_cats, hide_index=True, width="stretch",
                key="cats_animales_editor",
                disabled=["id"],
                column_config={
                    "id": st.column_config.NumberColumn(
                        "ID", width="small",
                    ),
                    "nombre": st.column_config.TextColumn(
                        "Nombre", required=True,
                    ),
                    "adpv_default_kg_dia": st.column_config.NumberColumn(
                        "ADPV default (kg/día)",
                        min_value=0.0, max_value=2.5, step=0.05,
                        format="%.2f",
                        help=(
                            "Ganancia diaria esperada por default. Se "
                            "usa si el lote no tiene ADPV propio."
                        ),
                    ),
                    "orden": st.column_config.NumberColumn(
                        "Orden", min_value=0, step=5,
                        help=(
                            "Define el orden en los dropdowns "
                            "(menor = primero)."
                        ),
                    ),
                    "activo": st.column_config.CheckboxColumn("Activo"),
                    "notas": st.column_config.TextColumn(
                        "Notas", width="large",
                    ),
                },
            )
            if st.button(
                "💾 Guardar cambios en categorías",
                key="btn_save_cats",
                type="primary",
            ):
                _orig_ids = {c["id"]: c for c in _cats_admin}
                _cambios = 0
                for _r in _df_cats_ed.to_dict("records"):
                    _orig = _orig_ids.get(_r["id"])
                    if not _orig:
                        continue
                    _hay_cambio = (
                        _r["nombre"].strip().lower() != _orig["nombre"]
                        or float(_r["adpv_default_kg_dia"] or 0)
                            != float(_orig["adpv_default_kg_dia"] or 0)
                        or int(_r["orden"] or 0)
                            != int(_orig["orden"] or 0)
                        or int(bool(_r["activo"]))
                            != int(_orig["activo"])
                        or (_r["notas"] or "")
                            != (_orig.get("notas") or "")
                    )
                    if _hay_cambio:
                        try:
                            db.actualizar_categoria(
                                _r["id"],
                                nombre=_r["nombre"],
                                adpv_default_kg_dia=_r[
                                    "adpv_default_kg_dia"],
                                orden=_r["orden"],
                                activo=1 if _r["activo"] else 0,
                                notas=_r["notas"],
                            )
                            _cambios += 1
                        except Exception as _e:
                            st.error(
                                f"Error en '{_r['nombre']}': {_e}"
                            )
                if _cambios:
                    st.success(f"✅ {_cambios} categoría(s) actualizada(s).")
                    st.rerun()
                else:
                    st.info("No se detectaron cambios.")
        else:
            st.info(
                "No hay categorías cargadas. Creá la primera abajo."
            )

        st.markdown("---")
        st.markdown("##### ➕ Agregar nueva categoría")
        _col_nc1, _col_nc2, _col_nc3 = st.columns([2, 1, 1])
        with _col_nc1:
            _nc_nombre = st.text_input(
                "Nombre",
                placeholder="ej: vaca_seca, recria_pesada, ternero_holando",
                key="nc_nombre",
            )
        with _col_nc2:
            _nc_adpv = st.number_input(
                "ADPV default (kg/día)",
                min_value=0.0, max_value=2.5, step=0.05, value=0.8,
                key="nc_adpv",
            )
        with _col_nc3:
            _nc_orden = st.number_input(
                "Orden",
                min_value=0, step=5,
                value=(max(c["orden"] for c in _cats_admin) + 10
                       if _cats_admin else 10),
                key="nc_orden",
            )
        _nc_notas = st.text_input(
            "Notas (opcional)",
            placeholder="ej: hembra de descarte, fuera de servicio",
            key="nc_notas",
        )
        if st.button("➕ Crear categoría", key="btn_crear_cat"):
            _nm = (_nc_nombre or "").strip()
            if not _nm:
                st.error("Tenés que poner un nombre.")
            else:
                try:
                    db.crear_categoria(
                        _nm,
                        adpv_default_kg_dia=_nc_adpv,
                        orden=int(_nc_orden),
                        notas=_nc_notas,
                    )
                    st.success(f"✅ Categoría '{_nm.lower()}' creada.")
                    st.rerun()
                except ValueError as _e:
                    st.error(str(_e))

    # ---- Sub-tab: Memoria del agente ----
    with sub_mem:
        st.markdown(
            "Acá enseñás cosas al agente que se aplican en TODAS las "
            "conversaciones futuras. Correcciones técnicas, valores típicos "
            "de tu zona, criterios de manejo, etc."
        )

        col_mem_form, col_mem_list = st.columns([1, 2])
        with col_mem_form:
            with st.form("nueva_memoria_cfg", clear_on_submit=True):
                m_categoria = st.selectbox(
                    "Categoría",
                    ["correccion", "valor_local", "ingrediente",
                     "manejo", "cliente", "preferencia", "general"],
                    format_func=lambda x: {
                        "correccion": "🔧 Corrección técnica",
                        "valor_local": "📍 Valor típico de la zona",
                        "ingrediente": "🌾 Composición de ingrediente",
                        "manejo": "⚙️ Característica de manejo",
                        "cliente": "👤 Info de cliente",
                        "preferencia": "❤️ Preferencia personal",
                        "general": "📝 Otra",
                    }[x],
                    key="cfg_mem_cat",
                )
                m_texto = st.text_area(
                    "Texto a recordar", height=100,
                    placeholder="Ej: 'En La Pampa el silaje de maíz típico tiene 28% MS y 7% PB'",
                    key="cfg_mem_txt",
                )
                m_etiqueta = st.text_input(
                    "Etiqueta (opcional)", key="cfg_mem_etq",
                )
                if st.form_submit_button("💾 Guardar memoria", type="primary"):
                    if m_texto:
                        memoria.agregar_memoria(m_texto, m_categoria, m_etiqueta)
                        st.success("✅ Guardada — se aplica desde el próximo chat")
                        st.rerun()

        with col_mem_list:
            todas = memoria.listar_memorias(activas_solo=False)
            if not todas:
                st.info("Sin memorias todavía. Cargá la primera con el formulario.")
            else:
                for m in sorted(todas, key=lambda x: x["fecha"], reverse=True):
                    estado = "✅" if m.get("activa", True) else "⏸️"
                    cat_emoji = {
                        "correccion": "🔧", "valor_local": "📍",
                        "ingrediente": "🌾", "manejo": "⚙️",
                        "cliente": "👤", "preferencia": "❤️",
                        "general": "📝",
                    }.get(m["categoria"], "📝")
                    cols_m = st.columns([8, 1, 1])
                    cols_m[0].write(f"{estado} {cat_emoji} {m['texto']}")
                    if cols_m[1].button(
                        "⏸️" if m.get("activa", True) else "▶️",
                        key=f"cfg_tog_{m['id']}",
                    ):
                        if m.get("activa", True):
                            memoria.desactivar_memoria(m["id"])
                        else:
                            memoria.reactivar_memoria(m["id"])
                        st.rerun()
                    if cols_m[2].button("🗑️", key=f"cfg_del_{m['id']}"):
                        memoria.eliminar_memoria(m["id"])
                        st.rerun()

    # ---- Sub-tab: Marca HMS ----
    with sub_marca:
        st.markdown("### 🎨 Identidad HMS")
        st.caption("Logos para PDFs profesionales. Estos se usan también en la sidebar.")

        c_lc, c_lb = st.columns(2)
        for col, prefijo, label, fondo in [
            (c_lc, "logo", "Logo color (fondos blancos)", None),
            (c_lb, "logo_blanco", "Logo blanco (banda verde)", "#1B3E27"),
        ]:
            with col:
                st.markdown(f"**{label}**")
                # Buscar logo
                logo_path = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    p = Path(f"assets/{prefijo}{ext}")
                    if p.exists():
                        logo_path = p
                        break
                if logo_path:
                    if fondo:
                        # Mostrar con fondo verde para preview del logo blanco
                        import base64
                        st.markdown(
                            f'<div style="background:{fondo};padding:14px;'
                            f'border-radius:6px;text-align:center;">'
                            f'<img src="data:image/png;base64,{base64.b64encode(open(logo_path,"rb").read()).decode()}" '
                            f'style="max-width:140px;max-height:140px;"/>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.image(str(logo_path), width=140)
                    if st.button(f"🗑️ Borrar {prefijo}", key=f"cfg_del_{prefijo}"):
                        logo_path.unlink()
                        st.rerun()
                else:
                    st.warning(f"⚠️ Sin {label} cargado")

                up = st.file_uploader(
                    f"Subir {prefijo}", type=["png", "jpg", "jpeg"],
                    key=f"cfg_up_{prefijo}",
                )
                if up:
                    Path("assets").mkdir(exist_ok=True)
                    for old_ext in [".png", ".jpg", ".jpeg"]:
                        old = Path(f"assets/{prefijo}{old_ext}")
                        if old.exists():
                            old.unlink()
                    ext = Path(up.name).suffix.lower()
                    save_path = Path(f"assets/{prefijo}{ext}")
                    save_path.write_bytes(up.getvalue())
                    st.success(f"✅ Guardado: {save_path.name}")
                    st.rerun()

    # ---- Sub-tab: Calibración drone ----
    with sub_drone:
        st.markdown("### 📐 Parámetros del módulo drone")
        st.caption("Estos valores se aplican al procesamiento de imágenes y videos.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Captura")
            st.write(f"Altura de vuelo: **{cfg['captura']['altura_vuelo_m']} m**")
            st.write(f"Resolución: **{cfg['captura']['resolucion']}**")
            st.write(f"FPS: **{cfg['captura']['fps']}**")

        with c2:
            st.markdown("##### Referencia")
            st.write(f"Método: **{cfg['referencia']['metodo']}**")
            st.write(f"Lado del cuadrado: **{cfg['referencia']['lado_m']} m**")

        st.markdown("##### Modelo de peso (calibración alométrica)")
        c3, c4, c5 = st.columns(3)
        c3.metric("Coeficiente a", cfg["estimacion_peso"]["coef_a"])
        c4.metric("Coeficiente b", cfg["estimacion_peso"]["coef_b"])
        c5.metric("Modelo", cfg["estimacion_peso"]["modelo"])
        st.caption(
            "💡 Para edición de los parámetros del drone (altura, modelo "
            "YOLO, etc.) usá la **sidebar** mientras estás procesando una "
            "imagen o video. Para calibración profunda con balanza, andá a "
            "🎓 Entrenamiento → Calibrar pesos."
        )

    # ---- Sub-tab: Alertas por email ----
    with sub_email:
        from src import alertas_email as ae

        st.markdown("### 📧 Alertas climáticas automáticas por email")
        st.caption(
            "El sistema corre todos los días a las 08:00 y manda un mail con "
            "las alertas climáticas relevantes. Vos recibís un digest con todos "
            "los clientes; cada cliente con email cargado recibe sus alertas."
        )

        cfg_smtp_actual = ae.cargar_config_smtp() or {}

        # Inicializar session_state con la config guardada (si los campos
        # no fueron tocados aún en esta sesión)
        for k, v in [
            ("smtp_host", cfg_smtp_actual.get("host", "")),
            ("smtp_user", cfg_smtp_actual.get("user", "")),
            ("smtp_password", cfg_smtp_actual.get("password", "")),
            ("smtp_from_email", cfg_smtp_actual.get(
                "from_email", "mauricio@hmsnutricionanimal.com.ar")),
            ("smtp_from_name", cfg_smtp_actual.get(
                "from_name", "HMS Nutrición Animal")),
            ("smtp_port", int(cfg_smtp_actual.get("port", 587))),
            ("smtp_use_ssl", bool(cfg_smtp_actual.get("use_ssl", False))),
            ("smtp_use_tls", bool(cfg_smtp_actual.get("use_tls", True))),
            ("smtp_admin_email", cfg_smtp_actual.get(
                "admin_email", "mauricio@hmsnutricionanimal.com.ar")),
            ("smtp_bcc_clientes", cfg_smtp_actual.get(
                "bcc_clientes", "")),
            ("smtp_bcc_activo", bool(cfg_smtp_actual.get(
                "bcc_clientes"))),
            ("smtp_imap_host", cfg_smtp_actual.get("imap_host", "")),
            ("smtp_imap_user", cfg_smtp_actual.get("imap_user", "")),
            ("smtp_imap_password",
             cfg_smtp_actual.get("imap_password", "")),
        ]:
            if k not in st.session_state:
                st.session_state[k] = v

        st.markdown("##### Servidor SMTP de tu hosting")
        st.caption(
            "Pedile al proveedor de hmsnutricionanimal.com.ar los datos SMTP "
            "salientes (suelen ser puerto 587 con STARTTLS, o 465 con SSL)."
        )

        # NOTA: como ya inicializamos session_state arriba con los valores
        # guardados, NO pasamos `value=` para evitar el warning de Streamlit
        # ("widget creado con default pero también con session_state").
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            smtp_host = st.text_input(
                "Host SMTP",
                placeholder="ej: mail.hmsnutricionanimal.com.ar",
                key="smtp_host",
            )
            smtp_user = st.text_input(
                "Usuario",
                placeholder="mauricio@hmsnutricionanimal.com.ar",
                key="smtp_user",
            )
            smtp_from_email = st.text_input(
                "Remitente (From)",
                key="smtp_from_email",
            )
        with col_s2:
            smtp_port = st.number_input(
                "Puerto",
                min_value=1, max_value=65535,
                key="smtp_port",
            )
            smtp_password = st.text_input(
                "Contraseña",
                type="password",
                key="smtp_password",
            )
            smtp_from_name = st.text_input(
                "Nombre remitente",
                key="smtp_from_name",
            )

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            smtp_use_ssl = st.checkbox(
                "Usar SSL (puerto 465)",
                key="smtp_use_ssl",
            )
        with col_e2:
            smtp_use_tls = st.checkbox(
                "Usar STARTTLS (puerto 587)",
                key="smtp_use_tls",
            )

        admin_email = st.text_input(
            "Email del admin (vos) — recibe el digest de TODOS los clientes",
            key="smtp_admin_email",
        )

        # ────── BCC del admin en emails a clientes ──────
        st.markdown("##### 📧 Copia (BCC) en emails a clientes")
        st.caption(
            "Si está activado, cada vez que el sistema le manda un email "
            "a un cliente (alerta diaria, semanal, crítica, bienvenida), "
            "vos recibís una copia oculta para tener trazabilidad. "
            "Independiente del digest administrativo (que es un resumen "
            "agregado del día)."
        )
        col_bcc1, col_bcc2 = st.columns([1, 2])
        bcc_activo = col_bcc1.checkbox(
            "Activar copia oculta",
            key="smtp_bcc_activo",
        )
        bcc_clientes_input = col_bcc2.text_input(
            "Email(s) que reciben copia",
            key="smtp_bcc_clientes",
            placeholder="ej: mauricio@hmsnutricionanimal.com.ar",
            help=(
                "Podés poner uno o varios separados por coma. Si dejás "
                "vacío y activás el toggle, se usa el 'Email del admin' "
                "de arriba."
            ),
            disabled=not bcc_activo,
        )

        # ───── IMAP separado (opcional) ─────
        with st.expander(
            "📥 Procesar bajas — IMAP separado (opcional, dejá vacío "
            "para usar el mismo proveedor SMTP)",
            expanded=False,
        ):
            st.caption(
                "Si vas a recibir las respuestas BAJA en una cuenta distinta "
                "del SMTP de envío (por ej. enviás vía Gmail pero recibís en "
                "iCloud Custom Domain), cargá los datos IMAP acá."
            )
            col_im1, col_im2 = st.columns(2)
            with col_im1:
                imap_host = st.text_input(
                    "Host IMAP",
                    placeholder="ej: imap.mail.me.com (iCloud) o imap.gmail.com",
                    key="smtp_imap_host",
                )
                imap_user = st.text_input(
                    "Usuario IMAP",
                    placeholder="hmsna@icloud.com",
                    key="smtp_imap_user",
                )
            with col_im2:
                imap_password = st.text_input(
                    "Contraseña IMAP (app-specific)",
                    type="password",
                    key="smtp_imap_password",
                )
                st.caption(
                    "Para iCloud: app-specific password de "
                    "[appleid.apple.com](https://appleid.apple.com) → "
                    "Contraseñas de aplicaciones."
                )

        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])

        if col_b1.button("💾 Guardar configuración", type="primary",
                          key="smtp_save_btn"):
            # BCC del admin: si está activo, usar el campo si tiene
            # algo cargado, sino fallback al admin_email.
            if bcc_activo:
                _bcc_val = (bcc_clientes_input or "").strip()
                if not _bcc_val:
                    _bcc_val = admin_email.strip()
            else:
                _bcc_val = ""
            cfg_nuevo = {
                "host": smtp_host.strip(),
                "port": int(smtp_port),
                "user": smtp_user.strip(),
                "password": smtp_password,
                "from_email": smtp_from_email.strip(),
                "from_name": smtp_from_name.strip(),
                "use_ssl": smtp_use_ssl,
                "use_tls": smtp_use_tls,
                "admin_email": admin_email.strip(),
                "bcc_clientes": _bcc_val,
                # IMAP separado (si está cargado)
                "imap_host": imap_host.strip() if imap_host else "",
                "imap_user": imap_user.strip() if imap_user else "",
                "imap_password": imap_password if imap_password else "",
            }
            ae.guardar_config_smtp(cfg_nuevo)
            st.success("✅ Configuración guardada en data/smtp_config.json")

        if col_b2.button("📨 Enviar email de prueba",
                           key="smtp_test_btn"):
            cfg_temp = ae.cargar_config_smtp() or {}
            if not cfg_temp.get("host") or not cfg_temp.get("user") or \
                    not cfg_temp.get("password"):
                cfg_temp = {
                    "host": smtp_host.strip(),
                    "port": int(smtp_port),
                    "user": smtp_user.strip(),
                    "password": smtp_password,
                    "from_email": smtp_from_email.strip(),
                    "from_name": smtp_from_name.strip(),
                    "use_ssl": smtp_use_ssl,
                    "use_tls": smtp_use_tls,
                }
            destino_test = (cfg_temp.get("admin_email") or
                              admin_email.strip() or
                              cfg_temp.get("from_email") or
                              smtp_from_email.strip())
            with st.spinner(f"Enviando prueba a {destino_test}..."):
                ok, msg = ae.enviar_email_prueba(cfg_temp, destino_test)
            if ok:
                st.success(f"✅ {msg} — revisá tu bandeja de entrada de "
                            f"`{destino_test}`")
            else:
                st.error(f"❌ {msg}")

        st.divider()
        st.markdown("##### 🧪 Simulador — ver cómo se ve una alerta real")
        st.caption(
            "Genera un email completo con datos falsos para que veas el "
            "diseño antes de que llegue una alerta real al campo."
        )

        col_sim1, col_sim2 = st.columns([2, 1])
        with col_sim1:
            sim_escenario = st.selectbox(
                "Escenario climático",
                [
                    "🌡️ Estrés calórico extremo (THI 84+, verano)",
                    "🥶 Helada severa (-5°C, invierno)",
                    "💨 Frente frío con viento (T° 2°C, viento 50 km/h)",
                    "🌧️ Lluvia y barro (acumulado 90 mm 7 días)",
                    "📋 Día normal (sin alertas, solo resumen)",
                ],
                key="sim_email_esc",
            )
        with col_sim2:
            st.markdown("")
            st.markdown("")
            sim_btn = st.button("🚀 Enviar email simulado",
                                  type="primary", key="sim_email_btn")

        if sim_btn:
            # Preferir la config guardada (sobrevive recargas);
            # si está vacía, usar lo que hay en el form
            cfg_temp = ae.cargar_config_smtp() or {}
            if not cfg_temp.get("host") or not cfg_temp.get("user") or \
                    not cfg_temp.get("password"):
                cfg_temp = {
                    "host": smtp_host.strip(),
                    "port": int(smtp_port),
                    "user": smtp_user.strip(),
                    "password": smtp_password,
                    "from_email": smtp_from_email.strip(),
                    "from_name": smtp_from_name.strip(),
                    "use_ssl": smtp_use_ssl,
                    "use_tls": smtp_use_tls,
                }
            destino = (cfg_temp.get("admin_email") or
                        admin_email.strip() or
                        cfg_temp.get("from_email") or
                        smtp_from_email.strip())

            # Datos falsos del cliente
            cliente_sim = {
                "nombre": "Establecimiento PRUEBA (simulación)",
                "establecimiento": "La Esperanza",
                "localidad": "Catriló, La Pampa",
            }

            # Construir alertas según escenario
            if "calórico" in sim_escenario:
                clima_actual = {"temp_c": 36.2, "humedad_pct": 68,
                                 "thi": 86, "thi_estado": "🔴 Estrés severo"}
                # Usar la nueva interfaz oficial: WhatsApp + Email
                from src.clima import evaluar_y_componer_mensajes
                resultado = evaluar_y_componer_mensajes(
                    clima={
                        "thi": 86, "thi_proyectado": [82, 84, 86],
                        "viento_kmh": 8, "min_nocturna": 24,
                        "temperatura": 35, "lluvia_mm": 0,
                    },
                    ambiente={"sombra_m2_cab": 3, "barro": False},
                    historial={"horas_thi_alto_ayer": 5,
                                "dias_consecutivos_calor": 2},
                    categoria="novillito",
                )
                # Extraer la línea de RESUMEN como descripción corta
                desc_corta = ""
                lineas_email = resultado["email"].splitlines()
                for i_l, ln in enumerate(lineas_email):
                    if "RESUMEN" in ln and "**" in ln:
                        # La descripción es la línea siguiente que no esté vacía
                        for j in range(i_l + 1, len(lineas_email)):
                            if lineas_email[j].strip():
                                desc_corta = lineas_email[j].strip()
                                break
                        break
                if not desc_corta:
                    desc_corta = (
                        f"Riesgo {resultado['nivel']} de "
                        f"{resultado['tipo']}."
                    )
                alertas_sim = [{
                    "lote": "L-101", "categoria": "novillito 320 kg",
                    "alertas": [
                        {"severidad": "critica",
                         "tipo": resultado.get("tipo", ""),
                         "nivel": resultado.get("nivel", ""),
                         "titulo": f"{resultado['tipo'].upper()} — "
                                    f"{resultado['nivel'].upper()} "
                                    f"(THI {resultado['thi_ajustado']:.0f})",
                         "descripcion": "",
                         "accion": resultado["email"]},
                    ],
                }]
            elif "Helada" in sim_escenario:
                clima_actual = {"temp_c": -5.0, "humedad_pct": 90,
                                 "thi": 22, "thi_estado": "❄️ Frío severo"}
                from src.clima import evaluar_y_componer_mensajes
                # Helada con viento sur fuerte y barro escarchado:
                # temp -5 (+2) + viento 25 (+2) + lluvia 6 (+1) + barro (+2) = 7 → crítico
                resultado = evaluar_y_componer_mensajes(
                    clima={"thi": 22, "viento_kmh": 25,
                            "temperatura": -5, "lluvia_mm": 6},
                    ambiente={"barro": True},
                    historial={},
                    categoria="vaquillona",
                )
                desc_corta = ""
                for ln in resultado["email"].splitlines():
                    if ln.strip() and "**" not in ln:
                        desc_corta = ln.strip()
                        break
                alertas_sim = [{
                    "lote": "L-202", "categoria": "vaquillona 280 kg",
                    "alertas": [
                        {"severidad": "critica",
                         "tipo": resultado.get("tipo", ""),
                         "nivel": resultado.get("nivel", ""),
                         "titulo": f"{resultado['tipo'].upper()} — "
                                    f"{resultado['nivel'].upper()}",
                         "descripcion": "",
                         "accion": resultado["email"]},
                    ],
                }]
            elif "frío con viento" in sim_escenario:
                clima_actual = {"temp_c": 2.1, "humedad_pct": 75,
                                 "thi": 38, "thi_estado": "❄️ Frío con viento"}
                from src.clima import evaluar_y_componer_mensajes
                resultado = evaluar_y_componer_mensajes(
                    clima={"thi": 38, "viento_kmh": 50,
                            "temperatura": 2, "lluvia_mm": 0},
                    ambiente={"barro": False},
                    historial={},
                    categoria="ternero",
                )
                desc_corta = ""
                for ln in resultado["email"].splitlines():
                    if ln.strip() and "**" not in ln:
                        desc_corta = ln.strip()
                        break
                alertas_sim = [{
                    "lote": "L-303", "categoria": "ternero destete 180 kg",
                    "alertas": [
                        {"severidad": "critica",
                         "tipo": resultado.get("tipo", ""),
                         "nivel": resultado.get("nivel", ""),
                         "titulo": f"{resultado['tipo'].upper()} — "
                                    f"{resultado['nivel'].upper()}",
                         "descripcion": "",
                         "accion": resultado["email"]},
                    ],
                }]
            elif "Lluvia" in sim_escenario:
                clima_actual = {"temp_c": 9.0, "humedad_pct": 95,
                                 "thi": 50, "thi_estado": "🟢 Sin estrés"}
                from src.clima import evaluar_y_componer_mensajes
                resultado = evaluar_y_componer_mensajes(
                    clima={"thi": 50, "viento_kmh": 22,
                            "temperatura": 9, "lluvia_mm": 13},
                    ambiente={"barro": True},
                    historial={},
                    categoria="novillo",
                )
                desc_corta = ""
                for ln in resultado["email"].splitlines():
                    if ln.strip() and "**" not in ln:
                        desc_corta = ln.strip()
                        break
                severidad_nivel = "critica" if resultado["nivel"] == "critico" \
                    else ("warning" if resultado["nivel"] == "moderado"
                            else "info")
                alertas_sim = [{
                    "lote": "L-404", "categoria": "novillo 380 kg",
                    "alertas": [
                        {"severidad": severidad_nivel,
                         "tipo": resultado.get("tipo", ""),
                         "nivel": resultado.get("nivel", ""),
                         "titulo": f"{resultado['tipo'].upper()} — "
                                    f"{resultado['nivel'].upper()}",
                         "descripcion": "",
                         "accion": resultado["email"]},
                    ],
                }]
            else:
                clima_actual = {"temp_c": 22.0, "humedad_pct": 55,
                                 "thi": 68, "thi_estado": "🟢 Sin estrés"}
                alertas_sim = []

            # SMN simulado
            smn_sim = {
                "estacion": {"nombre": "Anguil INTA", "distancia_km": 8.5,
                              "lat": -36.55, "lon": -63.97},
                "observacion": {
                    "temp_c": clima_actual["temp_c"],
                    "humedad_pct": clima_actual["humedad_pct"],
                    "viento_kmh": 18,
                    "descripcion": "Datos simulados",
                },
                "fuente": "SIMULACIÓN",
                "url_publica": "",
            }

            subject, html, text = ae.componer_alerta_diaria(
                cliente_sim, alertas_sim, smn_sim, clima_actual,
            )
            subject = "🧪 [SIMULACIÓN] " + subject

            with st.spinner(f"Enviando simulación a {destino}..."):
                ok, msg = ae.enviar_email(cfg_temp, [destino], subject, html, text)

            if ok:
                st.success(
                    f"✅ Simulación enviada a `{destino}`. "
                    f"Abrí tu mail y vas a ver exactamente cómo se va a ver "
                    f"la alerta real."
                )
            else:
                st.error(f"❌ {msg}")

        st.divider()
        st.markdown("##### Programar el envío diario")
        st.caption(
            "Una vez configurado el SMTP, instalá la tarea automática para "
            "que corra todos los días a las 08:00 sin intervención tuya."
        )

        st.code(
            "# Desde la terminal, ejecutá una sola vez:\n"
            "cd \"/Users/hms/Documents/Claude/Projects/determinacion de pesos y conteo bovinos utilizando drone\"\n"
            "bash scripts/instalar_cron.sh\n",
            language="bash",
        )

        with st.expander("¿Querés probarlo ahora sin esperar a las 08:00?"):
            st.code(
                "# Test sin enviar (muestra qué haría):\n"
                "python3 scripts/alertas_diarias.py --dry-run\n\n"
                "# Envío real ahora mismo:\n"
                "python3 scripts/alertas_diarias.py\n\n"
                "# Solo un cliente puntual:\n"
                "python3 scripts/alertas_diarias.py --solo-cliente \"guinter\"\n",
                language="bash",
            )

        # Mostrar últimos envíos
        st.divider()
        st.markdown("##### Últimos envíos")
        try:
            with db.get_conn() as conn:
                rows = conn.execute(
                    "SELECT fecha, destinatario, asunto, n_alertas, estado, error "
                    "FROM alertas_enviadas ORDER BY fecha_creacion DESC LIMIT 20"
                ).fetchall()
                if rows:
                    df_envios = pd.DataFrame([dict(r) for r in rows])
                    st.dataframe(df_envios, width="stretch", hide_index=True)
                else:
                    st.info("Todavía no se enviaron alertas. "
                             "Probá con `python3 scripts/alertas_diarias.py --dry-run` "
                             "o esperá al primer envío automático.")
        except Exception as e:
            st.warning(f"No se pudo leer el historial: {e}")

    # ---- Sub-tab: WhatsApp ----
    with sub_whatsapp:
        from src import whatsapp as wa

        st.markdown("### 💬 Alertas por WhatsApp (Twilio)")
        st.caption(
            "Las alertas críticas se mandan por WhatsApp al toque, además del "
            "email matinal. Usa Twilio — fácil de configurar, ~USD 1-3/mes."
        )

        with st.expander("📋 Cómo arrancar con Twilio Sandbox (gratis, 5 minutos)",
                          expanded=True):
            st.markdown("""
**1. Crear cuenta gratis en Twilio**

Andá a [twilio.com/try-twilio](https://www.twilio.com/try-twilio) y registrate. Te dan crédito gratis.

**2. Activar el WhatsApp Sandbox**

En la consola Twilio: **Messaging → Try it out → Send a WhatsApp message**.
Vas a ver:
- Un número de WhatsApp Twilio (ej: `+1 415 523 8886`)
- Una palabra clave del estilo `join correct-horse`

**3. Activar tu número**

Desde tu WhatsApp personal, mandá `join correct-horse` (la palabra clave que te dio Twilio) al número de Twilio. Te confirma que estás conectado al sandbox.

**4. Cargar credenciales acá abajo**

En el dashboard de Twilio copiá:
- **Account SID** (empieza con `AC...`)
- **Auth Token** (apretá el ojo para verlo)
- **From number** (el número del sandbox, formato `+14155238886`)

Pegalos abajo, guardá, y apretá "Enviar test". Te tiene que llegar a tu WhatsApp.

**5. Que tus clientes reciban**

En el sandbox cada cliente que quiera recibir alertas también tiene que mandar el `join correct-horse` al número Twilio una vez. Sirve para probar con 1-2 clientes.

**Para producción real (sin necesidad de "join")**

Cuando estés cómodo, en Twilio: **Messaging → Senders → WhatsApp** comprás un número y lo verificás como business sender. Eso te permite mandar a clientes sin que ellos hagan nada. Costo: alta única + ~USD 0.005-0.05 por mensaje.
            """)

        cfg_wa_actual = wa.cargar_config() or {}

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            wa_account_sid = st.text_input(
                "Account SID",
                value=cfg_wa_actual.get("account_sid", ""),
                placeholder="ACxxxxxxxxxxxxxxxx",
                key="wa_account_sid",
            )
            wa_auth_token = st.text_input(
                "Auth Token",
                type="password",
                value=cfg_wa_actual.get("auth_token", ""),
                key="wa_auth_token",
            )
        with col_w2:
            wa_from = st.text_input(
                "From (número Twilio WhatsApp)",
                value=cfg_wa_actual.get("from_number", "+14155238886"),
                placeholder="+14155238886",
                help="En sandbox: +14155238886. En producción: tu número Twilio.",
                key="wa_from",
            )
            wa_admin_phone = st.text_input(
                "Tu WhatsApp (recibe las alertas críticas)",
                value=cfg_wa_actual.get("admin_phone", "+54 9 2954 51-7407"),
                placeholder="+54 9 2954 51-7407",
                key="wa_admin_phone",
            )

        wa_modo_sandbox = st.checkbox(
            "Estoy usando el Sandbox de Twilio (modo gratis para testear)",
            value=bool(cfg_wa_actual.get("modo_sandbox", True)),
            help="En sandbox los destinatarios deben mandar 'join <palabra>' "
                  "primero. Desactivá esto cuando ya tengas tu número Twilio "
                  "verificado para producción.",
            key="wa_modo_sandbox",
        )

        col_wb1, col_wb2 = st.columns([1, 1])

        if col_wb1.button("💾 Guardar configuración",
                            type="primary", key="wa_save"):
            cfg_nuevo = {
                "provider": "twilio",
                "account_sid": wa_account_sid.strip(),
                "auth_token": wa_auth_token.strip(),
                "from_number": wa_from.strip(),
                "admin_phone": wa_admin_phone.strip(),
                "modo_sandbox": wa_modo_sandbox,
            }
            wa.guardar_config(cfg_nuevo)
            st.success("✅ Configuración Twilio guardada")

        if col_wb2.button("📨 Enviar test a mi WhatsApp",
                            key="wa_test"):
            cfg_temp = {
                "account_sid": wa_account_sid.strip(),
                "auth_token": wa_auth_token.strip(),
                "from_number": wa_from.strip(),
                "modo_sandbox": wa_modo_sandbox,
            }
            destino = wa_admin_phone.strip()
            with st.spinner(f"Enviando test a {destino}..."):
                ok, msg = wa.enviar_test(cfg_temp, destino)
            if ok:
                st.success(f"✅ {msg} — revisá tu WhatsApp en {destino}")
            else:
                st.error(f"❌ {msg}")
                if "join" in msg.lower():
                    st.warning(
                        "👉 Mandá la palabra `join <código>` desde tu "
                        "WhatsApp al número Twilio sandbox primero. "
                        "Lo encontrás en Twilio Console → Messaging → "
                        "Try it out → Send a WhatsApp message."
                    )

        st.divider()
        st.markdown("##### 🧪 Simulador — ver cómo se ve un WhatsApp real")
        st.caption(
            "Manda un WhatsApp con datos falsos para que veas cómo va a "
            "llegar una alerta real cuando se dispare en el campo."
        )

        col_ws1, col_ws2 = st.columns([2, 1])
        with col_ws1:
            sim_wa_esc = st.selectbox(
                "Escenario",
                [
                    "🌡️ Estrés calórico extremo",
                    "🥶 Helada severa -5°C",
                    "💨 Frente frío con viento",
                    "📋 Resumen matinal (sin alertas)",
                ],
                key="sim_wa_esc",
            )
        with col_ws2:
            st.markdown("")
            st.markdown("")
            sim_wa_btn = st.button("🚀 Enviar WhatsApp simulado",
                                     type="primary", key="sim_wa_btn")

        if sim_wa_btn:
            cfg_temp = {
                "account_sid": wa_account_sid.strip(),
                "auth_token": wa_auth_token.strip(),
                "from_number": wa_from.strip(),
                "modo_sandbox": wa_modo_sandbox,
            }
            destino_wa = wa_admin_phone.strip()

            from src.clima import evaluar_y_componer_mensajes

            if "calórico" in sim_wa_esc:
                r = evaluar_y_componer_mensajes(
                    clima={"thi": 86, "thi_proyectado": [82, 84, 86],
                            "viento_kmh": 8, "min_nocturna": 24,
                            "temperatura": 35, "lluvia_mm": 0},
                    ambiente={"sombra_m2_cab": 3, "barro": False},
                    historial={"horas_thi_alto_ayer": 5,
                                "dias_consecutivos_calor": 2},
                    categoria="novillito",
                )
                ok, msg = wa.enviar_texto(cfg_temp, destino_wa, r["whatsapp"])
            elif "Helada" in sim_wa_esc:
                r = evaluar_y_componer_mensajes(
                    clima={"thi": 30, "viento_kmh": 5,
                            "temperatura": -3, "lluvia_mm": 0},
                    ambiente={"barro": False},
                    historial={},
                    categoria="vaquillona",
                )
                ok, msg = wa.enviar_texto(cfg_temp, destino_wa, r["whatsapp"])
            elif "Frente frío" in sim_wa_esc:
                r = evaluar_y_componer_mensajes(
                    clima={"thi": 38, "viento_kmh": 50,
                            "temperatura": 2, "lluvia_mm": 8},
                    ambiente={"barro": True},
                    historial={},
                    categoria="ternero",
                )
                ok, msg = wa.enviar_texto(cfg_temp, destino_wa, r["whatsapp"])
            else:
                ok, msg = wa.enviar_resumen_diario(
                    cfg_temp, destino_wa,
                    n_clientes=5, n_criticas=0, n_warning=0,
                )

            if ok:
                st.success(f"✅ Simulación enviada a `{destino_wa}`. "
                            f"Mirá tu WhatsApp.")
            else:
                st.error(f"❌ {msg}")

        st.divider()
        st.markdown("##### Activar el envío automático")
        st.caption(
            "Una sola vez en la terminal. Después corre solo todos los días "
            "08:00 (resumen) + cada 1 h (críticas instantáneas)."
        )
        st.code("bash scripts/instalar_cron.sh", language="bash")

        with st.expander("Probar sin esperar"):
            st.code(
                "# Ver qué mandaría sin enviar nada:\n"
                "python3 scripts/alertas_criticas.py --dry-run\n\n"
                "# Enviar de verdad ahora:\n"
                "python3 scripts/alertas_criticas.py",
                language="bash",
            )

        # Historial WhatsApp
        st.divider()
        st.markdown("##### Últimos WhatsApp enviados")
        try:
            with db.get_conn() as conn:
                rows_wa = conn.execute(
                    "SELECT fecha_creacion, destinatario, mensaje, estado, error "
                    "FROM alertas_whatsapp_enviadas "
                    "ORDER BY fecha_creacion DESC LIMIT 20"
                ).fetchall()
                if rows_wa:
                    df_wa = pd.DataFrame([dict(r) for r in rows_wa])
                    st.dataframe(df_wa, width="stretch",
                                  hide_index=True)
                else:
                    st.info("Todavía no se enviaron WhatsApp. Probá el botón "
                             "de test arriba.")
        except Exception as e:
            st.warning(f"No se pudo leer historial WhatsApp: {e}")


with tab_train:
    st.markdown(
        "### 🎓 Entrenamiento avanzado del modelo de drone\n"
        "Para **alta densidad** y **precisión profesional** hay que entrenar "
        "el modelo con tus videos. Acá tenés el flujo completo guiado."
    )

    sub_extr, sub_calib, sub_modelo, sub_guia = st.tabs([
        "1️⃣ Extraer frames",
        "2️⃣ Calibrar pesos con balanza",
        "3️⃣ Importar modelo entrenado",
        "📋 Guía completa",
    ])

    # ----- 1) Extracción de frames -----
    with sub_extr:
        st.markdown(
            "**Subí uno o varios videos del drone** (preferentemente con "
            "tropas densas y/o casos difíciles). El sistema saca frames "
            "para que vos los etiquetes en Roboflow."
        )

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            fps_obj = st.select_slider(
                "Frames por segundo a extraer",
                options=[0.5, 1.0, 2.0, 5.0],
                value=1.0,
                help="Más fps = más imágenes (más etiquetado, mejor modelo)",
            )
        with col_e2:
            max_por_video = st.number_input(
                "Máximo frames por video", min_value=20, max_value=500,
                value=100, step=10,
            )

        videos_train = st.file_uploader(
            "Videos (podés subir varios)",
            type=["mp4", "mov", "avi"],
            accept_multiple_files=True,
            key="videos_train",
        )

        if videos_train and st.button(
            "🎬 Extraer frames", type="primary",
        ):
            todos_frames = []
            barra = st.progress(0)
            for i, v in enumerate(videos_train):
                with st.spinner(f"Procesando {v.name}..."):
                    frames_v = training.extraer_frames_de_video(
                        v.getvalue(),
                        fps_objetivo=fps_obj,
                        max_frames=int(max_por_video),
                    )
                    # Renombrar para incluir el nombre del video
                    for f in frames_v:
                        f["nombre"] = (
                            f"{Path(v.name).stem}_" + f["nombre"]
                        )
                    todos_frames.extend(frames_v)
                barra.progress((i + 1) / len(videos_train))
            barra.empty()

            if todos_frames:
                st.success(
                    f"✅ {len(todos_frames)} frames extraídos de "
                    f"{len(videos_train)} video(s)"
                )
                # Generar ZIP
                zip_bytes = training.crear_zip_dataset(
                    todos_frames, nombre_proyecto="bovinos_drone",
                )
                st.download_button(
                    "📦 Descargar ZIP listo para Roboflow",
                    data=zip_bytes,
                    file_name=f"dataset_bovinos_{datetime.now():%Y%m%d}.zip",
                    mime="application/zip",
                    type="primary",
                )

                # Preview
                st.markdown("##### Preview de los primeros frames")
                cols_p = st.columns(4)
                for j, f in enumerate(todos_frames[:8]):
                    with cols_p[j % 4]:
                        st.image(f["bytes"], caption=f"t={f['segundo']:.1f}s",
                                 use_column_width=True)

    # ----- 2) Calibración con balanza -----
    with sub_calib:
        st.markdown(
            "**Subí los pesos de balanza vs los pesos que dio la app** "
            "para que calibre automáticamente el `ajuste_fino`. "
            "Cuantos más pares, más preciso."
        )
        st.caption(
            "El CSV debe tener 2 columnas: `peso_real_kg` (balanza) y "
            "`peso_app_kg` (lo que dio la app antes de cualquier ajuste). "
            "Mínimo 5 pares; ideal 15-30."
        )

        # Plantilla CSV descargable
        plantilla_csv = pd.DataFrame({
            "peso_real_kg": [285, 310, 295, 320, 280],
            "peso_app_kg": [275, 298, 287, 308, 270],
        })
        st.download_button(
            "📥 Descargar plantilla CSV",
            data=plantilla_csv.to_csv(index=False).encode("utf-8"),
            file_name="plantilla_calibracion.csv",
            mime="text/csv",
        )

        archivo_cal = st.file_uploader(
            "Subir CSV de pesadas comparativas",
            type=["csv"], key="csv_calibracion",
        )

        if archivo_cal:
            try:
                df_cal = pd.read_csv(archivo_cal)
                if "peso_real_kg" not in df_cal.columns or \
                        "peso_app_kg" not in df_cal.columns:
                    st.error("Faltan columnas: peso_real_kg, peso_app_kg")
                else:
                    pares = list(zip(
                        df_cal["peso_real_kg"].tolist(),
                        df_cal["peso_app_kg"].tolist(),
                    ))
                    aj_actual = st.session_state.get("ajuste_fino_value", 1.0)
                    res = training.calibrar_ajuste_fino(
                        pares, ajuste_actual=aj_actual,
                    )

                    st.success(
                        f"✅ Calibración con {res.n_muestras} pares completada"
                    )

                    cm1, cm2, cm3 = st.columns(3)
                    cm1.metric(
                        "Ajuste fino óptimo",
                        f"{res.ajuste_fino_optimo:.3f}",
                        f"vs actual {aj_actual:.2f}",
                    )
                    cm2.metric(
                        "MAPE óptimo",
                        f"{res.mape_optimo:.1f}%",
                        f"{res.mape_actual - res.mape_optimo:+.1f}% vs actual",
                    )
                    cm3.metric(
                        "R²", f"{res.r2:.3f}",
                        "calidad del ajuste",
                    )

                    # Tabla comparativa
                    st.markdown("##### Pesos antes / después de calibrar")
                    df_comp = pd.DataFrame({
                        "Real (kg)": [round(x, 1) for x in res.pesos_reales],
                        "App original (kg)": [round(x, 1) for x in res.pesos_app],
                        "App calibrada (kg)": [round(x, 1) for x in res.pesos_corregidos],
                        "Error post-calib (kg)": [
                            round(c - r, 1)
                            for r, c in zip(res.pesos_reales, res.pesos_corregidos)
                        ],
                    })
                    st.dataframe(df_comp, hide_index=True, width="stretch")

                    if res.mape_optimo < 5:
                        st.success(
                            f"🎯 Excelente — MAPE {res.mape_optimo:.1f}% < 5%. "
                            f"Aplicá ajuste_fino = {res.ajuste_fino_optimo:.3f} "
                            "en la sidebar."
                        )
                    elif res.mape_optimo < 8:
                        st.info(
                            f"✅ Buena calibración — MAPE {res.mape_optimo:.1f}%. "
                            "Para bajar más, sumá más pares o entrená el "
                            "modelo (paso 3)."
                        )
                    else:
                        st.warning(
                            f"⚠️ MAPE {res.mape_optimo:.1f}% — todavía alto. "
                            "Probablemente el modelo de detección está "
                            "subestimando o sobreestimando áreas. "
                            "Conviene fine-tuning con tus videos."
                        )

                    if abs(res.sesgo_kg) > 5:
                        st.warning(
                            f"⚠️ Sesgo sistemático de {res.sesgo_kg:+.1f} kg. "
                            "El modelo está siempre por encima/debajo. "
                            "Calibrar ayuda pero el fine-tuning es la solución real."
                        )

                    st.caption(
                        f"Rango de confianza individual (95%): "
                        f"{res.rangos_confianza[0]:.0f} a "
                        f"{res.rangos_confianza[1]:.0f} kg de error por animal"
                    )

            except Exception as e:
                st.error(f"Error procesando CSV: {e}")

    # ----- 3) Importar modelo entrenado -----
    with sub_modelo:
        st.markdown(
            "Si ya entrenaste un modelo con Colab (paso de la guía), "
            "**subí el archivo `.pt` acá** para que la app lo use."
        )

        modelo_subido = st.file_uploader(
            "Subir modelo .pt entrenado", type=["pt"],
            key="upload_pt",
        )
        if modelo_subido:
            Path("models").mkdir(exist_ok=True)
            modelo_dest = Path(f"models/{modelo_subido.name}")
            with open(modelo_dest, "wb") as f:
                f.write(modelo_subido.getvalue())

            with st.spinner("Validando modelo..."):
                info = training.validar_modelo_yolo(modelo_dest)

            if info["valido"]:
                st.success(
                    f"✅ Modelo cargado: {modelo_dest.name} "
                    f"({info['tamano_mb']:.1f} MB)"
                )
                st.markdown(f"**Clases**: {', '.join(info.get('clases', [])) or 'desconocidas'}")
                st.markdown(f"**Tarea**: {info.get('task', '?')}")
                st.info(
                    f"Para usarlo: en la sidebar, en el campo 'Modelo YOLO', "
                    f"escribí `{modelo_dest}` (en lugar de yolov8m.pt)"
                )
            else:
                st.error(f"❌ Modelo no válido: {info.get('error', 'desconocido')}")

        # Listar modelos importados
        modelos_locales = list(Path("models").glob("*.pt")) if Path("models").exists() else []
        if modelos_locales:
            st.markdown("##### Modelos disponibles localmente")
            for m in modelos_locales:
                col_m1, col_m2 = st.columns([3, 1])
                col_m1.markdown(f"📦 **{m.name}** ({m.stat().st_size / 1024 / 1024:.1f} MB)")
                if col_m2.button("🗑️", key=f"del_model_{m.name}"):
                    m.unlink()
                    st.rerun()

    # ----- 4) Guía completa -----
    with sub_guia:
        st.markdown("""
### 📋 Flujo completo para tener un modelo profesional

#### Etapa A: Generar dataset (1-2 hs total)

1. **Filmar variedad de casos**: tropas chicas, tropas densas, distintas razas, distintos pisos. Cuanto más variado, mejor el modelo.
2. **Extraer frames** (paso 1️⃣ arriba): subí 5-10 videos, extraé ~500-1000 frames totales.
3. **Subí el ZIP a [Roboflow](https://roboflow.com)** (gratis hasta 10k imágenes):
   - Nuevo proyecto → Object Detection → YOLOv8
   - Drag & drop el ZIP
   - **Auto-Label** asistido: te pre-dibuja las cajas con clase "cow"/"sheep"
   - Vos sólo corregís, agregás los que faltaron, sacás falsos positivos
   - **Importante**: usar UNA SOLA clase llamada "bovino"

#### Etapa B: Entrenar (1-2 hs en Colab gratis con GPU)

1. **Bajá el notebook** desde el botón abajo
2. Subilo a [Google Colab](https://colab.research.google.com)
3. **Activá GPU**: Entorno de ejecución → Cambiar tipo → T4 GPU (gratis)
4. Subí el ZIP del dataset etiquetado de Roboflow al panel "Files"
5. Ejecutá las celdas en orden
6. Al final descargás el `best.pt`

#### Etapa C: Usar el modelo en la app

1. Volvé al paso 3️⃣ "Importar modelo entrenado"
2. Subí el `best.pt`
3. En la sidebar, en "Modelo YOLO", elegí o escribí `models/best.pt`
4. ¡Listo! La app usa tu modelo.

#### Etapa D: Calibrar pesos contra balanza

1. Pasá un lote por la app (con tu modelo nuevo) y por balanza real
2. Anotá ambos pesos individuales en CSV (al menos 15-20 pares)
3. Subí el CSV en el paso 2️⃣ "Calibrar pesos con balanza"
4. La app calcula automáticamente el `ajuste_fino` óptimo
5. Anotalo y dejalo fijo en el slider de la sidebar

#### 🎯 Resultado esperado

| Métrica | Sin entrenar | Con fine-tuning + calibración |
|---------|--------------|-------------------------------|
| Recall conteo (tropa densa) | 50-60% | 92-98% |
| MAPE peso individual | 8-15% | 2-4% |
| MAPE peso promedio del lote | 5-8% | <2% |

#### Costo

| Recurso | Costo |
|---------|-------|
| Roboflow (etiquetado) | Gratis hasta 10k imgs |
| Google Colab (entrenar) | Gratis con GPU T4 |
| **TOTAL** | **$0** + tu tiempo |
""")

        # Botón para descargar el notebook Colab
        st.markdown("---")
        # Solo tiene sentido si las libs de drone (ultralytics/cv2) están
        # disponibles: en Streamlit Cloud (free tier) las deshabilitamos
        # por límite de RAM, así que ocultamos el botón (evita crash con
        # data=None cuando training es _DroneStub).
        if _DRONE_LIBS_OK:
            notebook_bytes = training.generar_notebook_colab()
            st.download_button(
                "📥 Descargar notebook Colab listo",
                data=notebook_bytes,
                file_name="finetune_bovinos.ipynb",
                mime="application/json",
                type="primary",
            )
        else:
            st.info(
                "🚫 Descarga del notebook Colab no disponible en el "
                "deploy cloud. Requiere `ultralytics + torch + opencv`, "
                "que superan el límite de RAM del free tier. Usá esta "
                "pestaña desde tu Mac local."
            )


# ----------------------------- AYUDA ----------------------------------
with tab_help:
    st.markdown(
        """
### Cómo usar la app

1. **Captura con drone** — vuelo cenital (90°) a ~10 m de altura, 4K @ 30 fps.
   Incluí en el encuadre una **referencia conocida** en el piso (un marcador
   ArUco impreso de 1,02 m × 1,02 m, o un cuadrado de cinta de color sólido).

2. **Subí la imagen o video** en la pestaña correspondiente.

3. **Configurá la raza predominante** en la barra lateral. Eso ajusta el
   factor de corrección del modelo de peso.

4. **Calibración personalizada**: si ya tenés un dataset propio (imagen +
   peso real), corré el script de calibración:

   ```bash
   python scripts/calibrate_weight.py --dataset data/calibracion.csv \\
       --output models/weight_model.json
   ```

   Luego, en la barra lateral, activá *“Usar modelo de peso calibrado”* y
   subí el JSON resultante.

### Para alcanzar el <5 % de error

- Usá el modelo `yolov8m-seg.pt` (segmentación) — el área proyectada es
  mucho más fiel que el bounding box.
- Calibrá los coeficientes con **al menos 30 muestras** de tu rodeo
  (área observada vs peso real de balanza).
- Mantené altura de vuelo y ángulo constantes.
- Asegurate que la referencia de 1,02 m esté siempre visible y plana.

### Estructura del proyecto

```
app.py                   ← UI Streamlit
config.yaml              ← Parámetros
src/
  calibration.py         ← Detección de la referencia
  detector.py            ← YOLO (cow class)
  weight_estimator.py    ← Modelo Peso = a·Area^b · factor_raza + c
  processor.py           ← Pipeline imagen / video con tracking
scripts/
  calibrate_weight.py    ← Ajuste de coeficientes con tu dataset
data/
  calibracion_template.csv
```
        """
    )
