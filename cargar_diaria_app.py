"""Mini-app Streamlit pública para que el encargado del lote cargue
cuánto puso al comedero ese día.

Cómo se usa:
    1. Correr en un puerto separado del app principal:
           streamlit run cargar_diaria_app.py --server.port 8502
    2. Exponer ese puerto con ngrok o Cloudflare Tunnel:
           ngrok http 8502
       (URL pública tipo https://abc.ngrok.app)
    3. Configurar la URL base en config (usado por el cron 17:00).

El cron `whatsapp_pedido_carga.py` manda un WhatsApp al encargado con
un link tipo `https://abc.ngrok.app/?token=7.20260525.d560a047`.
Esta app:
    - Valida el token HMAC (firma + ventana de 48 hs).
    - Trae la dieta vigente del lote.
    - Muestra un form con un input numérico por cada ingrediente.
    - Al enviar, registra la carga en cargas_silocomedero con
      tipo_carga='lineal_diario' y desglose por ingrediente.

Sin login. La seguridad viene del token firmado.
"""
from __future__ import annotations

import sys
from datetime import datetime, date
from pathlib import Path

import streamlit as st

# Permitir importar src desde el root del proyecto
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import database as db  # noqa: E402
from src import carga_diaria_token as tok  # noqa: E402
from src import stock_producto as sp  # noqa: E402


# ───── Config UI ─────
st.set_page_config(
    page_title="HMS — Carga diaria",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Branding compacto
st.markdown(
    """
    <style>
      header[data-testid="stHeader"] { background:transparent; }
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      h1 { color: #1B3E27; }
      .hms-card {
        background: #F4F8F1;
        border-left: 4px solid #1B3E27;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍽️ Carga del comedero — HMS")


# ───── Lectura del token ─────
qp = st.query_params
token_raw = qp.get("token") or qp.get("t") or ""
if isinstance(token_raw, list):
    token_raw = token_raw[0] if token_raw else ""

if not token_raw:
    st.error(
        "⚠️ Esta página se abre con un link que te llega por "
        "WhatsApp. Si llegaste acá por error, cerrá la pestaña."
    )
    st.stop()

ok, lote_id, fecha_token, err = tok.validar_token(token_raw)
if not ok:
    st.error(f"❌ {err}")
    st.caption(
        "Pedile al asesor (Mauricio) que te reenvíe el link de hoy."
    )
    st.stop()

# ───── Traer lote + dieta vigente ─────
lote = db.obtener_lote(lote_id)
if not lote:
    st.error("❌ El lote referenciado no existe.")
    st.stop()

# Cliente (para mostrar contexto)
cli = None
try:
    for c in db.listar_clientes():
        if c["id"] == lote["cliente_id"]:
            cli = c
            break
except Exception:
    pass

dietas = db.listar_dietas(lote_id)
fecha_iso = fecha_token.strftime("%Y-%m-%d")
dieta_vig = sp._dieta_vigente(dietas, fecha_iso) if dietas else None

if not dieta_vig:
    st.warning(
        "⚠️ No hay una dieta cargada para este lote todavía. "
        "Avisale al asesor antes de cargar."
    )
    st.stop()

# Cantidad de animales vigente (respeta movimientos)
cant_animales = db.cantidad_vigente_lote(lote_id, fecha_iso) or 0

# ───── Cabecera ─────
nombre_cli = cli.get("nombre") if cli else "—"
encargado = (lote.get("encargado_nombre") or "").strip() or None
saludo = f"Hola {encargado}" if encargado else "Hola"

st.markdown(
    f"""
    <div class="hms-card">
      {saludo}, te pasamos lo que tenés que cargar hoy al
      comedero del lote <strong>{lote.get('identificador', '?')}</strong>
      (cliente <strong>{nombre_cli}</strong>).
      <br><span style='color:#666; font-size:0.9em;'>
        Fecha: {fecha_iso} · {cant_animales} cab. ·
        Dieta vigente del {dieta_vig.get('fecha', '—')}
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ───── Lo que la dieta recomienda hoy ─────
st.markdown("### 📋 Lo que tendrías que cargar hoy")

composicion = dieta_vig.get("composicion") or []
ingredientes_no_libre = []
mezcla_total_esp = 0.0
for c in composicion:
    nombre = (c.get("nombre") or "").strip()
    if not nombre:
        continue
    es_libre = sp._es_a_discrecion(nombre)
    if es_libre:
        st.caption(
            f"_{nombre}: a libre disposición — no hace falta "
            "cargarlo por kg, solo asegurate que tengan disponible._"
        )
        continue
    kg_animal = float(c.get("kg_tal_cual") or 0)
    kg_lote = kg_animal * cant_animales
    ingredientes_no_libre.append({
        "nombre": nombre,
        "kg_animal": round(kg_animal, 2),
        "kg_lote": round(kg_lote, 1),
    })
    mezcla_total_esp += kg_lote

if not ingredientes_no_libre:
    st.warning("No hay ingredientes a cargar en esta dieta.")
    st.stop()

# Tabla con lo recomendado
import pandas as pd
df_rec = pd.DataFrame([
    {
        "Ingrediente": i["nombre"],
        "kg por animal": f"{i['kg_animal']:.2f}",
        "kg para el lote": f"{i['kg_lote']:.0f}",
    }
    for i in ingredientes_no_libre
])
st.dataframe(df_rec, hide_index=True, width="stretch")
st.caption(
    f"Mezcla total recomendada: "
    f"**{mezcla_total_esp:.0f} kg** para los {cant_animales} animales."
)

# ───── Form: lo que cargaste ─────
st.markdown("### ✍️ ¿Cuánto cargaste realmente?")
st.caption(
    "Anotá lo que efectivamente tiraste al comedero. Si no usaste un "
    "ingrediente, dejá el 0. El asesor compara con lo recomendado y "
    "te avisa si hay un desvío grande."
)

# Detectar si ya cargó hoy
ya_cargadas = db.listar_cargas_silocomedero(lote_id, limit=20)
ya_hoy = [
    c for c in ya_cargadas
    if (c.get("fecha_carga") or "")[:10] == fecha_iso
]
if ya_hoy:
    _ya_total = sum(float(c.get("kg_cargados") or 0) for c in ya_hoy)
    _ya_lineas = []
    for c in sorted(
        ya_hoy, key=lambda x: (x.get("hora_carga") or "00:00")
    ):
        _h = c.get("hora_carga") or "sin hora"
        _kg = float(c.get("kg_cargados") or 0)
        _ya_lineas.append(f"  · **{_h}** → {_kg:.0f} kg")
    _detalle_hoy = "\n".join(_ya_lineas)
    st.info(
        f"ℹ️ Ya tenés **{len(ya_hoy)} carga(s)** registrada(s) hoy "
        f"(total {_ya_total:.0f} kg):\n\n{_detalle_hoy}\n\n"
        "Si esta es otra comida del día, completá los kg de esta "
        "comida (no el total) y enviá. Si te equivocaste antes, "
        "avisale al asesor para que la borre."
    )

with st.form("carga_form", clear_on_submit=False):
    # Hora de la carga (default = ahora)
    from datetime import time as _time
    _hora_default = datetime.now().time().replace(second=0, microsecond=0)
    hora_input = st.time_input(
        "Hora de la carga",
        value=_hora_default,
        help=(
            "Si fueron 2 comidas en el día, registrá ésta con su "
            "hora y volvé a entrar al link más tarde para registrar "
            "la segunda."
        ),
    )

    inputs = {}
    for i in ingredientes_no_libre:
        # Default = kg_lote recomendado para facilitar (modificar el
        # valor solo si fue distinto).
        inputs[i["nombre"]] = st.number_input(
            f"{i['nombre']} (kg al lote)",
            min_value=0.0,
            max_value=100000.0,
            value=float(i["kg_lote"]),
            step=1.0,
            key=f"inp_{i['nombre']}",
            help=f"Esperado: {i['kg_lote']:.0f} kg",
        )

    obs = st.text_input(
        "Observaciones (opcional)",
        placeholder=(
            "Ej: 'mañana cargo doble porque mañana no vengo' o "
            "'estaba mojado'"
        ),
    )

    total_ingresado = sum(inputs.values())
    st.markdown(
        f"**Total ingresado: {total_ingresado:.0f} kg** "
        f"(recomendado: {mezcla_total_esp:.0f} kg)"
    )
    if mezcla_total_esp > 0:
        desv_pct = (
            (total_ingresado - mezcla_total_esp) / mezcla_total_esp
            * 100
        )
        if abs(desv_pct) <= 5:
            st.caption(f"🟢 Desvío {desv_pct:+.1f}% — dentro del plan.")
        elif abs(desv_pct) <= 10:
            st.caption(f"🟠 Desvío {desv_pct:+.1f}% — atención.")
        else:
            st.caption(f"🔴 Desvío {desv_pct:+.1f}% — diferencia grande.")

    submit = st.form_submit_button(
        "💾 Confirmar y enviar al asesor",
        type="primary",
    )
    if submit:
        desglose = [
            {"nombre": k, "kg": float(v)}
            for k, v in inputs.items() if v > 0
        ]
        total = sum(d["kg"] for d in desglose)
        if total <= 0:
            st.error(
                "❌ Tenés que cargar al menos un ingrediente "
                "con kg > 0."
            )
        else:
            try:
                _hora_str = hora_input.strftime("%H:%M")
                db.crear_carga_silocomedero(
                    lote_id=lote_id,
                    fecha_carga=fecha_iso,
                    kg_cargados=total,
                    detalles=obs or "",
                    tipo_carga="lineal_diario",
                    desglose_ingredientes=desglose,
                    dias_cubiertos=1,
                    hora_carga=_hora_str,
                )
                st.success(
                    f"✅ Listo, gracias. Quedó registrada una carga "
                    f"de {total:.0f} kg a las {_hora_str}. El asesor "
                    "ya tiene el dato."
                )
                st.caption(
                    "Si fue la primera comida del día, podés volver "
                    "a entrar al link más tarde para registrar la "
                    "segunda."
                )
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error guardando: {e}")
                st.caption("Avisale al asesor.")

st.divider()
st.caption(
    "🌱 HMS Nutrición Animal — Si tenés un problema con esta carga "
    "comunicate con Mauricio al **2954-517407**."
)
