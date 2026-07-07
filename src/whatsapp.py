"""
Cliente de WhatsApp vía Twilio API.

Twilio API Messages:
  POST https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT_SID}/Messages.json
  Auth: Basic (AccountSid, AuthToken)
  Body form-encoded: From=whatsapp:+...  To=whatsapp:+...  Body=...

Modos:
  - Sandbox (gratis, para testear): From="whatsapp:+14155238886", el destinatario
    debe haber mandado "join <palabra-clave>" al número sandbox antes.
  - Producción: From=número WhatsApp Business comprado/verificado en Twilio.

Config en data/whatsapp_config.json (gitignore):
  {
    "provider": "twilio",
    "account_sid": "ACxxxxxxxxxxxxxxxx",
    "auth_token": "xxxxxxxxxxxxxxxxxx",
    "from_number": "+14155238886",          # sandbox o tu número Twilio
    "admin_phone": "+5492954517407",
    "modo_sandbox": true                    # afecta los mensajes de ayuda
  }

Las firmas públicas (enviar_texto, enviar_alerta_critica, enviar_resumen_diario,
enviar_test, normalizar_telefono, clave_dedup, componer_*) se mantienen iguales
que cuando usábamos Meta, así los scripts no cambian.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CONFIG_PATH = Path("data/whatsapp_config.json")
TWILIO_BASE = "https://api.twilio.com/2010-04-01"
SANDBOX_FROM = "+14155238886"


# =====================================================================
# CONFIG
# =====================================================================

def cargar_config() -> Optional[Dict]:
    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def guardar_config(cfg: Dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def config_valida(cfg: Optional[Dict]) -> Tuple[bool, str]:
    if not cfg:
        return False, "No hay config WhatsApp cargada"
    requeridos = ("account_sid", "auth_token", "from_number")
    faltantes = [k for k in requeridos if not cfg.get(k)]
    if faltantes:
        return False, f"Faltan: {', '.join(faltantes)}"
    if not str(cfg["account_sid"]).startswith("AC"):
        return False, "Account SID debe empezar con 'AC'"
    return True, ""


# =====================================================================
# TELÉFONO
# =====================================================================

def normalizar_telefono(tel: str) -> Optional[str]:
    """Convierte cualquier número argentino a formato E.164 con '+'.

    Ejemplos:
      "+54 9 2954 51-7407" → "+5492954517407"
      "02954 517407"        → "+5492954517407"
      "2954517407"          → "+5492954517407"
    """
    if not tel:
        return None
    digitos = re.sub(r"\D", "", tel)
    if not digitos:
        return None

    if digitos.startswith("0"):
        digitos = "549" + digitos[1:]
    elif len(digitos) == 10:
        digitos = "549" + digitos
    elif digitos.startswith("54") and not digitos.startswith("549"):
        if len(digitos) == 12:
            digitos = "549" + digitos[2:]
    elif digitos.startswith("9") and len(digitos) == 11:
        digitos = "54" + digitos

    if 8 <= len(digitos) <= 15:
        return "+" + digitos
    return None


def _wa_addr(tel: str) -> Optional[str]:
    """Formato Twilio: 'whatsapp:+5492954517407'."""
    n = normalizar_telefono(tel)
    if not n:
        return None
    return f"whatsapp:{n}"


def _from_addr(cfg: Dict) -> str:
    """Devuelve el From en formato Twilio. Acepta '+14155238886' o ya prefijado."""
    fn = (cfg.get("from_number") or "").strip()
    if not fn:
        return f"whatsapp:{SANDBOX_FROM}"
    if fn.startswith("whatsapp:"):
        return fn
    n = normalizar_telefono(fn) or fn
    if not n.startswith("+"):
        n = "+" + n
    return f"whatsapp:{n}"


# =====================================================================
# REQUEST A TWILIO
# =====================================================================

def _post_twilio(cfg: Dict, payload: Dict,
                  timeout: int = 15) -> Tuple[bool, Dict]:
    """POST a Twilio Messages API con Basic Auth."""
    import ssl
    sid = cfg["account_sid"]
    token = cfg["auth_token"]
    url = f"{TWILIO_BASE}/Accounts/{sid}/Messages.json"

    body = urllib.parse.urlencode(payload).encode("utf-8")
    auth = base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("ascii")

    # Crear contexto SSL robusto (fix para macOS Python sin certs del sistema)
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return True, data
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            err_body = {"message": str(e), "code": e.code}
        return False, err_body
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return False, {"message": str(e), "code": "network"}


# =====================================================================
# ENVÍO
# =====================================================================

def enviar_texto(cfg: Dict, destinatario: str,
                  mensaje: str) -> Tuple[bool, str]:
    """Envía texto por WhatsApp via Twilio.

    En sandbox: el destinatario debe haber mandado "join <code>" antes.
    En producción: requiere número Twilio con WhatsApp habilitado.
    """
    ok, err = config_valida(cfg)
    if not ok:
        return False, err

    to_addr = _wa_addr(destinatario)
    if not to_addr:
        return False, f"Número inválido: {destinatario}"

    payload = {
        "From": _from_addr(cfg),
        "To": to_addr,
        "Body": mensaje[:1600],   # Twilio WhatsApp body limit
    }
    ok, resp = _post_twilio(cfg, payload)
    if ok:
        sid = resp.get("sid", "")
        return True, f"Enviado (sid: {sid})"
    code = resp.get("code")
    msg = resp.get("message", "Error desconocido")
    if code == 63016:
        return False, ("Fuera de ventana 24hs — el cliente debe escribirte "
                        "primero o usás un Content Template aprobado.")
    if code == 63015 or "join" in str(msg).lower():
        return False, ("Sandbox: el destinatario debe mandar 'join <código>' "
                        f"al número sandbox primero. {msg}")
    return False, f"Twilio error {code}: {msg}"


def enviar_alerta_critica(cfg: Dict, destinatario: str,
                            cliente: str, lote: str, alerta: str,
                            accion: str = "") -> Tuple[bool, str]:
    """Compone alerta crítica y la manda como texto.

    En Twilio Sandbox y dentro de ventana 24hs: anda como texto plano.
    Para producción fuera de ventana: requiere Content Template aprobado
    (configurar `content_sid_alerta_critica` en cfg si lo necesitás).
    """
    content_sid = cfg.get("content_sid_alerta_critica")
    if content_sid:
        # Modo template (producción, fuera de ventana 24hs)
        return _enviar_content_template(
            cfg, destinatario, content_sid,
            variables={
                "1": cliente, "2": lote, "3": alerta,
                "4": accion or "Ver detalle en el sistema",
            },
        )

    # Modo texto (sandbox o ventana 24hs activa)
    mensaje = (
        f"⛔ *ALERTA CRÍTICA — {cliente}*\n"
        f"Lote {lote}: {alerta}\n"
    )
    if accion:
        mensaje += f"\n👉 *Acción:* {accion}\n"
    mensaje += "\n_HMS Nutrición Animal — alerta automática_"
    return enviar_texto(cfg, destinatario, mensaje)


def enviar_resumen_diario(cfg: Dict, destinatario: str,
                            n_clientes: int, n_criticas: int,
                            n_warning: int) -> Tuple[bool, str]:
    """Resumen matinal — texto plano (sandbox o ventana activa)."""
    content_sid = cfg.get("content_sid_resumen_diario")
    if content_sid:
        return _enviar_content_template(
            cfg, destinatario, content_sid,
            variables={"1": str(n_clientes), "2": str(n_criticas),
                        "3": str(n_warning)},
        )

    fecha = datetime.now().strftime("%d/%m/%Y")
    icono = "⛔" if n_criticas > 0 else ("⚠️" if n_warning > 0 else "✅")
    mensaje = (
        f"{icono} *HMS — Resumen {fecha}*\n\n"
        f"📊 {n_clientes} clientes monitoreados\n"
        f"⛔ {n_criticas} alertas críticas\n"
        f"⚠️ {n_warning} de atención\n\n"
        f"_Detalle en el email._"
    )
    return enviar_texto(cfg, destinatario, mensaje)


def _enviar_content_template(cfg: Dict, destinatario: str,
                               content_sid: str,
                               variables: Dict[str, str]) -> Tuple[bool, str]:
    """Envía un Content Template aprobado (producción fuera de ventana)."""
    ok, err = config_valida(cfg)
    if not ok:
        return False, err
    to_addr = _wa_addr(destinatario)
    if not to_addr:
        return False, f"Número inválido: {destinatario}"

    payload = {
        "From": _from_addr(cfg),
        "To": to_addr,
        "ContentSid": content_sid,
        "ContentVariables": json.dumps(variables, ensure_ascii=False),
    }
    ok, resp = _post_twilio(cfg, payload)
    if ok:
        return True, f"Enviado (sid: {resp.get('sid', '')})"
    return False, (
        f"Twilio template error {resp.get('code')}: {resp.get('message')}"
    )


def enviar_test(cfg: Dict, destinatario: str) -> Tuple[bool, str]:
    """Envía un mensaje de prueba simple."""
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    msg = (
        f"✅ *Test HMS Nutrición Animal*\n"
        f"Si recibís este mensaje, la integración WhatsApp funciona.\n"
        f"_{fecha}_"
    )
    return enviar_texto(cfg, destinatario, msg)


# =====================================================================
# COMPOSICIÓN (texto plano para WhatsApp) — sin cambios respecto a versión Meta
# =====================================================================

def componer_resumen_admin(n_clientes: int, n_criticos: int,
                             n_operativos: int,
                             top_alertas: List[Dict],
                             n_atencion: int = 0) -> str:
    """Compone resumen WhatsApp al admin con conteos por nivel productivo.

    Args:
      n_clientes: total de clientes monitoreados
      n_criticos: clientes con nivel productivo CRÍTICO hoy
      n_operativos: clientes con nivel productivo OPERATIVO hoy
      n_atencion: clientes con nivel productivo ATENCIÓN hoy
      top_alertas: [{cliente, nivel, titulo}, ...] ordenadas por nivel
    """
    fecha = datetime.now().strftime("%d/%m/%Y")
    if n_criticos > 0:
        icono = "🔴"
    elif n_operativos > 0:
        icono = "🟠"
    elif n_atencion > 0:
        icono = "🟡"
    else:
        icono = "🟢"

    lineas = [
        f"{icono} *HMS — Resumen {fecha}*",
        "",
        f"📊 {n_clientes} clientes monitoreados",
        f"🔴 {n_criticos} con riesgo crítico",
        f"🟠 {n_operativos} con riesgo operativo",
        f"🟡 {n_atencion} de atención leve",
    ]
    if top_alertas:
        lineas.append("")
        lineas.append("*Clientes con riesgo:*")
        nivel_icono = {"critico": "🔴", "operativo": "🟠",
                         "atencion": "🟡"}
        for a in top_alertas[:5]:
            ic = nivel_icono.get(a.get("nivel"), "•")
            lineas.append(
                f"{ic} {a.get('cliente', '')} — "
                f"{a.get('titulo', '')[:60]}"
            )
    lineas += ["", "_Detalle completo en el email._"]
    return "\n".join(lineas)


def componer_alerta_critica_texto(cliente: Dict, lote: str,
                                    alerta: Dict) -> str:
    nombre = cliente.get("nombre", "")
    establ = cliente.get("establecimiento", "") or cliente.get("localidad", "")
    titulo = alerta.get("titulo", "Alerta climática")
    desc = alerta.get("descripcion") or alerta.get("mensaje") or ""
    accion = alerta.get("accion", "")

    lineas = [
        f"⛔ *ALERTA CRÍTICA — {nombre}*",
        f"_{establ}_",
        "",
        f"*Lote:* {lote}",
        f"*Situación:* {titulo}",
        "",
        desc,
    ]
    if accion:
        lineas += ["", f"👉 *Acción:* {accion}"]
    lineas += ["", "_HMS Nutrición Animal — alerta automática_"]
    return "\n".join(lineas)


def clave_dedup(cliente_id: int, lote: str, severidad: str,
                  titulo: str) -> str:
    base = f"{cliente_id}|{lote}|{severidad}|{titulo[:60]}"
    return base.lower().replace(" ", "_")


def componer_bienvenida(cliente: Dict, contacto: Dict) -> str:
    """Mensaje de bienvenida vía WhatsApp para enviar la PRIMERA VEZ.

    Corto, claro, explica qué es y cómo darse de baja.
    """
    nombre_cliente = cliente.get("nombre", "")
    establecimiento = (cliente.get("establecimiento") or "").strip()
    nombre_dest = (contacto.get("nombre") or "").strip()
    rol = (contacto.get("rol") or "").strip()

    saludo = f"Hola {nombre_dest}" if nombre_dest else "Hola"
    if rol:
        saludo += f" ({rol})"

    contexto_quien = nombre_cliente
    if establecimiento and establecimiento.lower() != nombre_cliente.lower():
        contexto_quien = f"{nombre_cliente} — {establecimiento}"

    lineas = [
        "👋 *HMS Nutrición Animal*",
        f"{saludo}.",
        "",
        f"Sos parte del equipo de *{contexto_quien}*. Desde hoy vas a "
        f"recibir avisos de HMS por este WhatsApp cuando haya clima que "
        f"pueda afectar a los animales (calor, frío, viento).",
        "",
        "📩 El detalle completo te llega por email.",
        "📵 Solo te escribimos los días que hace falta.",
        "",
        "Si no querés recibirlos, respondé *BAJA* y listo.",
        "",
        "_Mauricio Suárez — HMS Nutrición Animal_",
    ]
    return "\n".join(lineas)


def componer_alerta_stock_cliente(
    cliente: Dict, productos_bajos: List[Dict],
) -> str:
    """Mensaje corto de WhatsApp al cliente con productos por agotarse.
    Consolida varios productos en un solo mensaje."""
    dias_min = min(
        (p.get("dias_restantes") or 0 for p in productos_bajos),
        default=999,
    )
    if dias_min <= 0:
        emoji = "🔴"
        cabecera = "*HMS — STOCK AGOTADO*"
    elif dias_min <= 3:
        emoji = "🔴"
        cabecera = "*HMS — Stock crítico*"
    elif dias_min <= 7:
        emoji = "🟠"
        cabecera = "*HMS — Stock bajo*"
    else:
        emoji = "🟡"
        cabecera = "*HMS — Reposición a coordinar*"

    lineas = [f"{emoji} {cabecera}", ""]
    for p in productos_bajos:
        lineas.append(
            f"• {p['producto']}: *{p['kg_restantes']:.0f} kg* "
            f"(~{p.get('dias_restantes', 0)} días)"
        )
    lineas.extend([
        "",
        "Coordinemos la próxima entrega con tiempo.",
        "WhatsApp: 2954-517407",
        "",
        "_Mauricio Suárez — HMS Nutrición Animal_",
    ])
    return "\n".join(lineas)


def componer_alerta_silocomedero_cliente(
    cliente: Dict, lotes_alerta: List[Dict],
) -> str:
    """Mensaje corto de WhatsApp avisando que la carga del
    silocomedero se está por terminar. Aviso operativo: hay que
    preparar mezcla nueva."""
    dias_min = min(
        (l.get("dias_restantes") or 0 for l in lotes_alerta),
        default=999,
    )
    if dias_min <= 0:
        emoji = "🔴"
        cabecera = "*HMS — Silocomedero vacío*"
        accion = "Hay que preparar la mezcla *hoy*."
    elif dias_min == 1:
        emoji = "🟠"
        cabecera = "*HMS — Silocomedero por terminarse*"
        accion = "Preparar la mezcla *mañana*."
    else:
        emoji = "🟡"
        cabecera = "*HMS — Fin de carga próximo*"
        accion = (
            f"En *{dias_min} días* hay que preparar mezcla nueva."
        )

    lineas = [f"{emoji} {cabecera}", ""]
    for l in lotes_alerta:
        ident = l.get("lote_ident", "?")
        kg_rest = l.get("kg_restantes", 0)
        dr = l.get("dias_restantes", 0)
        lineas.append(
            f"• Lote {ident}: *{kg_rest:.0f} kg* restantes "
            f"({dr} día{'s' if dr != 1 else ''})"
        )
    lineas.extend([
        "",
        accion,
        "Si necesitás reponer producto antes, avisame al 2954-517407.",
        "",
        "_Mauricio Suárez — HMS Nutrición Animal_",
    ])
    return "\n".join(lineas)


def componer_alerta_cambio_fase_cliente(
    cliente: Dict, cambios: List[Dict],
) -> str:
    """WhatsApp corto: mañana cambia la fase de adaptación. Muestra el
    diff de ingredientes principal (los que cambian) en formato compacto.
    """
    if len(cambios) == 1:
        cabecera = "*HMS — Mañana cambia la fase del lote*"
    else:
        cabecera = (
            f"*HMS — Mañana cambian {len(cambios)} fases de adaptación*"
        )

    # Importamos detector de libre disposición para no mostrar kg
    # operativos de forrajes que el animal consume a voluntad.
    from src.stock_producto import _es_a_discrecion

    lineas = [f"📋 {cabecera}", ""]
    for cam in cambios:
        ident = cam.get("lote_ident", "?")
        fase_a = (cam["fase_actual"].get("observaciones")
                  or "fase actual")
        fase_n = (cam["fase_nueva"].get("observaciones")
                  or "fase nueva")
        cant_anim = int(cam.get("cantidad_animales") or 0)
        fecha_fin = cam["fase_nueva"].get("fecha_fin")
        dur_dias = cam["fase_nueva"].get("duracion_dias")
        lineas.append(f"• Lote *{ident}*: {fase_a} → {fase_n}")
        # Rango de la fase: del X al Y (Z días). Si no hay siguiente,
        # se omite (es la última fase, queda hasta nuevo aviso).
        if fecha_fin and dur_dias:
            lineas.append(
                f"   _Del {cam['fecha_cambio']} al {fecha_fin}_ "
                f"_({dur_dias} días)_"
            )
        else:
            lineas.append(
                f"   _Arranca {cam['fecha_cambio']} — falta cargar "
                f"fecha objetivo del lote_"
            )

        # En WhatsApp lo más útil es saber CUÁNTO preparar para el
        # lote completo. Si tenemos cantidad de animales, mostramos
        # kg totales (lo operativo). Si no, fallback a kg/animal.
        comp_nueva = cam["fase_nueva"].get("composicion") or []
        medibles = [
            c for c in comp_nueva
            if not _es_a_discrecion(
                c.get("nombre") or c.get("ingrediente") or ""
            )
            and float(c.get("kg_tal_cual") or 0) > 0
        ]
        if cant_anim > 0 and medibles:
            lineas.append(
                f"   _Total a preparar (lote {cant_anim} cab.):_"
            )
            total = 0.0
            for c_n in medibles:
                nm = c_n.get("nombre") or c_n.get("ingrediente")
                kg_unit = float(c_n.get("kg_tal_cual") or 0)
                kg_lote = kg_unit * cant_anim
                total += kg_lote
                lineas.append(
                    f"   {nm}: *{kg_lote:.0f} kg*"
                )
            lineas.append(f"   → Mezcla total: *{total:.0f} kg*")
        else:
            # Fallback: kg por animal (top 3 que cambian)
            diffs_medibles = [
                d for d in cam["diff"]
                if not _es_a_discrecion(d.get("ingrediente", ""))
            ]
            diffs_orden = sorted(
                diffs_medibles,
                key=lambda d: abs(d["delta_kg"]), reverse=True,
            )
            for d in diffs_orden[:3]:
                delta = d["delta_kg"]
                signo = "+" if delta > 0 else ""
                lineas.append(
                    f"   {d['ingrediente']}: "
                    f"{d['kg_actual']:.1f} → *{d['kg_nueva']:.1f}* kg "
                    f"({signo}{delta:.2f})"
                )

        # Si hay forrajes a libre disposición, los mencionamos abajo
        # para que el productor sepa que están contemplados.
        libres = []
        for c_n in comp_nueva:
            nm = (c_n.get("nombre")
                  or c_n.get("ingrediente") or "").strip()
            if nm and _es_a_discrecion(nm) and nm not in libres:
                libres.append(nm)
        if libres:
            lineas.append(
                f"   _{', '.join(libres)}: a libre disposición_"
            )

    lineas.extend([
        "",
        "Preparar la mezcla nueva mañana con esas proporciones.",
        "Dudas: WhatsApp 2954-517407.",
        "",
        "_Mauricio Suárez — HMS Nutrición Animal_",
    ])
    return "\n".join(lineas)
