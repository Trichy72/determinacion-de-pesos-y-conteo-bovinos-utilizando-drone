"""
Sistema de alertas climáticas por email.

Composición de mails HTML con branding HMS Nutrición Animal y envío vía SMTP.
Soporta:
  - SMTP genérico (host/puerto/user/password/from)
  - SSL (puerto 465) o STARTTLS (587)
  - Logo HMS embebido (CID)
  - Versión texto plano fallback
  - Multi-destinatario por mail (BCC opcional)

Funciones públicas:
  - cargar_config_smtp(): lee data/smtp_config.json
  - guardar_config_smtp(cfg): persiste config
  - enviar_email(cfg, to, subject, html, text=None, attachments=None)
  - componer_alerta_diaria(cliente, alertas, smn_resumen): HTML+texto
  - enviar_alerta_diaria(cfg, cliente, alertas, smn_resumen, to)

La config nunca se commitea — vive en data/smtp_config.json (en .gitignore).
"""

from __future__ import annotations

import json
import smtplib
import ssl
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CONFIG_PATH = Path("data/smtp_config.json")
LOGO_PATH = Path("assets/logo.png")

# Branding HMS
COLOR_VERDE = "#1B3E27"
COLOR_LIMA = "#8BC53F"
COLOR_GRIS = "#444444"
COLOR_ALERTA_CRITICA = "#C0392B"
COLOR_ALERTA_WARNING = "#E67E22"
COLOR_ALERTA_INFO = "#2980B9"


# =====================================================================
# CONFIG SMTP
# =====================================================================

def cargar_config_smtp() -> Optional[Dict]:
    """Lee la config SMTP. Devuelve None si no existe."""
    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def guardar_config_smtp(cfg: Dict) -> None:
    """Guarda config SMTP (host, port, user, password, from_email, from_name)."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def config_valida(cfg: Optional[Dict]) -> Tuple[bool, str]:
    """Verifica que la config tenga los campos mínimos."""
    if not cfg:
        return False, "No hay config SMTP cargada"
    requeridos = ("host", "port", "user", "password", "from_email")
    faltantes = [k for k in requeridos if not cfg.get(k)]
    if faltantes:
        return False, f"Faltan: {', '.join(faltantes)}"
    return True, ""


# =====================================================================
# ENVÍO
# =====================================================================

def enviar_email(cfg: Dict, to: List[str], subject: str,
                 html: str, text: Optional[str] = None,
                 bcc: Optional[List[str]] = None,
                 embed_logo: bool = True,
                 con_bcc_admin: bool = True) -> Tuple[bool, str]:
    """Envía un email vía SMTP. Devuelve (ok, mensaje_o_error).

    Si en `cfg` está seteado `bcc_clientes` (una dirección o lista de
    direcciones) y `con_bcc_admin=True`, esa dirección se suma como BCC
    automáticamente. Esto permite que Mauricio reciba copia de cada
    email que va a un cliente, para tener registro centralizado.

    Para emails que YA van al admin (digest, prueba, etc.), pasar
    `con_bcc_admin=False` para evitar duplicación.
    """
    ok, err = config_valida(cfg)
    if not ok:
        return False, err
    if not to:
        return False, "Sin destinatarios"

    # Inyectar BCC del admin si está configurado y el destinatario no
    # coincide con el admin (evitar mandarse a uno mismo dos veces).
    bcc_list = list(bcc) if bcc else []
    if con_bcc_admin:
        bcc_admin_cfg = cfg.get("bcc_clientes") or ""
        if bcc_admin_cfg:
            # Soporta string con comas o lista
            if isinstance(bcc_admin_cfg, (list, tuple)):
                admins = [str(x).strip() for x in bcc_admin_cfg]
            else:
                admins = [
                    x.strip() for x in str(bcc_admin_cfg).split(",")
                ]
            destinos_norm = {x.lower().strip() for x in (to or [])}
            for a in admins:
                if (a and a.lower() not in destinos_norm
                        and a.lower() not in {
                            x.lower() for x in bcc_list
                        }):
                    bcc_list.append(a)

    msg = MIMEMultipart("related")
    from_name = cfg.get("from_name", "HMS Nutrición Animal")
    msg["From"] = f"{from_name} <{cfg['from_email']}>"
    msg["To"] = ", ".join(to)
    # Reply-To explícito al alias profesional. Aunque Gmail reescriba el
    # From cuando el alias no está verificado en "Send as", el Reply-To
    # asegura que las respuestas (incluido BAJA) vayan al buzón correcto.
    reply_to = cfg.get("reply_to") or cfg.get("from_email")
    if reply_to:
        msg["Reply-To"] = reply_to
    # Sender: respaldo adicional — algunos clientes muestran este campo
    # como "From" cuando el From principal coincide con el SMTP user.
    msg["Sender"] = cfg.get("from_email", "")
    if bcc_list:
        msg["Bcc"] = ", ".join(bcc_list)
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    if text:
        alt.attach(MIMEText(text, "plain", "utf-8"))
    alt.attach(MIMEText(html, "html", "utf-8"))

    # Embeber logo si existe — con dos CIDs (header y footer) para clientes
    # que no soportan reusar el mismo CID en distintos puntos del HTML.
    if embed_logo and LOGO_PATH.exists():
        try:
            with open(LOGO_PATH, "rb") as f:
                logo_bytes = f.read()
            for cid in ("hms-logo", "hms-logo-firma"):
                img = MIMEImage(logo_bytes)
                img.add_header("Content-ID", f"<{cid}>")
                img.add_header("Content-Disposition", "inline",
                                filename=f"{cid}.png")
                msg.attach(img)
        except OSError:
            pass

    try:
        port = int(cfg.get("port", 587))
        host = cfg["host"]
        use_ssl = bool(cfg.get("use_ssl", port == 465))
        timeout = int(cfg.get("timeout", 30))

        all_recipients = list(to) + bcc_list

        # Crear contexto SSL robusto. En macOS, Python a veces no encuentra
        # los certificados raíz. Si certifi está disponible, usamos su CA bundle.
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()

        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=timeout, context=ctx) as server:
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["from_email"], all_recipients, msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=timeout) as server:
                server.ehlo()
                if cfg.get("use_tls", port == 587):
                    server.starttls(context=ctx)
                    server.ehlo()
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["from_email"], all_recipients, msg.as_string())

        return True, "Email enviado"
    except smtplib.SMTPAuthenticationError as e:
        return False, f"Error de autenticación: {e}"
    except smtplib.SMTPException as e:
        return False, f"Error SMTP: {e}"
    except (OSError, TimeoutError) as e:
        return False, f"Error de red: {e}"


# =====================================================================
# COMPOSICIÓN HTML
# =====================================================================

def _severidad_color(sev: str) -> str:
    return {
        "critica": COLOR_ALERTA_CRITICA,
        "warning": COLOR_ALERTA_WARNING,
        "info": COLOR_ALERTA_INFO,
    }.get((sev or "").lower(), COLOR_ALERTA_INFO)


def _severidad_icono(sev: str) -> str:
    return {
        "critica": "⛔",
        "warning": "⚠️",
        "info": "ℹ️",
    }.get((sev or "").lower(), "•")


def _alerta_html(a: Dict) -> str:
    color = _severidad_color(a.get("severidad", ""))
    icono = _severidad_icono(a.get("severidad", ""))
    titulo = a.get("titulo", "Alerta")
    detalle = a.get("descripcion") or a.get("mensaje") or ""
    accion = a.get("accion") or ""

    def _to_html(txt: str) -> str:
        # Convertir texto plano con marcadores **bold** y saltos de línea a HTML.
        if not txt:
            return ""
        # Escapar < y > básicos primero
        txt = txt.replace("<", "&lt;").replace(">", "&gt;")
        # Convertir **texto** en <strong>texto</strong>
        import re
        txt = re.sub(r"\*\*(.+?)\*\*",
                       r"<strong style='color:#1B3E27;'>\1</strong>", txt)
        # Doble salto = párrafo, simple = <br>
        bloques = txt.split("\n\n")
        salida = []
        for b in bloques:
            b = b.replace("\n", "<br>")
            salida.append(f"<div style='margin:8px 0;'>{b}</div>")
        return "".join(salida)

    detalle_html = _to_html(detalle)
    accion_html = _to_html(accion)

    return f"""
    <div style="border-left:4px solid {color}; padding:12px 16px;
                margin:10px 0; background:#FAFAFA; border-radius:0 4px 4px 0;
                line-height:1.55;">
      <div style="font-weight:600; color:{color}; font-size:14px;
                   margin-bottom:8px;">
        {icono} {titulo}
      </div>
      <div style="color:{COLOR_GRIS}; font-size:13px;">
        {detalle_html}
      </div>
      {f'<div style="margin-top:10px; font-size:13px; color:#222;">{accion_html}</div>' if accion else ''}
    </div>
    """


def _alerta_texto(a: Dict) -> str:
    sev = (a.get("severidad", "") or "info").upper()
    titulo = a.get("titulo", "Alerta")
    det = a.get("descripcion") or a.get("mensaje") or ""
    accion = a.get("accion") or ""
    out = f"[{sev}] {titulo}\n  {det}"
    if accion:
        out += f"\n  → Acción: {accion}"
    return out


def componer_alerta_diaria(cliente: Dict, alertas_por_lote: List[Dict],
                            smn_resumen: Optional[Dict] = None,
                            clima_actual: Optional[Dict] = None,
                            alertas_oficiales: Optional[List[Dict]] = None,
                            etapa: str = "inicio",
                            dias_alerta_previos: int = 0,
                            lectura_out: Optional[Dict] = None) -> Tuple[str, str, str]:
    """Compone email de alerta diaria.

    Args:
      cliente: dict con nombre, establecimiento, localidad
      alertas_por_lote: [{lote, categoria, alertas: [...]}]
      smn_resumen: salida de clima_smn.resumen_smn() (opcional)
      clima_actual: temp_c/humedad/viento si los hay (de Open-Meteo)

    Returns: (subject, html, text)
    """
    fecha = datetime.now().strftime("%d/%m/%Y")
    nombre = cliente.get("nombre", "")
    establ = cliente.get("establecimiento", "") or cliente.get("localidad", "")

    n_criticas = sum(
        1 for l in alertas_por_lote for a in l.get("alertas", [])
        if a.get("severidad") == "critica"
    )
    n_warning = sum(
        1 for l in alertas_por_lote for a in l.get("alertas", [])
        if a.get("severidad") == "warning"
    )

    # Detectar tipo dominante (calor/frío) de la peor alerta
    peor_tipo = ""
    rank_sev = {"critica": 3, "warning": 2, "info": 1, "preventiva": 1}
    peor_rank = -1
    for l in alertas_por_lote:
        for a in l.get("alertas", []):
            r = rank_sev.get(a.get("severidad", ""), 0)
            if r > peor_rank:
                peor_rank = r
                peor_tipo = (a.get("tipo") or "").lower()

    tipo_label = {"calor": "calor", "frio": "frío"}.get(peor_tipo, "clima")

    # Subject variable según severidad + etapa del evento
    etapa_subject = {
        "inicio": "Inicio",
        "persistencia": f"Día {dias_alerta_previos + 1}",
        "acumulacion": f"Acumulado {dias_alerta_previos + 1} días",
        "recuperacion": "Recuperación",
    }.get(etapa, "")
    etapa_suffix = f" · {etapa_subject}" if etapa_subject else ""

    if n_criticas > 0:
        subject = (f"🔴 ACTUAR HOY: {tipo_label} crítico{etapa_suffix} — "
                     f"{establ or nombre}")
    elif n_warning > 0:
        subject = (f"🟠 Atención: {tipo_label} previsto{etapa_suffix} — "
                     f"{establ or nombre}")
    else:
        # Caso preventiva o resumen — solo si llega acá por dry-run
        subject = (f"🟡 Pronóstico {tipo_label}{etapa_suffix} — "
                     f"{establ or nombre}")

    # ─── 📅 EVENTO PRONOSTICADO / EN CURSO ───
    # Importante: separar claramente del "estado actual". Si el evento
    # todavía no ocurre, usar lenguaje preventivo. Si está en curso,
    # lenguaje activo con etapa.
    from src.clima import texto_etapa_evento
    # Detectar si el evento ya está ocurriendo hoy: si las condiciones
    # actuales tienen estrés relevante (T° baja o THI alto), está
    # ocurriendo. Si están normales pero hay alertas en próximos días,
    # es preventivo.
    t_actual = (clima_actual or {}).get("temp_c")
    h_actual = (clima_actual or {}).get("humedad_pct", 0) or 0
    thi_actual = (clima_actual or {}).get("thi", 0) or 0
    # Heurísticas para detectar si el evento ya está pasando HOY:
    # — frío extremo (T<5) o calor (THI>=75) en este momento
    # — etapa de persistencia/acumulación/recuperación
    # — hay alertas CRÍTICAS hoy (la severidad ya las marcó como evento)
    # — frío con agravantes evidentes en el momento (T<12 + humedad alta
    #   o lluvia/barro/viento) → no decir "condiciones de HOY normales"
    frio_con_agravantes_hoy = (
        peor_tipo == "frio"
        and isinstance(t_actual, (int, float)) and t_actual < 12
        and h_actual >= 80
    )
    ocurre_hoy = (
        (isinstance(t_actual, (int, float)) and t_actual < 5)
        or thi_actual >= 75
        or etapa in ("persistencia", "acumulacion", "recuperacion")
        or n_criticas > 0
        or frio_con_agravantes_hoy
    )
    # Fecha del primer día del evento (de la primera alerta proyectada)
    fecha_inicio_evento = ""
    for l in alertas_por_lote:
        for a in l.get("alertas", []):
            ctx = a.get("_contexto", {}) or {}
            if ctx.get("fecha"):
                fecha_inicio_evento = ctx["fecha"]
                break
        if fecha_inicio_evento:
            break

    info_etapa = texto_etapa_evento(
        etapa, tipo_clima=peor_tipo,
        dias_alerta_previos=dias_alerta_previos,
        fecha_inicio=fecha_inicio_evento,
        ocurre_hoy=ocurre_hoy,
    )

    # ─── 📖 LECTURA TÉCNICA ───
    # Frase narrativa que explica QUÉ LE PASA AL ANIMAL/CONSUMO/MEZCLA
    # antes de listar acciones. Primero intentamos generar el análisis
    # con Claude (personalizado al cliente, lote, etapa y condiciones
    # del día). Si falla, caemos a la biblioteca de frases pre-escritas.
    from src.clima import lectura_tecnica_evento
    # Detectar contexto: barro / lluvia agregados desde las alertas
    barro_ctx = False
    lluvia_ctx = 0.0
    for l in alertas_por_lote:
        for a in l.get("alertas", []):
            ctx_a = a.get("_contexto", {}) or {}
            if ctx_a.get("barro"):
                barro_ctx = True
            ll = float(ctx_a.get("lluvia_mm", 0) or 0)
            if ll > lluvia_ctx:
                lluvia_ctx = ll

    # Calcular días hasta el evento desde fecha_inicio_evento (si está)
    dias_hasta = 0
    if fecha_inicio_evento:
        try:
            from datetime import datetime as _dt, date as _date
            f_evt = _dt.strptime(fecha_inicio_evento, "%Y-%m-%d").date()
            dias_hasta = (f_evt - _dt.now().date()).days
            if dias_hasta < 0:
                dias_hasta = 0
        except (ValueError, TypeError):
            dias_hasta = 0

    # Intento de análisis personalizado con LLM
    lectura_txt = None
    fuente_llm = False
    # Calcular impacto productivo cuantificado para el lote de peor nivel
    # (si es un evento de frío). Le pasaremos esto al LLM como dato a
    # citar, para evitar que invente números. Solo en frío por ahora.
    impacto_lectura_txt = None
    impacto_lectura_dict = None  # Para el auditor post-LLM
    try:
        if peor_tipo == "frio" and alertas_por_lote:
            from src.impacto_productivo import (
                estimar_impacto_frio, formato_impacto_texto,
            )
            # Tomamos el primer lote con datos suficientes
            for _l_imp in alertas_por_lote:
                peso_l = _l_imp.get("peso_promedio_kg")
                if not peso_l or peso_l <= 0:
                    continue
                # Extraer condiciones del peor agravante de ese lote
                t_min_imp = None
                viento_imp = None
                humedad_imp = None
                barro_imp = False
                pelaje_mojado_imp = False
                for _a_imp in _l_imp.get("alertas", []):
                    ctx_imp = _a_imp.get("_contexto", {}) or {}
                    t_imp_cand = ctx_imp.get("t_min") or ctx_imp.get("temp_min")
                    if (t_imp_cand is not None
                        and (t_min_imp is None or t_imp_cand < t_min_imp)):
                        t_min_imp = t_imp_cand
                    if ctx_imp.get("viento_kmh"):
                        viento_imp = max(viento_imp or 0,
                                          ctx_imp["viento_kmh"])
                    if ctx_imp.get("barro"):
                        barro_imp = True
                    if (ctx_imp.get("lluvia_mm") or 0) > 5:
                        pelaje_mojado_imp = True
                # Caer al clima_actual si no hay contexto en alertas
                if t_min_imp is None:
                    t_min_imp = (clima_actual or {}).get("temp_min")
                if humedad_imp is None:
                    humedad_imp = (clima_actual or {}).get("humedad_pct")
                if viento_imp is None:
                    viento_imp = (clima_actual or {}).get("viento_kmh")
                # Días del evento: por defecto el día actual + previos
                dias_evt = max(1, dias_alerta_previos + 1)
                _imp = estimar_impacto_frio(
                    peso_kg=peso_l,
                    categoria=_l_imp.get("categoria", ""),
                    raza=_l_imp.get("raza", ""),
                    t_min_c=t_min_imp,
                    viento_kmh=viento_imp,
                    humedad_pct=humedad_imp,
                    barro=barro_imp,
                    pelaje_mojado=pelaje_mojado_imp,
                    dias_evento=dias_evt,
                    cantidad=_l_imp.get("cantidad_animales"),
                    adpv_objetivo_kg=_l_imp.get("adpv_objetivo_kg"),
                    energia_dieta_mcal_em_kg_ms=_l_imp.get(
                        "energia_dieta_mcal_em_kg_ms"
                    ),
                )
                if _imp:
                    impacto_lectura_txt = formato_impacto_texto(_imp)
                    impacto_lectura_dict = _imp
                    break
    except Exception:
        impacto_lectura_txt = None
        impacto_lectura_dict = None

    # Memoria: traer las últimas 2-3 lecturas técnicas enviadas a este
    # cliente para que el LLM evite repetir frases / verbos / enfoque.
    # Anti banner-blindness: si cada email aporta un ángulo distinto,
    # el productor mantiene atención. Mezclamos canales (diaria + tarde
    # + semanal) — el LLM ve TODO lo que ya le dijimos al cliente.
    lecturas_previas_lst = []
    try:
        from src import database as _dbm
        _cli_id = cliente.get("id") if cliente else None
        if _cli_id:
            lecturas_previas_lst = _dbm.obtener_lecturas_recientes(
                cliente_id=_cli_id, tipo=None, limite=3,
            )
    except Exception:
        lecturas_previas_lst = []

    try:
        from src.ai_analisis_semanal import generar_analisis_diario_llm
        lectura_txt = generar_analisis_diario_llm(
            cliente=cliente,
            alertas_por_lote=alertas_por_lote,
            clima_actual=clima_actual,
            etapa=etapa,
            ocurre_hoy=ocurre_hoy,
            dias_hasta_evento=dias_hasta,
            fecha_inicio_evento=fecha_inicio_evento,
            dias_alerta_previos=dias_alerta_previos,
            peor_tipo=peor_tipo,
            impacto_productivo_txt=impacto_lectura_txt,
            lecturas_previas=lecturas_previas_lst,
        )
        if lectura_txt:
            fuente_llm = True
            # Auditar el output del LLM por si se equivocó con el
            # total del lote (sesgo común: dividir/multiplicar mal y
            # presentar resultado como "acumulado en el evento").
            if impacto_lectura_dict:
                try:
                    from src.impacto_productivo import auditar_texto_llm
                    lectura_txt = auditar_texto_llm(
                        lectura_txt, impacto_lectura_dict,
                    )
                except Exception:
                    pass
    except Exception:
        lectura_txt = None

    # Exponer la lectura técnica al caller para que la persista en DB.
    # Esto permite usarla como MEMORIA en el próximo email del mismo
    # cliente (anti banner-blindness).
    if isinstance(lectura_out, dict):
        lectura_out["texto"] = lectura_txt or ""
        lectura_out["fuente_llm"] = bool(fuente_llm)

    # Fallback: biblioteca de frases pre-escritas
    if not lectura_txt:
        lectura_txt = lectura_tecnica_evento(
            peor_tipo, etapa,
            contexto={"barro": barro_ctx, "lluvia_mm": lluvia_ctx},
        )

    lectura_html = ""
    if lectura_txt:
        # Si el texto viene del LLM y tiene saltos de línea dobles,
        # los respetamos como párrafos; si es la biblioteca, va en una
        # sola línea.
        cuerpo = lectura_txt.replace("\n\n", "<br><br>").replace("\n", " ")
        # Badge destacado de "Análisis personalizado": es el valor
        # monetizable del sistema. NO puede pasar desapercibido.
        nota_personalizado = ""
        if fuente_llm:
            nota_personalizado = f"""
            <div style="display:inline-block; background:{COLOR_VERDE};
              color:#FFFFFF; padding:5px 12px; border-radius:14px;
              font-size:11.5px; font-weight:700;
              letter-spacing:0.3px; margin:8px 0 4px;
              text-transform:uppercase;">
              ✨ Análisis personalizado para tu lote
            </div>
            <div style="font-size:12px; color:{COLOR_VERDE};
              font-weight:600; margin:0 0 8px;">
              Generado por HMS para tu lote y las condiciones
              climáticas de hoy en {establ or 'tu zona'}.
            </div>"""
        lectura_html = f"""
        <div style="background:#FFFCF5; border-left:4px solid {COLOR_VERDE};
          padding:14px 16px; margin:14px 0; border-radius:6px;
          box-shadow:0 1px 2px rgba(0,0,0,0.04);">
          <div style="font-size:13px; color:{COLOR_VERDE};
            font-weight:700; letter-spacing:0.3px;">
            📖 LECTURA TÉCNICA
          </div>
          {nota_personalizado}
          <p style="margin:8px 0 0; font-size:13.5px; color:#2a2a2a;
            line-height:1.6;">
            {cuerpo}
          </p>
        </div>"""
    etapa_html = ""
    if etapa != "estable":
        # Header de bloque según si es preventivo o activo
        if info_etapa.get("lenguaje") == "preventivo":
            header_bloque = "📅 EVENTO PRONOSTICADO"
            color_borde = COLOR_ALERTA_INFO
        elif info_etapa.get("lenguaje") == "post":
            header_bloque = "✅ ETAPA DE RECUPERACIÓN"
            color_borde = COLOR_VERDE
        else:
            header_bloque = "🔁 EVENTO EN CURSO"
            color_borde = COLOR_LIMA

        etapa_html = f"""
        <div style="background:#F8F8F8; border-left:3px solid {color_borde};
          padding:12px 14px; margin:14px 0; border-radius:4px;">
          <div style="font-size:12px; color:{COLOR_GRIS}; font-weight:600;">
            {header_bloque}
          </div>
          <strong style="color:{COLOR_VERDE}; display:block;
            margin-top:6px;">
            {info_etapa['titulo']}</strong>
          <p style="margin:6px 0 0; font-size:13px; color:#444;">
            {info_etapa['mensaje']}
          </p>
          {f'<p style="margin:6px 0 0; font-size:12px; color:#666;"><strong>Prioridad técnica:</strong> {info_etapa["prioridad"]}</p>' if info_etapa.get('prioridad') else ''}
        </div>"""

    # ---------- HTML ----------
    # Personalizar las acciones + la línea de RESUMEN OPERATIVO por LLM
    # antes de armar los bloques. Por cada alerta de cada lote, intentar
    # generar acciones a medida del contexto (categoría, clima, etapa,
    # timing) y variar la redacción del dato climático para evitar la
    # plantilla rígida. Si LLM falla, se mantiene el `accion` original
    # del motor de reglas (fallback transparente).
    try:
        from src.ai_analisis_semanal import (
            generar_acciones_llm, generar_resumen_operativo_llm,
        )
        from src.impacto_productivo import (
            estimar_impacto_frio as _estimar_imp_frio,
            formato_impacto_texto as _formato_imp_txt,
        )
        from datetime import datetime as _dt2
        for l in alertas_por_lote:
            categoria_lote = l.get("categoria", "")
            for a in l.get("alertas", []):
                # Solo personalizar warning/critica (info no vale el costo)
                if a.get("severidad") not in ("warning", "critica"):
                    continue
                ctx_a = a.get("_contexto", {}) or {}
                clima_a = {
                    "temperatura": (ctx_a.get("temp_max")
                                    or (clima_actual or {}).get("temp_c")),
                    "min_nocturna": ctx_a.get("t_min"),
                    "viento_kmh": (ctx_a.get("viento_kmh")
                                    or (clima_actual or {}).get("viento_kmh")
                                    or 0),
                    "lluvia_mm": (ctx_a.get("lluvia_mm")
                                   or (clima_actual or {}).get("lluvia_mm")
                                   or 0),
                    "humedad_pct": (clima_actual or {}).get("humedad_pct") or 0,
                    "thi": (clima_actual or {}).get("thi") or 0,
                }
                fecha_evt = ctx_a.get("fecha", "")
                dias_hasta_a = 0
                ocurre_hoy_a = True
                if fecha_evt:
                    try:
                        f_e = _dt2.strptime(fecha_evt, "%Y-%m-%d").date()
                        dias_hasta_a = (f_e - _dt2.now().date()).days
                        if dias_hasta_a > 0:
                            ocurre_hoy_a = False
                        else:
                            dias_hasta_a = 0
                    except (ValueError, TypeError):
                        pass

                # Calcular impacto productivo del evento de FRÍO sobre
                # este lote, para que el LLM cite los rangos exactos en
                # las acciones nutricionales (no invente kg).
                impacto_a_txt = None
                impacto_a_dict = None  # Para el auditor post-LLM
                if (a.get("tipo") or "").lower() == "frio":
                    peso_l = l.get("peso_promedio_kg")
                    if peso_l and peso_l > 0:
                        try:
                            _imp_a = _estimar_imp_frio(
                                peso_kg=peso_l,
                                categoria=categoria_lote,
                                raza=l.get("raza", ""),
                                t_min_c=(ctx_a.get("t_min")
                                          or clima_a.get("min_nocturna")
                                          or clima_a.get("temperatura")),
                                viento_kmh=clima_a.get("viento_kmh"),
                                humedad_pct=clima_a.get("humedad_pct"),
                                barro=bool(ctx_a.get("barro")),
                                pelaje_mojado=(
                                    (clima_a.get("lluvia_mm") or 0) > 5
                                ),
                                dias_evento=max(1, dias_alerta_previos + 1),
                                cantidad=l.get("cantidad_animales"),
                                adpv_objetivo_kg=l.get("adpv_objetivo_kg"),
                                energia_dieta_mcal_em_kg_ms=l.get(
                                    "energia_dieta_mcal_em_kg_ms"
                                ),
                            )
                            if _imp_a:
                                impacto_a_txt = _formato_imp_txt(_imp_a)
                                impacto_a_dict = _imp_a
                        except Exception:
                            impacto_a_txt = None
                            impacto_a_dict = None

                acciones_personalizadas = generar_acciones_llm(
                    cliente=cliente,
                    tipo=a.get("tipo", "") or "",
                    nivel=a.get("nivel", "") or "",
                    categoria=categoria_lote,
                    clima=clima_a,
                    etapa=etapa,
                    dias_alerta_previos=dias_alerta_previos,
                    ocurre_hoy=ocurre_hoy_a,
                    dias_hasta_evento=dias_hasta_a,
                    impacto_productivo_txt=impacto_a_txt,
                )
                # Auditar cada acción del LLM por si se equivocó con
                # el total del lote (sesgo común al confundir
                # kg/día×cab con total del evento).
                if acciones_personalizadas and impacto_a_dict:
                    try:
                        from src.impacto_productivo import auditar_texto_llm
                        for _cat in ("inmediatas", "operativas",
                                      "nutricionales"):
                            _items = acciones_personalizadas.get(_cat) or []
                            acciones_personalizadas[_cat] = [
                                auditar_texto_llm(_it, impacto_a_dict)
                                for _it in _items
                            ]
                    except Exception:
                        pass
                # Reescribir la línea de condición climática del RESUMEN
                # OPERATIVO para evitar la plantilla rígida. Mantiene
                # todos los datos numéricos, varía la redacción.
                clima_a_con_barro = dict(clima_a)
                if ctx_a.get("barro"):
                    clima_a_con_barro["barro"] = True
                resumen_op_llm = generar_resumen_operativo_llm(
                    tipo=a.get("tipo", "") or "",
                    nivel=a.get("nivel", "") or "",
                    clima=clima_a_con_barro,
                )

                if not acciones_personalizadas and not resumen_op_llm:
                    continue
                # Reconstruir el campo `accion` (texto markdown). Si
                # tenemos acciones nuevas, las reemplazamos; si tenemos
                # resumen LLM, reemplazamos la línea de condición.
                accion_actual = a.get("accion", "") or ""
                # 1) Separar cabecera (RESUMEN OPERATIVO + Cuándo + condición)
                #    de las acciones viejas. Inicializamos _resto para
                #    cubrir el caso en que el separador no aparezca.
                _resto = ""
                if "**⚡ ACCIONES CLAVE**" in accion_actual:
                    cabecera, _resto = accion_actual.split(
                        "**⚡ ACCIONES CLAVE**", 1
                    )
                else:
                    cabecera = accion_actual

                # 2) Si tenemos resumen_op_llm, reemplazar la línea de
                #    condición climática dentro de la cabecera.
                #    Estructura esperada:
                #      **📌 RESUMEN OPERATIVO**
                #      📅 Cuándo: ...
                #      <cond_line>
                #      (línea vacía)
                if resumen_op_llm:
                    lineas_cab = cabecera.split("\n")
                    nueva_cab = []
                    saltada_cond = False
                    for idx, ln in enumerate(lineas_cab):
                        if (not saltada_cond
                            and idx > 0
                            and not ln.startswith("📅")
                            and not ln.startswith("**")
                            and ln.strip()
                            and "RESUMEN OPERATIVO" not in ln):
                            # Esta es la línea de cond_line vieja —
                            # reemplazarla por la versión LLM
                            nueva_cab.append(resumen_op_llm)
                            saltada_cond = True
                        else:
                            nueva_cab.append(ln)
                    cabecera = "\n".join(nueva_cab)

                # 3) Armar acciones (LLM si está disponible, sino las
                #    originales del motor)
                nuevas_lineas = [cabecera.rstrip(), "",
                                  "**⚡ ACCIONES CLAVE**"]
                if acciones_personalizadas:
                    if acciones_personalizadas["inmediatas"]:
                        nuevas_lineas.append("**Inmediatas**")
                        for it in acciones_personalizadas["inmediatas"]:
                            nuevas_lineas.append(f"• {it}")
                        nuevas_lineas.append("")
                    if acciones_personalizadas["operativas"]:
                        nuevas_lineas.append("**Operativas**")
                        for it in acciones_personalizadas["operativas"]:
                            nuevas_lineas.append(f"• {it}")
                        nuevas_lineas.append("")
                    if acciones_personalizadas["nutricionales"]:
                        nuevas_lineas.append("**Nutricionales**")
                        for it in acciones_personalizadas["nutricionales"]:
                            nuevas_lineas.append(f"• {it}")
                        nuevas_lineas.append("")
                else:
                    # Si LLM de acciones falló, dejar las acciones del
                    # motor (que vienen en _resto)
                    nuevas_lineas.append(_resto.lstrip("\n"))
                a["accion"] = "\n".join(nuevas_lineas)
    except Exception as _e_llm:
        # Si algo falla en la personalización, seguimos con las acciones
        # del motor (fallback transparente). Logueamos para diagnóstico
        # futuro — esto es CLAVE para entender por qué a veces no se
        # ven los textos LLM en los emails.
        import logging as _logging
        import traceback as _tb
        _logger = _logging.getLogger(__name__)
        _logger.warning(
            "Fallo personalización LLM en componer_alerta_diaria: %s\n%s",
            _e_llm, _tb.format_exc(),
        )

    bloques_lotes = []
    for l in alertas_por_lote:
        if not l.get("alertas"):
            continue
        alertas_html = "".join(_alerta_html(a) for a in l["alertas"])
        bloques_lotes.append(f"""
        <div style="margin:14px 0;">
          <div style="font-size:15px; font-weight:600; color:{COLOR_VERDE};
                      border-bottom:1px solid #DDD; padding-bottom:4px;">
            Lote {l.get('lote', '')} — {l.get('categoria', '')}
          </div>
          {alertas_html}
        </div>
        """)
    if not bloques_lotes:
        bloques_lotes.append(f"""
        <div style="padding:20px; background:#F0F8E8; border-radius:6px;
                    text-align:center; color:{COLOR_VERDE};">
          ✅ Sin alertas climáticas activas hoy. Clima dentro de rangos normales
          para todos los lotes.
        </div>
        """)

    # ─── 📍 ESTADO ACTUAL ───
    # Las condiciones reales de HOY. Importante: distinguir el estrés
    # calórico (THI) de otros estresores (frío, viento, barro). THI
    # bajo NO equivale a "sin estrés"; puede haber riesgo por frío.
    clima_html = ""
    if clima_actual:
        t = clima_actual.get("temp_c")
        h = clima_actual.get("humedad_pct")
        thi = clima_actual.get("thi")
        thi_estado = clima_actual.get("thi_estado", "")
        t_str = f"{t:.1f}" if isinstance(t, (int, float)) else "—"
        h_str = f"{h:.0f}" if isinstance(h, (int, float)) else "—"
        thi_str = f"{thi:.0f}" if isinstance(thi, (int, float)) else "—"

        # Interpretación del THI explícita: SOLO mide estrés CALÓRICO
        if isinstance(thi, (int, float)):
            if thi >= 78:
                thi_interp = "⚠️ con riesgo calórico"
            elif thi >= 70:
                thi_interp = "atención calórica"
            else:
                thi_interp = "sin estrés calórico"
        else:
            thi_interp = ""

        # Frase de estado actual del momento — adaptada a la estación.
        # En otoño/invierno con T° baja NO tiene sentido hablar de "calor".
        # Si la alerta dominante es de FRÍO, el foco son agravantes del
        # frío, no el THI.
        if isinstance(t, (int, float)) and t < 5:
            estado_actual_txt = (
                f"❄️ <strong>T° actual baja ({t:.0f}°C).</strong> "
                f"El estrés calórico no aplica; el riesgo se evalúa "
                f"por frío, viento y humedad."
            )
        elif isinstance(t, (int, float)) and t < 15 and peor_tipo == "frio":
            # Otoño/invierno con alerta de frío en curso o pronosticada
            estado_actual_txt = (
                f"🌬️ Condiciones frescas ({t:.0f}°C). Sin riesgo "
                f"calórico — el foco está en frío, viento y barro."
            )
        elif isinstance(t, (int, float)) and t < 15:
            estado_actual_txt = (
                f"🌤️ Condiciones frescas ({t:.0f}°C). El THI no aplica "
                f"en esta estación; vigilar frío, viento y humedad."
            )
        elif thi_interp == "⚠️ con riesgo calórico":
            estado_actual_txt = (
                f"🌡️ <strong>Estrés calórico actual presente</strong> "
                f"(THI {thi_str})."
            )
        else:
            estado_actual_txt = (
                f"✅ Condiciones actuales {thi_interp.replace('atención calórica', 'tranquilas')}."
            )

        clima_html = f"""
        <div style="background:#F5F5F5; padding:12px; border-radius:6px;
          margin:14px 0; border-left:3px solid {COLOR_VERDE};">
          <div style="font-size:12px; color:{COLOR_GRIS}; font-weight:600;">
            📍 ESTADO ACTUAL
          </div>
          <div style="margin-top:6px; font-size:14px;">
            🌡️ {t_str}°C &nbsp; 💧 {h_str}% HR &nbsp; THI {thi_str}
            <span style="font-size:11px; color:#888;">
              (índice que combina temperatura y humedad)
            </span>
          </div>
          <div style="margin-top:4px; font-size:13px; color:#555;">
            {estado_actual_txt}
          </div>
          <div style="font-size:11px; color:#888; margin-top:4px;">
            Fuente: Open-Meteo · El THI solo mide estrés calórico, no
            riesgos por frío, viento o barro.
          </div>
        </div>
        """

    # Alertas oficiales del SMN (amarillo/naranja/rojo)
    alertas_oficiales_html = ""
    if alertas_oficiales:
        bloques = []
        for ao in alertas_oficiales:
            nivel = ao.get("nivel", "amarillo")
            color_borde = {
                "rojo": "#C0392B",
                "naranja": "#E67E22",
                "amarillo": "#F1C40F",
            }.get(nivel, "#F1C40F")
            icono_nivel = {
                "rojo": "🔴",
                "naranja": "🟠",
                "amarillo": "🟡",
            }.get(nivel, "🟡")
            zonas_str = ", ".join(ao.get("zonas", [])[:3])
            vigencia = ""
            if ao.get("valida_desde") or ao.get("valida_hasta"):
                vigencia = (f"<br><span style='font-size:11px; color:#888;'>"
                              f"Vigencia: {ao.get('valida_desde','')} → "
                              f"{ao.get('valida_hasta','')}</span>")
            bloques.append(f"""
            <div style="border-left:4px solid {color_borde}; padding:10px 14px;
                        margin:6px 0; background:#FFFCF0; border-radius:0 4px 4px 0;">
              <div style="font-weight:600; color:{color_borde}; font-size:13px;">
                {icono_nivel} ALERTA OFICIAL SMN — {nivel.upper()}
              </div>
              <div style="margin-top:4px; font-size:13px; color:#333;">
                {ao.get('titulo', '')}
              </div>
              <div style="margin-top:4px; font-size:12px; color:#555;">
                {ao.get('descripcion', '')[:280]}
              </div>
              <div style="margin-top:6px; font-size:11px; color:#888;">
                Zona: {zonas_str}{vigencia}
              </div>
            </div>
            """)
        alertas_oficiales_html = (
            f"<div style='margin:14px 0;'>"
            f"<div style='font-size:12px; color:{COLOR_GRIS}; "
            f"font-weight:600; margin-bottom:6px;'>"
            f"⚠️ ALERTAS OFICIALES VIGENTES (Servicio Meteorológico Nacional)</div>"
            f"{''.join(bloques)}</div>"
        )

    # Resumen SMN — solo si hay observación con datos reales
    smn_html = ""
    if smn_resumen and smn_resumen.get("estacion"):
        est = smn_resumen["estacion"]
        obs = smn_resumen.get("observacion") or {}
        # Mostrar la sección solo si tenemos al menos T° o humedad
        tiene_datos = (obs.get("temp_c") is not None or
                         obs.get("humedad_pct") is not None or
                         obs.get("viento_kmh") is not None)
        if tiene_datos:
            t_smn = obs.get("temp_c")
            h_smn = obs.get("humedad_pct")
            v_smn = obs.get("viento_kmh")
            t_smn_str = f"{t_smn:.1f}" if isinstance(t_smn, (int, float)) else "—"
            h_smn_str = f"{h_smn:.0f}" if isinstance(h_smn, (int, float)) else "—"
            v_smn_str = f"{v_smn:.0f}" if isinstance(v_smn, (int, float)) else "—"
            smn_html = f"""
            <div style="background:#FFF8E1; padding:12px; border-radius:6px; margin:14px 0;
                        border-left:3px solid {COLOR_LIMA};">
              <div style="font-size:13px; color:{COLOR_GRIS};">
                ESTACIÓN OFICIAL SMN — {est.get('nombre')} ({est.get('distancia_km')} km)
              </div>
              <div style="margin-top:6px; font-size:14px;">
                🌡️ {t_smn_str}°C &nbsp;
                💧 {h_smn_str}% &nbsp;
                💨 {v_smn_str} km/h
              </div>
              <div style="font-size:11px; color:#888; margin-top:4px;">
                {obs.get('descripcion', '') or 'Sin descripción'}
              </div>
            </div>
            """

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0; padding:0; background:#F4F4F4; font-family:Arial,sans-serif;">
  <div style="max-width:640px; margin:0 auto; background:white;">
    <!-- Banda fina verde como acento de marca -->
    <div style="height:6px; background:{COLOR_VERDE};"></div>

    <!-- Header limpio con logo en blanco -->
    <div style="padding:24px 28px 18px; background:white; border-bottom:1px solid #EEE;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td style="vertical-align:middle;">
            <img src="cid:hms-logo" alt="HMS Nutrición Animal"
                 style="height:42px; width:auto; max-width:160px; display:block;">
          </td>
          <td style="vertical-align:middle; text-align:right;">
            <div style="font-size:12px; color:{COLOR_GRIS}; letter-spacing:0.5px;
                         text-transform:uppercase;">{fecha}</div>
            <div style="font-size:20px; font-weight:600; color:{COLOR_VERDE};
                         margin-top:4px;">Alerta climática diaria</div>
          </td>
        </tr>
      </table>
    </div>

    <div style="padding:24px;">
      <div style="font-size:16px; color:{COLOR_VERDE}; font-weight:600;">
        {nombre}
      </div>
      <div style="font-size:13px; color:{COLOR_GRIS};">
        {establ}
      </div>

      <div style="margin:14px 0; padding:10px 14px; background:#F0F8E8;
                  border-radius:6px; font-size:13px; color:{COLOR_VERDE};">
        <strong>{n_criticas + n_warning}</strong> alertas activas
        ({n_criticas} críticas / {n_warning} de atención)
      </div>

      {alertas_oficiales_html}
      {clima_html}
      {etapa_html}
      {lectura_html}
      {smn_html}

      {''.join(bloques_lotes)}

      <!-- Fuentes técnicas: respaldo bibliográfico del informe -->
      <div style="margin-top:20px; padding:12px 14px; background:#F4F6F2;
                  border-radius:6px; font-size:10.5px; color:#555;
                  line-height:1.55;">
        <div style="color:{COLOR_VERDE}; font-weight:700; font-size:11px;
                    margin-bottom:4px; letter-spacing:0.3px;">
          📚 FUENTES TÉCNICAS DEL INFORME
        </div>
        Cálculos energéticos y zona de confort térmico:
        <strong>NRC/NASEM 2016</strong> (Nutrient Requirements of
        Beef Cattle) · Ajustes y rangos para Pampa Húmeda:
        <strong>Pordomingo, Latimori, Pezzola, INTA Anguil</strong>
        · Referencias de mercado y categorías argentinas:
        <strong>IPCVA, AACREA</strong> · Datos climáticos en tiempo
        real: <strong>Open-Meteo</strong>.
      </div>

      <!-- Cierre del informe con logo integrado -->
      <div style="margin-top:24px; padding-top:18px; border-top:2px solid {COLOR_LIMA};
                  text-align:center;">
        <img src="cid:hms-logo-firma" alt="HMS Nutrición Animal" style="height:42px; width:auto; max-width:160px;"
             style="display:inline-block; margin-bottom:10px;">
        <div style="font-size:12px; color:{COLOR_GRIS}; line-height:1.6;">
          <strong style="color:{COLOR_VERDE}; font-size:13px;">
            HMS Nutrición Animal
          </strong><br>
          Mauricio Suárez — Asesor en Nutrición Animal<br>
          Ruta 5 km 525 — Catriló, La Pampa<br>
          2954-517407 &nbsp;|&nbsp; mauricio@hmsnutricionanimal.com.ar
        </div>
      </div>

      <div style="margin-top:18px; font-size:10px; color:#AAA; text-align:center;">
        Reporte generado automáticamente. Para dejar de recibir estas alertas, respondé "BAJA".
      </div>
    </div>

    <div style="background:{COLOR_VERDE}; padding:8px 24px; color:white;
                font-size:10px; text-align:center;">
      Sistema de seguimiento climático HMS · {fecha}
    </div>
  </div>
</body>
</html>"""

    # ---------- TEXTO ----------
    lineas = [
        f"HMS Nutrición Animal — Alerta climática diaria",
        f"Fecha: {fecha}",
        "",
        f"Cliente: {nombre}",
        f"Establecimiento: {establ}",
        "",
        f"Resumen: {n_criticas} alertas críticas, {n_warning} de atención",
        "",
    ]
    if clima_actual:
        t_v = clima_actual.get("temp_c")
        h_v = clima_actual.get("humedad_pct")
        thi_v = clima_actual.get("thi")
        lineas += [
            "CONDICIONES ACTUALES (Open-Meteo):",
            f"  Temperatura: {t_v:.1f}°C" if isinstance(t_v, (int, float))
                else f"  Temperatura: —°C",
            f"  Humedad: {h_v:.0f}%" if isinstance(h_v, (int, float))
                else f"  Humedad: —%",
            f"  THI: {thi_v:.0f} {clima_actual.get('thi_estado', '')}"
                if isinstance(thi_v, (int, float))
                else f"  THI: — {clima_actual.get('thi_estado', '')}",
            "",
        ]
    if smn_resumen and smn_resumen.get("estacion"):
        est = smn_resumen["estacion"]
        obs = smn_resumen.get("observacion") or {}
        ts = obs.get("temp_c")
        hs = obs.get("humedad_pct")
        vs = obs.get("viento_kmh")
        lineas += [
            f"ESTACIÓN SMN — {est.get('nombre')} ({est.get('distancia_km')} km):",
            f"  Temperatura: {ts:.1f}°C" if isinstance(ts, (int, float))
                else "  Temperatura: —°C",
            f"  Humedad: {hs:.0f}%" if isinstance(hs, (int, float))
                else "  Humedad: —%",
            f"  Viento: {vs:.0f} km/h" if isinstance(vs, (int, float))
                else "  Viento: — km/h",
            "",
        ]
    for l in alertas_por_lote:
        if not l.get("alertas"):
            continue
        lineas.append(f"=== Lote {l.get('lote', '')} — {l.get('categoria', '')} ===")
        for a in l["alertas"]:
            lineas.append(_alerta_texto(a))
        lineas.append("")
    lineas += [
        "—",
        "HMS Nutrición Animal",
        "Mauricio Suárez — Asesor en Nutrición Animal",
        "Ruta 5 km 525, Catriló, La Pampa",
        "Tel: 2954-517407 — mauricio@hmsnutricionanimal.com.ar",
    ]
    text = "\n".join(lineas)

    return subject, html, text


def _parece_telefono_o_email(texto: str) -> bool:
    """¿Este 'nombre' parece más un teléfono o email que un nombre real?
    Útil cuando el asesor cargó el dato de contacto en el campo nombre
    por confusión."""
    if not texto:
        return False
    t = texto.strip()
    if not t:
        return False
    # Empieza con + o tiene mayoría de dígitos → teléfono
    if t.startswith("+") or t.startswith("0"):
        return sum(c.isdigit() for c in t) >= len(t) * 0.5
    if sum(c.isdigit() for c in t) >= 6:
        return True
    # Tiene @ → email
    if "@" in t:
        return True
    return False


def _primer_nombre(nombre_completo: str) -> str:
    """Devuelve el primer nombre o, si no hay espacios, el string
    entero. Usado para saludos en emails."""
    if not nombre_completo:
        return ""
    partes = nombre_completo.strip().split()
    return partes[0] if partes else nombre_completo


def componer_bienvenida(cliente: Dict, contacto: Dict) -> Tuple[str, str, str]:
    """Mensaje de bienvenida que se manda la PRIMERA VEZ a un destinatario.

    Explica qué es HMS, por qué le va a llegar la alerta, y cómo darse de
    baja. Sin esto, el cliente piensa que es spam.
    """
    nombre_cliente = (cliente.get("nombre") or "").strip()
    establecimiento = (cliente.get("establecimiento") or "").strip()
    nombre_dest_raw = (contacto.get("nombre", "") or "").strip()
    rol = (contacto.get("rol", "") or "").strip()

    # Si el "nombre del contacto" en realidad es un teléfono o email
    # (caso típico: el dato se cargó en el campo Contacto en lugar del
    # campo Nombre), usar el nombre del cliente como destinatario del
    # saludo. Si no, usar el primer nombre del contacto.
    if _parece_telefono_o_email(nombre_dest_raw) or not nombre_dest_raw:
        nombre_saludo = _primer_nombre(nombre_cliente) or "buen día"
        es_cliente_principal = True
    else:
        nombre_saludo = _primer_nombre(nombre_dest_raw)
        # Si el nombre del contacto coincide con el del cliente, es el
        # cliente principal; si tiene rol cargado (encargado, etc.), es
        # un colaborador.
        es_cliente_principal = (
            not rol and (
                nombre_dest_raw.lower() in nombre_cliente.lower()
                or nombre_cliente.lower() in nombre_dest_raw.lower()
            )
        )

    saludo = f"Hola {nombre_saludo}"

    # Armar el "para qué establecimiento" — solo si hay datos útiles
    referencia_campo = ""
    if establecimiento and nombre_cliente:
        referencia_campo = f"de <strong>{establecimiento}</strong>"
    elif establecimiento:
        referencia_campo = f"de <strong>{establecimiento}</strong>"
    elif nombre_cliente and not es_cliente_principal:
        referencia_campo = f"de <strong>{nombre_cliente}</strong>"

    if es_cliente_principal:
        # Caso típico: el email va al productor mismo. Tono directo,
        # personal, sin asumir empleados/jerarquías.
        if referencia_campo:
            contexto = (
                f"Soy <strong>Mauricio Suárez</strong>, de "
                f"<strong>HMS Nutrición Animal</strong>. Acompañamos el "
                f"seguimiento nutricional de la hacienda {referencia_campo}. "
                f"Como parte de ese trabajo, desde hoy vas a recibir en "
                f"este correo las <strong>alertas climáticas</strong> "
                f"del campo: cuándo se viene calor, frío o lluvia que "
                f"puedan afectar el bienestar y el consumo de los "
                f"animales, y qué hacer para anticiparte."
            )
        else:
            contexto = (
                f"Soy <strong>Mauricio Suárez</strong>, de "
                f"<strong>HMS Nutrición Animal</strong>. Acompañamos el "
                f"seguimiento nutricional de tu hacienda. Como parte "
                f"de ese trabajo, desde hoy vas a recibir en este "
                f"correo las <strong>alertas climáticas</strong> del "
                f"campo: cuándo se viene calor, frío o lluvia que "
                f"puedan afectar el bienestar y el consumo de los "
                f"animales, y qué hacer para anticiparte."
            )
    else:
        # Destinatario es un colaborador (encargado, personal de
        # comedero) que recibe en nombre del cliente principal.
        rol_str = f" ({rol})" if rol else ""
        contexto = (
            f"Trabajás{rol_str} en <strong>{nombre_cliente}</strong>"
            f"{(' — ' + establecimiento) if establecimiento else ''}, "
            f"establecimiento que acompañamos desde <strong>HMS "
            f"Nutrición Animal</strong> en el seguimiento nutricional "
            f"de la hacienda. Desde hoy vas a recibir en este correo "
            f"las <strong>alertas climáticas</strong> del campo: "
            f"cuándo se viene calor, frío o lluvia que puedan afectar "
            f"el bienestar y el consumo de los animales, y qué hacer "
            f"para anticiparte."
        )

    subject = "👋 Bienvenido al sistema de alertas HMS Nutrición Animal"
    html = f"""<!DOCTYPE html>
<html><body style="margin:0; padding:0; background:#F4F4F4;
  font-family:Arial,sans-serif;">
  <div style="max-width:640px; margin:0 auto; background:white;">
    <div style="background:white; padding:20px 24px; border-bottom:3px solid {COLOR_VERDE};">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td style="vertical-align:middle;">
            <div style="font-size:19px; font-weight:600; color:{COLOR_VERDE};">
              Bienvenido al sistema de alertas
            </div>
          </td>
          <td style="vertical-align:middle; text-align:right; width:170px;">
            <img src="cid:hms-logo" alt="HMS Nutrición Animal" style="height:105px; width:auto; max-width:260px; display:block; margin-left:auto; margin-right:-8px;">
          </td>
        </tr>
      </table>
    </div>
    <div style="padding:24px; color:{COLOR_GRIS}; font-size:14px;
      line-height:1.55;">
      <p style="font-size:15px;">{saludo},</p>

      <p>{contexto}</p>

      <div style="background:#F0F8E8; border-left:3px solid {COLOR_LIMA};
        padding:14px 16px; margin:18px 0; border-radius:4px;">
        <strong style="color:{COLOR_VERDE};">¿Por qué este sistema marca la diferencia?</strong>
        <p style="margin:8px 0 10px 0;">
          Este sistema toma el pronóstico y lo traduce en algo
          concreto: <strong>cómo se va a comportar tu hacienda y qué
          hacer</strong> para que el clima no se traduzca en pérdida
          de kilos.
        </p>
        <p style="margin:0 0 10px 0;">
          Cada aviso une <strong>tres miradas que normalmente viajan
          por separado</strong>: el pronóstico real para tu zona,
          <strong>la dieta que armamos para tu lote</strong> (y cómo
          va a responder el animal según categoría), y la experiencia
          de manejo a campo. Cuando llegan juntas, dejan de ser un
          dato y pasan a ser una decisión.
        </p>
        <p style="margin:0 0 4px 0;">
          <strong style="color:{COLOR_VERDE};">Lo que ganás:</strong>
        </p>
        <ul style="margin:4px 0 0 0; padding-left:20px;">
          <li><strong>Anticipación real</strong> — el aviso llega
              antes del evento, no después.</li>
          <li><strong>Acciones a medida del lote</strong> — pensadas
              para la categoría, la dieta y la zona, no recetas
              genéricas.</li>
          <li><strong>Kilos en el corral</strong> — cada día de estrés
              sin manejar es aumento perdido. Los ajustes simples a
              tiempo lo evitan.</li>
          <li><strong>Más presencia técnica</strong> — como no
              podemos estar todos los días en los corrales, este
              sistema te extiende los ojos sobre el campo.</li>
        </ul>
      </div>

      <p><strong style="color:{COLOR_VERDE};">¿Qué te llega?</strong></p>
      <ul style="padding-left:20px;">
        <li><strong>Informe semanal los lunes</strong> — un panorama
            de lo que se viene en los próximos días y cómo puede
            impactar a la hacienda.</li>
        <li><strong>Actualización a mitad de semana</strong> si el
            pronóstico cambia fuerte respecto al del lunes —
            para que no te encuentres con una sorpresa.</li>
        <li><strong>Avisos puntuales</strong> los días que el clima
            puede afectar a los animales, con las acciones concretas
            para ese día. Cuando no hay riesgo, silencio.</li>
        <li><strong>Lectura del campo</strong>: qué pasa con el
            consumo, con el rumen, con los kilos — y qué conviene
            hacer.</li>
      </ul>

      <p><strong style="color:{COLOR_VERDE};">¿No querés recibirlo más?</strong></p>
      <p>
        Respondé este email con la palabra <strong>BAJA</strong> en
        cualquier momento. Te damos de baja automáticamente y no te
        llegan más alertas. Sin trámites.
      </p>

      <p>Cualquier consulta, escribime directamente a
        <a href="mailto:mauricio@hmsnutricionanimal.com.ar"
           style="color:{COLOR_VERDE};">
        mauricio@hmsnutricionanimal.com.ar</a>
        o al WhatsApp <strong>2954-517407</strong>.
      </p>

      <p style="margin-top:24px; color:#888; font-size:12px;">
        — Mauricio Suárez —<br>
        HMS Nutrición Animal<br>
        <span style="font-style:italic;">
          Información que anticipa. Decisiones que rinden.
        </span>
      </p>
    </div>
    <div style="background:{COLOR_VERDE}; padding:12px 24px; color:white;
      font-size:11px; text-align:center;">
      <strong>HMS Nutrición Animal</strong> — Catriló, La Pampa<br>
      <a href="mailto:mauricio@hmsnutricionanimal.com.ar" style="color:white; text-decoration:underline;">mauricio@hmsnutricionanimal.com.ar</a> · 2954-517407
    </div>
  </div>
</body></html>"""
    # Versión texto plano del contexto (sin HTML)
    if es_cliente_principal:
        if establecimiento:
            contexto_text = (
                f"Soy Mauricio Suárez, de HMS Nutrición Animal. "
                f"Acompañamos el seguimiento nutricional de la "
                f"hacienda de {establecimiento}. Desde hoy vas a "
                f"recibir en este correo las alertas climáticas del "
                f"campo: cuándo se viene calor, frío o lluvia que "
                f"puedan afectar el bienestar y el consumo de los "
                f"animales, y qué hacer para anticiparte."
            )
        else:
            contexto_text = (
                f"Soy Mauricio Suárez, de HMS Nutrición Animal. "
                f"Acompañamos el seguimiento nutricional de tu "
                f"hacienda. Desde hoy vas a recibir en este correo "
                f"las alertas climáticas del campo: cuándo se viene "
                f"calor, frío o lluvia que puedan afectar el "
                f"bienestar y el consumo de los animales, y qué "
                f"hacer para anticiparte."
            )
    else:
        rol_str = f" ({rol})" if rol else ""
        contexto_text = (
            f"Trabajás{rol_str} en {nombre_cliente}"
            f"{(' — ' + establecimiento) if establecimiento else ''}, "
            f"establecimiento que acompañamos desde HMS Nutrición "
            f"Animal en el seguimiento nutricional de la hacienda. "
            f"Desde hoy vas a recibir en este correo las alertas "
            f"climáticas del campo: cuándo se viene calor, frío o "
            f"lluvia que puedan afectar el bienestar y el consumo de "
            f"los animales, y qué hacer para anticiparte."
        )

    text = (
        f"{saludo},\n\n"
        f"{contexto_text}\n\n"
        f"¿POR QUÉ ESTE SISTEMA MARCA LA DIFERENCIA?\n"
        f"Este sistema toma el pronóstico y lo traduce en algo "
        f"concreto: CÓMO SE VA A COMPORTAR TU HACIENDA Y QUÉ HACER "
        f"para que el clima no se traduzca en pérdida de kilos.\n\n"
        f"Cada aviso une TRES MIRADAS QUE NORMALMENTE VIAJAN POR "
        f"SEPARADO: el pronóstico real para tu zona, LA DIETA QUE "
        f"ARMAMOS PARA TU LOTE (y cómo va a responder el animal según "
        f"categoría), y la experiencia de manejo a campo. Cuando "
        f"llegan juntas, dejan de ser un dato y pasan a ser una "
        f"decisión.\n\n"
        f"LO QUE GANÁS:\n"
        f"- Anticipación real: el aviso llega antes del evento, no "
        f"después.\n"
        f"- Acciones a medida del lote: pensadas para la categoría, "
        f"la dieta y la zona, no recetas genéricas.\n"
        f"- Kilos en el corral: cada día de estrés sin manejar es "
        f"aumento perdido. Los ajustes simples a tiempo lo evitan.\n"
        f"- Más presencia técnica: como no podemos estar todos los "
        f"días en los corrales, este sistema te extiende los ojos "
        f"sobre el campo.\n\n"
        f"¿QUÉ TE LLEGA?\n"
        f"- INFORME SEMANAL LOS LUNES: un panorama de lo que se viene "
        f"en los próximos días y cómo puede impactar a la hacienda.\n"
        f"- ACTUALIZACIÓN A MITAD DE SEMANA si el pronóstico cambia "
        f"fuerte respecto al del lunes — para que no te encuentres con "
        f"una sorpresa.\n"
        f"- AVISOS PUNTUALES los días que el clima puede afectar a los "
        f"animales, con las acciones concretas para ese día. Cuando no "
        f"hay riesgo, silencio.\n"
        f"- LECTURA DEL CAMPO: qué pasa con el consumo, con el rumen, "
        f"con los kilos — y qué conviene hacer.\n\n"
        f"¿NO QUERÉS RECIBIRLO MÁS?\n"
        f"Respondé este email con la palabra BAJA. Te damos de baja "
        f"automáticamente.\n\n"
        f"Consultas: mauricio@hmsnutricionanimal.com.ar - "
        f"WhatsApp 2954-517407.\n\n"
        f"— Mauricio Suárez —\nHMS Nutrición Animal\nInformación que anticipa. Decisiones que rinden."
    )
    return subject, html, text


def enviar_email_prueba(cfg: Dict, destinatario: str) -> Tuple[bool, str]:
    """Envía un email de prueba para validar la config SMTP."""
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    subject = f"✅ Prueba SMTP HMS Nutrición Animal — {fecha}"
    html = f"""<!DOCTYPE html>
<html><body style="font-family:Arial; padding:20px;">
  <div style="max-width:520px; margin:0 auto;">
    <div style="background:white; padding:18px; border-bottom:3px solid {COLOR_VERDE}; border-radius:6px 6px 0 0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td style="vertical-align:middle;">
            <div style="font-size:18px; font-weight:600; color:{COLOR_VERDE};">Prueba de configuración SMTP</div>
          </td>
          <td style="vertical-align:middle; text-align:right; width:170px;">
            <img src="cid:hms-logo" alt="HMS Nutrición Animal" style="height:105px; width:auto; max-width:260px; display:block; margin-left:auto; margin-right:-8px;">
          </td>
        </tr>
      </table>
    </div>
    <div style="background:#F8F8F8; padding:24px; border-radius:0 0 6px 6px;">
      <p style="color:{COLOR_VERDE}; font-size:16px;">✅ La configuración SMTP funciona correctamente.</p>
      <p style="color:{COLOR_GRIS};">
        Servidor: {cfg.get('host')}:{cfg.get('port')}<br>
        Remitente: {cfg.get('from_email')}<br>
        Hora: {fecha}
      </p>
      <p style="color:#999; font-size:12px;">
        El sistema ya está listo para enviar las alertas climáticas diarias.
      </p>
    </div>
  </div>
</body></html>"""
    text = (
        f"Prueba SMTP HMS Nutrición Animal — {fecha}\n\n"
        f"La configuración funciona correctamente.\n"
        f"Servidor: {cfg.get('host')}:{cfg.get('port')}\n"
        f"Remitente: {cfg.get('from_email')}\n"
    )
    # Prueba SMTP: no agregar BCC al admin (es un envío puntual, no a
    # un cliente).
    return enviar_email(
        cfg, [destinatario], subject, html, text,
        con_bcc_admin=False,
    )


def componer_alerta_stock_cliente(
    cliente: Dict, contacto: Dict,
    productos_bajos: List[Dict],
) -> Tuple[str, str, str]:
    """Email al cliente cuando uno o más productos HMS están por
    agotarse. Consolida todos los productos con stock bajo en un solo
    email (no manda uno por producto).

    Args:
        cliente: dict con nombre, establecimiento.
        contacto: dict con nombre, email del destinatario.
        productos_bajos: lista de dicts con lote_ident, producto,
            kg_restantes, consumo_kg_dia, dias_restantes,
            fecha_agotamiento. Producida por
            stock_producto.clientes_con_stock_bajo().

    Returns:
        (subject, html, text) listo para mandar.
    """
    nombre_dest_raw = (contacto.get("nombre") or "").strip()
    nombre_cliente = (cliente.get("nombre") or "").strip()
    if _parece_telefono_o_email(nombre_dest_raw) or not nombre_dest_raw:
        nombre_saludo = _primer_nombre(nombre_cliente) or "buen día"
    else:
        nombre_saludo = _primer_nombre(nombre_dest_raw)

    # Urgencia: mínimo de días entre todos los productos
    dias_min = min(
        (p.get("dias_restantes") or 0 for p in productos_bajos),
        default=999,
    )
    if dias_min <= 0:
        sev_emoji = "🔴"
        sev_titulo = "Stock AGOTADO"
        urgencia_txt = "ya se agotó alguno de los productos"
    elif dias_min <= 3:
        sev_emoji = "🔴"
        sev_titulo = "Stock crítico"
        urgencia_txt = (
            f"queda muy poco — apenas {dias_min} día(s) de autonomía"
        )
    elif dias_min <= 7:
        sev_emoji = "🟠"
        sev_titulo = "Stock bajo"
        urgencia_txt = (
            f"en {dias_min} días se está por terminar uno de los "
            f"productos"
        )
    else:
        sev_emoji = "🟡"
        sev_titulo = "Reposición a coordinar"
        urgencia_txt = (
            f"en aproximadamente {dias_min} días se está por agotar "
            f"un producto"
        )

    if len(productos_bajos) == 1:
        subject = (
            f"{sev_emoji} HMS — {productos_bajos[0]['producto']} "
            f"por agotarse ({dias_min} días)"
        )
    else:
        subject = (
            f"{sev_emoji} HMS — {len(productos_bajos)} productos "
            f"por reponer en tu campo"
        )

    # ───── Filas HTML por producto ─────
    filas_html = []
    for p in productos_bajos:
        dr = p.get("dias_restantes") or 0
        color = (
            "#A32D2D" if dr <= 7
            else ("#854F0B" if dr <= 14 else "#0F6E56")
        )
        bg = (
            "#FCEBEB" if dr <= 7
            else ("#FAEEDA" if dr <= 14 else "#EAF3DE")
        )
        filas_html.append(
            f"""<tr>
              <td style="padding:10px 12px; border-bottom:1px solid #E8E8E8;">
                <div style="font-weight:600; color:{COLOR_VERDE};">
                  {p['producto']}
                </div>
                <div style="font-size:12px; color:{COLOR_GRIS};">
                  Lote {p.get('lote_ident','?')}
                </div>
              </td>
              <td style="padding:10px 12px; border-bottom:1px solid #E8E8E8; text-align:right;">
                <div style="font-weight:600;">{p['kg_restantes']:.0f} kg</div>
                <div style="font-size:12px; color:{COLOR_GRIS};">
                  consumo {p['consumo_kg_dia']:.1f} kg/día
                </div>
              </td>
              <td style="padding:10px 12px; border-bottom:1px solid #E8E8E8; text-align:right;">
                <span style="background:{bg}; color:{color};
                  padding:4px 10px; border-radius:4px;
                  font-weight:600; font-size:13px;">
                  {dr} día(s)
                </span>
                <div style="font-size:11px; color:{COLOR_GRIS}; margin-top:3px;">
                  se acaba {p.get('fecha_agotamiento','—')}
                </div>
              </td>
            </tr>"""
        )

    # ───── Versión texto plano (para WhatsApp Mail y backups) ─────
    text_filas = []
    for p in productos_bajos:
        text_filas.append(
            f"- {p['producto']} (Lote {p.get('lote_ident','?')})\n"
            f"  Stock actual: {p['kg_restantes']:.0f} kg\n"
            f"  Consumo: {p['consumo_kg_dia']:.1f} kg/día\n"
            f"  Días restantes: {p.get('dias_restantes', 0)}\n"
            f"  Fecha estimada de agotamiento: "
            f"{p.get('fecha_agotamiento', '—')}"
        )

    html = f"""<!DOCTYPE html>
<html><body style="margin:0; padding:0; background:#F4F4F4;
  font-family:Arial,sans-serif;">
  <div style="max-width:640px; margin:0 auto; background:white;">
    <div style="background:white; padding:20px 24px; border-bottom:3px solid {COLOR_VERDE};">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td style="vertical-align:middle;">
            <div style="font-size:19px; font-weight:600; color:{COLOR_VERDE};">
              {sev_emoji} {sev_titulo} — Reposición de producto
            </div>
          </td>
          <td style="vertical-align:middle; text-align:right; width:170px;">
            <img src="cid:hms-logo" alt="HMS Nutrición Animal" style="height:105px; width:auto; max-width:260px; display:block; margin-left:auto; margin-right:-8px;">
          </td>
        </tr>
      </table>
    </div>
    <div style="padding:24px; color:{COLOR_GRIS}; font-size:14px;
      line-height:1.55;">
      <p style="font-size:15px;">Hola {nombre_saludo},</p>

      <p>
        Te aviso que <strong>{urgencia_txt}</strong> en tu campo. Te
        dejo el detalle abajo para que coordinemos la próxima
        entrega con tiempo:
      </p>

      <table style="width:100%; border-collapse:collapse;
        margin:14px 0; border-top:1px solid #E8E8E8;">
        <tbody>
          {''.join(filas_html)}
        </tbody>
      </table>

      <div style="background:#F0F8E8; border-left:3px solid {COLOR_LIMA};
        padding:12px 14px; margin:18px 0; border-radius:4px;">
        <strong style="color:{COLOR_VERDE};">📅 ¿Cómo seguimos?</strong>
        <p style="margin:6px 0 0 0;">
          Respondé este mail o escribime al WhatsApp
          <strong>2954-517407</strong> para coordinar la próxima
          entrega. Cuanto antes lo planifiquemos, mejor podemos
          asegurar que no se interrumpa la ración.
        </p>
      </div>

      <p style="font-size:12px; color:#888; margin-top:18px;">
        Este cálculo se hace con la dieta que tenés cargada en el
        sistema y el consumo diario proyectado del lote. Si cambiaste
        algo del manejo o las cantidades, avisame y lo ajustamos.
      </p>

      <p style="margin-top:24px; color:#888; font-size:12px;">
        — Mauricio Suárez —<br>
        HMS Nutrición Animal<br>
        <span style="font-style:italic;">
          Información que anticipa. Decisiones que rinden.
        </span>
      </p>
    </div>
    <div style="background:{COLOR_VERDE}; padding:12px 24px; color:white;
      font-size:11px; text-align:center;">
      <strong>HMS Nutrición Animal</strong> — Catriló, La Pampa<br>
      <a href="mailto:mauricio@hmsnutricionanimal.com.ar" style="color:white; text-decoration:underline;">mauricio@hmsnutricionanimal.com.ar</a> · 2954-517407
    </div>
  </div>
</body></html>"""

    text = (
        f"Hola {nombre_saludo},\n\n"
        f"Te aviso que {urgencia_txt} en tu campo. Detalle:\n\n"
        + "\n\n".join(text_filas)
        + "\n\n"
        f"¿CÓMO SEGUIMOS?\n"
        f"Respondé este mail o escribime al WhatsApp 2954-517407 "
        f"para coordinar la próxima entrega.\n\n"
        f"— Mauricio Suárez —\nHMS Nutrición Animal\nInformación que anticipa. Decisiones que rinden."
    )

    return subject, html, text


def componer_alerta_silocomedero_cliente(
    cliente: Dict, contacto: Dict, lotes_alerta: List[Dict],
) -> Tuple[str, str, str]:
    """Email al cliente cuando se está por terminar la carga del
    silocomedero en uno o más lotes. Es un aviso OPERATIVO: hay que
    preparar la próxima mezcla mañana (o ya).

    Diferencia con alerta de stock:
      - Stock = se acaba el producto HMS (logística / reposición).
      - Silocomedero = se acaba la mezcla cargada (trabajo del día —
        preparar el mixer).

    Args:
        cliente: dict con nombre, establecimiento.
        contacto: dict con nombre, email del destinatario.
        lotes_alerta: lista producida por
            stock_producto.lotes_silocomedero_proximos_agotamiento().
            Cada item tiene lote_ident, categoria, kg_cargados,
            kg_restantes, consumo_diario_kg, dias_restantes,
            fecha_agotamiento, fecha_carga.

    Returns:
        (subject, html, text).
    """
    nombre_dest_raw = (contacto.get("nombre") or "").strip()
    nombre_cliente = (cliente.get("nombre") or "").strip()
    if _parece_telefono_o_email(nombre_dest_raw) or not nombre_dest_raw:
        nombre_saludo = _primer_nombre(nombre_cliente) or "buen día"
    else:
        nombre_saludo = _primer_nombre(nombre_dest_raw)

    dias_min = min(
        (l.get("dias_restantes") or 0 for l in lotes_alerta),
        default=999,
    )
    if dias_min <= 0:
        sev_emoji = "🔴"
        sev_titulo = "Silocomedero — preparar mezcla HOY"
        urgencia_txt = "se agotó la carga del silocomedero"
    elif dias_min == 1:
        sev_emoji = "🟠"
        sev_titulo = "Silocomedero — preparar mezcla mañana"
        urgencia_txt = (
            "mañana se termina la carga del silocomedero"
        )
    else:
        sev_emoji = "🟡"
        sev_titulo = "Silocomedero — fin de carga próximo"
        urgencia_txt = (
            f"en {dias_min} días se termina la carga del silocomedero"
        )

    if len(lotes_alerta) == 1:
        l0 = lotes_alerta[0]
        subject = (
            f"{sev_emoji} HMS — Silo del lote "
            f"{l0.get('lote_ident', '?')} se agota "
            f"({dias_min} día{'s' if dias_min != 1 else ''})"
        )
    else:
        subject = (
            f"{sev_emoji} HMS — {len(lotes_alerta)} silocomederos "
            f"por recargar"
        )

    # ───── Filas HTML por lote ─────
    filas_html = []
    for l in lotes_alerta:
        dr = l.get("dias_restantes") or 0
        color = "#A32D2D" if dr <= 1 else "#854F0B"
        bg = "#FCEBEB" if dr <= 1 else "#FAEEDA"
        filas_html.append(
            f"""<tr>
              <td style="padding:10px 12px; border-bottom:1px solid #E8E8E8;">
                <div style="font-weight:600; color:{COLOR_VERDE};">
                  Lote {l.get('lote_ident', '?')}
                </div>
                <div style="font-size:12px; color:{COLOR_GRIS};">
                  {l.get('categoria') or '—'}
                </div>
              </td>
              <td style="padding:10px 12px; border-bottom:1px solid #E8E8E8; text-align:right;">
                <div style="font-weight:600;">
                  {l.get('kg_restantes', 0):.0f} kg
                </div>
                <div style="font-size:12px; color:{COLOR_GRIS};">
                  cargados {l.get('kg_cargados', 0):.0f} kg
                  el {l.get('fecha_carga', '—')}
                </div>
              </td>
              <td style="padding:10px 12px; border-bottom:1px solid #E8E8E8; text-align:right;">
                <span style="background:{bg}; color:{color};
                  padding:4px 10px; border-radius:4px;
                  font-weight:600; font-size:13px;">
                  {dr} día{'s' if dr != 1 else ''}
                </span>
                <div style="font-size:11px; color:{COLOR_GRIS}; margin-top:3px;">
                  se agota {l.get('fecha_agotamiento', '—')}
                </div>
                <div style="font-size:11px; color:{COLOR_GRIS}; margin-top:2px;">
                  consumo {l.get('consumo_diario_kg', 0):.0f} kg/día
                </div>
              </td>
            </tr>"""
        )

    # ───── Texto plano ─────
    text_filas = []
    for l in lotes_alerta:
        text_filas.append(
            f"- Lote {l.get('lote_ident', '?')} "
            f"({l.get('categoria') or '—'})\n"
            f"  Carga: {l.get('kg_cargados', 0):.0f} kg "
            f"el {l.get('fecha_carga', '—')}\n"
            f"  Restan: {l.get('kg_restantes', 0):.0f} kg "
            f"(consumo {l.get('consumo_diario_kg', 0):.0f} kg/día)\n"
            f"  Se agota: {l.get('fecha_agotamiento', '—')} "
            f"({l.get('dias_restantes', 0)} día(s))"
        )

    html = f"""<!DOCTYPE html>
<html><body style="margin:0; padding:0; background:#F4F4F4;
  font-family:Arial,sans-serif;">
  <div style="max-width:640px; margin:0 auto; background:white;">
    <div style="background:white; padding:20px 24px; border-bottom:3px solid {COLOR_VERDE};">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td style="vertical-align:middle;">
            <div style="font-size:19px; font-weight:600; color:{COLOR_VERDE};">
              {sev_emoji} {sev_titulo}
            </div>
          </td>
          <td style="vertical-align:middle; text-align:right; width:170px;">
            <img src="cid:hms-logo" alt="HMS Nutrición Animal" style="height:105px; width:auto; max-width:260px; display:block; margin-left:auto; margin-right:-8px;">
          </td>
        </tr>
      </table>
    </div>
    <div style="padding:24px; color:{COLOR_GRIS}; font-size:14px;
      line-height:1.55;">
      <p style="font-size:15px;">Hola {nombre_saludo},</p>

      <p>
        Te aviso que <strong>{urgencia_txt}</strong>. Te dejo el detalle
        abajo para que puedas preparar la próxima mezcla a tiempo y los
        animales no queden sin ración:
      </p>

      <table style="width:100%; border-collapse:collapse;
        margin:14px 0; border-top:1px solid #E8E8E8;">
        <tbody>
          {''.join(filas_html)}
        </tbody>
      </table>

      <div style="background:#F0F8E8; border-left:3px solid {COLOR_LIMA};
        padding:12px 14px; margin:18px 0; border-radius:4px;">
        <strong style="color:{COLOR_VERDE};">⚙️ Cómo seguir</strong>
        <p style="margin:6px 0 0 0;">
          Preparar la próxima mezcla para que los animales no queden
          sin ración. Si necesitás reponer producto antes de la
          próxima carga, escribime al WhatsApp
          <strong>2954-517407</strong> y lo coordinamos.
        </p>
      </div>

      <p style="font-size:12px; color:#888; margin-top:18px;">
        Este cálculo lo hago en base a la última carga registrada y al
        consumo diario que sale de la dieta del lote. Si cambiaste algo
        del manejo o de las cantidades, avisame y lo actualizo.
      </p>

      <p style="margin-top:24px; color:#888; font-size:12px;">
        — Mauricio Suárez —<br>
        HMS Nutrición Animal<br>
        <span style="font-style:italic;">
          Información que anticipa. Decisiones que rinden.
        </span>
      </p>
    </div>
    <div style="background:{COLOR_VERDE}; padding:12px 24px; color:white;
      font-size:11px; text-align:center;">
      <strong>HMS Nutrición Animal</strong> — Catriló, La Pampa<br>
      <a href="mailto:mauricio@hmsnutricionanimal.com.ar" style="color:white; text-decoration:underline;">mauricio@hmsnutricionanimal.com.ar</a> · 2954-517407
    </div>
  </div>
</body></html>"""

    text = (
        f"Hola {nombre_saludo},\n\n"
        f"Te aviso que {urgencia_txt}. Detalle:\n\n"
        + "\n\n".join(text_filas)
        + "\n\n"
        f"CÓMO SEGUIR\n"
        f"Preparar la próxima mezcla. Si necesitás reponer producto "
        f"antes de la próxima carga, escribime al 2954-517407 y lo "
        f"coordinamos.\n\n"
        f"— Mauricio Suárez —\nHMS Nutrición Animal\nInformación que anticipa. Decisiones que rinden."
    )

    return subject, html, text


def componer_alerta_cambio_fase_cliente(
    cliente: Dict, contacto: Dict, cambios: List[Dict],
) -> Tuple[str, str, str]:
    """Email al cliente avisando que mañana arranca una nueva fase del
    plan de adaptación. Muestra el diff de ingredientes (qué cambia y
    cuánto) para que el productor pueda preparar la mezcla nueva.

    Args:
        cliente: dict con nombre, establecimiento.
        contacto: dict con nombre, email del destinatario.
        cambios: lista producida por
            stock_producto.lotes_con_cambio_fase_proximo(). Cada item
            tiene lote_ident, categoria, fecha_cambio, fase_actual,
            fase_nueva, diff.

    Returns:
        (subject, html, text).
    """
    nombre_dest_raw = (contacto.get("nombre") or "").strip()
    nombre_cliente = (cliente.get("nombre") or "").strip()
    if _parece_telefono_o_email(nombre_dest_raw) or not nombre_dest_raw:
        nombre_saludo = _primer_nombre(nombre_cliente) or "buen día"
    else:
        nombre_saludo = _primer_nombre(nombre_dest_raw)

    if len(cambios) == 1:
        c0 = cambios[0]
        subject = (
            f"📋 HMS — Mañana cambia la fase del lote "
            f"{c0.get('lote_ident', '?')}"
        )
    else:
        subject = (
            f"📋 HMS — Mañana cambian {len(cambios)} fases de plan "
            f"de adaptación"
        )

    # Importamos el detector de forrajes a libre disposición
    from src.stock_producto import _es_a_discrecion

    # ───── Bloques HTML, uno por lote con cambio ─────
    bloques_html = []
    bloques_text = []
    for cam in cambios:
        fase_a = cam["fase_actual"]
        fase_n = cam["fase_nueva"]
        nombre_fase_a = fase_a.get("observaciones") or "fase actual"
        nombre_fase_n = fase_n.get("observaciones") or "fase nueva"

        # Construimos un índice de la fase ACTUAL para poder buscar
        # qué cantidad tenía cada ingrediente antes.
        idx_actual = {}
        for c in fase_a.get("composicion") or []:
            nm = (c.get("nombre") or c.get("ingrediente") or "").strip()
            if nm:
                idx_actual[nm.lower()] = c

        # La tabla muestra TODA la composición nueva — para que el
        # productor pueda preparar la mezcla sin tener que consultar
        # la dieta aparte. Para los forrajes a libre disposición no
        # mostramos kg (el animal regula). Para los que cambian, el
        # delta queda visible; para los que no, columna "=".
        filas = []
        hay_libre_disposicion = False
        comp_nueva = fase_n.get("composicion") or []
        for c_n in comp_nueva:
            nombre = (
                c_n.get("nombre") or c_n.get("ingrediente") or "?"
            )
            kg_nueva = float(c_n.get("kg_tal_cual") or 0)
            c_a = idx_actual.get(nombre.lower(), {})
            kg_actual = float(c_a.get("kg_tal_cual") or 0)
            delta = kg_nueva - kg_actual

            es_libre = _es_a_discrecion(nombre)
            if es_libre:
                hay_libre_disposicion = True
                cel_actual = (
                    f"<span style='color:#888;'>a libre disposición</span>"
                )
                cel_nueva = (
                    f"<span style='color:{COLOR_VERDE};'>"
                    f"a libre disposición</span>"
                )
                cel_delta = (
                    f"<span style='color:#888; font-size:11px;'>"
                    f"el animal regula</span>"
                )
            else:
                cel_actual = (
                    f"{kg_actual:.2f} kg" if kg_actual > 0
                    else "<span style='color:#888;'>—</span>"
                )
                cel_nueva = (
                    f"<strong>{kg_nueva:.2f} kg</strong>"
                    if kg_nueva > 0
                    else "<span style='color:#888;'>—</span>"
                )
                if abs(delta) < 0.05:
                    cel_delta = (
                        f"<span style='color:#666;'>=</span>"
                    )
                else:
                    signo = "+" if delta > 0 else ""
                    color = (
                        "#0F6E56" if delta > 0 else "#A32D2D"
                    )
                    cel_delta = (
                        f"<span style='color:{color};"
                        f" font-weight:600;'>{signo}{delta:.2f} kg</span>"
                    )
            filas.append(
                f"""<tr>
                  <td style="padding:8px 10px; border-bottom:1px solid #EEE;">
                    <strong>{nombre}</strong>
                  </td>
                  <td style="padding:8px 10px; border-bottom:1px solid #EEE; text-align:right; color:#666;">
                    {cel_actual}
                  </td>
                  <td style="padding:8px 10px; border-bottom:1px solid #EEE; text-align:right;">
                    {cel_nueva}
                  </td>
                  <td style="padding:8px 10px; border-bottom:1px solid #EEE; text-align:right;">
                    {cel_delta}
                  </td>
                </tr>"""
            )

        # También mostramos los que SOLO estaban en la fase actual y se
        # sacan (delta negativo total). Útil si una fase elimina un
        # ingrediente.
        for c_a in fase_a.get("composicion") or []:
            nm = (
                c_a.get("nombre") or c_a.get("ingrediente") or ""
            ).strip()
            if not nm:
                continue
            ya_listado = any(
                ((c.get("nombre") or c.get("ingrediente") or "")
                 .strip().lower() == nm.lower())
                for c in comp_nueva
            )
            if ya_listado:
                continue
            kg_actual = float(c_a.get("kg_tal_cual") or 0)
            es_libre = _es_a_discrecion(nm)
            if es_libre:
                cel_actual = (
                    f"<span style='color:#888;'>a libre disposición</span>"
                )
                cel_nueva = (
                    f"<span style='color:#A32D2D;'>se saca</span>"
                )
                cel_delta = (
                    f"<span style='color:#A32D2D; font-size:11px;'>"
                    f"no va más</span>"
                )
            else:
                cel_actual = f"{kg_actual:.2f} kg"
                cel_nueva = (
                    f"<span style='color:#A32D2D;'>se saca</span>"
                )
                cel_delta = (
                    f"<span style='color:#A32D2D; font-weight:600;'>"
                    f"-{kg_actual:.2f} kg</span>"
                )
            filas.append(
                f"""<tr>
                  <td style="padding:8px 10px; border-bottom:1px solid #EEE;">
                    <strong>{nm}</strong>
                  </td>
                  <td style="padding:8px 10px; border-bottom:1px solid #EEE; text-align:right; color:#666;">
                    {cel_actual}
                  </td>
                  <td style="padding:8px 10px; border-bottom:1px solid #EEE; text-align:right;">
                    {cel_nueva}
                  </td>
                  <td style="padding:8px 10px; border-bottom:1px solid #EEE; text-align:right;">
                    {cel_delta}
                  </td>
                </tr>"""
            )

        tabla_html = (
            f"""<table style="width:100%; border-collapse:collapse;
              margin:8px 0 12px 0; font-size:13px;">
              <thead>
                <tr style="background:#F4F4F4;">
                  <th style="padding:8px 10px; text-align:left; color:#666;">
                    Ingrediente
                  </th>
                  <th style="padding:8px 10px; text-align:right; color:#666;">
                    Hoy
                  </th>
                  <th style="padding:8px 10px; text-align:right; color:#666;">
                    Mañana
                  </th>
                  <th style="padding:8px 10px; text-align:right; color:#666;">
                    Δ
                  </th>
                </tr>
              </thead>
              <tbody>
                {''.join(filas)}
              </tbody>
            </table>"""
        )
        nota_libre = ""
        if hay_libre_disposicion:
            nota_libre = (
                f"<p style='font-size:11px; color:#888; "
                f"margin:0 0 10px 0;'>"
                f"Los forrajes \"a libre disposición\" no se "
                f"preparan en kg medidos — quedan en el corral y el "
                f"animal regula su consumo."
                f"</p>"
            )

        # ─── Bloque "Total a preparar mañana" (kg × animales) ───
        # Es lo realmente operativo: cuántos kg de cada ingrediente
        # se carga al silo/mixer para todo el lote. Solo ingredientes
        # medibles (sin forrajes a libre disposición).
        cant_anim = int(cam.get("cantidad_animales") or 0)
        bloque_total = ""
        if cant_anim > 0:
            filas_total = []
            total_general = 0.0
            for c_n in comp_nueva:
                nombre = (
                    c_n.get("nombre") or c_n.get("ingrediente") or "?"
                )
                if _es_a_discrecion(nombre):
                    continue
                kg_unit = float(c_n.get("kg_tal_cual") or 0)
                if kg_unit <= 0:
                    continue
                kg_lote = kg_unit * cant_anim
                total_general += kg_lote
                filas_total.append(
                    f"""<tr>
                      <td style="padding:6px 10px;">
                        <strong>{nombre}</strong>
                      </td>
                      <td style="padding:6px 10px; text-align:right;
                        color:#666; font-size:12px;">
                        {kg_unit:.2f} kg/animal
                      </td>
                      <td style="padding:6px 10px; text-align:right;
                        font-size:15px; font-weight:600;
                        color:{COLOR_VERDE};">
                        {kg_lote:,.0f} kg
                      </td>
                    </tr>""".replace(",", ".")
                )

            if filas_total:
                bloque_total = (
                    f"""<div style="background:#FFFBE6;
                      border:1px solid #F0D679; border-radius:6px;
                      padding:12px 14px; margin:6px 0 14px 0;">
                      <div style="font-size:13px; font-weight:600;
                        color:#6F5402; margin-bottom:6px;">
                        🧮 Total a preparar mañana
                        <span style="color:#9C7E27; font-weight:400;">
                          (lote de {cant_anim} animales)
                        </span>
                      </div>
                      <table style="width:100%;
                        border-collapse:collapse;">
                        <tbody>
                          {''.join(filas_total)}
                          <tr style="border-top:2px solid #F0D679;">
                            <td style="padding:8px 10px;
                              font-weight:600; color:#6F5402;">
                              Mezcla total
                            </td>
                            <td></td>
                            <td style="padding:8px 10px;
                              text-align:right; font-size:16px;
                              font-weight:700; color:#6F5402;">
                              {total_general:,.0f} kg
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>""".replace(",", ".")
                )

        # Duración de la fase nueva:
        #   - Si hay siguiente fase → rango entre las dos.
        #   - Si es la última → usa objetivo_fecha del lote (fecha de
        #     salida planificada del ciclo).
        #   - Si no hay objetivo_fecha → aviso para cargarla.
        fecha_fin = fase_n.get("fecha_fin")
        dur_dias = fase_n.get("duracion_dias")
        if fecha_fin and dur_dias:
            rango_txt = (
                f"Va del <strong>{cam['fecha_cambio']}</strong> al "
                f"<strong>{fecha_fin}</strong> "
                f"(<strong>{dur_dias} días</strong>)."
            )
        else:
            rango_txt = (
                f"Arranca el <strong>{cam['fecha_cambio']}</strong>. "
                f"<span style='color:#A06200;'>Para calcular la "
                f"duración cargá la fecha objetivo de salida del lote "
                f"en su ficha — el sistema actualiza el rango "
                f"automáticamente.</span>"
            )

        bloques_html.append(
            f"""<div style="border:1px solid #E8E8E8; border-radius:6px;
              padding:16px; margin:14px 0;">
              <div style="font-size:15px; font-weight:600;
                color:{COLOR_VERDE};">
                Lote {cam.get('lote_ident', '?')}
                <span style="color:{COLOR_GRIS}; font-weight:400;
                  font-size:13px;">
                  · {cam.get('categoria') or '—'}
                </span>
              </div>
              <div style="font-size:13px; color:{COLOR_GRIS};
                margin:4px 0 4px 0;">
                Pasa de <strong>{nombre_fase_a}</strong> a
                <strong>{nombre_fase_n}</strong>.
              </div>
              <div style="font-size:13px; color:{COLOR_GRIS};
                margin:0 0 10px 0;">
                {rango_txt}
              </div>
              {tabla_html}
              {nota_libre}
              {bloque_total}
            </div>"""
        )

        # ─── Texto plano: misma idea ───
        if fecha_fin and dur_dias:
            rango_text = (
                f"  Va del {cam['fecha_cambio']} al {fecha_fin} "
                f"({dur_dias} días)"
            )
        else:
            rango_text = (
                f"  Arranca el {cam['fecha_cambio']} — falta cargar "
                f"la fecha objetivo de salida del lote para calcular "
                f"la duración"
            )
        text_filas = [
            f"- Lote {cam.get('lote_ident', '?')} "
            f"({cam.get('categoria') or '—'})",
            f"  {nombre_fase_a} → {nombre_fase_n}",
            rango_text,
            f"  Composición nueva (por animal/día, tal cual):",
        ]
        for c_n in comp_nueva:
            nombre = (
                c_n.get("nombre") or c_n.get("ingrediente") or "?"
            )
            kg_nueva = float(c_n.get("kg_tal_cual") or 0)
            c_a = idx_actual.get(nombre.lower(), {})
            kg_actual = float(c_a.get("kg_tal_cual") or 0)
            delta = kg_nueva - kg_actual
            if _es_a_discrecion(nombre):
                text_filas.append(
                    f"    • {nombre}: a libre disposición "
                    f"(el animal regula)"
                )
            else:
                if abs(delta) < 0.05:
                    text_filas.append(
                        f"    • {nombre}: {kg_nueva:.2f} kg (igual)"
                    )
                else:
                    signo = "+" if delta > 0 else ""
                    text_filas.append(
                        f"    • {nombre}: {kg_actual:.2f} → "
                        f"{kg_nueva:.2f} kg ({signo}{delta:.2f})"
                    )
        # ingredientes que se sacan
        for c_a in fase_a.get("composicion") or []:
            nm = (
                c_a.get("nombre") or c_a.get("ingrediente") or ""
            ).strip()
            if not nm:
                continue
            ya = any(
                ((c.get("nombre") or c.get("ingrediente") or "")
                 .strip().lower() == nm.lower())
                for c in comp_nueva
            )
            if ya:
                continue
            text_filas.append(f"    • {nm}: se saca (no va más)")

        # Total por lote en texto plano
        if cant_anim > 0:
            tot_general = 0.0
            text_filas.append("")
            text_filas.append(
                f"  TOTAL A PREPARAR MAÑANA "
                f"(lote de {cant_anim} animales):"
            )
            for c_n in comp_nueva:
                nombre = (
                    c_n.get("nombre")
                    or c_n.get("ingrediente") or "?"
                )
                if _es_a_discrecion(nombre):
                    continue
                kg_unit = float(c_n.get("kg_tal_cual") or 0)
                if kg_unit <= 0:
                    continue
                kg_lote = kg_unit * cant_anim
                tot_general += kg_lote
                text_filas.append(
                    f"    • {nombre}: {kg_lote:.0f} kg"
                )
            if tot_general > 0:
                text_filas.append(
                    f"    → Mezcla total: {tot_general:.0f} kg"
                )
        bloques_text.append("\n".join(text_filas))

    html = f"""<!DOCTYPE html>
<html><body style="margin:0; padding:0; background:#F4F4F4;
  font-family:Arial,sans-serif;">
  <div style="max-width:640px; margin:0 auto; background:white;">
    <div style="background:white; padding:20px 24px; border-bottom:3px solid {COLOR_VERDE};">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td style="vertical-align:middle;">
            <div style="font-size:19px; font-weight:600; color:{COLOR_VERDE};">
              📋 Mañana cambia la fase de adaptación
            </div>
          </td>
          <td style="vertical-align:middle; text-align:right; width:170px;">
            <img src="cid:hms-logo" alt="HMS Nutrición Animal" style="height:105px; width:auto; max-width:260px; display:block; margin-left:auto; margin-right:-8px;">
          </td>
        </tr>
      </table>
    </div>
    <div style="padding:24px; color:{COLOR_GRIS}; font-size:14px;
      line-height:1.55;">
      <p style="font-size:15px;">Hola {nombre_saludo},</p>

      <p>
        Te aviso que <strong>mañana arranca una fase nueva del plan
        de adaptación</strong>. Te dejo el detalle de qué cambia para
        que prepares la mezcla con la nueva proporción:
      </p>

      {''.join(bloques_html)}

      <div style="background:#F0F8E8; border-left:3px solid {COLOR_LIMA};
        padding:12px 14px; margin:18px 0; border-radius:4px;">
        <strong style="color:{COLOR_VERDE};">⚙️ Cómo seguir</strong>
        <p style="margin:6px 0 0 0;">
          Preparar la próxima mezcla con la composición nueva. Si
          tenés dudas con la transición o querés revisar las cantidades,
          escribime al WhatsApp <strong>2954-517407</strong>.
        </p>
      </div>

      <p style="font-size:12px; color:#888; margin-top:18px;">
        Los kg de la tabla son por animal por día (tal cual, no MS),
        y el bloque "Total a preparar mañana" ya multiplica por la
        cantidad de animales del lote — ese es el que usás para cargar
        la mezcla. Si cambió algo en el manejo, avisame y lo ajusto.
      </p>

      <p style="margin-top:24px; color:#888; font-size:12px;">
        — Mauricio Suárez —<br>
        HMS Nutrición Animal<br>
        <span style="font-style:italic;">
          Información que anticipa. Decisiones que rinden.
        </span>
      </p>
    </div>
    <div style="background:{COLOR_VERDE}; padding:12px 24px; color:white;
      font-size:11px; text-align:center;">
      <strong>HMS Nutrición Animal</strong> — Catriló, La Pampa<br>
      <a href="mailto:mauricio@hmsnutricionanimal.com.ar" style="color:white; text-decoration:underline;">mauricio@hmsnutricionanimal.com.ar</a> · 2954-517407
    </div>
  </div>
</body></html>"""

    text = (
        f"Hola {nombre_saludo},\n\n"
        f"Te aviso que mañana arranca una fase nueva del plan de "
        f"adaptación. Detalle:\n\n"
        + "\n\n".join(bloques_text)
        + "\n\n"
        f"CÓMO SEGUIR\n"
        f"Preparar la mezcla con esas cantidades totales. Si tenés "
        f"dudas, escribime al 2954-517407.\n\n"
        f"— Mauricio Suárez —\nHMS Nutrición Animal\nInformación que anticipa. Decisiones que rinden."
    )

    return subject, html, text


def _kpis_pv_html(lt: Dict) -> str:
    """Mini-bloque con kg mezcla/animal y % del peso vivo.

    Se muestra dentro del bloque del lote, debajo de la tabla. Si no
    hay peso vivo estimado, sólo mostramos kg/animal sin el %.
    """
    kg_anim = lt.get("kg_mezcla_animal_dia") or 0
    pv = lt.get("peso_vivo_estimado_kg") or 0
    pct_pv = lt.get("pct_pv_mezcla") or 0
    if kg_anim <= 0:
        return ""
    if pv > 0 and pct_pv > 0:
        return (
            f"<div style='font-size:12px; margin-top:8px;"
            f" padding:8px 12px; background:#F4F8F1;"
            f" border-left:3px solid {COLOR_VERDE};"
            f" border-radius:3px;'>"
            f"🐄 <strong>{kg_anim:.2f} kg de mezcla por animal/día</strong>"
            f" · {pct_pv:.2f}% del peso vivo"
            f" <span style='color:#888;'>(PV estimado "
            f"{pv:,.0f} kg)</span>"
            f"</div>"
        ).replace(",", ".")
    # Sin PV: sólo el kg/animal/día
    return (
        f"<div style='font-size:12px; margin-top:8px;"
        f" padding:8px 12px; background:#F4F8F1;"
        f" border-left:3px solid {COLOR_VERDE};"
        f" border-radius:3px;'>"
        f"🐄 <strong>{kg_anim:.2f} kg de mezcla por animal/día</strong>"
        f" <span style='color:#888;'>(sin peso vivo cargado para "
        f"calcular % PV)</span>"
        f"</div>"
    )


def componer_informe_demanda_cliente(
    cliente: Dict, demanda: Dict,
) -> Tuple[str, str, str]:
    """Informe INTERNO de planificación logística por cliente.

    No es para enviar al cliente — es para Mauricio. Resume:
      - KPIs cabecera (lotes activos, animales totales, mezcla/día).
      - Por cada lote/corral: tabla con kg por animal y por lote.
      - Total cliente consolidado (kg/día, /semana, /mes), separando
        productos HMS (los que él vende) del resto.

    Args:
        cliente: dict con nombre, establecimiento.
        demanda: dict producido por
            stock_producto.demanda_insumos_cliente().

    Returns:
        (subject, html, text).
    """
    nombre_cli = (cliente.get("nombre") or "").strip()
    est = (cliente.get("establecimiento") or "").strip()
    fecha_ref = demanda.get("fecha_referencia") or ""
    tot = demanda.get("total_cliente") or {}
    lotes = demanda.get("lotes") or []
    n_lotes = len(lotes)
    n_animales = tot.get("cantidad_animales_total") or 0
    mezcla_total = tot.get("mezcla_total_kg_dia") or 0

    subject = (
        f"📊 Demanda de insumos — {nombre_cli} "
        f"({n_lotes} lote{'s' if n_lotes != 1 else ''}, "
        f"{n_animales} cab.)"
    )

    # ───── KPIs cabecera ─────
    kpis_html = f"""
    <table style="width:100%; border-collapse:collapse; margin:14px 0;">
      <tr>
        <td style="width:33%; padding:14px; background:#F4F8F1;
          border-radius:6px; text-align:center;">
          <div style="font-size:11px; color:{COLOR_GRIS};
            text-transform:uppercase; letter-spacing:0.5px;">
            Lotes activos
          </div>
          <div style="font-size:24px; font-weight:700;
            color:{COLOR_VERDE}; margin-top:4px;">
            {n_lotes}
          </div>
        </td>
        <td style="width:8px;"></td>
        <td style="width:33%; padding:14px; background:#F4F8F1;
          border-radius:6px; text-align:center;">
          <div style="font-size:11px; color:{COLOR_GRIS};
            text-transform:uppercase; letter-spacing:0.5px;">
            Animales
          </div>
          <div style="font-size:24px; font-weight:700;
            color:{COLOR_VERDE}; margin-top:4px;">
            {n_animales}
          </div>
        </td>
        <td style="width:8px;"></td>
        <td style="width:34%; padding:14px; background:#FFFBE6;
          border-radius:6px; text-align:center;">
          <div style="font-size:11px; color:#6F5402;
            text-transform:uppercase; letter-spacing:0.5px;">
            Mezcla / día
          </div>
          <div style="font-size:24px; font-weight:700;
            color:#6F5402; margin-top:4px;">
            {mezcla_total:,.0f} kg
          </div>
        </td>
      </tr>
    </table>""".replace(",", ".")

    # ───── Bloques por lote ─────
    bloques_lote_html = []
    bloques_lote_text = []
    for lt in lotes:
        fase = lt.get("fase_vigente") or ""
        fase_html = (
            f" <span style='color:{COLOR_GRIS}; font-weight:400;'>"
            f"· {fase}</span>" if fase else ""
        )
        # Filas tabla
        filas_html = []
        for ing in lt.get("ingredientes") or []:
            if ing.get("es_libre_disposicion"):
                pct_cel = "<span style='color:#888;'>—</span>"
                kg_an = "<span style='color:#888;'>libre disp.</span>"
                kg_dia = "<span style='color:#888;'>—</span>"
                kg_sem = "<span style='color:#888;'>—</span>"
            else:
                pct_val = ing.get("pct_mezcla") or 0
                pct_cel = (
                    f"<strong style='color:{COLOR_VERDE};'>"
                    f"{pct_val:.1f}%</strong>"
                )
                kg_an = f"{ing['kg_animal_dia']:.2f} kg"
                kg_dia = (
                    f"<strong>{ing['kg_lote_dia']:,.1f} kg</strong>"
                    .replace(",", ".")
                )
                kg_sem = (
                    f"{ing['kg_lote_semana']:,.0f} kg"
                    .replace(",", ".")
                )
            badge_hms = (
                f"<span style='background:{COLOR_VERDE};"
                f" color:white; padding:1px 6px; border-radius:3px;"
                f" font-size:10px; font-weight:600;"
                f" margin-left:6px;'>HMS</span>"
                if ing.get("es_hms") else ""
            )
            filas_html.append(
                f"""<tr>
                  <td style="padding:6px 10px; border-bottom:1px solid #EEE;">
                    {ing['nombre']}{badge_hms}
                  </td>
                  <td style="padding:6px 10px; border-bottom:1px solid #EEE; text-align:right;">
                    {pct_cel}
                  </td>
                  <td style="padding:6px 10px; border-bottom:1px solid #EEE; text-align:right; color:#666;">
                    {kg_an}
                  </td>
                  <td style="padding:6px 10px; border-bottom:1px solid #EEE; text-align:right;">
                    {kg_dia}
                  </td>
                  <td style="padding:6px 10px; border-bottom:1px solid #EEE; text-align:right; color:#666;">
                    {kg_sem}
                  </td>
                </tr>"""
            )

        bloques_lote_html.append(
            f"""<div style="border:1px solid #E8E8E8; border-radius:6px;
              padding:14px; margin:12px 0;">
              <div style="font-size:14px; font-weight:600;
                color:{COLOR_VERDE};">
                {lt.get('lote_ident', '?')}
                <span style="color:{COLOR_GRIS}; font-weight:400;">
                  — {lt.get('categoria') or '—'} ·
                  {lt.get('cantidad_animales', 0)} cab.{fase_html}
                </span>
              </div>
              <table style="width:100%; border-collapse:collapse;
                margin:8px 0 4px 0; font-size:13px;">
                <thead>
                  <tr style="background:#F4F4F4;">
                    <th style="padding:6px 10px; text-align:left; color:#666;">
                      Ingrediente
                    </th>
                    <th style="padding:6px 10px; text-align:right; color:#666;">
                      % mezcla
                    </th>
                    <th style="padding:6px 10px; text-align:right; color:#666;">
                      kg/animal·día
                    </th>
                    <th style="padding:6px 10px; text-align:right; color:#666;">
                      kg/lote·día
                    </th>
                    <th style="padding:6px 10px; text-align:right; color:#666;">
                      kg/lote·sem
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(filas_html)}
                </tbody>
              </table>
              <div style="font-size:11px; color:#888; margin-top:4px;">
                Mezcla del lote:
                <strong>{lt.get('mezcla_total_kg_dia', 0):,.1f}
                kg/día</strong> (sin contar libre disposición)
              </div>
              {_kpis_pv_html(lt)}
            </div>""".replace(",", ".")
        )

        # Texto plano por lote
        t_filas = [
            f"  {lt.get('lote_ident', '?')} — "
            f"{lt.get('categoria') or '—'} · "
            f"{lt.get('cantidad_animales', 0)} cab. "
            f"{('· ' + fase) if fase else ''}".rstrip()
        ]
        for ing in lt.get("ingredientes") or []:
            tag_hms = " [HMS]" if ing.get("es_hms") else ""
            if ing.get("es_libre_disposicion"):
                t_filas.append(
                    f"    • {ing['nombre']}{tag_hms}: "
                    f"a libre disposición"
                )
            else:
                pct_val = ing.get("pct_mezcla") or 0
                t_filas.append(
                    f"    • {ing['nombre']}{tag_hms} "
                    f"({pct_val:.1f}% de la mezcla): "
                    f"{ing['kg_animal_dia']:.2f} kg/animal × "
                    f"{lt.get('cantidad_animales', 0)} = "
                    f"{ing['kg_lote_dia']:.1f} kg/día "
                    f"({ing['kg_lote_semana']:.0f} kg/semana)"
                )
        t_filas.append(
            f"    Mezcla del lote: "
            f"{lt.get('mezcla_total_kg_dia', 0):.1f} kg/día"
        )
        _kg_anim = lt.get("kg_mezcla_animal_dia") or 0
        _pv = lt.get("peso_vivo_estimado_kg") or 0
        _pct_pv = lt.get("pct_pv_mezcla") or 0
        if _kg_anim > 0:
            if _pv > 0 and _pct_pv > 0:
                t_filas.append(
                    f"    Mezcla por animal: {_kg_anim:.2f} kg/día "
                    f"({_pct_pv:.2f}% PV, PV est. {_pv:.0f} kg)"
                )
            else:
                t_filas.append(
                    f"    Mezcla por animal: {_kg_anim:.2f} kg/día"
                )
        bloques_lote_text.append("\n".join(t_filas))

    # ───── Total cliente consolidado ─────
    filas_tot_html = []
    for ing in tot.get("ingredientes") or []:
        if ing.get("es_libre_disposicion"):
            kg_d = "<span style='color:#888;'>libre disp.</span>"
            kg_s = "<span style='color:#888;'>—</span>"
            kg_m = "<span style='color:#888;'>—</span>"
        else:
            kg_d = (
                f"<strong>{ing['kg_dia']:,.1f}</strong>"
                .replace(",", ".")
            )
            kg_s = (
                f"{ing['kg_semana']:,.0f}".replace(",", ".")
            )
            kg_m = (
                f"{ing['kg_mes']:,.0f}".replace(",", ".")
            )
        badge_hms = (
            f"<span style='background:{COLOR_VERDE};"
            f" color:white; padding:1px 6px; border-radius:3px;"
            f" font-size:10px; font-weight:600;"
            f" margin-left:6px;'>HMS</span>"
            if ing.get("es_hms") else ""
        )
        filas_tot_html.append(
            f"""<tr>
              <td style="padding:8px 10px; border-bottom:1px solid #EEE;">
                {ing['nombre']}{badge_hms}
                <span style="color:#888; font-size:11px;">
                  · {ing.get('lotes_que_lo_usan', 0)} lote(s)
                </span>
              </td>
              <td style="padding:8px 10px; border-bottom:1px solid #EEE; text-align:right;">
                {kg_d}
              </td>
              <td style="padding:8px 10px; border-bottom:1px solid #EEE; text-align:right; color:#666;">
                {kg_s}
              </td>
              <td style="padding:8px 10px; border-bottom:1px solid #EEE; text-align:right; color:#666;">
                {kg_m}
              </td>
            </tr>"""
        )

    bloque_total_html = f"""<div style="border:2px solid #F0D679;
      border-radius:6px; padding:16px; margin:18px 0;
      background:#FFFBE6;">
      <div style="font-size:14px; font-weight:600; color:#6F5402;
        margin-bottom:8px;">
        🧮 Total cliente — suma de todos los lotes
      </div>
      <table style="width:100%; border-collapse:collapse;
        font-size:13px;">
        <thead>
          <tr style="background:#FAEFC2;">
            <th style="padding:8px 10px; text-align:left; color:#6F5402;">
              Ingrediente
            </th>
            <th style="padding:8px 10px; text-align:right; color:#6F5402;">
              kg / día
            </th>
            <th style="padding:8px 10px; text-align:right; color:#6F5402;">
              kg / semana
            </th>
            <th style="padding:8px 10px; text-align:right; color:#6F5402;">
              kg / mes
            </th>
          </tr>
        </thead>
        <tbody>
          {''.join(filas_tot_html)}
          <tr style="border-top:2px solid #F0D679;">
            <td style="padding:10px; font-weight:700; color:#6F5402;">
              Mezcla total ({n_animales} cab.)
            </td>
            <td style="padding:10px; text-align:right; font-weight:700; color:#6F5402;">
              {mezcla_total:,.0f} kg
            </td>
            <td style="padding:10px; text-align:right; color:#6F5402;">
              {mezcla_total*7:,.0f} kg
            </td>
            <td style="padding:10px; text-align:right; color:#6F5402;">
              {mezcla_total*30:,.0f} kg
            </td>
          </tr>
        </tbody>
      </table>
      <div style="font-size:11px; color:#9C7E27; margin-top:8px;">
        Los ingredientes marcados <strong>HMS</strong> son los que
        coordinás vos. El resto los compra el productor por su lado.
      </div>
    </div>""".replace(",", ".")

    html = f"""<!DOCTYPE html>
<html><body style="margin:0; padding:0; background:#F4F4F4;
  font-family:Arial,sans-serif;">
  <div style="max-width:720px; margin:0 auto; background:white;">
    <div style="background:white; padding:20px 24px; border-bottom:3px solid {COLOR_VERDE};">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td style="vertical-align:middle;">
            <div style="font-size:19px; font-weight:600; color:{COLOR_VERDE};">
              📊 Demanda de insumos — {nombre_cli}
            </div>
            <div style="font-size:13px; color:{COLOR_GRIS}; margin-top:4px;">
              {est}{(' · ' if est else '')}{fecha_ref}
            </div>
          </td>
          <td style="vertical-align:middle; text-align:right; width:170px;">
            <img src="cid:hms-logo" alt="HMS Nutrición Animal" style="height:105px; width:auto; max-width:260px; display:block; margin-left:auto; margin-right:-8px;">
          </td>
        </tr>
      </table>
    </div>
    <div style="padding:22px; color:{COLOR_GRIS}; font-size:14px;
      line-height:1.55;">
      <p style="margin-top:0;">
        Resumen de la demanda diaria/semanal/mensual de cada insumo
        para todos los lotes activos del cliente. Sirve para planificar
        entregas y coordinar logística.
      </p>

      {kpis_html}

      <h3 style="color:{COLOR_VERDE}; font-size:15px;
        border-bottom:1px solid #E0E0E0; padding-bottom:6px;
        margin-top:24px;">
        Por lote / corral
      </h3>
      {''.join(bloques_lote_html)}

      {bloque_total_html}

      <p style="font-size:11px; color:#888; margin-top:18px;">
        El cálculo usa la dieta vigente de cada lote en
        <strong>{fecha_ref}</strong> y la cantidad de animales vigente
        ese mismo día (restando movimientos: muertes, ventas,
        traslados, ingresos).
      </p>
    </div>
    <div style="background:{COLOR_VERDE}; padding:12px 24px; color:white;
      font-size:11px; text-align:center;">
      <strong>HMS Nutrición Animal</strong> — Catriló, La Pampa<br>
      <a href="mailto:mauricio@hmsnutricionanimal.com.ar" style="color:white; text-decoration:underline;">mauricio@hmsnutricionanimal.com.ar</a> · 2954-517407
    </div>
  </div>
</body></html>"""

    # ───── Texto plano ─────
    text_partes = [
        f"DEMANDA DE INSUMOS — {nombre_cli}",
        f"{est}  ·  {fecha_ref}",
        "",
        f"Lotes activos: {n_lotes}",
        f"Animales totales: {n_animales}",
        f"Mezcla total: {mezcla_total:.0f} kg/día",
        "",
        "POR LOTE / CORRAL",
        "",
        "\n\n".join(bloques_lote_text),
        "",
        "TOTAL CLIENTE (suma de todos los lotes)",
    ]
    for ing in tot.get("ingredientes") or []:
        tag_hms = " [HMS]" if ing.get("es_hms") else ""
        if ing.get("es_libre_disposicion"):
            text_partes.append(
                f"  • {ing['nombre']}{tag_hms}: "
                f"a libre disposición "
                f"({ing.get('lotes_que_lo_usan', 0)} lote(s))"
            )
        else:
            text_partes.append(
                f"  • {ing['nombre']}{tag_hms}: "
                f"{ing['kg_dia']:.1f} kg/día · "
                f"{ing['kg_semana']:.0f} kg/sem · "
                f"{ing['kg_mes']:.0f} kg/mes "
                f"({ing.get('lotes_que_lo_usan', 0)} lote(s))"
            )
    text_partes.append("")
    text_partes.append(
        f"Mezcla total cliente: {mezcla_total:.0f} kg/día · "
        f"{mezcla_total*7:.0f} kg/semana · "
        f"{mezcla_total*30:.0f} kg/mes"
    )
    text = "\n".join(text_partes)

    return subject, html, text
