"""
Procesador automático de bajas por email (IMAP).

Se conecta a Gmail vía IMAP, busca respuestas con "BAJA" en el asunto o
cuerpo del mensaje, identifica al cliente por el email remitente, desactiva
las alertas (email y WhatsApp) en la DB y manda un email de confirmación.

Configuración: usa la misma cuenta SMTP definida en data/smtp_config.json.
Si la cuenta es Gmail, también se usa para IMAP automáticamente.
Para otros proveedores, se puede sumar `imap_host` en la config.

Uso:
    from src.procesador_bajas import procesar_bajas_pendientes
    n = procesar_bajas_pendientes()
"""

from __future__ import annotations

import email
import imaplib
import logging
import re
import ssl
from email.header import decode_header
from typing import Dict, List, Optional, Tuple

from . import database as db
from . import alertas_email as ae


log = logging.getLogger(__name__)


def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _imap_host_from_smtp(smtp_host: str) -> Optional[str]:
    """Inferir el host IMAP a partir del host SMTP."""
    if not smtp_host:
        return None
    h = smtp_host.lower()
    if "gmail" in h:
        return "imap.gmail.com"
    if "outlook" in h or "hotmail" in h or "office365" in h:
        return "outlook.office365.com"
    if "icloud" in h or "me.com" in h:
        return "imap.mail.me.com"
    if "yahoo" in h:
        return "imap.mail.yahoo.com"
    # fallback: cambiar smtp por imap
    if h.startswith("smtp."):
        return "imap." + h[5:]
    return None


def _decode_header_value(raw) -> str:
    """Decodifica un header (subject, from) que puede venir en RFC2047."""
    if not raw:
        return ""
    parts = decode_header(raw)
    salida = []
    for txt, charset in parts:
        if isinstance(txt, bytes):
            try:
                salida.append(txt.decode(charset or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                salida.append(txt.decode("utf-8", errors="replace"))
        else:
            salida.append(txt)
    return "".join(salida)


def _extraer_email_remitente(from_header: str) -> Optional[str]:
    """Extrae el email del header 'From'."""
    if not from_header:
        return None
    m = re.search(r"<([^>]+@[^>]+)>", from_header)
    if m:
        return m.group(1).strip().lower()
    m = re.search(r"([\w._+-]+@[\w.-]+\.[A-Za-z]{2,})", from_header)
    return m.group(1).strip().lower() if m else None


def _extraer_cuerpo_texto(msg: email.message.Message) -> str:
    """Extrae el texto plano de un email (incluso si es multipart)."""
    if msg.is_multipart():
        partes = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    try:
                        partes.append(
                            payload.decode(part.get_content_charset()
                                            or "utf-8", errors="replace")
                        )
                    except (LookupError, UnicodeDecodeError):
                        partes.append(payload.decode("utf-8", errors="replace"))
        return "\n".join(partes)
    payload = msg.get_payload(decode=True)
    if payload:
        try:
            return payload.decode(msg.get_content_charset()
                                    or "utf-8", errors="replace")
        except (LookupError, UnicodeDecodeError):
            return payload.decode("utf-8", errors="replace")
    return str(msg.get_payload() or "")


def _es_baja(asunto: str, cuerpo: str) -> bool:
    """Detecta si el email pide baja (BAJA, UNSUBSCRIBE, etc.)."""
    texto = f"{asunto} {cuerpo}".lower()
    palabras_baja = ["baja", "unsubscribe", "darme de baja", "dar de baja",
                       "no quiero más alertas", "cancelar alertas",
                       "stop alertas", "dejar de recibir"]
    return any(p in texto for p in palabras_baja)


def procesar_bajas_pendientes(cfg_smtp: Optional[Dict] = None,
                                max_emails: int = 50) -> Tuple[int, List[str]]:
    """Conecta IMAP, busca pedidos de baja y los procesa.

    Returns:
        (n_bajas_procesadas, [lista de emails dados de baja])
    """
    cfg = cfg_smtp or ae.cargar_config_smtp()
    if not cfg:
        log.warning("No hay config SMTP — no se puede procesar bajas")
        return 0, []

    # Si la config tiene un bloque IMAP separado, usar esos datos.
    # Si no, fallback a inferir desde el SMTP (mismo proveedor).
    imap_host = cfg.get("imap_host")
    user = cfg.get("imap_user")
    password = cfg.get("imap_password")

    # Fallback: usar datos del SMTP (cuando IMAP y SMTP son el mismo proveedor)
    if not imap_host:
        imap_host = _imap_host_from_smtp(cfg.get("host", ""))
    if not user:
        user = cfg.get("user")
    if not password:
        password = cfg.get("password")

    if not imap_host:
        log.warning(f"No se pudo inferir host IMAP desde "
                     f"SMTP={cfg.get('host')}")
        return 0, []

    if not user or not password:
        log.warning("Falta usuario o password IMAP")
        return 0, []

    bajas_procesadas: List[str] = []

    try:
        ctx = _ssl_ctx()
        with imaplib.IMAP4_SSL(imap_host, ssl_context=ctx) as M:
            M.login(user, password)
            M.select("INBOX")

            # Buscar emails que contengan "BAJA" en asunto o cuerpo
            # (más eficiente que iterar por toda la bandeja UNSEEN).
            # Buscamos en los últimos 30 días para no traer muy viejos.
            from datetime import datetime as _dt, timedelta as _td
            desde = (_dt.now() - _td(days=30)).strftime("%d-%b-%Y")
            ids: List[bytes] = []
            for criterio in (
                f'(SINCE "{desde}" SUBJECT "BAJA")',
                f'(SINCE "{desde}" BODY "BAJA")',
                f'(SINCE "{desde}" SUBJECT "darme de baja")',
                f'(SINCE "{desde}" BODY "darme de baja")',
                f'(SINCE "{desde}" SUBJECT "unsubscribe")',
            ):
                try:
                    status, data = M.search(None, criterio)
                    if status == "OK" and data and data[0]:
                        ids.extend(data[0].split())
                except imaplib.IMAP4.error:
                    continue

            # Deduplicar IDs y limitar
            ids = list(dict.fromkeys(ids))[:max_emails]
            if not ids:
                return 0, []

            for msg_id in ids:
                status, msg_data = M.fetch(msg_id, "(RFC822)")
                if status != "OK" or not msg_data:
                    continue

                msg = email.message_from_bytes(msg_data[0][1])
                asunto = _decode_header_value(msg.get("Subject", ""))
                from_h = _decode_header_value(msg.get("From", ""))
                cuerpo = _extraer_cuerpo_texto(msg)
                email_remit = _extraer_email_remitente(from_h)

                if not email_remit:
                    continue

                # Verificar si pide baja
                if not _es_baja(asunto, cuerpo[:500]):
                    continue

                # Buscar cliente por email
                clientes = db.listar_clientes()
                cliente = None
                for c in clientes:
                    cli_email = (c.get("email") or "").strip().lower()
                    if cli_email and cli_email == email_remit:
                        cliente = c
                        break

                if not cliente:
                    log.info(f"  Baja recibida de {email_remit} pero "
                              f"no coincide con ningún cliente. Skip.")
                    continue

                # Marcar como dado de baja (estado='baja') Y desactivar
                # alertas. El cliente queda archivado: no aparece en la
                # tabla principal pero se puede reactivar desde la ficha.
                db.dar_de_baja_cliente(
                    cliente["id"],
                    motivo=(
                        f"Solicitud automática vía email "
                        f"({email_remit})"
                    ),
                    desactivar_alertas=True,
                )
                bajas_procesadas.append(email_remit)
                log.info(f"  ✓ Baja procesada: {cliente['nombre']} "
                          f"({email_remit})")

                # Marcar el email como leído
                M.store(msg_id, "+FLAGS", "\\Seen")

                # Enviar confirmación
                try:
                    _enviar_confirmacion_baja(cfg, cliente, email_remit)
                except Exception as e:
                    log.warning(f"  No se pudo confirmar baja a "
                                 f"{email_remit}: {e}")

    except (imaplib.IMAP4.error, OSError, ssl.SSLError) as e:
        log.warning(f"Error IMAP: {e}")
        return 0, []

    return len(bajas_procesadas), bajas_procesadas


def _enviar_confirmacion_baja(cfg: Dict, cliente: Dict, email_remit: str) -> None:
    """Manda email confirmando la baja."""
    nombre = cliente.get("nombre", "")
    subject = "✅ Baja confirmada — HMS Nutrición Animal"
    html = f"""<!DOCTYPE html>
<html><body style="font-family:Arial; padding:20px; max-width:520px;">
  <div style="background:#1B3E27; padding:14px; color:white; border-radius:6px 6px 0 0;">
    <strong>HMS Nutrición Animal</strong>
  </div>
  <div style="background:#F8F8F8; padding:24px; border-radius:0 0 6px 6px;">
    <p>Hola {nombre},</p>
    <p>Recibimos tu pedido de baja. <strong>Ya no vas a recibir más alertas</strong>
    climáticas por este medio.</p>
    <p>Si en algún momento querés volver a recibir las alertas, escribime a
    <a href="mailto:mauricio@hmsnutricionanimal.com.ar">
    mauricio@hmsnutricionanimal.com.ar</a> o al WhatsApp 2954-517407.</p>
    <p style="color:#888; font-size:12px;">— Mauricio Suárez<br>
    HMS Nutrición Animal</p>
  </div>
</body></html>"""
    text = (f"Hola {nombre},\n\n"
              "Recibimos tu pedido de baja. Ya no vas a recibir más alertas "
              "climáticas por este medio.\n\n"
              "Si querés volver, escribime a mauricio@hmsnutricionanimal.com.ar "
              "o al WhatsApp 2954-517407.\n\n"
              "— Mauricio Suárez\nHMS Nutrición Animal")
    ae.enviar_email(cfg, [email_remit], subject, html, text)
