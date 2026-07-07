"""Tokens firmados HMAC para el link de carga diaria por WhatsApp.

Filosofía: que el link que recibe el encargado por WhatsApp NO se pueda
forjar ni reutilizar de un día a otro sin permiso. Token determinístico
por (lote_id, fecha) firmado con una clave secreta local.

Formato del token URL-safe:
    <lote_id>.<YYYYMMDD>.<sig8>

donde sig8 son los primeros 8 caracteres del HMAC-SHA256 (base16) sobre
"<lote_id>:<YYYYMMDD>" usando la clave secreta del archivo
`data/.carga_secret`. Se trunca a 8 chars (32 bits) — suficiente contra
fuerza bruta porque la URL caduca en 48 hs y cada intento es una HTTP
request al servidor.

Validación:
- Verifica firma HMAC.
- Verifica que la fecha esté dentro de la ventana válida (default: día
  de la fecha incluida ± 1 día).

El archivo de secret se crea al primer uso si no existe (32 bytes
random hex).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Tuple


_ROOT = Path(__file__).resolve().parents[1]
_SECRET_FILE = _ROOT / "data" / ".carga_secret"


def _load_or_create_secret() -> bytes:
    """Lee la clave secreta del archivo. Si no existe, la genera y
    la guarda con permisos restrictivos (600).

    Returns:
        Clave secreta en bytes.
    """
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_bytes()
    _SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    nuevo = secrets.token_bytes(32)
    _SECRET_FILE.write_bytes(nuevo)
    try:
        os.chmod(_SECRET_FILE, 0o600)
    except OSError:
        pass
    return nuevo


def generar_token(lote_id: int, fecha: Optional[date] = None) -> str:
    """Token URL-safe para el lote+fecha indicado.

    Args:
        lote_id: id del lote.
        fecha: fecha de la carga (default: hoy).

    Returns:
        Token formato '<lote_id>.<YYYYMMDD>.<sig8>'.
    """
    fecha = fecha or date.today()
    payload = f"{lote_id}:{fecha.strftime('%Y%m%d')}"
    sig = hmac.new(
        _load_or_create_secret(), payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:8]
    return f"{lote_id}.{fecha.strftime('%Y%m%d')}.{sig}"


def validar_token(
    token: str, ventana_dias: int = 1,
) -> Tuple[bool, Optional[int], Optional[date], str]:
    """Valida un token y devuelve sus componentes.

    Args:
        token: el token recibido en el query string.
        ventana_dias: cuántos días antes/después de la fecha del token
            se considera todavía válido (default: 1 → 48 hs útiles).

    Returns:
        Tupla (valido, lote_id, fecha, mensaje_error).
        Si valido=True, mensaje_error=''.
    """
    if not token or token.count(".") != 2:
        return False, None, None, "Token con formato inválido."
    try:
        lote_str, fecha_str, sig = token.split(".")
        lote_id = int(lote_str)
        fecha = datetime.strptime(fecha_str, "%Y%m%d").date()
    except (ValueError, TypeError):
        return False, None, None, "Token con formato inválido."

    # Re-firmar y comparar
    payload = f"{lote_id}:{fecha_str}"
    sig_esperada = hmac.new(
        _load_or_create_secret(), payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:8]
    if not hmac.compare_digest(sig, sig_esperada):
        return False, None, None, "Token con firma inválida."

    # Ventana de validez
    hoy = date.today()
    if abs((hoy - fecha).days) > ventana_dias:
        return (
            False, lote_id, fecha,
            f"Token vencido (fecha {fecha} vs hoy {hoy})."
        )
    return True, lote_id, fecha, ""


def url_carga_diaria(
    base_url: str, lote_id: int, fecha: Optional[date] = None,
) -> str:
    """URL completa para mandar por WhatsApp.

    Args:
        base_url: dominio base del túnel/deploy
            (ej. 'https://hms.ngrok.app').
        lote_id: id del lote.
        fecha: fecha (default hoy).

    Returns:
        URL completa con el token en query string.
    """
    token = generar_token(lote_id, fecha)
    base = base_url.rstrip("/")
    return f"{base}/?token={token}"
