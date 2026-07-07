#!/usr/bin/env python3
"""
Diagnóstico IMAP: prueba varias combinaciones de host/user/password
para descubrir cuál autentica contra iCloud.

Uso:  python3 scripts/test_imap.py
"""
from __future__ import annotations

import imaplib
import ssl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import alertas_email as ae


def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def probar(host: str, user: str, password: str) -> tuple:
    """Devuelve (ok, mensaje)."""
    try:
        with imaplib.IMAP4_SSL(host, ssl_context=_ssl_ctx()) as M:
            M.login(user, password)
            M.select("INBOX")
            return True, "OK — login exitoso, INBOX accesible"
    except imaplib.IMAP4.error as e:
        return False, f"IMAP error: {e}"
    except (OSError, ssl.SSLError) as e:
        return False, f"Conexión: {e}"


def main() -> int:
    cfg = ae.cargar_config_smtp()
    if not cfg:
        print("❌ No hay config SMTP")
        return 1

    host = cfg.get("imap_host", "imap.mail.me.com")
    user_actual = cfg.get("imap_user", "")
    pwd_actual = cfg.get("imap_password", "")

    pwd_sin_guiones = pwd_actual.replace("-", "")

    print(f"Host:           {host}")
    print(f"User actual:    {user_actual}")
    print(f"Password (con guiones):    {pwd_actual}")
    print(f"Password (sin guiones):    {pwd_sin_guiones}")
    print()

    casos = [
        (host, user_actual, pwd_actual,       "host+user actual + pwd con guiones"),
        (host, user_actual, pwd_sin_guiones,  "host+user actual + pwd sin guiones"),
    ]

    # Variantes adicionales si el user es un Apple ID con @gmail.com
    if "@gmail.com" in user_actual.lower():
        # Probar también con la cuenta @icloud.com auto-asignada
        user_local = user_actual.split("@")[0]
        casos.append((host, f"{user_local}@icloud.com", pwd_actual,
                       f"host + user @icloud.com derivado + pwd con guiones"))
        casos.append((host, f"{user_local}@icloud.com", pwd_sin_guiones,
                       f"host + user @icloud.com derivado + pwd sin guiones"))

    for h, u, p, desc in casos:
        print(f"🔎 {desc}")
        print(f"   user = {u}")
        ok, msg = probar(h, u, p)
        emoji = "✅" if ok else "❌"
        print(f"   {emoji} {msg}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
