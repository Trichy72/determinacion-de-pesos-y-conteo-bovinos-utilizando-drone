#!/usr/bin/env python3
"""
Script de prueba del mensaje de bienvenida.

Manda el email y/o WhatsApp de bienvenida (el que se mandaría la primera
vez a un contacto nuevo) al destinatario que pases por argumento.
NO toca la base de datos — solo manda el mensaje para que vos veas
cómo llega.

Uso:
    # Email de bienvenida a un destinatario:
    python3 scripts/test_bienvenida.py --email hms002@gmail.com

    # WhatsApp (el destinatario tiene que estar en el sandbox Twilio):
    python3 scripts/test_bienvenida.py --whatsapp "+5492954517407"

    # Los dos a la vez:
    python3 scripts/test_bienvenida.py \
        --email hms002@gmail.com \
        --whatsapp "+5492954517407"

    # Personalizar el contexto (cliente / contacto que aparecen en el mensaje):
    python3 scripts/test_bienvenida.py \
        --email hms002@gmail.com \
        --cliente "Estancia La Soñada" \
        --establecimiento "Lote Sur" \
        --nombre "Carlos Pérez" \
        --rol "Encargado"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import alertas_email as ae
from src import whatsapp as wa


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--email", default=None,
                    help="Email destinatario para mandar la bienvenida")
    p.add_argument("--whatsapp", default=None,
                    help="WhatsApp destinatario E.164 (ej +5492954517407)")
    p.add_argument("--cliente", default="Estancia La Soñada (PRUEBA)",
                    help="Nombre del cliente que aparece en el mensaje")
    p.add_argument("--establecimiento", default="Lote Sur",
                    help="Establecimiento del cliente")
    p.add_argument("--nombre", default="Carlos Pérez",
                    help="Nombre del contacto destinatario")
    p.add_argument("--rol", default="Encargado",
                    help="Rol del destinatario (Encargado, Capataz, etc.)")
    args = p.parse_args()

    if not args.email and not args.whatsapp:
        print("ERROR: tenés que pasar --email y/o --whatsapp")
        return 1

    cliente = {
        "nombre": args.cliente,
        "establecimiento": args.establecimiento,
        "contacto": args.nombre,
        "localidad": "Catriló",
    }
    contacto = {
        "nombre": args.nombre,
        "rol": args.rol,
        "email": args.email or "",
        "whatsapp": args.whatsapp or "",
    }

    # ──────── EMAIL ────────
    if args.email:
        cfg = ae.cargar_config_smtp()
        ok_cfg, err = ae.config_valida(cfg)
        if not ok_cfg:
            print(f"❌ Config SMTP inválida: {err}")
            print("   Configurá SMTP en la pestaña Configuración de la app.")
            return 2
        subject, html, text = ae.componer_bienvenida(cliente, contacto)
        print(f"\n📧 Enviando bienvenida email -> {args.email}")
        print(f"   Subject: {subject}")
        ok, msg = ae.enviar_email(cfg, [args.email], subject, html, text)
        print(f"   {'✓' if ok else '✗'} {msg}")
        if not ok:
            return 3

    # ──────── WHATSAPP ────────
    if args.whatsapp:
        cfg_wa = wa.cargar_config()
        ok_cfg, err = wa.config_valida(cfg_wa)
        if not ok_cfg:
            print(f"❌ Config WhatsApp inválida: {err}")
            return 4
        wa_text = wa.componer_bienvenida(cliente, contacto)
        print(f"\n📱 Enviando bienvenida WhatsApp -> {args.whatsapp}")
        print("─" * 50)
        print(wa_text)
        print("─" * 50)
        ok, msg = wa.enviar_texto(cfg_wa, args.whatsapp, wa_text)
        print(f"   {'✓' if ok else '✗'} {msg}")
        if not ok:
            return 5

    print("\n✅ Listo. Revisá tu inbox / WhatsApp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
