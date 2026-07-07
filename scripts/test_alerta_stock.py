"""Script de prueba: manda el email de alerta de stock al admin
(NO al cliente), para ver cómo queda armado el mensaje en formato real.

Usa datos ficticios simulando que un cliente tiene stock bajo. Es solo
para preview visual — no escribe en la DB, no notifica al cliente.

Uso:
    .venv/bin/python scripts/test_alerta_stock.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import alertas_email as ae


def main() -> int:
    cfg = ae.cargar_config_smtp() or {}
    if not cfg.get("host"):
        print("❌ No hay config SMTP cargada. "
              "Andá a Configuración primero.")
        return 1

    destinatario_prueba = cfg.get("admin_email") or cfg.get("from_email")
    print(f"📨 Destinatario de prueba: {destinatario_prueba}")
    print("    (es vos, no el cliente — esto es solo para ver el formato)")

    # Caso 1: un solo producto con stock crítico (5 días)
    cliente_demo_1 = {
        "nombre": "Ezequiel Pezzola",
        "establecimiento": "Lonquimay",
    }
    contacto_demo_1 = {
        "nombre": "Ezequiel Pezzola",
        "email": destinatario_prueba,
    }
    productos_demo_1 = [
        {
            "lote_ident": "Terneros destete",
            "producto": "Fibroter (BALCOOP Destete Precoz)",
            "kg_restantes": 26,
            "consumo_kg_dia": 5.2,
            "dias_restantes": 5,
            "fecha_agotamiento": "2026-05-26",
        },
    ]

    # Caso 2: dos productos consolidados, uno crítico y uno moderado
    cliente_demo_2 = {
        "nombre": "Miguel Bergondi",
        "establecimiento": "La Cancha",
    }
    contacto_demo_2 = {
        "nombre": "Miguel Bergondi",
        "email": destinatario_prueba,
    }
    productos_demo_2 = [
        {
            "lote_ident": "Engorde vacas",
            "producto": "Fibrogreen plus",
            "kg_restantes": 90,
            "consumo_kg_dia": 12.0,
            "dias_restantes": 8,
            "fecha_agotamiento": "2026-05-29",
        },
        {
            "lote_ident": "Recría B",
            "producto": "Fibroter",
            "kg_restantes": 168,
            "consumo_kg_dia": 12.0,
            "dias_restantes": 14,
            "fecha_agotamiento": "2026-06-04",
        },
    ]

    for label, cliente, contacto, productos in [
        ("STOCK CRÍTICO (5 días, 1 producto)",
         cliente_demo_1, contacto_demo_1, productos_demo_1),
        ("STOCK BAJO + MODERADO (consolidado, 2 productos)",
         cliente_demo_2, contacto_demo_2, productos_demo_2),
    ]:
        print(f"\n=== {label} ===")
        try:
            subject, html, text = ae.componer_alerta_stock_cliente(
                cliente, contacto, productos,
            )
            # Prefijar [PRUEBA] al subject para que se distinga del envío real
            subject_test = f"[PRUEBA] {subject}"
            ok, msg = ae.enviar_email(
                cfg, [destinatario_prueba], subject_test, html, text,
                con_bcc_admin=False,
            )
            if ok:
                print(f"✅ Enviado: {subject_test}")
            else:
                print(f"❌ Falló: {msg}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n💌 Revisá tu inbox en {destinatario_prueba}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
