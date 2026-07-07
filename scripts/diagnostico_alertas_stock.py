#!/usr/bin/env python3
"""Diagnóstico de alertas de stock para un cliente específico.

Responde a la pregunta operativa: ¿por qué a este cliente no le
llegó otra alerta de falta de producto?

Verifica:
  1. Si el cliente tiene stock bajo HOY (productos a punto de
     agotarse)
  2. Cuándo se envió la última alerta de tipo 'stock'
  3. Si el dedup de 3 días está bloqueando un nuevo envío
  4. Si el cliente tiene email/WhatsApp activos
  5. Cuándo se vence el dedup (si está activo)

Uso:
    python3 scripts/diagnostico_alertas_stock.py Salvadori
    python3 scripts/diagnostico_alertas_stock.py --id 6
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import database as db  # noqa: E402
from src.stock_producto import clientes_con_stock_bajo  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "cliente", nargs="?", default=None,
        help="Nombre (parcial) del cliente. Ej: Salvadori",
    )
    p.add_argument(
        "--id", type=int, default=None,
        help="Id del cliente (alternativa al nombre).",
    )
    p.add_argument(
        "--umbral", type=int, default=14,
        help="Umbral de días para 'stock bajo' (default 14).",
    )
    args = p.parse_args()

    # Resolver cliente
    cliente_id = args.id
    cliente_nombre = None
    if cliente_id is None and args.cliente:
        clientes = db.listar_clientes()
        match = None
        for c in clientes:
            if args.cliente.lower() in (c.get("nombre") or "").lower():
                match = c
                break
        if not match:
            print(f"❌ Cliente '{args.cliente}' no encontrado.")
            print("\nClientes disponibles:")
            for c in clientes:
                print(f"  - {c['nombre']} (id={c['id']})")
            return 1
        cliente_id = match["id"]
        cliente_nombre = match["nombre"]
    elif cliente_id:
        c = db.obtener_cliente(cliente_id)
        cliente_nombre = (c or {}).get("nombre", f"id={cliente_id}")
    else:
        print("❌ Pasá nombre o --id")
        return 1

    hoy = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*70}")
    print(f"📋 DIAGNÓSTICO DE ALERTAS DE STOCK")
    print(f"   Cliente: {cliente_nombre} (id={cliente_id})")
    print(f"   Hoy: {hoy}")
    print(f"   Umbral de stock bajo: {args.umbral} días")
    print(f"{'='*70}\n")

    # ── 1. Estado actual del stock ──
    print("1️⃣  ¿Tiene stock bajo HOY?")
    todos_bajo = clientes_con_stock_bajo(umbral_dias=args.umbral)
    items_cliente = [
        it for it in todos_bajo
        if it["cliente"]["id"] == cliente_id
    ]
    if not items_cliente:
        print("   🟢 NO — el cliente NO tiene productos por agotarse.")
        print(
            "   Posibles motivos: ya recibió entrega, "
            "consumo ajustado, sin lotes activos."
        )
    else:
        for it in items_cliente:
            for p_ in it["productos"]:
                print(
                    f"   🟠 {p_['producto']}: "
                    f"{p_['kg_restantes']:.0f} kg, "
                    f"{p_['dias_restantes']} días restantes"
                )

    # ── 2. Configuración de alertas ──
    print("\n2️⃣  ¿Tiene email/WhatsApp activos?")
    cli_full = db.obtener_cliente(cliente_id) or {}
    contactos = db.listar_destinatarios(cli_full)
    if not contactos:
        print(
            "   ❌ NO tiene contactos cargados. Por eso no le "
            "puede llegar nada."
        )
    else:
        for ct in contactos:
            _email = ct.get("email") or ""
            _wa = ct.get("whatsapp") or ""
            _email_act = ct.get("alertas_email_activas", 1)
            _wa_act = ct.get("alertas_whatsapp_activas", 1)
            print(
                f"   👤 {ct.get('nombre', '—')}: "
                + (
                    f"📧 {_email} "
                    + ("✓ activo" if _email_act else "✗ desactivado")
                    if _email else ""
                )
                + (" · " if _email and _wa else "")
                + (
                    f"📱 {_wa} "
                    + ("✓ activo" if _wa_act else "✗ desactivado")
                    if _wa else ""
                )
            )

    # ── 3. Últimas alertas de stock enviadas ──
    print("\n3️⃣  Últimas alertas de stock enviadas a este cliente:")
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT fecha, destinatario, asunto, estado,
                       n_alertas
               FROM alertas_enviadas
               WHERE cliente_id = ? AND tipo = 'stock'
               ORDER BY date(fecha) DESC LIMIT 10""",
            (cliente_id,),
        ).fetchall()
    if not rows:
        print(
            "   📭 No tiene ninguna alerta de stock registrada "
            "en la base."
        )
    else:
        for r in rows:
            d = dict(r)
            print(
                f"   📅 {d['fecha']} · "
                f"{d.get('estado','?'):<10} · "
                f"{d.get('destinatario','—')[:35]:<35} · "
                f"{(d.get('asunto') or '')[:50]}"
            )

    # ── 4. Estado del dedup ──
    print("\n4️⃣  Estado del dedup (ventana 3 días):")
    with db.get_conn() as conn:
        r_dedup = conn.execute(
            """SELECT MAX(fecha) AS ultima
               FROM alertas_enviadas
               WHERE cliente_id = ? AND tipo = 'stock'
                 AND date(fecha) >= date(?, '-3 days')
                 AND estado = 'enviada'""",
            (cliente_id, hoy),
        ).fetchone()
    ultima = r_dedup["ultima"] if r_dedup else None
    if ultima:
        # Calcular cuándo se libera el dedup
        try:
            f_ultima = datetime.strptime(
                str(ultima)[:10], "%Y-%m-%d"
            ).date()
            f_libera = f_ultima + timedelta(days=3)
            dias_a_libera = (
                f_libera - datetime.now().date()
            ).days
        except Exception:
            f_libera = None
            dias_a_libera = None
        print(
            f"   🚫 DEDUP ACTIVO — última alerta enviada el "
            f"{ultima[:10]}."
        )
        if f_libera:
            if dias_a_libera <= 0:
                print(
                    f"   ✅ El dedup se libera HOY — el próximo "
                    "cron diario puede mandar otra."
                )
            else:
                print(
                    f"   ⏳ Dedup se libera el "
                    f"{f_libera.isoformat()} "
                    f"(en {dias_a_libera} día(s))."
                )
        print(
            "   💡 Si querés forzar el envío AHORA (saltando "
            "dedup), ejecutá:"
        )
        print(
            f"       python3 scripts/forzar_alerta_stock.py "
            f"\"{cliente_nombre}\" --force"
        )
    else:
        print(
            "   ✅ DEDUP NO ACTIVO — no hay alertas de stock en "
            "los últimos 3 días."
        )
        print(
            "   El próximo cron diario va a evaluar si "
            "corresponde mandar."
        )

    # ── 5. Resumen / próximos pasos ──
    print("\n5️⃣  Resumen:")
    if not items_cliente:
        print(
            "   ➡️  No hay nada que mandar — el cliente NO tiene "
            "stock bajo HOY."
        )
    elif not contactos:
        print(
            "   ➡️  Hay stock bajo pero el cliente NO tiene "
            "contactos. Cargá email/WhatsApp en la ficha."
        )
    elif ultima:
        print(
            f"   ➡️  Hay stock bajo y contactos activos, pero el "
            f"dedup de 3 días bloqueó el envío (última: "
            f"{ultima[:10]})."
        )
        print(
            "       Es normal — protege al cliente del spam. "
            "Espera o usá --force si es urgente."
        )
    else:
        print(
            "   ➡️  Hay stock bajo, contactos activos y dedup "
            "NO bloqueado."
        )
        print(
            "       El próximo cron diario (cada 24hs) debería "
            "mandar. Si no llega, revisá:"
        )
        print("       - Logs del cron (~/Library/Logs/launchd/...)")
        print("       - Plist launchd activo")
        print(
            "       - Forzar manualmente: "
            f"python3 scripts/forzar_alerta_stock.py "
            f"\"{cliente_nombre}\""
        )

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
