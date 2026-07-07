#!/usr/bin/env python3
"""Dispara MANUALMENTE la alerta de stock bajo (email + WhatsApp) a
un cliente específico, sin pasar por el cron diario completo.

Útil cuando querés forzar el envío inmediato (ej. para probar, o
porque el cron crasheó por un bug en otra parte del flujo).

Uso:
    python3 scripts/forzar_alerta_stock.py Salvadori
    python3 scripts/forzar_alerta_stock.py --id 6
    python3 scripts/forzar_alerta_stock.py Salvadori --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import database as db  # noqa: E402
from src import alertas_email as ae  # noqa: E402
from src import whatsapp as wa  # noqa: E402
from src.stock_producto import clientes_con_stock_bajo  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "cliente", nargs="?", default=None,
        help="Nombre (parcial) del cliente.",
    )
    p.add_argument(
        "--id", type=int, default=None,
        help="Id del cliente (alternativa al nombre).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--umbral", type=int, default=14,
        help="Umbral de días para considerar stock bajo (default 14).",
    )
    p.add_argument(
        "--force", action="store_true",
        help=(
            "Saltar el dedup de 3 días (mandar igual aunque ya se "
            "haya enviado alerta de stock al cliente recientemente). "
            "Usar SOLO para casos especiales — el default protege "
            "al cliente de spam."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("forzar_alerta_stock")

    items = clientes_con_stock_bajo(umbral_dias=args.umbral)
    if not items:
        log.warning("No hay clientes con stock bajo (umbral %d días).",
                    args.umbral)
        return 0

    # Filtrar el cliente buscado
    target = []
    for it in items:
        c = it["cliente"]
        if args.id and c["id"] == args.id:
            target.append(it)
        elif args.cliente and args.cliente.lower() in c["nombre"].lower():
            target.append(it)
        elif not args.id and not args.cliente:
            target.append(it)
    if not target:
        log.error(
            "Cliente no encontrado o sin stock bajo: %s / id=%s",
            args.cliente, args.id,
        )
        log.info("Clientes con stock bajo disponibles:")
        for it in items:
            log.info("  - %s (id=%d)",
                     it["cliente"]["nombre"], it["cliente"]["id"])
        return 1

    cfg_email = ae.cargar_config_smtp() or {}
    cfg_wa = wa.cargar_config() or {}

    enviados = 0
    errores = 0
    saltados = 0
    fecha_db = datetime.now().strftime("%Y-%m-%d")
    for it in target:
        cli = it["cliente"]
        productos = it["productos"]
        nombre_cli = cli["nombre"]
        log.info("=== %s (id=%d) — %d producto(s) bajos ===",
                 nombre_cli, cli["id"], len(productos))

        # ── Dedup anti-spam: si ya se mandó alerta de stock al
        # cliente en los últimos 3 días, saltar (a menos que
        # --force). Mismo criterio que alertas_diarias.py.
        if not args.force:
            try:
                with db.get_conn() as conn:
                    r = conn.execute(
                        """SELECT COUNT(*) AS n
                           FROM alertas_enviadas
                           WHERE cliente_id = ?
                             AND tipo = 'stock'
                             AND date(fecha) >= date(?, '-3 days')
                             AND estado = 'enviada'""",
                        (cli["id"], fecha_db),
                    ).fetchone()
                    n_recientes = (r["n"] if r else 0)
                if n_recientes > 0:
                    log.warning(
                        "  ⏭️  %s — ya hay %d alerta(s) de stock "
                        "en los últimos 3 días. Saltando para no "
                        "spamear (usá --force para forzar).",
                        nombre_cli, n_recientes,
                    )
                    saltados += 1
                    continue
            except Exception as e:
                log.warning("  No pude chequear dedup: %s", e)
        for p_ in productos:
            log.info("  · %s · %d kg · %d días",
                     p_["producto"], p_["kg_restantes"],
                     p_["dias_restantes"])

        contactos = db.listar_destinatarios(cli)
        if not contactos:
            log.warning(
                "  Sin contactos cargados para %s — no se manda.",
                nombre_cli,
            )
            continue

        for contacto in contactos:
            email_dest = (contacto.get("email") or "").strip()
            wa_dest = (contacto.get("whatsapp") or "").strip()

            # ── EMAIL ──
            if (email_dest and contacto.get(
                    "alertas_email_activas", 1) and cfg_email):
                try:
                    s, h, t = ae.componer_alerta_stock_cliente(
                        cli, contacto, productos,
                    )
                    if args.dry_run:
                        log.info("  [DRY] email → %s | %s",
                                 email_dest, s)
                    else:
                        ok, msg = ae.enviar_email(
                            cfg_email, [email_dest], s, h, t,
                        )
                        estado = "enviada" if ok else "fallo"
                        # Registrar en DB
                        try:
                            db.registrar_alerta_enviada(
                                fecha=(
                                    datetime.now().strftime("%Y-%m-%d")
                                ),
                                cliente_id=cli["id"],
                                destinatario=email_dest,
                                asunto=s,
                                n_alertas=len(productos),
                                estado=estado,
                                error=("" if ok else str(msg)),
                                tipo="stock",
                            )
                        except Exception:
                            pass
                        if ok:
                            log.info("  ✓ email → %s", email_dest)
                            enviados += 1
                        else:
                            log.warning(
                                "  ⚠ email falló %s: %s",
                                email_dest, msg,
                            )
                            errores += 1
                except Exception as e:
                    log.exception("Error componiendo email: %s", e)
                    errores += 1

            # ── WHATSAPP ──
            if (wa_dest and contacto.get(
                    "alertas_whatsapp_activas", 1) and cfg_wa):
                try:
                    msg_wa = wa.componer_alerta_stock_cliente(
                        cli, productos,
                    )
                    if args.dry_run:
                        log.info("  [DRY] whatsapp → %s", wa_dest)
                    else:
                        ok_w, info_w = wa.enviar_texto(
                            cfg_wa, wa_dest, msg_wa,
                        )
                        if ok_w:
                            log.info("  ✓ whatsapp → %s", wa_dest)
                            enviados += 1
                        else:
                            log.warning(
                                "  ⚠ whatsapp falló %s: %s",
                                wa_dest, info_w,
                            )
                            errores += 1
                except Exception as e:
                    log.exception("Error WA: %s", e)
                    errores += 1

    log.info(
        "=== FIN — Enviados %d, saltados por dedup %d, errores %d ===",
        enviados, saltados, errores,
    )
    if saltados > 0 and not args.force:
        log.info(
            "💡 Tip: usá --force si querés que mande igual aunque "
            "ya se haya enviado alerta en los últimos 3 días.",
        )
    return 0 if errores == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
