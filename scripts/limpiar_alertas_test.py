#!/usr/bin/env python3
"""Limpia entradas de prueba de la tabla alertas_enviadas.

Pensado para borrar los registros de testing (ej. las 8 alertas de
stock que se mandaron a Salvadori el 27/05 mientras probábamos
forzar_alerta_stock.py antes de que tuviera dedup).

Uso:
    # Ver qué se borraría (no toca la DB)
    python3 scripts/limpiar_alertas_test.py Salvadori --tipo stock --dry-run

    # Borrar los registros de stock del cliente Salvadori del 27/05
    python3 scripts/limpiar_alertas_test.py Salvadori --tipo stock --fecha 2026-05-27

    # Borrar todos los de hoy del cliente (cualquier tipo)
    python3 scripts/limpiar_alertas_test.py Salvadori --hoy
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "cliente", nargs="?", default=None,
        help="Nombre (parcial) del cliente. Default: todos.",
    )
    p.add_argument(
        "--id", type=int, default=None,
        help="Id del cliente (alternativa al nombre).",
    )
    p.add_argument(
        "--tipo", default=None,
        help=(
            "Tipo de alerta a borrar (stock / diaria / tarde / "
            "semanal / etc.). Si no se pasa, borra todos los tipos."
        ),
    )
    p.add_argument(
        "--fecha", default=None,
        help=(
            "Fecha exacta YYYY-MM-DD a borrar. Si no se pasa, "
            "borra el rango más amplio dado por las otras flags."
        ),
    )
    p.add_argument(
        "--hoy", action="store_true",
        help="Atajo: --fecha = hoy.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Solo mostrar qué se borraría, sin tocar la DB.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("limpiar_alertas")

    # Resolver cliente_id si vino nombre
    cliente_id = args.id
    cliente_nombre = "TODOS"
    if cliente_id is None and args.cliente:
        clientes = db.listar_clientes()
        match = None
        for c in clientes:
            if args.cliente.lower() in (c.get("nombre", "") or "").lower():
                match = c
                break
        if not match:
            log.error("Cliente '%s' no encontrado.", args.cliente)
            return 1
        cliente_id = match["id"]
        cliente_nombre = match["nombre"]
    elif cliente_id:
        c = db.obtener_cliente(cliente_id)
        cliente_nombre = (c or {}).get("nombre", f"id={cliente_id}")

    fecha = args.fecha
    if args.hoy:
        fecha = datetime.now().strftime("%Y-%m-%d")

    # Armar WHERE dinámico
    where = []
    params = []
    if cliente_id is not None:
        where.append("cliente_id = ?")
        params.append(cliente_id)
    if args.tipo:
        where.append("tipo = ?")
        params.append(args.tipo)
    if fecha:
        where.append("date(fecha) = date(?)")
        params.append(fecha)

    if not where:
        log.error(
            "Hace falta al menos un filtro (cliente, --tipo, "
            "--fecha o --hoy). No vamos a borrar todo sin filtro.",
        )
        return 1

    sql_select = (
        "SELECT id, fecha, cliente_id, destinatario, asunto, "
        "tipo, estado FROM alertas_enviadas WHERE "
        + " AND ".join(where)
        + " ORDER BY fecha DESC"
    )
    sql_delete = (
        "DELETE FROM alertas_enviadas WHERE "
        + " AND ".join(where)
    )

    with db.get_conn() as conn:
        rows = conn.execute(sql_select, params).fetchall()
        log.info("Filtros aplicados:")
        log.info("  Cliente: %s (id=%s)", cliente_nombre, cliente_id)
        log.info("  Tipo:    %s", args.tipo or "—")
        log.info("  Fecha:   %s", fecha or "—")
        log.info("Encontré %d registro(s):", len(rows))
        for r in rows:
            d = dict(r)
            log.info(
                "  [%d] %s · %s · %s · %s",
                d["id"], d["fecha"], d.get("tipo", "—"),
                d.get("destinatario", "—"),
                (d.get("asunto") or "")[:60],
            )

        if not rows:
            log.info("Nada para borrar — chau.")
            return 0

        if args.dry_run:
            log.info("(DRY-RUN — no se borró nada.)")
            return 0

        # Confirmación interactiva
        try:
            ans = input(
                f"\n¿Borrar estos {len(rows)} registros? [y/N]: "
            ).strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes", "s", "si", "sí"):
            log.info("Cancelado por el usuario.")
            return 0

        conn.execute(sql_delete, params)
        log.info("✅ %d registros borrados.", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
