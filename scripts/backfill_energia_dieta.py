#!/usr/bin/env python3
"""Backfill del campo energia_dieta_mcal_em_kg_ms del lote, usando la
última dieta de cada lote para calcular la concentración EM
(em_mcal_dia / consumo_ms_kg).

Este script es idempotente y SOLO actualiza los lotes que tienen el
campo vacío (energia_dieta_mcal_em_kg_ms IS NULL). No pisa valores ya
cargados a mano. Usá `--force` si querés recalcular todos.

A partir de ahora, cada vez que el agente IA guarde una dieta o plan
de adaptación, el campo se actualiza automáticamente. Este script es
solo para arrancar con los lotes que ya tenían dieta antes del cambio.

Uso:
    python3 scripts/backfill_energia_dieta.py
    python3 scripts/backfill_energia_dieta.py --force
    python3 scripts/backfill_energia_dieta.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import database as db  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--force", action="store_true",
        help="Sobrescribir el campo aunque ya tenga valor.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    actualizados = 0
    saltados = 0
    rechazados = 0
    sin_datos = 0

    for c in db.listar_clientes():
        for lote in db.listar_lotes(cliente_id=c["id"], estado="activo"):
            valor_actual = lote.get("energia_dieta_mcal_em_kg_ms")
            if valor_actual and not args.force:
                saltados += 1
                continue

            dietas = db.listar_dietas(lote["id"])
            if not dietas:
                sin_datos += 1
                continue

            # Usar la última dieta vigente
            ult = dietas[-1]
            em_dia = float(ult.get("em_mcal_dia") or 0)
            consumo_ms = float(ult.get("consumo_ms_kg") or 0)

            if em_dia <= 0 or consumo_ms <= 0:
                sin_datos += 1
                continue

            em_conc = round(em_dia / consumo_ms, 3)

            if not (1.8 <= em_conc <= 3.5):
                rechazados += 1
                print(
                    f"  ⚠ {c['nombre'][:25]:25} lote {lote['id']}: "
                    f"valor fuera de rango ({em_conc}), no se actualiza."
                )
                continue

            print(
                f"  {'(DRY) ' if args.dry_run else '✓ '}"
                f"{c['nombre'][:25]:25} lote {lote['id']}: "
                f"{em_conc} Mcal EM/kg MS "
                f"(antes: {valor_actual or 'vacío'})"
            )
            if not args.dry_run:
                db.actualizar_lote(
                    lote["id"],
                    energia_dieta_mcal_em_kg_ms=em_conc,
                )
                actualizados += 1

    print()
    print(f"=== Resumen ===")
    print(f"  Actualizados:      {actualizados}")
    print(f"  Ya tenían valor:   {saltados}")
    print(f"  Sin dieta válida:  {sin_datos}")
    print(f"  Fuera de rango:    {rechazados}")
    if args.dry_run:
        print()
        print("(Modo dry-run: no se modificó nada en la DB.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
