#!/usr/bin/env python3
"""Renombra "Maíz molido" / "Grano de maíz molido" → "Maíz grano" en todas
las dietas ya cargadas a la base. Por decisión del 25/05/2026: no se usa
maíz molido (riesgo de acidosis), se usa maíz GRANO (entero o partido
grueso).

Idempotente: si las dietas ya están actualizadas no hace nada.

Uso:
    python3 scripts/renombrar_maiz_molido.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "cattle_tracker.db"

# Comparación case-insensitive; primer término ⇒ reemplazo
RENAMES_LOWER = {
    "maíz molido": "Maíz grano",
    "grano de maíz molido": "Maíz grano",
    "maiz molido": "Maíz grano",
    "grano de maiz molido": "Maíz grano",
}


def main() -> int:
    if not DB.exists():
        print(f"❌ No existe la DB en {DB}")
        return 1

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id, composicion_json FROM dietas")
    rows = cur.fetchall()

    n_changed = 0
    for did, raw in rows:
        if not raw:
            continue
        try:
            comp = json.loads(raw)
        except Exception:
            continue
        changed = False
        for ing in comp:
            nom = (ing.get("nombre") or "").strip()
            lo = nom.lower()
            if lo in RENAMES_LOWER:
                ing["nombre"] = RENAMES_LOWER[lo]
                changed = True
        if changed:
            cur.execute(
                "UPDATE dietas SET composicion_json=? WHERE id=?",
                (json.dumps(comp, ensure_ascii=False), did),
            )
            n_changed += 1
            print(f"  ✓ dieta id={did} actualizada")

    con.commit()
    con.close()

    if n_changed == 0:
        print("👍 No había nada para actualizar (ya estaban en 'Maíz grano').")
    else:
        print(f"\n✅ Total dietas modificadas: {n_changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
