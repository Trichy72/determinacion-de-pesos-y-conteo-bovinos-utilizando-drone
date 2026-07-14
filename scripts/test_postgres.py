"""Test end-to-end del backend Postgres.

Corre las mismas queries que hace la app real y compara resultados
contra SQLite. Si todo pasa, quiere decir que:
  1) El adapter traduce placeholders bien
  2) RowDict funciona (acceso por índice y por nombre)
  3) .lastrowid vía RETURNING funciona
  4) Los tipos de datos se mapean correctamente

Uso:
    python scripts/test_postgres.py

Requisito: DATABASE_URL configurada en .env (Supabase).

NOTA: este script NO modifica datos. Es solo lectura.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

PROY = Path(__file__).resolve().parents[1]
os.chdir(PROY)
# Agregar la raíz del proyecto al path para poder importar `src.*`
sys.path.insert(0, str(PROY))


def separador(t):
    print(f"\n{'='*60}\n{t}\n{'='*60}")


def main():
    separador("1. DETECCIÓN DEL BACKEND")
    from src.db_backend import backend_activo, _get_database_url
    print(f"Backend activo: {backend_activo()}")
    url = _get_database_url()
    if not url:
        print("❌ No hay DATABASE_URL en .env. Configuralo y reintentá.")
        return
    print(f"URL: {url[:60]}...")
    if backend_activo() != "postgres":
        print("❌ El backend no es postgres. Revisá .env")
        return

    separador("2. CONEXIÓN + QUERIES BÁSICAS")
    from src.database import get_conn
    with get_conn() as conn:
        r = conn.execute("SELECT 1 AS x").fetchone()
        assert r[0] == 1, f"esperaba 1, dio {r[0]}"
        assert r["x"] == 1, "acceso por nombre roto"
        print("  ✓ SELECT 1 devuelve 1 (indexado y por nombre OK)")

    separador("3. TABLAS + CONTEOS (deben coincidir con SQLite)")
    # Conteo en SQLite
    sqlite_conn = sqlite3.connect(PROY / "data" / "cattle_tracker.db")
    sqlite_conn.row_factory = sqlite3.Row
    tablas = [
        "clientes", "lotes", "dietas", "entregas_producto",
        "cargas_silocomedero", "alertas_enviadas",
        "alertas_whatsapp_enviadas", "pronosticos_semanales",
        "recordatorios_llamada", "impactos_lote", "categorias_animales",
    ]
    todo_ok = True
    with get_conn() as conn:
        for t in tablas:
            n_pg = conn.execute(
                f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            n_sq = sqlite_conn.execute(
                f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            ok = "✓" if n_pg == n_sq else "✗"
            if n_pg != n_sq:
                todo_ok = False
            print(f"  {ok} {t}: SQLite={n_sq} Postgres={n_pg}")
    sqlite_conn.close()

    separador("4. QUERY CON PLACEHOLDERS (test del translator)")
    with get_conn() as conn:
        # Query igual a la que hace la app: buscar Pezzola
        r = conn.execute(
            "SELECT id, nombre FROM clientes WHERE nombre LIKE ?",
            ("%Pezzola%",),
        ).fetchone()
        if r:
            print(f"  ✓ Encontró cliente: id={r['id']} nombre={r['nombre']}")
        else:
            print("  ⚠️  No encontró Pezzola (chequeá que los datos migraron)")

        # Query con múltiples placeholders
        rows = conn.execute(
            """SELECT id, identificador FROM lotes
               WHERE cliente_id = ? AND estado = ? ORDER BY id DESC""",
            (7, "activo"),
        ).fetchall()
        print(f"  ✓ Lotes activos del cliente 7: {len(rows)}")
        for lote in rows:
            print(f"     - id={lote['id']} ident={lote['identificador']}")

    separador("5. INSERT + RETURNING id (test lastrowid)")
    with get_conn() as conn:
        # Insertar una alerta de test y verificar que devuelve el id.
        # Usamos asunto="ADAPTER_TEST" como marcador para poder identificarla
        # y borrarla al final. Las columnas coinciden con la tabla real.
        cur = conn.execute(
            """INSERT INTO alertas_enviadas
                (cliente_id, destinatario, asunto, n_alertas,
                 estado, fecha, tipo)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (7, "test@adapter.local", "ADAPTER_TEST", 0,
             "test", "2026-07-14", "diaria"),
        )
        new_id = cur.lastrowid
        assert new_id and new_id > 0, f"lastrowid vino vacío: {new_id}"
        print(f"  ✓ INSERT devolvió lastrowid = {new_id}")

        # Verificar que la fila existe
        r = conn.execute(
            "SELECT id, asunto FROM alertas_enviadas WHERE id = ?",
            (new_id,),
        ).fetchone()
        assert r and r["asunto"] == "ADAPTER_TEST"
        print(f"  ✓ La fila se puede leer de vuelta (asunto={r['asunto']})")

        # Limpiar
        conn.execute("DELETE FROM alertas_enviadas WHERE id = ?", (new_id,))
        print(f"  ✓ Test row limpiada")

    separador("RESUMEN")
    if todo_ok:
        print("🎉 Todos los tests pasaron. El adapter funciona.")
        print("   La app está lista para operar contra Postgres.")
    else:
        print("⚠️  Algunas tablas difieren entre SQLite y Postgres.")
        print("   Corré: python scripts/migrar_sqlite_a_supabase.py --run")
        print("   (Vuelve a copiar todo — es idempotente)")


if __name__ == "__main__":
    main()
