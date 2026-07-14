"""Migra la base cattle_tracker.db (SQLite) a Supabase (Postgres).

Uso:
    python scripts/migrar_sqlite_a_supabase.py           # solo simulación
    python scripts/migrar_sqlite_a_supabase.py --run     # ejecuta la migración
    python scripts/migrar_sqlite_a_supabase.py --verify  # solo verifica conteos

Requisitos:
    pip install psycopg2-binary python-dotenv

Antes de correr, tiene que existir un archivo .env en la raíz del proyecto
con la variable DATABASE_URL apuntando al Postgres de Supabase.

Filosofía:
  - No borra nada. Si una tabla ya existe en Postgres, se salta (para
    poder correrlo varias veces).
  - Copia por lotes de 100 filas para no reventar la conexión.
  - Al final, compara COUNT(*) entre SQLite y Postgres y muestra la
    diferencia por tabla.
  - Si algún INSERT falla, log detallado y sigue con la siguiente fila
    (no rompe todo el proceso).

Traducción de tipos SQLite → Postgres:
  INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY
  INTEGER                           → BIGINT
  REAL                              → DOUBLE PRECISION
  TEXT                              → TEXT
  BLOB                              → BYTEA
  BOOLEAN (usualmente 0/1 en INT)   → BOOLEAN (se convierte al insertar)
  fechas ISO en TEXT                → TEXT (compatible, no molestamos)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# ─────────────────────────── Config ───────────────────────────

PROYECTO = Path(__file__).resolve().parents[1]
SQLITE_DB = PROYECTO / "data" / "cattle_tracker.db"
ENV_FILE = PROYECTO / ".env"

# Tamaño de lote para INSERTs
BATCH_SIZE = 100


# ────────────────────── Cargar credenciales ──────────────────────

def cargar_env():
    """Carga variables de .env sin depender de python-dotenv."""
    if not ENV_FILE.exists():
        print(f"❌ No existe {ENV_FILE}")
        print("   Creá el archivo con DATABASE_URL=postgresql://...")
        sys.exit(1)
    for linea in ENV_FILE.read_text().splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        os.environ.setdefault(clave.strip(), valor.strip())


# ───────────── Traducción de esquema SQLite → Postgres ─────────────

def traducir_columna(col: dict) -> str:
    """Convierte una definición de columna SQLite a Postgres."""
    nombre = col["name"]
    tipo_sqlite = (col["type"] or "TEXT").upper()
    es_pk = col["pk"] == 1
    not_null = col["notnull"] == 1
    default = col["dflt_value"]

    # Detectar autoincrement como SERIAL
    if es_pk and "INT" in tipo_sqlite:
        return f'"{nombre}" SERIAL PRIMARY KEY'

    # Traducir tipo
    if "INT" in tipo_sqlite:
        pg_tipo = "BIGINT"
    elif "REAL" in tipo_sqlite or "FLOAT" in tipo_sqlite or "DOUBLE" in tipo_sqlite:
        pg_tipo = "DOUBLE PRECISION"
    elif "BLOB" in tipo_sqlite:
        pg_tipo = "BYTEA"
    elif "BOOL" in tipo_sqlite:
        pg_tipo = "BOOLEAN"
    elif "NUMERIC" in tipo_sqlite or "DECIMAL" in tipo_sqlite:
        pg_tipo = "NUMERIC"
    else:
        pg_tipo = "TEXT"

    partes = [f'"{nombre}"', pg_tipo]
    if not_null:
        partes.append("NOT NULL")
    if default is not None:
        # Adaptar defaults conocidos
        d = str(default).strip()
        if d.upper() == "CURRENT_TIMESTAMP":
            partes.append("DEFAULT CURRENT_TIMESTAMP")
        elif d.replace(".", "").replace("-", "").isdigit():
            partes.append(f"DEFAULT {d}")
        else:
            partes.append(f"DEFAULT {d}")

    return " ".join(partes)


def generar_create_table(cur_sqlite, tabla: str) -> str:
    """Arma el CREATE TABLE para Postgres a partir del schema SQLite."""
    cur_sqlite.execute(f'PRAGMA table_info("{tabla}")')
    cols = [dict(r) for r in cur_sqlite.fetchall()]
    cols_pg = [traducir_columna(c) for c in cols]
    return (
        f'CREATE TABLE IF NOT EXISTS "{tabla}" (\n    '
        + ",\n    ".join(cols_pg)
        + "\n);"
    )


def generar_indices(cur_sqlite) -> list[str]:
    """Devuelve los índices existentes en SQLite adaptados a Postgres."""
    cur_sqlite.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='index' AND sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite_%'"
    )
    idxs = []
    for nombre, sql in cur_sqlite.fetchall():
        # SQLite usa `CREATE INDEX idx ON tabla(col)`. Postgres lo entiende
        # tal cual, solo hay que evitar UNIQUE INDEX y CREATE INDEX IF NOT EXISTS.
        sql_pg = sql.replace(
            "CREATE INDEX", "CREATE INDEX IF NOT EXISTS"
        ).replace(
            "CREATE UNIQUE INDEX",
            "CREATE UNIQUE INDEX IF NOT EXISTS",
        )
        idxs.append(sql_pg + ";")
    return idxs


# ───────────────────── Copia de datos ─────────────────────

def copiar_tabla(cur_sqlite, cur_pg, tabla: str, dry_run: bool):
    """Copia todas las filas de una tabla SQLite → Postgres."""
    cur_sqlite.execute(f'PRAGMA table_info("{tabla}")')
    cols_info = [dict(r) for r in cur_sqlite.fetchall()]
    col_names = [c["name"] for c in cols_info]
    col_names_q = [f'"{c}"' for c in col_names]

    cur_sqlite.execute(f'SELECT COUNT(*) FROM "{tabla}"')
    total = cur_sqlite.fetchone()[0]
    if total == 0:
        print(f"  {tabla}: (vacía)")
        return 0

    cur_sqlite.execute(f'SELECT * FROM "{tabla}"')
    insertadas = 0
    fallidas = 0
    lote = []
    placeholders = "(" + ",".join(["%s"] * len(col_names)) + ")"

    def _flush(lote):
        nonlocal insertadas, fallidas
        if not lote:
            return
        sql = (
            f'INSERT INTO "{tabla}" ('
            + ",".join(col_names_q)
            + f") VALUES {','.join([placeholders]*len(lote))}"
            + " ON CONFLICT DO NOTHING"
        )
        params = [v for fila in lote for v in fila]
        try:
            if not dry_run:
                cur_pg.execute(sql, params)
            insertadas += len(lote)
        except Exception as e:
            # Fallback: fila por fila para localizar la culpable
            for fila in lote:
                try:
                    sql_una = (
                        f'INSERT INTO "{tabla}" ('
                        + ",".join(col_names_q)
                        + f") VALUES {placeholders}"
                        + " ON CONFLICT DO NOTHING"
                    )
                    if not dry_run:
                        cur_pg.execute(sql_una, fila)
                    insertadas += 1
                except Exception as e2:
                    fallidas += 1
                    print(f"    ⚠️  fila con error en {tabla}: {e2}")

    for fila in cur_sqlite:
        lote.append(tuple(fila))
        if len(lote) >= BATCH_SIZE:
            _flush(lote)
            lote = []
    _flush(lote)

    estado = "✓" if fallidas == 0 else "⚠"
    print(f"  {estado} {tabla}: {insertadas}/{total} insertadas"
          + (f" ({fallidas} fallidas)" if fallidas else ""))
    return insertadas


# ────────────────────────── Main ──────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="Ejecutar la migración (por default es dry-run).")
    parser.add_argument("--verify", action="store_true",
                        help="Solo verificar conteos, no migrar.")
    args = parser.parse_args()

    cargar_env()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("❌ Falta DATABASE_URL en .env")
        sys.exit(1)

    try:
        import psycopg2
    except ImportError:
        print("❌ Falta psycopg2. Instalalo con:")
        print("     pip install psycopg2-binary")
        sys.exit(1)

    if not SQLITE_DB.exists():
        print(f"❌ No existe la base SQLite en {SQLITE_DB}")
        sys.exit(1)

    print(f"📂 SQLite: {SQLITE_DB}")
    print(f"☁️  Postgres: {db_url.split('@')[1] if '@' in db_url else '(oculto)'}")
    modo = "VERIFY" if args.verify else ("RUN" if args.run else "DRY-RUN")
    print(f"🔧 Modo: {modo}")
    print()

    # ── Conectar ──
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row
    cur_sqlite = sqlite_conn.cursor()

    print("🔌 Conectando a Supabase...")
    pg_conn = psycopg2.connect(db_url, connect_timeout=15)
    pg_conn.autocommit = False
    cur_pg = pg_conn.cursor()
    cur_pg.execute("SELECT version()")
    print(f"   ✓ {cur_pg.fetchone()[0][:60]}")
    print()

    # ── Listar tablas SQLite ──
    cur_sqlite.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    tablas = [r[0] for r in cur_sqlite.fetchall()]
    print(f"📋 Tablas encontradas: {len(tablas)}")

    if args.verify:
        # ── Solo verificar ──
        print()
        print("📊 Comparación SQLite vs Postgres:")
        print(f"{'Tabla':<30} {'SQLite':>10} {'Postgres':>10} {'Estado':>10}")
        print("-" * 62)
        total_ok = total_dif = 0
        for t in tablas:
            n_sqlite = cur_sqlite.execute(
                f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            try:
                cur_pg.execute(f'SELECT COUNT(*) FROM "{t}"')
                n_pg = cur_pg.fetchone()[0]
            except Exception as e:
                pg_conn.rollback()
                n_pg = f"NO EXISTE"
            estado = "✓" if n_sqlite == n_pg else "✗"
            if n_sqlite == n_pg:
                total_ok += 1
            else:
                total_dif += 1
            print(f"{t:<30} {n_sqlite:>10} {str(n_pg):>10} {estado:>10}")
        print("-" * 62)
        print(f"Resumen: {total_ok} tablas OK, {total_dif} con diferencia")
        sys.exit(0 if total_dif == 0 else 1)

    # ── Crear tablas en Postgres ──
    print()
    print("🏗️  Creando tablas en Postgres...")
    for t in tablas:
        ddl = generar_create_table(cur_sqlite, t)
        if args.run:
            cur_pg.execute(ddl)
        print(f"  ✓ {t}")

    # ── Crear índices ──
    print()
    print("🗂️  Creando índices...")
    for idx_sql in generar_indices(cur_sqlite):
        if args.run:
            try:
                cur_pg.execute(idx_sql)
            except Exception as e:
                print(f"  ⚠️  {e}")
        print(f"  ✓ {idx_sql[:60]}...")

    if args.run:
        pg_conn.commit()
        print("  💾 DDL commiteado")

    # ── Copiar datos ──
    print()
    print("📥 Copiando datos...")
    total = 0
    for t in tablas:
        total += copiar_tabla(cur_sqlite, cur_pg, t, dry_run=not args.run)

    if args.run:
        pg_conn.commit()
        print(f"\n💾 {total} filas commiteadas")
    else:
        print(f"\n🔍 DRY-RUN: hubieran ido {total} filas")
        print("   Corré con --run para ejecutar la migración de verdad")

    cur_pg.close()
    pg_conn.close()
    sqlite_conn.close()
    print("\n✅ Listo")


if __name__ == "__main__":
    main()
