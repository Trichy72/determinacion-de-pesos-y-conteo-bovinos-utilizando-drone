"""Backend de base de datos — SQLite (local) o Postgres (Supabase).

El resto del código llama a `get_conn()` y obtiene una conexión que se
comporta como sqlite3 aunque por detrás sea Postgres. La elección se
hace por variable de entorno:

    DATABASE_URL vacío       → SQLite en data/cattle_tracker.db (default)
    DATABASE_URL=postgres... → Supabase

Diseño:
  - Un `ConnectionAdapter` que envuelve psycopg2 y expone `execute()`,
    `executemany()`, `commit()`, `close()`, `cursor()`, `row_factory`
    con la misma semántica que sqlite3.
  - Traducción automática de placeholders `?` → `%s`.
  - `CursorAdapter` con `.lastrowid` implementado vía trick RETURNING.
  - `Row` que soporta indexado por posición Y por nombre (como sqlite3.Row).

Filosofía: cero cambios en el código que consume `get_conn()`. Solo se
modifica `database.py` para importar de acá en lugar de sqlite3 directo.

Rollback plan: si algo se rompe con Postgres, basta con borrar
DATABASE_URL del .env y la app vuelve a usar SQLite automáticamente.
"""
from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


# =====================================================================
# Detección del backend
# =====================================================================

def _get_database_url() -> Optional[str]:
    """Devuelve la URL de Postgres si está en env, sino None (usa SQLite).

    Lee de:
      1. Variable de entorno DATABASE_URL
      2. Archivo .env en la raíz del proyecto (para desarrollo local)
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    # Fallback: leer .env manualmente (sin depender de python-dotenv)
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for linea in env_file.read_text().splitlines():
            linea = linea.strip()
            if linea.startswith("DATABASE_URL="):
                return linea.split("=", 1)[1].strip()
    return None


def usando_postgres() -> bool:
    return _get_database_url() is not None


# =====================================================================
# Adaptador de placeholders SQLite → Postgres
# =====================================================================

# Convierte `?` en `%s`, pero solo si NO están dentro de un string
# literal (evita romper cosas como "hola? qué tal?" en un valor).
# psycopg2 usa %s, mismo caracter que Python format — ponemos %% en
# caso de que el usuario escriba literalmente %s en una query (raro).
_STRING_RE = re.compile(r"'(?:[^']|'')*'")


def _traducir_placeholders(sql: str) -> str:
    """`SELECT * FROM t WHERE x = ? AND y = ?` → `... x = %s AND y = %s`.

    Escapa literales primero para no tocar `?` dentro de strings, y
    también duplica `%` a `%%` para que psycopg2 no los interprete.
    """
    # Guardar posiciones de literales string y reemplazar con placeholder temporal
    literales = []

    def _stash(m):
        literales.append(m.group(0))
        return f"\x00LIT{len(literales)-1}\x00"

    sql_sin_strings = _STRING_RE.sub(_stash, sql)

    # Duplicar % que no sean ya %s (por si hay LIKE '%foo%' — pero esos
    # están dentro de literales, ya extraídos). Fuera de literales, %
    # es raro. Igual duplicamos por seguridad.
    sql_sin_strings = sql_sin_strings.replace("%", "%%")

    # Reemplazar ? por %s
    sql_sin_strings = sql_sin_strings.replace("?", "%s")

    # Restaurar literales originales
    def _restore(m):
        idx = int(m.group(1))
        return literales[idx]

    sql_final = re.sub(r"\x00LIT(\d+)\x00", _restore, sql_sin_strings)
    return sql_final


# =====================================================================
# Adaptadores tipo sqlite3
# =====================================================================

class RowDict(dict):
    """Emula sqlite3.Row: acceso por índice, por nombre, y como dict.

    sqlite3.Row soporta:
        r[0]         → primer valor por posición
        r["col"]     → valor por nombre de columna
        dict(r)      → convierte a dict
        r.keys()     → nombres de columnas
    Esta clase soporta todo eso encima de dict.
    """
    def __init__(self, keys: Sequence[str], values: Sequence[Any]):
        super().__init__(zip(keys, values))
        self._keys = list(keys)
        self._values = list(values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)

    def keys(self):
        return self._keys


class CursorAdapter:
    """Wrappea psycopg2 cursor con API de sqlite3."""

    def __init__(self, pg_cursor):
        self._cur = pg_cursor
        self._last_sql = ""
        self._last_lastrowid = None
        # description se replica en el shape sqlite: lista de tuplas (name, ...).
        # psycopg2 ya tiene .description con .name, así que exponemos igual.
        self.arraysize = 1

    @property
    def description(self):
        return self._cur.description

    @property
    def rowcount(self):
        return self._cur.rowcount

    @property
    def lastrowid(self):
        return self._last_lastrowid

    def _fila_a_row(self, row):
        if row is None:
            return None
        cols = [d.name for d in self._cur.description] if self._cur.description else []
        return RowDict(cols, row)

    def execute(self, sql: str, params: Iterable[Any] = ()):
        # Traducir placeholders sqlite → postgres
        sql_pg = _traducir_placeholders(sql)

        # Truco para .lastrowid: si es un INSERT y no tiene RETURNING,
        # agregamos "RETURNING id" para poder rescatar el id generado.
        # Solo si detectamos INSERT INTO <tabla> (...).
        agregar_returning = (
            re.match(r"\s*INSERT\s+INTO\s+", sql_pg, re.IGNORECASE)
            and "RETURNING" not in sql_pg.upper()
        )
        if agregar_returning:
            sql_pg = sql_pg.rstrip("; \n\t") + " RETURNING id"

        try:
            self._cur.execute(sql_pg, tuple(params))
        except Exception as e:
            # Enriquecer error para debug fácil
            e.args = (
                f"{e.args[0] if e.args else ''}\n"
                f"  SQL original: {sql[:200]}\n"
                f"  SQL traducido: {sql_pg[:200]}",
            ) + e.args[1:]
            raise

        # Rescatar lastrowid si aplica
        if agregar_returning:
            try:
                r = self._cur.fetchone()
                self._last_lastrowid = r[0] if r else None
            except Exception:
                self._last_lastrowid = None
        else:
            self._last_lastrowid = None

        self._last_sql = sql_pg
        return self

    def executemany(self, sql: str, seq_params: Iterable[Iterable[Any]]):
        sql_pg = _traducir_placeholders(sql)
        self._cur.executemany(sql_pg, [tuple(p) for p in seq_params])
        return self

    def fetchone(self):
        try:
            r = self._cur.fetchone()
        except Exception:
            return None
        return self._fila_a_row(r)

    def fetchall(self):
        rows = self._cur.fetchall()
        return [self._fila_a_row(r) for r in rows]

    def fetchmany(self, size: int = -1):
        if size < 0:
            size = self.arraysize
        rows = self._cur.fetchmany(size)
        return [self._fila_a_row(r) for r in rows]

    def __iter__(self):
        while True:
            r = self.fetchone()
            if r is None:
                return
            yield r

    def close(self):
        try:
            self._cur.close()
        except Exception:
            pass


class ConnectionAdapter:
    """Wrappea psycopg2 connection con API de sqlite3."""

    def __init__(self, pg_conn):
        self._conn = pg_conn
        self._conn.autocommit = False
        # sqlite3 row_factory se lee/escribe. Lo ignoramos: RowDict siempre.
        self.row_factory = None

    def execute(self, sql: str, params: Iterable[Any] = ()):
        """Sqlite permite conn.execute() sin cursor explícito."""
        cur = CursorAdapter(self._conn.cursor())
        cur.execute(sql, params)
        return cur

    def executemany(self, sql: str, seq_params: Iterable[Iterable[Any]]):
        cur = CursorAdapter(self._conn.cursor())
        cur.executemany(sql, seq_params)
        return cur

    def executescript(self, script: str):
        """Sqlite permite un batch de statements separados por `;`.

        Postgres también, pero psycopg2 no soporta múltiples statements
        en un execute salvo que no usemos parámetros. Los partimos.
        """
        # Partir por `;` de nivel top (ojo con `;` dentro de strings —
        # el schema del proyecto no tiene, así que usamos split simple).
        stmts = [s.strip() for s in script.split(";") if s.strip()]
        cur = self._conn.cursor()
        try:
            for stmt in stmts:
                # Traducir tipos SQLite → Postgres al vuelo (para el schema)
                stmt_pg = _adaptar_ddl_a_postgres(stmt)
                if not stmt_pg:
                    continue
                try:
                    cur.execute(stmt_pg)
                except Exception as e:
                    # Muchas de estas ya existen (IF NOT EXISTS); ignoramos
                    # errores de "ya existe". El init es idempotente.
                    if "already exists" in str(e).lower():
                        self._conn.rollback()
                        continue
                    raise
        finally:
            cur.close()

    def cursor(self):
        return CursorAdapter(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


# =====================================================================
# Adaptación DDL SQLite → Postgres
# =====================================================================

def _adaptar_ddl_a_postgres(ddl: str) -> str:
    """Adapta un CREATE TABLE / CREATE INDEX / ALTER TABLE SQLite a Postgres.

    Cambios principales:
      INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY
      INTEGER PRIMARY KEY               → SERIAL PRIMARY KEY (si sola)
      DATETIME DEFAULT CURRENT_TIMESTAMP → TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      REAL   → DOUBLE PRECISION
      BLOB   → BYTEA
      Los tipos con `AUTOINCREMENT` los normalizamos.
    """
    d = ddl

    # INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY
    d = re.sub(
        r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b",
        "SERIAL PRIMARY KEY",
        d, flags=re.IGNORECASE,
    )
    # INTEGER PRIMARY KEY (sin AUTOINCREMENT) → SERIAL PRIMARY KEY
    d = re.sub(
        r"\bINTEGER\s+PRIMARY\s+KEY\b(?!\s+AUTOINCREMENT)",
        "SERIAL PRIMARY KEY",
        d, flags=re.IGNORECASE,
    )
    # Tipos simples
    d = re.sub(r"\bDATETIME\b", "TIMESTAMP", d, flags=re.IGNORECASE)
    d = re.sub(r"\bREAL\b", "DOUBLE PRECISION", d, flags=re.IGNORECASE)
    d = re.sub(r"\bBLOB\b", "BYTEA", d, flags=re.IGNORECASE)
    d = re.sub(r"\bBOOLEAN\b", "BOOLEAN", d, flags=re.IGNORECASE)
    return d


# =====================================================================
# Factory: get_conn
# =====================================================================

# Cache de la conexión de Postgres (reusa la misma para evitar overhead
# de crear una nueva cada vez — Postgres es caro de conectar).
_pg_pool: Any = None


def _crear_conexion_postgres(url: str):
    try:
        import psycopg2
        import psycopg2.extras  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Falta psycopg2 para conectar a Postgres. Instalalo con:\n"
            "    pip install psycopg2-binary"
        )
    conn = psycopg2.connect(url, connect_timeout=15)
    return ConnectionAdapter(conn)


@contextmanager
def get_conn(db_path: Optional[Path] = None):
    """Context manager que devuelve una conexión a la DB activa.

    - Si hay DATABASE_URL en env/.env → Postgres (Supabase).
    - Si no → SQLite en el path que se pase (o el default de database.py).

    Uso idéntico al `get_conn()` original:
        with get_conn() as conn:
            r = conn.execute("SELECT * FROM t WHERE x = ?", (val,)).fetchone()
    """
    url = _get_database_url()
    if url:
        # Modo Postgres — no hay pool, creamos y cerramos por ciclo.
        # (Simple y suficiente para volumen HMS. Si escala mucho, se
        # puede meter psycopg2.pool.SimpleConnectionPool).
        conn = _crear_conexion_postgres(url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        # Modo SQLite — comportamiento original.
        from src.database import DB_PATH  # import lazy p/ evitar ciclo
        path = db_path or DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def backend_activo() -> str:
    """Devuelve 'postgres' o 'sqlite' según qué backend está en uso.

    Útil para logs, diagnóstico y para saltear código SQLite-only
    (como los PRAGMA de migración inline).
    """
    return "postgres" if usando_postgres() else "sqlite"
