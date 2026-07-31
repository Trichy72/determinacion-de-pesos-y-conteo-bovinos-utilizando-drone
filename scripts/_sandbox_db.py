"""Fuerza que un test corra contra SQLite temporal y NUNCA contra Supabase.

**Por qué existe este archivo.** El 31/07/2026 dos scripts de test se
conectaron a la base de PRODUCCIÓN y le dejaron 28 clientes de prueba,
cinco de ellos duplicando el nombre de clientes reales. El test hacía
`os.environ.pop("DATABASE_URL")` dando por sentado que sin esa variable
el backend cae a SQLite. No es así: `db_backend._get_database_url()`,
cuando no encuentra la variable de entorno, **lee el `.env` del
proyecto directamente**. El pop no sirvió de nada.

La lección: sacar la variable de entorno NO alcanza. Hay que
intervenir el resolutor, y después verificar que la intervención
funcionó — porque un test que escribe en producción no avisa, pasa
en verde.

Uso, SIEMPRE como primera importación del test y antes de importar
`src.database`:

    from _sandbox_db import base_temporal
    TMP = base_temporal("test_loquesea_")

    from src import database as db     # recién ahora
"""
import os
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def base_temporal(prefijo: str = "test_") -> Path:
    """Crea una base SQLite temporal, deja el proceso apuntado ahí y
    verifica que no haya forma de llegar a Postgres.

    Aborta el proceso si la verificación falla: es preferible un test
    que no corre a un test que escribe en la base de producción.
    """
    sys.path.insert(0, str(RAIZ))

    # 1) Sacar la variable de entorno (necesario pero NO suficiente).
    os.environ.pop("DATABASE_URL", None)

    # 2) Intervenir el resolutor, que es el que lee el .env a espaldas
    #    del entorno. Esto tiene que pasar ANTES de que se importe
    #    src.database, que resuelve la conexión al importarse.
    from src import db_backend

    db_backend._get_database_url = lambda: None

    # 3) Verificar. Si algo quedó apuntando a Postgres, cortar acá.
    if db_backend.usando_postgres():
        sys.exit(
            "ABORTADO: el test resolvió una conexión a Postgres.\n"
            "Un test NUNCA debe escribir en la base de producción."
        )
    url = db_backend._get_database_url()
    if url:
        sys.exit(f"ABORTADO: _get_database_url() devolvió {url!r}")

    # 4) Recién ahora, la base temporal.
    tmp = Path(tempfile.mkdtemp(prefix=prefijo))
    os.chdir(tmp)
    (tmp / "data").mkdir(exist_ok=True)
    return tmp


def verificar_sqlite(db_mod) -> None:
    """Segunda verificación, ya con `database` importado: que la
    conexión real sea SQLite. Se llama después de `init_db()`."""
    try:
        with db_mod.get_conn() as conn:
            fila = conn.execute("SELECT 1 AS x").fetchone()
            if fila is None:
                sys.exit("ABORTADO: la base temporal no responde.")
    except Exception as e:
        sys.exit(f"ABORTADO: no pude abrir la base temporal: {e}")

    from src import db_backend
    if db_backend.usando_postgres():
        sys.exit(
            "ABORTADO: después de init_db la conexión es Postgres.\n"
            "Un test NUNCA debe escribir en la base de producción."
        )
