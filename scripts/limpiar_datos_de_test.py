#!/usr/bin/env python3
"""Borra de la base los clientes que dejaron los tests del 31/07/2026.

**Qué pasó.** Los scripts `test_consumo_tal_cual.py` y
`test_peso_proyectado.py` hacían `os.environ.pop("DATABASE_URL")`
creyendo que así caían a SQLite. Pero `db_backend._get_database_url()`,
cuando no encuentra la variable de entorno, lee el `.env` del proyecto
directamente. Los tests se conectaron a Supabase y le dejaron 28
clientes de prueba con sus lotes, dietas, pesadas, entregas,
movimientos e impactos.

Cinco de esos clientes se llaman IGUAL que clientes reales, porque la
primera versión del test usaba los nombres de producción y la tabla
`clientes` en Postgres no tiene el UNIQUE del DDL de SQLite. Quedaron
duplicados.

**Cómo se distingue el duplicado del real:** por `fecha_alta`. Los de
test se crearon todos el 2026-07-31; los reales son de mayo. El script
NUNCA borra un cliente cuya `fecha_alta` no sea del día del incidente,
aunque el nombre coincida.

Uso:
    python3 scripts/limpiar_datos_de_test.py            # solo informa
    python3 scripts/limpiar_datos_de_test.py --confirmar # borra

Sin `--confirmar` no toca nada: lista exactamente qué borraría.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import database as db          # noqa: E402
from src import db_backend              # noqa: E402

# Día en que corrieron los tests contra producción. Un cliente con
# cualquier otra fecha de alta NO se toca, pase lo que pase.
FECHA_INCIDENTE = "2026-07-31"

# Nombres exactos creados por los tests. Escritos a mano uno por uno
# desde el código de los scripts: nada de patrones tipo "termina en SA",
# que podrían barrer un cliente real que se llame así.
NOMBRES_TEST = [
    # --- test_consumo_tal_cual.py (primera versión, sin anonimizar) ---
    "Ezequiel Pezzola", "Jackie Graves", "Mario Salvadori",
    "Miguel Bergondi", "Pedro Manuel Pezzola",
    # --- test_consumo_tal_cual.py (versión anonimizada) ---
    "A — destete", "B — recría", "C — novillos", "D — vacas",
    "E — novillos",
    # --- test_consumo_tal_cual.py (casos sintéticos) ---
    "Base SA", "Incoherente SA", "Legacy SA", "Legacy2 SA", "Vacio SA",
    "Rollo SA", "Ajeno SA", "SinDieta SA", "Adaptacion SA", "Clima SA",
    "Bajas SA", "EndToEnd SA",
    # --- test_peso_proyectado.py ---
    "Drone SA", "DosPesadas SA", "Juntas SA", "Absurda SA",
    "Perdida SA", "SinPesadas SA", "SinAdpv SA", "MismoDia SA",
    "Futura SA", "Techo SA", "Silo SA",
    # --- test_serie_peso.py (por si alguna vez se corrió) ---
    "Limpio SA", "Frio SA", "PocaCobertura SA", "Superpuesto SA",
    "Confirmado SA", "Extremo SA", "Pesadas SA", "AdpvClima SA",
    "SinPeso SA", "Rango SA", "SinFecha SA", "UnaPesada SA",
]

CONFIRMAR = "--confirmar" in sys.argv

print("=" * 72)
print("LIMPIEZA DE DATOS DE TEST EN PRODUCCIÓN")
print(f"Base: {'Postgres (Supabase)' if db_backend.usando_postgres() else 'SQLite local'}")
print(f"Modo: {'BORRADO REAL' if CONFIRMAR else 'SIMULACRO (no toca nada)'}")
print("=" * 72)

def _leer_clientes(intentos=4):
    import time
    ultimo = None
    for n in range(intentos):
        try:
            return db.listar_clientes()
        except Exception as e:
            ultimo = e
            print(f"  (se cortó la conexión, reintento {n + 1}/{intentos}…)")
            if n < intentos - 1:
                time.sleep(2 * (n + 1))
    raise ultimo


todos = _leer_clientes()
por_nombre = {}
for c in todos:
    por_nombre.setdefault(c["nombre"], []).append(c)

a_borrar = []
protegidos = []

for nombre in NOMBRES_TEST:
    for c in por_nombre.get(nombre, []):
        alta = (c.get("fecha_alta") or "")[:10]
        if alta == FECHA_INCIDENTE:
            a_borrar.append(c)
        else:
            protegidos.append((c, alta))

if protegidos:
    print("\nPROTEGIDOS (mismo nombre, pero NO son del incidente):")
    for c, alta in protegidos:
        print(f"  · id={c['id']:<4} {c['nombre']:<28} alta {alta}  → NO se toca")

if not a_borrar:
    print("\nNo hay nada para borrar. La base está limpia.")
    sys.exit(0)

print(f"\nA BORRAR: {len(a_borrar)} clientes creados el {FECHA_INCIDENTE}")
print("-" * 72)
print("  %-5s %-30s %-12s" % ("id", "nombre", "alta"))
print("-" * 72)
for c in a_borrar:
    print("  %-5s %-30s %-12s" % (
        c["id"], c["nombre"][:30], (c.get("fecha_alta") or "")[:10]))
print("-" * 72)
print(f"  TOTAL: {len(a_borrar)} clientes")

if not CONFIRMAR:
    print("\nSIMULACRO: no se borró nada.")
    print("Revisá la lista de arriba. Si está bien, volvé a correrlo con:")
    print("    python3 scripts/limpiar_datos_de_test.py --confirmar")
    sys.exit(0)

def con_reintentos(fn, intentos=4):
    """La base está en Oregón y la conexión se corta sola. Reintenta
    en vez de dejar la limpieza a medias."""
    import time
    ultimo = None
    for n in range(intentos):
        try:
            return fn()
        except Exception as e:
            ultimo = e
            if n < intentos - 1:
                time.sleep(2 * (n + 1))
    raise ultimo


print("\nBorrando...")
errores = []
for c in a_borrar:
    try:
        # Los hijos primero: no confío en que el borrado en cascada
        # esté declarado igual en las dos bases.
        lotes = con_reintentos(lambda: db.listar_lotes(cliente_id=c["id"]))
        for l in lotes:
            con_reintentos(lambda l=l: db.eliminar_lote(l["id"]))
        con_reintentos(lambda: db.eliminar_cliente(c["id"]))
        print(f"  borrado id={c['id']} {c['nombre']}")
    except Exception as e:
        errores.append((c, e))
        print(f"  ERROR id={c['id']} {c['nombre']}: {e}")

print("-" * 72)
if errores:
    print(f"Terminó con {len(errores)} errores. Revisalos arriba.")
    sys.exit(1)

try:
    restantes = len(db.listar_clientes())
    print(f"Listo. Quedan {restantes} clientes en la base.")
except Exception:
    print("Listo. No pude releer la base para contar, pero el borrado terminó sin errores.")
print("Verificá en la app que el header diga la cantidad que esperás.")
