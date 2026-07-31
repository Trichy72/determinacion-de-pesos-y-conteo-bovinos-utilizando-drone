#!/usr/bin/env python3
"""Tests de `estimar_peso_vivo_lote` y `factor_escala_consumo_pv`.

Cubre el arreglo del 31/07/2026: la última pesada se proyecta hacia
adelante en vez de devolverse congelada. Antes, un lote CON pesada
quedaba sin escalado por peso (factor 1,000 exacto) y uno SIN datos sí
lo recibía.

Uso:   python3 scripts/test_peso_proyectado.py
"""
import shutil
import sys
from pathlib import Path

# Tiene que ir ANTES de importar src.database: fuerza SQLite temporal y
# aborta si algo resuelve una conexión a Postgres. Ver el docstring de
# _sandbox_db.py — el 31/07/2026 estos tests escribieron en producción.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sandbox_db import base_temporal, verificar_sqlite   # noqa: E402

TMP = base_temporal("test_peso_")

from src import database as db          # noqa: E402
from src import stock_producto as sp    # noqa: E402

db.init_db()
verificar_sqlite(db)

FALLAS = []
CASOS = 0


def check(nombre, obtenido, esperado, tol=0.05):
    global CASOS
    CASOS += 1
    if esperado is None:
        ok = obtenido is None
        det = f"obtenido={obtenido}"
    elif obtenido is None:
        ok = False
        det = f"obtenido=None esperado={esperado}"
    else:
        ok = abs(obtenido - esperado) <= tol
        det = f"obtenido={obtenido:.3f} esperado={esperado:.3f}"
    print(f"  {'OK  ' if ok else 'FALLA'}  {nombre}  ({det})")
    if not ok:
        FALLAS.append(nombre)


def afirmar(nombre, condicion, detalle=""):
    global CASOS
    CASOS += 1
    print(f"  {'OK  ' if condicion else 'FALLA'}  {nombre}  {detalle}")
    if not condicion:
        FALLAS.append(nombre)


def nuevo_lote(nombre, cant=10, fecha_ingreso="2026-05-01", peso=300,
               adpv=1.0):
    cid = db.crear_cliente(nombre)
    lid = db.crear_lote(
        cid, "L1", fecha_ingreso=fecha_ingreso, cantidad_inicial=cant,
        peso_ingreso_kg=peso, adpv_objetivo_kg=adpv,
    )
    return cid, lid


def pesar(lid, fecha, peso, cant=10):
    db.guardar_pesada(lid, fecha, "drone", cant, peso, peso * cant, None)


# =====================================================================
print("\n1. EL BUG: una sola pesada anulaba el escalado por completo")
print("=" * 70)

cid, lid = nuevo_lote("Drone SA", fecha_ingreso="2026-05-01",
                      peso=300, adpv=1.2)
pesar(lid, "2026-06-01", 340)
lote = db.obtener_lote(lid)
dieta = {"fecha": "2026-06-09"}

p_dieta = sp.estimar_peso_vivo_lote(lote, "2026-06-09")
p_hoy = sp.estimar_peso_vivo_lote(lote, "2026-07-31")
factor, info = sp.factor_escala_consumo_pv(lote, dieta, "2026-07-31")

# Antes: las dos fechas devolvían 340 → factor 1,000
check("peso a la fecha de dieta (340 + 1,2 × 8 días)", p_dieta, 349.6)
check("peso hoy (340 + 1,2 × 60 días)", p_hoy, 412.0)
afirmar("el factor ya NO queda clavado en 1,000",
        abs(factor - 1.0) > 0.01, f"(factor={factor:.3f})")
check("factor = 412,0 / 349,6", factor, 1.1785, tol=0.005)
afirmar("origen del factor es 'adg'", info.get("origen") == "adg",
        f"(origen={info.get('origen')})")

# =====================================================================
print("\n2. Con dos pesadas usa la ganancia MEDIDA, no el objetivo")
print("=" * 70)
print("  Es lo que incorpora el efecto del clima sin tener que modelarlo:")
print("  si el invierno frenó la ganancia, las pesadas lo muestran.")

cid, lid = nuevo_lote("DosPesadas SA", fecha_ingreso="2026-05-01",
                      peso=300, adpv=1.4)
pesar(lid, "2026-05-01", 300)
pesar(lid, "2026-06-30", 350)      # 50 kg en 60 días = 0,833 kg/día real
lote = db.obtener_lote(lid)

# 31 días después de la última pesada: 350 + 0,833 × 31 = 375,8
check("proyecta con el ADPV medido (0,833) y no con 1,4",
      sp.estimar_peso_vivo_lote(lote, "2026-07-31"), 375.83, tol=0.5)
afirmar("con el objetivo habría dado ~393,4 (17 kg de más)",
        abs(350 + 1.4 * 31 - 393.4) < 0.1)

# =====================================================================
print("\n3. Pesadas muy juntas: no se les cree la ganancia")
print("=" * 70)

cid, lid = nuevo_lote("Juntas SA", fecha_ingreso="2026-05-01",
                      peso=300, adpv=1.0)
pesar(lid, "2026-06-01", 340)
pesar(lid, "2026-06-06", 355)      # 3 kg/día en 5 días: ruido del drone
lote = db.obtener_lote(lid)
# Cae al objetivo 1,0 → 355 + 1,0 × 55 = 410
check("5 días entre pesadas → usa el objetivo", sp.estimar_peso_vivo_lote(
    lote, "2026-07-31"), 410.0)

# =====================================================================
print("\n4. Ganancia medida absurda: también cae al objetivo")
print("=" * 70)

cid, lid = nuevo_lote("Absurda SA", fecha_ingreso="2026-05-01",
                      peso=300, adpv=1.0)
pesar(lid, "2026-05-01", 300)
pesar(lid, "2026-06-30", 700)      # 6,7 kg/día: imposible
lote = db.obtener_lote(lid)
check("6,7 kg/día se descarta → objetivo 1,0", sp.estimar_peso_vivo_lote(
    lote, "2026-07-31"), 731.0)     # 700 + 1,0 × 31

# =====================================================================
print("\n5. Pérdida de peso real: se respeta (es dato, no error)")
print("=" * 70)

cid, lid = nuevo_lote("Perdida SA", fecha_ingreso="2026-05-01",
                      peso=300, adpv=1.0)
pesar(lid, "2026-05-01", 320)
pesar(lid, "2026-06-30", 305)      # -0,25 kg/día
lote = db.obtener_lote(lid)
check("proyecta hacia abajo", sp.estimar_peso_vivo_lote(
    lote, "2026-07-31"), 297.25, tol=0.5)

_f, _i = sp.factor_escala_consumo_pv(lote, {"fecha": "2026-06-30"},
                                     "2026-07-31")
afirmar("el factor baja pero no perfora el piso de 0,85",
        0.85 <= _f < 1.0, f"(factor={_f:.3f})")

# =====================================================================
print("\n6. Sin pesadas se comporta igual que antes")
print("=" * 70)

cid, lid = nuevo_lote("SinPesadas SA", fecha_ingreso="2026-05-01",
                      peso=300, adpv=1.0)
lote = db.obtener_lote(lid)
check("300 + 1,0 × 91 días", sp.estimar_peso_vivo_lote(
    lote, "2026-07-31"), 391.0)

cid, lid = nuevo_lote("SinAdpv SA", fecha_ingreso="2026-05-01",
                      peso=300, adpv=0)
lote = db.obtener_lote(lid)
check("sin ADPV devuelve el peso de ingreso", sp.estimar_peso_vivo_lote(
    lote, "2026-07-31"), 300.0)
_f, _i = sp.factor_escala_consumo_pv(lote, {"fecha": "2026-06-09"},
                                     "2026-07-31")
check("sin ADPV el factor es 1,0", _f, 1.0)
afirmar("y lo dice en origen", _i.get("origen") == "sin_adg",
        f"(origen={_i.get('origen')})")

# =====================================================================
print("\n7. Pesada del mismo día y pesadas futuras")
print("=" * 70)

cid, lid = nuevo_lote("MismoDia SA", fecha_ingreso="2026-05-01",
                      peso=300, adpv=1.0)
pesar(lid, "2026-07-31", 400)
lote = db.obtener_lote(lid)
check("pesada de hoy → sin proyección", sp.estimar_peso_vivo_lote(
    lote, "2026-07-31"), 400.0)

cid, lid = nuevo_lote("Futura SA", fecha_ingreso="2026-05-01",
                      peso=300, adpv=1.0)
pesar(lid, "2026-06-01", 340)
pesar(lid, "2026-12-01", 500)      # posterior a la referencia: se ignora
lote = db.obtener_lote(lid)
check("la pesada futura no se usa", sp.estimar_peso_vivo_lote(
    lote, "2026-07-31"), 400.0)     # 340 + 1,0 × 60

# =====================================================================
print("\n8. El techo del factor sigue vigente")
print("=" * 70)

cid, lid = nuevo_lote("Techo SA", fecha_ingreso="2026-01-01",
                      peso=200, adpv=2.5)
lote = db.obtener_lote(lid)
_f, _i = sp.factor_escala_consumo_pv(lote, {"fecha": "2026-02-01"},
                                     "2026-07-31")
check("clampeado a 1,40", _f, 1.40)
afirmar("y el factor bruto queda registrado",
        (_i.get("factor_bruto") or 0) > 1.40,
        f"(bruto={_i.get('factor_bruto')})")

# =====================================================================
print("\n9. Punta a punta: la carga de silo con y sin pesada")
print("=" * 70)

cid, lid = nuevo_lote("Silo SA", cant=100, fecha_ingreso="2026-05-01",
                      peso=300, adpv=1.2)
db.guardar_dieta(
    lid, "2026-06-09",
    [{"nombre": "Fibrogreen", "pct_ms": 14.0, "kg_tal_cual": 1.0},
     {"nombre": "Maíz grano", "pct_ms": 86.0, "kg_tal_cual": 6.0}],
    consumo_ms_kg=6.16,
)
d_sin = sp.desglose_carga_silocomedero(lid, 5, "2026-07-31")
pesar(lid, "2026-06-01", 340, cant=100)
d_con = sp.desglose_carga_silocomedero(lid, 5, "2026-07-31")

afirmar("sin pesada la carga ya escalaba", d_sin["escala_pv"]["factor_aplicado"] > 1.0,
        f"(factor={d_sin['escala_pv']['factor_aplicado']})")
afirmar("con pesada AHORA también escala (antes daba 1,000)",
        d_con["escala_pv"]["factor_aplicado"] > 1.0,
        f"(factor={d_con['escala_pv']['factor_aplicado']})")
afirmar("y la carga a preparar cambia en consecuencia",
        abs(d_con["kg_total_mezcla"] - d_sin["kg_total_mezcla"]) > 1,
        f"({d_sin['kg_total_mezcla']} → {d_con['kg_total_mezcla']} kg)")

# =====================================================================
print("\n" + "=" * 70)
print(f"RESULTADO: {CASOS - len(FALLAS)}/{CASOS} casos OK")
if FALLAS:
    print("FALLARON: " + ", ".join(FALLAS))
print("=" * 70)

shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FALLAS else 0)
