#!/usr/bin/env python3
"""Tests de `calcular_consumo_diario_kg` contra una base SQLite temporal.

Cubre el cambio del 31/07/2026: el consumo diario de producto sale de
`kg_tal_cual` de la dieta y no de `DMI × pct_ms / 100`.

Uso:   python3 scripts/test_consumo_tal_cual.py
Salida: una línea por caso y un resumen. Exit code 1 si algo falla.
"""
import shutil
import sys
from pathlib import Path

# Tiene que ir ANTES de importar src.database: fuerza SQLite temporal y
# aborta si algo resuelve una conexión a Postgres. Ver el docstring de
# _sandbox_db.py — el 31/07/2026 estos tests escribieron en producción.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sandbox_db import base_temporal, verificar_sqlite   # noqa: E402

TMP = base_temporal("test_consumo_")

from src import database as db          # noqa: E402
from src import stock_producto as sp    # noqa: E402

db.init_db()
verificar_sqlite(db)

FALLAS = []
CASOS = 0


def check(nombre, obtenido, esperado, tol=0.01):
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
    return ok


def nuevo_lote(nombre_cli, ident, cantidad, fecha_ingreso="2026-05-01",
               peso=300, adpv=1.0):
    cid = db.crear_cliente(nombre_cli)
    lid = db.crear_lote(
        cid, ident, fecha_ingreso=fecha_ingreso,
        cantidad_inicial=cantidad, peso_ingreso_kg=peso,
        adpv_objetivo_kg=adpv,
    )
    return cid, lid


# =====================================================================
print("\n1. Los 5 lotes de producción — contra los valores de campo")
print("   (validados a campo el 30/07/2026; clientes anonimizados: repo público)")
print("=" * 70)

# Datos reales leídos de Supabase el 31/07/2026 con diag_consumo.py.
#
# ESTE REPO ES PÚBLICO: los clientes van anonimizados a propósito. Para
# el test sirven los números, no quién es cada uno. El mapa contra los
# nombres reales está en la sesión del 31/07 de ESTADO_GESTION.md, que
# tampoco los necesita para que se entienda el caso.
PRODUCCION = [
    # caso, lote, cant, dieta, DMI, producto, pct_ms, kg_tc, real_campo
    ("A — destete", "Terneros destete", 5, "2026-06-09", 2.63,
     "Producto destete precoz", 33.0, 0.400, 1.10),
    ("B — recría", "Recria hembras", 82, "2026-06-09", 5.32,
     "Producto recría", 20.0, 0.600, 0.62),
    ("C — novillos", "Novillos", 18, "2026-06-09", 11.10,
     "Producto engorde", 12.0, 1.200, 1.28),
    ("D — vacas", "Engorde vacas", 30, "2026-06-05", 5.68,
     "Producto engorde", 14.0, 0.900, 0.98),
    ("E — novillos", "novillos", 38, "2026-05-26", 7.86,
     "Producto engorde", 10.0, 0.700, 0.75),
]

print("  %-22s %8s %8s %8s %8s" % (
    "caso", "VIEJO", "NUEVO", "CAMPO", "error"))
print("  " + "-" * 60)

peor_viejo, peor_nuevo = 0.0, 0.0
for (cli, ident, cant, f_dieta, dmi, prod, pct, kgtc, real) in PRODUCCION:
    cid, lid = nuevo_lote(cli, ident, cant)
    db.guardar_dieta(
        lid, f_dieta,
        [{"nombre": prod, "pct_ms": pct, "kg_tal_cual": kgtc}],
        consumo_ms_kg=dmi,
    )
    info = sp.calcular_consumo_diario_kg(lid, prod, fecha_referencia="2026-07-31")
    nuevo = info["kg_dia"] / cant
    viejo = dmi * pct / 100.0          # fórmula anterior
    e_v = abs(viejo / real - 1) * 100
    e_n = abs(nuevo / real - 1) * 100
    peor_viejo = max(peor_viejo, e_v)
    peor_nuevo = max(peor_nuevo, e_n)
    print("  %-22s %8.3f %8.3f %8.2f %7.0f%%" % (
        cli[:22], viejo, nuevo, real, e_n))

print(f"\n  Peor error con la fórmula VIEJA: {peor_viejo:.0f}%")
print(f"  Peor error con la fórmula NUEVA: {peor_nuevo:.0f}%")
CASOS += 1
if peor_nuevo < peor_viejo:
    print("  OK    la fórmula nueva es mejor en el peor caso")
else:
    print("  FALLA la fórmula nueva NO mejora el peor caso")
    FALLAS.append("peor caso")

# El caso A es el dato roto: kg_tal_cual 0,400 no cierra con pct 33%.
# El test no lo tapa — deja constancia de que sigue mal hasta que se
# corrija el dato en la base.
CASOS += 1
if peor_nuevo > 50:
    print("  OK    el lote con el dato roto (caso A) sigue marcado como")
    print("        desviado: es un problema de DATO, no de código")
else:
    print("  FALLA se esperaba que el caso A siguiera desviado")
    FALLAS.append("caso A dato roto")

# =====================================================================
print("\n2. Fórmula base: kg_dia = kg_tal_cual × cantidad")
print("=" * 70)

cid, lid = nuevo_lote("Base SA", "L1", 100)
db.guardar_dieta(
    lid, "2026-06-01",
    [{"nombre": "Fibrogreen", "pct_ms": 14.0, "kg_tal_cual": 0.9},
     {"nombre": "Maíz grano", "pct_ms": 86.0, "kg_tal_cual": 5.56}],
    consumo_ms_kg=5.68,
)
i = sp.calcular_consumo_diario_kg(lid, "Fibrogreen", fecha_referencia="2026-07-31")
check("0,9 kg × 100 animales", i["kg_dia"], 90.0)
check("kg_tal_cual_animal reportado", i["kg_tal_cual_animal"], 0.9)
check("segundo ingrediente", sp.calcular_consumo_diario_kg(
    lid, "Maíz grano", fecha_referencia="2026-07-31")["kg_dia"], 556.0)

CASOS += 1
if "tal cual" in (i.get("fuente_kg") or ""):
    print(f"  OK    fuente_kg dice de dónde salió: '{i['fuente_kg']}'")
else:
    print(f"  FALLA fuente_kg inesperado: {i.get('fuente_kg')}")
    FALLAS.append("fuente_kg")

# MS implícita: 5.68 × 14% / 0.9 = 88%
check("ms_implicita_pct detecta dieta coherente", i["ms_implicita_pct"], 88.0, tol=1)

# =====================================================================
print("\n3. MS implícita imposible (>100%) — se calcula igual, se avisa")
print("=" * 70)

cid, lid = nuevo_lote("Incoherente SA", "L1", 10)
db.guardar_dieta(
    lid, "2026-06-01",
    [{"nombre": "Fibroter", "pct_ms": 33.0, "kg_tal_cual": 0.4}],
    consumo_ms_kg=2.63,
)
i = sp.calcular_consumo_diario_kg(lid, "Fibroter", fecha_referencia="2026-07-31")
check("usa kg_tal_cual igual", i["kg_dia"], 4.0)
check("ms_implicita_pct la delata", i["ms_implicita_pct"], 217.0, tol=1)

# =====================================================================
print("\n4. Dieta vieja sin kg_tal_cual → cae al cálculo por DMI × %")
print("=" * 70)

cid, lid = nuevo_lote("Legacy SA", "L1", 20)
db.guardar_dieta(
    lid, "2026-06-01",
    [{"nombre": "Fibrogreen", "pct_ms": 10.0}],   # sin kg_tal_cual
    consumo_ms_kg=8.0,
)
i = sp.calcular_consumo_diario_kg(lid, "Fibrogreen", fecha_referencia="2026-07-31")
check("8 × 10% × 20 animales", i["kg_dia"], 16.0)
CASOS += 1
if "estimado" in (i.get("fuente_kg") or ""):
    print(f"  OK    queda marcado como estimado: '{i['fuente_kg']}'")
else:
    print(f"  FALLA no se marcó como estimado: {i.get('fuente_kg')}")
    FALLAS.append("fallback marcado")

# kg_tal_cual en 0 explícito: mismo camino
cid, lid = nuevo_lote("Legacy2 SA", "L1", 20)
db.guardar_dieta(
    lid, "2026-06-01",
    [{"nombre": "Fibrogreen", "pct_ms": 10.0, "kg_tal_cual": 0}],
    consumo_ms_kg=8.0,
)
check("kg_tal_cual = 0 cae al fallback", sp.calcular_consumo_diario_kg(
    lid, "Fibrogreen", fecha_referencia="2026-07-31")["kg_dia"], 16.0)

# Ni kg_tal_cual ni DMI → None (no inventar un número)
cid, lid = nuevo_lote("Vacio SA", "L1", 20)
db.guardar_dieta(
    lid, "2026-06-01", [{"nombre": "Fibrogreen", "pct_ms": 10.0}],
    consumo_ms_kg=0,
)
check("sin kg_tal_cual y sin DMI → None", sp.calcular_consumo_diario_kg(
    lid, "Fibrogreen", fecha_referencia="2026-07-31"), None)

# =====================================================================
print("\n5. Rollo a voluntad (pct_ms = 0 pero kg_tal_cual cargado)")
print("=" * 70)
print("  Antes devolvía None porque salía por pct <= 0. Ese forraje SÍ")
print("  se consume y ahora se puede seguir.")

cid, lid = nuevo_lote("Rollo SA", "L1", 18)
db.guardar_dieta(
    lid, "2026-06-09",
    [{"nombre": "Cebada en grano", "pct_ms": 88.0, "kg_tal_cual": 8.9},
     {"nombre": "Rollo (a voluntad)", "pct_ms": 0.0, "kg_tal_cual": 2.4}],
    consumo_ms_kg=11.10,
)
check("rollo con pct 0 → 2,4 × 18", sp.calcular_consumo_diario_kg(
    lid, "Rollo (a voluntad)", fecha_referencia="2026-07-31")["kg_dia"], 43.2)

# =====================================================================
print("\n6. Producto que no está en la dieta → None")
print("=" * 70)

cid, lid = nuevo_lote("Ajeno SA", "L1", 10)
db.guardar_dieta(
    lid, "2026-06-01",
    [{"nombre": "Fibrogreen", "pct_ms": 10.0, "kg_tal_cual": 0.5}],
    consumo_ms_kg=8.0,
)
check("producto ausente", sp.calcular_consumo_diario_kg(
    lid, "Producto Inexistente", fecha_referencia="2026-07-31"), None)
check("lote sin dietas", sp.calcular_consumo_diario_kg(
    nuevo_lote("SinDieta SA", "L1", 10)[1], "Fibrogreen",
    fecha_referencia="2026-07-31"), None)

# =====================================================================
print("\n7. Plan de adaptación: la dieta vigente cambia con la fecha")
print("=" * 70)

cid, lid = nuevo_lote("Adaptacion SA", "L1", 50, fecha_ingreso="2026-06-01")
db.guardar_dieta(lid, "2026-06-01",
                 [{"nombre": "Fibroter", "pct_ms": 10.0, "kg_tal_cual": 0.2}],
                 consumo_ms_kg=4.0, observaciones="fase 1")
db.guardar_dieta(lid, "2026-06-10",
                 [{"nombre": "Fibroter", "pct_ms": 20.0, "kg_tal_cual": 0.5}],
                 consumo_ms_kg=4.5, observaciones="fase 2")
db.guardar_dieta(lid, "2026-06-20",
                 [{"nombre": "Fibroter", "pct_ms": 33.0, "kg_tal_cual": 0.9}],
                 consumo_ms_kg=5.0, observaciones="fase 3")

check("fase 1 (05/06)", sp.calcular_consumo_diario_kg(
    lid, "Fibroter", fecha_referencia="2026-06-05")["kg_dia"], 10.0)
check("fase 2 (15/06)", sp.calcular_consumo_diario_kg(
    lid, "Fibroter", fecha_referencia="2026-06-15")["kg_dia"], 25.0)
check("fase 3 (31/07)", sp.calcular_consumo_diario_kg(
    lid, "Fibroter", fecha_referencia="2026-07-31")["kg_dia"], 45.0)

i = sp.calcular_consumo_diario_kg(lid, "Fibroter", fecha_referencia="2026-06-15")
CASOS += 1
if i.get("fase_vigente") == "fase 2":
    print("  OK    fase_vigente sigue reportando la fase correcta")
else:
    print(f"  FALLA fase_vigente = {i.get('fase_vigente')}")
    FALLAS.append("fase_vigente")

# Fecha anterior a toda dieta → usa la más antigua (fallback original)
check("antes de la primera dieta", sp.calcular_consumo_diario_kg(
    lid, "Fibroter", fecha_referencia="2026-05-01")["kg_dia"], 10.0)

# =====================================================================
print("\n8. Override de DMI (ajuste por clima) escala, no reemplaza")
print("=" * 70)

cid, lid = nuevo_lote("Clima SA", "L1", 10)
db.guardar_dieta(
    lid, "2026-06-01",
    [{"nombre": "Fibrogreen", "pct_ms": 14.0, "kg_tal_cual": 1.0}],
    consumo_ms_kg=5.0,
)
# override 5.5 sobre DMI 5.0 → +10% → 1,1 kg × 10 animales
check("override +10%", sp.calcular_consumo_diario_kg(
    lid, "Fibrogreen", dmi_kg_dia_override=5.5,
    fecha_referencia="2026-07-31")["kg_dia"], 11.0)
check("override -20%", sp.calcular_consumo_diario_kg(
    lid, "Fibrogreen", dmi_kg_dia_override=4.0,
    fecha_referencia="2026-07-31")["kg_dia"], 8.0)
check("sin override no cambia", sp.calcular_consumo_diario_kg(
    lid, "Fibrogreen", fecha_referencia="2026-07-31")["kg_dia"], 10.0)

# =====================================================================
print("\n9. Bajas del lote: la cantidad vigente sigue mandando")
print("=" * 70)

cid, lid = nuevo_lote("Bajas SA", "L1", 100, fecha_ingreso="2026-06-01")
db.guardar_dieta(
    lid, "2026-06-01",
    [{"nombre": "Fibrogreen", "pct_ms": 14.0, "kg_tal_cual": 1.0}],
    consumo_ms_kg=5.0,
)
db.crear_movimiento_lote(lid, "2026-07-01", "muerte", 10)
check("antes de la baja (100)", sp.calcular_consumo_diario_kg(
    lid, "Fibrogreen", fecha_referencia="2026-06-15")["kg_dia"], 100.0)
check("después de la baja (90)", sp.calcular_consumo_diario_kg(
    lid, "Fibrogreen", fecha_referencia="2026-07-31")["kg_dia"], 90.0)

# =====================================================================
print("\n10. Punta a punta: stock, días restantes y alerta")
print("=" * 70)

cid, lid = nuevo_lote("EndToEnd SA", "L1", 50, fecha_ingreso="2026-07-01")
db.guardar_dieta(
    lid, "2026-07-01",
    [{"nombre": "Fibrogreen", "pct_ms": 14.0, "kg_tal_cual": 1.0}],
    consumo_ms_kg=5.0,
)
# 50 animales × 1 kg = 50 kg/día. Entrego 30 bolsas = 900 kg el 21/07.
db.crear_entrega(cid, "Fibrogreen", 900.0, "2026-07-21", lote_id=lid,
                 formato="bolsa", cantidad_bolsas=30, kg_por_bolsa=30)
st = sp.calcular_stock_actual(cid, lid, "Fibrogreen",
                              fecha_referencia="2026-07-31")
check("consumo diario", st["consumo_diario_kg"], 50.0)
# 10 días consumidos (21→30 inclusive) = 500 kg → quedan 400
check("kg restantes", st["kg_restantes_hoy"], 400.0)
check("días restantes", st["dias_restantes"], 8.0, tol=1)

CASOS += 1
if st["diagnostico_uso"] != "historial_incompleto":
    print(f"  OK    diagnóstico sano: '{st['diagnostico_uso']}'")
else:
    print("  FALLA quedó marcado como historial incompleto")
    FALLAS.append("diagnostico")

# El mismo lote con la fórmula vieja habría dado 5.0 × 14% = 0,7 kg/animal
# = 35 kg/día, o sea 500 kg restantes y 14 días: 6 días de más.
print("  (con la fórmula vieja daba 35 kg/día → 14 días: 6 de más)")

# =====================================================================
print("\n" + "=" * 70)
print(f"RESULTADO: {CASOS - len(FALLAS)}/{CASOS} casos OK")
if FALLAS:
    print("FALLARON: " + ", ".join(FALLAS))
print("=" * 70)

shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FALLAS else 0)
