#!/usr/bin/env python3
"""
Smoke test del flujo completo con override de lote.

Este script:
  1. Aplica la migración de DB (sumar columnas nuevas si faltan).
  2. Carga overrides en el lote real (ADPV objetivo y energía dieta).
  3. Lee el lote desde la DB (igual que lo haría el cron).
  4. Calcula el impacto productivo SIN override (default por categoría).
  5. Calcula el impacto CON override (datos cargados por el productor).
  6. Llama al LLM diario con cada uno y muestra la diferencia en la
     lectura técnica que recibiría el productor.

Uso:
    .venv/bin/python scripts/test_flujo_completo.py
    .venv/bin/python scripts/test_flujo_completo.py --lote-id 2
    .venv/bin/python scripts/test_flujo_completo.py --reset
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from src import database as db
from src.impacto_productivo import estimar_impacto_frio, formato_impacto_texto
from src.ai_analisis_semanal import generar_analisis_diario_llm


def _print_seccion(titulo: str) -> None:
    print()
    print("=" * 72)
    print(titulo)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lote-id", type=int, default=None,
        help="ID del lote a usar. Si no se pasa, busca el primer lote "
              "activo con peso conocido.",
    )
    parser.add_argument(
        "--adpv-override", type=float, default=1.05,
        help="ADPV objetivo a cargar (kg/día). Default: 1.05.",
    )
    parser.add_argument(
        "--energia-override", type=float, default=2.75,
        help="Energía dieta a cargar (Mcal EM/kg MS). Default: 2.75.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Borra los overrides del lote y termina (sin tests).",
    )
    parser.add_argument(
        "--sin-llm", action="store_true",
        help="No llama al LLM (solo muestra los cálculos numéricos).",
    )
    args = parser.parse_args()

    # 1) Migración
    print("→ Aplicando migración de DB (idempotente)...")
    db.init_db()
    print("  ✓ Migración OK")

    # 2) Identificar lote a usar
    lotes_activos = []
    for c in db.listar_clientes():
        if c.get("estado", "activo") != "activo":
            continue
        for l in db.listar_lotes(cliente_id=c["id"], estado="activo"):
            peso = l.get("ultimo_peso_kg") or l.get("peso_ingreso_kg")
            if peso and peso > 0:
                lotes_activos.append((c, l))

    if not lotes_activos:
        print("✗ No hay lotes activos con peso cargado. "
              "Cargá un peso en algún lote y reintenta.")
        sys.exit(1)

    if args.lote_id:
        elegido = next(((c, l) for c, l in lotes_activos
                         if l["id"] == args.lote_id), None)
        if not elegido:
            print(f"✗ Lote id={args.lote_id} no encontrado entre los "
                  f"activos con peso.")
            sys.exit(1)
        cliente, lote = elegido
    else:
        cliente, lote = lotes_activos[0]

    print(f"→ Lote elegido: id={lote['id']} · "
          f"{cliente['nombre']} · {lote['identificador']} · "
          f"{lote.get('categoria')} {lote.get('raza')}")

    # 3) Aplicar override (o resetear si --reset)
    if args.reset:
        db.actualizar_lote(
            lote["id"],
            adpv_objetivo_kg=None,
            energia_dieta_mcal_em_kg_ms=None,
        )
        print("✓ Overrides borrados. El lote vuelve a usar defaults "
              "por categoría.")
        sys.exit(0)

    print(f"→ Cargando overrides en el lote: "
          f"ADPV={args.adpv_override} kg/día, "
          f"Energía={args.energia_override} Mcal EM/kg MS")
    db.actualizar_lote(
        lote["id"],
        adpv_objetivo_kg=args.adpv_override,
        energia_dieta_mcal_em_kg_ms=args.energia_override,
    )

    # 4) Releer el lote (con override aplicado)
    lote = db.obtener_lote(lote["id"])
    peso_kg = lote.get("ultimo_peso_kg") or lote.get("peso_ingreso_kg")
    cantidad = lote.get("cantidad_inicial") or 1

    # Escenario climático de frío operativo (T° mín 1°C, viento 20,
    # HR 85%, 2 días). Idéntico al smoke test anterior para comparar.
    escenario = {
        "t_min_c": 1, "viento_kmh": 20, "humedad_pct": 85,
        "dias_evento": 2, "barro": False, "pelaje_mojado": False,
    }

    _print_seccion("CONDICIONES DEL TEST")
    print(f"  Lote: {lote['identificador']} ({lote.get('categoria')}, "
          f"{lote.get('raza')})")
    print(f"  Peso promedio: {peso_kg} kg · Cantidad: {cantidad} cab.")
    print(f"  Clima simulado: T° mín 1°C, viento 20 km/h, HR 85%, "
          f"2 días consecutivos")
    print(f"  Override cargado:")
    print(f"    - ADPV objetivo: {lote.get('adpv_objetivo_kg')} kg/día")
    print(f"    - Energía dieta: "
          f"{lote.get('energia_dieta_mcal_em_kg_ms')} Mcal EM/kg MS")

    # 5) Impacto SIN override
    imp_sin = estimar_impacto_frio(
        peso_kg=peso_kg,
        categoria=lote.get("categoria", ""),
        raza=lote.get("raza", ""),
        cantidad=cantidad,
        **escenario,
    )
    txt_sin = formato_impacto_texto(imp_sin) if imp_sin else ""

    # 6) Impacto CON override (usando los campos guardados en DB)
    imp_con = estimar_impacto_frio(
        peso_kg=peso_kg,
        categoria=lote.get("categoria", ""),
        raza=lote.get("raza", ""),
        cantidad=cantidad,
        adpv_objetivo_kg=lote.get("adpv_objetivo_kg"),
        energia_dieta_mcal_em_kg_ms=lote.get(
            "energia_dieta_mcal_em_kg_ms"
        ),
        **escenario,
    )
    txt_con = formato_impacto_texto(imp_con) if imp_con else ""

    _print_seccion("IMPACTO SIN OVERRIDE (defaults categoría)")
    print(txt_sin)

    _print_seccion("IMPACTO CON OVERRIDE (datos cargados por productor)")
    print(txt_con)

    if args.sin_llm:
        return

    # 7) Llamar al LLM diario con cada uno para ver cómo cambia el
    #    texto que recibe el productor.
    contexto_base = dict(
        cliente={
            "nombre": cliente["nombre"],
            "establecimiento": cliente.get("establecimiento", ""),
            "localidad": cliente.get("localidad", ""),
        },
        alertas_por_lote=[{
            "lote": lote["identificador"],
            "categoria": lote.get("categoria", ""),
            "raza": lote.get("raza", ""),
            "peso_promedio_kg": peso_kg,
            "cantidad_animales": cantidad,
            "alertas": [{
                "tipo": "frio",
                "nivel": "operativo",
                "severidad": "warning",
                "titulo": "Frío operativo en curso",
                "_contexto": {"t_min": 1, "viento_kmh": 20, "barro": False},
            }],
        }],
        clima_actual={"temp_c": 4, "temp_min": 1, "humedad_pct": 85,
                       "viento_kmh": 20, "lluvia_mm": 0, "thi": 50},
        etapa="inicio", dias_alerta_previos=1,
        peor_tipo="frio", ocurre_hoy=True, dias_hasta_evento=0,
        fecha_inicio_evento="",
    )

    _print_seccion("📖 LECTURA TÉCNICA · LLM con impacto SIN override")
    print(generar_analisis_diario_llm(
        **contexto_base, impacto_productivo_txt=txt_sin,
    ) or "(LLM no devolvió texto)")

    _print_seccion("📖 LECTURA TÉCNICA · LLM con impacto CON override")
    print(generar_analisis_diario_llm(
        **contexto_base, impacto_productivo_txt=txt_con,
    ) or "(LLM no devolvió texto)")

    _print_seccion("VERIFICACIÓN MANUAL")
    print("  ¿La segunda lectura cita el % de objetivo más preciso?")
    print("  ¿Ambas usan los rangos exactos sin inventar números?")
    print()
    print("Para revertir los overrides del lote y volver a defaults:")
    print(f"  .venv/bin/python {sys.argv[0]} --lote-id {lote['id']} --reset")


if __name__ == "__main__":
    main()
