#!/usr/bin/env python3
"""
Smoke test end-to-end del módulo de impacto productivo.

Carga el lote real "Vaquillonas" de la DB, simula un evento de frío
operativo (T° mín 1°C, viento 20 km/h, HR 85%, 2 días) y dispara:
  - el cálculo NRC/NASEM
  - el LLM diario (lectura técnica)
  - el LLM de acciones

Verifica que el LLM cite los rangos exactos calculados y no invente kg.

Uso:
    .venv/bin/python scripts/test_impacto_llm.py
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from src.impacto_productivo import estimar_impacto_frio, formato_impacto_texto
from src.ai_analisis_semanal import (
    generar_analisis_diario_llm, generar_acciones_llm,
)


def main():
    # Datos del lote real (Mauricio, vaquillonas 220 kg, 50 cab.)
    cliente = {
        "nombre": "Mauricio Suarez",
        "establecimiento": "Catriló",
        "localidad": "Catriló, La Pampa",
    }
    lote = {
        "lote": "Vaquillonas",
        "categoria": "vaquillona",
        "raza": "angus",
        "peso_promedio_kg": 220,
        "cantidad_animales": 50,
    }

    # Escenario B: frío operativo
    clima_actual = {
        "temp_c": 4,
        "temp_min": 1,
        "humedad_pct": 85,
        "viento_kmh": 20,
        "lluvia_mm": 0,
        "thi": 50,
    }
    alertas = [{
        "tipo": "frio",
        "nivel": "operativo",
        "severidad": "warning",
        "titulo": "Frío operativo previsto",
        "accion": "(placeholder del motor)",
        "_contexto": {
            "fecha": "2026-05-15",
            "temp_min": 1,
            "t_min": 1,
            "viento_kmh": 20,
            "barro": False,
            "lluvia_mm": 0,
        },
    }]
    alertas_por_lote = [{**lote, "alertas": alertas}]

    # === Cálculo NRC ===
    imp = estimar_impacto_frio(
        peso_kg=lote["peso_promedio_kg"],
        categoria=lote["categoria"],
        raza=lote["raza"],
        t_min_c=1, viento_kmh=20, humedad_pct=85,
        barro=False, pelaje_mojado=False,
        dias_evento=2, cantidad=lote["cantidad_animales"],
    )
    imp_txt = formato_impacto_texto(imp)
    print("=" * 72)
    print("BLOQUE INYECTADO AL LLM (dato a citar tal cual):")
    print("=" * 72)
    print(imp_txt)
    print()

    # === LLM diario ===
    print("=" * 72)
    print("📖 LECTURA TÉCNICA generada por el LLM:")
    print("=" * 72)
    lectura = generar_analisis_diario_llm(
        cliente=cliente,
        alertas_por_lote=alertas_por_lote,
        clima_actual=clima_actual,
        etapa="inicio",
        dias_alerta_previos=1,
        peor_tipo="frio",
        ocurre_hoy=True,
        dias_hasta_evento=0,
        fecha_inicio_evento="2026-05-15",
        impacto_productivo_txt=imp_txt,
    )
    print(lectura or "(LLM no devolvió texto)")
    print()

    # === LLM acciones ===
    print("=" * 72)
    print("⚡ ACCIONES generadas por el LLM:")
    print("=" * 72)
    clima_a = {
        "temperatura": clima_actual["temp_c"],
        "min_nocturna": clima_actual["temp_min"],
        "viento_kmh": clima_actual["viento_kmh"],
        "lluvia_mm": clima_actual["lluvia_mm"],
        "humedad_pct": clima_actual["humedad_pct"],
        "thi": clima_actual["thi"],
    }
    acciones = generar_acciones_llm(
        cliente=cliente,
        tipo="frio",
        nivel="operativo",
        categoria="vaquillona",
        clima=clima_a,
        etapa="inicio",
        dias_alerta_previos=1,
        ocurre_hoy=True,
        dias_hasta_evento=0,
        impacto_productivo_txt=imp_txt,
    )
    if acciones:
        for cat in ("inmediatas", "operativas", "nutricionales"):
            items = acciones.get(cat, [])
            if items:
                print(f"\n{cat.upper()}:")
                for it in items:
                    print(f"  • {it}")
    else:
        print("(LLM de acciones no devolvió nada — revisar log)")
    print()
    print("=" * 72)
    print("VERIFICACIÓN MANUAL:")
    print(" - ¿La lectura técnica cierra con los rangos 0.12-0.18 kg/día")
    print("   o 12-18 kg en el lote, sin inventar otros números?")
    print(" - ¿Alguna acción nutricional cita esos mismos rangos para")
    print("   anclar el costo de no actuar?")
    print("=" * 72)


if __name__ == "__main__":
    main()
