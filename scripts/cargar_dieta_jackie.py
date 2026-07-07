#!/usr/bin/env python3
"""Carga la dieta de Jackie Graves (Recria hembras, lote id=7) al
historial. Esta dieta se perdió por el corte de saldo del 22/05/26 y
fue reconstruida a partir de los datos validados con ChatGPT.

Uso:
    python3 scripts/cargar_dieta_jackie.py

Si el lote ya tiene una dieta cargada con esa misma fecha, no
inserta nada (idempotente).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permitir importar src desde el root del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import database as db  # noqa: E402


LOTE_ID = 7
FECHA = "2026-05-22"

COMPOSICION = [
    {
        "nombre": "Maíz grano",
        "kg_tal_cual": 2.50,
        "kg_ms": 2.15,
        "pct_ms": 39.0,
    },
    {
        "nombre": "Fibrogreen Plus",
        "kg_tal_cual": 1.00,
        "kg_ms": 0.88,
        "pct_ms": 16.0,
    },
    {
        "nombre": "Silaje planta entera maíz",
        "kg_tal_cual": 7.00,
        "kg_ms": 2.47,
        "pct_ms": 45.0,
    },
]

OBSERVACIONES = (
    "Dieta de recría para vaquillonas 220 kg (Angus). "
    "Objetivo: 320-340 kg al servicio (octubre 2026). "
    "ADPV: 0,65-0,75 kg/día. DMI total: 5,5 kg MS/cab/día. "
    "Maíz + Fibrogreen Plus mezclados en comedero LINEAL (2 "
    "comidas/día para reducir acidosis). Silaje planta entera "
    "maíz APARTE en autoconsumo (libre disposición). "
    "Monensina en dieta final: ~38 ppm (240 ppm × 16% inclusión). "
    "NNP: 35 g/cab/día (manejable con adaptación + fibra "
    "disponible). Validada con criterios técnicos del 22/05/2026."
)


def main() -> None:
    # Idempotencia: si ya hay una dieta con esta fecha para el lote,
    # no la duplicamos.
    existentes = db.listar_dietas(LOTE_ID)
    if any(d.get("fecha") == FECHA for d in existentes):
        print(
            f"⚠️  Ya existe una dieta para el lote {LOTE_ID} con "
            f"fecha {FECHA}. Skip."
        )
        return

    dieta_id = db.guardar_dieta(
        lote_id=LOTE_ID,
        fecha=FECHA,
        composicion=COMPOSICION,
        costo_dia=2000.0,
        pb_pct=12.0,
        em_mcal_dia=14.5,
        consumo_ms_kg=5.5,
        nnp_pct=0.65,
        observaciones=OBSERVACIONES,
    )
    print(
        f"✅ Dieta guardada al lote {LOTE_ID} (Jackie Graves — "
        f"Recria hembras) con id={dieta_id}."
    )
    print("Composición:")
    for ing in COMPOSICION:
        print(
            f"  - {ing['nombre']:30}  "
            f"{ing['kg_tal_cual']:.2f} kg tal cual  "
            f"({ing['kg_ms']:.2f} kg MS, {ing['pct_ms']}%)"
        )


if __name__ == "__main__":
    main()
