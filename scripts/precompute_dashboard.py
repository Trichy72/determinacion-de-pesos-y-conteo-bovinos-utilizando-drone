"""Precomputa el bloque de logística del dashboard y guarda el
resultado como blob JSON en la tabla `dashboard_cache` de Supabase.

Se corre desde un cron externo (GitHub Actions cada 5 min en horario
laboral AR) — así cuando el asesor abre la app en el campo, el
dashboard usa esa data cacheada (1 query ~200ms) en lugar de
recalcular ~50 queries × 20 clientes cada visita (30-60s).

Uso:
    DATABASE_URL=postgresql://... python scripts/precompute_dashboard.py

Salida: log a stdout con "Precompute OK — N clientes, X entregas, Y s"
Exit code 0 si todo bien, != 0 si hubo un error (para que el cron marque
el run como failed y aparezca en la vista de Actions).

Requisitos:
  - DATABASE_URL en env (Supabase session pooler).
  - Las mismas dependencias de la app (psycopg2-binary sobre todo).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


# Permitir `from src...` desde cualquier cwd (por si el runner
# corre desde otro directorio).
PROYECTO = Path(__file__).resolve().parents[1]
if str(PROYECTO) not in sys.path:
    sys.path.insert(0, str(PROYECTO))


def main() -> int:
    inicio = time.time()

    # Import diferido para que el sys.path manipulation aplique.
    from src import database as db
    from src.dashboard_precompute import calcular_dashboard_logistica

    # Log del backend en uso (útil para debug de runs de GH Actions:
    # si algún día DATABASE_URL desaparece de los secrets, esto
    # canta que fue a SQLite en el runner y no a Supabase).
    try:
        backend = db._backend_activo()
    except Exception:
        backend = "desconocido"
    print(f"[precompute_dashboard] backend={backend}", flush=True)

    # Correr el cálculo pesado.
    try:
        resultado = calcular_dashboard_logistica()
    except Exception as e:
        print(
            f"[precompute_dashboard] ERROR calculando: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 2

    # Métricas para el log
    n_clientes = len({
        f.get("Cliente") for f in resultado.get("filas_log", [])
    } | {
        a.get("cliente") for a in resultado.get("autonomia", [])
    })
    n_alertas = len(resultado.get("filas_log", []))
    n_autonomia = len(resultado.get("autonomia", []))
    n_sin_dieta = len(resultado.get("entregas_sin_dieta", []))

    # Guardar en la tabla dashboard_cache
    try:
        db.guardar_dashboard_cache("logistica_v1", resultado)
    except Exception as e:
        print(
            f"[precompute_dashboard] ERROR guardando cache: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 3

    elapsed = time.time() - inicio
    # Formato del log — parseable a ojo desde la UI de GH Actions.
    print(
        f"Precompute OK — {n_clientes} clientes, "
        f"{n_autonomia} entregas ({n_alertas} alertas, "
        f"{n_sin_dieta} sin dieta), "
        f"tiempo {elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
