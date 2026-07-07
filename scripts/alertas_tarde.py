#!/usr/bin/env python3
"""
Cron de las 18:00 — pronóstico nocturno HMS.

A diferencia de alertas_diarias.py (8:00 AM), este script SOLO manda si
hay algo concreto para las próximas 12-18 hs:
  - Frío nocturno con severidad warning/critica
  - Tormenta / lluvia fuerte / granizo / vientos intensos
  - Calor que no afloja en la noche (sostenido > 22°C con humedad)
  - Alertas oficiales del SMN para esa zona

Si está todo tranquilo, NO MANDA NADA (silencio total, no spam).

Tampoco manda digest al admin (eso ya se mandó a la mañana).

Uso:
    python scripts/alertas_tarde.py [--dry-run] [--solo-cliente NOMBRE]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import database as db
from src.locking import adquirir_lock_proceso, liberar_lock
from src import alertas_email as ae
from src import whatsapp as wa
from src.procesador_bajas import procesar_bajas_pendientes

# Reusar las funciones del cron diario para no duplicar lógica
from scripts.alertas_diarias import (
    calcular_alertas_cliente,
    enviar_alertas_a_cliente,
)


# =====================================================================
# LOGGING
# =====================================================================

def setup_logging() -> logging.Logger:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"tarde_{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger("alertas_tarde")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# =====================================================================
# FILTRO DE ALERTAS NOCTURNAS
# =====================================================================

# Palabras que indican que una alerta es relevante para mandar a la tarde
# (cosas que afectan la noche / madrugada / mañana siguiente).
_PALABRAS_NOCTURNAS = (
    "tormenta", "lluvia", "granizo", "viento",
    "frente frío", "frente frio", "helada", "escarcha",
    "anegamient", "precipitación", "precipitacion",
    "nevada", "nieve", "alerta oficial",
)


def es_alerta_relevante_tarde(alerta: dict) -> bool:
    """¿La alerta es importante para mandar a las 18:00?

    Criterio:
      - Severidad warning o critica (las preventivas/info no se mandan)
      - Tipo "frio" (frío nocturno siempre relevante)
      - Tipo "calor" SÓLO si severidad crítica (ola de calor que no afloja)
      - Cualquier título que mencione tormenta, viento, lluvia, granizo,
        helada, nieve, alerta oficial.
    """
    sev = (alerta.get("severidad") or "").lower()
    if sev not in ("warning", "critica"):
        return False

    tipo = (alerta.get("tipo") or "").lower()
    if tipo == "frio":
        return True
    if tipo == "calor" and sev == "critica":
        return True

    titulo = (alerta.get("titulo") or "").lower()
    descripcion = (alerta.get("descripcion") or "").lower()
    blob = f"{titulo} {descripcion}"
    if any(p in blob for p in _PALABRAS_NOCTURNAS):
        return True

    return False


def filtrar_datos_para_tarde(datos: dict) -> dict:
    """Reduce datos["alertas_por_lote"] dejando SOLO alertas nocturnas
    relevantes. Devuelve los datos modificados (in-place).

    Si después del filtro no queda ninguna alerta, devuelve datos con
    `alertas_por_lote = []` y `tiene_algo = False` para que el caller
    sepa que no hay nada que mandar.
    """
    nuevos_lotes = []
    for l in datos.get("alertas_por_lote", []):
        relevantes = [
            a for a in l.get("alertas", [])
            if es_alerta_relevante_tarde(a)
        ]
        if relevantes:
            nuevos_lotes.append({
                **l,
                "alertas": relevantes,
            })

    datos["alertas_por_lote"] = nuevos_lotes
    n_total = sum(len(l["alertas"]) for l in nuevos_lotes)
    datos["tiene_algo"] = n_total > 0
    return datos


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="No envía mails, solo muestra qué haría")
    parser.add_argument("--solo-cliente", default=None,
                         help="Solo procesa este cliente por nombre")
    args = parser.parse_args()

    log = setup_logging()

    # Lock global: evitar dos instancias en paralelo (launchd al despertar
    # la Mac puede disparar el job atrasado + el StartCalendarInterval).
    lock_fd = adquirir_lock_proceso("alertas_tarde")
    if lock_fd is None:
        log.info("=== abortado: otra instancia ya corre ===")
        return 0

    log.info("=== ALERTAS TARDE (pronóstico nocturno) — INICIO ===")

    db.init_db()

    cfg = ae.cargar_config_smtp()
    ok, err = ae.config_valida(cfg)
    if not ok and not args.dry_run:
        log.error(f"Config SMTP inválida: {err}")
        return 1

    # Procesar bajas (igual que en la mañana, por si llegaron BAJAs durante
    # el día y queremos no mandar a clientes dados de baja recién).
    if not args.dry_run and ok:
        try:
            n_bajas, emails_dados_baja = procesar_bajas_pendientes(cfg)
            if n_bajas > 0:
                log.info(f"  ✓ {n_bajas} baja(s) procesada(s): "
                          f"{', '.join(emails_dados_baja)}")
        except Exception as e:
            log.warning(f"  Error al procesar bajas: {e}")

    fecha = datetime.now().strftime("%d/%m/%Y")

    # Domingos: solo críticas (no warning) — para no molestar.
    es_domingo = datetime.now().weekday() == 6

    clientes = db.listar_clientes()
    if args.solo_cliente:
        clientes = [c for c in clientes
                    if args.solo_cliente.lower() in c["nombre"].lower()]
    log.info(f"Clientes a evaluar: {len(clientes)}")

    enviados = 0
    errores = 0
    n_clientes_con_aviso = 0
    n_total_clientes_ok = 0

    asunto_tarde = f"🌙 Pronóstico nocturno HMS — {fecha}"
    cabecera_wa_tarde = "🌙 NOCTURNO HMS"

    for c in clientes:
        log.info(f"Cliente: {c['nombre']}")
        try:
            datos = calcular_alertas_cliente(c, log)
            if datos.get("error_api"):
                log.warning(f"  {c['nombre']}: error de API, skip.")
                continue
            n_total_clientes_ok += 1

            datos = filtrar_datos_para_tarde(datos)
            if not datos["tiene_algo"]:
                log.info(f"  {c['nombre']}: sin alertas nocturnas, skip.")
                continue

            # Coherencia: usar el mismo clasificador productivo que la
            # diaria y la semanal. Tarde manda si operativo/crítico,
            # NO si solo atención.
            from scripts.alertas_diarias import _nivel_productivo_maximo
            nivel_max = _nivel_productivo_maximo(datos)
            log.info(f"  {c['nombre']}: nivel productivo nocturno = "
                      f"{nivel_max}")

            if es_domingo:
                if nivel_max != "critico":
                    log.info(f"  {c['nombre']}: domingo, nivel={nivel_max}, "
                              f"no es crítico → no se manda.")
                    continue
            else:
                if nivel_max not in ("operativo", "critico"):
                    log.info(f"  {c['nombre']}: nivel={nivel_max} no "
                              f"justifica aviso nocturno → no se manda.")
                    continue

            n_alerts = sum(len(l["alertas"]) for l in datos["alertas_por_lote"])
            log.info(f"  {c['nombre']}: {n_alerts} alerta(s) nocturna(s) "
                      f"en {len(datos['alertas_por_lote'])} lote(s)")
            n_clientes_con_aviso += 1

            e, err_n = enviar_alertas_a_cliente(
                cfg, datos, log,
                dry_run=args.dry_run,
                tipo="tarde",
                asunto_override=asunto_tarde,
                cabecera_wa_override=cabecera_wa_tarde,
                nivel_productivo=nivel_max,
            )
            enviados += e
            errores += err_n
        except Exception as e:
            log.exception(f"  Error procesando {c['nombre']}: {e}")
            errores += 1

    log.info(
        f"=== FIN — Clientes OK: {n_total_clientes_ok}, con aviso "
        f"nocturno: {n_clientes_con_aviso}, enviados: {enviados}, "
        f"errores: {errores} ==="
    )
    if n_clientes_con_aviso == 0:
        log.info("  → Día sin alertas nocturnas. No se mandó nada (correcto).")

    liberar_lock(lock_fd)
    return 0 if errores == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
