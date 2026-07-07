#!/usr/bin/env python3
"""
Cron semanal — lunes 8:00 AM.

Manda al ADMIN (Mauricio) un email por cada cliente activo con la
demanda diaria/semanal/mensual de insumos por lote, marcando los
productos HMS. Útil para planificar la logística de la semana antes
de que arranque.

A diferencia de las alertas climáticas, este informe es INTERNO:
no se manda al cliente, solo a vos. El cliente nunca ve este mail.

Filosofía:
- 1 email por cliente, cada lunes 8 AM
- Solo clientes activos con al menos un lote y una dieta cargada
- Muestra demanda por lote + total cliente
- Distingue productos HMS (los que vos vendés) del resto

Uso:
    python scripts/informe_demanda_semanal.py
    python scripts/informe_demanda_semanal.py --dry-run
    python scripts/informe_demanda_semanal.py --solo-cliente Bergondi
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
from src import stock_producto as sp


def setup_logging() -> logging.Logger:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        log_dir / f"demanda_semanal_{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    logger = logging.getLogger("informe_demanda_semanal")
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Informe semanal de demanda de insumos por cliente."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Calcular y mostrar el log, pero no mandar emails.",
    )
    parser.add_argument(
        "--solo-cliente", type=str, default=None,
        help="Procesar solo el cliente cuyo nombre matchea (case-insensitive).",
    )
    args = parser.parse_args()

    log = setup_logging()
    log.info(f"=== INFORME DEMANDA SEMANAL "
             f"{datetime.now().isoformat(timespec='seconds')} ===")
    if args.dry_run:
        log.info("MODO DRY-RUN — no se enviará nada.")

    # Lock para evitar doble ejecución si launchd dispara dos veces
    lock_fd = adquirir_lock_proceso("informe_demanda_semanal")
    if not lock_fd:
        log.warning(
            "Otro proceso está corriendo. Salgo para no duplicar envíos."
        )
        return 0

    try:
        # Config SMTP — el destinatario es siempre el admin
        cfg = ae.cargar_config_smtp() or {}
        if not cfg.get("host"):
            log.error("No hay config SMTP cargada. Saliendo.")
            return 1
        admin_email = cfg.get("admin_email") or cfg.get("from_email")
        if not admin_email:
            log.error(
                "No hay admin_email ni from_email en la config SMTP."
            )
            return 1
        log.info(f"Destinatario: {admin_email}")

        # Iterar clientes activos
        clientes = db.listar_clientes()
        log.info(f"Total clientes en DB: {len(clientes)}")

        procesados = 0
        enviados = 0
        errores = 0
        for c in clientes:
            if (c.get("estado") or "activo") != "activo":
                continue
            if args.solo_cliente:
                nombre = (c.get("nombre") or "").lower()
                if args.solo_cliente.lower() not in nombre:
                    continue

            procesados += 1
            nombre_cli = c.get("nombre") or "(sin nombre)"
            try:
                demanda = sp.demanda_insumos_cliente(c["id"])
            except Exception as e:
                log.exception(
                    f"  Error calculando demanda de {nombre_cli}: {e}"
                )
                errores += 1
                continue

            if not demanda.get("lotes"):
                log.info(
                    f"  {nombre_cli}: sin lotes activos con dieta. Skip."
                )
                continue

            tot = demanda.get("total_cliente") or {}
            n_lotes = len(demanda["lotes"])
            n_anim = tot.get("cantidad_animales_total") or 0
            mezcla = tot.get("mezcla_total_kg_dia") or 0
            log.info(
                f"  {nombre_cli}: {n_lotes} lote(s), {n_anim} cab., "
                f"mezcla {mezcla:.0f} kg/día"
            )

            try:
                subject, html, text = (
                    ae.componer_informe_demanda_cliente(c, demanda)
                )
            except Exception as e:
                log.exception(
                    f"  Error componiendo email de {nombre_cli}: {e}"
                )
                errores += 1
                continue

            if args.dry_run:
                log.info(f"  [DRY-RUN] -> {admin_email}: {subject}")
                continue

            try:
                # con_bcc_admin=False porque YA va al admin (no al cliente)
                ok, msg = ae.enviar_email(
                    cfg, [admin_email], subject, html, text,
                    con_bcc_admin=False,
                )
                if ok:
                    log.info(
                        f"  ✓ Enviado: {nombre_cli} -> {admin_email}"
                    )
                    enviados += 1
                else:
                    log.warning(
                        f"  ⚠ Falló envío {nombre_cli}: {msg}"
                    )
                    errores += 1
            except Exception as e:
                log.exception(
                    f"  Error enviando email {nombre_cli}: {e}"
                )
                errores += 1

        log.info(
            f"=== FIN — Procesados {procesados}, "
            f"enviados {enviados}, errores {errores} ==="
        )
        return 0 if errores == 0 else 2

    finally:
        liberar_lock(lock_fd)


if __name__ == "__main__":
    sys.exit(main())
