#!/usr/bin/env python3
"""
Job horario de alertas CRÍTICAS por WhatsApp.

Corre cada 1-2 hs (cron: 0 */1 * * *). Detecta alertas severidad='critica'
recién aparecidas (no enviadas en las últimas 12 hs) y manda WhatsApp:

  - al admin (vos): mensaje breve con el cliente afectado
  - al cliente: alerta detallada usando plantilla aprobada de Meta
                (porque está "fuera de ventana 24hs" en general)

Deduplicación: cada alerta se identifica por (cliente_id, lote, severidad,
título). Si ya se mandó el mismo evento en las últimas 12 hs, se skipea.

Uso:
    python scripts/alertas_criticas.py [--dry-run] [--solo-cliente NOMBRE]

Cron Linux:
    0 */1 * * *  cd /ruta/proyecto && /usr/bin/python3 scripts/alertas_criticas.py

launchd macOS: ver scripts/com.hms.alertas-criticas.plist
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
from src import whatsapp as wa
from src.clima import (
    obtener_clima, generar_alertas_predictivas, calcular_thi, clasificar_thi,
    geocodificar, geocodificar_manual, evaluar_y_componer_mensajes,
)


def setup_logging() -> logging.Logger:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"criticas_{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger("alertas_criticas")
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


def detectar_alertas_criticas_cliente(cliente: dict, log: logging.Logger) -> list:
    """Devuelve [{lote, alerta_dict}] solo con severidad='critica'."""
    nombre = cliente.get("nombre", "")
    lat = cliente.get("lat")
    lon = cliente.get("lon")
    localidad = cliente.get("localidad", "")

    if lat and lon:
        geo = geocodificar_manual(float(lat), float(lon), localidad)
    elif localidad:
        geo = geocodificar(localidad)
    else:
        return []

    if not geo:
        return []

    clima = obtener_clima(geo["lat"], geo["lon"])
    if not clima:
        return []

    lotes = db.listar_lotes(cliente_id=cliente["id"], estado="activo")
    criticas = []
    for l in lotes:
        alertas = generar_alertas_predictivas(clima, categoria=l.get("categoria", ""))
        for a in alertas:
            if a.get("severidad") == "critica":
                criticas.append({
                    "lote": l["identificador"],
                    "categoria": l.get("categoria", ""),
                    "alerta": a,
                })
    return criticas


def procesar_cliente(cfg_wa: dict, cliente: dict, admin_phone: str,
                       log: logging.Logger, dry_run: bool = False) -> tuple:
    """Detecta alertas críticas nuevas y manda WhatsApp.

    Returns: (n_enviados_admin, n_enviados_cliente, n_errores)
    """
    nombre = cliente.get("nombre", "")
    criticas = detectar_alertas_criticas_cliente(cliente, log)
    if not criticas:
        return 0, 0, 0

    enviados_admin = 0
    enviados_cli = 0
    errores = 0

    # Multi-contacto: iterar TODOS los destinatarios del cliente
    # (productor + encargado + comedero, etc.)
    destinatarios = [
        d for d in db.listar_destinatarios(cliente)
        if d.get("whatsapp") and d.get("alertas_whatsapp_activas", 1)
    ]

    for c in criticas:
        lote = c["lote"]
        alerta = c["alerta"]
        clave = wa.clave_dedup(
            cliente["id"], lote, "critica", alerta.get("titulo", ""),
        )

        # 1) Admin (con su propia clave de dedup, ventana 12hs)
        if admin_phone:
            clave_admin = f"admin:{clave}"
            # Ventana 24hs: si la alerta crítica persiste varios días,
            # mandamos máximo 1 vez por día (no cada hora).
            if db.whatsapp_ya_enviado(clave_admin, ventana_horas=24):
                log.info(f"  [{nombre}] admin skip duplicado: "
                          f"{alerta.get('titulo')}")
            else:
                mensaje_admin = (
                    f"⛔ *ALERTA CRÍTICA*\n"
                    f"Cliente: {nombre}\n"
                    f"Lote: {lote}\n"
                    f"Situación: {alerta.get('titulo')}\n\n"
                    f"{alerta.get('descripcion', '')[:200]}"
                )
                if dry_run:
                    log.info(f"  [DRY] -> admin {admin_phone}: "
                              f"{alerta.get('titulo')}")
                    enviados_admin += 1
                else:
                    ok, msg = wa.enviar_texto(cfg_wa, admin_phone, mensaje_admin)
                    if not ok and "ventana 24hs" in msg.lower():
                        ok, msg = wa.enviar_alerta_critica(
                            cfg_wa, admin_phone, nombre, lote,
                            alerta.get("titulo", ""),
                            alerta.get("accion", ""),
                        )
                    if ok:
                        log.info(f"  ✓ admin -> {nombre} | "
                                  f"{alerta.get('titulo')}")
                        db.registrar_whatsapp_enviado(
                            cliente["id"], admin_phone, clave_admin,
                            mensaje_admin, "enviada",
                        )
                        enviados_admin += 1
                    else:
                        log.error(f"  ✗ admin: {msg}")
                        db.registrar_whatsapp_enviado(
                            cliente["id"], admin_phone, clave_admin,
                            mensaje_admin, "error", msg,
                        )
                        errores += 1

        # 2) TODOS los contactos del cliente (productor + encargado + comedero)
        for d in destinatarios:
            phone_d = d["whatsapp"]
            nom_d = d.get("nombre", "") or "destinatario"
            clave_dest = f"cli:{phone_d}:{clave}"

            # Ventana 24hs: máximo 1 alerta por día por destinatario
            # por la misma alerta climática.
            if db.whatsapp_ya_enviado(clave_dest, ventana_horas=24):
                continue

            # Si no recibió bienvenida todavía, mandar primero la
            # bienvenida para que entienda qué es esto.
            if not d.get("bienvenida_whatsapp_enviada", 0) and not dry_run:
                try:
                    wa_bienv = wa.componer_bienvenida(cliente, d)
                    ok_b, _ = wa.enviar_texto(cfg_wa, phone_d, wa_bienv)
                    if ok_b:
                        log.info(f"  ✓ bienvenida wa -> {phone_d} ({nom_d})")
                        db.marcar_bienvenida_enviada(
                            d["origen"], d["id"], "whatsapp",
                        )
                except Exception as e:
                    log.warning(f"  ⚠ Error bienvenida wa a {phone_d}: {e}")

            if dry_run:
                log.info(f"  [DRY] -> {phone_d} ({nom_d}): {alerta.get('titulo')}")
                enviados_cli += 1
                continue

            ok, msg = wa.enviar_alerta_critica(
                cfg_wa, phone_d, nombre, lote,
                alerta.get("titulo", ""), alerta.get("accion", ""),
            )
            if ok:
                log.info(f"  ✓ {nom_d} {phone_d} | {alerta.get('titulo')}")
                db.registrar_whatsapp_enviado(
                    cliente["id"], phone_d, clave_dest,
                    alerta.get("titulo", ""), "enviada",
                )
                enviados_cli += 1
            else:
                log.error(f"  ✗ {nom_d} {phone_d}: {msg}")
                db.registrar_whatsapp_enviado(
                    cliente["id"], phone_d, clave_dest,
                    alerta.get("titulo", ""), "error", msg,
                )
                errores += 1

    return enviados_admin, enviados_cli, errores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="No envía WhatsApp, solo muestra qué haría")
    parser.add_argument("--solo-cliente", default=None)
    args = parser.parse_args()

    log = setup_logging()

    # Lock global: evitar dos instancias en paralelo (launchd al despertar
    # la Mac puede disparar el job atrasado + el StartCalendarInterval).
    lock_fd = adquirir_lock_proceso("alertas_criticas")
    if lock_fd is None:
        log.info("=== abortado: otra instancia ya corre ===")
        return 0

    log.info("=== ALERTAS CRÍTICAS WHATSAPP ===")

    db.init_db()

    cfg_wa = wa.cargar_config()
    ok, err = wa.config_valida(cfg_wa)
    if not ok and not args.dry_run:
        log.error(f"Config WhatsApp inválida: {err}")
        log.error("Configurá WhatsApp en Configuración → WhatsApp.")
        return 1

    admin_phone = (cfg_wa or {}).get("admin_phone", "")
    if admin_phone:
        admin_phone = wa.normalizar_telefono(admin_phone) or admin_phone

    clientes = db.listar_clientes()
    if args.solo_cliente:
        clientes = [c for c in clientes
                    if args.solo_cliente.lower() in c["nombre"].lower()]
    log.info(f"Clientes a evaluar: {len(clientes)}")

    total_admin = 0
    total_cli = 0
    total_err = 0
    for c in clientes:
        try:
            ea, ec, ee = procesar_cliente(
                cfg_wa or {}, c, admin_phone, log, dry_run=args.dry_run,
            )
            total_admin += ea
            total_cli += ec
            total_err += ee
        except Exception as e:
            log.exception(f"Error en {c.get('nombre')}: {e}")
            total_err += 1

    log.info(f"=== FIN — admin: {total_admin}, clientes: {total_cli}, "
              f"errores: {total_err} ===")
    liberar_lock(lock_fd)
    return 0 if total_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
