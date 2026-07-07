#!/usr/bin/env python3
"""Cron CADA 15 MINUTOS — pide a cada encargado de lote, por WhatsApp,
cuánto cargó al comedero, en los horarios que el lote tiene
configurados.

Lógica de envío por lote (comedero lineal):
  - cant_comidas_diarias = 1 o 2.
  - Para cada comida N (1, 2):
      hora_objetivo = lote.hora_comida_N (HH:MM)
      - Intento 1: ventana [hora_objetivo - 10 min, hora_objetivo + 10 min]
      - Intento 2 (recordatorio): ventana
         [hora_objetivo + 50 min, hora_objetivo + 70 min],
         SOLO si todavía no hay carga registrada hoy para esa comida.
  - Dedup: tabla pedidos_carga_enviados con UNIQUE
    (lote_id, fecha, comida_n, intento_n).

Para silocomedero la lógica de frecuencia se mantiene: solo manda el
día que toca recargar (última silo_carga >= dias_cubiertos - 1 atrás).
En silocomedero la pregunta usa hora_comida_1 únicamente.

Configuración (en data/whatsapp_config.json):
  - Credenciales Twilio + 'carga_base_url' (URL pública del túnel).

Uso:
    python3 scripts/whatsapp_pedido_carga.py
    python3 scripts/whatsapp_pedido_carga.py --dry-run
    python3 scripts/whatsapp_pedido_carga.py --solo-lote 7
    python3 scripts/whatsapp_pedido_carga.py --force
        (ignora dedup y ventana de hora — solo para tests manuales)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import database as db  # noqa: E402
from src import whatsapp as wa  # noqa: E402
from src import carga_diaria_token as tok  # noqa: E402
from src import stock_producto as sp  # noqa: E402
from src.locking import adquirir_lock_proceso, liberar_lock  # noqa: E402


# Tolerancia: ±10 min sobre cada "punto de envío".
TOLERANCIA_MIN = 10
# Cuándo es el recordatorio respecto del horario inicial (en min).
RECORDATORIO_MIN = 60
# Cuántos minutos alrededor del horario objetivo cuenta como "ya
# cargó" — si hay una carga registrada con hora dentro de esa
# ventana, no insistir con esa comida.
VENTANA_YA_CARGO_MIN = 90


def setup_logging() -> logging.Logger:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        log_dir
        / f"pedido_carga_{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    logger = logging.getLogger("pedido_carga_whatsapp")
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


def _parse_hora(s: str, default: str) -> time:
    try:
        s = (s or default).strip()
        hh, mm = s.split(":")[:2]
        return time(int(hh), int(mm))
    except Exception:
        hh, mm = default.split(":")
        return time(int(hh), int(mm))


def _minutos_entre(t1: time, t2: time) -> int:
    """Diferencia t1 - t2 en minutos (puede ser negativa)."""
    return (t1.hour * 60 + t1.minute) - (t2.hour * 60 + t2.minute)


def _componer_mensaje(
    lote: dict, cli: dict, url: str, mezcla_recom_kg: float,
    comida_n: int, hora_obj_str: str, intento_n: int,
) -> str:
    """Mensaje de WhatsApp al encargado."""
    encargado = (lote.get("encargado_nombre") or "").strip()
    saludo = f"Hola {encargado}" if encargado else "Hola"

    if intento_n >= 2:
        intro = (
            f"{saludo}, te recuerdo cargar la dieta del lote "
            f"*{lote.get('identificador', '?')}* "
            f"({cli.get('nombre', '—')}) — todavía no recibí los kg "
            f"de la comida de las {hora_obj_str}."
        )
    else:
        if comida_n == 2:
            intro = (
                f"{saludo}, te paso el link para la **segunda comida** "
                f"({hora_obj_str}) del lote "
                f"*{lote.get('identificador', '?')}* "
                f"({cli.get('nombre', '—')})."
            )
        else:
            intro = (
                f"{saludo}, te paso el link para cargar la dieta de "
                f"hoy del lote *{lote.get('identificador', '?')}* "
                f"({cli.get('nombre', '—')})."
            )

    return (
        f"{intro}\n\n"
        f"📋 Recomendado total del día: ~{mezcla_recom_kg:.0f} kg "
        f"de mezcla.\n\n"
        f"👉 {url}\n\n"
        "El link tiene un form pre-cargado con los ingredientes — "
        "solo ajustá los kg si fue distinto y dale Enviar.\n\n"
        "Gracias!\n— HMS Nutrición Animal"
    )


def _ya_cargo_comida(
    lote_id: int, fecha_iso: str, hora_obj: time,
) -> bool:
    """¿Hay alguna carga del día con hora cercana a hora_obj
    (±VENTANA_YA_CARGO_MIN)? Si sí, asumimos que ya cargó esa comida."""
    try:
        cargas = db.listar_cargas_silocomedero(lote_id, limit=30)
    except Exception:
        return False
    for c in cargas:
        if (c.get("fecha_carga") or "")[:10] != fecha_iso:
            continue
        h_str = c.get("hora_carga")
        if not h_str:
            # Carga vieja sin hora — la consideramos cubriendo TODO
            # el día (para no insistir).
            return True
        try:
            hh, mm = h_str.split(":")[:2]
            h = time(int(hh), int(mm))
        except Exception:
            continue
        if abs(_minutos_entre(h, hora_obj)) <= VENTANA_YA_CARGO_MIN:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pide carga del día por WhatsApp en horarios configurados."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="No envía nada, solo loguea lo que mandaría.",
    )
    parser.add_argument(
        "--solo-lote", type=int, default=None,
        help="Procesar solo el lote con este id.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Ignora ventana de hora y dedup. Útil para tests manuales — "
            "manda inmediatamente."
        ),
    )
    args = parser.parse_args()

    log = setup_logging()
    log.info(
        f"=== PEDIDO CARGA WHATSAPP "
        f"{datetime.now().isoformat(timespec='seconds')} ==="
    )

    lock_fd = adquirir_lock_proceso("pedido_carga_whatsapp")
    if not lock_fd:
        log.warning("Otro proceso corriendo. Salgo para no duplicar.")
        return 0

    try:
        cfg_wa = wa.cargar_config() or {}
        ok, err = wa.config_valida(cfg_wa)
        if not ok:
            log.error(f"Config WhatsApp inválida: {err}")
            return 1

        base_url = (
            cfg_wa.get("carga_base_url")
            or cfg_wa.get("base_url")
            or ""
        ).strip()
        if not base_url:
            log.error(
                "Falta 'carga_base_url' en whatsapp_config.json."
            )
            return 1

        if args.solo_lote:
            lt0 = db.obtener_lote(args.solo_lote)
            lotes = [lt0] if lt0 else []
        else:
            lotes = db.listar_lotes(estado="activo")
        log.info(f"Lotes activos a evaluar: {len(lotes)}")

        clientes = {c["id"]: c for c in db.listar_clientes()}

        ahora = datetime.now()
        hoy = ahora.date()
        hoy_iso = hoy.strftime("%Y-%m-%d")
        ahora_t = ahora.time().replace(second=0, microsecond=0)

        enviados = 0
        skip = 0
        errores = 0

        for lt in lotes:
            lid = lt["id"]
            ident = lt.get("identificador", "?")

            wa_enc = (lt.get("encargado_whatsapp") or "").strip()
            if not wa_enc:
                continue
            if not int(lt.get("carga_diaria_activa") or 0):
                continue

            dietas = db.listar_dietas(lid)
            dieta_v = (
                sp._dieta_vigente(dietas, hoy_iso) if dietas else None
            )
            if not dieta_v:
                skip += 1
                log.info(
                    f"  [skip] lote {lid} ({ident}): sin dieta vigente."
                )
                continue

            tipo_com = (
                lt.get("tipo_comedero_concentrado") or ""
            ).lower()

            # Silocomedero: solo el día que toca recargar
            if tipo_com == "silocomedero":
                cargas_prev = db.listar_cargas_silocomedero(
                    lid, limit=10
                )
                ultima_silo = None
                for cp in cargas_prev:
                    if (
                        (cp.get("tipo_carga") or "silo_carga")
                        == "silo_carga"
                    ):
                        ultima_silo = cp
                        break
                if ultima_silo:
                    try:
                        f_ult = datetime.strptime(
                            (ultima_silo.get("fecha_carga") or "")
                            [:10],
                            "%Y-%m-%d",
                        ).date()
                    except Exception:
                        f_ult = None
                    dias_cub = float(
                        ultima_silo.get("dias_cubiertos") or 5
                    )
                    if f_ult:
                        dias_pasados = (hoy - f_ult).days
                        if (
                            not args.force
                            and dias_pasados < (dias_cub - 1)
                        ):
                            skip += 1
                            log.info(
                                f"  [skip] lote {lid} ({ident}): "
                                f"silo cargado hace {dias_pasados}d "
                                f"de {dias_cub:.0f}d — todavía no toca."
                            )
                            continue
                # Silocomedero usa solo hora_comida_1
                comidas_a_evaluar = [1]
            else:
                cant_com = int(lt.get("cant_comidas_diarias") or 1)
                cant_com = 1 if cant_com not in (1, 2) else cant_com
                comidas_a_evaluar = (
                    [1] if cant_com == 1 else [1, 2]
                )

            # Calcular mezcla total recomendada (para mostrar en msg)
            cant_animales = (
                db.cantidad_vigente_lote(lid, hoy_iso) or 0
            )
            mezcla_total = 0.0
            for cing in dieta_v.get("composicion") or []:
                nm = (cing.get("nombre") or "").strip()
                if not nm or sp._es_a_discrecion(nm):
                    continue
                mezcla_total += (
                    float(cing.get("kg_tal_cual") or 0) * cant_animales
                )

            for comida_n in comidas_a_evaluar:
                # Resolver hora objetivo de esa comida
                if comida_n == 1:
                    hora_obj = _parse_hora(
                        lt.get("hora_comida_1"), "08:30",
                    )
                else:
                    hora_obj = _parse_hora(
                        lt.get("hora_comida_2"), "16:00",
                    )
                hora_obj_str = hora_obj.strftime("%H:%M")

                # ¿En qué intento estamos según la hora actual?
                dt_obj = ahora.replace(
                    hour=hora_obj.hour, minute=hora_obj.minute,
                    second=0, microsecond=0,
                )
                dt_obj_rec = dt_obj + timedelta(
                    minutes=RECORDATORIO_MIN
                )
                en_ventana_1 = (
                    abs((ahora - dt_obj).total_seconds() / 60)
                    <= TOLERANCIA_MIN
                )
                en_ventana_2 = (
                    abs((ahora - dt_obj_rec).total_seconds() / 60)
                    <= TOLERANCIA_MIN
                )

                if args.force:
                    intento_n = 1
                elif en_ventana_1:
                    intento_n = 1
                elif en_ventana_2:
                    # Solo mandamos recordatorio si todavía no cargó.
                    if _ya_cargo_comida(lid, hoy_iso, hora_obj):
                        log.info(
                            f"  [skip] lote {lid} ({ident}) "
                            f"comida {comida_n} @ {hora_obj_str}: "
                            "ya hay carga del día, no recuerdo."
                        )
                        skip += 1
                        continue
                    intento_n = 2
                else:
                    # No estamos en ninguna ventana — skip silencioso.
                    continue

                # Dedup
                if not args.force and db.pedido_carga_ya_enviado(
                    lid, hoy_iso, comida_n, intento_n,
                ):
                    log.info(
                        f"  [skip] lote {lid} ({ident}) "
                        f"comida {comida_n} intento {intento_n}: "
                        "ya enviado."
                    )
                    skip += 1
                    continue

                # URL firmada
                url = tok.url_carga_diaria(base_url, lid, hoy)
                cli = clientes.get(lt.get("cliente_id")) or {}
                mensaje = _componer_mensaje(
                    lt, cli, url, mezcla_total,
                    comida_n, hora_obj_str, intento_n,
                )

                if args.dry_run:
                    log.info(
                        f"  [DRY] lote {lid} ({ident}) "
                        f"comida {comida_n} intento {intento_n} "
                        f"→ {wa_enc} (hora_obj {hora_obj_str})\n"
                        f"        {mensaje[:160]}..."
                    )
                    continue

                ok_env, resp = wa.enviar_texto(
                    cfg_wa, wa_enc, mensaje,
                )
                if ok_env:
                    enviados += 1
                    db.registrar_pedido_carga_enviado(
                        lid, hoy_iso, comida_n, intento_n,
                    )
                    log.info(
                        f"  ✓ Enviado lote {lid} ({ident}) "
                        f"comida {comida_n} intento {intento_n} "
                        f"→ {wa_enc} | {resp}"
                    )
                else:
                    errores += 1
                    log.warning(
                        f"  ⚠ Falló lote {lid} ({ident}) "
                        f"comida {comida_n}: {resp}"
                    )

        log.info(
            f"=== FIN — Enviados {enviados}, skip {skip}, "
            f"errores {errores} ==="
        )
        return 0 if errores == 0 else 2

    finally:
        liberar_lock(lock_fd)


if __name__ == "__main__":
    sys.exit(main())
