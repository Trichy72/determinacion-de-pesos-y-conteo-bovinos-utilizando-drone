#!/usr/bin/env python3
"""
Cron de mitad de semana — miércoles 7:30 AM.

Compara el pronóstico actual con el snapshot que se guardó el lunes.
Si hay cambios significativos en los próximos días, manda un email
"update" al cliente. Si no hay cambios, NO manda nada (silencio total).

Cambios que disparan envío:
  🆕 Nuevo: día que ahora tiene alerta y antes no.
  ⬆️ Empeoró: severidad subió (normal→warning, warning→critica, etc.)
  ⬇️ Mejoró: severidad bajó (warning→normal, etc.) — buena noticia.

Uso:
    python scripts/alertas_semanales_update.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import database as db
from src.locking import adquirir_lock_proceso, liberar_lock
from src import alertas_email as ae
from src.clima import (
    obtener_clima, generar_alertas_predictivas,
    geocodificar_manual, geocodificar,
)
from scripts.alertas_semanales import _snapshot_pronostico


def setup_logging() -> logging.Logger:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"semanal-update_{datetime.now().strftime('%Y-%m-%d')}.log"
    logger = logging.getLogger("alertas_semanales_update")
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


_RANK = {"normal": 0, "preventiva": 1, "info": 1,
           "warning": 2, "critica": 3}

_DIA_NOMBRE = ["lunes", "martes", "miércoles", "jueves",
                "viernes", "sábado", "domingo"]


def _formato_dia(fecha_iso: str) -> str:
    """'2026-05-14' → 'jueves 14/05'."""
    try:
        d = datetime.strptime(fecha_iso, "%Y-%m-%d").date()
    except ValueError:
        return fecha_iso
    return f"{_DIA_NOMBRE[d.weekday()]} {d.strftime('%d/%m')}"


def _icono_sev(sev: str) -> str:
    return {"critica": "🔴", "warning": "🟠",
            "preventiva": "🟡", "info": "🟡",
            "normal": "🟢"}.get(sev or "normal", "🟢")


def detectar_cambios(snapshot_lunes: list, snapshot_hoy: list,
                       hoy_iso: str) -> dict:
    """Compara los dos snapshots y devuelve {nuevos, empeoraron, mejoraron}.

    Solo considera fechas >= hoy (los días pasados no importan).
    """
    cambios = {"nuevos": [], "empeoraron": [], "mejoraron": []}

    # Indexar por fecha
    lunes_by_fecha = {it["fecha"]: it for it in (snapshot_lunes or [])}
    hoy_by_fecha = {it["fecha"]: it for it in (snapshot_hoy or [])}

    todas_fechas = sorted(set(lunes_by_fecha) | set(hoy_by_fecha))
    for fecha in todas_fechas:
        if fecha < hoy_iso:
            continue
        sev_l = (lunes_by_fecha.get(fecha) or {}).get("severidad", "normal")
        sev_h = (hoy_by_fecha.get(fecha) or {}).get("severidad", "normal")
        tipo_h = (hoy_by_fecha.get(fecha) or {}).get("tipo")
        rank_l = _RANK.get(sev_l, 0)
        rank_h = _RANK.get(sev_h, 0)

        if rank_h > rank_l and rank_l == 0:
            # Antes era normal, ahora hay alerta → NUEVO
            cambios["nuevos"].append({
                "fecha": fecha, "severidad": sev_h, "tipo": tipo_h,
            })
        elif rank_h > rank_l:
            # Empeoró
            cambios["empeoraron"].append({
                "fecha": fecha, "antes": sev_l, "ahora": sev_h, "tipo": tipo_h,
            })
        elif rank_h < rank_l:
            # Mejoró
            cambios["mejoraron"].append({
                "fecha": fecha, "antes": sev_l, "ahora": sev_h,
            })
    return cambios


def hay_cambios_significativos(cambios: dict) -> bool:
    """¿Vale la pena mandar email? Solo si hay al menos 1 cambio."""
    return bool(cambios["nuevos"]
                or cambios["empeoraron"]
                or cambios["mejoraron"])


def _cambios_son_significativos(cambios: dict) -> bool:
    """¿Vale la pena invocar al LLM para explicar el cambio?
    Criterio: ≥3 cambios totales, o aparece un nuevo crítico, o un día
    empeoró hasta crítico. Cambios menores quedan con la plantilla simple."""
    total = (len(cambios.get("nuevos", []))
             + len(cambios.get("empeoraron", []))
             + len(cambios.get("mejoraron", [])))
    if total >= 3:
        return True
    if any(c.get("severidad") == "critica"
           for c in cambios.get("nuevos", [])):
        return True
    if any(c.get("ahora") == "critica"
           for c in cambios.get("empeoraron", [])):
        return True
    return False


def componer_email_update(cliente: dict, cambios: dict,
                            snapshot_hoy: list,
                            snapshot_lunes: list = None,
                            lotes: list = None,
                            clima: dict = None) -> tuple:
    """Compone (subject, html, text) del email de update."""
    nombre = cliente.get("nombre", "")
    establ = (cliente.get("establecimiento", "")
                or cliente.get("localidad", ""))
    fecha_hoy = datetime.now().strftime("%d/%m/%Y")

    n_nuevos = len(cambios["nuevos"])
    n_peor = len(cambios["empeoraron"])
    n_mejor = len(cambios["mejoraron"])

    # Análisis LLM personalizado solo si los cambios son significativos.
    # Si falla la llamada, queda en cadena vacía (el email sale igual).
    analisis_html = ""
    if _cambios_son_significativos(cambios):
        try:
            from src.ai_analisis_semanal import (
                generar_analisis_update_llm, texto_a_html_parrafos,
            )
            # Si tenemos el clima fresco, calculamos el impacto del
            # peor evento que arroja la semana actualizada y se lo
            # pasamos al LLM como dato a citar tal cual.
            impacto_upd_txt = None
            if clima:
                try:
                    from src.impacto_productivo import (
                        estimar_impacto_peor_dia_semanal as _imp_sem_upd,
                        formato_impacto_texto as _fmt_imp_upd,
                    )
                    _imp_upd = _imp_sem_upd(clima, lotes)
                    if _imp_upd:
                        impacto_upd_txt = _fmt_imp_upd(_imp_upd)
                except Exception:
                    impacto_upd_txt = None
            texto = generar_analisis_update_llm(
                cliente=cliente, cambios=cambios,
                snapshot_hoy=snapshot_hoy,
                snapshot_lunes=snapshot_lunes,
                lotes=lotes,
                impacto_productivo_txt=impacto_upd_txt,
            )
            if texto:
                cuerpo = texto_a_html_parrafos(texto)
                analisis_html = f"""
                <div style="margin-top:14px; padding:12px 14px;
                  background:#FAF9F2; border-left:3px solid {ae.COLOR_VERDE};
                  border-radius:4px; font-size:13px; color:#333;">
                  <strong style="color:#1B3E27;">📖 Qué implica este cambio
                  </strong>
                  <p style="margin:4px 0 0; font-size:11px; color:#888;
                    font-style:italic;">
                    Análisis personalizado generado para tu rodeo.
                  </p>
                  {cuerpo}
                </div>"""
        except Exception:
            analisis_html = ""

    # Subject según el tipo de cambio dominante
    if n_peor or n_nuevos:
        if any(c["severidad"] == "critica" for c in cambios["nuevos"]) or \
                any(c["ahora"] == "critica" for c in cambios["empeoraron"]):
            subject = (f"⚠️ Update pronóstico — empeoró: {establ or nombre}")
        else:
            subject = (f"📅 Update pronóstico — {establ or nombre}")
    else:
        subject = (f"✅ Update pronóstico — mejoró: {establ or nombre}")

    bloques_html = []

    if cambios["nuevos"]:
        items_html = "".join(
            f"<li>{_icono_sev(c['severidad'])} <strong>{_formato_dia(c['fecha'])}</strong>: "
            f"alerta {'crítica' if c['severidad']=='critica' else 'moderada'}"
            f"{' de ' + c['tipo'] if c['tipo'] else ''}</li>"
            for c in cambios["nuevos"]
        )
        bloques_html.append(
            f"<h3 style='color:#C0392B; margin-top:14px;'>"
            f"🆕 Días que el lunes no tenían alerta:</h3>"
            f"<ul style='padding-left:18px;'>{items_html}</ul>"
        )

    if cambios["empeoraron"]:
        items_html = "".join(
            f"<li>{_icono_sev(c['ahora'])} <strong>{_formato_dia(c['fecha'])}</strong>: "
            f"pasó de {c['antes']} → <strong>{c['ahora']}</strong>"
            f"{' (' + c['tipo'] + ')' if c['tipo'] else ''}</li>"
            for c in cambios["empeoraron"]
        )
        bloques_html.append(
            f"<h3 style='color:#E67E22; margin-top:14px;'>"
            f"⬆️ Días que empeoraron:</h3>"
            f"<ul style='padding-left:18px;'>{items_html}</ul>"
        )

    if cambios["mejoraron"]:
        items_html = "".join(
            f"<li>{_icono_sev(c['ahora'])} <strong>{_formato_dia(c['fecha'])}</strong>: "
            f"pasó de {c['antes']} → <strong>{c['ahora']}</strong></li>"
            for c in cambios["mejoraron"]
        )
        bloques_html.append(
            f"<h3 style='color:#1B3E27; margin-top:14px;'>"
            f"✅ Días que mejoraron:</h3>"
            f"<ul style='padding-left:18px;'>{items_html}</ul>"
        )

    html = f"""<!DOCTYPE html>
<html><body style="margin:0; padding:0; background:#F4F4F4;
  font-family:Arial,sans-serif;">
  <div style="max-width:680px; margin:0 auto; background:white;">
    <div style="background:{ae.COLOR_VERDE}; padding:18px 24px;
      color:white;">
      <table width="100%"><tr>
        <td><img src="cid:hms-logo" height="44"></td>
        <td style="text-align:right;">
          <div style="font-size:13px; opacity:0.85;">{fecha_hoy}</div>
          <div style="font-size:18px; font-weight:600;">
            🔄 Update del pronóstico semanal
          </div>
        </td>
      </tr></table>
    </div>
    <div style="padding:24px; color:#333; line-height:1.55;">
      <p style="font-size:15px;">Hola {nombre},</p>
      <p>Hoy es miércoles. El pronóstico que te mandamos el lunes
      <strong>cambió</strong> respecto a lo que se preveía:</p>

      {''.join(bloques_html)}

      {analisis_html}

      <p style="margin-top:18px; color:#666; font-size:13px;">
        💡 Este update solo se manda cuando hay cambios reales
        respecto al lunes. Las alertas puntuales del día siguen
        llegando como siempre.
      </p>

      <p style="margin-top:18px; color:#888; font-size:12px;">
        — Mauricio Suárez<br>HMS Nutrición Animal
      </p>
    </div>
    <div style="background:{ae.COLOR_VERDE}; padding:12px 24px;
      color:white; font-size:11px; text-align:center;">
      <strong>HMS Nutrición Animal</strong> — alerta automática<br>
      Para darse de baja, respondé este email con la palabra BAJA.
    </div>
  </div>
</body></html>"""

    text_lines = [
        f"Update del pronóstico — {fecha_hoy}",
        "",
        f"Hola {nombre},",
        "",
        "El pronóstico que te mandamos el lunes cambió:",
        "",
    ]
    if cambios["nuevos"]:
        text_lines.append("NUEVOS días con alerta:")
        for c in cambios["nuevos"]:
            text_lines.append(
                f"  - {_formato_dia(c['fecha'])}: {c['severidad']}"
                f"{' (' + c['tipo'] + ')' if c['tipo'] else ''}"
            )
        text_lines.append("")
    if cambios["empeoraron"]:
        text_lines.append("Días que EMPEORARON:")
        for c in cambios["empeoraron"]:
            text_lines.append(
                f"  - {_formato_dia(c['fecha'])}: {c['antes']} → {c['ahora']}"
            )
        text_lines.append("")
    if cambios["mejoraron"]:
        text_lines.append("Días que MEJORARON:")
        for c in cambios["mejoraron"]:
            text_lines.append(
                f"  - {_formato_dia(c['fecha'])}: {c['antes']} → {c['ahora']}"
            )
        text_lines.append("")

    text_lines.append(
        "— Mauricio Suárez — HMS Nutrición Animal\n\n"
        "Para darse de baja, respondé este email con la palabra BAJA."
    )
    return subject, html, "\n".join(text_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--solo-cliente", default=None)
    args = parser.parse_args()

    log = setup_logging()

    # Lock global: evitar dos instancias en paralelo (launchd al despertar
    # la Mac puede disparar el job atrasado + el StartCalendarInterval).
    lock_fd = adquirir_lock_proceso("alertas_semanales_update")
    if lock_fd is None:
        log.info("=== abortado: otra instancia ya corre ===")
        return 0

    log.info("=== UPDATE SEMANAL (miércoles) — INICIO ===")

    db.init_db()

    cfg = ae.cargar_config_smtp()
    ok, err = ae.config_valida(cfg)
    if not ok and not args.dry_run:
        log.error(f"Config SMTP inválida: {err}")
        return 1

    hoy = datetime.now().date()
    hoy_iso = hoy.isoformat()
    lunes = hoy - timedelta(days=hoy.weekday())
    fecha_db = hoy_iso

    clientes = db.listar_clientes()
    if args.solo_cliente:
        clientes = [c for c in clientes
                    if args.solo_cliente.lower() in c["nombre"].lower()]
    log.info(f"Clientes a procesar: {len(clientes)}")
    log.info(f"Lunes de referencia: {lunes.isoformat()}")

    enviados = 0
    errores = 0
    n_con_cambios = 0

    for c in clientes:
        log.info(f"Cliente: {c['nombre']}")
        try:
            snap_lunes = db.obtener_snapshot_pronostico(
                c["id"], lunes.isoformat(),
            )
            if snap_lunes is None:
                log.info(f"  {c['nombre']}: no hay snapshot del lunes "
                          f"({lunes.isoformat()}). Skip.")
                continue

            # Recalcular pronóstico hoy
            lat = c.get("lat")
            lon = c.get("lon")
            localidad = c.get("localidad", "")
            if lat and lon:
                geo = geocodificar_manual(float(lat), float(lon), localidad)
            elif localidad:
                geo = geocodificar(localidad)
            else:
                continue
            if not geo:
                continue
            clima = obtener_clima(geo["lat"], geo["lon"])
            if not clima:
                log.warning(f"  {c['nombre']}: Open-Meteo no respondió.")
                continue

            lotes = db.listar_lotes(cliente_id=c["id"], estado="activo")
            alertas_por_lote = []
            for l in lotes:
                alertas = generar_alertas_predictivas(
                    clima,
                    categoria=l.get("categoria", ""),
                    raza=l.get("raza", ""),
                )
                alertas_por_lote.append({
                    "lote": l["identificador"],
                    "categoria": l.get("categoria", ""),
                    "alertas": alertas,
                })

            snap_hoy = _snapshot_pronostico(alertas_por_lote, clima)
            cambios = detectar_cambios(snap_lunes, snap_hoy, hoy_iso)

            if not hay_cambios_significativos(cambios):
                log.info(f"  {c['nombre']}: pronóstico sin cambios. Silencio.")
                continue

            n_con_cambios += 1
            log.info(f"  {c['nombre']}: cambios detectados — "
                      f"nuevos={len(cambios['nuevos'])}, "
                      f"empeor={len(cambios['empeoraron'])}, "
                      f"mejor={len(cambios['mejoraron'])}")

            # Mandar a TODOS los destinatarios
            destinatarios = db.listar_destinatarios(c)
            subject, html, text = componer_email_update(
                c, cambios, snap_hoy,
                snapshot_lunes=snap_lunes, lotes=lotes,
                clima=clima,
            )

            for d in destinatarios:
                email = (d.get("email") or "").strip()
                if not email or not d.get("alertas_email_activas", 1):
                    continue

                if db.alerta_ya_enviada_hoy(c["id"], email, fecha_db,
                                              tipo="semanal_update"):
                    continue

                if args.dry_run:
                    log.info(f"  [DRY-RUN] update -> {email}")
                    enviados += 1
                    continue

                ok2, msg = ae.enviar_email(cfg, [email], subject, html, text)
                if ok2:
                    log.info(f"  ✓ update -> {email}")
                    db.registrar_alerta_enviada(
                        fecha_db, c["id"], email, subject, 0,
                        "enviada", "", tipo="semanal_update",
                    )
                    enviados += 1
                else:
                    log.error(f"  ✗ update -> {email}: {msg}")
                    errores += 1
        except Exception as e:
            log.exception(f"  Error procesando {c['nombre']}: {e}")
            errores += 1

    log.info(
        f"=== FIN — clientes con cambios: {n_con_cambios}, "
        f"enviados: {enviados}, errores: {errores} ==="
    )
    if n_con_cambios == 0:
        log.info("  → Pronóstico sin cambios para nadie. No se mandó nada.")
    liberar_lock(lock_fd)
    return 0 if errores == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
