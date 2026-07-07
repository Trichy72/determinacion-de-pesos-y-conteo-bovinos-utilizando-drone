#!/usr/bin/env python3
"""
Cron semanal — lunes 7:30 AM.

Manda a cada cliente un email-resumen del PRONÓSTICO de la semana:
- Clima previsto día por día
- Alertas anticipadas (frío, calor, frente, lluvia)
- Acciones generales recomendadas para la semana
- Período sugerido para manejos (vacunas, traslados, pesadas)

Filosofía:
- 1 email por semana, lunes temprano
- Ayuda al cliente a planificar la semana antes de que arranque
- Se manda incluso si no hay alertas críticas (es un pronóstico
  semanal, no una alarma puntual)

Uso:
    python scripts/alertas_semanales.py [--dry-run] [--solo-cliente NOMBRE]
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
    obtener_clima, generar_alertas_predictivas, calcular_thi, clasificar_thi,
    geocodificar_manual, geocodificar, clasificar_nivel_productivo,
)


# =====================================================================
# LOGGING
# =====================================================================

def setup_logging() -> logging.Logger:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"semanal_{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger("alertas_semanales")
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
# COMPOSICIÓN DEL EMAIL SEMANAL
# =====================================================================

_DIA_SEMANA = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def _snapshot_pronostico(alertas_por_lote: list, clima: dict) -> list:
    """Construye el snapshot del pronóstico semanal: lista de días con
    severidad máxima, tipo predominante y motivo principal.

    Cada item: {fecha, severidad, tipo, motivo, nivel_productivo}
      severidad: "normal" | "preventiva" | "warning" | "critica"
      tipo: "calor" | "frio" | None
      motivo: string corto del motivo principal del día
      nivel_productivo: "normal" | "atencion" | "operativo" | "critico"
    """
    daily = clima.get("daily", {}) or {}
    fechas = daily.get("time", [])
    precip = daily.get("precipitation_sum", [])
    t_min = daily.get("temperature_2m_min", [])
    hum_max = daily.get("relative_humidity_2m_max", [])
    hoy = datetime.now().date()

    # Calcular horas con HR ≥85% por día desde los datos horarios.
    # Esto distingue picos de madrugada (1-3h, transitorios) de humedad
    # sostenida (8+ horas, que realmente moja el pelaje y altera el
    # consumo).
    hourly = clima.get("hourly", {}) or {}
    hr_times = hourly.get("time", []) or []
    hr_hum = hourly.get("relative_humidity_2m", []) or []
    horas_hr_alta_por_fecha = {}
    for idx_h, ts in enumerate(hr_times):
        if idx_h >= len(hr_hum):
            break
        try:
            fecha_h = ts[:10]
        except (TypeError, IndexError):
            continue
        valor = hr_hum[idx_h]
        if valor is not None and valor >= 85:
            horas_hr_alta_por_fecha[fecha_h] = (
                horas_hr_alta_por_fecha.get(fecha_h, 0) + 1
            )

    snapshot = []
    rank = {"normal": 0, "preventiva": 1, "info": 1,
             "warning": 2, "critica": 3}

    for i, fstr in enumerate(fechas):
        try:
            f = datetime.strptime(fstr, "%Y-%m-%d").date()
        except ValueError:
            continue
        if f < hoy:
            continue
        if (f - hoy).days > 7:
            break

        # Buscar peor severidad para ese día y el motivo (titulo de la alerta)
        sev_top = "normal"
        tipo_top = None
        titulo_top = ""
        for l in alertas_por_lote:
            for a in l.get("alertas", []):
                ctx = a.get("_contexto", {}) or {}
                if ctx.get("fecha") != fstr:
                    continue
                s = a.get("severidad", "")
                if rank.get(s, 0) > rank.get(sev_top, 0):
                    sev_top = s
                    tipo_top = (a.get("tipo") or "").lower() or None
                    titulo_top = a.get("titulo", "") or ""

        # Detectar contexto adicional (barro, humedad, lluvia 3 días)
        idx_3d = list(range(max(0, i - 2), i + 1))
        precip_3d = sum((precip[k] or 0) for k in idx_3d if k < len(precip))
        ll_dia = precip[i] if i < len(precip) and precip[i] else 0
        hum_dia = hum_max[i] if i < len(hum_max) and hum_max[i] else 0
        tmin_dia = t_min[i] if i < len(t_min) and t_min[i] is not None else None
        barro = precip_3d > 30
        # Humedad alta SOSTENIDA: ≥85% durante al menos 6 horas. Picos
        # de madrugada de 1-3h se secan con el sol y no impactan al
        # animal de la misma forma.
        horas_hr_alta = horas_hr_alta_por_fecha.get(fstr, 0)
        humedad_alta = hum_dia >= 85 and horas_hr_alta >= 6
        humedad_pico = hum_dia >= 85 and horas_hr_alta < 6

        # Clasificar con la función unificada (misma lógica que diaria/tarde)
        nivel_prod = clasificar_nivel_productivo(
            sev_top, tipo=tipo_top,
            contexto={
                "barro": barro,
                "lluvia_mm": ll_dia,
                "humedad_pct": hum_dia,
                "precip_3d_mm": precip_3d,
                "temp_min": tmin_dia,
            },
        )

        # Motivo: usar el título de la alerta + agregar contexto si aplica
        if nivel_prod == "critico":
            motivo = titulo_top or "Riesgo crítico"
        elif nivel_prod == "operativo":
            if titulo_top:
                motivo = titulo_top + (
                    " + barro" if barro else (
                        " + lluvia" if ll_dia > 5 else (
                            f" ({precip_3d:.0f}mm acum.)"
                            if precip_3d > 20 else ""
                        )
                    )
                )
            else:
                motivo = ("Barro persistente" if barro else
                          f"Lluvia acumulada {precip_3d:.0f}mm"
                          if precip_3d > 20 else "Riesgo operativo")
        elif nivel_prod == "atencion":
            # Motivo explicativo: conecta el factor climático con el efecto
            # sobre el animal. El productor debe entender QUÉ pasa, no
            # solo qué número leyó la estación meteorológica.
            if titulo_top:
                motivo = titulo_top
            elif humedad_alta and tmin_dia is not None and tmin_dia < 8:
                motivo = (f"Frío húmedo sostenido ({horas_hr_alta}h con "
                          f"HR ≥85%) — pelaje no se seca, sube gasto "
                          f"energético de mantenimiento")
            elif humedad_alta and tmin_dia is not None and tmin_dia < 13:
                motivo = (f"Humedad sostenida ({horas_hr_alta}h ≥85%) + "
                          f"fresco — pelaje tarda en secar, mayor "
                          f"consumo de calorías")
            elif humedad_alta:
                motivo = (f"Humedad sostenida ({horas_hr_alta}h ≥85%) — "
                          f"mezcla en comedero se deteriora, animal "
                          f"selecciona más")
            elif humedad_pico:
                motivo = (f"Pico de humedad de madrugada "
                          f"({horas_hr_alta}h ≥85%) — efecto transitorio, "
                          f"el pelaje se seca con el sol")
            elif ll_dia > 10:
                motivo = ("Lluvia significativa — barro de acceso "
                          "reduce visitas al comedero, consumo cae")
            else:
                motivo = "Atención moderada — sin agravantes claros"
        else:
            motivo = ""

        snapshot.append({
            "fecha": fstr,
            "severidad": sev_top,
            "tipo": tipo_top,
            "motivo": motivo,
            "nivel_productivo": nivel_prod,
        })

    # Riesgo residual: si AYER fue crítico de frío con barro o lluvia,
    # HOY puede mantener riesgo operativo aunque el clima ya mejoró.
    for i in range(1, len(snapshot)):
        ant = snapshot[i - 1]
        cur = snapshot[i]
        if (ant["nivel_productivo"] == "critico"
                and cur["nivel_productivo"] in ("normal", "atencion")):
            # Buscar si el día siguiente tiene barro / humedad arrastrada
            idx = fechas.index(cur["fecha"]) if cur["fecha"] in fechas else None
            if idx is not None:
                idx_3d = list(range(max(0, idx - 2), idx + 1))
                precip_3d = sum((precip[k] or 0) for k in idx_3d
                                 if k < len(precip))
                if precip_3d > 20:
                    cur["nivel_productivo"] = "operativo"
                    cur["motivo"] = (cur["motivo"] + " (residual)"
                                       if cur["motivo"]
                                       else "Riesgo residual post-evento")

    # ─── ACUMULACIÓN DE ESTRÉS ───
    # Un día de "atención" no es lo mismo que el día 5 consecutivo de
    # atención. El animal compensa la recuperación nocturna los primeros
    # 1-2 días, pero a partir del día 3 las reservas se agotan, la rumia
    # baja y el patrón de consumo se altera. Escalamos el semáforo según
    # la racha de días con frío/humedad sostenidos.
    racha = 0  # días consecutivos con nivel >= atención
    for i, dia in enumerate(snapshot):
        nivel = dia.get("nivel_productivo", "normal")
        if nivel in ("atencion", "operativo", "critico"):
            racha += 1
        else:
            racha = 0
            continue

        # Anotar la etapa (útil para la columna motivo)
        if racha == 1:
            etapa_txt = ""
        elif racha == 2:
            etapa_txt = "día 2"
        elif racha in (3, 4):
            etapa_txt = f"día {racha} — recuperación nocturna insuficiente"
        else:  # 5+
            etapa_txt = (f"día {racha} — acumulación, baja rumia y "
                         f"pérdida productiva real")

        # Escalar nivel productivo según racha. No degradamos crítico
        # (ya es el máximo).
        # Día 3-4 con atención → operativo (la recuperación nocturna ya
        # no compensa el gasto del día).
        # Día 5+ con atención → operativo más enfático.
        # Día 5+ con operativo → crítico (pérdida productiva real).
        if racha >= 5 and nivel == "operativo":
            dia["nivel_productivo"] = "critico"
            nivel = "critico"
        elif racha >= 3 and nivel == "atencion":
            dia["nivel_productivo"] = "operativo"
            nivel = "operativo"

        # Sumar al motivo la etapa de acumulación
        if etapa_txt:
            motivo_actual = dia.get("motivo", "") or ""
            if motivo_actual:
                dia["motivo"] = f"{motivo_actual} ({etapa_txt})"
            else:
                dia["motivo"] = etapa_txt.capitalize()

    return snapshot


def _icono_severidad(sev: str) -> str:
    return {"critica": "🔴", "warning": "🟠",
            "info": "🟡", "preventiva": "🟡"}.get(sev or "", "🟢")


def _tendencia_semana(snapshot: list) -> str:
    """Devuelve 'mejorando' | 'estable' | 'empeorando' comparando primera
    mitad vs segunda mitad de la semana."""
    if not snapshot or len(snapshot) < 4:
        return "estable"
    rank = {"normal": 0, "preventiva": 1, "info": 1,
             "warning": 2, "critica": 3}
    mid = len(snapshot) // 2
    primera = [rank.get(d.get("severidad", "normal"), 0) for d in snapshot[:mid]]
    segunda = [rank.get(d.get("severidad", "normal"), 0) for d in snapshot[mid:]]
    avg1 = sum(primera) / len(primera) if primera else 0
    avg2 = sum(segunda) / len(segunda) if segunda else 0
    if avg2 > avg1 + 0.5:
        return "empeorando"
    if avg1 > avg2 + 0.5:
        return "mejorando"
    return "estable"


def _detectar_eventos_semana(clima: dict, snapshot: list) -> dict:
    """Detecta tipos de eventos que se esperan en la semana para
    generar advertencias específicas:
      - calor: hay días con tipo='calor' warning/critica
      - frio: hay días con tipo='frio' warning/critica
      - lluvia: hay días con precipitación >5mm
      - barro: hay días con precipitación 3d >30mm
    """
    daily = clima.get("daily", {}) or {}
    precip = daily.get("precipitation_sum", [])
    fechas = daily.get("time", [])
    eventos = {"calor": False, "frio": False, "lluvia": False, "barro": False}

    # Calor / frío desde el snapshot
    for d in snapshot:
        if d.get("severidad") in ("warning", "critica"):
            t = (d.get("tipo") or "").lower()
            if t == "calor":
                eventos["calor"] = True
            elif t == "frio":
                eventos["frio"] = True

    # Lluvia (>5mm en algún día) y barro (>30mm en 3 días)
    hoy = datetime.now().date()
    for i, fstr in enumerate(fechas):
        try:
            f = datetime.strptime(fstr, "%Y-%m-%d").date()
        except ValueError:
            continue
        if f < hoy or (f - hoy).days > 7:
            continue
        p = precip[i] if i < len(precip) and precip[i] else 0
        if p > 5:
            eventos["lluvia"] = True
        # Barro: lluvia acumulada 3 días
        idx_3d = list(range(max(0, i - 2), i + 1))
        precip_3d = sum((precip[k] or 0) for k in idx_3d if k < len(precip))
        if precip_3d > 30:
            eventos["barro"] = True
    return eventos


def _categorias_sensibles(lotes_cliente: list) -> list:
    """Lista de categorías más sensibles cargadas en los lotes activos."""
    sensibles_orden = ["ternero", "vaquillona", "vaca_adulta", "toro"]
    cats = set()
    for l in lotes_cliente:
        cat = (l.get("categoria") or "").lower()
        for k in sensibles_orden:
            if k in cat:
                cats.add(k)
    nombres = {
        "ternero": "terneros / recría",
        "vaquillona": "vaquillonas",
        "vaca_adulta": "vacas adultas",
        "toro": "toros",
    }
    return [nombres[k] for k in sensibles_orden if k in cats]


_NIVEL_PROD_INFO = {
    "normal":    ("🟢", "Normal",          "#F8F8F8", "#666"),
    "atencion":  ("🟡", "Atención",        "#FFF9E6", "#9A7B00"),
    "operativo": ("🟠", "Riesgo operativo","#FFF3E0", "#C77400"),
    "critico":   ("🔴", "Crítico",         "#FFEBEE", "#C0392B"),
}


def _tabla_pronostico_html(clima: dict, alertas_por_lote: list,
                              snapshot: list = None) -> str:
    """Tabla con clima + semáforo productivo por día.

    snapshot: lista del _snapshot_pronostico con nivel_productivo y motivo.
    """
    daily = clima.get("daily", {})
    fechas = daily.get("time", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    viento = daily.get("wind_speed_10m_max", [])
    hum_max = daily.get("relative_humidity_2m_max", [])

    # Datos horarios para calcular cuántas horas estuvo HR ≥85% por día.
    # La HR máxima diaria es típicamente un pico de madrugada y no refleja
    # si el animal estuvo expuesto a humedad alta sostenida (lo que
    # realmente impacta el pelaje y el consumo).
    hourly = clima.get("hourly", {}) or {}
    hr_times = hourly.get("time", []) or []
    hr_hum = hourly.get("relative_humidity_2m", []) or []
    horas_hr_alta_por_fecha = {}
    for idx_h, ts in enumerate(hr_times):
        if idx_h >= len(hr_hum):
            break
        try:
            fecha_h = ts[:10]  # 'YYYY-MM-DDTHH:MM' → 'YYYY-MM-DD'
        except (TypeError, IndexError):
            continue
        valor = hr_hum[idx_h]
        if valor is None:
            continue
        if valor >= 85:
            horas_hr_alta_por_fecha[fecha_h] = (
                horas_hr_alta_por_fecha.get(fecha_h, 0) + 1
            )

    # Indexar el snapshot por fecha para lookup rápido
    snap_by_fecha = {s["fecha"]: s for s in (snapshot or [])}

    hoy = datetime.now().date()
    filas_html = []
    for i, fstr in enumerate(fechas):
        try:
            f = datetime.strptime(fstr, "%Y-%m-%d").date()
        except ValueError:
            continue
        if f < hoy:
            continue
        if (f - hoy).days > 7:
            break

        nombre_dia = _DIA_SEMANA[f.weekday()]
        fecha_corta = f.strftime("%d/%m")
        tmax = t_max[i] if i < len(t_max) and t_max[i] is not None else "—"
        tmin = t_min[i] if i < len(t_min) and t_min[i] is not None else "—"
        ll = precip[i] if i < len(precip) and precip[i] is not None else 0
        vt = viento[i] if i < len(viento) and viento[i] is not None else 0
        hum = (hum_max[i] if i < len(hum_max) and hum_max[i] is not None
                else None)

        # Nivel productivo + motivo del snapshot
        snap_item = snap_by_fecha.get(fstr, {})
        nivel_prod = snap_item.get("nivel_productivo", "normal")
        motivo = snap_item.get("motivo", "")
        icono, label, bg_color, txt_color = _NIVEL_PROD_INFO.get(
            nivel_prod, _NIVEL_PROD_INFO["normal"]
        )

        tmax_s = f"{tmax:.0f}°" if isinstance(tmax, (int, float)) else "—"
        tmin_s = f"{tmin:.0f}°" if isinstance(tmin, (int, float)) else "—"
        ll_s = f"{ll:.0f}mm" if ll > 0 else "—"
        vt_s = f"{vt:.0f}km/h"
        # Humedad: mostrar pico máximo + cuántas horas estuvo ≥85% (lo
        # que realmente importa para el animal). Un pico de 95% que
        # dura 2 horas en la madrugada no tiene el mismo efecto que
        # 95% sostenido 12 horas con neblina o post-lluvia.
        horas_alta = horas_hr_alta_por_fecha.get(fstr, 0)
        if hum is None:
            hum_s = "—"
        elif horas_alta >= 8:
            # Sostenida: pelaje no se seca, gasto energético real.
            hum_s = (
                f'<strong style="color:#C77400;">{hum:.0f}%</strong>'
                f'<br><span style="color:#888; font-size:10px;">'
                f'{horas_alta}h ≥85%</span>'
            )
        elif horas_alta >= 4:
            # Persistente: el pelaje se moja varias horas.
            hum_s = (
                f'<strong style="color:#9A7B00;">{hum:.0f}%</strong>'
                f'<br><span style="color:#888; font-size:10px;">'
                f'{horas_alta}h ≥85%</span>'
            )
        elif horas_alta > 0:
            # Solo pico de madrugada — efecto transitorio.
            hum_s = (
                f'{hum:.0f}%'
                f'<br><span style="color:#888; font-size:10px;">'
                f'{horas_alta}h ≥85%</span>'
            )
        else:
            hum_s = f"{hum:.0f}%"

        # Celda con color + nivel + motivo
        motivo_html = ""
        if motivo:
            motivo_html = (
                f'<br><span style="color:#666; font-size:11px;">'
                f'{motivo}</span>'
            )
        celda_riesgo = (
            f'<span style="color:{txt_color}; font-weight:600;">'
            f'{icono} {label}</span>{motivo_html}'
        )

        filas_html.append(f"""
        <tr style="background:{bg_color};">
          <td style="padding:8px 10px;"><strong>{nombre_dia}</strong> {fecha_corta}</td>
          <td style="padding:8px 10px; text-align:center;">{tmin_s} / {tmax_s}</td>
          <td style="padding:8px 10px; text-align:center;">{hum_s}</td>
          <td style="padding:8px 10px; text-align:center;">{ll_s}</td>
          <td style="padding:8px 10px; text-align:center;">{vt_s}</td>
          <td style="padding:8px 10px;">{celda_riesgo}</td>
        </tr>""")

    return f"""
    <table cellspacing="0" cellpadding="0" style="width:100%;
      border-collapse:collapse; font-size:13px; margin:12px 0;">
      <thead style="background:#1B3E27; color:white;">
        <tr>
          <th style="padding:8px 10px; text-align:left;">Día</th>
          <th style="padding:8px 10px;">Min/Max</th>
          <th style="padding:8px 10px;">HR%</th>
          <th style="padding:8px 10px;">Lluvia</th>
          <th style="padding:8px 10px;">Viento</th>
          <th style="padding:8px 10px; text-align:left;">Riesgo productivo</th>
        </tr>
      </thead>
      <tbody>{''.join(filas_html)}</tbody>
    </table>
    <p style="font-size:11px; color:#888; margin:4px 0 0;">
      <strong>HR%</strong> = humedad relativa máxima del día (pico) +
      cuántas horas estuvo ≥85%. Un pico breve de madrugada (1–3h) se
      seca con el sol; cuando son <strong>8 o más horas</strong>, el
      pelaje del animal no se seca y el gasto energético se vuelve
      sostenido (resaltado en naranja oscuro).
    </p>
    """


def _resumen_alertas_semana(alertas_por_lote: list,
                              snapshot: list = None) -> tuple:
    """Devuelve (n_dias_alerta, lista_strings_alertas, has_critica).

    Si se pasa `snapshot`, usa el nivel productivo del día (no la
    severidad climática raw) para construir los bullets. Esto evita
    que aparezca "Frío crítico" como bullet cuando el día se reclasificó
    a operativo o atención por contexto seco/adultos.
    """
    if snapshot:
        # Indexar nivel productivo por fecha
        nivel_por_fecha = {s["fecha"]: s.get("nivel_productivo", "normal")
                              for s in snapshot}
        # Mapeo nivel productivo → ícono + label suavizado
        nivel_label = {
            "critico": ("🔴", "Riesgo crítico"),
            "operativo": ("🟠", "Riesgo operativo"),
            "atencion": ("🟡", "Atención"),
        }

        dias_alerta = set()
        bullets = []
        # Construir un bullet por lote x día, pero filtrar/suavizar
        for l in alertas_por_lote:
            for a in l.get("alertas", []):
                ctx = a.get("_contexto", {}) or {}
                fecha = ctx.get("fecha")
                if not fecha:
                    continue
                nivel_dia = nivel_por_fecha.get(fecha, "normal")
                if nivel_dia not in ("critico", "operativo", "atencion"):
                    continue
                # Solo bullets para warning/critica del clima
                if a.get("severidad") not in ("warning", "critica"):
                    continue
                dias_alerta.add(fecha)
                icono, label = nivel_label[nivel_dia]
                tipo = (a.get("tipo") or "").lower()
                # Suavizar título según nivel: si el clima dice "crítico"
                # pero el contexto lo bajó, ajustar texto.
                titulo_clima = a.get("titulo", "Alerta")
                if nivel_dia == "atencion":
                    if "crítico" in titulo_clima.lower():
                        titulo_clima = titulo_clima.lower().replace(
                            "crítico", "moderado",
                        ).capitalize()
                elif nivel_dia == "operativo":
                    if "crítico" in titulo_clima.lower():
                        titulo_clima = titulo_clima.lower().replace(
                            "crítico", "operativo",
                        ).capitalize()
                cat_lote = l.get("categoria", "")
                bullets.append(
                    f"{icono} {label}: {titulo_clima.lower()} "
                    f"— lote {cat_lote}"
                )
        return (len(dias_alerta), bullets[:6],
                any(nivel_por_fecha.get(d) == "critico"
                    for d in dias_alerta))

    # Fallback (sin snapshot): comportamiento previo
    dias_alerta = set()
    bullets = []
    has_critica = False
    for l in alertas_por_lote:
        for a in l.get("alertas", []):
            sev = a.get("severidad")
            if sev not in ("critica", "warning"):
                continue
            ctx = a.get("_contexto", {}) or {}
            fecha = ctx.get("fecha")
            if fecha:
                dias_alerta.add(fecha)
            if sev == "critica":
                has_critica = True
            titulo = a.get("titulo", "Alerta")
            cat_lote = l.get("categoria", "")
            icono = _icono_severidad(sev)
            bullets.append(f"{icono} {titulo} — lote {cat_lote}")
    return len(dias_alerta), bullets[:6], has_critica


def componer_email_semanal(cliente: dict, clima: dict,
                              alertas_por_lote: list,
                              lotes_cliente: list = None) -> tuple:
    """Compone (subject, html, text) del email semanal con foco
    productivo y de seguimiento operativo."""
    nombre = cliente.get("nombre", "")
    establ = cliente.get("establecimiento", "") or cliente.get("localidad", "")
    fecha_lunes = datetime.now().strftime("%d/%m/%Y")
    fecha_domingo = (
        datetime.now() + timedelta(days=6)
    ).strftime("%d/%m")

    # Construir snapshot primero (la tabla productiva y los bullets
    # dependen de él para mantener coherencia).
    snapshot = _snapshot_pronostico(alertas_por_lote, clima)
    n_dias, bullets, has_critica = _resumen_alertas_semana(
        alertas_por_lote, snapshot=snapshot,
    )
    tendencia = _tendencia_semana(snapshot)

    # Contar días por nivel productivo
    cnt = {"normal": 0, "atencion": 0, "operativo": 0, "critico": 0}
    for s in snapshot:
        nivel = s.get("nivel_productivo", "normal")
        cnt[nivel] = cnt.get(nivel, 0) + 1
    eventos = _detectar_eventos_semana(clima, snapshot)
    cats_sensibles = _categorias_sensibles(lotes_cliente or [])

    # Subject según el peor nivel productivo de la semana
    if cnt["critico"] > 0:
        icono_titulo = "🔴"
        subject = (f"📅 Semana {fecha_lunes}–{fecha_domingo}: "
                     f"{cnt['critico']} día(s) crítico(s) — "
                     f"{establ or nombre}")
    elif cnt["operativo"] > 0:
        icono_titulo = "🟠"
        subject = (f"📅 Semana {fecha_lunes}–{fecha_domingo}: "
                     f"{cnt['operativo']} día(s) con riesgo operativo — "
                     f"{establ or nombre}")
    elif cnt["atencion"] > 0:
        icono_titulo = "🟡"
        subject = (f"📅 Semana {fecha_lunes}–{fecha_domingo}: "
                     f"{cnt['atencion']} día(s) de atención — "
                     f"{establ or nombre}")
    else:
        icono_titulo = "🟢"
        subject = (f"📅 Semana {fecha_lunes}–{fecha_domingo}: "
                     f"normal — buena ventana para manejos — "
                     f"{establ or nombre}")

    # ─── 1. TENDENCIA GENERAL ───
    tendencia_texto = {
        "mejorando": "📉 Tendencia <strong>mejorando</strong> — la semana "
                       "arranca con más exigencia y se estabiliza hacia el "
                       "fin de semana.",
        "estable": "➖ Tendencia <strong>estable</strong> durante toda la "
                     "semana.",
        "empeorando": "📈 Tendencia <strong>empeorando</strong> hacia el "
                        "fin de semana — anticipar acciones temprano.",
    }[tendencia]

    # ─── Copete proporcional al riesgo REAL del semáforo ───
    # Usa el conteo de niveles productivos (cnt) para mantener
    # coherencia entre header, copete, tabla y bullets.
    # Frase orientadora que recorre todos los copetes: el clima no es
    # un dato meteorológico aislado, es el principal disparador del
    # consumo y de la estabilidad ruminal.
    frase_lente = (
        "<br><span style='font-size:13px; color:#555;'>Cuando cambia "
        "el clima, lo primero que cambia es el <strong>consumo</strong> "
        "y la <strong>estabilidad del rumen</strong> — mucho antes que "
        "lo veamos en la balanza.</span>"
    )

    if cnt["critico"] > 0:
        copete = (
            f"<strong style='color:{ae.COLOR_ALERTA_CRITICA};'>"
            f"{cnt['critico']} día(s) con riesgo crítico sobre el "
            f"rodeo.</strong> Las decisiones de manejo en esos días "
            f"impactan directo sobre consumo, ganancia diaria y bienestar."
            f"{frase_lente}"
        )
    elif cnt["operativo"] > 0:
        copete = (
            f"<strong style='color:{ae.COLOR_ALERTA_WARNING};'>"
            f"Semana con {cnt['operativo']} día(s) de riesgo operativo.</strong> "
            f"Revisar acceso a comedero, mezcla y reparos en esos días "
            f"ayuda a sostener el consumo y evitar inestabilidad ruminal."
            f"{frase_lente}"
        )
    elif cnt["atencion"] > 0:
        # Predominio de atención = semana estable, sin lenguaje alarmista
        copete = (
            f"<strong style='color:#9A7B00;'>"
            f"Semana estable con {cnt['atencion']} día(s) de atención "
            f"leve.</strong> Condiciones para monitorear (humedad, frío "
            f"moderado, etc.), sin riesgos productivos importantes "
            f"previstos."
            f"{frase_lente}"
        )
    else:
        copete = (
            f"<strong style='color:{ae.COLOR_VERDE};'>"
            f"Semana climáticamente estable.</strong> "
            f"Sin riesgos productivos previstos — buen momento para "
            f"tareas planificadas: vacunaciones, pesadas, traslados, "
            f"ajustes de dieta."
        )

    # Flag global de "riesgo serio" — controla qué bloques mostrar
    riesgo_serio = cnt["critico"] > 0 or cnt["operativo"] > 0

    # ─── 2. ALERTAS PUNTUALES ───
    bullets_html = ""
    if bullets:
        # Título según el nivel dominante: si hay críticos, "riesgo
        # esperado"; si solo operativo/atención, "días para seguir".
        titulo_bullets = (
            "Días con riesgo esperado:"
            if (cnt["critico"] > 0 or cnt["operativo"] > 0)
            else "Días para seguir:"
        )
        bullets_html = (
            f"<h3 style='color:#1B3E27; margin-top:18px;'>"
            f"{titulo_bullets}</h3>"
            "<ul style='padding-left:18px;'>"
            + "".join(f"<li>{b}</li>" for b in bullets)
            + "</ul>"
        )

    # ─── 3. PREPARACIÓN OPERATIVA ───
    prep_items = []
    if eventos["frio"]:
        prep_items += [
            "Reparos disponibles (monte, cortina, galpón) revisados",
            "Bebederos sin riesgo de congelamiento",
            "Cama seca y drenajes funcionales",
            "Stock de concentrado funcional con fibra activa",
        ]
    if eventos["calor"]:
        prep_items += [
            "Sombra disponible (objetivo ≥ 4 m²/cab)",
            "Caudal y limpieza de bebederos",
            "Plan de horario de comidas adelantado (5-7 hs / 19-21 hs)",
            "Mezcla protegida del calor — evitar fermentación",
        ]
    if eventos["lluvia"] or eventos["barro"]:
        prep_items += [
            "Drenaje de corrales revisado",
            "Accesos a comedero sin barro profundo",
            "Mezcla protegida de humedad — pellet bajo cobertura",
            "Sobrantes de comedero retirados diariamente",
        ]

    prep_html = ""
    # Solo mostrar la lista completa si hay riesgo operativo/crítico.
    # Si solo hay atención, no llenar al cliente de preparativos
    # innecesarios.
    if prep_items and riesgo_serio:
        items_li = "".join(
            f"<li>{x}</li>" for x in prep_items
        )
        prep_html = (
            f"<h3 style='color:#1B3E27; margin-top:18px;'>"
            f"👉 Preparativos sugeridos para la semana:</h3>"
            f"<ul style='padding-left:18px;'>{items_li}</ul>"
        )

    # ─── 4. ADVERTENCIAS ESPECÍFICAS POR TIPO DE EVENTO ───
    # Solo mostrar bloques alarmistas si el riesgo del semáforo
    # lo justifica. Si solo hay 🟡 atención, mostrar versión liviana.
    adv_html_partes = []
    if eventos["calor"] and riesgo_serio:
        adv_html_partes.append(f"""
        <div style="background:#FFF3E0; border-left:3px solid
          {ae.COLOR_ALERTA_WARNING}; padding:12px 14px;
          margin:10px 0; border-radius:4px;">
          <strong style="color:{ae.COLOR_ALERTA_WARNING};">
            🥵 Calor previsto:</strong>
          <ul style="margin:6px 0 0; padding-left:18px;">
            <li>Posible caída del consumo de materia seca</li>
            <li>Mayor demanda de agua — revisar caudal y temperatura</li>
            <li>Riesgo de alimento caliente o fermentado en comedero</li>
            <li>Menor recuperación nocturna si las noches no afloja</li>
            <li>El riesgo se acumula si hay varios días seguidos</li>
          </ul>
        </div>""")
    if (eventos["lluvia"] or eventos["barro"]) and riesgo_serio:
        adv_html_partes.append(f"""
        <div style="background:#E3F2FD; border-left:3px solid
          {ae.COLOR_ALERTA_INFO}; padding:12px 14px;
          margin:10px 0; border-radius:4px;">
          <strong style="color:{ae.COLOR_ALERTA_INFO};">
            🌧️ Lluvia / humedad / barro:</strong>
          <ul style="margin:6px 0 0; padding-left:18px;">
            <li>Posible deterioro físico de la ración (pellet húmedo,
                desarmado, polvillo)</li>
            <li>Mayor selección y rechazo en comedero</li>
            <li>Riesgo de fermentación en mezcla mojada</li>
            <li>Menor acceso al comedero por barro</li>
            <li>Si persiste varios días: riesgo de pododermatitis</li>
          </ul>
        </div>""")
    elif (eventos["lluvia"] or eventos["barro"]) and cnt["atencion"] > 0:
        # Versión suave: solo seguimiento, no advertencia alarmista
        adv_html_partes.append(f"""
        <div style="background:#FFFBF0; border-left:3px solid
          #9A7B00; padding:12px 14px; margin:10px 0;
          border-radius:4px;">
          <strong style="color:#9A7B00;">
            💧 Humedad / lluvias leves a seguir:</strong>
          <p style="margin:6px 0 0;">
            Sin impacto productivo serio esperado, pero conviene
            seguir el estado de la mezcla y el acceso al comedero
            si la humedad persiste.
          </p>
        </div>""")
    if eventos["frio"] and riesgo_serio:
        adv_html_partes.append(f"""
        <div style="background:#E8F4F8; border-left:3px solid
          {ae.COLOR_ALERTA_INFO}; padding:12px 14px;
          margin:10px 0; border-radius:4px;">
          <strong style="color:#0277BD;">❄️ Frío previsto:</strong>
          <ul style="margin:6px 0 0; padding-left:18px;">
            <li>Aumento del requerimiento de mantenimiento</li>
            <li>Si es frío SECO: consumo se mantiene o sube</li>
            <li>Si hay lluvia o barro: <strong>posible caída de
                consumo</strong> por deterioro de la ración</li>
            <li>Acumulación: 2+ días seguidos = riesgo energético y
                sanitario</li>
          </ul>
        </div>""")
    elif eventos["frio"] and cnt["atencion"] > 0:
        # Frío moderado, sin combinación de agravantes serios
        adv_html_partes.append(f"""
        <div style="background:#FFFBF0; border-left:3px solid
          #9A7B00; padding:12px 14px; margin:10px 0;
          border-radius:4px;">
          <strong style="color:#9A7B00;">
            ❄️ Frío moderado a seguir:</strong>
          <p style="margin:6px 0 0;">
            Condiciones para monitorear. En animales adultos de razas
            británicas adaptadas y piso seco, el frío moderado por sí
            solo no implica impacto productivo serio. Mantener agua y
            reparos disponibles.
          </p>
        </div>""")
    advertencias_html = "".join(adv_html_partes)

    # ─── 5. CATEGORÍAS SENSIBLES ───
    cats_html = ""
    if cats_sensibles and (has_critica or n_dias > 0):
        cats_html = (
            f"<p style='background:#F0F8E8; border-left:3px solid "
            f"{ae.COLOR_LIMA}; padding:10px 14px; margin:14px 0;'>"
            f"🐄 <strong>Categorías más sensibles esta semana:</strong> "
            f"{', '.join(cats_sensibles)}. Priorizar atención y "
            f"recursos sobre estos lotes.</p>"
        )

    tabla_html = _tabla_pronostico_html(clima, alertas_por_lote,
                                            snapshot=snapshot)

    # ─── 5b. ANÁLISIS TÉCNICO LLM (personalizado por cliente) ───
    # Llama a Claude con los datos de la semana del cliente y arma un
    # párrafo a medida. Si la API falla, retorna None y el email cae a
    # la biblioteca de frases (no se rompe el envío).
    analisis_llm_html = ""
    try:
        from src.ai_analisis_semanal import (
            generar_analisis_llm, texto_a_html_parrafos,
        )
        # Calcular el impacto del peor evento de la semana sobre el lote
        # más sensible. Si hay frío relevante y peso disponible, pasamos
        # el rango al LLM para que cite los kg perdidos exactos.
        impacto_sem_txt = None
        try:
            from src.impacto_productivo import (
                estimar_impacto_peor_dia_semanal as _imp_sem,
                formato_impacto_texto as _fmt_imp,
            )
            _imp = _imp_sem(clima, lotes_cliente)
            if _imp:
                impacto_sem_txt = _fmt_imp(_imp)
        except Exception:
            impacto_sem_txt = None
        texto_llm = generar_analisis_llm(
            cliente=cliente, snapshot=snapshot, eventos=eventos,
            cnt=cnt, lotes=lotes_cliente,
            impacto_productivo_txt=impacto_sem_txt,
        )
        if texto_llm:
            cuerpo_llm = texto_a_html_parrafos(texto_llm)
            analisis_llm_html = f"""
            <div style="margin-top:18px; padding:14px 16px;
              background:#FAF9F2; border-left:3px solid {ae.COLOR_VERDE};
              border-radius:4px; font-size:13.5px; color:#333;">
              <strong style="color:#1B3E27;">📖 Análisis técnico para tu rodeo
              </strong>
              <p style="margin:4px 0 0; font-size:11px; color:#888;
                font-style:italic;">
                Análisis personalizado generado para tus datos de esta semana.
              </p>
              {cuerpo_llm}
            </div>"""
    except Exception:
        # Cualquier error → seguimos sin el bloque LLM, biblioteca cubre.
        analisis_llm_html = ""

    html = f"""<!DOCTYPE html>
<html><body style="margin:0; padding:0; background:#F4F4F4;
  font-family:Arial,sans-serif;">
  <div style="max-width:680px; margin:0 auto; background:white;">
    <div style="background:{ae.COLOR_VERDE}; padding:18px 24px;
      color:white;">
      <table width="100%"><tr>
        <td><img src="cid:hms-logo" height="44"></td>
        <td style="text-align:right;">
          <div style="font-size:13px; opacity:0.85;">
            Semana del {fecha_lunes} al {fecha_domingo}
          </div>
          <div style="font-size:18px; font-weight:600;">
            {icono_titulo} Monitoreo productivo semanal
          </div>
          <div style="font-size:12px; opacity:0.85; margin-top:3px;">
            Riesgo climático y operativo sobre consumo, manejo
            y bienestar animal.
          </div>
        </td>
      </tr></table>
    </div>
    <div style="padding:24px; color:#333; line-height:1.55;">
      <p style="font-size:15px;">Hola {nombre},</p>
      <p>{copete}</p>
      <p style="margin:10px 0; padding:8px 12px; background:#F8F8F8;
        border-radius:4px;">{tendencia_texto}</p>

      {bullets_html}

      {analisis_llm_html}

      <h3 style="color:#1B3E27; margin-top:18px;">
        Riesgo productivo día por día:
      </h3>
      <p style="font-size:12px; color:#666;">
        El semáforo refleja riesgo sobre consumo, barro, estrés térmico
        y acceso al comedero — no solo el dato meteorológico.
      </p>
      {tabla_html}

      {cats_html}

      {advertencias_html}

      {prep_html}

      <div style="margin-top:18px; padding:12px 14px;
        background:#F8F8F8; border-left:3px solid #888;
        border-radius:4px; font-size:12px; color:#555;">
        <strong style="color:#1B3E27;">🎯 Cómo afecta el clima al consumo
        y al rumen</strong>

        <p style="margin:6px 0 0;">
          Cuando hablamos de "riesgo productivo" estamos mirando dos
          cosas que el clima impacta antes que la balanza:
          <strong>cuánto come el animal</strong> y
          <strong>qué tan estable está su rumen</strong>. Si el consumo
          cae o cambia su patrón, la fermentación ruminal se
          desordena — y de ahí en adelante todo lo demás (ganancia,
          condición corporal, sanidad) se resiente.
        </p>

        <p style="margin:8px 0 0;">
          El semáforo combina temperatura, humedad, viento y lluvia
          para estimar ese impacto. Así actúa cada factor:
        </p>

        <p style="margin:10px 0 4px;"><strong>Frío + humedad alta (HR ≥ 85%)</strong></p>
        <p style="margin:0 0 0 6px;">
          El pelaje se moja y pierde poder aislante. El animal gasta
          10–25% más de energía sólo para mantener temperatura corporal.
          Si no compensa comiendo más (y muchas veces no puede,
          porque hay barro o la mezcla se moja), tira de reservas y
          pierde condición. El rumen entra en déficit energético.
        </p>

        <p style="margin:10px 0 4px;"><strong>Frío + viento (windchill)</strong></p>
        <p style="margin:0 0 0 6px;">
          El viento amplifica la pérdida de calor: 10°C reales con
          viento de 20 km/h se sienten como 5°C. Más windchill = más
          gasto de mantenimiento. El animal busca reparo y reduce
          tiempo en comedero — el consumo cae aunque la dieta esté bien
          formulada.
        </p>

        <p style="margin:10px 0 4px;"><strong>Lluvia + barro de acceso</strong></p>
        <p style="margin:0 0 0 6px;">
          El barro en la zona del comedero hace que el animal dude al
          comer, reduce las visitas y selecciona más (deja sobrantes).
          Consumo real puede caer 5–15%. Si además la mezcla se moja,
          fermenta y pierde palatabilidad — el animal come MENOS Y PEOR,
          combinación riesgosa para la estabilidad del rumen.
        </p>

        <p style="margin:10px 0 4px;"><strong>Comidas espaciadas o concentradas</strong></p>
        <p style="margin:0 0 0 6px;">
          Cuando el animal cambia su patrón habitual de consumo
          (come todo de golpe cuando para el viento, o se saltea
          horarios), aparecen picos de ácido en rumen y caídas de pH.
          Eso es inestabilidad ruminal: menos rumia, menos producción
          de proteína microbiana, riesgo de acidosis subclínica. La
          ración formulada en papel deja de ser la que se aprovecha.
        </p>

        <p style="margin:10px 0 4px;"><strong>Acumulación día tras día</strong></p>
        <p style="margin:0 0 0 6px;">
          Un día de estrés se compensa con la recuperación nocturna.
          Pero 3–5 días seguidos agotan reservas, bajan rumia y la
          pérdida productiva se vuelve real. Por eso el semáforo se
          carga a medida que el cuadro se sostiene — no es alarma:
          es que el rumen ya no tiene margen para compensar.
        </p>

        <p style="margin:10px 0 0;">
          <em>Un dato climático aislado no implica pérdida automática.</em>
          El impacto depende del lote: infraestructura, categoría
          animal, calidad y disponibilidad de la mezcla, reparos y
          manejo del comedero. El objetivo no es predecir el clima —
          es <strong>anticipar su impacto sobre consumo y rumen</strong>
          antes que se pierdan días de eficiencia.
        </p>
      </div>

      <p style="margin-top:18px; padding:12px 14px;
        background:#F0F8E8; border-radius:4px;
        color:{ae.COLOR_VERDE}; font-size:13px;">
        🔔 Cada día que el clima requiera una acción concreta, vas a
        recibir un aviso puntual con el detalle del lote. Este email
        es <em>solo</em> el monitoreo semanal — sirve para preparar
        operativa, mezcla y reparos con tiempo.
      </p>

      <p style="margin-top:18px; color:#888; font-size:12px;">
        — Mauricio Suárez<br>HMS Nutrición Animal
      </p>
    </div>
    <div style="background:{ae.COLOR_VERDE}; padding:12px 24px;
      color:white; font-size:11px; text-align:center;">
      <strong>HMS Nutrición Animal</strong> — monitoreo automático<br>
      Para darse de baja, respondé este email con la palabra BAJA.
    </div>
  </div>
</body></html>"""

    text_lines = [
        f"Monitoreo productivo semanal — {fecha_lunes} al {fecha_domingo}",
        f"Riesgo climático y operativo sobre consumo, manejo y bienestar.",
        "",
        f"Hola {nombre},",
        "",
    ]
    if cnt["critico"] > 0:
        text_lines.append(
            f"RIESGO ALTO: {cnt['critico']} día(s) crítico(s) sobre el "
            f"rodeo."
        )
    elif cnt["operativo"] > 0:
        text_lines.append(
            f"Semana con {cnt['operativo']} día(s) de riesgo operativo."
        )
    elif cnt["atencion"] > 0:
        text_lines.append(
            f"Semana estable con {cnt['atencion']} día(s) de atención "
            f"leve. Condiciones para monitorear."
        )
    else:
        text_lines.append(
            "Semana climáticamente estable. Sin riesgos productivos "
            "previstos."
        )
    text_lines.append("")
    text_lines.append({
        "mejorando": "Tendencia: MEJORANDO hacia el fin de semana.",
        "estable": "Tendencia: ESTABLE durante toda la semana.",
        "empeorando": "Tendencia: EMPEORANDO hacia el fin de semana.",
    }[tendencia])
    text_lines.append("")

    if bullets:
        text_lines.append("Días con riesgo esperado:")
        for b in bullets:
            text_lines.append(f"- {b}")
        text_lines.append("")

    if cats_sensibles and (has_critica or n_dias > 0):
        text_lines.append(
            f"Categorías más sensibles: {', '.join(cats_sensibles)}"
        )
        text_lines.append("")

    if eventos["calor"] and riesgo_serio:
        text_lines += [
            "CALOR PREVISTO — advertencias:",
            "  - Posible caída de consumo",
            "  - Mayor demanda de agua",
            "  - Riesgo de alimento caliente/fermentado",
            "  - Menor recuperación nocturna",
            "  - El riesgo se acumula si son varios días seguidos",
            "",
        ]
    if (eventos["lluvia"] or eventos["barro"]) and riesgo_serio:
        text_lines += [
            "LLUVIA / BARRO — advertencias:",
            "  - Posible deterioro físico de la ración",
            "  - Mayor selección, menor acceso al comedero",
            "  - Riesgo de fermentación si la mezcla se moja",
            "  - Si persiste: riesgo de pododermatitis",
            "",
        ]
    elif ((eventos["lluvia"] or eventos["barro"]) and cnt["atencion"] > 0):
        text_lines += [
            "Humedad / lluvias leves a seguir:",
            "  Sin impacto productivo serio esperado. Conviene seguir",
            "  el estado de la mezcla y el acceso al comedero.",
            "",
        ]
    if eventos["frio"] and riesgo_serio:
        text_lines += [
            "FRÍO PREVISTO — advertencias:",
            "  - Aumento del requerimiento de mantenimiento",
            "  - Frío SECO: consumo se mantiene o sube",
            "  - Frío + lluvia/barro: posible caída de consumo",
            "  - Acumulación 2+ días: riesgo energético y sanitario",
            "",
        ]
    elif eventos["frio"] and cnt["atencion"] > 0:
        text_lines += [
            "Frío moderado a seguir:",
            "  Condiciones para monitorear. En animales adultos de razas",
            "  británicas adaptadas y piso seco, el frío moderado por sí",
            "  solo no implica impacto productivo serio.",
            "",
        ]

    if prep_items and riesgo_serio:
        text_lines.append("PREPARATIVOS sugeridos:")
        for x in prep_items:
            text_lines.append(f"  - {x}")
        text_lines.append("")

    text_lines += [
        "CÓMO INTERPRETAR ESTE RIESGO",
        "El semáforo refleja riesgo productivo (consumo, barro, estrés,",
        "acceso al alimento, sanitario), no solo el dato meteorológico.",
        "",
        "El clima es un disparador. El impacto real depende del contexto:",
        "  - Categoría animal (sensibilidad)",
        "  - Acumulación (días consecutivos)",
        "  - Barro, humedad prolongada",
        "  - Acceso a agua y comedero",
        "  - Calidad física de la ración",
        "  - Sombra, reparos, manejo",
        "  - Recuperación nocturna",
        "",
        "Un THI alto no implica pérdidas automáticas. Si el lote tiene",
        "buena infraestructura y manejo, el impacto real puede ser menor;",
        "si está expuesto, puede ser peor.",
        "",
        "Cada día que el clima lo requiera, vas a recibir un aviso "
        "puntual con detalle.",
        "",
        "— Mauricio Suárez — HMS Nutrición Animal",
        "",
        "Para darse de baja, respondé este email con la palabra BAJA.",
    ]

    return subject, html, "\n".join(text_lines)


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--solo-cliente", default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Saltar el dedup y re-enviar aunque ya se haya mandado hoy.",
    )
    args = parser.parse_args()

    log = setup_logging()

    # Lock global: evitar dos instancias en paralelo (launchd al despertar
    # la Mac puede disparar el job atrasado + el StartCalendarInterval).
    lock_fd = adquirir_lock_proceso("alertas_semanales")
    if lock_fd is None:
        log.info("=== abortado: otra instancia ya corre ===")
        return 0

    log.info("=== ALERTAS SEMANALES — INICIO ===")

    db.init_db()

    cfg = ae.cargar_config_smtp()
    ok, err = ae.config_valida(cfg)
    if not ok and not args.dry_run:
        log.error(f"Config SMTP inválida: {err}")
        return 1

    fecha_db = datetime.now().strftime("%Y-%m-%d")

    clientes = db.listar_clientes()
    if args.solo_cliente:
        clientes = [c for c in clientes
                    if args.solo_cliente.lower() in c["nombre"].lower()]
    log.info(f"Clientes a procesar: {len(clientes)}")

    enviados = 0
    errores = 0

    for c in clientes:
        log.info(f"Cliente: {c['nombre']}")
        try:
            # Resolver coordenadas y clima
            lat = c.get("lat")
            lon = c.get("lon")
            localidad = c.get("localidad", "")
            if lat and lon:
                geo = geocodificar_manual(float(lat), float(lon), localidad)
            elif localidad:
                geo = geocodificar(localidad)
            else:
                log.warning(f"  {c['nombre']}: sin coordenadas. Skip.")
                continue
            if not geo:
                log.warning(f"  {c['nombre']}: no se pudo geocodificar. Skip.")
                continue

            clima = obtener_clima(geo["lat"], geo["lon"])
            if not clima:
                log.warning(f"  {c['nombre']}: Open-Meteo no respondió. Skip.")
                continue

            # Generar alertas de la semana por lote
            lotes = db.listar_lotes(cliente_id=c["id"], estado="activo")
            alertas_por_lote = []
            for l in lotes:
                alertas = generar_alertas_predictivas(
                    clima,
                    categoria=l.get("categoria", ""),
                    raza=l.get("raza", ""),
                )
                relevantes = [
                    a for a in alertas
                    if a.get("severidad") in ("warning", "critica")
                ]
                if relevantes or l:
                    alertas_por_lote.append({
                        "lote": l["identificador"],
                        "categoria": l.get("categoria", ""),
                        "alertas": relevantes,
                    })

            # Componer + enviar a TODOS los destinatarios
            destinatarios = db.listar_destinatarios(c)
            if not destinatarios:
                continue

            subject, html, text = componer_email_semanal(
                c, clima, alertas_por_lote, lotes_cliente=lotes,
            )

            # Guardar snapshot para que el cron del miércoles pueda
            # comparar y detectar cambios significativos.
            try:
                hoy_dt = datetime.now().date()
                # Lunes de esta semana = hoy si es lunes; sino retroceder.
                lunes = hoy_dt - timedelta(days=hoy_dt.weekday())
                snapshot = _snapshot_pronostico(alertas_por_lote, clima)
                db.guardar_snapshot_pronostico(
                    c["id"], lunes.isoformat(), snapshot,
                )
                log.info(f"  Snapshot pronóstico guardado "
                          f"({len(snapshot)} días)")
            except Exception as e:
                log.warning(f"  No se pudo guardar snapshot: {e}")

            for d in destinatarios:
                email = (d.get("email") or "").strip()
                if not email or not d.get("alertas_email_activas", 1):
                    continue

                # Dedup tipo='semanal' — distinto de diaria/tarde
                # --force salta este chequeo (útil para pruebas y reenvíos).
                if not args.force and db.alerta_ya_enviada_hoy(
                    c["id"], email, fecha_db, tipo="semanal",
                ):
                    log.info(f"  {email}: semanal ya enviada hoy. Skip.")
                    continue
                if args.force:
                    log.info(f"  {email}: --force activado, ignorando dedup.")

                if args.dry_run:
                    log.info(f"  [DRY-RUN] semanal -> {email}")
                    enviados += 1
                    continue

                ok2, msg = ae.enviar_email(cfg, [email], subject, html, text)
                if ok2:
                    log.info(f"  ✓ semanal -> {email}")
                    db.registrar_alerta_enviada(
                        fecha_db, c["id"], email, subject,
                        len([a for l in alertas_por_lote
                              for a in l["alertas"]]),
                        "enviada", "", tipo="semanal",
                    )
                    enviados += 1
                else:
                    log.error(f"  ✗ semanal -> {email}: {msg}")
                    db.registrar_alerta_enviada(
                        fecha_db, c["id"], email, subject, 0,
                        "error", msg, tipo="semanal",
                    )
                    errores += 1

        except Exception as e:
            log.exception(f"  Error procesando {c['nombre']}: {e}")
            errores += 1

    log.info(
        f"=== FIN — Enviados: {enviados}, errores: {errores} ==="
    )
    liberar_lock(lock_fd)
    return 0 if errores == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
