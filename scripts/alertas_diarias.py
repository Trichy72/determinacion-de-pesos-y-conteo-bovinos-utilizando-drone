#!/usr/bin/env python3
"""
Job diario de alertas climáticas — corre vía cron / launchd / Task Scheduler.

Itera todos los clientes activos con coordenadas y email habilitado,
calcula alertas climáticas (Open-Meteo + SMN), agrupa por destinatario
y manda los emails.

Uso:
    python scripts/alertas_diarias.py [--dry-run] [--solo-cliente NOMBRE]

Cron (crontab -e):
    30 7 * * *  cd /ruta/al/proyecto && /usr/bin/python3 scripts/alertas_diarias.py

launchd (macOS): ver scripts/com.hms.alertas-diarias.plist (generado abajo)

Salida:
    - log en data/logs/alertas_YYYY-MM-DD.log
    - tabla alertas_enviadas en SQLite
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Permitir ejecutar como `python scripts/alertas_diarias.py` desde la raíz
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import database as db
from src import alertas_email as ae
from src import whatsapp as wa
from src.locking import adquirir_lock_proceso, liberar_lock
from src.procesador_bajas import procesar_bajas_pendientes
from src.clima import (
    obtener_clima, generar_alertas_predictivas, calcular_thi, clasificar_thi,
    geocodificar_manual, geocodificar, clasificar_nivel_productivo,
    clasificar_etapa_evento,
)
from src.clima_smn import resumen_smn
from src.clima_weatherapi import (
    obtener_alertas_oficiales as wapi_obtener_alertas_oficiales,
)


# =====================================================================
# LOGGING
# =====================================================================

def setup_logging() -> logging.Logger:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"alertas_{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger("alertas_diarias")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # También adjuntar los handlers al logger del módulo de análisis
    # LLM (`hms.ai_analisis`) para que sus warnings (fallas de Claude,
    # JSON inválido, API caída, etc.) queden en el mismo archivo de log.
    # Sin esto, los warnings caen al void y nunca los vemos.
    ai_logger = logging.getLogger("hms.ai_analisis")
    ai_logger.setLevel(logging.INFO)
    ai_logger.handlers.clear()
    ai_logger.addHandler(fh)
    ai_logger.addHandler(sh)
    ai_logger.propagate = False  # evitar duplicar líneas

    return logger


# =====================================================================
# LÓGICA
# =====================================================================

def _nivel_productivo_maximo(datos: dict) -> str:
    """Devuelve el peor nivel productivo entre todos los lotes del
    cliente para HOY. Usa la lógica unificada de `clasificar_nivel_productivo`.

    Returns: 'normal' | 'atencion' | 'operativo' | 'critico'
    """
    rank = {"normal": 0, "atencion": 1, "operativo": 2, "critico": 3}
    inverso = {v: k for k, v in rank.items()}
    peor = 0

    clima_actual = datos.get("clima_actual") or {}
    # Contexto compartido para todos los lotes (clima de hoy + ambiente)
    contexto_base = {
        "humedad_pct": clima_actual.get("humedad_pct"),
        # Otros campos dependientes del lote se setean abajo si aplica
    }

    for l in datos.get("alertas_por_lote", []):
        for a in l.get("alertas", []):
            sev = a.get("severidad", "")
            tipo = (a.get("tipo") or "").lower()
            ctx = a.get("_contexto", {}) or {}
            contexto = {
                **contexto_base,
                "barro": ctx.get("barro", False),
                "lluvia_mm": ctx.get("lluvia_mm", 0),
                "precip_3d_mm": ctx.get("precip_3d_mm", 0),
                "temp_min": ctx.get("min_nocturna") or ctx.get("temp_min"),
            }
            nivel = clasificar_nivel_productivo(sev, tipo=tipo,
                                                  contexto=contexto)
            peor = max(peor, rank.get(nivel, 0))
    return inverso[peor]


def calcular_alertas_cliente(cliente: dict, log: logging.Logger) -> dict:
    """Calcula alertas y datos climáticos de un cliente.

    Returns: {
      'cliente': cliente_dict,
      'alertas_por_lote': [{lote, categoria, alertas: [...]}],
      'clima_actual': {temp_c, humedad_pct, thi, thi_estado} | None,
      'smn': resumen_smn() | None,
      'tiene_algo': bool,
    }
    """
    nombre = cliente.get("nombre", "")
    lat = cliente.get("lat")
    lon = cliente.get("lon")
    localidad = cliente.get("localidad", "")

    # Resolver coordenadas (manual > geocodificar)
    if lat and lon:
        geo = geocodificar_manual(float(lat), float(lon), localidad)
    elif localidad:
        geo = geocodificar(localidad)
    else:
        log.warning(f"  {nombre}: sin coordenadas ni localidad. Skip.")
        return {"cliente": cliente, "alertas_por_lote": [], "clima_actual": None,
                "smn": None, "tiene_algo": False, "error_api": False}

    if not geo:
        log.warning(f"  {nombre}: no se pudo geocodificar. Skip.")
        return {"cliente": cliente, "alertas_por_lote": [], "clima_actual": None,
                "smn": None, "tiene_algo": False, "error_api": True}

    clima = obtener_clima(geo["lat"], geo["lon"])
    if not clima:
        log.warning(f"  {nombre}: Open-Meteo no respondió. Skip.")
        return {"cliente": cliente, "alertas_por_lote": [], "clima_actual": None,
                "smn": None, "tiene_algo": False, "error_api": True}

    # Datos actuales
    actual = clima.get("current", {}) or {}
    t = actual.get("temperature_2m")
    hr = actual.get("relative_humidity_2m")
    clima_actual = None
    if t is not None and hr is not None:
        thi = calcular_thi(t, hr)
        clima_actual = {
            "temp_c": t,
            "humedad_pct": hr,
            "thi": thi,
            "thi_estado": clasificar_thi(thi),
        }

    # Auto-finalizar lotes cuya fecha_salida ya pasó (los animales ya no
    # están en el corral, no tiene sentido seguir alertando).
    hoy_str = datetime.now().date().isoformat()
    todos_lotes = db.listar_lotes(cliente_id=cliente["id"], estado="activo")
    for l in todos_lotes:
        fecha_salida = (l.get("fecha_salida") or "").strip()
        if fecha_salida and fecha_salida < hoy_str:
            db.actualizar_lote(l["id"], estado="finalizado")
            log.info(f"  {nombre}: lote {l.get('identificador')} "
                       f"finalizado automáticamente (fecha_salida "
                       f"{fecha_salida} ya pasó).")

    # Alertas por lote (solo los que siguen activos)
    lotes = db.listar_lotes(cliente_id=cliente["id"], estado="activo")
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
        if relevantes:
            # Peso actual del lote: última pesada si existe, sino peso de
            # ingreso. Cantidad: cantidad_inicial del lote.
            peso_actual = (l.get("ultimo_peso_kg")
                            or l.get("peso_ingreso_kg"))

            # === Dieta vigente del lote ===
            # Si el agente IA armó una dieta (o un plan de adaptación),
            # leemos la versión activa HOY — eso permite que la alerta
            # haga recomendaciones específicas sobre la mezcla real,
            # no genéricas. Usamos la misma lógica vigente-by-date que
            # stock_producto, así si hay plan de adaptación de 4 fases,
            # toma la fase que corresponde al día actual.
            dieta_vigente = None
            try:
                dietas = db.listar_dietas(l["id"])
                if dietas:
                    # listar_dietas devuelve ORDER BY fecha DESC. Buscamos
                    # la primera con fecha <= hoy.
                    for d in dietas:
                        if (d.get("fecha") or "") <= hoy_str:
                            dieta_vigente = d
                            break
                    # Si todas son futuras (plan cargado pero todavía no
                    # arrancó), no hay dieta vigente — el LLM seguirá
                    # con lógica genérica.
            except Exception as e:
                log.warning(
                    f"  {nombre}: error leyendo dieta del lote "
                    f"{l.get('identificador')}: {e}"
                )

            # === Diagnóstico del sistema de alimentación ===
            # Define qué tan rápido se puede ajustar la mezcla ante un
            # evento climático. Forzamos lineal diario si está en los
            # primeros 15 días de adaptación.
            try:
                diag_alim = db.diagnostico_alimentacion_lote(l)
            except Exception as e:
                log.warning(
                    f"  {nombre}: error en diagnóstico_alimentacion del "
                    f"lote {l.get('identificador')}: {e}"
                )
                diag_alim = None

            alertas_por_lote.append({
                "lote": l["identificador"],
                "lote_id": l["id"],
                "categoria": l.get("categoria", ""),
                "raza": l.get("raza", ""),
                "peso_promedio_kg": peso_actual,
                "cantidad_animales": l.get("cantidad_inicial"),
                # Overrides para refinar el cálculo NRC con los valores
                # reales del lote (si están cargados en la ficha).
                "adpv_objetivo_kg": l.get("adpv_objetivo_kg"),
                "energia_dieta_mcal_em_kg_ms": l.get(
                    "energia_dieta_mcal_em_kg_ms"
                ),
                # Dieta y sistema de alimentación — para que el LLM
                # personalice las acciones a la mezcla y al comedero
                # reales del lote.
                "dieta_vigente": dieta_vigente,
                "diagnostico_alimentacion": diag_alim,
                "alertas": relevantes,
            })

    # Resumen SMN (datos observados oficiales)
    smn = None
    try:
        smn = resumen_smn(geo["lat"], geo["lon"])
    except Exception as e:
        log.warning(f"  {nombre}: error SMN: {e}")

    # Alertas oficiales: el SMN no expone API pública (Cloudflare bloquea
    # todo acceso programático) y WeatherAPI no sincroniza alertas
    # argentinas. Si en algún momento aparece una API confiable, sumarla acá.
    alertas_oficiales = []

    n_total = sum(len(l["alertas"]) for l in alertas_por_lote)
    log.info(f"  {nombre}: {n_total} alertas en {len(alertas_por_lote)} lotes")

    return {
        "cliente": cliente,
        "alertas_por_lote": alertas_por_lote,
        "clima_actual": clima_actual,
        "smn": smn,
        "alertas_oficiales": alertas_oficiales,
        "tiene_algo": (n_total > 0 or len(lotes) > 0
                        or len(alertas_oficiales) > 0),
    }


def _enviar_email_a_destinatario(cfg: dict, cliente: dict, contacto: dict,
                                    datos: dict, log: logging.Logger,
                                    dry_run: bool,
                                    tipo: str = "diaria",
                                    asunto_override: str = None,
                                    force: bool = False) -> tuple:
    """Manda el email de alertas a un destinatario puntual.

    Si el destinatario nunca recibió bienvenida, manda PRIMERO el email
    de bienvenida (explicando qué es HMS, cómo darse de baja, etc.)
    y después el email de alertas.

    `tipo` permite separar ventanas (diaria/tarde) en el dedup.
    `asunto_override` reemplaza el asunto si se pasa (útil para "tarde").

    Retorna (enviados, errores).
    """
    email = (contacto.get("email") or "").strip()
    nombre_dest = contacto.get("nombre", "") or "destinatario"
    fecha = datetime.now().strftime("%Y-%m-%d")

    if not email:
        return 0, 0

    # Dedup por destinatario+cliente, separado por tipo de envío.
    # --force salta el chequeo para reenvíos manuales (pruebas).
    if not force and db.alerta_ya_enviada_hoy(cliente["id"], email, fecha,
                                                tipo=tipo):
        return 0, 0

    enviados = 0
    errores = 0

    # ─── BIENVENIDA (solo la primera vez) ───
    if not contacto.get("bienvenida_email_enviada", 0):
        if dry_run:
            log.info(f"  [DRY-RUN bienvenida email] -> {email}")
        else:
            try:
                bs, bh, bt = ae.componer_bienvenida(cliente, contacto)
                ok_b, msg_b = ae.enviar_email(cfg, [email], bs, bh, bt)
                if ok_b:
                    log.info(f"  ✓ bienvenida email -> {email} ({nombre_dest})")
                    db.marcar_bienvenida_enviada(
                        contacto["origen"], contacto["id"], "email",
                    )
                else:
                    log.warning(
                        f"  ⚠ bienvenida email a {email} falló: {msg_b}. "
                        f"Igual mando la alerta."
                    )
            except Exception as e:
                log.warning(f"  ⚠ Error en bienvenida email: {e}")

    # ─── ALERTA REGULAR ───
    # `lectura_out` captura el texto de la "Lectura técnica" generada por
    # el LLM para poder guardarla en DB y usarla como MEMORIA del LLM
    # en el próximo email del cliente (anti banner-blindness).
    lectura_out: dict = {}
    subject, html, text = ae.componer_alerta_diaria(
        cliente=cliente,
        alertas_por_lote=datos["alertas_por_lote"],
        smn_resumen=datos["smn"],
        clima_actual=datos["clima_actual"],
        alertas_oficiales=datos.get("alertas_oficiales", []),
        etapa=datos.get("etapa_evento", "inicio"),
        dias_alerta_previos=datos.get("dias_alerta_previos", 0),
        lectura_out=lectura_out,
    )
    if asunto_override:
        subject = asunto_override
    n_alertas = sum(len(l["alertas"]) for l in datos["alertas_por_lote"])
    lectura_tecnica_txt = (
        lectura_out.get("texto", "") if lectura_out.get("fuente_llm") else ""
    ) or None

    if dry_run:
        log.info(f"  [DRY-RUN email] -> {email} ({nombre_dest}): {subject}")
        return enviados + 1, errores

    ok, msg = ae.enviar_email(cfg, [email], subject, html, text)
    if ok:
        log.info(f"  ✓ email -> {email} ({nombre_dest})")
        db.registrar_alerta_enviada(
            fecha, cliente["id"], email, subject, n_alertas,
            "enviada", "", tipo=tipo,
            lectura_tecnica=lectura_tecnica_txt,
        )
        enviados += 1
    else:
        log.error(f"  ✗ email -> {email}: {msg}")
        db.registrar_alerta_enviada(
            fecha, cliente["id"], email, subject, n_alertas,
            "error", msg, tipo=tipo,
        )
        errores += 1
    return enviados, errores


def _enviar_whatsapp_a_destinatario(cfg_wa: dict, cliente: dict,
                                      contacto: dict, datos: dict,
                                      log: logging.Logger,
                                      dry_run: bool,
                                      tipo: str = "diaria",
                                      cabecera_override: str = None,
                                      nivel_productivo: str = "critico",
                                      force: bool = False) -> tuple:
    """Manda el WhatsApp corto a un destinatario puntual.

    Si nunca recibió bienvenida, manda primero el WhatsApp de bienvenida.

    `tipo` separa el dedup por ventana (diaria/tarde).
    `cabecera_override` reemplaza el "ALERTA HMS — ..." (útil para tarde).
    """
    from src import whatsapp as wa

    whatsapp_dest = (contacto.get("whatsapp") or "").strip()
    nombre_dest = contacto.get("nombre", "") or "destinatario"

    if not whatsapp_dest:
        return 0, 0
    if not datos.get("alertas_por_lote"):
        # Sin alertas no mandamos WhatsApp (ni siquiera bienvenida — esperamos
        # a que haya algo concreto que reportar).
        return 0, 0

    enviados = 0
    errores = 0

    # ─── BIENVENIDA WhatsApp (solo la primera vez) ───
    if not contacto.get("bienvenida_whatsapp_enviada", 0):
        if dry_run:
            log.info(f"  [DRY-RUN bienvenida wa] -> {whatsapp_dest}")
        else:
            try:
                wa_bienv = wa.componer_bienvenida(cliente, contacto)
                ok_b, msg_b = wa.enviar_texto(cfg_wa, whatsapp_dest, wa_bienv)
                if ok_b:
                    log.info(
                        f"  ✓ bienvenida wa -> {whatsapp_dest} ({nombre_dest})"
                    )
                    db.marcar_bienvenida_enviada(
                        contacto["origen"], contacto["id"], "whatsapp",
                    )
                else:
                    log.warning(
                        f"  ⚠ bienvenida wa a {whatsapp_dest} falló: {msg_b}."
                    )
            except Exception as e:
                log.warning(f"  ⚠ Error en bienvenida wa: {e}")

    # ─── ALERTA WHATSAPP ───
    # Tomar el peor lote para componer un único mensaje al destinatario
    peor_lote = None
    peor_rank = -1
    rank_sev = {"critica": 3, "warning": 2, "info": 1}
    for l in datos["alertas_por_lote"]:
        for a in l.get("alertas", []):
            r = rank_sev.get(a.get("severidad", ""), 0)
            if r > peor_rank:
                peor_rank = r
                peor_lote = (l, a)

    if not peor_lote:
        return enviados, errores

    lote_obj, alerta_obj = peor_lote
    nombre_cliente = cliente.get("nombre", "")
    sev = alerta_obj.get("severidad", "")
    tipo_clima = alerta_obj.get("tipo", "")
    nivel_alerta = alerta_obj.get("nivel", "")
    cat_corto = (lote_obj.get("categoria", "")
                  or alerta_obj.get("titulo", ""))[:30]

    # Cabecera según nivel PRODUCTIVO (no severidad climática raw).
    # Tres niveles bien diferenciados (alineado feedback CRM 360):
    #   - crítico: "ALERTA HMS" (rojo, riesgo serio)
    #   - operativo: "SEGUIMIENTO OPERATIVO HMS" (naranja, manejo activo)
    #   - atención: "ATENCIÓN HMS" (amarillo, monitoreo)
    if nivel_productivo == "critico":
        icono = "🔴"
        prefijo_cabecera = "ALERTA HMS"
    elif nivel_productivo == "operativo":
        icono = "🟠"
        prefijo_cabecera = "SEGUIMIENTO OPERATIVO HMS"
    else:
        # Atención (o fallback): tono medido, no alarmista
        icono = "🟡"
        prefijo_cabecera = "ATENCIÓN HMS"

    if cabecera_override:
        cabecera_principal = cabecera_override
        if cat_corto and cat_corto.lower() not in cabecera_override.lower():
            cabecera_principal += f" ({cat_corto})"
    else:
        cabecera_principal = f"{icono} {prefijo_cabecera} — {nombre_cliente}"
        if cat_corto:
            cabecera_principal += f" ({cat_corto})"

    # Tipo+nivel: el label varía según nivel productivo para no
    # ser alarmista cuando no corresponde.
    if nivel_productivo == "critico":
        tipo_label = (tipo_clima.upper() if tipo_clima
                        else alerta_obj.get("titulo", "ALERTA")[:25])
        nivel_label = nivel_alerta.upper() if nivel_alerta else ""
    elif nivel_productivo == "operativo":
        # Operativo: lenguaje más medido (Frío en lugar de FRÍO CRÍTICO)
        tipo_label = (tipo_clima.capitalize() if tipo_clima
                        else "Manejo")
        nivel_label = "operativo"
    else:
        # Atención: tono de monitoreo, sin urgencia
        tipo_label = (tipo_clima.capitalize() if tipo_clima
                        else "Manejo")
        nivel_label = "sostenido"
    cond = (alerta_obj.get("descripcion", "") or "").strip()
    cond = cond.replace("Temperatura ", "T° ").strip()
    if len(cond) > 60:
        cond = cond[:57] + "..."
    tipo_nivel = f"{tipo_label} {nivel_label}".strip()
    cabecera_secundaria = f"{tipo_nivel} · {cond}" if cond else tipo_nivel

    acciones_raw = alerta_obj.get("acciones", []) or []
    inmediatas = [a for a in acciones_raw if "INMEDIATA" in a]
    operativas = [a for a in acciones_raw if "OPERATIVA" in a]
    nutricionales = [a for a in acciones_raw if "NUTRICIONAL" in a]

    def _limpiar(linea: str) -> str:
        for p in ("⚡ INMEDIATA: ", "🔧 OPERATIVA: ", "🌾 NUTRICIONAL: "):
            if linea.startswith(p):
                linea = linea[len(p):]
                break
        if len(linea) > 80:
            linea = linea[:77] + "..."
        return linea

    bullets = []
    for it in inmediatas[:2]:
        bullets.append(f"• {_limpiar(it)}")
    for it in operativas[:1]:
        bullets.append(f"• {_limpiar(it)}")
    for it in nutricionales[:1]:
        bullets.append(f"• {_limpiar(it)}")

    # Frase LLM corta de impacto — reemplaza el genérico "Riesgo:" por
    # algo específico al lote y a las condiciones. Si falla, no rompe:
    # el WhatsApp sale igual sin la frase.
    frase_impacto = None
    try:
        from src.ai_analisis_semanal import generar_whatsapp_llm
        # Combinar datos del contexto de la alerta (pico del evento)
        # con el clima ACTUAL (datos del día del cliente). Si el motor
        # predictivo no devolvió temp_max/viento (None), usamos lo que
        # el clima actual reporte así el LLM nunca se queda sin datos.
        ctx_alerta = alerta_obj.get("_contexto", {}) or {}
        clima_act = datos.get("clima_actual", {}) or {}
        ctx_clima_wa = {
            "temperatura": (ctx_alerta.get("temp_max")
                            or clima_act.get("temp_c")),
            "min_nocturna": (ctx_alerta.get("t_min")
                              or clima_act.get("temp_c")),
            "viento_kmh": (ctx_alerta.get("viento_kmh")
                            or clima_act.get("viento_kmh") or 0),
            "lluvia_mm": (ctx_alerta.get("lluvia_mm")
                           or clima_act.get("lluvia_mm") or 0),
            "humedad_pct": clima_act.get("humedad_pct") or 0,
            "thi": clima_act.get("thi") or 0,
        }
        # Calcular si el evento ocurre hoy o en X días
        from datetime import datetime as _dt
        fecha_evt = (alerta_obj.get("_contexto", {}) or {}).get("fecha", "")
        dias_hasta_evt = 0
        ocurre_hoy_wa = True
        if fecha_evt:
            try:
                f_e = _dt.strptime(fecha_evt, "%Y-%m-%d").date()
                dias_hasta_evt = (f_e - _dt.now().date()).days
                if dias_hasta_evt > 0:
                    ocurre_hoy_wa = False
                else:
                    dias_hasta_evt = 0
            except (ValueError, TypeError):
                pass
        etapa_wa = datos.get("etapa_evento", "inicio")
        dias_prev_wa = datos.get("dias_alerta_previos", 0)

        # Calcular impacto productivo (solo si es frío y tenemos peso
        # del lote). Le pasamos los rangos al LLM como dato a citar.
        impacto_wa_txt = None
        impacto_wa_dict = None  # Para el auditor post-LLM
        if (tipo_clima or "").lower() == "frio":
            try:
                from src.impacto_productivo import (
                    estimar_impacto_frio as _imp_fn,
                    formato_impacto_texto as _imp_fmt,
                )
                _peso_l = lote_obj.get("peso_promedio_kg")
                if _peso_l and _peso_l > 0:
                    _imp_wa = _imp_fn(
                        peso_kg=_peso_l,
                        categoria=lote_obj.get("categoria", ""),
                        raza=lote_obj.get("raza", ""),
                        t_min_c=ctx_clima_wa.get("min_nocturna"),
                        viento_kmh=ctx_clima_wa.get("viento_kmh"),
                        humedad_pct=ctx_clima_wa.get("humedad_pct"),
                        barro=bool(ctx_alerta.get("barro")),
                        pelaje_mojado=(
                            (ctx_clima_wa.get("lluvia_mm") or 0) > 5
                        ),
                        dias_evento=max(1, dias_prev_wa + 1),
                        cantidad=lote_obj.get("cantidad_animales"),
                        adpv_objetivo_kg=lote_obj.get("adpv_objetivo_kg"),
                        energia_dieta_mcal_em_kg_ms=lote_obj.get(
                            "energia_dieta_mcal_em_kg_ms"
                        ),
                    )
                    if _imp_wa:
                        impacto_wa_txt = _imp_fmt(_imp_wa)
                        impacto_wa_dict = _imp_wa
            except Exception:
                impacto_wa_txt = None
                impacto_wa_dict = None

        frase_impacto = generar_whatsapp_llm(
            cliente=cliente,
            tipo=tipo_clima or "",
            nivel=nivel_alerta or nivel_productivo,
            clima=ctx_clima_wa,
            categoria=cat_corto or "",
            etapa=etapa_wa,
            dias_alerta_previos=dias_prev_wa,
            ocurre_hoy=ocurre_hoy_wa,
            dias_hasta_evento=dias_hasta_evt,
            impacto_productivo_txt=impacto_wa_txt,
        )
        # Auditar la frase del WhatsApp por si el LLM se equivocó
        # con el total del lote (mismo sesgo que en el email).
        if frase_impacto and impacto_wa_dict:
            try:
                from src.impacto_productivo import auditar_texto_llm
                frase_impacto = auditar_texto_llm(
                    frase_impacto, impacto_wa_dict,
                )
            except Exception:
                pass
    except Exception:
        frase_impacto = None

    lineas = [cabecera_principal, cabecera_secundaria]
    if frase_impacto:
        lineas.append("")
        lineas.append(f"_{frase_impacto}_")
    lineas += bullets
    lineas.append("_Detalle en el email._")
    wa_text = "\n".join(lineas)

    # Dedup por destinatario (sumamos al hash el destino para que cada
    # contacto reciba su WhatsApp aunque sean del mismo cliente).
    # Sufijo `tipo` para que el WhatsApp de la mañana NO bloquee el de la tarde.
    clave_wa = wa.clave_dedup(
        cliente["id"], lote_obj.get("lote", ""),
        alerta_obj.get("severidad", "info"),
        f"{whatsapp_dest}|{tipo}|{alerta_obj.get('titulo', '')}",
    )
    if not force and db.whatsapp_ya_enviado(clave_wa, ventana_horas=12):
        log.info(f"  {nombre_dest}: WhatsApp ya enviado hoy. Skip.")
        return enviados, errores
    if force:
        log.info(f"  {nombre_dest}: --force activado, "
                  f"ignorando dedup de WhatsApp.")

    if dry_run:
        log.info(f"  [DRY-RUN whatsapp] -> {whatsapp_dest}: "
                  f"{alerta_obj.get('titulo', '')[:60]}")
        return enviados + 1, errores

    ok, msg = wa.enviar_texto(cfg_wa, whatsapp_dest, wa_text)
    if ok:
        log.info(f"  ✓ whatsapp -> {whatsapp_dest} ({nombre_dest})")
        db.registrar_whatsapp_enviado(
            cliente["id"], whatsapp_dest, clave_wa,
            wa_text, "enviada", "",
        )
        enviados += 1
    else:
        log.error(f"  ✗ whatsapp -> {whatsapp_dest}: {msg}")
        db.registrar_whatsapp_enviado(
            cliente["id"], whatsapp_dest, clave_wa,
            wa_text, "error", msg,
        )
        errores += 1
    return enviados, errores


def enviar_alertas_a_cliente(cfg: dict, datos: dict, log: logging.Logger,
                               dry_run: bool = False,
                               tipo: str = "diaria",
                               asunto_override: str = None,
                               cabecera_wa_override: str = None,
                               nivel_productivo: str = "critico",
                               force: bool = False) -> tuple:
    """Envía email + WhatsApp a TODOS los contactos del cliente.

    Itera sobre el contacto principal (productor) + extras (encargado,
    comedero, etc.). Cada uno recibe su mensaje de bienvenida la
    primera vez, y después las alertas regulares.

    `tipo` separa ventanas (diaria/tarde) en el dedup.
    `asunto_override` y `cabecera_wa_override` permiten al cron de la
    tarde personalizar el copy del email y el WhatsApp.
    """
    from src import whatsapp as wa

    cliente = datos["cliente"]
    nombre = cliente.get("nombre", "")

    destinatarios = db.listar_destinatarios(cliente)
    if not destinatarios:
        return 0, 0

    cfg_wa = wa.cargar_config()
    ok_cfg_wa, _ = wa.config_valida(cfg_wa)

    enviados = 0
    errores = 0

    for d in destinatarios:
        # Email
        if d.get("email") and d.get("alertas_email_activas", 1):
            e_n, err_n = _enviar_email_a_destinatario(
                cfg, cliente, d, datos, log, dry_run,
                tipo=tipo, asunto_override=asunto_override,
                force=force,
            )
            enviados += e_n
            errores += err_n
        # WhatsApp — solo si nivel productivo es operativo o crítico.
        # Atención y normal no disparan WhatsApp puntual (evita
        # fatiga de alertas). --force salta esa filosofía para pruebas.
        if (d.get("whatsapp") and d.get("alertas_whatsapp_activas", 1)
                and ok_cfg_wa
                and (nivel_productivo in ("operativo", "critico")
                     or force)):
            e_n, err_n = _enviar_whatsapp_a_destinatario(
                cfg_wa, cliente, d, datos, log, dry_run,
                tipo=tipo, cabecera_override=cabecera_wa_override,
                nivel_productivo=nivel_productivo,
                force=force,
            )
            enviados += e_n
            errores += err_n

    return enviados, errores


def enviar_digest_admin(cfg: dict, todos_datos: list, admin_email: str,
                         log: logging.Logger, dry_run: bool = False) -> tuple:
    """Envía un digest agregado al admin (Mauricio) con resumen de TODOS los clientes."""
    if not admin_email:
        return 0, 0
    fecha = datetime.now().strftime("%d/%m/%Y")
    fecha_db = datetime.now().strftime("%Y-%m-%d")

    if db.alerta_ya_enviada_hoy(None, admin_email, fecha_db):
        log.info(f"  Digest admin ya enviado hoy. Skip.")
        return 0, 0

    n_clientes = len(todos_datos)
    # Conteos por NIVEL PRODUCTIVO (no severidad climática raw).
    cnt_nivel = {"critico": 0, "operativo": 0, "atencion": 0, "normal": 0}
    for d in todos_datos:
        nivel = d.get("nivel_productivo", "normal")
        cnt_nivel[nivel] = cnt_nivel.get(nivel, 0) + 1
    n_con_alertas = (cnt_nivel["critico"] + cnt_nivel["operativo"]
                       + cnt_nivel["atencion"])
    total_criticos = cnt_nivel["critico"]
    total_operativos = cnt_nivel["operativo"]
    total_atencion = cnt_nivel["atencion"]

    # Subject según el peor nivel productivo del día
    if total_criticos > 0:
        subject = (
            f"🔴 Resumen diario — {fecha} — "
            f"{total_criticos} cliente(s) con riesgo crítico"
        )
    elif total_operativos > 0:
        subject = (
            f"🟠 Resumen diario — {fecha} — "
            f"{total_operativos} cliente(s) con riesgo operativo"
        )
    elif total_atencion > 0:
        subject = (
            f"🟡 Resumen diario — {fecha} — "
            f"{total_atencion} cliente(s) de atención leve"
        )
    else:
        subject = (
            f"🟢 Resumen diario — {fecha} — sin riesgos previstos"
        )

    def _fmt_num(v):
        """Formato sin decimales para valores numéricos; '—' si no hay."""
        if v is None or v == "":
            return "—"
        try:
            return f"{float(v):.0f}"
        except (TypeError, ValueError):
            return str(v)

    secciones = []
    for d in todos_datos:
        if not d["alertas_por_lote"]:
            continue
        c = d["cliente"]
        ca = d["clima_actual"] or {}
        bloques = []
        for l in d["alertas_por_lote"]:
            for a in l["alertas"]:
                bloques.append(ae._alerta_html(a))
        temp_s = _fmt_num(ca.get("temp_c"))
        hum_s = _fmt_num(ca.get("humedad_pct"))
        thi_s = _fmt_num(ca.get("thi"))
        secciones.append(f"""
        <div style="margin:18px 0; padding:14px; background:#FAFAFA; border-radius:6px;
                    border-left:3px solid {ae.COLOR_LIMA};">
          <div style="font-size:15px; color:{ae.COLOR_VERDE}; font-weight:600;">
            {c.get('nombre')} — {c.get('establecimiento') or c.get('localidad','')}
          </div>
          <div style="font-size:12px; color:#888; margin-bottom:8px;">
            🌡️ {temp_s}°C &nbsp; 💧 {hum_s}%
            &nbsp; THI {thi_s} {ca.get('thi_estado','')}
          </div>
          {''.join(bloques)}
        </div>
        """)

    if not secciones:
        secciones.append(f"""
        <div style="padding:24px; background:#F0F8E8; text-align:center;
                    color:{ae.COLOR_VERDE}; border-radius:6px;">
          ✅ Sin alertas en ningún cliente. Clima dentro de rangos normales.
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html><body style="margin:0; padding:0; background:#F4F4F4; font-family:Arial,sans-serif;">
  <div style="max-width:700px; margin:0 auto; background:white;">
    <div style="background:{ae.COLOR_VERDE}; padding:18px 24px; color:white;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td><img src="cid:hms-logo" height="44"></td>
          <td style="text-align:right; color:white;">
            <div style="font-size:13px; opacity:0.85;">{fecha}</div>
            <div style="font-size:18px; font-weight:600;">Resumen diario — todos los clientes</div>
          </td>
        </tr>
      </table>
    </div>
    <div style="padding:24px;">
      <div style="display:flex; gap:10px; margin-bottom:18px;">
        <div style="flex:1; padding:14px; background:#F5F5F5; border-radius:6px; text-align:center;">
          <div style="font-size:24px; font-weight:600; color:{ae.COLOR_VERDE};">{n_clientes}</div>
          <div style="font-size:11px; color:#888;">CLIENTES</div>
        </div>
        <div style="flex:1; padding:14px; background:#FFEBEE; border-radius:6px; text-align:center;">
          <div style="font-size:24px; font-weight:600; color:{ae.COLOR_ALERTA_CRITICA};">{total_criticos}</div>
          <div style="font-size:11px; color:#888;">🔴 CRÍTICOS</div>
        </div>
        <div style="flex:1; padding:14px; background:#FFF3E0; border-radius:6px; text-align:center;">
          <div style="font-size:24px; font-weight:600; color:{ae.COLOR_ALERTA_WARNING};">{total_operativos}</div>
          <div style="font-size:11px; color:#888;">🟠 OPERATIVOS</div>
        </div>
        <div style="flex:1; padding:14px; background:#FFFBE6; border-radius:6px; text-align:center;">
          <div style="font-size:24px; font-weight:600; color:#9A7B00;">{total_atencion}</div>
          <div style="font-size:11px; color:#888;">🟡 ATENCIÓN</div>
        </div>
      </div>
      {''.join(secciones)}
    </div>
    <div style="background:{ae.COLOR_VERDE}; padding:12px 24px; color:white;
                font-size:11px; text-align:center;">
      <strong>HMS Nutrición Animal</strong> — Sistema autónomo de monitoreo<br>
      mauricio@hmsnutricionanimal.com.ar
    </div>
  </div>
</body></html>"""

    text_lines = [f"Resumen diario {fecha}", "",
                  f"Clientes monitoreados: {n_clientes}",
                  f"🔴 Riesgo crítico:   {total_criticos}",
                  f"🟠 Riesgo operativo: {total_operativos}",
                  f"🟡 Atención leve:    {total_atencion}",
                  ""]
    for d in todos_datos:
        if not d["alertas_por_lote"]:
            continue
        c = d["cliente"]
        text_lines.append(f"=== {c.get('nombre')} — {c.get('localidad','')} ===")
        for l in d["alertas_por_lote"]:
            text_lines.append(f"  Lote {l['lote']} ({l['categoria']}):")
            for a in l["alertas"]:
                text_lines.append(f"    {ae._alerta_texto(a)}")
        text_lines.append("")

    if dry_run:
        log.info(f"  [DRY-RUN] -> admin {admin_email}: {subject}")
        return 1, 0

    # El digest YA va al admin como destinatario principal; pasamos
    # con_bcc_admin=False para que no se duplique en BCC.
    ok, msg = ae.enviar_email(cfg, [admin_email], subject, html,
                                "\n".join(text_lines),
                                con_bcc_admin=False)
    total_alertas = total_criticos + total_operativos + total_atencion
    if ok:
        log.info(f"  ✓ -> admin {admin_email}")
        db.registrar_alerta_enviada(
            fecha_db, None, admin_email, subject,
            total_alertas, "enviada", "",
        )
        return 1, 0
    else:
        log.error(f"  ✗ -> admin {admin_email}: {msg}")
        db.registrar_alerta_enviada(
            fecha_db, None, admin_email, subject,
            total_alertas, "error", msg,
        )
        return 0, 1


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="No envía mails, solo muestra qué haría")
    parser.add_argument("--solo-cliente", default=None,
                         help="Solo procesa este cliente por nombre")
    parser.add_argument("--admin-email", default=None,
                         help="Email del admin (override config)")
    parser.add_argument(
        "--force", action="store_true",
        help="Saltar el dedup y re-enviar aunque ya se haya mandado hoy.",
    )
    args = parser.parse_args()

    log = setup_logging()

    # Lock global: si otro proceso del mismo script ya está corriendo
    # (ej. launchd disparó un job atrasado al despertar la Mac y a la vez
    # el cron rescate), abortamos silenciosamente para evitar envíos
    # duplicados. La razón está en data/logs/.
    lock_fd = adquirir_lock_proceso("alertas_diarias")
    if lock_fd is None:
        log.info("=== ALERTAS DIARIAS — abortado: otra instancia ya corre ===")
        return 0

    log.info("=== ALERTAS DIARIAS — INICIO ===")

    db.init_db()

    cfg = ae.cargar_config_smtp()
    ok, err = ae.config_valida(cfg)
    if not ok and not args.dry_run:
        log.error(f"Config SMTP inválida: {err}")
        log.error("Configurá SMTP en la pestaña Configuración de la app.")
        return 1

    # ───────── PROCESAR BAJAS ANTES DE MANDAR NUEVAS ALERTAS ─────────
    # Lee la bandeja IMAP, busca respuestas con "BAJA" y desactiva
    # alertas para los clientes que pidieron baja.
    if not args.dry_run and ok:
        try:
            n_bajas, emails_dados_baja = procesar_bajas_pendientes(cfg)
            if n_bajas > 0:
                log.info(f"  ✓ {n_bajas} baja(s) procesada(s): "
                          f"{', '.join(emails_dados_baja)}")
            else:
                log.info("  No hay pedidos de baja pendientes.")
        except Exception as e:
            log.warning(f"  Error al procesar bajas: {e}")

    admin_email = args.admin_email or (cfg or {}).get("admin_email", "") \
        or "mauricio@hmsnutricionanimal.com.ar"

    clientes = db.listar_clientes()
    if args.solo_cliente:
        clientes = [c for c in clientes
                    if args.solo_cliente.lower() in c["nombre"].lower()]
    log.info(f"Clientes a procesar: {len(clientes)}")

    todos_datos = []
    enviados = 0
    errores = 0

    # Día de semana: 0=lunes, 6=domingo. Los domingos solo mandamos
    # alertas CRÍTICAS al cliente (no warning ni preventiva). El clima
    # crítico no espera al lunes, pero las alertas moderadas sí.
    es_domingo = datetime.now().weekday() == 6

    for c in clientes:
        log.info(f"Cliente: {c['nombre']}")
        try:
            datos = calcular_alertas_cliente(c, log)
            todos_datos.append(datos)

            # Coherencia con el reporte semanal: usar el MISMO clasificador
            # de nivel productivo. La alerta diaria SOLO dispara para
            # operativo o crítico; nunca para atención (eso queda en el
            # resumen del lunes para no spamear).
            nivel_max = _nivel_productivo_maximo(datos)
            datos["nivel_productivo"] = nivel_max

            # Detectar etapa del evento usando historial de envíos
            fecha_hoy_iso = datetime.now().strftime("%Y-%m-%d")
            dias_previos = db.dias_alerta_consecutivos_previos(
                c["id"], fecha_hoy_iso,
            )
            etapa = clasificar_etapa_evento(nivel_max, dias_previos)
            datos["etapa_evento"] = etapa
            datos["dias_alerta_previos"] = dias_previos

            log.info(f"  {c['nombre']}: nivel productivo del día = "
                      f"{nivel_max} · etapa = {etapa} "
                      f"(días previos con alerta: {dias_previos})")

            # Domingos: solo crítica dispara envío al cliente
            if es_domingo:
                manda_a_cliente = (nivel_max == "critico")
                if nivel_max == "operativo":
                    log.info(f"  {c['nombre']}: domingo + operativo "
                              f"→ posterga (no es crítico).")
            else:
                manda_a_cliente = nivel_max in ("operativo", "critico")
                if nivel_max == "atencion":
                    log.info(f"  {c['nombre']}: nivel atención solamente "
                              f"→ no se manda diaria (queda en resumen "
                              f"del lunes).")

            # --force también ignora la filosofía silencio (útil para
            # pruebas manuales, ej: testear el bloque LLM del email diario
            # aunque el cliente esté en nivel atención hoy).
            if args.force and not manda_a_cliente and nivel_max != "normal":
                log.info(f"  {c['nombre']}: --force activado, ignorando "
                          f"filosofía silencio (nivel={nivel_max}).")
                manda_a_cliente = True

            if manda_a_cliente and not args.dry_run:
                e, err_n = enviar_alertas_a_cliente(
                    cfg, datos, log,
                    dry_run=args.dry_run,
                    nivel_productivo=nivel_max,
                    force=args.force,
                )
                enviados += e
                errores += err_n
            elif manda_a_cliente:
                e, err_n = enviar_alertas_a_cliente(
                    cfg, datos, log,
                    dry_run=True,
                    nivel_productivo=nivel_max,
                    force=args.force,
                )
                enviados += e
            else:
                log.info(f"  {c['nombre']}: sin riesgo productivo "
                          f"(nivel={nivel_max}). No se manda al cliente.")
        except Exception as e:
            log.exception(f"  Error procesando {c['nombre']}: {e}")
            errores += 1

    # Digest al admin — pero solo si tuvimos al menos un cliente OK.
    # Si TODOS los clientes fallaron por error de API (Open-Meteo caído),
    # no tiene sentido mandar un digest vacío; mejor que el cron reintente.
    n_total = len(todos_datos)
    n_ok = sum(1 for d in todos_datos if not d.get("error_api"))
    n_api_error = n_total - n_ok

    if n_total > 0 and n_ok == 0:
        log.error(
            f"Todos los clientes ({n_total}) fallaron por error de API "
            f"(Open-Meteo). No se manda digest vacío. El próximo run del "
            f"cron lo reintentará."
        )
        errores += n_api_error
    elif admin_email:
        if n_api_error > 0:
            log.warning(
                f"Atención: {n_api_error} cliente(s) con error de API. "
                f"El digest va con {n_ok} cliente(s) OK."
            )
        log.info(f"Enviando digest al admin: {admin_email}")
        try:
            e, err_n = enviar_digest_admin(cfg or {}, todos_datos, admin_email,
                                             log, dry_run=args.dry_run)
            enviados += e
            errores += err_n
        except Exception as e:
            log.exception(f"  Error enviando digest: {e}")
            errores += 1

    # WhatsApp resumen matinal al admin (si hay config WhatsApp).
    # Mismo criterio que con el digest: si todos los clientes fallaron por
    # error de API, no mandar el resumen vacío.
    cfg_wa = wa.cargar_config()
    ok_wa, _ = wa.config_valida(cfg_wa)
    if ok_wa and not (n_total > 0 and n_ok == 0):
        admin_phone = (cfg_wa or {}).get("admin_phone", "")
        if admin_phone:
            log.info(f"Enviando resumen WhatsApp al admin: {admin_phone}")
            try:
                # Conteos por NIVEL PRODUCTIVO (coherente con email digest)
                n_clientes = len(todos_datos)
                n_criticos = sum(1 for d in todos_datos
                                    if d.get("nivel_productivo") == "critico")
                n_operativos = sum(1 for d in todos_datos
                                       if d.get("nivel_productivo") == "operativo")
                n_atencion = sum(1 for d in todos_datos
                                    if d.get("nivel_productivo") == "atencion")

                # Top alertas: priorizar por nivel productivo del cliente
                rank_nivel = {"critico": 0, "operativo": 1,
                              "atencion": 2, "normal": 3}
                top = []
                for d in todos_datos:
                    nivel_d = d.get("nivel_productivo", "normal")
                    if nivel_d == "normal":
                        continue
                    # Tomar la peor alerta del cliente
                    peor_titulo = ""
                    for l in d["alertas_por_lote"]:
                        for a in l["alertas"]:
                            if a.get("titulo"):
                                peor_titulo = a["titulo"]
                                break
                        if peor_titulo:
                            break
                    top.append({
                        "cliente": d["cliente"]["nombre"],
                        "nivel": nivel_d,
                        "titulo": peor_titulo or nivel_d.capitalize(),
                    })
                top.sort(key=lambda x: rank_nivel.get(x["nivel"], 9))

                if args.dry_run:
                    log.info(f"  [DRY-RUN WA] -> {admin_phone}: "
                              f"{n_clientes} clientes, {n_criticos} críticos, "
                              f"{n_operativos} operativos, "
                              f"{n_atencion} atención")
                    enviados += 1
                else:
                    msg = wa.componer_resumen_admin(
                        n_clientes, n_criticos, n_operativos, top,
                        n_atencion=n_atencion,
                    )
                    ok, info = wa.enviar_texto(cfg_wa, admin_phone, msg)
                    if not ok and "ventana 24hs" in info.lower():
                        ok, info = wa.enviar_resumen_diario(
                            cfg_wa, admin_phone, n_clientes,
                            n_criticos, n_operativos,
                        )
                    if ok:
                        log.info(f"  ✓ WhatsApp admin: {info}")
                        enviados += 1
                    else:
                        log.error(f"  ✗ WhatsApp admin: {info}")
                        errores += 1
            except Exception as e:
                log.exception(f"  Error WhatsApp admin: {e}")
                errores += 1

    # ════════════════════════════════════════════════════════════
    # ALERTAS DE STOCK BAJO AL CLIENTE
    # ════════════════════════════════════════════════════════════
    # Después de las alertas climáticas, escaneamos los clientes con
    # productos HMS por agotarse (≤14 días) y les avisamos por email
    # y WhatsApp. Dedup de 3 días por cliente (no spamear).
    try:
        from src.stock_producto import clientes_con_stock_bajo
        # NOTA: NO re-importar 'wa' acá. El import top-level (línea 36)
        # ya lo provee. Si se re-importa adentro de main() Python lo
        # trata como variable local en TODA la función → UnboundLocalError
        # en cualquier referencia previa a 'wa' dentro de main.
        log.info("--- Chequeo de stock bajo ---")
        stocks_bajos = clientes_con_stock_bajo(umbral_dias=14)
        log.info(f"  Clientes con stock bajo: {len(stocks_bajos)}")
        fecha_db = datetime.now().strftime("%Y-%m-%d")
        for item in stocks_bajos:
            cli = item["cliente"]
            productos = item["productos"]
            nombre_cli = cli["nombre"]

            # Dedup: si ya se mandó alerta de stock al cliente en los
            # últimos 3 días, saltar
            try:
                with db.get_conn() as conn:
                    r = conn.execute(
                        """SELECT COUNT(*) AS n
                           FROM alertas_enviadas
                           WHERE cliente_id = ?
                             AND tipo = 'stock'
                             AND date(fecha) >= date(?, '-3 days')""",
                        (cli["id"], fecha_db),
                    ).fetchone()
                    ya_enviada_reciente = (r and r["n"] > 0)
            except Exception:
                ya_enviada_reciente = False

            if ya_enviada_reciente and not args.force:
                log.info(
                    f"  {nombre_cli}: alerta stock ya enviada en "
                    f"últimos 3 días. Skip."
                )
                continue

            # Buscar contactos del cliente (los mismos que reciben
            # alertas climáticas — usamos su email y WhatsApp).
            contactos = db.listar_destinatarios(cli)
            if not contactos:
                continue
            cfg_wa = wa.cargar_config() or {}

            envio_ok_alguno = False
            for contacto in contactos:
                email_dest = (contacto.get("email") or "").strip()
                wa_dest = (contacto.get("whatsapp") or "").strip()
                # Email
                if (email_dest and contacto.get(
                        "alertas_email_activas", 1)):
                    try:
                        s, h, t = ae.componer_alerta_stock_cliente(
                            cli, contacto, productos,
                        )
                        if args.dry_run:
                            log.info(
                                f"  [DRY-RUN stock email] -> "
                                f"{email_dest}"
                            )
                            envio_ok_alguno = True
                        else:
                            ok, msg = ae.enviar_email(
                                cfg, [email_dest], s, h, t,
                            )
                            if ok:
                                log.info(
                                    f"  ✓ stock email -> "
                                    f"{email_dest} ({nombre_cli})"
                                )
                                envio_ok_alguno = True
                                enviados += 1
                            else:
                                log.warning(
                                    f"  ⚠ stock email {email_dest}: "
                                    f"{msg}"
                                )
                                errores += 1
                    except Exception as e:
                        log.warning(f"  Error stock email: {e}")
                        errores += 1
                # WhatsApp
                if (wa_dest and contacto.get(
                        "alertas_whatsapp_activas", 1)):
                    try:
                        wa_msg = wa.componer_alerta_stock_cliente(
                            cli, productos,
                        )
                        if args.dry_run:
                            log.info(
                                f"  [DRY-RUN stock wa] -> {wa_dest}"
                            )
                            envio_ok_alguno = True
                        else:
                            ok_wa, info_wa = wa.enviar_alerta_critica(
                                cfg_wa, wa_dest, wa_msg,
                            )
                            if ok_wa:
                                log.info(
                                    f"  ✓ stock wa -> {wa_dest} "
                                    f"({nombre_cli})"
                                )
                                envio_ok_alguno = True
                                enviados += 1
                            else:
                                log.warning(
                                    f"  ⚠ stock wa {wa_dest}: "
                                    f"{info_wa}"
                                )
                                errores += 1
                    except Exception as e:
                        log.warning(f"  Error stock wa: {e}")
                        errores += 1

            # Registrar el envío en alertas_enviadas para el dedup
            if envio_ok_alguno and not args.dry_run:
                try:
                    db.registrar_alerta_enviada(
                        fecha_db, cli["id"],
                        cli.get("email") or "—",
                        f"Stock bajo: {len(productos)} producto(s)",
                        len(productos), "ok", "", tipo="stock",
                    )
                except Exception as e:
                    log.warning(
                        f"  No se pudo registrar dedup stock: {e}"
                    )
    except Exception as e:
        log.exception(f"Error en chequeo de stock bajo: {e}")

    # ════════════════════════════════════════════════════════════
    # ALERTAS DE FIN DE CARGA DEL SILOCOMEDERO
    # ════════════════════════════════════════════════════════════
    # Lotes con silocomedero cuya carga actual se agota en ≤1 día.
    # Es un aviso OPERATIVO (preparar la próxima mezcla), distinto
    # del aviso de stock (reposición de producto HMS). Por eso usa
    # su propio tipo de dedup ('silocomedero') y ventana corta de
    # 1 día (no querés perder ningún aviso de mezcla por dedup).
    try:
        from src.stock_producto import (
            lotes_silocomedero_proximos_agotamiento,
        )
        # 'wa' viene del import top-level (línea 36).
        log.info("--- Chequeo de fin de carga de silocomedero ---")
        silos_alerta = lotes_silocomedero_proximos_agotamiento(
            umbral_dias=1,
        )
        log.info(f"  Clientes con silo por agotarse: {len(silos_alerta)}")
        fecha_db = datetime.now().strftime("%Y-%m-%d")
        for item in silos_alerta:
            cli = item["cliente"]
            lotes_a = item["lotes"]
            nombre_cli = cli["nombre"]

            # Dedup: ya enviada en las últimas 24 horas
            try:
                with db.get_conn() as conn:
                    r = conn.execute(
                        """SELECT COUNT(*) AS n
                           FROM alertas_enviadas
                           WHERE cliente_id = ?
                             AND tipo = 'silocomedero'
                             AND date(fecha) >= date(?, '-1 days')""",
                        (cli["id"], fecha_db),
                    ).fetchone()
                    ya_enviada_reciente = (r and r["n"] > 0)
            except Exception:
                ya_enviada_reciente = False

            if ya_enviada_reciente and not args.force:
                log.info(
                    f"  {nombre_cli}: alerta silo ya enviada en "
                    f"últimas 24h. Skip."
                )
                continue

            contactos = db.listar_destinatarios(cli)
            if not contactos:
                continue
            cfg_wa = wa.cargar_config() or {}

            envio_ok_alguno = False
            for contacto in contactos:
                email_dest = (contacto.get("email") or "").strip()
                wa_dest = (contacto.get("whatsapp") or "").strip()
                # Email
                if (email_dest and contacto.get(
                        "alertas_email_activas", 1)):
                    try:
                        s, h, t = (
                            ae.componer_alerta_silocomedero_cliente(
                                cli, contacto, lotes_a,
                            )
                        )
                        if args.dry_run:
                            log.info(
                                f"  [DRY-RUN silo email] -> "
                                f"{email_dest}"
                            )
                            envio_ok_alguno = True
                        else:
                            ok, msg = ae.enviar_email(
                                cfg, [email_dest], s, h, t,
                            )
                            if ok:
                                log.info(
                                    f"  ✓ silo email -> "
                                    f"{email_dest} ({nombre_cli})"
                                )
                                envio_ok_alguno = True
                                enviados += 1
                            else:
                                log.warning(
                                    f"  ⚠ silo email {email_dest}: "
                                    f"{msg}"
                                )
                                errores += 1
                    except Exception as e:
                        log.warning(f"  Error silo email: {e}")
                        errores += 1
                # WhatsApp
                if (wa_dest and contacto.get(
                        "alertas_whatsapp_activas", 1)):
                    try:
                        wa_msg = (
                            wa.componer_alerta_silocomedero_cliente(
                                cli, lotes_a,
                            )
                        )
                        if args.dry_run:
                            log.info(
                                f"  [DRY-RUN silo wa] -> {wa_dest}"
                            )
                            envio_ok_alguno = True
                        else:
                            ok_wa, info_wa = (
                                wa.enviar_alerta_critica(
                                    cfg_wa, wa_dest, wa_msg,
                                )
                            )
                            if ok_wa:
                                log.info(
                                    f"  ✓ silo wa -> {wa_dest} "
                                    f"({nombre_cli})"
                                )
                                envio_ok_alguno = True
                                enviados += 1
                            else:
                                log.warning(
                                    f"  ⚠ silo wa {wa_dest}: "
                                    f"{info_wa}"
                                )
                                errores += 1
                    except Exception as e:
                        log.warning(f"  Error silo wa: {e}")
                        errores += 1

            if envio_ok_alguno and not args.dry_run:
                try:
                    db.registrar_alerta_enviada(
                        fecha_db, cli["id"],
                        cli.get("email") or "—",
                        f"Silocomedero: {len(lotes_a)} lote(s)",
                        len(lotes_a), "ok", "", tipo="silocomedero",
                    )
                except Exception as e:
                    log.warning(
                        f"  No se pudo registrar dedup silo: {e}"
                    )
    except Exception as e:
        log.exception(f"Error en chequeo de silocomedero: {e}")

    # ════════════════════════════════════════════════════════════
    # ALERTAS DE CAMBIO DE FASE EN PLAN DE ADAPTACIÓN
    # ════════════════════════════════════════════════════════════
    # Detecta lotes donde MAÑANA arranca una nueva dieta del plan de
    # adaptación. Se manda una sola vez por cambio (dedup ventana 24h,
    # tipo='cambio_fase').
    try:
        from src.stock_producto import lotes_con_cambio_fase_proximo
        # 'wa' viene del import top-level (línea 36).
        log.info("--- Chequeo de cambio de fase ---")
        cambios_pendientes = lotes_con_cambio_fase_proximo(
            dias_anticipo=1,
        )
        log.info(
            f"  Clientes con cambio de fase mañana: "
            f"{len(cambios_pendientes)}"
        )
        fecha_db = datetime.now().strftime("%Y-%m-%d")
        for item in cambios_pendientes:
            cli = item["cliente"]
            cambios = item["cambios"]
            nombre_cli = cli["nombre"]

            # Dedup: una alerta cada 24h por cambio
            try:
                with db.get_conn() as conn:
                    r = conn.execute(
                        """SELECT COUNT(*) AS n
                           FROM alertas_enviadas
                           WHERE cliente_id = ?
                             AND tipo = 'cambio_fase'
                             AND date(fecha) >= date(?, '-1 days')""",
                        (cli["id"], fecha_db),
                    ).fetchone()
                    ya_enviada_reciente = (r and r["n"] > 0)
            except Exception:
                ya_enviada_reciente = False

            if ya_enviada_reciente and not args.force:
                log.info(
                    f"  {nombre_cli}: alerta cambio fase ya enviada "
                    f"en últimas 24h. Skip."
                )
                continue

            contactos = db.listar_destinatarios(cli)
            if not contactos:
                continue
            cfg_wa = wa.cargar_config() or {}

            envio_ok_alguno = False
            for contacto in contactos:
                email_dest = (contacto.get("email") or "").strip()
                wa_dest = (contacto.get("whatsapp") or "").strip()
                # Email
                if (email_dest and contacto.get(
                        "alertas_email_activas", 1)):
                    try:
                        s, h, t = (
                            ae.componer_alerta_cambio_fase_cliente(
                                cli, contacto, cambios,
                            )
                        )
                        if args.dry_run:
                            log.info(
                                f"  [DRY-RUN cambio_fase email] -> "
                                f"{email_dest}"
                            )
                            envio_ok_alguno = True
                        else:
                            ok, msg = ae.enviar_email(
                                cfg, [email_dest], s, h, t,
                            )
                            if ok:
                                log.info(
                                    f"  ✓ cambio_fase email -> "
                                    f"{email_dest} ({nombre_cli})"
                                )
                                envio_ok_alguno = True
                                enviados += 1
                            else:
                                log.warning(
                                    f"  ⚠ cambio_fase email "
                                    f"{email_dest}: {msg}"
                                )
                                errores += 1
                    except Exception as e:
                        log.warning(
                            f"  Error cambio_fase email: {e}"
                        )
                        errores += 1
                # WhatsApp
                if (wa_dest and contacto.get(
                        "alertas_whatsapp_activas", 1)):
                    try:
                        wa_msg = (
                            wa.componer_alerta_cambio_fase_cliente(
                                cli, cambios,
                            )
                        )
                        if args.dry_run:
                            log.info(
                                f"  [DRY-RUN cambio_fase wa] -> "
                                f"{wa_dest}"
                            )
                            envio_ok_alguno = True
                        else:
                            ok_wa, info_wa = (
                                wa.enviar_alerta_critica(
                                    cfg_wa, wa_dest, wa_msg,
                                )
                            )
                            if ok_wa:
                                log.info(
                                    f"  ✓ cambio_fase wa -> "
                                    f"{wa_dest} ({nombre_cli})"
                                )
                                envio_ok_alguno = True
                                enviados += 1
                            else:
                                log.warning(
                                    f"  ⚠ cambio_fase wa {wa_dest}: "
                                    f"{info_wa}"
                                )
                                errores += 1
                    except Exception as e:
                        log.warning(
                            f"  Error cambio_fase wa: {e}"
                        )
                        errores += 1

            if envio_ok_alguno and not args.dry_run:
                try:
                    db.registrar_alerta_enviada(
                        fecha_db, cli["id"],
                        cli.get("email") or "—",
                        f"Cambio de fase: {len(cambios)} lote(s)",
                        len(cambios), "ok", "",
                        tipo="cambio_fase",
                    )
                except Exception as e:
                    log.warning(
                        f"  No se pudo registrar dedup "
                        f"cambio_fase: {e}"
                    )
    except Exception as e:
        log.exception(f"Error en chequeo de cambio de fase: {e}")

    log.info(f"=== FIN — Enviados: {enviados}, errores: {errores} ===")
    liberar_lock(lock_fd)
    return 0 if errores == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
