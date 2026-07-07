"""Ficha clínica del lote — historia médica del paciente.

Convierte el stream de evaluaciones registradas en cada conversación
con el cliente en una historia clínica acumulativa del lote:

- Línea de tiempo: cada evaluación es una "consulta médica"
- Tally de mortandad por causa (¿es problema sanitario estructural?)
- Patrones detectados: síntomas recurrentes (¿se repite el "comedero
  vacío" 3 evaluaciones seguidas?)
- Diagnósticos activos: temas que requieren seguimiento
- Resumen clínico generado con IA periódicamente

Se accede desde la ficha del lote, en una sección dedicada.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# =====================================================================
# RECOPILACIÓN DE DATOS
# =====================================================================

@dataclass
class EvaluacionRegistrada:
    """Una evaluación parseada lista para mostrar."""
    rid: int
    fecha: str  # ISO YYYY-MM-DD HH:MM
    cliente: str
    tipo_contacto: str
    atendio: str
    # Datos clínicos clave
    aspecto_animales: str
    bajas: int
    causa_muerte: str
    enfermos: int
    ventas: int
    comedero: str
    heces: str
    agua: str
    cama: str
    reparos: str
    # Stock
    maiz_kg: float
    fg_kg: float
    silo_pct: int
    # Análisis
    resumen_semaforo: str  # 🔴/🟡/🟢
    n_sugerencias_urgentes: int
    n_sugerencias_atencion: int
    # Texto libre
    observaciones: str
    acciones_acordadas: str
    notas_md: str  # markdown completo por si lo quiere ver entero


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return default


def recopilar_evaluaciones_lote(
    lote_id: int, db_mod, limit: int = 30,
) -> List[EvaluacionRegistrada]:
    """Devuelve las evaluaciones registradas para un lote, más
    reciente primero.

    Args:
        lote_id: id del lote.
        db_mod: módulo src.database (pasado para no importar acá).
        limit: máximo de evaluaciones a devolver.

    Returns:
        Lista de EvaluacionRegistrada (puede estar vacía).
    """
    out: List[EvaluacionRegistrada] = []
    try:
        with db_mod.get_conn() as conn:
            # Buscamos los recordatorios completados del lote.
            # OJO: solo desde que se agregó la columna lote_id
            # tenemos esta relación directa. Para evaluaciones
            # viejas, fallback por cliente del lote.
            rows = conn.execute(
                """SELECT r.*, c.nombre AS cliente_nombre
                   FROM recordatorios_llamada r
                   JOIN clientes c ON c.id = r.cliente_id
                   WHERE r.estado = 'hecho'
                     AND (r.lote_id = ?
                          OR r.cliente_id IN (
                              SELECT cliente_id FROM lotes WHERE id = ?
                          ))
                   ORDER BY date(r.completado_en) DESC,
                            r.id DESC
                   LIMIT ?""",
                (lote_id, lote_id, limit),
            ).fetchall()
    except Exception:
        rows = []

    for r in rows:
        d = dict(r)
        # Parsear el JSON estructurado si existe
        ev_struct = {}
        raw_json = d.get("evaluacion_json") or ""
        if raw_json:
            try:
                ev_struct = json.loads(raw_json)
            except json.JSONDecodeError:
                ev_struct = {}

        # Si la evaluación NO tiene JSON estructurado (es vieja
        # o fue una conversación libre sin cuestionario), no
        # la incluimos en la ficha clínica — no podemos extraer
        # datos confiables del markdown libre.
        if not ev_struct:
            continue

        out.append(EvaluacionRegistrada(
            rid=int(d.get("id", 0)),
            fecha=(
                d.get("completado_en")
                or d.get("fecha_objetivo")
                or ""
            )[:16],
            cliente=d.get("cliente_nombre", ""),
            tipo_contacto=ev_struct.get("tipo_contacto", ""),
            atendio=ev_struct.get("atendio", ""),
            aspecto_animales=ev_struct.get("aspecto_animales", ""),
            bajas=_safe_int(ev_struct.get("bajas_48hs")),
            causa_muerte=ev_struct.get("causa_muerte", ""),
            enfermos=_safe_int(ev_struct.get("animales_enfermos")),
            ventas=_safe_int(ev_struct.get("ventas_48hs")),
            comedero=ev_struct.get("estado_comedero", ""),
            heces=ev_struct.get("heces", ""),
            agua=ev_struct.get("estado_agua", ""),
            cama=ev_struct.get("estado_cama", ""),
            reparos=ev_struct.get("estado_reparos", ""),
            maiz_kg=_safe_float(ev_struct.get("maiz_kg_disponible")),
            fg_kg=_safe_float(
                ev_struct.get("fibrogreen_kg_disponible")
            ),
            silo_pct=_safe_int(
                ev_struct.get("silo_nivel_pct"), -1,
            ),
            resumen_semaforo=ev_struct.get(
                "resumen_semaforo", "🟢",
            ),
            n_sugerencias_urgentes=_safe_int(
                ev_struct.get("n_sugerencias_urgentes")
            ),
            n_sugerencias_atencion=_safe_int(
                ev_struct.get("n_sugerencias_atencion")
            ),
            observaciones=ev_struct.get("observaciones", ""),
            acciones_acordadas=ev_struct.get(
                "acciones_acordadas", "",
            ),
            notas_md=d.get("notas_cierre", "") or "",
        ))
    return out


# =====================================================================
# AGREGADOS Y PATRONES
# =====================================================================

def tally_mortandad_por_causa(
    evals: List[EvaluacionRegistrada],
) -> Dict[str, int]:
    """Cuenta total de muertes agrupadas por causa."""
    counter: Counter = Counter()
    for e in evals:
        if e.bajas > 0:
            causa = e.causa_muerte or "Sin determinar"
            counter[causa] += e.bajas
    return dict(counter)


def total_ventas(evals: List[EvaluacionRegistrada]) -> int:
    return sum(e.ventas for e in evals)


def total_muertes(evals: List[EvaluacionRegistrada]) -> int:
    return sum(e.bajas for e in evals)


def detectar_patrones_sintomas(
    evals: List[EvaluacionRegistrada], n_recientes: int = 5,
) -> List[Dict[str, Any]]:
    """Detecta síntomas recurrentes en las últimas N evaluaciones.

    Si un mismo síntoma anormal aparece en >=50% de las últimas
    evaluaciones, lo marca como patrón.
    """
    if not evals:
        return []
    recientes = evals[:n_recientes]
    n = len(recientes)
    if n < 2:
        return []

    # Lo que es "anormal" en cada campo
    patrones: List[Dict[str, Any]] = []

    def _detectar(campo: str, label: str,
                  fn_es_anormal, sugerencia: str):
        casos = [
            getattr(e, campo) for e in recientes
            if fn_es_anormal(getattr(e, campo) or "")
        ]
        if len(casos) >= max(2, n // 2):
            patrones.append({
                "label": label,
                "frecuencia": (
                    f"{len(casos)}/{n} evaluaciones recientes"
                ),
                "sugerencia": sugerencia,
                "casos": casos,
            })

    _detectar(
        "comedero",
        "Comedero vacío sostenido",
        lambda v: "Vacío" in v,
        "Animal hambreado → ajustar oferta diaria al alza. "
        "Riesgo de acidosis por consumo desparejo.",
    )
    _detectar(
        "comedero",
        "Sobras de mezcla sostenidas",
        lambda v: "Sobra" in v,
        "Caída de consumo. Investigar: calor, mezcla en mal "
        "estado, problema sanitario subclínico.",
    )
    _detectar(
        "heces",
        "Heces alteradas sostenidas",
        lambda v: ("Pastosas" in v or "Líquidas" in v
                    or "diarrea" in v.lower()),
        "Posible acidosis subclínica recurrente. Revisar "
        "estructuralmente la proporción de Fibrogreen y la "
        "homogeneidad de la mezcla.",
    )
    _detectar(
        "agua",
        "Problemas de agua recurrentes",
        lambda v: ("Hielo" in v or "Sin agua" in v
                    or "Sucia" in v),
        "Problema estructural con el sistema de agua. Mover "
        "bebedero, instalar reparo del viento, evaluar caudal.",
    )
    _detectar(
        "cama",
        "Cama comprometida sostenida",
        lambda v: ("Embarrada" in v or "Sin cama" in v
                    or "Húmeda" in v),
        "Drenaje o reposición de cama deficitarios. Revisar "
        "pendiente del corral y régimen de reposición de paja.",
    )
    _detectar(
        "reparos",
        "Reparos insuficientes crónicos",
        lambda v: "Insuficientes" in v,
        "Inversión pendiente: cortavientos en oeste/suroeste. "
        "Cada evento de frío deja secuelas productivas.",
    )
    return patrones


def diagnosticos_activos(
    evals: List[EvaluacionRegistrada],
) -> List[Dict[str, Any]]:
    """Temas detectados en la ÚLTIMA evaluación que requieren
    seguimiento próximo. Si la última fue normal, no devuelve nada.
    """
    if not evals:
        return []
    ult = evals[0]
    activos: List[Dict[str, Any]] = []
    if ult.bajas > 0:
        activos.append({
            "label": f"💀 {ult.bajas} muerte(s) — {ult.causa_muerte}",
            "fecha_deteccion": ult.fecha[:10],
            "estado": "Pendiente investigación / próximo control",
        })
    if ult.enfermos > 0:
        activos.append({
            "label": f"🤒 {ult.enfermos} animal(es) enfermo(s)",
            "fecha_deteccion": ult.fecha[:10],
            "estado": "En seguimiento veterinario",
        })
    if "Decaídos" in ult.aspecto_animales or (
        "Apagados" in ult.aspecto_animales
    ):
        activos.append({
            "label": f"📉 Aspecto: {ult.aspecto_animales}",
            "fecha_deteccion": ult.fecha[:10],
            "estado": "Monitorear evolución",
        })
    if "Vacío" in ult.comedero:
        activos.append({
            "label": "🍽️ Comedero vacío — oferta corta",
            "fecha_deteccion": ult.fecha[:10],
            "estado": "Ajustar oferta diaria",
        })
    elif "Sobra TODO" in ult.comedero:
        activos.append({
            "label": "🚨 Animal no consume",
            "fecha_deteccion": ult.fecha[:10],
            "estado": "Investigación urgente",
        })
    if "Líquidas" in ult.heces or "diarrea" in ult.heces.lower():
        activos.append({
            "label": "💩 Diarrea",
            "fecha_deteccion": ult.fecha[:10],
            "estado": "Revisar acidosis / infeccioso",
        })
    return activos


# =====================================================================
# FICHA CLÍNICA COMPLETA
# =====================================================================

def armar_ficha_clinica_lote(
    lote_id: int, db_mod,
) -> Dict[str, Any]:
    """Recopila todo lo necesario para renderizar la ficha clínica.

    Returns:
        dict con:
        - lote_info: datos del paciente
        - evaluaciones: lista de EvaluacionRegistrada
        - tally_mortandad: {causa: total_muertes}
        - patrones: lista de patrones detectados
        - diagnosticos_activos: lista de items abiertos
        - total_muertes, total_ventas
    """
    lote = db_mod.obtener_lote(lote_id) or {}
    evals = recopilar_evaluaciones_lote(lote_id, db_mod, limit=30)
    return {
        "lote_info": lote,
        "evaluaciones": evals,
        "tally_mortandad": tally_mortandad_por_causa(evals),
        "patrones": detectar_patrones_sintomas(evals),
        "diagnosticos_activos": diagnosticos_activos(evals),
        "total_muertes": total_muertes(evals),
        "total_ventas": total_ventas(evals),
        "n_evaluaciones": len(evals),
    }


# =====================================================================
# RESUMEN CLÍNICO CON LLM
# =====================================================================

def generar_resumen_clinico_llm(
    ficha: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 600,
) -> Dict[str, Any]:
    """Pide a Claude un resumen clínico breve del lote.

    Estilo: como un médico que repasa la historia del paciente
    en 3-5 oraciones antes de la próxima consulta.
    """
    out = {"exito": False, "resumen_md": "", "error": ""}
    try:
        from anthropic import Anthropic
    except ImportError:
        out["error"] = "Falta paquete 'anthropic'."
        return out

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        out["error"] = "Sin ANTHROPIC_API_KEY."
        return out

    if not ficha.get("evaluaciones"):
        out["error"] = "Sin evaluaciones para resumir."
        return out

    lote_info = ficha.get("lote_info") or {}
    evals = ficha.get("evaluaciones", [])
    tally = ficha.get("tally_mortandad", {})
    patrones = ficha.get("patrones", [])
    activos = ficha.get("diagnosticos_activos", [])

    # Resumen de evaluaciones para el prompt (las 6 más recientes)
    evals_str = []
    for e in evals[:6]:
        evals_str.append(
            f"- {e.fecha[:10]} ({e.tipo_contacto}): "
            f"aspecto={e.aspecto_animales or '—'} · "
            f"comedero={e.comedero or '—'} · "
            f"heces={e.heces or '—'} · "
            f"bajas={e.bajas}"
            + (f" ({e.causa_muerte})" if e.causa_muerte else "")
            + f" · enfermos={e.enfermos}"
        )
    evals_block = "\n".join(evals_str)

    tally_str = ", ".join(
        f"{k}: {v}" for k, v in tally.items()
    ) if tally else "Sin mortandad registrada"

    patrones_str = (
        "\n".join(f"- {p['label']} ({p['frecuencia']})"
                   for p in patrones)
        if patrones else "Sin patrones recurrentes"
    )

    activos_str = (
        "\n".join(f"- {a['label']} ({a['estado']})"
                   for a in activos)
        if activos else "Sin diagnósticos activos"
    )

    # ─── Composición del system prompt ───
    # Filosofía única HMS + perfil "resumen_clinico"
    from . import perfiles_llm as _perfiles_llm
    system_prompt = _perfiles_llm.armar_system_prompt(
        "resumen_clinico",
    )

    # Bloque viejo deshabilitado (referencia histórica)
    _system_prompt_viejo = (
        "Sos Mauricio Suárez de HMS Nutrición Animal. Revisás la "
        "historia clínica del lote como un médico antes de la "
        "próxima consulta.\n\n"
        "Devolvé un resumen MUY breve (máximo 4 oraciones) en "
        "prosa criolla técnica con:\n"
        "1. Estado clínico general del lote (en qué momento "
        "productivo está, cómo viene)\n"
        "2. Si hay un patrón / hilo conductor entre las "
        "evaluaciones, mencionalo\n"
        "3. Qué es lo más importante a monitorear / resolver\n\n"
        "REGLAS:\n"
        "- No repitas datos puntuales — abstraé\n"
        "- No uses bullets, escribí en prosa\n"
        "- No inventes datos que no estén\n"
        "- Si no hay problemas serios, decilo en una línea\n"
        "- Si hay UN solo problema crítico, ese es el foco\n"
        "- Estilo: como cuando le contás a un colega 'cómo viene "
        "este lote'"
    )

    # Contexto unificado (ADG real, sub-consumo, fase del plan,
    # movimientos recientes) si está disponible — mismo bloque
    # que usa el análisis climático y el chat conversacional.
    ctx_unificado = ficha.get("contexto_unificado") or ""
    user_msg = (
        "=== DATOS DEL LOTE ===\n"
        f"Lote: {lote_info.get('identificador','—')} · "
        f"{lote_info.get('categoria','')} {lote_info.get('raza','')} "
        f"· {lote_info.get('cantidad_inicial', 0)} animales\n"
        f"Ingreso: {lote_info.get('fecha_ingreso','—')} · "
        f"PV ingreso: {lote_info.get('peso_ingreso_kg','?')} kg · "
        f"ADG objetivo: "
        f"{lote_info.get('adpv_objetivo_kg','?')} kg/día\n\n"
        f"=== ÚLTIMAS {len(evals[:6])} EVALUACIONES ===\n"
        f"{evals_block}\n\n"
        f"=== MORTANDAD ACUMULADA ===\n{tally_str}\n\n"
        f"=== PATRONES DETECTADOS ===\n{patrones_str}\n\n"
        f"=== DIAGNÓSTICOS ACTIVOS HOY ===\n{activos_str}\n"
        + (f"\n{ctx_unificado}\n" if ctx_unificado else "")
    )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        partes = []
        for block in resp.content:
            if hasattr(block, "text"):
                partes.append(block.text)
        out["resumen_md"] = "\n".join(partes).strip()
        out["exito"] = True
    except Exception as e:
        out["error"] = f"LLM falló: {e}"
    return out
