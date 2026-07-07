"""
Memoria persistente del agente IA.

Guarda correcciones, preferencias del asesor, valores típicos de la zona,
ingredientes locales con sus análisis reales, etc. Esa información se
inyecta automáticamente en el system prompt cada vez que se inicia una
conversación, así el agente "recuerda" entre sesiones.

Categorías de memoria:
  - "preferencia": cómo le gusta trabajar al asesor (formato, tono)
  - "correccion": correcciones técnicas que hizo el asesor a respuestas pasadas
  - "valor_local": valores típicos de la zona (silaje 28% MS, agua salina, etc.)
  - "ingrediente": composición real de un ingrediente que usa el asesor
  - "cliente": información persistente sobre un cliente
  - "manejo": características del establecimiento o sistema productivo
  - "general": cualquier otra cosa
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


MEMORIA_PATH = Path("data/agent_memory.json")


def _cargar() -> List[Dict]:
    """Lee la lista de memorias del disco. Si no existe, devuelve vacío."""
    if not MEMORIA_PATH.exists():
        return []
    try:
        with MEMORIA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _guardar(memorias: List[Dict]) -> None:
    """Persiste la lista de memorias."""
    MEMORIA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MEMORIA_PATH.open("w", encoding="utf-8") as f:
        json.dump(memorias, f, indent=2, ensure_ascii=False)


def listar_memorias(categoria: Optional[str] = None,
                     activas_solo: bool = True) -> List[Dict]:
    """Lista todas las memorias, opcionalmente filtradas por categoría."""
    memorias = _cargar()
    if activas_solo:
        memorias = [m for m in memorias if m.get("activa", True)]
    if categoria:
        memorias = [m for m in memorias if m.get("categoria") == categoria]
    return memorias


def agregar_memoria(texto: str, categoria: str = "general",
                     etiqueta: str = "") -> Dict:
    """Agrega una nueva memoria al sistema."""
    memorias = _cargar()
    nueva = {
        "id": max([m.get("id", 0) for m in memorias], default=0) + 1,
        "texto": texto.strip(),
        "categoria": categoria,
        "etiqueta": etiqueta.strip(),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "activa": True,
    }
    memorias.append(nueva)
    _guardar(memorias)
    return nueva


def actualizar_memoria(memoria_id: int, **campos) -> bool:
    """Actualiza campos de una memoria existente."""
    memorias = _cargar()
    for m in memorias:
        if m.get("id") == memoria_id:
            m.update(campos)
            _guardar(memorias)
            return True
    return False


def eliminar_memoria(memoria_id: int) -> bool:
    """Elimina una memoria del archivo."""
    memorias = _cargar()
    nuevas = [m for m in memorias if m.get("id") != memoria_id]
    if len(nuevas) == len(memorias):
        return False
    _guardar(nuevas)
    return True


def desactivar_memoria(memoria_id: int) -> bool:
    """Desactiva (sin borrar) una memoria — para no inyectarla, pero conservarla."""
    return actualizar_memoria(memoria_id, activa=False)


def reactivar_memoria(memoria_id: int) -> bool:
    return actualizar_memoria(memoria_id, activa=True)


# =====================================================================
# CONTEXTO INYECTABLE EN EL SYSTEM PROMPT
# =====================================================================

def construir_bloque_memoria() -> str:
    """Genera el texto que se inyecta en el system prompt con todas las
    memorias activas, agrupadas por categoría."""
    memorias = listar_memorias(activas_solo=True)
    if not memorias:
        return ""

    por_categoria: Dict[str, List[Dict]] = {}
    for m in memorias:
        cat = m.get("categoria", "general")
        por_categoria.setdefault(cat, []).append(m)

    titulos = {
        "preferencia": "Preferencias del asesor",
        "correccion": "Correcciones técnicas previas",
        "valor_local": "Valores típicos de la zona",
        "ingrediente": "Composición real de ingredientes locales",
        "cliente": "Información persistente de clientes",
        "manejo": "Características de manejo / sistema productivo",
        "general": "Notas generales",
    }

    lineas = [
        "═══════════════════════════════════════════════════════════════",
        "MEMORIA DEL ASESOR — conocimiento acumulado del usuario",
        "═══════════════════════════════════════════════════════════════",
        "Esta es información que el asesor te enseñó en sesiones anteriores. "
        "Tomalas como REGLAS o CRITERIOS aprendidos: cuando aparezca un "
        "tema relacionado, aplicá esto sin que el usuario lo tenga que repetir.",
        "",
    ]
    for cat in ["correccion", "valor_local", "ingrediente", "manejo",
                 "cliente", "preferencia", "general"]:
        if cat not in por_categoria:
            continue
        lineas.append(f"▸ {titulos.get(cat, cat).upper()}:")
        for m in por_categoria[cat]:
            etq = f" [{m['etiqueta']}]" if m.get("etiqueta") else ""
            lineas.append(f"   • {m['texto']}{etq}")
        lineas.append("")
    return "\n".join(lineas)
