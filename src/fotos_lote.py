"""Gestión de fotos de inspección del lote.

Las fotos se cargan durante una consulta/entrevista (recordatorio_llamada)
y se guardan físicamente en:

    data/fotos_lote/<lote_id>/<recordatorio_id>/<timestamp>_<tipo>.<ext>

Las referencias se persisten en la tabla `fotos_lote` (ver
src/database.py). Este módulo expone:

  - guardar_archivo_subido(): toma un UploadedFile de Streamlit y lo
    deja en disco con nombre seguro.
  - directorio_fotos_consulta(): devuelve el Path donde van las fotos
    de una consulta puntual.
  - listar_fotos_categorizadas(): organiza las fotos por tipo para
    armar la galería o el anexo PDF.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import database as db


# Raíz donde viven todas las fotos. Está dentro de data/ (no de ~/Documents
# en sentido sensible), Python tiene FDA, así que no hay problema TCC.
FOTOS_ROOT = Path(__file__).resolve().parents[1] / "data" / "fotos_lote"


# Extensiones permitidas. Si el usuario sube HEIC (iPhone), Streamlit
# lo devuelve igual — lo guardamos como .heic y el viewer del PDF
# se encarga de convertir si hace falta.
EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif"}


def directorio_fotos_consulta(lote_id: int,
                               recordatorio_id: Optional[int]) -> Path:
    """Devuelve el Path del directorio para las fotos de esa consulta.

    Si `recordatorio_id` es None (foto suelta sin consulta asociada),
    cae en `data/fotos_lote/<lote_id>/sueltas/`.
    """
    if recordatorio_id is None:
        return FOTOS_ROOT / str(lote_id) / "sueltas"
    return FOTOS_ROOT / str(lote_id) / str(recordatorio_id)


def _nombre_seguro(tipo: str, original: str) -> str:
    """Arma un nombre de archivo seguro: timestamp + tipo + ext."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = Path(original).suffix.lower()
    if ext not in EXTENSIONES_VALIDAS:
        ext = ".jpg"  # default conservador
    # Sanitizar tipo (solo letras minúsculas + dígitos)
    tipo_safe = "".join(ch for ch in tipo.lower() if ch.isalnum())[:20] or "otros"
    return f"{ts}_{tipo_safe}{ext}"


def guardar_archivo_subido(uploaded_file: Any, lote_id: int,
                            recordatorio_id: Optional[int],
                            tipo: str,
                            comentario: str = "") -> Dict[str, Any]:
    """Guarda un archivo subido por Streamlit + registra en DB.

    Args:
      uploaded_file: objeto tipo UploadedFile (tiene .name, .read()).
      lote_id: lote al que pertenece la foto.
      recordatorio_id: consulta asociada (puede ser None).
      tipo: una clave de db.TIPOS_FOTO_LOTE (bosta/animales/...).
      comentario: texto opcional.

    Returns:
      dict con id (db), archivo_path (str), tipo, comentario.
    """
    if uploaded_file is None:
        raise ValueError("uploaded_file vacío")

    destino_dir = directorio_fotos_consulta(lote_id, recordatorio_id)
    destino_dir.mkdir(parents=True, exist_ok=True)

    nombre = _nombre_seguro(tipo, uploaded_file.name)
    destino = destino_dir / nombre

    # Streamlit UploadedFile soporta .getbuffer() / .read()
    try:
        contenido = uploaded_file.getbuffer()
    except Exception:
        contenido = uploaded_file.read()
    destino.write_bytes(contenido)

    # Registrar en DB con path absoluto (más portable cross-platform)
    foto_id = db.registrar_foto_lote(
        lote_id=lote_id,
        recordatorio_id=recordatorio_id,
        tipo=tipo,
        archivo_path=str(destino),
        comentario=comentario or "",
    )
    return {
        "id": foto_id,
        "archivo_path": str(destino),
        "tipo": tipo,
        "comentario": comentario or "",
    }


def listar_fotos_categorizadas(lote_id: int,
                                 recordatorio_id: Optional[int] = None
                                 ) -> Dict[str, List[Dict]]:
    """Devuelve dict {tipo: [fotos]} agrupado para galería o PDF.

    Cada foto incluye un campo `existe` (True/False) según si el
    archivo físico todavía está en disco (a veces el usuario lo borró
    a mano y la referencia en DB queda colgada).
    """
    fotos = db.listar_fotos_lote(lote_id, recordatorio_id)
    out: Dict[str, List[Dict]] = {}
    for f in fotos:
        f["existe"] = Path(f["archivo_path"]).exists() if f.get("archivo_path") else False
        out.setdefault(f["tipo"], []).append(f)
    return out


def contar_fotos_lote(lote_id: int,
                       recordatorio_id: Optional[int] = None) -> int:
    """Conteo rápido. Útil para mostrar badge "📸 N fotos" en UI."""
    return len(db.listar_fotos_lote(lote_id, recordatorio_id))
