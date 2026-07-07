"""
Cliente del Servicio Meteorológico Nacional (SMN) Argentina.

El SMN expone un endpoint público (sin auth, sin API key) usado por su sitio web:
  - https://ws.smn.gob.ar/map_items/weather   → observaciones actuales todas las estaciones
  - https://ws.smn.gob.ar/map_items/forecast/2  → pronóstico (hoy/mañana)

Funciones principales:
  - estaciones_smn():            lista de estaciones con lat/lon (cacheada)
  - estacion_mas_cercana(lat, lon): devuelve la estación SMN más cercana
  - obtener_obs(estacion_id):    observación actual (T, HR, viento, presión)
  - obtener_pronostico(est_id):  pronóstico oficial SMN

A diferencia de Open-Meteo (modelo numérico), SMN da observaciones reales
de estaciones físicas. Útil para validar pronósticos.

Si SMN está caído o cambia el formato, retornamos None y el caller hace fallback
a Open-Meteo (que ya está integrado en clima.py).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import urllib.request
import urllib.error


URL_OBS = "https://ws.smn.gob.ar/map_items/weather"
URL_FORECAST = "https://ws.smn.gob.ar/map_items/forecast/2"
URL_ALERTAS = "https://ws.smn.gob.ar/alerts/type/AL"  # alertas oficiales activas

CACHE_DIR = Path("data/cache_smn")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_OBS = CACHE_DIR / "obs.json"
CACHE_FORECAST = CACHE_DIR / "forecast.json"
CACHE_TTL_SEC = 30 * 60  # 30 min


# Estaciones SMN relevantes para La Pampa y zonas pampeanas vecinas.
# Lista pre-cargada como fallback si el endpoint live no responde.
# Coordenadas: SMN (lat, lon) en grados decimales.
ESTACIONES_FALLBACK: List[Dict] = [
    {"id": 87623, "nombre": "Santa Rosa Aero", "provincia": "La Pampa",
     "lat": -36.5667, "lon": -64.2667},
    {"id": 87532, "nombre": "General Pico Aero", "provincia": "La Pampa",
     "lat": -35.6667, "lon": -63.7500},
    {"id": 87544, "nombre": "Anguil INTA", "provincia": "La Pampa",
     "lat": -36.5500, "lon": -63.9667},
    {"id": 87648, "nombre": "Victorica", "provincia": "La Pampa",
     "lat": -36.2150, "lon": -65.4400},
    {"id": 87650, "nombre": "General Acha", "provincia": "La Pampa",
     "lat": -37.3833, "lon": -64.6000},
    {"id": 87506, "nombre": "Río Cuarto Aero", "provincia": "Córdoba",
     "lat": -33.1167, "lon": -64.2333},
    {"id": 87480, "nombre": "Villa Reynolds Aero", "provincia": "San Luis",
     "lat": -33.7300, "lon": -65.3833},
    {"id": 87453, "nombre": "Laboulaye Aero", "provincia": "Córdoba",
     "lat": -34.1333, "lon": -63.3667},
    {"id": 87750, "nombre": "Bahía Blanca Aero", "provincia": "Buenos Aires",
     "lat": -38.7333, "lon": -62.1667},
    {"id": 87532, "nombre": "Pehuajó Aero", "provincia": "Buenos Aires",
     "lat": -35.8500, "lon": -61.9000},
    {"id": 87540, "nombre": "Trenque Lauquen", "provincia": "Buenos Aires",
     "lat": -35.9667, "lon": -62.7333},
    {"id": 87715, "nombre": "Coronel Suárez Aero", "provincia": "Buenos Aires",
     "lat": -37.4400, "lon": -61.8833},
    {"id": 87765, "nombre": "Olavarría Aero", "provincia": "Buenos Aires",
     "lat": -36.8833, "lon": -60.2167},
]


# =====================================================================
# CACHE HELPERS
# =====================================================================

def _cache_valido(path: Path, ttl: int = CACHE_TTL_SEC) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl


def _leer_cache(path: Path) -> Optional[List]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _escribir_cache(path: Path, data) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass


# =====================================================================
# FETCH SMN
# =====================================================================

def _fetch_smn(url: str, timeout: int = 10) -> Optional[List]:
    """Fetch JSON del SMN con headers que el WS acepta."""
    import ssl
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; HMS-Nutricion/1.0)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return data
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError):
        return None


def _obtener_observaciones_live() -> Optional[List[Dict]]:
    """Fetch de TODAS las observaciones actuales del SMN (cacheado 30 min)."""
    if _cache_valido(CACHE_OBS):
        cached = _leer_cache(CACHE_OBS)
        if cached:
            return cached
    data = _fetch_smn(URL_OBS)
    if data:
        _escribir_cache(CACHE_OBS, data)
    return data


def _obtener_pronosticos_live() -> Optional[List[Dict]]:
    """Fetch del pronóstico SMN para todas las estaciones (cacheado 30 min)."""
    if _cache_valido(CACHE_FORECAST):
        cached = _leer_cache(CACHE_FORECAST)
        if cached:
            return cached
    data = _fetch_smn(URL_FORECAST)
    if data:
        _escribir_cache(CACHE_FORECAST, data)
    return data


# =====================================================================
# API PÚBLICA
# =====================================================================

def estaciones_smn() -> List[Dict]:
    """Lista de estaciones con datos en vivo + fallback estático.

    Cada estación: {id, nombre, provincia, lat, lon}.
    """
    obs = _obtener_observaciones_live()
    if not obs:
        return ESTACIONES_FALLBACK

    estaciones = []
    for e in obs:
        if not isinstance(e, dict):
            continue
        # SMN devuelve nombre, province, lat, lon, station_id
        try:
            est = {
                "id": e.get("station_id") or e.get("id") or e.get("int_id"),
                "nombre": e.get("name") or e.get("nombre") or "",
                "provincia": e.get("province") or e.get("provincia") or "",
                "lat": float(e.get("lat") or 0),
                "lon": float(e.get("lon") or 0),
            }
            if est["lat"] and est["lon"]:
                estaciones.append(est)
        except (TypeError, ValueError):
            continue
    return estaciones if estaciones else ESTACIONES_FALLBACK


def _distancia_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine simple, suficiente para distancias < 500 km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def estacion_mas_cercana(lat: float, lon: float,
                          max_km: float = 250) -> Optional[Dict]:
    """Devuelve {id, nombre, provincia, lat, lon, distancia_km} o None."""
    if not (lat and lon):
        return None
    estaciones = estaciones_smn()
    if not estaciones:
        return None
    mejor = None
    mejor_d = float("inf")
    for e in estaciones:
        d = _distancia_km(lat, lon, e["lat"], e["lon"])
        if d < mejor_d:
            mejor_d = d
            mejor = {**e, "distancia_km": round(d, 1)}
    if mejor and mejor_d <= max_km:
        return mejor
    return None


def obtener_obs(estacion_id) -> Optional[Dict]:
    """Observación actual de una estación específica."""
    if estacion_id is None:
        return None
    obs = _obtener_observaciones_live()
    if not obs:
        return None
    for e in obs:
        if not isinstance(e, dict):
            continue
        eid = e.get("station_id") or e.get("id") or e.get("int_id")
        if eid == estacion_id:
            # Normalizar campos comunes
            w = e.get("weather", {}) or {}
            return {
                "estacion_id": eid,
                "nombre": e.get("name") or "",
                "provincia": e.get("province") or "",
                "temp_c": _to_float(w.get("temp")),
                "humedad_pct": _to_float(w.get("humidity")),
                "viento_kmh": _to_float(w.get("wind_speed")),
                "viento_dir": w.get("wind_deg") or w.get("wind_direction"),
                "presion_hpa": _to_float(w.get("pressure")),
                "descripcion": w.get("description") or w.get("tendency", ""),
                "ts": e.get("ts") or e.get("timestamp"),
            }
    return None


def obtener_pronostico(estacion_id) -> Optional[Dict]:
    """Pronóstico SMN (hoy/mañana) para una estación."""
    if estacion_id is None:
        return None
    fc = _obtener_pronosticos_live()
    if not fc:
        return None
    for e in fc:
        if not isinstance(e, dict):
            continue
        eid = e.get("station_id") or e.get("id") or e.get("int_id")
        if eid == estacion_id:
            return e
    return None


def _to_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


# =====================================================================
# RESUMEN PARA AGENTE / EMAILS
# =====================================================================

def obtener_alertas_oficiales(provincia: str = "") -> List[Dict]:
    """Obtiene las alertas oficiales activas del SMN (amarillo/naranja/rojo).

    Args:
      provincia: si se pasa, filtra solo las que mencionen esa provincia/zona
                  en el campo de zonas afectadas (busqueda case-insensitive).

    Returns: lista de dicts con {id, titulo, descripcion, zonas, nivel,
                                  fenomenos, valida_desde, valida_hasta}.
             Lista vacía si no hay alertas o el endpoint no responde.
    """
    data = _fetch_smn(URL_ALERTAS)
    if not data or not isinstance(data, list):
        return []

    salida: List[Dict] = []
    prov_lower = (provincia or "").lower().strip()

    for a in data:
        if not isinstance(a, dict):
            continue

        # Extraer zonas afectadas (varios formatos posibles)
        zonas: List[str] = []
        for campo in ("zones", "regions", "areas", "zonas", "regiones"):
            items = a.get(campo) or []
            if not isinstance(items, list):
                continue
            for z in items:
                if isinstance(z, dict):
                    nombre = (z.get("name") or z.get("nombre")
                                or z.get("title") or "")
                    if nombre:
                        zonas.append(str(nombre))
                elif isinstance(z, str):
                    zonas.append(z)

        # Color/nivel (amarillo|naranja|rojo)
        nivel = str(a.get("color") or a.get("level") or a.get("severity")
                    or a.get("nivel") or "amarillo").lower().strip()
        # Normalizar valores comunes
        if nivel in ("yellow", "amarilla"): nivel = "amarillo"
        elif nivel in ("orange", "naranja"): nivel = "naranja"
        elif nivel in ("red", "roja"): nivel = "rojo"

        # Fenómenos
        fenomenos = a.get("phenomenon") or a.get("phenomena") \
            or a.get("fenomenos") or a.get("fenomeno") or []
        if isinstance(fenomenos, str):
            fenomenos = [fenomenos]
        elif not isinstance(fenomenos, list):
            fenomenos = []

        # Filtrar por provincia si se especificó
        if prov_lower:
            zonas_str = " | ".join(zonas).lower()
            descripcion_str = str(a.get("description")
                                    or a.get("descripcion") or "").lower()
            if (prov_lower not in zonas_str
                    and prov_lower not in descripcion_str):
                continue

        salida.append({
            "id": a.get("id") or a.get("_id"),
            "titulo": str(a.get("title") or a.get("titulo") or ""),
            "descripcion": str(a.get("description")
                                or a.get("descripcion") or ""),
            "zonas": zonas,
            "nivel": nivel,
            "fenomenos": [str(f) for f in fenomenos],
            "valida_desde": a.get("valid_from") or a.get("desde")
                or a.get("valid_at"),
            "valida_hasta": a.get("valid_to") or a.get("hasta")
                or a.get("expires_at"),
        })

    return salida


def resumen_smn(lat: float, lon: float) -> Optional[Dict]:
    """Resumen de la estación SMN más cercana + observación actual.

    Devuelve None si no hay estación cercana o el SMN no responde.
    Estructura útil para inyectar en el agente IA o mostrar en email.
    """
    est = estacion_mas_cercana(lat, lon)
    if not est:
        return None
    obs = obtener_obs(est["id"])
    fc = obtener_pronostico(est["id"])
    return {
        "estacion": est,
        "observacion": obs,
        "pronostico": fc,
        "fuente": "SMN — Servicio Meteorológico Nacional",
        "url_publica": "https://www.smn.gob.ar/clima",
    }
