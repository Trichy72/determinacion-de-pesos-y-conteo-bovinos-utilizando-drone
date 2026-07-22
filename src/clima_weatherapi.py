"""
Cliente de WeatherAPI.com — usado SOLO para alertas oficiales.

El SMN deprecó su endpoint público de alertas (/alerts/type/AL devuelve 404).
WeatherAPI.com integra automáticamente alertas oficiales de organismos
meteorológicos nacionales (incluyendo SMN para Argentina) en su endpoint
forecast.json con `alerts=yes`.

API key:
  Free tier 1.000.000 calls/mes — suficiente para 30+ clientes consultando
  cada hora. Se guarda en data/weatherapi_config.json (gitignore).

El pronóstico CLIMÁTICO sigue siendo Open-Meteo (gratis, sin API key, mejor
resolución por ser modelo numérico). WeatherAPI lo usamos solo para `alerts`.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


CONFIG_PATH = Path("data/weatherapi_config.json")
URL_FORECAST = "https://api.weatherapi.com/v1/forecast.json"


# =====================================================================
# CONFIG
# =====================================================================

def cargar_config() -> Optional[Dict]:
    """Prioridad: archivo JSON local → env var WEATHERAPI_KEY → None."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    key = os.getenv("WEATHERAPI_KEY")
    if key:
        return {"api_key": key}
    return None


def guardar_config(cfg: Dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def obtener_api_key() -> Optional[str]:
    cfg = cargar_config()
    if cfg:
        return cfg.get("api_key")
    return None


# =====================================================================
# HTTP
# =====================================================================

def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch(url: str, timeout: int = 10) -> Optional[Dict]:
    """GET JSON con SSL robusto."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "HMS-Nutricion/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError):
        return None


# =====================================================================
# ALERTAS OFICIALES
# =====================================================================

# Severity → nivel local (amarillo/naranja/rojo)
_SEVERITY_MAP = {
    "minor": "amarillo",
    "moderate": "amarillo",
    "severe": "naranja",
    "extreme": "rojo",
}


def obtener_alertas_oficiales(lat: float, lon: float,
                                api_key: Optional[str] = None,
                                lang: str = "es") -> List[Dict]:
    """Devuelve alertas oficiales activas para un punto (lat, lon).

    Cada item: {titulo, descripcion, zonas, nivel, fenomenos,
                valida_desde, valida_hasta, instrucciones}.
    Lista vacía si no hay alertas o falla la API.
    """
    if api_key is None:
        api_key = obtener_api_key()
    if not api_key or not lat or not lon:
        return []

    params = urllib.parse.urlencode({
        "key": api_key,
        "q": f"{lat},{lon}",
        "days": 1,
        "alerts": "yes",
        "aqi": "no",
        "lang": lang,
    })
    url = f"{URL_FORECAST}?{params}"

    data = _fetch(url)
    if not data or not isinstance(data, dict):
        return []

    alerts_data = data.get("alerts", {}) or {}
    alert_list = alerts_data.get("alert", []) or []
    if not isinstance(alert_list, list):
        return []

    salida: List[Dict] = []
    for a in alert_list:
        if not isinstance(a, dict):
            continue
        sev = str(a.get("severity") or "").lower()
        nivel = _SEVERITY_MAP.get(sev, "amarillo")
        zonas_raw = a.get("areas") or ""
        # WeatherAPI a veces concatena con ";" o ","
        if isinstance(zonas_raw, str):
            zonas = [z.strip() for z in zonas_raw.replace(";", ",").split(",")
                     if z.strip()]
        elif isinstance(zonas_raw, list):
            zonas = [str(z) for z in zonas_raw]
        else:
            zonas = []

        salida.append({
            "titulo": a.get("headline") or a.get("event") or "",
            "descripcion": a.get("desc") or a.get("description") or "",
            "instrucciones": a.get("instruction") or "",
            "zonas": zonas,
            "nivel": nivel,
            "fenomenos": [str(a.get("event"))] if a.get("event") else [],
            "valida_desde": a.get("effective") or a.get("onset"),
            "valida_hasta": a.get("expires"),
            "categoria": a.get("category"),
            "urgencia": a.get("urgency"),
            "certeza": a.get("certainty"),
        })
    return salida


def test_api_key(api_key: str) -> tuple[bool, str]:
    """Verifica que la API key funcione llamando con coords de Buenos Aires."""
    if not api_key:
        return False, "API key vacía"
    params = urllib.parse.urlencode({
        "key": api_key,
        "q": "-34.6,-58.4",
        "days": 1,
        "alerts": "no",
        "aqi": "no",
    })
    url = f"{URL_FORECAST}?{params}"
    data = _fetch(url)
    if data is None:
        return False, "No respondió (red o SSL)"
    if "error" in data:
        return False, f"Error: {data['error'].get('message', 'desconocido')}"
    if "current" in data:
        return True, "OK"
    return False, f"Respuesta inesperada: {list(data.keys())[:5]}"
