"""
Cliente para el Sistema de Estaciones Meteorológicas Provincial — La Pampa.

URL: https://produccion.lapampa.gob.ar/sistema-de-estaciones-meteorologicas-provincial.html

Este módulo descarga datos en tiempo real de las 47 estaciones
agro-meteorológicas del Ministerio de Producción de La Pampa.

Como el sitio no expone una API REST documentada, hacemos scraping del
HTML buscando estructuras comunes (tablas, divs con id/class predictibles).

NOTA: el código está armado de forma defensiva — si los selectors fallan,
intenta variantes y devuelve todo lo que pudo extraer. Si la página
cambia su estructura, hay que actualizar los selectors en este archivo.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

URL_BASE = "https://lapampa.redesclimaticas.com"
URL_SISTEMA = URL_BASE
URL_LEGACY = "https://produccion.lapampa.gob.ar/sistema-de-estaciones-meteorologicas-provincial.html"
USER_AGENT = "HMS-Nutricion-Animal/1.0 (cliente meteorológico)"
CACHE_PATH = Path("data/.lapampa_cache.json")
CACHE_TTL_MINUTOS = 30   # los datos del Gobierno se actualizan cada 30 min aprox.


# =====================================================================
# Lista preconfigurada de estaciones más relevantes
# =====================================================================
# Datos aproximados de las estaciones principales del sistema provincial.
# Cuando hagamos scraping del sitio podremos extraerlas todas, pero estas
# 12 cubren la mayor parte de la provincia agropecuaria.

ESTACIONES_LA_PAMPA = [
    {"nombre": "Anguil",            "lat": -36.520, "lon": -63.990},
    {"nombre": "Catriló",           "lat": -36.410, "lon": -63.420},
    {"nombre": "Colonia La Carlota","lat": -37.450, "lon": -63.733},
    {"nombre": "Eduardo Castex",    "lat": -35.910, "lon": -64.300},
    {"nombre": "General Acha",      "lat": -37.380, "lon": -64.610},
    {"nombre": "General Pico",      "lat": -35.660, "lon": -63.760},
    {"nombre": "Guatraché",         "lat": -37.667, "lon": -63.533},
    {"nombre": "Macachín",          "lat": -37.150, "lon": -63.680},
    {"nombre": "Quemú Quemú",       "lat": -36.060, "lon": -63.580},
    {"nombre": "Realicó",           "lat": -35.040, "lon": -64.250},
    {"nombre": "Santa Rosa",        "lat": -36.620, "lon": -64.290},
    {"nombre": "Toay",              "lat": -36.680, "lon": -64.380},
    {"nombre": "Trenel",            "lat": -35.700, "lon": -64.100},
    {"nombre": "Victorica",         "lat": -36.220, "lon": -65.430},
    {"nombre": "Winifreda",         "lat": -36.230, "lon": -64.250},
]


@dataclass
class DatosEstacion:
    """Datos meteorológicos extraídos de una estación La Pampa."""
    estacion: str
    fecha_consulta: datetime
    temperatura_c: Optional[float] = None
    humedad_pct: Optional[float] = None
    sensacion_termica_c: Optional[float] = None
    viento_kmh: Optional[float] = None
    direccion_viento: Optional[str] = None
    precipitacion_mm_24h: Optional[float] = None
    precipitacion_mm_7d: Optional[float] = None
    precipitacion_mm_30d: Optional[float] = None
    temp_max_24h: Optional[float] = None
    temp_min_24h: Optional[float] = None
    horas_frio_acum: Optional[float] = None   # horas debajo de 7°C
    presion_hpa: Optional[float] = None
    fuente: str = "Sistema Estaciones Meteorológicas Provincial - La Pampa"
    raw_html: str = ""   # para debugging


# =====================================================================
# 1) ESTACIÓN MÁS CERCANA A UNA COORDENADA
# =====================================================================

def estacion_mas_cercana(lat: float, lon: float) -> Dict:
    """Devuelve la estación de La Pampa más cercana a una coordenada."""
    import math

    def dist(e):
        # Distancia haversina simplificada
        dlat = math.radians(e["lat"] - lat)
        dlon = math.radians(e["lon"] - lon)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat))
             * math.cos(math.radians(e["lat"]))
             * math.sin(dlon / 2) ** 2)
        return 2 * 6371 * math.asin(math.sqrt(a))

    return min(ESTACIONES_LA_PAMPA, key=dist)


def listar_estaciones() -> List[Dict]:
    """Devuelve todas las estaciones preconfiguradas."""
    return list(ESTACIONES_LA_PAMPA)


# =====================================================================
# 2) DESCARGA Y PARSEO DEL SITIO
# =====================================================================

def _descargar_html(url: str, timeout: int = 15) -> Optional[str]:
    """Descarga el HTML del sitio del Gobierno de La Pampa."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        log.warning("Error descargando %s: %s", url, e)
        return None


def _parsear_numero(texto: str) -> Optional[float]:
    """Extrae el primer número (acepta coma decimal) de un string."""
    if not texto:
        return None
    # Reemplazar coma por punto para facilitar
    t = texto.replace(",", ".").replace("\xa0", " ")
    m = re.search(r"-?\d+\.?\d*", t)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass
    return None


def _intentar_parsear_html(html: str, estacion: str) -> DatosEstacion:
    """
    Intenta extraer datos del HTML usando varias estrategias.
    Defensivo: si la estructura del sitio cambia, aún así devuelve lo que
    pudo encontrar.
    """
    datos = DatosEstacion(estacion=estacion, fecha_consulta=datetime.now())
    datos.raw_html = html[:500]   # primeros 500 chars para debug

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("Falta beautifulsoup4 — no puedo parsear HTML")
        return datos

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        log.warning("BS4 error: %s", e)
        return datos

    # ESTRATEGIA 1: buscar tabla de datos meteorológicos por keywords
    # típicos del sitio del Gobierno
    keywords = {
        "temperatura": "temperatura_c",
        "humedad": "humedad_pct",
        "sensación térmica": "sensacion_termica_c",
        "sensacion termica": "sensacion_termica_c",
        "viento": "viento_kmh",
        "precipitación": "precipitacion_mm_24h",
        "precipitacion": "precipitacion_mm_24h",
        "lluvia": "precipitacion_mm_24h",
        "máxima": "temp_max_24h",
        "mínima": "temp_min_24h",
        "presión": "presion_hpa",
        "presion": "presion_hpa",
        "horas de frío": "horas_frio_acum",
        "horas frio": "horas_frio_acum",
    }

    # Buscar todas las tablas
    for tabla in soup.find_all(["table", "div"]):
        texto = tabla.get_text(separator=" ", strip=True).lower()
        for clave, atributo in keywords.items():
            if clave in texto:
                # Buscar números cerca del keyword
                idx = texto.find(clave)
                if idx >= 0:
                    sub = texto[idx:idx + 80]
                    valor = _parsear_numero(sub)
                    if valor is not None and getattr(datos, atributo) is None:
                        setattr(datos, atributo, valor)

    # ESTRATEGIA 2: si hay un JSON embebido en JavaScript (común en
    # dashboards modernos)
    json_pattern = re.compile(
        r"var\s+\w+\s*=\s*(\{[^}]+\}|\[[^\]]+\])\s*;",
        re.DOTALL,
    )
    for match in json_pattern.finditer(html[:50000]):
        try:
            data = json.loads(match.group(1))
            # Buscar campos conocidos
            if isinstance(data, dict):
                for k, v in data.items():
                    k_low = k.lower()
                    if "temp" in k_low and datos.temperatura_c is None:
                        if isinstance(v, (int, float)):
                            datos.temperatura_c = float(v)
                    elif "hum" in k_low and datos.humedad_pct is None:
                        if isinstance(v, (int, float)):
                            datos.humedad_pct = float(v)
                    # etc...
        except (json.JSONDecodeError, TypeError):
            continue

    return datos


def obtener_datos_estacion(estacion: str = "Catriló",
                            url_base: str = URL_SISTEMA) -> Optional[DatosEstacion]:
    """
    Intenta obtener datos en tiempo real de una estación específica del
    Gobierno de La Pampa.

    Como no conocemos exactamente la URL del endpoint de cada estación
    (no hay API documentada), probamos algunas estrategias:
      1. URL principal del sistema
      2. URL con parámetro de estación: ?estacion=Catriló
      3. URL del clima específico: /clima/<estacion>

    Si todo falla, devolvemos None y el sistema cae al fallback de
    Open-Meteo.
    """
    cache = _cargar_cache()
    cache_key = f"lapampa|{estacion}"
    if cache_key in cache:
        cache_entry = cache[cache_key]
        try:
            cached_at = datetime.fromisoformat(cache_entry["fecha_consulta"])
            if (datetime.now() - cached_at).total_seconds() < CACHE_TTL_MINUTOS * 60:
                # Reconstruir el dataclass desde cache
                d = DatosEstacion(
                    estacion=cache_entry.get("estacion", estacion),
                    fecha_consulta=cached_at,
                )
                for k, v in cache_entry.items():
                    if hasattr(d, k) and v is not None:
                        try:
                            setattr(d, k, v if k.endswith("_html") or
                                            k == "estacion" or
                                            k == "fuente"
                                            else float(v) if v else None)
                        except (TypeError, ValueError):
                            pass
                d.fecha_consulta = cached_at
                return d
        except Exception:
            pass

    urls_a_probar = [
        url_base,
        f"{url_base}?estacion={urllib.parse.quote(estacion)}",
        f"{URL_BASE}/clima/{urllib.parse.quote(estacion)}",
        f"{URL_BASE}/api/estacion/{urllib.parse.quote(estacion)}",
    ]

    for url in urls_a_probar:
        html = _descargar_html(url)
        if not html:
            continue
        datos = _intentar_parsear_html(html, estacion)
        # Si extrajimos al menos T° o humedad, lo damos por bueno
        if datos.temperatura_c is not None or datos.humedad_pct is not None:
            _guardar_cache(cache_key, datos)
            return datos

    log.info("No pude extraer datos de la estación %s", estacion)
    return None


def _cargar_cache() -> Dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _guardar_cache(key: str, datos: DatosEstacion) -> None:
    cache = _cargar_cache()
    cache[key] = {
        "estacion": datos.estacion,
        "fecha_consulta": datos.fecha_consulta.isoformat(),
        "temperatura_c": datos.temperatura_c,
        "humedad_pct": datos.humedad_pct,
        "sensacion_termica_c": datos.sensacion_termica_c,
        "viento_kmh": datos.viento_kmh,
        "direccion_viento": datos.direccion_viento,
        "precipitacion_mm_24h": datos.precipitacion_mm_24h,
        "precipitacion_mm_7d": datos.precipitacion_mm_7d,
        "precipitacion_mm_30d": datos.precipitacion_mm_30d,
        "temp_max_24h": datos.temp_max_24h,
        "temp_min_24h": datos.temp_min_24h,
        "horas_frio_acum": datos.horas_frio_acum,
        "presion_hpa": datos.presion_hpa,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                           encoding="utf-8")


# =====================================================================
# 3) RESUMEN PARA EL AGENTE / UI
# =====================================================================

def guardar_datos_manuales(estacion: str, datos: Dict) -> None:
    """Guarda datos cargados manualmente desde Redes Climáticas.

    `datos` puede contener: temperatura_c, humedad_pct, sensacion_termica_c,
    viento_kmh, precipitacion_mm_24h, precipitacion_mm_7d, temp_max_24h,
    temp_min_24h, horas_frio_acum, etc.
    """
    cache = _cargar_cache()
    key = f"manual|{estacion}"
    entry = {
        "estacion": estacion,
        "fecha_consulta": datetime.now().isoformat(),
        "fuente": "Redes Climáticas (manual)",
    }
    entry.update({k: v for k, v in datos.items() if v is not None})
    cache[key] = entry
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def obtener_datos_manuales(estacion: str,
                            ttl_horas: float = 24) -> Optional[Dict]:
    """Lee datos cargados manualmente si todavía están vigentes."""
    cache = _cargar_cache()
    key = f"manual|{estacion}"
    if key not in cache:
        return None
    entry = cache[key]
    try:
        cargado = datetime.fromisoformat(entry["fecha_consulta"])
        horas = (datetime.now() - cargado).total_seconds() / 3600
        if horas > ttl_horas:
            return None
    except Exception:
        return None
    return entry


def resumen_estacion_oficial(lat: float, lon: float) -> Optional[str]:
    """Si la coord está cerca de La Pampa, busca datos en este orden:
      1. Datos cargados manualmente (de Redes Climáticas) — preferidos
      2. Scraping (intento — puede no funcionar si requiere login)

    Solo si la coord está en el rango de La Pampa (lat -39 a -34, lon -67 a -62).
    """
    if not (-39 <= lat <= -34 and -67 <= lon <= -62):
        return None

    estacion = estacion_mas_cercana(lat, lon)
    nombre = estacion["nombre"]

    # 1) Intentar datos cargados manualmente desde Redes Climáticas
    manual = obtener_datos_manuales(nombre, ttl_horas=24)
    if manual:
        try:
            cargado = datetime.fromisoformat(manual["fecha_consulta"])
        except Exception:
            cargado = datetime.now()
        lineas = [
            "═══════════════════════════════════════════════════════════════",
            f"📡 ESTACIÓN — {nombre} (Redes Climáticas, La Pampa)",
            "═══════════════════════════════════════════════════════════════",
            f"Datos cargados manualmente por el asesor",
            f"Hace {(datetime.now() - cargado).total_seconds() / 3600:.1f} horas",
        ]
        for label, key, fmt in [
            ("🌡️ Temperatura", "temperatura_c", "{:.1f}°C"),
            ("💧 Humedad", "humedad_pct", "{:.0f}%"),
            ("❄️ Sensación térmica", "sensacion_termica_c", "{:.1f}°C"),
            ("💨 Viento", "viento_kmh", "{:.0f} km/h"),
            ("🌧️ Lluvia 24h", "precipitacion_mm_24h", "{:.1f} mm"),
            ("🌧️ Lluvia 7 días", "precipitacion_mm_7d", "{:.1f} mm"),
            ("🌧️ Lluvia 30 días", "precipitacion_mm_30d", "{:.1f} mm"),
            ("📈 T° máxima 24h", "temp_max_24h", "{:.1f}°C"),
            ("📉 T° mínima 24h", "temp_min_24h", "{:.1f}°C"),
            ("❄️ Horas de frío", "horas_frio_acum", "{:.0f} hs"),
        ]:
            v = manual.get(key)
            if v is not None:
                try:
                    lineas.append(f"{label}: {fmt.format(float(v))}")
                except (TypeError, ValueError):
                    pass
        return "\n".join(lineas)

    # 2) Fallback: scraping automático (probablemente falla por login)
    datos = obtener_datos_estacion(nombre)
    if not datos or (datos.temperatura_c is None and datos.humedad_pct is None):
        return None

    lineas = [
        "═══════════════════════════════════════════════════════════════",
        f"📡 ESTACIÓN — {datos.estacion} (Redes Climáticas)",
        "═══════════════════════════════════════════════════════════════",
        f"Última actualización: {datos.fecha_consulta.strftime('%d/%m/%Y %H:%M')}",
    ]
    if datos.temperatura_c is not None:
        lineas.append(f"🌡️ Temperatura: {datos.temperatura_c:.1f}°C")
    if datos.humedad_pct is not None:
        lineas.append(f"💧 Humedad: {datos.humedad_pct:.0f}%")
    if datos.sensacion_termica_c is not None:
        lineas.append(f"❄️ Sensación térmica: {datos.sensacion_termica_c:.1f}°C")
    if datos.viento_kmh is not None:
        lineas.append(f"💨 Viento: {datos.viento_kmh:.0f} km/h")
    if datos.precipitacion_mm_24h is not None:
        lineas.append(f"🌧️ Precipitación 24h: {datos.precipitacion_mm_24h:.1f} mm")
    if datos.precipitacion_mm_7d is not None:
        lineas.append(f"🌧️ Precipitación 7 días: {datos.precipitacion_mm_7d:.1f} mm")
    if datos.temp_max_24h is not None and datos.temp_min_24h is not None:
        lineas.append(
            f"📈 Rango T° últimas 24h: "
            f"{datos.temp_min_24h:.1f}°C a {datos.temp_max_24h:.1f}°C"
        )
    if datos.horas_frio_acum is not None:
        lineas.append(f"❄️ Horas de frío acumuladas: {datos.horas_frio_acum:.0f}")
    return "\n".join(lineas)
