"""
Datos climáticos para el contexto del agente nutricional.

Usa Open-Meteo (https://open-meteo.com) — gratis, sin API key, sin límites
para uso no comercial. Cobertura global incluyendo Argentina.

Provee:
  - Geocodificación de localidades argentinas (Catriló, Realicó, etc.)
  - Clima actual (temperatura, humedad, viento)
  - Histórico de los últimos 7 días (para detectar olas de calor/frío)
  - Pronóstico de los próximos 7 días
  - Cálculo del Índice de Temperatura-Humedad (THI) para detectar estrés
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

GEOCODE_CACHE = Path("data/.geocode_cache.json")


def _ssl_ctx():
    """Contexto SSL robusto. En macOS, Python a veces no encuentra los CAs
    del sistema. Si certifi está disponible, lo usamos."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


# =====================================================================
# 1) GEOCODIFICACIÓN
# =====================================================================

def _cargar_cache_geocode() -> Dict[str, Dict]:
    if not GEOCODE_CACHE.exists():
        return {}
    try:
        return json.loads(GEOCODE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _guardar_cache_geocode(cache: Dict[str, Dict]) -> None:
    GEOCODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GEOCODE_CACHE.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _intentar_open_meteo(consulta: str, pais: str = "Argentina") -> Optional[Dict]:
    """Intenta geocodificar con Open-Meteo Geocoding API."""
    url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        f"name={urllib.parse.quote(consulta)}&count=10&language=es"
    )
    try:
        with urllib.request.urlopen(url, timeout=8, context=_ssl_ctx()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.debug("Open-Meteo geocoding error: %s", e)
        return None

    if not data.get("results"):
        return None
    pais_lower = pais.lower()
    for r in data["results"]:
        if r.get("country", "").lower() == pais_lower:
            return {
                "lat": r["latitude"], "lon": r["longitude"],
                "nombre": r.get("name", ""),
                "admin1": r.get("admin1", ""),
                "country": r.get("country", ""),
                "fuente": "Open-Meteo",
            }
    r = data["results"][0]
    return {
        "lat": r["latitude"], "lon": r["longitude"],
        "nombre": r.get("name", ""),
        "admin1": r.get("admin1", ""),
        "country": r.get("country", ""),
        "fuente": "Open-Meteo",
    }


def _intentar_nominatim(consulta: str, pais: str = "Argentina") -> Optional[Dict]:
    """Intenta geocodificar con Nominatim (OpenStreetMap) — mejor cobertura
    para localidades pequeñas argentinas."""
    # Agregar el país a la consulta para filtrar mejor
    consulta_full = f"{consulta}, {pais}"
    url = (
        "https://nominatim.openstreetmap.org/search?"
        f"q={urllib.parse.quote(consulta_full)}"
        "&format=json&limit=5&accept-language=es&addressdetails=1"
    )
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "HMS-Nutricion-Animal/1.0",
        })
        with urllib.request.urlopen(req, timeout=8, context=_ssl_ctx()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.debug("Nominatim error: %s", e)
        return None

    if not data:
        return None

    # Preferir resultados en Argentina y de tipo "village/town/city"
    pais_lower = pais.lower()
    for r in data:
        addr = r.get("address", {}) or {}
        if pais_lower in (addr.get("country", "") or "").lower():
            return {
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "nombre": addr.get("city") or addr.get("town")
                          or addr.get("village") or r.get("name", "")
                          or consulta,
                "admin1": addr.get("state", ""),
                "country": addr.get("country", ""),
                "fuente": "OpenStreetMap",
            }
    # Si no hay match con país, usar el primer resultado
    r = data[0]
    addr = r.get("address", {}) or {}
    return {
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "nombre": addr.get("city") or addr.get("town")
                  or addr.get("village") or consulta,
        "admin1": addr.get("state", ""),
        "country": addr.get("country", ""),
        "fuente": "OpenStreetMap",
    }


def geocodificar(localidad: str, pais: str = "Argentina") -> Optional[Dict]:
    """Busca lat/lon de una localidad usando múltiples estrategias.

    Orden de búsqueda:
      1. Cache local
      2. Open-Meteo Geocoding (rápido, pero cobertura limitada)
      3. Nominatim OpenStreetMap (más completo, especialmente para Argentina)
      4. Variaciones del nombre (sin "Colonia", "Estancia", etc.)
      5. Solo el último componente (ej. "La Pampa" si la localidad falla)

    Devuelve dict {lat, lon, nombre, admin1, country, fuente} o None.
    """
    if not localidad or len(localidad.strip()) < 2:
        return None

    key = f"{localidad.strip().lower()}|{pais.strip().lower()}"
    cache = _cargar_cache_geocode()
    if key in cache:
        return cache[key]

    nombre_base = localidad.strip()

    # Generar variaciones del nombre para intentar
    variaciones = [nombre_base]

    # Quitar prefijos comunes de localidades
    for prefijo in ["Colonia ", "Estancia ", "Pueblo ", "Villa ",
                     "Pueblo de ", "Localidad de "]:
        if nombre_base.lower().startswith(prefijo.lower()):
            variaciones.append(nombre_base[len(prefijo):])

    # Si tiene coma (ej. "La Carlota, Córdoba"), probar también solo la primera parte
    if "," in nombre_base:
        partes = [p.strip() for p in nombre_base.split(",")]
        if partes[0] and partes[0] != nombre_base:
            variaciones.append(partes[0])

    # Intentar cada variación con cada API
    for v in variaciones:
        if not v:
            continue
        # 1) Open-Meteo
        result = _intentar_open_meteo(v, pais)
        if result:
            cache[key] = result
            _guardar_cache_geocode(cache)
            log.info("Geocoding ✓ via Open-Meteo: %s → %s", localidad, result["nombre"])
            return result

        # 2) Nominatim (más lento pero mejor cobertura)
        result = _intentar_nominatim(v, pais)
        if result:
            cache[key] = result
            _guardar_cache_geocode(cache)
            log.info("Geocoding ✓ via Nominatim: %s → %s", localidad, result["nombre"])
            return result

    log.warning("Geocoding ✗ falló para: %s", localidad)
    return None


def geocodificar_manual(lat: float, lon: float, nombre: str = "") -> Dict:
    """Crea un objeto de geocoding desde coordenadas cargadas a mano."""
    return {
        "lat": float(lat),
        "lon": float(lon),
        "nombre": nombre or f"({lat:.2f}, {lon:.2f})",
        "admin1": "",
        "country": "Argentina",
        "fuente": "manual",
    }


# =====================================================================
# Circuit breaker para Open-Meteo
# =====================================================================
# Cuando Open-Meteo está caído (5xx, timeouts, 429) y los reintentos
# fallan repetidamente, dejamos de consultar por X minutos. Evita que
# la UI de Streamlit quede bloqueada esperando respuestas que nunca
# llegan, y limpia el log de spam.

import time as _time_mod
_CIRCUIT_BREAKER = {
    # api_url base → {fallos_consecutivos: int, abierto_hasta: float|None}
    "forecast": {"fallos": 0, "abierto_hasta": None},
    "archive": {"fallos": 0, "abierto_hasta": None},
}
_CIRCUIT_UMBRAL_FALLOS = 2     # Después de 2 fallos seguidos, abre.
_CIRCUIT_COOLDOWN_SEG = 600    # 10 minutos sin consultar.


def _circuit_abierto(servicio: str) -> bool:
    """¿Está el circuit breaker abierto (no consultar) para servicio?"""
    estado = _CIRCUIT_BREAKER.get(servicio) or {}
    abierto_hasta = estado.get("abierto_hasta")
    if abierto_hasta is None:
        return False
    if _time_mod.time() >= abierto_hasta:
        # Tiempo agotado, cerramos el breaker y reseteamos fallos
        estado["abierto_hasta"] = None
        estado["fallos"] = 0
        return False
    return True


def _circuit_registrar_fallo(servicio: str) -> None:
    estado = _CIRCUIT_BREAKER.setdefault(
        servicio, {"fallos": 0, "abierto_hasta": None}
    )
    estado["fallos"] += 1
    if estado["fallos"] >= _CIRCUIT_UMBRAL_FALLOS:
        estado["abierto_hasta"] = (
            _time_mod.time() + _CIRCUIT_COOLDOWN_SEG
        )
        log.warning(
            "Circuit breaker ABIERTO para Open-Meteo (%s) — "
            "no se consultará por %d min.",
            servicio, _CIRCUIT_COOLDOWN_SEG // 60,
        )


def _circuit_registrar_exito(servicio: str) -> None:
    estado = _CIRCUIT_BREAKER.setdefault(
        servicio, {"fallos": 0, "abierto_hasta": None}
    )
    estado["fallos"] = 0
    estado["abierto_hasta"] = None


# =====================================================================
# 2) CLIMA ACTUAL + HISTÓRICO + PRONÓSTICO
# =====================================================================

def obtener_clima(lat: float, lon: float,
                   dias_pasado: int = 7,
                   dias_futuro: int = 7) -> Optional[Dict]:
    """Devuelve un resumen climático integrado: actual + último N días + pronóstico."""
    hoy = datetime.now().date()
    desde = hoy - timedelta(days=dias_pasado)
    hasta = hoy + timedelta(days=dias_futuro)

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "precipitation,weather_code,apparent_temperature"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "precipitation,apparent_temperature"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "wind_speed_10m_max,relative_humidity_2m_max,relative_humidity_2m_min,"
        "apparent_temperature_max,apparent_temperature_min,"
        "cloud_cover_mean,shortwave_radiation_sum"
        f"&start_date={desde.isoformat()}&end_date={hasta.isoformat()}"
        "&timezone=America%2FArgentina%2FBuenos_Aires"
        "&models=best_match"
    )
    # Circuit breaker: si Open-Meteo viene fallando, no insistir.
    if _circuit_abierto("forecast"):
        return None

    # Reintento con backoff: a veces Open-Meteo tiene picos de latencia
    # (especialmente en horas pico). 3 intentos con 2s, 5s, 10s de espera.
    import time as _time
    ultimo_error = None
    for intento, espera in enumerate([2, 5, 10], start=1):
        try:
            with urllib.request.urlopen(
                url, timeout=15, context=_ssl_ctx()
            ) as resp:
                resultado = json.loads(resp.read().decode("utf-8"))
                _circuit_registrar_exito("forecast")
                return resultado
        except Exception as e:
            ultimo_error = e
            log.warning(
                "Clima intento %d/3 falló: %s. Reintento en %ds...",
                intento, e, espera
            )
            if intento < 3:
                _time.sleep(espera)

    log.warning("Clima error tras 3 intentos: %s", ultimo_error)
    _circuit_registrar_fallo("forecast")
    return None


def obtener_clima_historico(lat: float, lon: float,
                              fecha_desde: str,
                              fecha_hasta: str) -> Optional[Dict]:
    """Devuelve clima HISTÓRICO REAL (datos observados) para un rango
    de fechas pasadas, usando la API archive de Open-Meteo
    (https://archive-api.open-meteo.com/).

    Es lo que se debe usar para CONFIRMAR un impacto productivo
    con datos del clima que efectivamente ocurrió, en lugar del
    pronóstico que se había proyectado.

    Args:
        lat, lon: coordenadas del campo.
        fecha_desde: YYYY-MM-DD del primer día a consultar.
        fecha_hasta: YYYY-MM-DD del último día (inclusive).

    Returns:
        Dict con sección `daily` igual al de obtener_clima(), o None
        si la API falla. Datos son OBSERVADOS, no pronosticados.

    Nota: la API archive tiene un delay de ~5 días — datos del día
    de ayer pueden no estar disponibles todavía. Por eso conviene
    confirmar impactos de semanas con al menos 7 días de antigüedad.
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={fecha_desde}&end_date={fecha_hasta}"
        "&daily=temperature_2m_max,temperature_2m_min,"
        "precipitation_sum,wind_speed_10m_max,"
        "relative_humidity_2m_max,relative_humidity_2m_min,"
        "apparent_temperature_max,apparent_temperature_min"
        "&hourly=temperature_2m,relative_humidity_2m,"
        "wind_speed_10m,precipitation,apparent_temperature"
        "&timezone=America%2FArgentina%2FBuenos_Aires"
    )
    # Circuit breaker: si la API archive viene fallando, no insistir.
    if _circuit_abierto("archive"):
        return None

    import time as _time
    ultimo_error = None
    for intento, espera in enumerate([2, 5, 10], start=1):
        try:
            with urllib.request.urlopen(
                url, timeout=15, context=_ssl_ctx()
            ) as resp:
                resultado = json.loads(resp.read().decode("utf-8"))
                _circuit_registrar_exito("archive")
                return resultado
        except Exception as e:
            ultimo_error = e
            log.warning(
                "Clima histórico intento %d/3 falló: %s. "
                "Reintento en %ds...", intento, e, espera,
            )
            if intento < 3:
                _time.sleep(espera)
    log.warning(
        "Clima histórico error tras 3 intentos: %s", ultimo_error,
    )
    _circuit_registrar_fallo("archive")
    return None


def calcular_wind_chill(temp_c: float, viento_kmh: float) -> float:
    """Índice de sensación térmica (Canadian Wind Chill Index).

    Solo válido cuando viento ≥ 5 km/h. Por debajo, devuelve la T° real.

    Para bovinos, el viento potencia el estrés frío significativamente:
    una T° de 0°C con viento de 30 km/h equivale a sensación de -5°C.
    """
    if viento_kmh < 5 or temp_c > 10:
        return temp_c
    return (
        13.12 + 0.6215 * temp_c
        - 11.37 * (viento_kmh ** 0.16)
        + 0.3965 * temp_c * (viento_kmh ** 0.16)
    )


def temp_critica_inferior(categoria: str, mojado: bool = False,
                            pelo_invernal: bool = True) -> float:
    """LCT (Lower Critical Temperature) — temperatura debajo de la cual el
    animal empieza a gastar energía extra en termorregulación.

    Por debajo de la LCT, cada °C menos = +2 a +3% del requerimiento de
    mantenimiento. Fuente: NASEM 2016 cap. 7.
    """
    cat = (categoria or "").lower()
    if mojado:
        # Pelo mojado pierde 90% de su poder aislante
        if cat == "ternero":
            return 18.0    # crítico ya con frío leve
        return 14.0
    if not pelo_invernal:
        # Pelo de transición o muda, otoño/primavera
        if cat == "ternero":
            return 8.0
        return 0.0
    # Pelo invernal completo (julio-agosto)
    if cat == "ternero":
        return -3.0
    if cat == "vaquillona":
        return -8.0
    return -10.0  # vacas adultas, novillos, toros


def calcular_thi(temp_c: float, humedad_rel: float) -> float:
    """Índice de Temperatura-Humedad (THI) para bovinos.
    Fórmula NRC: THI = (1.8 × T + 32) − ((0.55 − 0.0055 × HR) × (1.8 × T − 26))
    Umbrales para bovinos:
      <72  - sin estrés
      72-78 - estrés leve
      79-88 - estrés moderado
      >88  - estrés severo
    """
    t_f = 1.8 * temp_c + 32
    return t_f - (0.55 - 0.0055 * humedad_rel) * (t_f - 58)


def clasificar_thi(thi: float) -> str:
    if thi < 72:
        return "🟢 Sin estrés"
    if thi < 79:
        return "🟡 Estrés leve"
    if thi < 89:
        return "🟠 Estrés moderado"
    return "🔴 Estrés severo"


# =====================================================================
# 3) EVALUADOR DE ESTRÉS CALÓRICO con THI ajustado por agravantes
# =====================================================================
# Lógica especificada por Mauricio Suárez (asesor 20 años, La Pampa).
# Suma +1 al THI por cada agravante presente:
#   - Viento <10 km/h
#   - Temperatura mínima nocturna >22°C
#   - Sombra <4 m²/cab
#   - ≥4 horas con THI alto el día anterior
#   - ≥2 días consecutivos de calor
# Umbrales feedlot 300-450 kg: leve ≥76, moderado ≥81, crítico ≥85.
# Salida: dict estructurado con tipo_alerta, riesgo, thi_ajustado, acciones.

UMBRAL_THI_LEVE = 76          # Default novillos preventiva (compat)
UMBRAL_THI_MODERADO = 81
UMBRAL_THI_CRITICO = 85

# Estaciones (hemisferio sur)
def estacion_actual() -> str:
    """Devuelve 'verano' / 'otoño' / 'invierno' / 'primavera' (hemisferio sur)."""
    m = datetime.now().month
    if m in (12, 1, 2): return "verano"
    if m in (3, 4, 5): return "otoño"
    if m in (6, 7, 8): return "invierno"
    return "primavera"


def texto_etapa_evento(etapa: str, tipo_clima: str = "frio",
                          dias_alerta_previos: int = 0,
                          duracion_prevista: int = 0,
                          fecha_inicio: str = "",
                          ocurre_hoy: bool = True) -> Dict[str, str]:
    """Devuelve textos contextuales según etapa para el email.

    Args:
      etapa: 'inicio' | 'persistencia' | 'acumulacion' | 'recuperacion' | 'estable'
      tipo_clima: 'frio' | 'calor' | etc — para nombrar el evento
      dias_alerta_previos: días consecutivos previos con alerta
      duracion_prevista: cuántos días más se espera el evento
      fecha_inicio: ISO date del primer día del evento (opcional)
      ocurre_hoy: True si el evento está pasando HOY,
                  False si está sólo en el pronóstico

    Returns:
      {'titulo', 'mensaje', 'prioridad', 'lenguaje': 'preventivo'|'activo'}
    """
    nombre_evento = {
        "frio": "frío",
        "calor": "calor",
        "lluvia": "lluvia",
        "barro": "barro persistente",
    }.get((tipo_clima or "").lower(), "clima adverso")

    if etapa == "inicio":
        if ocurre_hoy:
            duracion_txt = (f" — pronóstico {duracion_prevista} días"
                              if duracion_prevista > 1 else "")
            return {
                "titulo": f"🆕 Inicio del evento de {nombre_evento}",
                "mensaje": (
                    f"Hoy arranca un evento de {nombre_evento}"
                    f"{duracion_txt}. Conviene preparar reparos, agua y "
                    f"comederos AHORA para sostener consumo cuando se "
                    f"agrave."
                ),
                "prioridad": (
                    "Preparación operativa: anticipar acciones de manejo "
                    "antes de que el problema se asiente."
                ),
                "lenguaje": "activo",
            }
        else:
            # El evento todavía NO ocurre — solo está pronosticado
            cuando_txt = (f"para el {fecha_inicio}" if fecha_inicio
                            else "para los próximos días")
            return {
                "titulo": (f"📅 Evento de {nombre_evento} pronosticado"),
                "mensaje": (
                    f"Se pronostica un evento de {nombre_evento} "
                    f"{cuando_txt}. Las condiciones de HOY son normales, "
                    f"pero conviene revisar reparos, agua y mezcla "
                    f"antes de que se instale."
                ),
                "prioridad": (
                    "Seguimiento preventivo: planificar la operativa con "
                    "tiempo para llegar preparado al evento."
                ),
                "lenguaje": "preventivo",
            }
    if etapa == "persistencia":
        dia_n = dias_alerta_previos + 1
        return {
            "titulo": f"🔁 Evento de {nombre_evento} en curso · día {dia_n}",
            "mensaje": (
                f"El evento de {nombre_evento} lleva {dia_n} día(s) "
                f"seguidos. El foco pasa a sostener el consumo, controlar "
                f"el acceso al comedero y la calidad física de la mezcla."
            ),
            "prioridad": (
                "Sostener consumo: vigilar barro, selección en comedero, "
                "deterioro de mezcla y disponibilidad de agua."
            ),
            "lenguaje": "activo",
        }
    if etapa == "acumulacion":
        dia_n = dias_alerta_previos + 1
        return {
            "titulo": (f"📉 Acumulación de estrés por {nombre_evento} · "
                         f"día {dia_n}"),
            "mensaje": (
                f"El evento de {nombre_evento} lleva {dia_n} días seguidos. "
                f"Aparece riesgo de caída de eficiencia productiva, "
                f"pérdida de condición corporal y riesgo sanitario. "
                f"La recuperación nocturna puede estar siendo insuficiente."
            ),
            "prioridad": (
                "Estabilidad ruminal y condición corporal: ajustar fibra "
                "activa, vigilar rumia, controlar pisos secos y reparos. "
                "El impacto se acumula — no sumar más estrés con manejos."
            ),
            "lenguaje": "activo",
        }
    if etapa == "recuperacion":
        return {
            "titulo": "✅ Recuperación post-evento",
            "mensaje": (
                f"Las condiciones actuales mejoraron respecto al "
                f"{dias_alerta_previos}-día(s) previo(s) de "
                f"{nombre_evento}. Foco: normalizar dieta de forma "
                f"gradual, evaluar consumo real y revisar impacto "
                f"residual (barro, acceso al comedero, condición corporal)."
            ),
            "prioridad": (
                "Retorno gradual: no hacer cambios bruscos en la dieta. "
                "Vigilar consumo individual, recuperación de condición "
                "y secuelas del barro (pododermatitis, acceso al comedero)."
            ),
            "lenguaje": "post",
        }
    # estable
    return {
        "titulo": "Condiciones estables",
        "mensaje": "Sin riesgo productivo previsto. Operativa normal.",
        "prioridad": "",
        "lenguaje": "estable",
    }


def lectura_tecnica_evento(tipo_clima: str, etapa: str,
                              nivel: str = "operativo",
                              contexto: Optional[Dict] = None) -> str:
    """Devuelve una frase narrativa explicando QUÉ LE PASA AL ANIMAL.

    Es la "lectura técnica" del asesor: une fisiología, consumo,
    mezcla y manejo antes de listar acciones.

    Args:
      tipo_clima: 'frio' | 'calor' | 'lluvia'
      etapa: 'inicio' | 'persistencia' | 'acumulacion' | 'recuperacion'
      nivel: 'operativo' | 'critico' | 'atencion'
      contexto: dict opcional con flags relevantes
    """
    tipo = (tipo_clima or "").lower()
    ctx = contexto or {}
    barro = bool(ctx.get("barro"))
    lluvia = float(ctx.get("lluvia_mm", 0) or 0)
    cat_sensible = bool(ctx.get("categoria_sensible"))

    # ─── FRÍO ───
    if tipo == "frio":
        if etapa == "inicio":
            if barro or lluvia > 5:
                return (
                    "Con frío + humedad el animal pierde poder aislante del "
                    "pelaje y empieza a gastar reservas energéticas para "
                    "mantener temperatura corporal. Si el comedero tiene "
                    "barro de acceso, además come menos y cambia su patrón "
                    "habitual — eso descompensa la fermentación ruminal "
                    "antes que veamos caída en la ganancia. Conviene "
                    "anticipar reparos y mezcla protegida ANTES de que el "
                    "cuadro se instale."
                )
            return (
                "El frío seco solo aumenta el gasto energético de "
                "mantenimiento — el animal compensa comiendo más, siempre "
                "que tenga agua disponible y reparo del viento. La clave "
                "en esta etapa es preparar para que el consumo se sostenga "
                "y el rumen no entre en déficit energético."
            )
        if etapa == "persistencia":
            if barro or lluvia > 5:
                return (
                    "El animal lleva varios días gastando energía extra "
                    "para regular temperatura. Si la mezcla se moja, "
                    "selecciona más, deja sobrantes y consume menos. La "
                    "ración formulada en papel deja de ser la que entra al "
                    "rumen: aparecen picos de ácido y caídas de pH "
                    "(inestabilidad ruminal subclínica). Resultado: balance "
                    "energético negativo y pérdida incipiente de condición."
                )
            return (
                "El animal ya se acomodó al frío seco — sigue comiendo "
                "pero con mayor demanda energética. Hay que vigilar que "
                "el agua no se congele y que el comedero sea accesible "
                "todo el día, así el patrón de consumo no se altera y el "
                "rumen mantiene estabilidad."
            )
        if etapa == "acumulacion":
            return (
                "Después de varios días seguidos de frío, el animal "
                "agotó reservas de grasa de cobertura, baja su rumia y "
                "se vuelve más sensible a problemas respiratorios. Menor "
                "rumia = menos producción de proteína microbiana y rumen "
                "más inestable. La recuperación nocturna ya no compensa "
                "el gasto del día — es el punto donde empieza la pérdida "
                "productiva real."
            )
        if etapa == "recuperacion":
            return (
                "El clima mejoró pero el animal viene de varios días "
                "con balance energético negativo y rumen castigado. La "
                "condición corporal se recupera lentamente. Hay que "
                "normalizar la dieta sin subir grano de golpe (alto riesgo "
                "de acidosis post-estrés porque el rumen viene "
                "desestabilizado) y vigilar consumo individual durante "
                "los próximos 3-4 días."
            )

    # ─── CALOR ───
    if tipo == "calor":
        if etapa == "inicio":
            return (
                "El animal empieza a respirar más rápido para disipar "
                "calor. Si las noches no refrescan, no recupera. Baja el "
                "consumo de materia seca en las horas centrales y aumenta "
                "demanda de agua. Conviene anticipar sombra y horarios "
                "de alimentación antes que se instale el cuadro."
            )
        if etapa == "persistencia":
            return (
                "El animal ya está jadeando y reduce consumo en horas "
                "calurosas. Si la mezcla se calienta en comedero o "
                "fermenta, agrava el rechazo. La rumia baja, el agua "
                "se duplica y la ganancia diaria se resiente."
            )
        if etapa == "acumulacion":
            return (
                "Después de varios días de calor sin recuperación "
                "nocturna, el animal entra en deuda térmica. Mayor riesgo "
                "de muerte súbita en lotes encerrados, fertilidad caída "
                "en hembras y pérdida marcada de ganancia. Cada día que "
                "se prolonga, el impacto productivo se acumula."
            )
        if etapa == "recuperacion":
            return (
                "El calor cedió. Hay que evaluar el consumo de los "
                "próximos días — si bajó la rumia, vigilar acidosis y "
                "subir gradualmente el plano nutricional. La fertilidad "
                "y el peso pueden tardar 2-3 semanas en normalizarse."
            )

    # Fallback genérico
    if etapa == "inicio":
        return ("Evento que recién comienza. Prioridad: preparación "
                "antes de que el cuadro se instale.")
    if etapa == "persistencia":
        return ("Evento sostenido. Foco en sostener consumo y acceso "
                "al alimento.")
    if etapa == "acumulacion":
        return ("Estrés acumulado. Riesgo de pérdida productiva real "
                "si no se actúa con criterio.")
    if etapa == "recuperacion":
        return ("El cuadro cedió. Evaluar impacto residual y normalizar "
                "gradualmente.")
    return ""


def clasificar_etapa_evento(nivel_hoy: str,
                              dias_alerta_previos: int) -> str:
    """Clasifica la etapa del evento climático-productivo.

    Args:
      nivel_hoy: nivel productivo de HOY ('normal'|'atencion'|'operativo'|'critico')
      dias_alerta_previos: cuántos días consecutivos previos hubo alerta enviada

    Returns:
      'inicio' | 'persistencia' | 'acumulacion' | 'recuperacion' | 'estable'

    Etapas:
      - 'inicio': hoy hay alerta y no había antes
      - 'persistencia': hoy hay alerta y AYER también (1-2 días seguidos)
      - 'acumulacion': hoy hay alerta y vienen 3+ días seguidos
      - 'recuperacion': hoy NO hay alerta pero ayer/anteayer sí
      - 'estable': sin alertas hoy ni en los días previos
    """
    hay_alerta_hoy = nivel_hoy in ("atencion", "operativo", "critico")

    if hay_alerta_hoy:
        if dias_alerta_previos == 0:
            return "inicio"
        if dias_alerta_previos <= 2:
            return "persistencia"
        return "acumulacion"
    else:
        if dias_alerta_previos > 0:
            return "recuperacion"
        return "estable"


def clasificar_nivel_productivo(severidad: str, tipo: Optional[str] = None,
                                  contexto: Optional[Dict] = None) -> str:
    """Clasificador unificado de nivel productivo para una alerta de un día.

    Devuelve uno de: 'normal' | 'atencion' | 'operativo' | 'critico'.

    Usado por:
      - reporte semanal (semáforo por día)
      - alertas diarias (decidir si mandar al cliente)
      - pronóstico nocturno

    Filosofía: el nivel productivo NO es la severidad climática raw.
    Refleja impacto esperado sobre consumo, manejo, bienestar.

    `contexto` opcional: {
        'barro': bool,
        'lluvia_mm': float,
        'humedad_pct': float,
        'precip_3d_mm': float,
        'pelaje_mojado': bool,
        'dias_consecutivos_frio': int,
        'dias_barro_consecutivos': int,
        'temp_min': float,
        'categoria': str,
        'raza': str,
    }
    """
    sev = (severidad or "").lower()
    ctx = contexto or {}
    barro = bool(ctx.get("barro", False))
    lluvia = float(ctx.get("lluvia_mm", 0) or 0)
    humedad = float(ctx.get("humedad_pct", 0) or 0)
    precip_3d = float(ctx.get("precip_3d_mm", 0) or 0)
    pelaje_mojado = bool(ctx.get("pelaje_mojado", False))
    dias_frio = int(ctx.get("dias_consecutivos_frio", 0) or 0)
    dias_barro = int(ctx.get("dias_barro_consecutivos", 0) or 0)
    t_min = ctx.get("temp_min")

    # ─── Crítica del clima → crítico productivo ───
    if sev == "critica":
        return "critico"

    # ─── Warning del clima → atencion u operativo según contexto ───
    if sev == "warning":
        # Agravantes que elevan a operativo
        agravantes_operativos = (
            barro
            or precip_3d > 20
            or pelaje_mojado
            or dias_barro >= 2
            or dias_frio >= 2
            or (t_min is not None and t_min < 2 and lluvia > 5)
        )
        return "operativo" if agravantes_operativos else "atencion"

    # ─── Preventiva/info → atención ───
    if sev in ("preventiva", "info"):
        return "atencion"

    # ─── Sin alerta climática: revisar contexto residual ───
    if barro or precip_3d > 30:
        return "operativo"
    if lluvia > 10:
        return "atencion"
    if humedad >= 85:
        return "atencion"
    return "normal"


def clasificar_raza(raza: str = "") -> Dict:
    """Clasifica la raza para aplicar modificadores de sensibilidad térmica.

    Razas británicas (Angus, Hereford):
      - Buena tolerancia al frío seco (atenuar alertas si piso seco
        + sin viento fuerte + adultos)
      - Alta sensibilidad al calor húmedo (bajar umbral THI en 1 punto)
      - Alta sensibilidad a barro prolongado (+1 score ambiental)
      - Angus negro: máxima sensibilidad al calor (absorbe más radiación)
      - Toros británicos: muy alta sensibilidad térmica

    Razas índicas (Brahman, Nelore, etc): mejor adaptadas al calor,
    no se ajustan los umbrales hacia abajo. (Por ahora no penalizamos
    a las índicas — futuro: subir umbral 1 punto por mejor tolerancia.)

    Cruzas: tratan como británicas con sensibilidad reducida (a definir
    cuando aparezcan datos reales). Por ahora se evalúan como británicas.
    """
    r = (raza or "").lower().strip()
    flags = {
        "britanica": False,
        "angus": False,
        "angus_negro": False,
        "hereford": False,
        "indica": False,
        "cruza": False,
    }
    if not r:
        return flags

    # Razas índicas (cebú, brahman, nelore, etc.)
    if any(k in r for k in ("brahman", "nelore", "cebu", "cebú",
                              "índic", "indic", "gyr", "guzerá")):
        flags["indica"] = True
        return flags

    # Cruzas con índica
    if any(k in r for k in ("brangus", "braford", "santa gertrudis",
                              "bonsmara")):
        # Brangus = Angus x Brahman; Braford = Hereford x Brahman.
        # Tienen tolerancia intermedia. Por ahora las trato como
        # británicas (precaución), pero marcamos cruza para futuras
        # reglas más finas.
        flags["britanica"] = True
        flags["cruza"] = True
        return flags

    # Razas británicas puras
    if "angus" in r:
        flags["britanica"] = True
        flags["angus"] = True
        if "negr" in r or "black" in r:
            flags["angus_negro"] = True
        return flags
    if "hereford" in r:
        flags["britanica"] = True
        flags["hereford"] = True
        return flags
    if "shorthorn" in r:
        flags["britanica"] = True
        return flags

    return flags


def umbrales_thi_por_categoria(categoria: str = "",
                                  raza: str = "") -> Dict[str, int]:
    """Devuelve umbrales (preventiva/moderado/critico) según categoría.

    Categorías reconocidas (prompt v3):
      - terneros/as: 78 / 81 / 85
      - novillos:    76 / 81 / 85
      - vaquillonas: 77 / 80 / 84
      - vacas:       78 / 82 / 86
      - toros:       74 / 79 / 83
    """
    cat = (categoria or "").lower().strip()

    # Toros (los más sensibles por afectar fertilidad)
    if "toro" in cat:
        base = {"preventiva": 74, "moderado": 79, "critico": 83}

    # Terneros / terneras / destete / recría (los más livianos)
    elif any(k in cat for k in ("ternero", "ternera", "recría", "recria",
                                "destete", "destetado", "guacho")):
        base = {"preventiva": 78, "moderado": 81, "critico": 85}

    # Vaquillonas
    elif "vaquillona" in cat or "vaquillon" in cat:
        base = {"preventiva": 77, "moderado": 80, "critico": 84}

    # Vacas adultas / madres
    elif any(k in cat for k in ("vaca", "madre", "vientre", "criadora")):
        base = {"preventiva": 78, "moderado": 82, "critico": 86}
    elif cat == "cría" or cat == "cria" or cat.endswith(" cría") \
            or cat.endswith(" cria"):
        base = {"preventiva": 78, "moderado": 82, "critico": 86}
    else:
        # Default = novillos
        base = {"preventiva": 76, "moderado": 81, "critico": 85}

    # Aplicar modificador racial (-1 puntos si británicas, más si Angus
    # negro o toro británico)
    return _ajustar_umbrales_por_raza(base, raza, categoria)


def _ajustar_umbrales_por_raza(umbrales: Dict[str, int], raza: str,
                                  categoria: str) -> Dict[str, int]:
    """Ajusta umbrales THI hacia abajo si la raza es británica (más
    sensible al calor que las índicas o cruzas).

    - Británicas: -1 punto en todos los umbrales (THI 84 se comporta
      como 85, etc.)
    - Angus negro: -2 puntos (absorbe más radiación, dispara antes)
    - Toros británicos: -2 puntos (sensibilidad térmica adicional por
      tema reproductivo)
    - Cruzas británica x índica: -0.5 (redondeo a -1 si pega justo)
    """
    flags = clasificar_raza(raza)
    if not flags["britanica"]:
        return umbrales  # índicas / sin raza cargada → sin ajuste

    # Base británica: -1
    delta = -1
    cat = (categoria or "").lower()

    # Angus negro absorbe más radiación → -1 adicional (= -2 total)
    if flags["angus_negro"]:
        delta -= 1

    # Toros británicos → -1 adicional
    if "toro" in cat:
        delta -= 1

    # Cruzas británica x índica → atenuar a la mitad
    if flags["cruza"]:
        # En vez de -1, dejar -1 también pero podríamos volver a 0 si
        # tuviéramos datos. Por ahora compromiso: -1 fijo.
        delta = max(delta, -1)

    # Aplicar ajuste a los umbrales
    return {k: v + delta for k, v in umbrales.items()}


def _agregar_unico(lista: List[str], item: str) -> None:
    """Suma `item` a la lista si no hay ya uno cuyo prefijo coincida.

    Evita duplicar items cuando ya existe una recomendación del mismo
    "tema" (agua, comedero, sombra, etc.). Compara la primera palabra
    en mayúsculas (antes del ":") como key.
    """
    key = item.split(":", 1)[0].lower().strip()
    for existente in lista:
        if existente.split(":", 1)[0].lower().strip() == key:
            return  # ya hay una reco del mismo tema
    lista.append(item)


def evaluar_estres_calorico(
    thi_base: float,
    viento_kmh: Optional[float] = None,
    temp_min_nocturna: Optional[float] = None,
    sombra_m2_cab: Optional[float] = None,
    horas_thi_alto_ayer: int = 0,
    dias_calor_consecutivos: int = 0,
    categoria: str = "",
    raza: str = "",
    barro: bool = False,
    humedad_pct: Optional[float] = None,
) -> Dict:
    """Evalúa estrés calórico aplicando agravantes al THI base.

    `raza` permite ajustar sensibilidad: razas británicas (Angus,
    Hereford) son más sensibles al calor húmedo y al barro prolongado.
    Esto se aplica vía `umbrales_thi_por_categoria(..., raza)` y
    agravantes adicionales acá.

    Returns: dict con tipo_alerta, riesgo, thi_base, thi_ajustado,
    agravantes, detalles_agravantes y acciones (inmediata/operativa/nutricional).
    """
    flags_raza = clasificar_raza(raza)

    agravantes = 0
    detalles: List[str] = []

    if viento_kmh is not None and viento_kmh < 10:
        agravantes += 1
        detalles.append(f"Viento bajo ({viento_kmh:.0f} km/h)")
    if temp_min_nocturna is not None and temp_min_nocturna > 22:
        agravantes += 1
        detalles.append(f"Noches calurosas (mínima {temp_min_nocturna:.0f}°C)")
    if sombra_m2_cab is not None and sombra_m2_cab < 4:
        agravantes += 1
        detalles.append(f"Sombra insuficiente ({sombra_m2_cab:.1f} m²/cab)")
    if horas_thi_alto_ayer >= 4:
        agravantes += 1
        detalles.append(f"Calor sostenido el día anterior ({horas_thi_alto_ayer} hs)")
    if dias_calor_consecutivos >= 2:
        agravantes += 1
        detalles.append(f"{dias_calor_consecutivos} días consecutivos de calor")

    # ─── MODIFICADOR RACIAL EN CALOR ───
    # Razas británicas tienen alta sensibilidad al barro + humedad
    # combinados (estrés ambiental crónico). Sumamos +1 al score si
    # el lote tiene barro Y la humedad es alta (>70%).
    if flags_raza["britanica"] and barro and (humedad_pct or 0) >= 70:
        agravantes += 1
        detalles.append("Británica + barro + humedad alta (sensibilidad racial)")

    # Angus negro absorbe más radiación → +0 acá porque ya bajamos el
    # umbral 1 punto extra en _ajustar_umbrales_por_raza. No
    # duplicamos el efecto.

    thi_ajustado = round(thi_base + agravantes, 1)

    # Clasificar riesgo según umbrales de la categoría + raza
    umb = umbrales_thi_por_categoria(categoria, raza)
    if thi_ajustado >= umb["critico"]:
        riesgo = "critico"
    elif thi_ajustado >= umb["moderado"]:
        riesgo = "moderado"
    elif thi_ajustado >= umb["preventiva"]:
        riesgo = "preventiva"
    else:
        riesgo = "normal"

    # ─── UMBRALES ABSOLUTOS DE TEMPERATURA EXTREMA ─────────────
    # Sin importar humedad, una T° muy alta ya es peligrosa de por sí.
    # thi_base se calcula con T° del momento o máxima del día.
    # Inferimos la T° aproximada desde el THI base (ya viene en escala T+HR).
    # Para el boost preferimos usar la temperatura mínima nocturna como
    # referencia inversa (no aplica acá), por lo que solo confiamos en thi.
    nivel_orden = {"normal": 0, "leve": 1, "preventiva": 1,
                     "moderado": 2, "critico": 3}

    def _subir_calor(nuevo: str, motivo: str):
        nonlocal riesgo
        if nivel_orden.get(nuevo, 0) > nivel_orden.get(riesgo, 0):
            riesgo = nuevo
        if motivo and motivo not in detalles:
            detalles.append(motivo)

    # Si THI ≥ 88 (cercano a "extremo absoluto") → crítico garantizado
    # incluso si la categoría es vaca (umbral más alto)
    if thi_ajustado >= 88:
        _subir_calor("critico", "THI extremo (≥88)")

    # ─── BOOST DE NIVEL POR ACUMULACIÓN ──────────────────────────
    # La acumulación de estrés debe subir el semáforo aunque el evento
    # puntual no sea extremo. Reglas:
    #
    #   - 3+ días consecutivos de calor → subir un nivel
    #   - 8+ horas acumuladas con THI alto → subir un nivel
    #
    # 2 días o 4 horas ya suman agravante (arriba); acá disparamos un
    # boost adicional solo cuando se pasa el umbral fuerte.
    nivel_a_siguiente = {"normal": "preventiva",
                          "preventiva": "moderado",
                          "leve": "moderado",
                          "moderado": "critico",
                          "critico": "critico"}
    # El boost solo se aplica si el día puntual ya tiene algún nivel
    # de riesgo. Si está en "normal" (THI bajo, sin estrés del día), la
    # acumulación previa por sí sola no inventa una alerta.
    if riesgo != "normal":
        if dias_calor_consecutivos >= 3:
            siguiente = nivel_a_siguiente.get(riesgo, "moderado")
            _subir_calor(siguiente,
                          f"Acumulación: {dias_calor_consecutivos} días "
                          f"consecutivos de calor")
        if horas_thi_alto_ayer >= 8:
            siguiente = nivel_a_siguiente.get(riesgo, "moderado")
            _subir_calor(siguiente,
                          f"Acumulación: {horas_thi_alto_ayer} hs sostenidas "
                          f"con THI alto")

    # Determinar tipo de alerta
    if riesgo == "critico":
        tipo_alerta = "critica"
        acciones = {
            "inmediata": [
                "Manejo: suspender vacunas, pesadas y traslados",
                "Agua: revisar caudal, presión y temperatura "
                "(que no esté caliente)",
                "Sombra: verificar disponibilidad (mín 4 m²/cab)",
            ],
            "operativa": [
                "Alimentación: comida principal 5-7 hs y 19-21 hs",
                "Monitoreo: respiración >80/min = riesgo de muerte súbita",
                "Movimientos: postergar manejo en horas centrales (12-16)",
            ],
            "nutricional": [
                "Estabilidad ruminal: evitar cambios bruscos de dieta",
                "Fibra: mantener FDN 25-30%",
                "Acidosis: sumar buffers si hay rumia caída",
            ],
        }
    elif riesgo == "moderado":
        tipo_alerta = "moderada"
        acciones = {
            "inmediata": [
                "Manejo: evitar horas pico (12-16 hs)",
                "Agua: revisar bebederos y caudal",
            ],
            "operativa": [
                "Alimentación: adelantar comida a antes de las 8 hs",
                "Alimentación: segunda comida después de las 19 hs",
            ],
            "nutricional": [
                "Dieta: mantenerla estable, sin subas bruscas de energía",
                "Consumo: vigilar individual y rumia",
            ],
        }
    elif riesgo == "preventiva":
        tipo_alerta = "preventiva"
        acciones = {
            "inmediata": [
                "Agua: verificar disponibilidad y sombra antes del evento",
            ],
            "operativa": [
                "Alimentación: planificar suministro temprano",
            ],
            "nutricional": [
                "Energía: no subir nivel anticipadamente",
            ],
        }
    else:
        tipo_alerta = "ninguna"
        acciones = {"inmediata": [], "operativa": [], "nutricional": []}

    # ─── RECOS OPERATIVAS POR CONTEXTO (calor) ────────────────────
    # Sumamos recomendaciones específicas según las condiciones reales:
    # agua, comedero, sombra, calidad de la mezcla. Esto va más allá del
    # animal — apunta a la operación del feedlot/corral en condiciones
    # de estrés calórico.
    if riesgo in ("preventiva", "moderado", "critico"):
        # AGUA en calor: caudal, limpieza, temperatura, presión
        _agregar_unico(acciones["inmediata"],
                        "Agua: revisar caudal, limpieza y temperatura "
                        "(no debe estar caliente)")
        if riesgo in ("moderado", "critico"):
            _agregar_unico(acciones["operativa"],
                            "Agua: evitar bebederos vacíos en horas pico "
                            "(12-16 hs)")

        # COMEDERO con calor: evitar mezcla caliente/fermentada
        _agregar_unico(acciones["operativa"],
                        "Comedero: revisar sobrantes diariamente, "
                        "evitar mezcla caliente o fermentada")
        if riesgo == "critico":
            _agregar_unico(acciones["operativa"],
                            "Comedero: reducir tiempo de permanencia de la "
                            "ración, priorizar suministro fresco")
            _agregar_unico(acciones["operativa"],
                            "Controlar selección y rechazo en comedero")

        # SOMBRA Y DESCANSO en calor
        if (sombra_m2_cab is not None) and sombra_m2_cab < 4:
            _agregar_unico(acciones["inmediata"],
                            "Sombra: ampliar superficie disponible "
                            "(objetivo 4 m²/cab)")
        if riesgo in ("moderado", "critico"):
            _agregar_unico(acciones["operativa"],
                            "Sombra: evitar hacinamiento, revisar barro "
                            "en zonas de descanso")

    # Comedero húmedo (cuando hay barro / humedad alta combinada con calor)
    if barro or (humedad_pct is not None and humedad_pct >= 80):
        _agregar_unico(acciones["operativa"],
                        "Pellet/mezcla: vigilar humedad, polvillo y pérdida "
                        "de homogeneidad → puede aumentar selección y "
                        "afectar consumo")

    return {
        "tipo_alerta": tipo_alerta,
        "riesgo": riesgo,
        "thi_base": round(thi_base, 1),
        "thi_ajustado": thi_ajustado,
        "agravantes": agravantes,
        "detalles_agravantes": detalles,
        "umbrales": umb,
        "categoria": categoria,
        "acciones": acciones,
    }


def acciones_a_lista_plana(acciones_dict: Dict) -> List[str]:
    """Aplana el dict de acciones a lista con prefijos para emails/WhatsApp."""
    out = []
    for cat, items in (("INMEDIATA", acciones_dict.get("inmediata", [])),
                         ("OPERATIVA", acciones_dict.get("operativa", [])),
                         ("NUTRICIONAL", acciones_dict.get("nutricional", []))):
        for it in items:
            out.append(f"[{cat}] {it}")
    return out


# =====================================================================
# EVALUADOR DE ESTRÉS POR FRÍO (score)
# =====================================================================
# Score:
#   - Temp <5°C → +2 ; <10°C → +1
#   - Viento >20 km/h → +2 ; >10 → +1
#   - Lluvia >5 mm → +1
#   - Barro → +2
# Clasificación:
#   - leve ≥2, moderado ≥4, crítico ≥6

def evaluar_estres_frio(
    temp_c: Optional[float] = None,
    viento_kmh: Optional[float] = None,
    lluvia_mm: float = 0,
    barro: bool = False,
    helada: bool = False,
    pelaje_mojado: bool = False,
    raza: str = "",
    categoria: str = "",
    dias_consecutivos_frio: int = 0,
    dias_barro_consecutivos: int = 0,
) -> Dict:
    """Evalúa estrés por frío con sistema de score.

    `raza` y `categoria` permiten atenuar la alerta si:
      - raza británica (Angus, Hereford) tolera bien el frío seco
      - el frío es seco (sin viento fuerte ni lluvia ni barro)
      - los animales son adultos (no terneros)

    `dias_consecutivos_frio` y `dias_barro_consecutivos` aplican boost
    de severidad por acumulación: el frío sostenido o el barro persistente
    suben el nivel del semáforo aunque el evento puntual no sea extremo.
    """
    score = 0
    detalles: List[str] = []

    if temp_c is not None:
        if temp_c < 5:
            score += 2
            detalles.append(f"Temperatura muy baja ({temp_c:.0f}°C)")
        elif temp_c < 10:
            score += 1
            detalles.append(f"Temperatura baja ({temp_c:.0f}°C)")

    if viento_kmh is not None:
        if viento_kmh > 20:
            score += 2
            detalles.append(f"Viento fuerte ({viento_kmh:.0f} km/h)")
        elif viento_kmh > 10:
            score += 1
            detalles.append(f"Viento elevado ({viento_kmh:.0f} km/h)")

    if lluvia_mm and lluvia_mm > 5:
        score += 1
        detalles.append(f"Lluvia significativa ({lluvia_mm:.0f} mm)")

    if barro:
        score += 2
        detalles.append("Barro presente")

    # Helada: explícita por flag o inferida por temperatura ≤0°C
    helada_explicita = bool(helada)
    helada_inferida = (temp_c is not None and temp_c <= 0)
    if helada_explicita or helada_inferida:
        score += 1
        detalles.append("Helada")

    # Pelaje mojado: explícito por flag o inferido por lluvia + frío
    pelaje_explicito = bool(pelaje_mojado)
    pelaje_inferido = ((lluvia_mm and lluvia_mm > 5) and
                         (temp_c is not None and temp_c < 10))
    if pelaje_explicito or pelaje_inferido:
        score += 2
        detalles.append("Pelaje mojado (lluvia + frío)")

    # ─── MODIFICADOR RACIAL EN FRÍO ───
    # Razas británicas (Angus, Hereford) toleran bien el frío seco si:
    #   - el piso está seco (sin barro, sin lluvia significativa)
    #   - el viento no es fuerte (≤ 20 km/h)
    #   - los animales son adultos (no terneros / recría)
    # En esos casos, atenuamos el score en 1 punto (mín 0).
    # Esto evita generar alertas innecesarias con un -2°C seco y
    # un Angus adulto pastoreando tranquilo.
    flags_raza = clasificar_raza(raza)
    cat_l = (categoria or "").lower()
    es_adulto = not any(k in cat_l for k in (
        "ternero", "ternera", "recría", "recria", "destete",
        "destetado", "guacho",
    ))
    frio_seco = (
        not barro
        and (not lluvia_mm or lluvia_mm <= 5)
        and not pelaje_mojado
        and (viento_kmh is None or viento_kmh <= 20)
    )
    if flags_raza["britanica"] and frio_seco and es_adulto and score > 0:
        score = max(0, score - 1)
        detalles.append("Atenuado: raza británica adulta + frío seco")

    if score >= 6:
        riesgo = "critico"
    elif score >= 4:
        riesgo = "moderado"
    elif score >= 2:
        riesgo = "leve"
    else:
        riesgo = "normal"

    # ─── UMBRALES ABSOLUTOS (boost) ────────────────────────────
    # Cualquiera de estos eleva el riesgo mínimo, sin importar el score.
    # Pensado para casos puntuales graves que el score no captaría solo
    # (ej: helada -5°C con viento calmo = score 2 leve, pero ES grave).

    nivel_orden = {"normal": 0, "leve": 1, "moderado": 2, "critico": 3}
    inverso = {0: "normal", 1: "leve", 2: "moderado", 3: "critico"}

    def _subir_a(nuevo_riesgo: str, motivo: str):
        nonlocal riesgo
        if nivel_orden[nuevo_riesgo] > nivel_orden[riesgo]:
            riesgo = nuevo_riesgo
        if motivo and motivo not in detalles:
            detalles.append(motivo)

    # ─── DETECTAR AGRAVANTES SERIOS PARA HABILITAR ESCALAR A CRÍTICO ───
    # Filosofía: el frío seco con animales adultos británicos NO es
    # automáticamente crítico. Solo se escala a "critico" si hay
    # combinación de agravantes que indiquen impacto productivo real:
    #   - barro persistente o lluvia (deterioro de ración + acceso)
    #   - animales mojados (pierden capacidad termorreguladora)
    #   - viento fuerte (wind chill + estrés respiratorio)
    #   - varios días consecutivos (acumulación)
    #   - terneros / destete / recría (categoría sensible)
    cat_l = (categoria or "").lower()
    es_cat_sensible = any(k in cat_l for k in (
        "ternero", "ternera", "recría", "recria", "destete",
        "destetado", "guacho",
    ))
    agravantes_serios = sum([
        bool(barro),
        bool(pelaje_mojado),
        bool(lluvia_mm and lluvia_mm > 5),
        bool(viento_kmh and viento_kmh > 25),
        bool(dias_consecutivos_frio >= 2),
        bool(dias_barro_consecutivos >= 2),
        bool(es_cat_sensible),
    ])

    def _subir_a_si_serio(temp_motivo: str):
        """Sube a crítico SOLO si hay ≥3 agravantes serios Y la
        temperatura es realmente baja (T° ≤ 0°C). Si no se cumplen
        ambas, queda en moderado.

        Filosofía (feedback agente externo CRM 360): la palabra
        'crítico' se reserva para eventos con riesgo de mortalidad o
        pérdida productiva mayor — combinaciones tipo lluvia fría +
        barro severo + viento fuerte + helada + varios días sostenidos
        + categoría sensible. NO para frío seco con un solo agravante."""
        temp_realmente_baja = (temp_c is not None and temp_c <= 0)
        if agravantes_serios >= 3 and temp_realmente_baja:
            _subir_a("critico",
                      f"{temp_motivo} + {agravantes_serios} agravantes "
                      f"serios + T° {temp_c}°C")
        else:
            _subir_a("moderado",
                      f"{temp_motivo} (sin combinación severa para crítico)")

    if temp_c is not None:
        if temp_c <= -10:
            # Helada extrema absoluta: siempre crítica (sale del rango
            # razonable de tolerancia incluso para razas británicas adultas).
            _subir_a("critico", f"Helada extrema ({temp_c:.0f}°C)")
        elif temp_c <= -5:
            # Severa: crítica SOLO con agravantes; si no, moderada.
            _subir_a_si_serio(f"Helada severa ({temp_c:.0f}°C)")
        elif temp_c <= 0:
            _subir_a("moderado", f"Helada esperada ({temp_c:.0f}°C)")

    if viento_kmh is not None:
        if viento_kmh >= 90:
            _subir_a("critico",
                      f"Tormenta de viento ({viento_kmh:.0f} km/h)")
        elif viento_kmh >= 70:
            _subir_a("moderado",
                      f"Vientos muy fuertes ({viento_kmh:.0f} km/h)")

    if lluvia_mm:
        if lluvia_mm >= 100:
            _subir_a("critico", f"Lluvia torrencial ({lluvia_mm:.0f} mm)")
        elif lluvia_mm >= 50:
            _subir_a("moderado", f"Lluvia intensa ({lluvia_mm:.0f} mm)")

    # ─── BOOST POR ACUMULACIÓN ────────────────────────────────────
    # Frío sostenido: +1 a riesgo si 2+ días consecutivos.
    #   Justificación: el frío de varios días acumulados gasta reservas
    #   de grasa, deprime inmunidad → riesgo energético + sanitario.
    # Barro persistente: +1 a riesgo si 3+ días con barro.
    #   Justificación: barro prolongado aumenta gasto energético al
    #   caminar, riesgo de pododermatitis y problemas reproductivos.
    # La acumulación NO debe escalar moderado → crítico por sí sola.
    # Filosofía: el rojo se reserva a combinación severa real. La
    # acumulación de días moderados sigue siendo moderado, no crítico.
    nivel_a_siguiente = {"normal": "leve",
                          "leve": "moderado",
                          "moderado": "moderado",  # ← antes era "critico"
                          "critico": "critico"}
    if dias_consecutivos_frio >= 2 and riesgo != "normal":
        siguiente = nivel_a_siguiente.get(riesgo, "moderado")
        _subir_a(siguiente,
                  f"Acumulación: {dias_consecutivos_frio} días consecutivos "
                  f"de frío (riesgo energético/sanitario)")
    if dias_barro_consecutivos >= 3 and riesgo != "normal":
        siguiente = nivel_a_siguiente.get(riesgo, "moderado")
        _subir_a(siguiente,
                  f"Acumulación: barro persistente "
                  f"({dias_barro_consecutivos} días)")

    # ─── FRENO FINAL: NO CRÍTICO SI NO HAY AGRAVANTES SERIOS ───
    # Si el score llevó a crítico pero el contexto es "frío seco con
    # adultos no sensibles", bajamos a moderado. El crítico requiere
    # combinación de factores que indiquen impacto productivo real.
    if (riesgo == "critico"
            and agravantes_serios < 2
            and (temp_c is None or temp_c > -10)):
        riesgo = "moderado"
        detalles.append(
            "↓ Re-clasificado a moderado: frío seco sin combinación "
            "de agravantes serios (sin barro, sin lluvia, sin viento "
            "fuerte, sin terneros, sin acumulación)"
        )

    if riesgo == "critico":
        acciones = {
            "inmediata": [
                "Reparo: proveer (monte, cortina forestal o rollos apilados como cortaviento)",
                "Barro: evitar profundidad en corral",
                "Viento: reducir exposición directa",
            ],
            "operativa": [
                "Alimentación: adelantar comida principal a la tarde",
                "Cama: asegurar superficie seca",
                "Drenaje: mejorar evacuación del corral",
            ],
            "nutricional": [
                "Concentrado: subir fibra activa 2 puntos "
                "(ej. 12% → 14%) por 3-4 días",
                "Vuelta: gradual al nivel base, sin cambios bruscos",
                "Estabilidad ruminal: NO subir grano de golpe",
            ],
        }
    elif riesgo == "moderado":
        acciones = {
            "inmediata": [
                "Reparo: verificar disponibilidad",
                "Agua: revisar bebederos al amanecer (hielo)",
            ],
            "operativa": [
                "Alimentación: adelantar comida a la tarde",
                "Cama: asegurar superficie seca",
            ],
            "nutricional": [
                "Concentrado: subir fibra activa 1-2 puntos "
                "(ej. 12% → 14%) por 3-4 días",
                "Vuelta: gradual al nivel base, sin cambios bruscos",
                "Estabilidad ruminal: NO subir grano de golpe",
            ],
        }
    else:
        acciones = {"inmediata": [], "operativa": [], "nutricional": []}

    # ─── RECOS OPERATIVAS POR CONTEXTO (frío / barro / lluvia) ────
    # Igual que en calor: sumamos recomendaciones específicas según las
    # condiciones reales (agua congelada, comedero con humedad, acceso
    # con barro, etc.).
    if riesgo in ("leve", "moderado", "critico"):
        # AGUA en frío: congelamiento + acceso
        if temp_c is not None and temp_c <= 2:
            _agregar_unico(acciones["inmediata"],
                            "Agua: revisar congelamiento, romper hielo "
                            "y asegurar disponibilidad")
        if barro:
            _agregar_unico(acciones["inmediata"],
                            "Agua: controlar barro alrededor de bebederos")

        # COMEDERO con humedad / lluvia
        if (lluvia_mm and lluvia_mm > 5) or pelaje_mojado:
            _agregar_unico(acciones["operativa"],
                            "Pellet/mezcla: revisar humedad, evitar "
                            "permanencia prolongada de ración mojada")
            _agregar_unico(acciones["operativa"],
                            "Comedero: controlar selección, evitar "
                            "sobrantes fermentados o deteriorados")

        # ACCESO al comedero con barro
        if barro:
            _agregar_unico(acciones["inmediata"],
                            "Acceso al comedero: revisar entrada, "
                            "evitar barro profundo en zona de comida")
            _agregar_unico(acciones["operativa"],
                            "Drenaje: revisar y reducir pérdida "
                            "energética por humedad/barro")

        # PIE / CAMA si barro persistente
        if dias_barro_consecutivos >= 3:
            _agregar_unico(acciones["operativa"],
                            "Cama: priorizar superficie seca para "
                            "descanso (riesgo de pododermatitis)")

    return {
        "tipo": "frio",
        "riesgo": riesgo,
        "score": score,
        "detalles": detalles,
        "acciones": acciones,
    }


# =====================================================================
# DOMINANTE CALOR vs FRÍO — formato JSON oficial (spec usuario)
# =====================================================================

# Severidad → ranking numérico para comparar
_RANK_SEVERIDAD = {
    "ninguno": 0,
    "normal": 0,
    "leve": 1,
    "preventiva": 2,
    "moderado": 3,
    "moderada": 3,
    "critico": 4,
    "critica": 4,
}


def evaluar_estres_ambiental(
    clima: Dict,
    ambiente: Optional[Dict] = None,
    historial: Optional[Dict] = None,
    categoria: str = "",
    raza: str = "",
) -> Dict:
    """Evalúa calor + frío y devuelve el dominante en formato JSON oficial.

    Input (clima/ambiente/historial — todos opcionales):
      clima: {thi, viento_kmh, min_nocturna, temperatura, lluvia_mm,
              thi_proyectado: [...]}
      ambiente: {sombra_m2_cab, barro}
      historial: {horas_thi_alto_ayer, dias_consecutivos_calor}

    Output (formato exacto del prompt):
      {
        "tipo": "calor" | "frio",
        "nivel": "preventiva" | "moderado" | "critico" | "ninguno",
        "acciones": {"inmediata":[], "operativa":[], "nutricional":[]}
      }
    """
    ambiente = ambiente or {}
    historial = historial or {}
    clima = clima or {}

    # Aceptar claves v3 o las viejas
    min_nocturna = (clima.get("minima_nocturna_c")
                      if clima.get("minima_nocturna_c") is not None
                      else clima.get("min_nocturna"))

    # Aceptar ambas claves de historial:
    #   - "horas_acumuladas_arriba_umbral" (formato del prompt maestro)
    #   - "horas_thi_alto_ayer" (legacy interno)
    horas_acum = (historial.get("horas_acumuladas_arriba_umbral")
                    if historial.get("horas_acumuladas_arriba_umbral") is not None
                    else historial.get("horas_thi_alto_ayer", 0))

    # ---- CALOR ----
    eval_calor = evaluar_estres_calorico(
        thi_base=float(clima.get("thi", 0)),
        viento_kmh=clima.get("viento_kmh"),
        temp_min_nocturna=min_nocturna,
        sombra_m2_cab=ambiente.get("sombra_m2_cab"),
        horas_thi_alto_ayer=int(horas_acum or 0),
        dias_calor_consecutivos=int(historial.get("dias_consecutivos_calor", 0)),
        categoria=categoria,
        raza=raza,
        barro=bool(ambiente.get("barro", False)),
        humedad_pct=clima.get("humedad_pct") or clima.get("hr"),
    )

    # Override para preventiva: si THI proyectado ≥85 (futuro) y aún no estás
    # en crítico, marcar como preventiva
    proy = clima.get("thi_proyectado") or []
    if proy:
        try:
            max_proy = max(float(x) for x in proy)
        except (TypeError, ValueError):
            max_proy = 0
        if max_proy >= 85 and eval_calor["riesgo"] in ("normal", "leve"):
            eval_calor["tipo_alerta"] = "preventiva"
            # acciones preventiva mínimas
            eval_calor["acciones"] = {
                "inmediata": ["Verificar agua y sombra antes del evento"],
                "operativa": ["Planificar alimentación temprana"],
                "nutricional": [],
            }

    # ---- FRÍO ----
    # Para frío usamos la T° MÍNIMA nocturna si está disponible
    # (clave para detectar heladas). Si no, fallback a la del día.
    temp_para_frio = min_nocturna
    if temp_para_frio is None:
        temp_para_frio = (clima.get("temperatura_actual_c")
                            if clima.get("temperatura_actual_c") is not None
                            else clima.get("temperatura"))

    eval_frio = evaluar_estres_frio(
        temp_c=temp_para_frio,
        viento_kmh=clima.get("viento_kmh"),
        lluvia_mm=float(clima.get("lluvia_mm", 0) or 0),
        barro=bool(ambiente.get("barro", False)),
        helada=bool(clima.get("helada", False)),
        pelaje_mojado=bool(ambiente.get("pelaje_mojado", False)),
        raza=raza,
        categoria=categoria,
        dias_consecutivos_frio=int(historial.get("dias_consecutivos_frio", 0)),
        dias_barro_consecutivos=int(historial.get("dias_barro_consecutivos", 0)),
    )

    # ---- DOMINANTE ----
    rank_calor = _RANK_SEVERIDAD.get(
        eval_calor.get("tipo_alerta", "ninguno"),
        _RANK_SEVERIDAD.get(eval_calor["riesgo"], 0),
    )
    rank_frio = _RANK_SEVERIDAD.get(eval_frio["riesgo"], 0)

    # Generar alerta solo si hay riesgo moderado/crítico (o preventiva calor)
    if rank_calor >= rank_frio and rank_calor >= 2:  # 2 = preventiva o más
        nivel = eval_calor.get("tipo_alerta", "ninguno")
        if nivel in ("critica",):
            nivel = "critico"
        elif nivel in ("moderada",):
            nivel = "moderado"
        return {
            "tipo": "calor",
            "nivel": nivel,
            "acciones": eval_calor["acciones"],
        }
    if rank_frio >= 3:  # moderado/crítico de frío
        return {
            "tipo": "frio",
            "nivel": eval_frio["riesgo"],
            "acciones": eval_frio["acciones"],
        }

    # Sin alerta digna
    return {
        "tipo": "calor" if rank_calor >= rank_frio else "frio",
        "nivel": "ninguno",
        "acciones": {"inmediata": [], "operativa": [], "nutricional": []},
    }


# =====================================================================
# COMPOSICIÓN DE MENSAJES (WhatsApp corto + Email detallado)
# Formato oficial según spec del usuario.
# =====================================================================

def _ranking_dominante(clima, ambiente, historial, categoria, raza=""):
    """Helper interno: obtiene los dos evaluadores y el dominante."""
    ambiente = ambiente or {}
    historial = historial or {}
    clima = clima or {}

    # Aceptar tanto las claves v3 (más explícitas) como las viejas
    min_nocturna = (clima.get("minima_nocturna_c")
                      if clima.get("minima_nocturna_c") is not None
                      else clima.get("min_nocturna"))

    # Aceptar ambas claves: "horas_acumuladas_arriba_umbral" (prompt
    # maestro) y "horas_thi_alto_ayer" (legacy interno).
    horas_acum = (historial.get("horas_acumuladas_arriba_umbral")
                    if historial.get("horas_acumuladas_arriba_umbral") is not None
                    else historial.get("horas_thi_alto_ayer", 0))

    eval_calor = evaluar_estres_calorico(
        thi_base=float(clima.get("thi", 0)),
        viento_kmh=clima.get("viento_kmh"),
        temp_min_nocturna=min_nocturna,
        sombra_m2_cab=ambiente.get("sombra_m2_cab"),
        horas_thi_alto_ayer=int(horas_acum or 0),
        dias_calor_consecutivos=int(historial.get("dias_consecutivos_calor", 0)),
        categoria=categoria,
        raza=raza,
        barro=bool(ambiente.get("barro", False)),
        humedad_pct=clima.get("humedad_pct") or clima.get("hr"),
    )

    proy = clima.get("thi_proyectado") or []
    if proy:
        try:
            max_proy = max(float(x) for x in proy)
        except (TypeError, ValueError):
            max_proy = 0
        if max_proy >= 85 and eval_calor["riesgo"] in ("normal", "leve"):
            eval_calor["tipo_alerta"] = "preventiva"

    # Para evaluar frío usamos la T° MÍNIMA NOCTURNA si está disponible
    # (clave para detectar heladas). Si no, fallback a temperatura del día.
    temp_para_frio = min_nocturna
    if temp_para_frio is None:
        temp_para_frio = (clima.get("temperatura_actual_c")
                            if clima.get("temperatura_actual_c") is not None
                            else clima.get("temperatura"))

    eval_frio = evaluar_estres_frio(
        temp_c=temp_para_frio,
        viento_kmh=clima.get("viento_kmh"),
        lluvia_mm=float(clima.get("lluvia_mm", 0) or 0),
        barro=bool(ambiente.get("barro", False)),
        helada=bool(clima.get("helada", False)),
        pelaje_mojado=bool(ambiente.get("pelaje_mojado", False)),
        raza=raza,
        categoria=categoria,
        dias_consecutivos_frio=int(historial.get("dias_consecutivos_frio", 0)),
        dias_barro_consecutivos=int(historial.get("dias_barro_consecutivos", 0)),
    )

    rank_calor = _RANK_SEVERIDAD.get(
        eval_calor.get("tipo_alerta", "ninguno"),
        _RANK_SEVERIDAD.get(eval_calor["riesgo"], 0),
    )
    rank_frio = _RANK_SEVERIDAD.get(eval_frio["riesgo"], 0)

    if rank_calor >= rank_frio:
        return "calor", eval_calor, eval_frio
    return "frio", eval_calor, eval_frio


def _impacto_calor(nivel: str, clima: Optional[Dict] = None,
                     ambiente: Optional[Dict] = None,
                     historial: Optional[Dict] = None,
                     categoria: str = "",
                     raza: str = "") -> List[str]:
    """Impacto sobre consumo en calor — rango calculado dinámicamente.

    Reglas:
      - leve: caída leve o sin cambios
      - moderado: 5-12%
      - critico: 10-20%
      - extremo / prolongado: puede superar 20%

    Agravantes (cada uno empuja el rango hacia arriba):
      - noches calurosas (T° min > 22)
      - días consecutivos de calor (≥3)
      - razas británicas negras (Angus negro)
      - animales pesados (vacas adultas, novillos terminación)
      - mala disponibilidad de agua (no detectable directamente)
      - sombra insuficiente (<4 m²/cab)
      - barro / mezcla deteriorada
    """
    clima = clima or {}
    ambiente = ambiente or {}
    historial = historial or {}

    if nivel == "critico":
        rango_min, rango_max = 10, 20
        items = []
    elif nivel == "moderado":
        rango_min, rango_max = 5, 12
        items = []
    elif nivel == "preventiva":
        return [
            "Riesgo bajo si se actúa con anticipación",
            "Monitorear consumo y comportamiento",
        ]
    else:
        return [
            "Atención si el evento se prolonga",
            "Monitorear consumo y comportamiento",
        ]

    # Agravantes que empujan hacia arriba
    agrav_n = 0
    motivos_extremo = []

    # Noches calurosas
    t_min = clima.get("min_nocturna") or clima.get("minima_nocturna_c")
    if t_min is not None and t_min > 22:
        agrav_n += 1
        motivos_extremo.append("noches calurosas")
    # Días consecutivos
    dias = int(historial.get("dias_consecutivos_calor", 0))
    if dias >= 3:
        agrav_n += 2  # peso doble: prolongado
        motivos_extremo.append(f"{dias} días consecutivos")
    elif dias >= 2:
        agrav_n += 1
    # Sombra insuficiente
    sombra = ambiente.get("sombra_m2_cab")
    if sombra is not None and sombra < 4:
        agrav_n += 1
        motivos_extremo.append("sombra insuficiente")
    # Raza Angus negro / británica
    flags_raza = clasificar_raza(raza)
    if flags_raza["angus_negro"]:
        agrav_n += 1
        motivos_extremo.append("Angus negro (más absorción de radiación)")
    elif flags_raza["britanica"]:
        agrav_n += 1
    # Categoría: animales pesados (vacas adultas, novillos terminación)
    cat = (categoria or "").lower()
    if any(k in cat for k in ("vaca", "novillo", "toro", "madre", "vientre")):
        agrav_n += 1  # los pesados acumulan más calor
    # Barro / humedad alta — deterioro de mezcla
    barro = bool(ambiente.get("barro", False))
    humedad = clima.get("humedad_pct") or clima.get("hr")
    if barro or (humedad is not None and humedad >= 80):
        agrav_n += 1
        motivos_extremo.append("comedero/mezcla con humedad")

    # Si hay 4+ agravantes y el nivel es crítico → escenario "extremo / prolongado"
    if nivel == "critico" and agrav_n >= 4:
        items.append(
            f"**Riesgo de caída de consumo de materia seca > 20%** "
            f"si el escenario se prolonga sin medidas"
        )
        items.append(
            "Riesgo de muerte súbita en lotes encerrados sin manejo"
        )
        items.append("Baja fertilidad sostenida")
        items.append("Pérdida severa de ganancia diaria")
    else:
        # Empujar el rango hacia arriba según agravantes acumulados
        # Cada agravante mueve el rango +1-2 pts hasta el techo
        ajuste = min(agrav_n * 2, rango_max - rango_min)
        rmin = rango_min + ajuste // 2
        rmax = min(rango_max + ajuste, rango_max + 5)
        items.append(
            f"Riesgo de caída de consumo de materia seca "
            f"{rmin}-{rmax}% (estimación según contexto)"
        )
        if nivel == "critico":
            items.append("Riesgo de muerte súbita si la noche no afloja")
            items.append("Baja fertilidad")
            items.append("Pérdida severa de ganancia diaria")
        else:  # moderado
            items.append("Menor ganancia diaria de peso")
            items.append("Mayor consumo de agua")

    if motivos_extremo and len(items) < 5:
        items.append(
            f"Factores que empujan el impacto: {', '.join(motivos_extremo)}"
        )
    # Disclaimer técnico: el impacto real depende del contexto del lote
    items.append(
        "ℹ️ El impacto real depende también del acceso a agua, sombra, "
        "calidad de la mezcla, manejo de comederos y recuperación nocturna"
    )
    return items


def _impacto_frio(nivel: str, clima: Optional[Dict] = None,
                    ambiente: Optional[Dict] = None,
                    historial: Optional[Dict] = None) -> List[str]:
    """Impacto sobre consumo en frío — depende mucho de SI hay barro/lluvia.

    Frío seco:
      - el consumo se mantiene o sube (los animales comen más para
        cubrir mayor mantenimiento)
    Frío + barro / lluvia / humedad:
      - puede haber caída de consumo
      - mayor selección
      - menor acceso al comedero
      - deterioro físico de la ración
    """
    clima = clima or {}
    ambiente = ambiente or {}
    historial = historial or {}

    if nivel not in ("leve", "moderado", "critico"):
        return ["Atención si el evento se prolonga"]

    barro = bool(ambiente.get("barro", False))
    lluvia = float(clima.get("lluvia_mm", 0) or 0)
    pelaje_mojado = bool(ambiente.get("pelaje_mojado", False))
    dias_barro = int(historial.get("dias_barro_consecutivos", 0))
    dias_frio = int(historial.get("dias_consecutivos_frio", 0))

    # Frío seco vs húmedo/barroso
    es_humedo = barro or lluvia > 5 or pelaje_mojado or dias_barro >= 2

    items = []

    # Aumento de requerimiento (siempre que haya frío significativo)
    if nivel == "critico":
        req_min, req_max = 25, 40
    elif nivel == "moderado":
        req_min, req_max = 10, 15
    else:  # leve
        req_min, req_max = 5, 10
    if dias_frio >= 3:
        req_max += 5
    items.append(
        f"Aumento {req_min}-{req_max}% del requerimiento de mantenimiento"
    )

    if es_humedo:
        # Caída de consumo + selección + deterioro
        if nivel == "critico":
            items.append(
                "Riesgo alto de **caída de consumo** por barro, lluvia "
                "y deterioro físico de la ración"
            )
            items.append(
                "Mayor selección en comedero y menor acceso al alimento"
            )
            items.append("Riesgo de hipotermia y enfermedad respiratoria")
            items.append("Pérdida acelerada de condición corporal")
        elif nivel == "moderado":
            items.append(
                "Posible caída de consumo si la mezcla pierde calidad "
                "física (humedad / barro)"
            )
            items.append("Mayor consumo energético")
            items.append("Pérdida lenta de condición si se prolonga")
        else:  # leve
            items.append("Vigilar acceso al comedero y calidad de la mezcla")
    else:
        # Frío seco: el consumo se mantiene o aumenta
        if nivel == "critico":
            items.append(
                "Frío SECO: consumo se mantiene o sube (más demanda "
                "energética)"
            )
            items.append("Riesgo de hipotermia si no hay reparo")
            items.append("Pérdida de condición corporal si se prolonga")
        elif nivel == "moderado":
            items.append(
                "Frío SECO: consumo estable, puede aumentar levemente"
            )
            items.append("Mayor consumo energético sostenido")
        else:  # leve
            items.append("Consumo estable; vigilar agua y reparo")

    if dias_barro >= 3:
        items.append(
            f"Acumulación: {dias_barro} días con barro persistente "
            f"→ riesgo de pododermatitis"
        )
    # Disclaimer técnico: contexto modula el impacto real
    items.append(
        "ℹ️ El impacto real depende del estado del corral, drenaje, "
        "acceso al comedero, calidad física de la mezcla y disponibilidad "
        "de reparos"
    )
    return items


def _recos_calor(nivel: str, categoria: str = "",
                   clima: Optional[Dict] = None,
                   ambiente: Optional[Dict] = None) -> Dict[str, List[str]]:
    """Recos de calor para el simulador / formato JSON v3.

    `clima` y `ambiente` opcionales: si vienen, sumamos recos
    contextuales (sombra insuficiente, barro+humedad, etc).
    """
    if nivel == "critico":
        recos = {
            "inmediatas": [
                "Manejo: suspender vacunas, pesadas y traslados",
                "Agua: revisar caudal, presión y temperatura "
                "(que no esté caliente)",
                "Sombra: verificar disponibilidad (mín 4 m²/cab)",
            ],
            "operativas": [
                "Alimentación: comida principal 5-7 hs y 19-21 hs",
                "Monitoreo: jadeo >80 resp/min = riesgo de muerte súbita",
                "Movimientos: postergar manejo en horas centrales (12-16)",
            ],
            "nutricionales": [
                "Energía/almidón: NO subir — agrava la fermentación",
                "Estabilidad ruminal: sin cambios bruscos de dieta",
                "Fibra (FDN): mantener 25-30%; sumar buffer si hay acidosis",
            ],
        }
    elif nivel == "moderado":
        recos = {
            "inmediatas": [
                "Manejo: evitar horas pico (12-16 hs)",
                "Agua: revisar bebederos y caudal",
            ],
            "operativas": [
                "Alimentación: adelantar comida a antes de las 8 hs",
                "Alimentación: segunda comida después de las 19 hs",
            ],
            "nutricionales": [
                "Dieta: mantenerla estable, sin subas bruscas de energía",
                "Consumo: vigilar individual y rumia",
            ],
        }
    elif nivel == "preventiva":
        recos = {
            "inmediatas": [
                "Agua: verificar disponibilidad y sombra antes del evento"
            ],
            "operativas": [
                "Alimentación: planificar suministro temprano"
            ],
            "nutricionales": [
                "Energía: no subir nivel anticipadamente"
            ],
        }
    else:
        return {"inmediatas": [], "operativas": [], "nutricionales": []}

    # ─── RECOS CONTEXTUALES POR CONDICIONES REALES ───
    clima = clima or {}
    ambiente = ambiente or {}
    sombra_m2 = ambiente.get("sombra_m2_cab")
    barro = bool(ambiente.get("barro", False))
    humedad = clima.get("humedad_pct") or clima.get("hr")

    # Sombra insuficiente
    if sombra_m2 is not None and sombra_m2 < 4:
        _agregar_unico(recos["inmediatas"],
                        f"Sombra: ampliar superficie disponible "
                        f"(actual {sombra_m2:.1f} m²/cab, objetivo ≥4)")

    # Comedero con calor
    if nivel in ("moderado", "critico"):
        _agregar_unico(recos["operativas"],
                        "Comedero: revisar sobrantes diariamente, evitar "
                        "mezcla caliente o fermentada")
        if nivel == "critico":
            _agregar_unico(recos["operativas"],
                            "Comedero: reducir tiempo de permanencia de "
                            "ración, priorizar suministro fresco")
            _agregar_unico(recos["operativas"],
                            "Comedero: controlar selección y rechazo")

    # Sombra y descanso
    if nivel in ("moderado", "critico"):
        _agregar_unico(recos["operativas"],
                        "Sombra: evitar hacinamiento, revisar barro "
                        "en zonas de descanso")

    # Pellet/mezcla con humedad alta + barro
    if barro or (humedad is not None and humedad >= 80):
        _agregar_unico(recos["operativas"],
                        "Pellet/mezcla: vigilar humedad, polvillo y pérdida "
                        "de homogeneidad → puede aumentar selección y "
                        "afectar consumo")

    return recos


def _recos_frio(nivel: str, clima: Optional[Dict] = None,
                  ambiente: Optional[Dict] = None) -> Dict[str, List[str]]:
    """Recos de frío para el simulador / formato JSON v3.

    `clima` y `ambiente` son opcionales. Si vienen, sumamos recos
    operativas según el contexto real (lluvia, barro, helada, viento).
    """
    if nivel == "critico":
        recos = {
            "inmediatas": [
                "Reparo: proveer (monte, cortina forestal o rollos apilados como cortaviento)",
                "Barro: evitar profundidad en corral",
                "Viento: reducir exposición directa",
            ],
            "operativas": [
                "Alimentación: adelantar comida principal a la tarde",
                "Cama: asegurar superficie seca",
                "Drenaje: mejorar evacuación del corral",
            ],
            "nutricionales": [
                "Concentrado: subir fibra activa 2 puntos "
                "(ej. 12% → 14%) por 3-4 días",
                "Vuelta: gradual al nivel base, sin cambios bruscos",
                "Estabilidad ruminal: NO subir grano ni almidón rápido",
            ],
        }
    elif nivel == "moderado":
        recos = {
            "inmediatas": [
                "Reparo: verificar disponibilidad",
                "Agua: revisar bebederos al amanecer (hielo)",
            ],
            "operativas": [
                "Alimentación: adelantar comida a la tarde",
                "Cama: asegurar superficie seca",
            ],
            "nutricionales": [
                "Concentrado: subir fibra activa 1-2 puntos "
                "(ej. 12% → 14%) por 3-4 días",
                "Vuelta: gradual al nivel base, sin cambios bruscos",
                "Estabilidad ruminal: NO subir grano de golpe",
            ],
        }
    else:
        return {"inmediatas": [], "operativas": [], "nutricionales": []}

    # ─── RECOS CONTEXTUALES POR CONDICIONES REALES ───
    clima = clima or {}
    ambiente = ambiente or {}
    temp = (clima.get("temperatura_actual_c")
              or clima.get("temperatura")
              or clima.get("min_nocturna")
              or clima.get("minima_nocturna_c"))
    lluvia_mm = float(clima.get("lluvia_mm", 0) or 0)
    barro = bool(ambiente.get("barro", False))
    pelaje_mojado = bool(ambiente.get("pelaje_mojado", False))
    dias_barro = int((clima.get("dias_barro_consecutivos") or 0))

    # AGUA en frío
    if temp is not None and temp <= 2:
        _agregar_unico(recos["inmediatas"],
                        "Agua: revisar congelamiento, romper hielo "
                        "y asegurar disponibilidad")
    if barro:
        _agregar_unico(recos["inmediatas"],
                        "Agua: controlar barro alrededor de bebederos")

    # COMEDERO con humedad / lluvia / pelaje mojado
    if lluvia_mm > 5 or pelaje_mojado:
        _agregar_unico(recos["operativas"],
                        "Pellet/mezcla: revisar humedad, evitar "
                        "permanencia prolongada de ración mojada")
        _agregar_unico(recos["operativas"],
                        "Comedero: controlar selección, evitar "
                        "sobrantes fermentados o deteriorados")

    # ACCESO al comedero con barro
    if barro:
        _agregar_unico(recos["inmediatas"],
                        "Acceso al comedero: revisar entrada, evitar "
                        "barro profundo en zona de comida")
        _agregar_unico(recos["operativas"],
                        "Drenaje: revisar y reducir pérdida energética "
                        "por humedad/barro")

    # CAMA priorizada si barro persistente
    if dias_barro >= 3:
        _agregar_unico(recos["operativas"],
                        "Cama: priorizar superficie seca para descanso "
                        "(riesgo de pododermatitis)")

    return recos


def _whatsapp_msg(tipo: str, nivel: str, thi_ajustado: float,
                    clima: Dict, ambiente: Dict,
                    recos: Dict[str, List[str]],
                    cliente_nombre: str = "",
                    lote_categoria: str = "",
                    localidad: str = "") -> str:
    """Compone WhatsApp con formato HMS extendido (~12 líneas).

    Formato:
        🔴 *ALERTA HMS — FRÍO CRÍTICO*
        _Lote: novillos | Catriló_

        *Condiciones:* 6°C + viento 22 km/h + lluvia 13mm + barro
        *Riesgo:* hipotermia + caída productiva si no se actúa hoy.

        ⚡ *INMEDIATAS:*
        • Reparo: monte, cortina forestal o rollos apilados
        • Agua: revisar congelamiento
        • Acceso al comedero: evitar barro profundo

        🔧 *OPERATIVAS:*
        • Pellet: revisar humedad
        • Cama: priorizar superficie seca

        🌾 *NUTRICIONAL:*
        • Subir fibra activa 2 pts (12% → 14%) por 3-4 días

        📩 _Detalle completo en el email._
    """
    if nivel == "ninguno":
        return ""

    icono = {
        "critico": "🔴", "moderado": "🟠", "preventiva": "🟡",
    }.get(nivel, "🟡")

    # ─── Cabecera ───
    tipo_label = "FRÍO" if tipo == "frio" else tipo.upper()
    nivel_label = nivel.upper().replace("CRITICO", "CRÍTICO")
    cabecera = f"{icono} *ALERTA HMS — {tipo_label} {nivel_label}*"

    # Subtítulo con lote y localidad si están disponibles
    subtitulo_partes = []
    if lote_categoria:
        subtitulo_partes.append(f"Lote: {lote_categoria}")
    if localidad:
        subtitulo_partes.append(localidad)
    elif cliente_nombre:
        subtitulo_partes.append(cliente_nombre)
    subtitulo = (
        f"_{' | '.join(subtitulo_partes)}_" if subtitulo_partes else ""
    )

    # ─── Condiciones ───
    if tipo == "calor":
        partes = [f"THI {thi_ajustado:.0f}"]
        t = clima.get("temperatura") or clima.get("temperatura_actual_c")
        if t is not None:
            partes.append(f"{t:.0f}°C")
        v = clima.get("viento_kmh")
        if v is not None and v < 10:
            partes.append(f"viento {v:.0f} km/h")
        elif v is not None and v >= 30:
            partes.append(f"viento {v:.0f} km/h")
        cond_line = " + ".join(partes)
        riesgo_line = (
            "muerte súbita y caída productiva si no se actúa."
            if nivel == "critico" else
            "caída de consumo y menor ganancia diaria."
            if nivel == "moderado" else
            "ola de calor en camino — preparar antes."
        )
    else:
        partes = []
        t = clima.get("min_nocturna")
        if t is None:
            t = clima.get("temperatura")
        v = clima.get("viento_kmh")
        l = clima.get("lluvia_mm", 0)
        if t is not None:
            partes.append(f"{t:.0f}°C")
        if v is not None and v > 10:
            partes.append(f"viento {v:.0f} km/h")
        if l and l > 5:
            partes.append(f"lluvia {l:.0f}mm")
        if (ambiente or {}).get("barro"):
            partes.append("barro")
        cond_line = " + ".join(partes) if partes else "Condición adversa"
        riesgo_line = (
            "hipotermia + caída productiva si no se actúa hoy."
            if nivel == "critico" else
            "mayor consumo energético y riesgo respiratorio."
            if nivel == "moderado" else
            "frente entrando — preparar reparos."
        )

    # ─── Bullets por sección ───
    inm = recos.get("inmediatas", [])
    op = recos.get("operativas", [])
    nut = recos.get("nutricionales", [])

    def _resumir(item: str, max_len: int = 75) -> str:
        """Acorta items largos para que entren cómodos en WhatsApp."""
        if len(item) <= max_len:
            return item
        return item[:max_len - 1].rstrip(",;:") + "…"

    lineas = [cabecera]
    if subtitulo:
        lineas.append(subtitulo)
    lineas.append("")
    lineas.append(f"*Condiciones:* {cond_line}")
    lineas.append(f"*Riesgo:* {riesgo_line}")

    if inm:
        lineas.append("")
        lineas.append("⚡ *INMEDIATAS:*")
        for it in inm[:3]:
            lineas.append(f"• {_resumir(it)}")
    if op:
        lineas.append("")
        lineas.append("🔧 *OPERATIVAS:*")
        for it in op[:2]:
            lineas.append(f"• {_resumir(it)}")
    if nut:
        lineas.append("")
        lineas.append("🌾 *NUTRICIONAL:*")
        for it in nut[:1]:
            lineas.append(f"• {_resumir(it, max_len=90)}")

    lineas.append("")
    lineas.append("📩 _Detalle completo en el email._")

    return "\n".join(lineas)


def _email_msg(tipo: str, nivel: str, thi_ajustado: float,
                 eval_calor: Dict, eval_frio: Dict,
                 recos: Dict[str, List[str]],
                 clima: Optional[Dict] = None,
                 ambiente: Optional[Dict] = None,
                 historial: Optional[Dict] = None,
                 categoria: str = "",
                 raza: str = "") -> str:
    """Compone email con golpe inicial + ACCIONES PRIMERO + agravantes/impacto/conclusión.
    Sigue el rediseño propuesto por el agente técnico externo (HMS CRM 360)."""
    if nivel == "ninguno":
        return "Sin alertas activas. Condiciones dentro de rangos normales."

    clima = clima or {}
    ambiente = ambiente or {}
    historial = historial or {}

    if tipo == "calor":
        agravantes = eval_calor.get("detalles_agravantes", [])
        impactos = _impacto_calor(nivel, clima=clima, ambiente=ambiente,
                                     historial=historial, categoria=categoria,
                                     raza=raza)
        cond_line = (
            f"Índice de calor (THI) ajustado: {thi_ajustado:.0f}"
        )
        riesgo_line = (
            "Riesgo de muerte súbita y caída productiva si no se actúa."
            if nivel == "critico"
            else "Riesgo de menor consumo y caída de ganancia diaria."
        )
    else:
        agravantes = eval_frio.get("detalles", [])
        impactos = _impacto_frio(nivel, clima=clima, ambiente=ambiente,
                                    historial=historial)
        # 1 línea con condiciones del frío — usa la mín nocturna si la hay
        partes = []
        t = clima.get("min_nocturna")
        if t is None:
            t = clima.get("temperatura")
        v = clima.get("viento_kmh")
        l = clima.get("lluvia_mm", 0)
        if t is not None:
            partes.append(f"Temperatura {t:.0f}°C")
        if v is not None and v > 10:
            partes.append("viento")
        if l and l > 5:
            partes.append("lluvia")
        if ambiente.get("barro"):
            partes.append("barro")
        cond_line = " + ".join(partes) if partes else "Condición adversa"
        riesgo_line = (
            "Riesgo de hipotermia y enfermedad respiratoria."
            if nivel == "critico"
            else "Riesgo de mayor consumo energético y pérdida de condición."
        )

    lineas = []

    # ───── 1) RESUMEN OPERATIVO (corto: cuándo + condición) ─────
    # La explicación de QUÉ pasa con el animal está en el bloque
    # LECTURA TÉCNICA del email (arriba), generado por LLM o biblioteca.
    # Acá solo dejamos lo accionable: cuándo y qué hacer.
    lineas.append("**📌 RESUMEN OPERATIVO**")
    lineas.append(cond_line)
    lineas.append("")

    # ───── 2) ACCIONES CLAVE ─────
    lineas.append("**⚡ ACCIONES CLAVE**")
    if recos.get("inmediatas"):
        lineas.append("**Inmediatas**")
        for it in recos["inmediatas"]:
            lineas.append(f"• {it}")
        lineas.append("")
    if recos.get("operativas"):
        lineas.append("**Operativas**")
        for it in recos["operativas"]:
            lineas.append(f"• {it}")
        lineas.append("")
    if recos.get("nutricionales"):
        lineas.append("**Nutricionales**")
        for it in recos["nutricionales"]:
            lineas.append(f"• {it}")
        lineas.append("")

    return "\n".join(lineas)


def evaluar_y_componer_mensajes(
    clima: Dict,
    ambiente: Optional[Dict] = None,
    historial: Optional[Dict] = None,
    categoria: str = "",
    raza: str = "",
) -> Dict:
    """Interfaz oficial — devuelve el JSON con tipo/nivel/thi_ajustado/whatsapp/email.

    Input (mismo que evaluar_estres_ambiental):
      clima: {thi, thi_proyectado, viento_kmh, min_nocturna, temperatura, lluvia_mm}
      ambiente: {sombra_m2_cab, barro}
      historial: {horas_thi_alto_ayer, dias_consecutivos_calor}
      categoria: ternero/novillito/vaca/toro/etc.

    Output (formato exacto del prompt):
      {
        "tipo": "calor" | "frio",
        "nivel": "preventiva" | "moderado" | "critico" | "ninguno",
        "thi_ajustado": numero,
        "whatsapp": "...",
        "email": "..."
      }
    """
    tipo, eval_calor, eval_frio = _ranking_dominante(
        clima, ambiente, historial, categoria, raza=raza,
    )
    rank_calor = _RANK_SEVERIDAD.get(
        eval_calor.get("tipo_alerta", "ninguno"),
        _RANK_SEVERIDAD.get(eval_calor["riesgo"], 0),
    )
    rank_frio = _RANK_SEVERIDAD.get(eval_frio["riesgo"], 0)

    if tipo == "calor" and rank_calor >= 2:
        nivel = eval_calor.get("tipo_alerta", "ninguno")
        nivel = "critico" if nivel in ("critica", "critico") else \
                 "moderado" if nivel in ("moderada", "moderado") else \
                 nivel
        recos = _recos_calor(nivel, categoria, clima=clima, ambiente=ambiente)
    elif tipo == "frio" and rank_frio >= 3:
        nivel = eval_frio["riesgo"]
        # Propagar contexto de barro persistente al rece de frío
        clima_ctx = dict(clima or {})
        if isinstance(historial, dict):
            clima_ctx["dias_barro_consecutivos"] = (
                historial.get("dias_barro_consecutivos", 0)
            )
        recos = _recos_frio(nivel, clima=clima_ctx, ambiente=ambiente)
    else:
        nivel = "ninguno"
        recos = {"inmediatas": [], "operativas": [], "nutricionales": []}

    thi_ajustado = eval_calor.get("thi_ajustado", 0)

    # "modo" reemplaza a "tipo" y agrega "transicion" cuando no hay alerta
    modo = tipo if nivel != "ninguno" else "transicion"

    return {
        "modo": modo,
        "tipo": tipo,   # mantengo por compatibilidad con código existente
        "nivel": nivel,
        "categoria": categoria,
        "thi_ajustado": thi_ajustado,
        "whatsapp": _whatsapp_msg(tipo, nivel, thi_ajustado,
                                     clima or {}, ambiente or {}, recos),
        "email": _email_msg(tipo, nivel, thi_ajustado,
                              eval_calor, eval_frio, recos,
                              clima=clima or {},
                              ambiente=ambiente or {},
                              historial=historial or {},
                              categoria=categoria,
                              raza=raza),
    }


def evaluar_desde_json(payload: Dict) -> Dict:
    """Evalúa desde un JSON completo v3 (con fecha, establecimiento, nutrición).

    Estructura esperada:
      {
        "fecha": "2026-01-15",
        "establecimiento": "Feedlot HMS",
        "categoria": "novillos",
        "clima": {
          "temperatura_actual_c": 35,
          "humedad_relativa": 68,
          "thi": 86,
          "thi_proyectado": [...],
          "viento_kmh": 8,
          "minima_nocturna_c": 24,
          "lluvia_mm": 0,
          "helada": false
        },
        "ambiente": {
          "sombra_m2_cab": 3,
          "barro": false,
          "pelaje_mojado": false,
          "estado_corrales": "normal"
        },
        "historial": {
          "horas_thi_alto_ayer": 5,
          "dias_consecutivos_calor": 2
        },
        "nutricion": {
          "porcentaje_concentrado_funcional": 12,
          "fibra_efectiva": true,
          "cambios_dieta_recientes": false
        }
      }

    Devuelve el JSON oficial con whatsapp/email personalizados según el % de
    concentrado funcional actual (si frío crítico).
    """
    payload = payload or {}
    resultado = evaluar_y_componer_mensajes(
        clima=payload.get("clima"),
        ambiente=payload.get("ambiente"),
        historial=payload.get("historial"),
        categoria=payload.get("categoria", ""),
    )

    # Personalizar la recomendación de concentrado funcional con el % real
    nutricion = payload.get("nutricion") or {}
    pct_actual = nutricion.get("porcentaje_concentrado_funcional")
    if pct_actual is not None and resultado["modo"] == "frio" \
            and resultado["nivel"] in ("critico", "moderado"):
        try:
            pct_actual_n = float(pct_actual)
            pct_objetivo = round(pct_actual_n + 2, 1)
            ajuste_personalizado = (
                f"Subir el concentrado funcional con fibra activa "
                f"de {pct_actual_n:.0f}% → {pct_objetivo:.0f}% por 3-4 días"
            )
            # Reemplazar el genérico en email y WhatsApp
            resultado["email"] = resultado["email"].replace(
                "Subir el concentrado funcional con fibra activa "
                "2 puntos (ej. 12% → 14%) por 3-4 días",
                ajuste_personalizado,
            )
            resultado["whatsapp"] = resultado["whatsapp"].replace(
                "Subir el concentrado funcional con fibra activa "
                "2 puntos (ej. 12% → 14%) por 3-4 días",
                ajuste_personalizado,
            )
        except (TypeError, ValueError):
            pass

    # Sumar campos del payload al resultado para trazabilidad
    if payload.get("fecha"):
        resultado["fecha"] = payload["fecha"]
    if payload.get("establecimiento"):
        resultado["establecimiento"] = payload["establecimiento"]
    if nutricion:
        resultado["nutricion_input"] = nutricion

    return resultado


# =====================================================================
# 4) SISTEMA DE ALERTAS PREDICTIVAS
# =====================================================================

SEVERIDAD_INFO = "info"
SEVERIDAD_WARNING = "warning"
SEVERIDAD_CRITICA = "critica"


def generar_alertas_predictivas(clima: Dict, categoria: str = "",
                                  ambiente: Optional[Dict] = None,
                                  raza: str = "") -> List[Dict]:
    """
    Analiza el pronóstico y devuelve alertas con acciones recomendadas.

    Usa el motor evaluar_estres_ambiental() para evaluar calor + frío y
    elegir el dominante. Itera los próximos 7 días, toma el peor, y
    devuelve la alerta en el formato compatible con email/WhatsApp.

    Args:
      clima: respuesta de obtener_clima() (dict con daily/current de Open-Meteo)
      categoria: categoría animal (afecta umbrales THI)
      ambiente: opcional {sombra_m2_cab, barro} si se cargan en la ficha del lote
    """
    alertas: List[Dict] = []
    if not clima or not clima.get("daily"):
        return alertas

    daily = clima["daily"]
    fechas = daily.get("time", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    viento_max = daily.get("wind_speed_10m_max", [])
    hum_max = daily.get("relative_humidity_2m_max", [])

    # Datos horarios (más precisos)
    hourly = clima.get("hourly", {}) or {}
    hr_times = hourly.get("time", [])
    hr_temp = hourly.get("temperature_2m", [])
    hr_hum = hourly.get("relative_humidity_2m", [])
    hr_app = hourly.get("apparent_temperature", [])  # sensación térmica
    hr_wind = hourly.get("wind_speed_10m", [])

    def _horas_thi_alto_dia(fecha_str: str, umbral: float) -> int:
        """Cuenta horas reales con THI ≥ umbral en una fecha específica."""
        if not hr_times or not hr_temp or not hr_hum:
            return 0
        cnt = 0
        for j, t_str in enumerate(hr_times):
            if not t_str.startswith(fecha_str):
                continue
            if j >= len(hr_temp) or j >= len(hr_hum):
                continue
            t = hr_temp[j]
            h = hr_hum[j]
            if t is None or h is None:
                continue
            thi_h = calcular_thi(t, h)
            if thi_h >= umbral:
                cnt += 1
        return cnt

    def _temp_min_nocturna(fecha_str: str) -> Optional[float]:
        """Mínima sensación térmica real entre 22:00 del día y 08:00 del día siguiente."""
        if not hr_times or not hr_app:
            return None
        try:
            f = datetime.strptime(fecha_str, "%Y-%m-%d").date()
        except ValueError:
            return None
        sig = (f + timedelta(days=1)).isoformat()
        candidatas = []
        for j, t_str in enumerate(hr_times):
            if j >= len(hr_app):
                continue
            v = hr_app[j]
            if v is None:
                continue
            # Hora de la noche: 22:00-23:00 del día f, o 00:00-08:00 del día sig
            if t_str.startswith(fecha_str) and t_str[11:13] in ("22", "23"):
                candidatas.append(v)
            elif t_str.startswith(sig) and t_str[11:13] in (
                "00", "01", "02", "03", "04", "05", "06", "07"
            ):
                candidatas.append(v)
        return min(candidatas) if candidatas else None

    hoy = datetime.now().date()
    idx_hoy = None
    for i, f in enumerate(fechas):
        try:
            if datetime.strptime(f, "%Y-%m-%d").date() == hoy:
                idx_hoy = i
                break
        except ValueError:
            continue
    if idx_hoy is None:
        return alertas

    # Próximos 7 días incluyendo hoy
    indices = list(range(idx_hoy, min(len(fechas), idx_hoy + 8)))
    if not indices:
        return alertas

    # Construir el array de THI proyectado (toda la ventana)
    thi_proyectado = []
    for i in indices:
        if i < len(t_max) and t_max[i] is not None:
            hr_dia = (hum_max[i] or 70) if i < len(hum_max) else 70
            thi_proyectado.append(round(calcular_thi(t_max[i], hr_dia * 0.7), 1))

    # Evaluar día por día. Quedarnos con el más severo.
    # Contadores de acumulación que se actualizan en cada iteración:
    #   - racha_calor: días consecutivos con THI ≥ preventiva
    #   - racha_frio:  días consecutivos con T° mínima < 5°C
    #   - racha_barro: días consecutivos con barro
    racha_calor = 0
    racha_frio = 0
    racha_barro = 0
    eval_peor = None
    fecha_peor = None
    rank_peor = -1

    for i in indices:
        if i >= len(t_max) or t_max[i] is None:
            continue
        hr_dia = (hum_max[i] or 70) if i < len(hum_max) else 70
        thi_dia = calcular_thi(t_max[i], hr_dia * 0.7)
        viento_dia = (viento_max[i]
                       if i < len(viento_max) and viento_max[i] is not None
                       else None)
        t_min_dia = (t_min[i]
                      if i < len(t_min) and t_min[i] is not None else None)
        precip_dia = (precip[i] if i < len(precip) and precip[i] else 0)

        # Historial — calculado del horario REAL del día anterior
        umb_cat = umbrales_thi_por_categoria(categoria)
        horas_thi_alto_ayer = 0
        if i > 0 and i - 1 < len(fechas):
            fecha_ayer = fechas[i - 1]
            horas_thi_alto_ayer = _horas_thi_alto_dia(
                fecha_ayer, umb_cat["preventiva"],
            )
            # Fallback al proxy diario si no hay datos horarios
            if horas_thi_alto_ayer == 0 and i - 1 < len(t_max) \
                    and t_max[i - 1] is not None:
                hr_ant = ((hum_max[i - 1] or 70)
                            if (i - 1) < len(hum_max) else 70)
                thi_ayer = calcular_thi(t_max[i - 1], hr_ant * 0.7)
                if thi_ayer >= umb_cat["preventiva"]:
                    horas_thi_alto_ayer = 5

        # Mín nocturna REAL del horario (sensación térmica con wind chill)
        if i < len(fechas):
            t_min_real = _temp_min_nocturna(fechas[i])
            if t_min_real is not None:
                t_min_dia = t_min_real

        if thi_dia >= umb_cat["preventiva"]:
            racha_calor += 1
        else:
            racha_calor = 0

        # Detectar barro: lluvia acumulada >30 mm en últimos 3 días
        idx_3d = list(range(max(0, i - 2), i + 1))
        precip_3d = sum((precip[k] or 0) for k in idx_3d if k < len(precip))
        barro = precip_3d > 30 or (ambiente or {}).get("barro", False)

        # Acumulación de frío: T° mínima < 5°C un día → suma a racha
        if t_min_dia is not None and t_min_dia < 5:
            racha_frio += 1
        else:
            racha_frio = 0

        # Acumulación de barro: barro presente en este día → suma a racha
        if barro:
            racha_barro += 1
        else:
            racha_barro = 0

        clima_in = {
            "thi": thi_dia,
            "thi_proyectado": thi_proyectado,
            "viento_kmh": viento_dia,
            "min_nocturna": t_min_dia,
            "temperatura": t_max[i],
            "lluvia_mm": precip_dia,
        }
        ambiente_in = {
            "sombra_m2_cab": (ambiente or {}).get("sombra_m2_cab"),
            "barro": barro,
        }
        historial_in = {
            "horas_thi_alto_ayer": horas_thi_alto_ayer,
            "dias_consecutivos_calor": racha_calor,
            "dias_consecutivos_frio": racha_frio,
            "dias_barro_consecutivos": racha_barro,
        }

        eval_dia = evaluar_estres_ambiental(
            clima_in, ambiente_in, historial_in,
            categoria=categoria, raza=raza,
        )
        nivel = eval_dia.get("nivel", "ninguno")
        rank = _RANK_SEVERIDAD.get(nivel, 0)

        if rank > rank_peor:
            rank_peor = rank
            eval_peor = eval_dia
            fecha_peor = fechas[i]
            # Guardar contexto del día para descripción
            eval_peor["_contexto"] = {
                "fecha": fechas[i],
                "thi": round(thi_dia, 1),
                "temp_max": t_max[i],
                "viento_kmh": viento_dia,
                "t_min": t_min_dia,
                "lluvia_mm": precip_dia,
                "barro": barro,
                "thi_proy_max": max(thi_proyectado) if thi_proyectado else 0,
            }

    # Si no hay alerta digna (nivel "ninguno"), no devolvemos nada
    if not eval_peor or eval_peor.get("nivel") == "ninguno":
        return alertas

    nivel = eval_peor["nivel"]
    tipo = eval_peor["tipo"]
    ctx = eval_peor.get("_contexto", {})

    # Mapear nivel → severidad para el formato externo
    if nivel in ("critico", "critica"):
        severidad = SEVERIDAD_CRITICA
        icono = "🔴" if tipo == "calor" else "❄️"
    elif nivel in ("moderado", "moderada"):
        severidad = SEVERIDAD_WARNING
        icono = "🟠" if tipo == "calor" else "🌨️"
    else:  # preventiva
        severidad = SEVERIDAD_INFO
        icono = "🟡"

    # Aplanar acciones para el formato externo (lista plana)
    acciones_planas: List[str] = []
    for it in eval_peor["acciones"].get("inmediata", []):
        acciones_planas.append(f"⚡ INMEDIATA: {it}")
    for it in eval_peor["acciones"].get("operativa", []):
        acciones_planas.append(f"🔧 OPERATIVA: {it}")
    for it in eval_peor["acciones"].get("nutricional", []):
        acciones_planas.append(f"🌾 NUTRICIONAL: {it}")

    # Título y descripción según tipo + nivel
    if tipo == "calor":
        if nivel == "critico":
            titulo = "ESTRÉS CALÓRICO CRÍTICO"
        elif nivel == "moderado":
            titulo = "Estrés calórico MODERADO"
        else:
            titulo = "Estrés calórico — alerta preventiva"
        desc = (
            f"THI hasta {ctx.get('thi', '?')} (T° max {ctx.get('temp_max', '?')}°C). "
            f"Viento {ctx.get('viento_kmh', '?')} km/h, "
            f"mínima nocturna {ctx.get('t_min', '?')}°C."
        )
        impacto = (
            "Caída del DMI 15-25%, riesgo de muerte súbita, baja fertilidad."
            if nivel == "critico" else
            "Caída del DMI 5-10%, ADG reducido, mayor consumo de agua."
        )
    else:
        if nivel == "critico":
            titulo = "ESTRÉS POR FRÍO CRÍTICO"
        elif nivel == "moderado":
            titulo = "Estrés por FRÍO MODERADO"
        else:
            titulo = "Frío — alerta"
        desc = (
            f"T° {ctx.get('temp_max', '?')}°C / mín {ctx.get('t_min', '?')}°C, "
            f"viento {ctx.get('viento_kmh', '?')} km/h, "
            f"lluvia {ctx.get('lluvia_mm', 0)} mm"
            f"{', barro presente' if ctx.get('barro') else ''}."
        )
        impacto = (
            "Aumento 25-40% requerimiento de mantenimiento. "
            "Riesgo alto de hipotermia y enfermedad respiratoria."
            if nivel == "critico" else
            "Aumento 10-15% requerimiento de mantenimiento. "
            "Pérdida lenta de CC si se prolonga."
        )

    dias_hasta = 0
    if fecha_peor:
        try:
            dias_hasta = (datetime.strptime(fecha_peor, "%Y-%m-%d").date() - hoy).days
        except ValueError:
            pass

    # Generar el email completo estructurado (mismo formato que el simulador)
    # llamando al evaluador top-level con los datos del peor día.
    ctx_peor = eval_peor.get("_contexto", {})
    clima_peor = {
        "thi": ctx_peor.get("thi", 0),
        "thi_proyectado": [ctx_peor.get("thi_proy_max", 0)],
        "viento_kmh": ctx_peor.get("viento_kmh"),
        "min_nocturna": ctx_peor.get("t_min"),
        "temperatura": ctx_peor.get("temp_max"),
        "lluvia_mm": ctx_peor.get("lluvia_mm", 0),
    }
    ambiente_peor = {
        "barro": ctx_peor.get("barro", False),
        "sombra_m2_cab": (ambiente or {}).get("sombra_m2_cab"),
    }
    historial_peor = {}  # ya considerado dentro de eval_peor
    try:
        msgs = evaluar_y_componer_mensajes(
            clima_peor, ambiente_peor, historial_peor, categoria=categoria,
        )
        email_completo = msgs.get("email", "") or ""
        # Si el evaluador puntual NO clasificó como alerta (umbral propio más
        # estricto que el del scoring semanal) y el motor semanal sí marcó
        # nivel >= preventiva, evitar emitir el placeholder "Sin alertas activas"
        # dentro de un bloque cuyo título sí dice MODERADO/CRITICO. En ese caso
        # construimos un email mínimo coherente con el título de la alerta.
        if (
            email_completo
            and "Sin alertas activas" in email_completo
            and nivel in ("preventiva", "moderado", "critico")
        ):
            partes_min = [
                "**📌 RESUMEN OPERATIVO**",
                desc,
                "",
            ]
            if acciones_planas:
                partes_min.append("**⚡ ACCIONES CLAVE**")
                for it in acciones_planas:
                    partes_min.append(f"• {it}")
                partes_min.append("")
            email_completo = "\n".join(partes_min)
    except Exception:
        email_completo = ""

    # Insertar línea "Cuándo" después del título RESUMEN OPERATIVO para
    # que el productor sepa si el evento es hoy, mañana o en varios días.
    if email_completo:
        if dias_hasta == 0:
            cuando_label = "hoy"
        elif dias_hasta == 1:
            cuando_label = f"mañana ({fecha_peor})"
        else:
            cuando_label = f"en {dias_hasta} días ({fecha_peor})"
        email_completo = email_completo.replace(
            "**📌 RESUMEN OPERATIVO**\n",
            f"**📌 RESUMEN OPERATIVO**\n📅 Cuándo: {cuando_label}\n",
            1,
        )

    alertas.append({
        "severidad": severidad,
        "icono": icono,
        "titulo": titulo,
        "cuando": (f"Hoy" if dias_hasta == 0
                    else f"En {dias_hasta} día(s) — {fecha_peor}"),
        "descripcion": "",          # vacío para no duplicar con el RESUMEN
        "impacto": impacto,
        "acciones": acciones_planas,
        # accion = email estructurado completo (RESUMEN/ACCIONES/AGRAVANTES/...)
        "accion": email_completo,
        # Campos extras del nuevo formato (por si se quieren consumir como JSON)
        "tipo": tipo,
        "nivel": nivel,
        "json_oficial": {
            "tipo": tipo,
            "nivel": nivel,
            "acciones": eval_peor["acciones"],
        },
    })

    # Alerta complementaria: viento extremo (fuera del score por umbral)
    if viento_max:
        viento_pico = max(
            (viento_max[i] for i in indices
             if i < len(viento_max) and viento_max[i] is not None),
            default=0,
        )
        if viento_pico >= 70:
            alertas.append({
                "severidad": SEVERIDAD_WARNING,
                "icono": "💨",
                "titulo": "Vientos fuertes previstos",
                "cuando": "Próximos días",
                "descripcion": f"Ráfagas previstas hasta {viento_pico:.0f} km/h.",
                "impacto": "Estrés mecánico, polvo en agua y comederos.",
                "acciones": [
                    "🔧 Asegurar techos, chapas y elementos sueltos.",
                    "🌳 Mover hacienda a potreros con monte de reparo.",
                    "💧 Revisar bebederos post-tormenta (sedimentos).",
                ],
            })

    return alertas



def alertas_a_texto(alertas: List[Dict]) -> str:
    """Convierte las alertas a texto plano para inyectar al system prompt."""
    if not alertas:
        return ""
    lineas = [
        "═══════════════════════════════════════════════════════════════",
        "🚨 ALERTAS CLIMÁTICAS PREDICTIVAS — usá esto proactivamente",
        "═══════════════════════════════════════════════════════════════",
        "Estas alertas se generaron analizando el pronóstico de los próximos",
        "7 días. NO esperes que el productor las identifique: incluí estas",
        "acciones en tu informe / recomendaciones para anticiparse.",
        "",
    ]
    for a in alertas:
        lineas.append(
            f"{a['icono']} [{a['severidad'].upper()}] {a['titulo']}"
        )
        lineas.append(f"   Cuándo: {a['cuando']}")
        lineas.append(f"   {a['descripcion']}")
        lineas.append(f"   Impacto: {a['impacto']}")
        lineas.append(f"   Acciones recomendadas:")
        for acc in a["acciones"]:
            lineas.append(f"      • {acc}")
        lineas.append("")
    return "\n".join(lineas)


# =====================================================================
# 4) RESUMEN PARA EL AGENTE
# =====================================================================

def resumen_clima_para_ia(localidad: str, categoria: str = "") -> str:
    """Genera un texto que se inyecta en el system prompt con el clima
    actual + análisis de los últimos 7 días + pronóstico próximos 7."""
    geo = geocodificar(localidad)
    if not geo:
        return ""

    clima = obtener_clima(geo["lat"], geo["lon"])
    if not clima:
        return ""

    actual = clima.get("current", {})
    daily = clima.get("daily", {})

    lineas = [
        "═══════════════════════════════════════════════════════════════",
        f"CLIMA — {geo['nombre']}, {geo['admin1']} ({geo['country']})",
        "═══════════════════════════════════════════════════════════════",
        f"📍 Lat {geo['lat']:.2f}, Lon {geo['lon']:.2f}",
        "",
    ]

    # Clima actual
    if actual:
        t = actual.get("temperature_2m")
        hr = actual.get("relative_humidity_2m")
        viento = actual.get("wind_speed_10m")
        precip = actual.get("precipitation", 0)
        if t is not None and hr is not None:
            thi = calcular_thi(t, hr)
            lineas.append(
                f"🌡️ AHORA: {t:.0f}°C  |  HR {hr:.0f}%  |  "
                f"Viento {viento:.0f} km/h  |  THI {thi:.0f} → {clasificar_thi(thi)}"
            )
        if precip and precip > 0:
            lineas.append(f"   Lluvia ahora: {precip:.1f} mm")
        lineas.append("")

    # Histórico últimos 7 días
    if daily and daily.get("time"):
        fechas = daily["time"]
        t_max = daily.get("temperature_2m_max", [])
        t_min = daily.get("temperature_2m_min", [])
        precip_sum = daily.get("precipitation_sum", [])
        viento_max = daily.get("wind_speed_10m_max", [])
        hum_max = daily.get("relative_humidity_2m_max", [])
        hum_min = daily.get("relative_humidity_2m_min", [])

        # Separar pasado y futuro
        hoy = datetime.now().date()
        idx_hoy = None
        for i, f in enumerate(fechas):
            try:
                if datetime.strptime(f, "%Y-%m-%d").date() == hoy:
                    idx_hoy = i
                    break
            except ValueError:
                continue

        if idx_hoy is not None:
            # Últimos 7 días (incluye hoy)
            ult7 = list(range(max(0, idx_hoy - 6), idx_hoy + 1))
            if ult7:
                lineas.append("📅 ÚLTIMOS 7 DÍAS:")
                t_max_ult = [t_max[i] for i in ult7 if i < len(t_max) and t_max[i] is not None]
                t_min_ult = [t_min[i] for i in ult7 if i < len(t_min) and t_min[i] is not None]
                precip_ult = sum((precip_sum[i] or 0) for i in ult7
                                  if i < len(precip_sum) and precip_sum[i] is not None)
                if t_max_ult:
                    lineas.append(
                        f"   T° máx promedio: {sum(t_max_ult)/len(t_max_ult):.0f}°C  "
                        f"(rango {min(t_max_ult):.0f}–{max(t_max_ult):.0f})"
                    )
                if t_min_ult:
                    lineas.append(
                        f"   T° mín promedio: {sum(t_min_ult)/len(t_min_ult):.0f}°C  "
                        f"(rango {min(t_min_ult):.0f}–{max(t_min_ult):.0f})"
                    )
                lineas.append(f"   Lluvia acumulada: {precip_ult:.1f} mm")

                # Detectar olas de calor (>32°C) o frío (<5°C)
                dias_calor = sum(1 for t in t_max_ult if t > 30)
                dias_frio = sum(1 for t in t_min_ult if t < 5)
                if dias_calor > 0:
                    lineas.append(
                        f"   ⚠️ {dias_calor} día(s) con T° máx >30°C — posible estrés calórico"
                    )
                if dias_frio > 0:
                    lineas.append(
                        f"   ❄️ {dias_frio} día(s) con T° mín <5°C — posible estrés por frío"
                    )
                if precip_ult > 50:
                    lineas.append(
                        f"   🌧️ Lluvia >50mm en 7 días — riesgo de barro en corral"
                    )
                lineas.append("")

            # Pronóstico próximos 7 días
            prox7 = list(range(idx_hoy + 1, min(len(fechas), idx_hoy + 8)))
            if prox7:
                lineas.append("🔮 PRÓXIMOS 7 DÍAS (pronóstico):")
                t_max_prox = [t_max[i] for i in prox7
                               if i < len(t_max) and t_max[i] is not None]
                precip_prox = sum((precip_sum[i] or 0) for i in prox7
                                   if i < len(precip_sum) and precip_sum[i] is not None)
                if t_max_prox:
                    lineas.append(
                        f"   T° máx esperada: {min(t_max_prox):.0f}–{max(t_max_prox):.0f}°C"
                    )
                lineas.append(f"   Lluvia prevista: {precip_prox:.1f} mm")

                dias_calor_prox = sum(1 for t in t_max_prox if t > 30)
                if dias_calor_prox > 0:
                    lineas.append(
                        f"   ⚠️ Se esperan {dias_calor_prox} día(s) con >30°C — "
                        "ajustar manejo (sombra, agua, frecuencia comidas)"
                    )

    # Anexar alertas predictivas estructuradas
    alertas = generar_alertas_predictivas(clima, categoria=categoria)
    if alertas:
        lineas.append("")
        lineas.append(alertas_a_texto(alertas))

    return "\n".join(lineas)
