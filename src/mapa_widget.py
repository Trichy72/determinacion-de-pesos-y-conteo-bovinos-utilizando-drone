"""
Widget de mapa interactivo para seleccionar coordenadas del campo.

Usa folium (OpenStreetMap) embebido en Streamlit. El usuario hace click
sobre el mapa y devolvemos lat/lon de ese punto. Sin API key, sin costo.

Características:
  - Centrado en Argentina por default
  - Si la localidad fue geocodificada, centra ahí
  - Soporta capa satelital (vista de campo)
  - Marca el punto seleccionado en tiempo real
  - Devuelve coordenadas listas para guardar
"""

from __future__ import annotations

from typing import Optional, Tuple


def parsear_link_ubicacion(texto: str) -> Optional[Tuple[float, float]]:
    """Extrae (lat, lon) de un link de ubicación pegado por el usuario.

    Soporta los formatos típicos que llegan por WhatsApp o que se
    copian desde Google Maps en el navegador/móvil:

      1. Google Maps directo:
         https://maps.google.com/?q=-36.42,-63.49
         https://www.google.com/maps?q=-36.42,-63.49
      2. Google Maps con /@lat,lon,zoom:
         https://www.google.com/maps/place/.../@-36.42,-63.49,17z/...
      3. Google Maps con !3dlat!4dlon (formato interno del data):
         https://www.google.com/maps/place/.../data=!3m1!4b1!4m6!.../!3d-36.42!4d-63.49
      4. Link corto (requiere seguir redirección HTTP):
         https://maps.app.goo.gl/abc123
         https://goo.gl/maps/abc123
      5. Coordenadas pegadas directo (formato libre):
         "-36.4234, -63.4567"
         "-36.4234,-63.4567"
         "-36.4234 -63.4567"
      6. Formato geo: (Android/Apple Maps):
         geo:-36.42,-63.49

    Args:
        texto: el string que pegó el usuario.

    Returns:
        Tupla (lat, lon) si pudo extraer, None si el formato no es
        reconocible. La lat debe estar entre -90 y 90, la lon entre
        -180 y 180.
    """
    import re
    if not texto or not isinstance(texto, str):
        return None
    texto = texto.strip()
    if not texto:
        return None

    def _validar(lat: float, lon: float) -> Optional[Tuple[float, float]]:
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
        return None

    # ── Caso 1+2+3: Google Maps formato @lat,lon o ?q=lat,lon o !3d!4d
    # Patrón @ (preferido si aparece, es la ubicación visible del mapa)
    m = re.search(
        r"@(-?\d+\.\d+),\s*(-?\d+\.\d+)", texto,
    )
    if m:
        r = _validar(float(m.group(1)), float(m.group(2)))
        if r:
            return r

    # Patrón ?q=lat,lon  (Google Maps "Compartir")
    m = re.search(
        r"[?&]q=(-?\d+\.\d+),\s*(-?\d+\.\d+)", texto,
    )
    if m:
        r = _validar(float(m.group(1)), float(m.group(2)))
        if r:
            return r

    # Patrón !3dlat!4dlon (data interno de Google)
    m = re.search(
        r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", texto,
    )
    if m:
        r = _validar(float(m.group(1)), float(m.group(2)))
        if r:
            return r

    # ── Caso 4: link corto → seguir redirección
    if "maps.app.goo.gl" in texto or "goo.gl/maps" in texto:
        try:
            import urllib.request
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(
                texto,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36"
                    ),
                },
            )
            with urllib.request.urlopen(
                req, timeout=8, context=ctx,
            ) as resp:
                url_final = resp.geturl()
            # Re-parsear con el URL ya expandido
            if url_final and url_final != texto:
                return parsear_link_ubicacion(url_final)
        except Exception:
            pass

    # ── Caso 6: geo:lat,lon
    m = re.match(
        r"geo:\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)", texto, re.IGNORECASE,
    )
    if m:
        r = _validar(float(m.group(1)), float(m.group(2)))
        if r:
            return r

    # ── Caso 5: coordenadas pegadas directo "lat, lon" o "lat lon"
    m = re.match(
        r"^\s*(-?\d{1,3}\.\d+)\s*[,\s]\s*(-?\d{1,3}\.\d+)\s*$",
        texto,
    )
    if m:
        r = _validar(float(m.group(1)), float(m.group(2)))
        if r:
            return r

    # Última oportunidad: buscar cualquier par lat,lon dentro del texto
    m = re.search(
        r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)", texto,
    )
    if m:
        r = _validar(float(m.group(1)), float(m.group(2)))
        if r:
            return r

    return None


def render_mapa_seleccion(
    lat_actual: Optional[float] = None,
    lon_actual: Optional[float] = None,
    localidad_busqueda: str = "",
    zoom: int = 6,
    altura: int = 450,
    key: str = "mapa",
) -> Tuple[Optional[float], Optional[float]]:
    """Renderiza un mapa interactivo de OpenStreetMap.

    Args:
      lat_actual, lon_actual: si ya hay coordenadas, las muestra como marker
      localidad_busqueda: si se geocodifica esta localidad, centra ahí
      zoom: 4 (Argentina entera) - 15 (cuadra)
      altura: alto del mapa en pixels
      key: clave Streamlit única para este componente

    Returns:
      (lat, lon) — None si no hay click reciente
    """
    try:
        import folium
        from folium.plugins import MousePosition
        from streamlit_folium import st_folium
        import streamlit as st
    except ImportError as e:
        import streamlit as st
        st.error(
            f"Faltan paquetes: {e}. "
            "Instalá con: `pip install folium streamlit-folium`"
        )
        return None, None

    # Determinar centro del mapa
    if lat_actual and lon_actual and lat_actual != 0:
        centro = [lat_actual, lon_actual]
        zoom_inicial = 13
    elif localidad_busqueda:
        # Intentar geocodificar para centrar
        try:
            from .clima import geocodificar
            geo = geocodificar(localidad_busqueda)
            if geo:
                centro = [geo["lat"], geo["lon"]]
                zoom_inicial = 11
            else:
                centro = [-36.0, -64.0]   # Argentina central
                zoom_inicial = 5
        except Exception:
            centro = [-36.0, -64.0]
            zoom_inicial = 5
    else:
        centro = [-36.0, -64.0]
        zoom_inicial = 5

    # Crear el mapa con dos capas: estándar + satélite
    m = folium.Map(
        location=centro,
        zoom_start=zoom_inicial,
        control_scale=True,
    )

    # Capa de mapa estándar (calles)
    folium.TileLayer(
        "OpenStreetMap",
        name="Mapa estándar",
    ).add_to(m)

    # Capa satelital (Esri World Imagery — gratis sin API key)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Vista satelital",
        overlay=False,
        control=True,
    ).add_to(m)

    # Capa híbrida (satélite + nombres)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Etiquetas (sobre satélite)",
        overlay=True,
        control=True,
    ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)

    # Mostrar coordenadas del cursor en tiempo real
    MousePosition(
        position="bottomleft",
        separator=" , ",
        prefix="📍 ",
        num_digits=4,
    ).add_to(m)

    # Marker del punto actual si existe
    if lat_actual and lon_actual and lat_actual != 0:
        folium.Marker(
            [lat_actual, lon_actual],
            popup=f"Actual: {lat_actual:.4f}, {lon_actual:.4f}",
            icon=folium.Icon(color="green", icon="leaf"),
        ).add_to(m)

    # Renderizar y capturar interacciones
    result = st_folium(
        m,
        height=altura,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key=key,
    )

    # Si el usuario hizo click, devolver esas coordenadas
    if result and result.get("last_clicked"):
        return (
            result["last_clicked"]["lat"],
            result["last_clicked"]["lng"],
        )

    return None, None
