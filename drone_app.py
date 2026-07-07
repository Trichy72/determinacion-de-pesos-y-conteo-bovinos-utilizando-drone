"""
Drone Cattle Weight — App standalone para comercialización.

Esta es la versión "modo producto" de la app: solo el módulo de drone
(conteo y peso por imagen/video). Sin asesor nutricional, sin formulación
de dietas, sin Claude API.

Pensada para:
  - Vender como producto independiente a productores y feedlots
  - Versión white-label (cualquiera la puede customizar con su marca)
  - Servicios de consultoría puntual (cobrar por análisis)

Cómo correrla:
    streamlit run drone_app.py

Configuración: editá branding.py para cambiar marca, colores, contacto.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.calibration import calibrate, calibrate_from_altitude
from src.detector import CattleDetector
from src.processor import (
    export_results_csv, process_image, process_video,
)
from src.weight_estimator import WeightModel

# =====================================================================
# CONFIGURACIÓN DE MARCA (editable para white-label)
# =====================================================================
BRAND = {
    "nombre": "Drone Cattle Weight",
    "tagline": "Conteo y estimación de peso por drone",
    "color_primario": "#1B3E27",
    "color_secundario": "#8BC53F",
    "logo_path": "assets/logo.png",
    "contacto": "",   # ej: "info@tucabaña.com.ar"
    "version": "1.0",
}

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")

# =====================================================================
# CONFIG STREAMLIT
# =====================================================================
st.set_page_config(
    page_title=BRAND["nombre"],
    page_icon="🐄",
    layout="wide",
)

# CSS de marca
st.markdown(f"""
<style>
    h1, h2, h3 {{ color: {BRAND['color_primario']}; }}
    .stButton > button[kind="primary"] {{
        background-color: {BRAND['color_primario']};
        color: white;
        border: 2px solid {BRAND['color_secundario']};
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {BRAND['color_secundario']};
        color: {BRAND['color_primario']};
    }}
    .stTabs [aria-selected="true"] {{
        color: {BRAND['color_primario']} !important;
        border-bottom: 3px solid {BRAND['color_secundario']} !important;
    }}
    [data-testid="stMetricValue"] {{ color: {BRAND['color_primario']}; }}
</style>
""", unsafe_allow_html=True)

# Header
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if Path(BRAND["logo_path"]).exists():
        st.image(BRAND["logo_path"], width=120)
with col_title:
    st.markdown(
        f"<h1 style='color:{BRAND['color_primario']};margin-bottom:0;'>"
        f"{BRAND['nombre']}</h1>"
        f"<p style='color:{BRAND['color_secundario']};font-size:1.2em;"
        f"margin-top:0;font-weight:600;'>{BRAND['tagline']}</p>",
        unsafe_allow_html=True,
    )
st.caption(
    f"Captura recomendada: drone 4K @ 30 fps, altura ~10 m, 90° (cenital), "
    f"con cuadrado de referencia de 1,02 m visible."
)


# =====================================================================
# HELPERS
# =====================================================================
def _hash_bytes(data: bytes, *extra) -> str:
    h = hashlib.sha1(data)
    for e in extra:
        h.update(b"|")
        h.update(str(e).encode())
    return h.hexdigest()


@st.cache_resource
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Cargando modelo YOLO…")
def load_detector(model_path: str, cow_class_id: int, conf: float, iou: float,
                   imgsz: int, modo_tropa_densa: bool = False):
    return CattleDetector(
        model_path=model_path,
        cow_class_id=cow_class_id,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        modo_tropa_densa=modo_tropa_densa,
    )


cfg = load_config()


# =====================================================================
# SIDEBAR — CONFIGURACIÓN DE CAPTURA Y DETECCIÓN
# =====================================================================
with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("Captura")
    altura = st.number_input(
        "Altura de vuelo (m)", min_value=2.0, max_value=50.0,
        value=float(cfg["captura"]["altura_vuelo_m"]), step=0.5,
    )
    cfg["captura"]["altura_vuelo_m"] = altura

    st.subheader("Referencia en piso")
    metodo = st.selectbox(
        "Método de detección", ["aruco", "color_square"],
        index=0 if cfg["referencia"]["metodo"] == "aruco" else 1,
    )
    cfg["referencia"]["metodo"] = metodo
    lado = st.number_input(
        "Lado del cuadrado (m)", min_value=0.3, max_value=3.0,
        value=float(cfg["referencia"]["lado_m"]), step=0.01, format="%.2f",
    )
    cfg["referencia"]["lado_m"] = lado

    st.subheader("Detección")
    modo_tropa_densa = st.toggle(
        "🐄🐄🐄 Modo tropa densa", value=False,
        help="Activar si los animales pasan apretados (mixer, manga)."
    )
    if modo_tropa_densa:
        modelo_path = st.selectbox(
            "Modelo YOLO",
            ["yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
            index=1,
        )
        conf = 0.05
        iou = 0.35
        imgsz = 1920
    else:
        modelo_path = st.selectbox(
            "Modelo YOLO",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt",
             "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt"],
            index=2,
        )
        conf = st.slider("Confianza mínima", 0.05, 0.9, 0.10, 0.05)
        iou = st.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05)
        imgsz = st.select_slider(
            "Tamaño inferencia", [640, 960, 1280, 1600, 1920], 1280,
        )

    st.subheader("Estimación de peso")
    raza = st.selectbox(
        "Raza predominante",
        ["angus", "hereford", "brangus", "braford", "cruza", "desconocido"],
        index=0,
    )
    categoria = st.selectbox(
        "Categoría / edad",
        ["ternero", "vaquillona", "novillo", "vaca_adulta", "toro"],
        index=1,
    )
    ajuste_fino = st.slider(
        "Ajuste fino de peso", 0.70, 1.30, 1.00, 0.01,
        help="Multiplicador para calibrar con tu balanza real.",
    )


# =====================================================================
# CARGAR DETECTOR Y MODELO DE PESO
# =====================================================================
detector = load_detector(
    modelo_path, cfg["deteccion"]["clase_cow_id"], conf, iou, imgsz,
    modo_tropa_densa=modo_tropa_densa,
)
weight_model = WeightModel.from_config(cfg)


# =====================================================================
# TABS — solo Imagen y Video
# =====================================================================
tab_img, tab_vid, tab_help = st.tabs(["📷 Imagen", "🎞️ Video", "ℹ️ Ayuda"])


# ----------------------------- IMAGEN ---------------------------------
with tab_img:
    file = st.file_uploader(
        "Subí una imagen del lote (JPG/PNG)",
        type=["jpg", "jpeg", "png"], key="img_upload",
    )
    if file:
        bytes_data = file.getvalue()
        cache_key = _hash_bytes(
            bytes_data, modelo_path, conf, iou, imgsz, raza, categoria,
            ajuste_fino, cfg["referencia"]["metodo"], cfg["referencia"]["lado_m"],
        )

        if st.session_state.get("img_cache_key") != cache_key:
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                st.error("No se pudo leer la imagen.")
                st.stop()

            with st.spinner("Detectando animales y estimando pesos…"):
                annotated, result = process_image(
                    img, detector, weight_model, cfg, raza, categoria,
                    ajuste_fino=ajuste_fino,
                )

            _, png_buf = cv2.imencode(".png", annotated)
            st.session_state["img_cache_key"] = cache_key
            st.session_state["img_annotated_rgb"] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.session_state["img_png_bytes"] = png_buf.tobytes()
            st.session_state["img_orig_name"] = file.name
            st.session_state["img_n"] = result.n_animales
            st.session_state["img_prom"] = result.peso_promedio_kg
            st.session_state["img_total"] = result.peso_total_kg
            st.session_state["img_desv"] = result.desvio_kg
            st.session_state["img_animales"] = [
                {"Animal": a.track_id, "Peso (kg)": round(a.peso_kg, 1)}
                for a in result.animales
            ]

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(
                st.session_state["img_annotated_rgb"],
                caption="Resultado", use_column_width=True,
            )
        with col2:
            st.metric("Animales detectados", st.session_state["img_n"])
            st.metric("Peso promedio", f"{st.session_state['img_prom']:.1f} kg")
            st.metric("Peso total", f"{st.session_state['img_total']:.0f} kg")
            st.metric("Desvío estándar", f"{st.session_state['img_desv']:.1f} kg")
            df = pd.DataFrame(st.session_state["img_animales"])
            st.dataframe(df, hide_index=True, width="stretch")

            base = Path(st.session_state["img_orig_name"]).stem
            st.download_button(
                "📥 PNG anotado",
                data=st.session_state["img_png_bytes"],
                file_name=f"{base}_anotado.png",
                mime="image/png",
            )
            st.download_button(
                "📥 CSV pesos",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{base}_pesos.csv",
                mime="text/csv",
            )


# ----------------------------- VIDEO ----------------------------------
with tab_vid:
    file = st.file_uploader(
        "Subí un video del lote (MP4/MOV)",
        type=["mp4", "mov", "avi"], key="vid_upload",
    )
    if file:
        bytes_data = file.getvalue()
        cache_key = _hash_bytes(
            bytes_data, modelo_path, conf, iou, imgsz, raza, categoria,
            ajuste_fino, cfg["referencia"]["metodo"], cfg["referencia"]["lado_m"],
        )

        if st.session_state.get("vid_cache_key") != cache_key:
            with tempfile.NamedTemporaryFile(
                suffix=Path(file.name).suffix, delete=False,
            ) as tmp:
                tmp.write(bytes_data)
                in_path = Path(tmp.name)
            out_path = in_path.with_name(in_path.stem + "_anotado.mp4")
            csv_path = in_path.with_name(in_path.stem + "_pesos.csv")

            progress = st.progress(0.0, text="Procesando video…")

            def cb(p, _pb=progress):
                _pb.progress(min(p, 1.0), text=f"Procesando… {p*100:.0f}%")

            with st.spinner("Procesando video — puede tardar varios minutos"):
                result = process_video(
                    in_path, out_path, detector, weight_model, cfg,
                    raza=raza, categoria=categoria,
                    ajuste_fino=ajuste_fino, progress_cb=cb,
                )
                export_results_csv(result, csv_path)
            progress.progress(1.0, text="¡Listo!")

            st.session_state["vid_cache_key"] = cache_key
            st.session_state["vid_out_path"] = str(out_path)
            st.session_state["vid_video_bytes"] = out_path.read_bytes()
            st.session_state["vid_csv_bytes"] = csv_path.read_bytes()
            st.session_state["vid_orig_name"] = file.name
            st.session_state["vid_n"] = result.n_animales
            st.session_state["vid_prom"] = result.peso_promedio_kg
            st.session_state["vid_total"] = result.peso_total_kg
            st.session_state["vid_desv"] = result.desvio_kg
            st.session_state["vid_calidad"] = result.calidad_captura_pct
            st.session_state["vid_animales"] = [
                {"Animal": a.track_id, "Peso (kg)": round(a.peso_kg, 1)}
                for a in result.animales
            ]

        col1, col2 = st.columns([3, 2])
        with col1:
            st.video(st.session_state["vid_out_path"])
        with col2:
            st.metric("Animales únicos", st.session_state["vid_n"])
            st.metric("Peso promedio", f"{st.session_state['vid_prom']:.1f} kg")
            st.metric("Peso total", f"{st.session_state['vid_total']:.0f} kg")
            st.metric("Desvío estándar", f"{st.session_state['vid_desv']:.1f} kg")
            calidad = st.session_state.get("vid_calidad", 100)
            if calidad >= 90:
                st.success(f"✅ Calidad: {calidad:.0f}%")
            elif calidad >= 70:
                st.warning(f"⚠️ Calidad: {calidad:.0f}%")
            else:
                st.error(f"🔴 Calidad: {calidad:.0f}% — refilmar recomendado")

            df = pd.DataFrame(st.session_state["vid_animales"])
            st.dataframe(df, hide_index=True, width="stretch")

            base = Path(st.session_state["vid_orig_name"]).stem
            st.download_button(
                "📥 Video anotado",
                data=st.session_state["vid_video_bytes"],
                file_name=f"{base}_anotado.mp4",
                mime="video/mp4",
            )
            st.download_button(
                "📥 CSV pesos",
                data=st.session_state["vid_csv_bytes"],
                file_name=f"{base}_pesos.csv",
                mime="text/csv",
            )


# ----------------------------- AYUDA ----------------------------------
with tab_help:
    st.markdown(f"""
### Cómo usar {BRAND['nombre']}

1. **Captura del drone**: vuelo cenital (90°) a ~10 m de altura, 4K @ 30 fps.
   Incluí en el encuadre una **referencia conocida en el piso**:
   - Marcador ArUco de 1,02 × 1,02 m, o
   - Cuadrado de cinta/lona de color sólido.

2. **Subí imagen o video** en la pestaña correspondiente.

3. **Configurá** en la barra lateral:
   - Raza predominante y categoría
   - **Ajuste fino**: calibrá con tu balanza real una sola vez
     y dejalo fijo

4. **Resultado**: conteo, peso individual, peso promedio del lote y
   peso total. Descargás imagen/video anotado + CSV con los datos.

### Precisión esperada

- Conteo: 85-95% (con condiciones óptimas de captura)
- Peso: ±5% al promedio del lote (después de calibrar el ajuste fino)

### Limitaciones honestas

- En tropas muy densas (animales tocándose), el conteo puede bajar a 50-70%
- YOLO genérico ve mejor lotes con ≤30 animales por frame
- Si tu drone se mueve mucho por viento, calidad de captura baja

---

**{BRAND['nombre']}** v{BRAND['version']}
{BRAND['contacto']}
    """)
