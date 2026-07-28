# -*- coding: utf-8 -*-
"""
Drone HMS v2 — app LOCAL de procesamiento de recorridas con drone.

Corre en la Mac (necesita el venv del repo con ultralytics):

    source .venv/bin/activate
    streamlit run drone_app.py

Qué hace:
  - "Recorrida completa": procesa una carpeta de fotos DJI con el pipeline
    CALIBRADO de src/recorrida.py (EXIF/XMP -> GSD, yolov8l-seg con peso
    por área de máscara SIN sombra, filtro de borde, peso solo con fotos
    <= 20 m, media recortada 10%, agrupación automática en corrales).
  - "Foto suelta": subís una o varias fotos y ves el resultado al instante.
  - GUARDAR EN LA FICHA: por grupo/corral elegís cliente y lote y el peso
    promedio estimado se inserta como pesada (metodo='drone') en la MISMA
    base que usa la app online (Supabase Postgres vía src/database.py),
    así aparece de inmediato en la ficha del lote con sus gráficos.

Conexión a la base: igual que app.py — DATABASE_URL de la variable de
entorno o del .env del repo (src/db_backend.py). Si no hay, la app pide
la URL una vez y la guarda en .hms_local.env (ignorado por git).
"""

from __future__ import annotations

import io
import os
import statistics
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_ENV = REPO_ROOT / ".hms_local.env"


def _cargar_env_local() -> None:
    """Carga .hms_local.env (creado por la pantalla de configuración) en
    el entorno, sin pisar variables ya seteadas. src/db_backend.py ya lee
    el .env del repo por su cuenta; esto es el fallback para Macs donde
    ese archivo no existe."""
    if not LOCAL_ENV.exists():
        return
    for linea in LOCAL_ENV.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        k, v = linea.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_cargar_env_local()

from src import database as db          # noqa: E402
from src import recorrida as rec        # noqa: E402
from src.db_backend import backend_activo  # noqa: E402

# =====================================================================
# Config Streamlit
# =====================================================================
st.set_page_config(page_title="Drone HMS — Recorridas",
                   page_icon="🚁", layout="wide")

DIR_FOTOS_DEFAULT = REPO_ROOT / "videos_drone" / "tarjeta_drone"
IMGSZ = rec.IMGSZ_DEFAULT  # 3840, casi resolución nativa (calibrado)


def _asegurar_gitignore() -> None:
    """Garantiza que .hms_local.env no se suba nunca al repo."""
    gi = REPO_ROOT / ".gitignore"
    try:
        contenido = gi.read_text(encoding="utf-8") if gi.exists() else ""
        if ".hms_local.env" not in contenido:
            gi.write_text(contenido.rstrip("\n") +
                          "\n\n# Config local de drone_app (URL de la base)\n"
                          ".hms_local.env\n", encoding="utf-8")
    except OSError:
        pass


def _guardar_database_url(url: str) -> None:
    LOCAL_ENV.write_text(f"DATABASE_URL={url.strip()}\n", encoding="utf-8")
    _asegurar_gitignore()
    os.environ["DATABASE_URL"] = url.strip()


# =====================================================================
# Pantalla de configuración inicial (solo si NO hay Postgres disponible)
# =====================================================================
backend = backend_activo()

if backend != "postgres" and not st.session_state.get("aceptar_sqlite"):
    st.title("Configuración inicial — conexión a la base")
    st.error(
        "No encontré la conexión a la base de la nube (DATABASE_URL). "
        "Sin ella, las pesadas se guardarían en un archivo SQLite local "
        "de esta Mac y **NO aparecerían en la ficha online del cliente**."
    )
    st.markdown(
        "Pegá la URL de Postgres de Supabase (la misma que usa la app "
        "online, empieza con `postgresql://`). Se guarda una sola vez en "
        "`.hms_local.env` (fuera de git) y no se vuelve a pedir."
    )
    url_nueva = st.text_input("DATABASE_URL", type="password",
                              placeholder="postgresql://usuario:clave@host:6543/postgres")
    c1, c2 = st.columns(2)
    if c1.button("Guardar y conectar", type="primary"):
        if url_nueva.strip().startswith(("postgres://", "postgresql://")):
            _guardar_database_url(url_nueva)
            st.success("Guardada. Conectando…")
            time.sleep(0.5)
            st.rerun()
        else:
            st.warning("La URL debe empezar con postgresql:// — revisala.")
    if c2.button("Continuar con SQLite local (no recomendado)"):
        st.session_state["aceptar_sqlite"] = True
        st.rerun()
    st.stop()


# =====================================================================
# Sidebar: conexión + parámetros del pipeline
# =====================================================================
st.sidebar.title("Drone HMS")

if backend == "postgres":
    st.sidebar.success("Base: **Postgres nube (Supabase)** — la misma que "
                       "la app online. Lo que guardes aparece en la ficha.")
else:
    st.sidebar.error("Base: **SQLite local** — ⚠️ lo que guardes acá NO se "
                     "va a ver en la app online. Configurá DATABASE_URL "
                     "(borrá .hms_local.env y recargá para volver a la "
                     "pantalla de configuración).")

modelos_disponibles = sorted(p.name for p in REPO_ROOT.glob("*.pt"))
if rec.MODELO_DEFAULT not in modelos_disponibles:
    modelos_disponibles.insert(0, rec.MODELO_DEFAULT)
# Modelo fine-tuneado propio (notebooks/finetune_hms.ipynb): si está en la
# raíz, va PRIMERO en el selector — es el mejor para detección/conteo.
MODELO_HMS = "hms_bovinos.pt"
if MODELO_HMS in modelos_disponibles:
    modelos_disponibles.remove(MODELO_HMS)
    modelos_disponibles.insert(0, MODELO_HMS)
modelo = st.sidebar.selectbox(
    "Modelo YOLO", modelos_disponibles,
    index=modelos_disponibles.index(rec.MODELO_DEFAULT),
    help="hms_bovinos (si está): fine-tuneado con tus fotos de drone, la "
         "mejor detección/conteo. Con un modelo -seg el peso se calcula "
         "por máscara (sin sombra). Sin -seg cae al método bbox x 0.69 "
         "(menos preciso).",
)
if modelo == MODELO_HMS:
    st.sidebar.info(
        "**hms_bovinos**: mejor detección/conteo (fine-tuneado con tus "
        "fotos). Es un modelo *detect* (sin máscaras): el peso cae al "
        "método bbox × 0.69. **Para peso seguí usando yolov8l-seg.**"
    )
conf = st.sidebar.slider("Confianza mínima YOLO", 0.05, 0.50,
                         rec.CONF_DEFAULT, 0.01)
categoria = st.sidebar.selectbox("Categoría (factor de peso)",
                                 rec.CATEGORIAS,
                                 index=rec.CATEGORIAS.index("novillo"))

try:
    _wm = rec.cargar_weight_model()
    st.sidebar.caption(
        f"Coeficiente calibrado (config.yaml, solo lectura): "
        f"a = {_wm.coef_a:g}, b = {_wm.coef_b:g} — calibrado con balanza "
        f"27/7/26. Peso confiable solo con fotos ≤ 20 m."
    )
except Exception as e:
    st.sidebar.caption(f"No pude leer config.yaml: {e}")


# =====================================================================
# Helpers de procesamiento y UI
# =====================================================================

def procesar_lote_fotos(fotos, dir_anotadas: Path):
    """Corre el pipeline calibrado sobre una lista de FotoMeta, con barra
    de progreso. Los imports pesados (ultralytics/cv2) ocurren acá."""
    barra = st.progress(0.0, text="Cargando modelo YOLO…")
    yolo = rec.crear_yolo(modelo)
    wm = rec.cargar_weight_model()
    es_seg = rec.es_modelo_seg(modelo)
    for i, meta in enumerate(fotos, 1):
        barra.progress(i / len(fotos),
                       text=f"[{i}/{len(fotos)}] {meta.path.name}…")
        if meta.error:
            continue
        meta.metodo = "mask" if es_seg else "bbox"
        try:
            rec.procesar_foto(meta, yolo, wm, categoria,
                              solo_conteo=False, anotar=True,
                              out_anotadas=dir_anotadas,
                              conf=conf, imgsz=IMGSZ)
        except Exception as e:
            meta.error = f"Error en detección: {e}"
    barra.progress(1.0, text="Listo.")
    return fotos


def csv_descarga(fotos) -> bytes:
    buf = io.StringIO()
    import csv as _csv
    _csv.writer(buf).writerows(rec.filas_csv(fotos, modelo))
    return buf.getvalue().encode("utf-8")


def fecha_recorrida(fotos) -> date:
    horas = [f.hora for f in fotos if f.hora]
    return min(horas).date() if horas else date.today()


def tabla_grupos(grupos) -> pd.DataFrame:
    filas = []
    for i, g in enumerate(grupos, 1):
        r = rec.resumen_grupo(g)
        peso = (f"{r['peso_est']:.0f} kg" if r["peso_est"] is not None
                else ("sin peso confiable (fotos >20 m)"
                      if r["alturas_altas"] else "—"))
        filas.append({
            "Grupo": f"Grupo {i}",
            "Fotos": f"{g[0].nombre} a {g[-1].nombre} ({len(g)})",
            "Hora": (f"{r['hora_ini']:%H:%M}–{r['hora_fin']:%H:%M}"
                     if r["hora_ini"] else "s/hora"),
            "Conteo (máx. por foto)": r["conteo_max"],
            "Animales pesados": r["n_pesados"],
            "Peso promedio (media recortada)": peso,
        })
    return pd.DataFrame(filas)


def ui_guardar_en_ficha(key: str, resumen: dict, fecha_default: date):
    """Selector de cliente/lote + botón que inserta la pesada en la MISMA
    base que la app online (tabla `pesadas`, metodo='drone')."""
    st.markdown("##### Guardar en la ficha del cliente")
    if resumen["peso_est"] is None:
        if resumen["alturas_altas"]:
            st.info("Este grupo no tiene peso confiable (solo fotos >20 m; "
                    "para pesar hay que volar a 10-20 m). No se guarda "
                    "pesada sin peso.")
        else:
            st.info("Este grupo no tiene animales completos pesados; no hay "
                    "peso para guardar en la ficha.")
        return

    try:
        clientes = db.listar_clientes()
    except Exception as e:
        st.error(f"No pude leer los clientes de la base: {e}")
        return
    if not clientes:
        st.warning("No hay clientes cargados en la base.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    cli = c1.selectbox(
        "Cliente", clientes, key=f"cli_{key}",
        format_func=lambda c: c["nombre"] + (
            f" ({c['establecimiento']})" if c.get("establecimiento") else ""),
    )
    lotes = db.listar_lotes(cliente_id=cli["id"])
    if not lotes:
        c2.warning("Ese cliente no tiene lotes.")
        return
    lote = c2.selectbox(
        "Lote", lotes, key=f"lote_{key}",
        format_func=lambda l: f"{l['identificador']}" + (
            f" — corral {l['corral']}" if l.get("corral") else ""),
    )
    fecha = c3.date_input("Fecha de la recorrida", value=fecha_default,
                          key=f"fecha_{key}")

    pesos = resumen["pesos"]
    desvio = statistics.stdev(pesos) if len(pesos) > 1 else 0.0
    conteo = resumen["conteo_max"]
    peso_prom = resumen["peso_est"]
    notas = (f"Drone: {conteo} detectados, {resumen['n_pesados']} pesados "
             f"(media recortada {rec.FRAC_RECORTE:.0%}, fotos ≤20 m, "
             f"máscara sin sombra)")
    st.caption(f"Se guardará: {conteo} animales, peso promedio "
               f"{peso_prom:.0f} kg, desvío {desvio:.0f} kg, método "
               f"'drone'. Observación: “{notas}”")

    if st.button("Guardar en ficha", key=f"btn_{key}", type="primary"):
        try:
            pid = db.guardar_pesada(
                lote_id=lote["id"],
                fecha=fecha.isoformat(),
                metodo="drone",
                cantidad_animales=conteo,
                peso_promedio_kg=round(peso_prom, 1),
                peso_total_kg=round(peso_prom * conteo, 1),
                desvio_kg=round(desvio, 1),
                pesos_individuales=[round(p, 1) for p in pesos],
                video_path="",
                notas=notas,
            )
        except Exception as e:
            st.error(f"No pude guardar la pesada: {e}")
            return
        destino = ("la ficha online" if backend == "postgres"
                   else "la base LOCAL (no visible online)")
        st.success(f"Pesada #{pid} guardada en el lote "
                   f"{lote['identificador']} de {cli['nombre']} "
                   f"({fecha.isoformat()}). Ya aparece en {destino}.")


@st.cache_data(show_spinner=False)
def _info_video_cacheada(path_str: str, mtime: float):
    """Duración/fps del MP4 leídos del header (barato). mtime invalida
    el cache si el archivo cambia. Devuelve None si cv2 no puede."""
    try:
        return rec.info_video(Path(path_str))
    except Exception:
        return None


def ui_guardar_conteo_video(res, nombre_video: str):
    """Selector cliente/lote + botón que registra el conteo del video en
    la ficha como movimiento 'conteo_drone' (signo 0: NO toca el stock).
    Sin peso: los MP4 del drone no traen altura, no hay escala posible."""
    st.markdown("##### Guardar el conteo en la ficha del cliente")
    if res.n_animales <= 0:
        st.info("No hay animales confirmados para guardar.")
        return
    try:
        clientes = db.listar_clientes()
    except Exception as e:
        st.error(f"No pude leer los clientes de la base: {e}")
        return
    if not clientes:
        st.warning("No hay clientes cargados en la base.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    cli = c1.selectbox(
        "Cliente", clientes, key="cli_video",
        format_func=lambda c: c["nombre"] + (
            f" ({c['establecimiento']})" if c.get("establecimiento") else ""),
    )
    lotes = db.listar_lotes(cliente_id=cli["id"])
    if not lotes:
        c2.warning("Ese cliente no tiene lotes.")
        return
    lote = c2.selectbox(
        "Lote", lotes, key="lote_video",
        format_func=lambda l: f"{l['identificador']}" + (
            f" — corral {l['corral']}" if l.get("corral") else ""),
    )
    fecha = c3.date_input("Fecha de la pasada", value=date.today(),
                          key="fecha_video")

    detalles = (f"Conteo por drone (video {nombre_video}): "
                f"{res.n_animales} animales únicos por tracking "
                f"({res.duracion_s/60:.1f} min de video, "
                f"1 de cada {res.stride} frames, BoT-SORT). "
                f"Sin peso: el video no registra altura de vuelo.")
    st.caption("Se guardará como movimiento **“Conteo por drone (no "
               "cambia el stock)”** en la ficha del lote — aparece en la "
               f"tabla de movimientos de la app online. Obs.: “{detalles}”")

    if st.button("Guardar conteo en ficha", key="btn_video",
                 type="primary"):
        try:
            mid = db.crear_movimiento_lote(
                lote_id=lote["id"],
                fecha=fecha.isoformat(),
                tipo="conteo_drone",
                cantidad=res.n_animales,
                detalles=detalles,
            )
        except Exception as e:
            st.error(f"No pude guardar el conteo: {e}")
            return
        destino = ("la ficha online" if backend == "postgres"
                   else "la base LOCAL (no visible online)")
        st.success(f"Conteo #{mid} guardado en el lote "
                   f"{lote['identificador']} de {cli['nombre']} "
                   f"({fecha.isoformat()}): {res.n_animales} animales. "
                   f"Ya aparece en {destino}.")


def galeria_anotadas(fotos, dir_anotadas: Path, key: str):
    anotadas = [dir_anotadas / f"{f.nombre}_anotada.jpg" for f in fotos]
    anotadas = [p for p in anotadas if p.exists()]
    if not anotadas:
        st.caption("Sin fotos anotadas (no hubo detecciones).")
        return
    with st.expander(f"Ver {len(anotadas)} fotos anotadas (contorno = "
                     "silueta sin sombra; 'borde' = no pesado)"):
        cols = st.columns(3)
        for i, p in enumerate(anotadas):
            cols[i % 3].image(str(p), caption=p.stem, use_container_width=True)


def mostrar_errores(fotos):
    errores = [f for f in fotos if f.error]
    if errores:
        with st.expander(f"⚠️ {len(errores)} fotos con error"):
            for f in errores:
                st.write(f"- {f.path.name}: {f.error}")


# =====================================================================
# Tabs
# =====================================================================
st.title("Recorridas con drone — conteo y peso calibrado")

tab_rec, tab_foto, tab_video = st.tabs(
    ["📂 Recorrida completa", "🖼️ Foto suelta", "🎬 Video (conteo)"])

# ---------------------------------------------------------------------
# TAB 1: Recorrida completa
# ---------------------------------------------------------------------
with tab_rec:
    # Menú de carpetas: subcarpetas de videos_drone/ que tengan JPGs,
    # ordenadas por cantidad de fotos, + opción de escribir otra ruta.
    _base_vd = DIR_FOTOS_DEFAULT.parent
    _opciones_carpetas = []
    try:
        for _d in sorted(_base_vd.iterdir()):
            if _d.is_dir() and _d.name != "resultados":
                _n_jpg = len(list(_d.glob("*.JPG"))) + len(list(_d.glob("*.jpg")))
                if _n_jpg:
                    _opciones_carpetas.append((f"{_d.name}  ({_n_jpg} fotos)", _d))
    except FileNotFoundError:
        pass
    _labels = [o[0] for o in _opciones_carpetas] + ["Otra carpeta (escribir ruta)…"]
    _sel = st.selectbox(
        "Carpeta con las fotos de la recorrida (JPG del drone)", _labels
    )
    if _sel == "Otra carpeta (escribir ruta)…":
        carpeta = st.text_input(
            "Ruta de la carpeta (arrastrá la carpeta desde el Finder "
            "hasta acá para pegar la ruta)",
            value=str(DIR_FOTOS_DEFAULT),
        )
    else:
        carpeta = str(dict(_opciones_carpetas)[_sel])
    if st.button("Procesar recorrida", type="primary"):
        try:
            fotos = rec.listar_fotos(Path(carpeta).expanduser())
        except FileNotFoundError:
            st.error(f"No existe la carpeta: {carpeta}")
            fotos = []
        if not fotos:
            if Path(carpeta).expanduser().is_dir():
                st.warning("No hay fotos JPG en esa carpeta.")
        else:
            st.write(f"{len(fotos)} fotos encontradas. Procesando con "
                     f"{modelo} (conf {conf}, imgsz {IMGSZ})…")
            dir_anotadas = Path(tempfile.mkdtemp(prefix="drone_anotadas_"))
            fotos = procesar_lote_fotos(fotos, dir_anotadas)
            st.session_state["recorrida"] = {
                "fotos": fotos,
                "grupos": rec.agrupar_fotos([f for f in fotos if not f.error]),
                "dir_anotadas": dir_anotadas,
                "cuando": datetime.now(),
            }

    res = st.session_state.get("recorrida")
    if res:
        fotos, grupos = res["fotos"], res["grupos"]
        st.markdown(f"### Resultados por grupo/corral "
                    f"({len(grupos)} grupos, {len(fotos)} fotos)")
        st.caption("Grupos = fotos contiguas del mismo vuelo (corte con "
                   ">4 min de diferencia). Conteo del corral = máximo en "
                   "una foto del grupo; peso = media recortada 10% de "
                   "animales completos en fotos ≤20 m.")
        st.dataframe(tabla_grupos(grupos), hide_index=True,
                     use_container_width=True)
        st.download_button(
            "⬇️ Descargar CSV por foto",
            data=csv_descarga(fotos),
            file_name=f"recorrida_{fecha_recorrida(fotos):%Y%m%d}.csv",
            mime="text/csv",
        )
        mostrar_errores(fotos)

        fdef = fecha_recorrida(fotos)
        for i, g in enumerate(grupos, 1):
            r = rec.resumen_grupo(g)
            peso_txt = (f"{r['peso_est']:.0f} kg"
                        if r["peso_est"] is not None else "s/peso")
            st.markdown("---")
            st.markdown(f"#### Grupo {i}: {g[0].nombre} a {g[-1].nombre} — "
                        f"{r['conteo_max']} animales, {peso_txt}")
            galeria_anotadas(g, res["dir_anotadas"], key=f"g{i}")
            ui_guardar_en_ficha(f"g{i}", r, fdef)

# ---------------------------------------------------------------------
# TAB 2: Foto suelta
# ---------------------------------------------------------------------
with tab_foto:
    subidas = st.file_uploader(
        "Subí una o varias fotos del drone (JPG con metadata DJI)",
        type=["jpg", "jpeg"], accept_multiple_files=True,
    )
    if subidas and st.button("Procesar fotos", type="primary",
                             key="btn_fotosueltas"):
        tmp = Path(tempfile.mkdtemp(prefix="drone_sueltas_"))
        fotos = []
        for up in subidas:
            destino = tmp / up.name
            destino.write_bytes(up.getbuffer())
            fotos.append(rec.leer_metadata_foto(destino))
        dir_anotadas = tmp / "anotadas"
        fotos = procesar_lote_fotos(fotos, dir_anotadas)
        st.session_state["sueltas"] = {
            "fotos": fotos, "dir_anotadas": dir_anotadas,
        }

    res_s = st.session_state.get("sueltas")
    if res_s:
        fotos = res_s["fotos"]
        for f in fotos:
            st.markdown("---")
            c1, c2 = st.columns([3, 2])
            anotada = res_s["dir_anotadas"] / f"{f.nombre}_anotada.jpg"
            c1.image(str(anotada if anotada.exists() else f.path),
                     use_container_width=True)
            with c2:
                st.markdown(f"**{f.path.name}**")
                if f.error:
                    st.error(f.error)
                    continue
                alt = (f"{f.altura_m:.0f} m" if f.altura_m is not None
                       else "sin altura XMP")
                gsd = (f"{f.gsd_cm_px:.2f} cm/px" if f.gsd_cm_px
                       else "sin GSD")
                st.write(f"Altura: {alt} · Escala: {gsd}")
                st.metric("Animales detectados", f.n_animales)
                if f.altura_m is not None and f.altura_m > rec.ALTURA_PESO_CONFIABLE_M:
                    st.warning("Foto a más de 20 m: el conteo vale, pero el "
                               "peso no es confiable (volar a 10-20 m).")
                elif f.pesos_kg:
                    _mr = rec.media_recortada(f.pesos_kg)
                    st.metric(
                        "Peso promedio (media recortada)",
                        f"{_mr:.0f} kg" if _mr is not None
                        else f"{sum(f.pesos_kg)/len(f.pesos_kg):.0f} kg",
                        help="Media recortada 10%: descarta los pesos "
                             "extremos (siluetas dobles o parciales). "
                             "Los animales cortados en el borde ya están "
                             "excluidos (etiqueta 'borde').",
                    )
                elif f.altura_m is None:
                    st.warning("Sin RelativeAltitude en el XMP: no se puede "
                               "calcular la escala ni el peso.")

        st.markdown("---")
        st.markdown("### Resumen del conjunto")
        r = rec.resumen_grupo(fotos)
        st.dataframe(tabla_grupos([fotos]), hide_index=True,
                     use_container_width=True)
        st.download_button("⬇️ Descargar CSV", data=csv_descarga(fotos),
                           file_name="fotos_sueltas.csv", mime="text/csv",
                           key="csv_sueltas")
        ui_guardar_en_ficha("sueltas", r, fecha_recorrida(fotos))

# ---------------------------------------------------------------------
# TAB 3: Video (conteo)
# ---------------------------------------------------------------------
with tab_video:
    st.info(
        "**El conteo por video es la referencia del corral completo; el "
        "peso sale de las fotos a 10-20 m.** Los MP4 del drone no traen "
        "altura (RelativeAltitude), así que acá NO se estima peso: solo "
        "se cuentan animales únicos con tracking (BoT-SORT) para no "
        "contar dos veces al que se mueve."
    )

    # Menú de videos: MP4/MOV de las carpetas de videos_drone/, con
    # tamaño y duración (leída del header, barata), + otra ruta a mano.
    # NO usamos file_uploader: son archivos de 300-500 MB y el uploader
    # los copiaría enteros; elegirlos del disco es directo.
    _videos = rec.listar_videos(DIR_FOTOS_DEFAULT.parent)
    _opciones_videos = []
    for _v in _videos:
        try:
            _stat = _v.stat()
        except OSError:
            continue
        _mb = _stat.st_size / 1_000_000
        _info = _info_video_cacheada(str(_v), _stat.st_mtime)
        _dur = (f", {_info['duracion_s']:.0f} s" if _info
                and _info.get("duracion_s") else "")
        _opciones_videos.append(
            (f"{_v.parent.name}/{_v.name}  ({_mb:.0f} MB{_dur})", _v))
    _labels_v = ([o[0] for o in _opciones_videos]
                 + ["Otra ruta (escribir)…"])
    _sel_v = st.selectbox("Video de la pasada (MP4 del drone)", _labels_v)
    if _sel_v == "Otra ruta (escribir)…":
        ruta_video = st.text_input(
            "Ruta del video (arrastrá el archivo desde el Finder hasta "
            "acá para pegar la ruta)", value="")
    else:
        ruta_video = str(dict(_opciones_videos)[_sel_v])

    cv1, cv2_ = st.columns([2, 1])
    stride = cv1.slider(
        "Procesar 1 de cada N frames", 1, 10, rec.VIDEO_STRIDE_DEFAULT,
        help="El pipeline viejo procesaba TODOS los frames (necesitaba "
             "la lona por frame para el peso). Para contar alcanza con "
             "menos: a 30 fps, 3 = 10 fps efectivos y BoT-SORT "
             "(track_buffer 90) mantiene los IDs sin problema. No subir "
             "de 5-6: el tracker necesita continuidad entre frames.")
    generar_video = cv2_.checkbox(
        "Generar video anotado", value=True,
        help="Apagalo para terminar antes: quedan igual 4 frames "
             "anotados de muestra.")

    if st.button("Contar animales del video", type="primary",
                 key="btn_procesar_video"):
        _p = Path(ruta_video).expanduser() if ruta_video.strip() else None
        if _p is None or not _p.is_file():
            st.error(f"No existe el video: {ruta_video or '(vacío)'}")
        else:
            barra_v = st.progress(0.0, text="Cargando modelo YOLO…")

            def _cb_video(frac: float, n_parcial: int) -> None:
                barra_v.progress(frac, text=f"Procesando… {frac*100:.0f}% "
                                 f"— {n_parcial} animales únicos hasta "
                                 f"ahora")

            _out_dir = Path(tempfile.mkdtemp(prefix="drone_video_"))
            try:
                _res_v = rec.contar_video(
                    _p, modelo=modelo, conf=conf, stride=stride,
                    out_dir=_out_dir, progress_cb=_cb_video,
                    generar_video=generar_video)
            except Exception as e:
                st.error(f"Error procesando el video: {e}")
                _res_v = None
            if _res_v is not None:
                barra_v.progress(1.0, text="Listo.")
                st.session_state["video_conteo"] = {
                    "res": _res_v, "video": _p.name,
                    "cuando": datetime.now(),
                }

    res_v = st.session_state.get("video_conteo")
    if res_v:
        r_v = res_v["res"]
        st.markdown(f"### Resultado — {res_v['video']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Conteo total del corral", f"{r_v.n_animales} animales")
        m2.metric("Duración procesada", f"{r_v.duracion_s/60:.1f} min")
        m3.metric("Frames procesados",
                  f"{r_v.n_frames_procesados} de {r_v.n_frames_video}")
        st.caption(
            f"Confirmado = track visto en ≥{r_v.min_frames} frames "
            f"procesados (mismo umbral que el pipeline histórico de "
            f"video); {r_v.n_tracks_totales} IDs vistos en total, "
            f"renumerados 1..{r_v.n_animales}. Tracker BoT-SORT "
            f"(botsort_robusto.yaml, track_buffer 90), stride "
            f"{r_v.stride}, modelo {modelo}."
        )
        if r_v.video_anotado and r_v.video_anotado.exists():
            if r_v.codec == "avc1":
                st.video(str(r_v.video_anotado))
            else:
                st.warning(
                    "El video anotado quedó en códec mp4v (el navegador "
                    "no lo reproduce). Abrilo con QuickTime: "
                    f"`{r_v.video_anotado}`")
        if r_v.frames_muestra:
            _existentes = [p for p in r_v.frames_muestra if p.exists()]
            if _existentes:
                with st.expander(
                        f"Ver {len(_existentes)} frames anotados de "
                        "muestra", expanded=not r_v.video_anotado):
                    cols_v = st.columns(2)
                    for i, p in enumerate(_existentes):
                        cols_v[i % 2].image(str(p), caption=p.stem,
                                            use_container_width=True)
        st.markdown("---")
        ui_guardar_conteo_video(r_v, res_v["video"])
