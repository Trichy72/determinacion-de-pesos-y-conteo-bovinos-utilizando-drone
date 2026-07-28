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
modelo = st.sidebar.selectbox(
    "Modelo YOLO", modelos_disponibles,
    index=modelos_disponibles.index(rec.MODELO_DEFAULT),
    help="Con un modelo -seg el peso se calcula por máscara (sin sombra). "
         "Sin -seg cae al método bbox x 0.69 (menos preciso).",
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

tab_rec, tab_foto = st.tabs(["📂 Recorrida completa", "🖼️ Foto suelta"])

# ---------------------------------------------------------------------
# TAB 1: Recorrida completa
# ---------------------------------------------------------------------
with tab_rec:
    carpeta = st.text_input(
        "Carpeta con las fotos de la recorrida (JPG del drone)",
        value=str(DIR_FOTOS_DEFAULT),
    )
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
                    st.metric("Peso promedio (animales completos)",
                              f"{sum(f.pesos_kg)/len(f.pesos_kg):.0f} kg")
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
