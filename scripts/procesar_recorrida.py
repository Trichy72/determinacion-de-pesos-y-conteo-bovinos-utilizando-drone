#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procesamiento batch de la recorrida con drone del 24/7/2026.  (v3: calibración)

El pipeline (EXIF/GSD, detección con máscaras, filtro de borde, media
recortada, CSV) vive en src/recorrida.py — MISMA fuente de verdad que
drone_app.py. Este script conserva lo específico de la recorrida del
24/7: mapeo hardcodeado de fotos -> corral (GRUPOS_REALES con la verdad
de campo), comparación estimado vs real y calibración automática.

Recorre las fotos DJI de videos_drone/tarjeta_drone/, y para cada una:
  1. Lee altura relativa (XMP RelativeAltitude) y datos de cámara (EXIF)
     para calcular la escala GSD (cm/píxel) sin necesidad del cuadrado
     de referencia en el piso.
  2. Detecta bovinos con YOLO de SEGMENTACIÓN (default yolov8l-seg.pt,
     clases COCO de cuadrúpedos grandes, conf baja 0.10, imgsz 3840 para
     usar casi toda la resolución 4000x2250 — clave para fotos de 50/100 m
     donde un animal mide ~50 px).
  3. Estima peso por animal con el ÁREA DE LA MÁSCARA de segmentación
     (píxeles de silueta x GSD²), que EXCLUYE LA SOMBRA del animal.
     Si el modelo elegido NO es de segmentación (--modelo yolov8m.pt),
     cae al método anterior: área bbox x 0.69.

Reglas calibradas (v3, 27/7/26 vs balanza La Esperanza):
  - FILTRO DE BORDE: detecciones cuya bbox toca el borde de la imagen
    (margen 10 px) cuentan para el CONTEO pero NO para el peso.
  - MEDIA RECORTADA: el peso promedio del corral descarta el 10% inferior
    y superior de los pesos individuales (animales completos del grupo).
  - Peso SOLO con fotos <= 20 m (fotos altas dan silueta gruesa: +22%).
  - GRUPOS_REALES: mapeo rango de archivo -> corral con datos de campo.
  - CALIBRACIÓN AUTOMÁTICA: con los corrales que tienen peso real se
    calcula f y se propone a_calibrado = a x f (tabla ANTES/DESPUÉS +
    videos_drone/resultados/calibracion.json). config.yaml NO se toca.

Salidas en videos_drone/resultados/:
  - resultados_fotos.csv   (una fila por foto)
  - anotadas/*.jpg         (todas las fotos con detecciones: contornos,
                            pesos y conteo)
  - resumen.txt            (por corral, con datos reales y calibración)
  - calibracion.json       (factor de calibración propuesto)

Uso en la Mac (desde la raíz del repo):
    source .venv/bin/activate
    python scripts/procesar_recorrida.py
    python scripts/procesar_recorrida.py --solo-conteo --conf 0.15
    python scripts/procesar_recorrida.py --desde DJI_0079 --hasta DJI_0148 \
        --categoria novillo
    python scripts/procesar_recorrida.py --modelo yolov8m.pt --imgsz 1920  # v1

NOTA: los imports pesados (ultralytics, cv2, numpy, yaml) se hacen recién
dentro de main()/procesar_foto() para que el parsing EXIF/GSD se pueda
testear sin GPU ni ultralytics instalado.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recorrida import (  # noqa: E402  (pipeline compartido)
    ALTURA_PESO_CONFIABLE_M,
    FRAC_RECORTE,
    MARGEN_BORDE_PX,
    FotoMeta,
    cargar_weight_model,
    crear_yolo,
    es_modelo_seg,
    escribir_csv,
    listar_fotos as _listar_fotos_dir,
    media_recortada,  # noqa: F401  (reexport histórico del script)
    procesar_foto,
    resumen_grupo,
    stats_pesos as _stats,
)

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
DIR_FOTOS = REPO_ROOT / "videos_drone" / "tarjeta_drone"
DIR_RESULTADOS = REPO_ROOT / "videos_drone" / "resultados"
DIR_ANOTADAS = DIR_RESULTADOS / "anotadas"

# Rango de fotos del cliente Roxdan (el resto es La Esperanza Argentina)
ROXDAN_DESDE, ROXDAN_HASTA = 63, 78

# ----------------------------------------------------------------------
# Verdad de campo de la recorrida del 24/7/2026: rango de numeración de
# archivo -> corral, con conteo real y rango de peso real (kg) declarado
# por el cliente. peso_real=None => corral sin dato de peso.
# ----------------------------------------------------------------------
GRUPOS_REALES = [
    {"corral": "Roxdan", "cliente": "Roxdan",
     "desde": 63, "hasta": 78,
     "conteo_real": None, "peso_real": None, "nota": "sin datos reales"},
    {"corral": "Corral 1", "cliente": "La Esperanza Argentina",
     "desde": 79, "hasta": 84,
     "conteo_real": 185, "peso_real": None, "nota": "terneras, sin peso"},
    {"corral": "Corral 3", "cliente": "La Esperanza Argentina",
     "desde": 86, "hasta": 116,
     "conteo_real": 78, "peso_real": (460, 500), "nota": "novillos"},
    {"corral": "Corral 5", "cliente": "La Esperanza Argentina",
     "desde": 117, "hasta": 129,
     "conteo_real": 78, "peso_real": (440, 460), "nota": "novillos"},
    {"corral": "Corral 6", "cliente": "La Esperanza Argentina",
     "desde": 132, "hasta": 148,
     "conteo_real": 96, "peso_real": (380, 440), "nota": "animales"},
]


def cliente_de(numero: int) -> str:
    if ROXDAN_DESDE <= numero <= ROXDAN_HASTA:
        return "Roxdan"
    return "La Esperanza Argentina"


def listar_fotos(desde: Optional[str], hasta: Optional[str]) -> List[FotoMeta]:
    """Lista las fotos JPG de la tarjeta, con filtro opcional --desde/--hasta."""
    try:
        return _listar_fotos_dir(DIR_FOTOS, desde, hasta, cliente_fn=cliente_de)
    except FileNotFoundError:
        sys.exit(f"ERROR: no existe la carpeta de fotos: {DIR_FOTOS}")


# ----------------------------------------------------------------------
# Corrales reales: estadísticas por corral y calibración automática
# ----------------------------------------------------------------------

def stats_corrales(fotos: List[FotoMeta]) -> List[dict]:
    """Agrupa las fotos según GRUPOS_REALES y calcula por corral:
    conteo máximo por foto, n animales pesados (completos), peso promedio
    estimado (media recortada 10%) y desvío % vs punto medio del rango real.

    El peso usa SOLO fotos <= ALTURA_PESO_CONFIABLE_M (regla calibrada
    27/7/26); si el corral no tiene ninguna (ej. Corral 6, fotografiado
    a 50 m) queda sin peso y se marca con `alturas_altas`.
    """
    out = []
    for g in GRUPOS_REALES:
        fg = [f for f in fotos if g["desde"] <= f.numero <= g["hasta"]]
        if not fg:
            continue
        c = dict(g)
        c["fotos"] = fg
        r = resumen_grupo(fg)
        for k in ("conteo_max", "foto_max", "n_pesados", "peso_est",
                  "alturas_altas"):
            c[k] = r[k]

        real = g["peso_real"]
        c["real_medio"] = (real[0] + real[1]) / 2.0 if real else None
        c["desvio_pct"] = (
            (c["peso_est"] - c["real_medio"]) / c["real_medio"] * 100.0
            if c["peso_est"] is not None and c["real_medio"] else None
        )
        out.append(c)
    return out


def calibrar(corrales: List[dict], a_original: float) -> Optional[dict]:
    """Factor global f = promedio ponderado (por n animales pesados) de
    peso_real_medio / peso_estimado, sobre los corrales con peso real.
    Propone a_calibrado = a_original x f. NO toca config.yaml."""
    usados = [c for c in corrales
              if c["real_medio"] and c["peso_est"] and c["n_pesados"] > 0]
    if not usados:
        return None
    den = sum(c["n_pesados"] for c in usados)
    f = sum(c["n_pesados"] * (c["real_medio"] / c["peso_est"])
            for c in usados) / den
    return {
        "factor": round(f, 4),
        "a_original": a_original,
        "a_calibrado": round(a_original * f, 2),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "corrales_usados": [c["corral"] for c in usados],
    }


def tabla_calibracion(corrales: List[dict], calib: dict) -> List[str]:
    """Tabla ANTES/DESPUÉS: estimado original y corregido por f vs real."""
    f = calib["factor"]
    lineas = [
        f"Factor global f = {f:.4f} (ponderado por n animales pesados; "
        f"corrales: {', '.join(calib['corrales_usados'])})",
        f"Coeficiente a: {calib['a_original']:.1f} -> a_calibrado = "
        f"{calib['a_calibrado']:.1f}",
        "",
        f"{'Corral':<10} {'Est. ANTES':>10} {'Est. x f':>10} {'Real medio':>10} "
        f"{'Err ANTES':>10} {'Err DESPUES':>11}",
    ]
    for c in corrales:
        if not (c["real_medio"] and c["peso_est"]):
            continue
        antes = c["peso_est"]
        despues = antes * f
        err_a = (antes - c["real_medio"]) / c["real_medio"] * 100.0
        err_d = (despues - c["real_medio"]) / c["real_medio"] * 100.0
        lineas.append(
            f"{c['corral']:<10} {antes:>7.0f} kg {despues:>7.0f} kg "
            f"{c['real_medio']:>7.0f} kg {err_a:>+9.1f}% {err_d:>+10.1f}%"
        )
    return lineas


def escribir_resumen(fotos: List[FotoMeta], path: Path, categoria: str,
                     solo_conteo: bool, corrales: List[dict],
                     calib: Optional[dict]) -> None:
    lineas = [
        "RESUMEN RECORRIDA DRONE v3 - conteo y peso estimado POR CORRAL",
        f"Generado: {datetime.now():%Y-%m-%d %H:%M}   "
        f"Categoria peso: {categoria}   Fotos procesadas: {len(fotos)}",
        "Peso por area de MASCARA de segmentacion (excluye la sombra),",
        "SOLO animales completos (bbox que no toca el borde de la imagen,",
        f"margen {MARGEN_BORDE_PX} px). Peso promedio del corral = MEDIA "
        f"RECORTADA {FRAC_RECORTE:.0%}",
        f"de los pesos individuales, priorizando fotos <= "
        f"{ALTURA_PESO_CONFIABLE_M:.0f} m.",
        "Conteo del corral = MAXIMO de animales detectados en una foto.",
        "=" * 70,
    ]
    cliente_actual = None
    for c in corrales:
        if c["cliente"] != cliente_actual:
            cliente_actual = c["cliente"]
            lineas += ["", f"CLIENTE: {cliente_actual}", "-" * 70]

        g = c["fotos"]
        horas = [f.hora for f in g if f.hora]
        rango_h = (f"{min(horas):%H:%M}-{max(horas):%H:%M}" if horas else "s/hora")
        alturas = sorted({round(f.altura_m) for f in g if f.altura_m is not None})

        lineas.append(
            f"{c['corral'].upper()} ({c['nota']}): {g[0].nombre} a "
            f"{g[-1].nombre} ({len(g)} fotos, {rango_h}, alturas {alturas} m)"
        )
        conteo_real = (f"{c['conteo_real']}" if c["conteo_real"] is not None
                       else "s/dato")
        lineas.append(
            f"  Conteo: maximo por foto {c['conteo_max']} "
            f"(en {c['foto_max']})  |  real: {conteo_real}"
        )
        if not solo_conteo:
            if c["peso_est"] is not None:
                lineas.append(
                    f"  Peso estimado (media recortada, {c['n_pesados']} "
                    f"animales completos): {c['peso_est']:.0f} kg"
                )
            elif c["alturas_altas"]:
                lineas.append(
                    "  Peso estimado: SIN PESO CONFIABLE (solo hay fotos "
                    ">20 m; para pesar volar a 10-20 m)"
                )
            else:
                lineas.append("  Peso estimado: sin animales completos pesados")
            if c["peso_real"]:
                r0, r1 = c["peso_real"]
                lineas.append(
                    f"  Peso real: {r0}-{r1} kg (punto medio "
                    f"{c['real_medio']:.0f} kg)"
                )
                if c["desvio_pct"] is not None:
                    lineas.append(
                        f"  Desvio estimado vs real: {c['desvio_pct']:+.1f}%"
                    )
            else:
                lineas.append("  Peso real: sin dato")
        conteos = ", ".join(f"{f.nombre}={f.n_animales}" for f in g)
        lineas.append(f"  Detalle conteos: {conteos}")
        lineas.append("")

    numeros_mapeados = {f.numero for c in corrales for f in c["fotos"]}
    sueltas = [f for f in fotos if f.numero not in numeros_mapeados]
    if sueltas:
        lineas += ["FOTOS FUERA DEL MAPEO DE CORRALES (no usadas arriba):",
                   "  " + ", ".join(f.nombre for f in sueltas), ""]

    if calib:
        lineas += ["=" * 70,
                   "CALIBRACION AUTOMATICA (corrales con peso real)",
                   "-" * 70]
        lineas += tabla_calibracion(corrales, calib)
        lineas += [
            "",
            "ERROR FINAL POR CORRAL TRAS CALIBRAR (columna 'Err DESPUES').",
            "Factor guardado en videos_drone/resultados/calibracion.json.",
            "config.yaml NO fue modificado: aplicar a_calibrado se decide",
            "con el usuario.",
        ]
    elif not solo_conteo:
        lineas += ["=" * 70,
                   "CALIBRACION: no se pudo calcular (ningun corral con peso "
                   "real y estimado a la vez)."]

    errores = [f for f in fotos if f.error]
    if errores:
        lineas += ["", "FOTOS CON ERROR:"] + [
            f"  {f.path.name}: {f.error}" for f in errores
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lineas) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Conteo y peso estimado de bovinos en fotos DJI de la recorrida."
    )
    ap.add_argument("--solo-conteo", action="store_true",
                    help="Solo contar animales, sin estimar peso")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="Umbral de confianza YOLO (default 0.10)")
    ap.add_argument("--categoria", default="novillo",
                    choices=["ternero", "vaquillona", "novillo",
                             "vaca_adulta", "toro", "desconocido"],
                    help="Categoría para el factor de peso (default novillo)")
    ap.add_argument("--desde", default=None, metavar="DJI_0063",
                    help="Primera foto a procesar (inclusive)")
    ap.add_argument("--hasta", default=None, metavar="DJI_0078",
                    help="Última foto a procesar (inclusive)")
    ap.add_argument("--modelo", default="yolov8l-seg.pt",
                    help="Modelo YOLO (default yolov8l-seg.pt; con un modelo "
                         "sin '-seg' el peso vuelve al método bbox x 0.69)")
    ap.add_argument("--imgsz", type=int, default=3840,
                    help="Tamaño de inferencia YOLO (default 3840, casi "
                         "resolución nativa; ultralytics lo ajusta a "
                         "múltiplo de 32)")
    ap.add_argument("--no-anotar", action="store_true",
                    help="No guardar JPGs anotados (default: se anotan "
                         "TODAS las fotos con detecciones)")
    args = ap.parse_args()

    print("Leyendo metadata EXIF/XMP de las fotos...")
    fotos = listar_fotos(args.desde, args.hasta)
    if not fotos:
        sys.exit("No hay fotos JPG que procesar con ese filtro.")
    print(f"  {len(fotos)} fotos encontradas en {DIR_FOTOS}")

    es_seg = es_modelo_seg(args.modelo)
    print(f"Cargando YOLO ({args.modelo}, conf={args.conf}, "
          f"imgsz={args.imgsz}, peso por {'mascara' if es_seg else 'bbox x 0.69'})...")
    yolo = crear_yolo(args.modelo)
    # El weight_model se necesita aun con --solo-conteo para el filtro
    # anti-sombra por área; solo se saltea la estimación de peso.
    weight_model = cargar_weight_model()

    for i, meta in enumerate(fotos, 1):
        alt = f"{meta.altura_m:.0f}m" if meta.altura_m is not None else "alt?"
        gsd = f"{meta.gsd_cm_px:.2f}cm/px" if meta.gsd_cm_px else "gsd?"
        print(f"[{i}/{len(fotos)}] {meta.path.name} ({alt}, {gsd}, "
              f"{meta.cliente})...", end=" ", flush=True)
        if meta.error:
            print(f"SALTEADA: {meta.error}")
            continue
        meta.metodo = "mask" if es_seg else "bbox"
        try:
            procesar_foto(meta, yolo, weight_model, args.categoria,
                          args.solo_conteo, not args.no_anotar,
                          DIR_ANOTADAS, args.conf, args.imgsz)
        except Exception as e:
            meta.error = f"Error en detección: {e}"
            print(f"ERROR: {e}")
            continue
        prom, _, _ = _stats(meta.pesos_kg)
        extra = f", peso prom {prom} kg" if prom else ""
        if meta.n_descartadas:
            extra += f" ({meta.n_descartadas} descartadas por area)"
        print(f"{meta.n_animales} animales{extra}")

    csv_path = DIR_RESULTADOS / "resultados_fotos.csv"
    resumen_path = DIR_RESULTADOS / "resumen.txt"
    calib_path = DIR_RESULTADOS / "calibracion.json"
    escribir_csv(fotos, csv_path, args.modelo)

    corrales = stats_corrales(fotos)
    calib = None
    if not args.solo_conteo:
        calib = calibrar(corrales, weight_model.coef_a)
        if calib:
            calib_path.parent.mkdir(parents=True, exist_ok=True)
            calib_path.write_text(
                json.dumps(calib, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    escribir_resumen(fotos, resumen_path, args.categoria, args.solo_conteo,
                     corrales, calib)

    print()
    print("=" * 60)
    if calib:
        print("CALIBRACION AUTOMATICA (ANTES/DESPUES vs peso real):")
        for linea in tabla_calibracion(corrales, calib):
            print("  " + linea)
        print(f"  Guardada en: {calib_path} (config.yaml NO modificado)")
        print("=" * 60)
    print("LISTO. Resultados en:")
    print(f"  CSV por foto:    {csv_path}")
    print(f"  Resumen corrales: {resumen_path}")
    if calib:
        print(f"  Calibracion:     {calib_path}")
    if not args.no_anotar:
        print(f"  Fotos anotadas:  {DIR_ANOTADAS}/ (contorno de mascara = "
              "silueta SIN sombra; 'borde' = animal cortado, no pesado)")
    print("Ver en resumen.txt el desvio % por corral vs datos reales y el")
    print("error final tras aplicar el factor de calibracion.")


if __name__ == "__main__":
    main()
