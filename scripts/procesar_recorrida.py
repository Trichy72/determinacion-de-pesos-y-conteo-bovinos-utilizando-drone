#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procesamiento batch de la recorrida con drone del 24/7/2026.

Recorre las fotos DJI de videos_drone/tarjeta_drone/, y para cada una:
  1. Lee altura relativa (XMP RelativeAltitude) y datos de cámara (EXIF)
     para calcular la escala GSD (cm/píxel) sin necesidad del cuadrado
     de referencia en el piso.
  2. Detecta bovinos con YOLO (reusa src/detector.py: clases COCO de
     cuadrúpedos grandes, conf bajo, tiles SAHI-style para fotos altas).
  3. Cuenta animales y estima peso por animal reusando el estimador del
     pipeline (src/weight_estimator.py: área silueta m² -> kg con modelo
     alométrico, factor bbox->silueta 0.69, factores por categoría).

Salidas en videos_drone/resultados/:
  - resultados_fotos.csv   (una fila por foto)
  - anotadas/*.jpg         (fotos ≤30 m anotadas con cajas, pesos y conteo)
  - resumen.txt            (por cliente y grupo de fotos contiguas)

Uso en la Mac (desde la raíz del repo):
    source .venv/bin/activate
    python scripts/procesar_recorrida.py
    python scripts/procesar_recorrida.py --solo-conteo --conf 0.2
    python scripts/procesar_recorrida.py --desde DJI_0079 --hasta DJI_0148 \
        --categoria novillo

NOTA: los imports pesados (ultralytics, cv2, numpy, yaml) se hacen recién
dentro de main()/run_deteccion() para que el parsing EXIF/GSD se pueda
testear sin GPU ni ultralytics instalado.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PIL import Image

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DIR_FOTOS = REPO_ROOT / "videos_drone" / "tarjeta_drone"
DIR_RESULTADOS = REPO_ROOT / "videos_drone" / "resultados"
DIR_ANOTADAS = DIR_RESULTADOS / "anotadas"

# EXIF tags (ids numéricos, evita depender de PIL.ExifTags en versiones viejas)
TAG_DATETIME_ORIGINAL = 36867
TAG_DATETIME = 306
TAG_FOCAL_LENGTH = 37386
TAG_FOCAL_35MM = 41989
TAG_MODEL = 272

# Ancho de sensor (mm) por modelo de cámara DJI, fallback si no hay
# FocalLengthIn35mmFilm en el EXIF. FC3682 = DJI Mini 3 / Mini 3 Pro.
SENSOR_WIDTH_MM = {
    "FC3682": 9.7,     # 1/1.3"
    "FC3582": 9.7,     # Mini 3 Pro
    "FC7303": 6.17,    # Mini 2, 1/2.3"
    "FC220": 6.17,     # Mavic Pro
    "FC3170": 6.4,     # Mavic Air 2
    "L1D-20C": 13.2,   # Mavic 2 Pro, 1"
}

# Rango de fotos del cliente Roxdan (el resto es La Esperanza Argentina)
ROXDAN_DESDE, ROXDAN_HASTA = 63, 78

# Umbrales de detección/agrupación
ALTURA_MODO_TILES_M = 35.0     # por encima de esto, inferencia por tiles
ALTURA_PESO_CONFIABLE_M = 20.5 # fotos ≤20 m: las más confiables para peso
GAP_GRUPO_SEG = 240            # >4 min entre fotos => nuevo grupo/corral


# ----------------------------------------------------------------------
# Metadata por foto: EXIF + XMP -> altura, hora, GSD
# ----------------------------------------------------------------------

@dataclass
class FotoMeta:
    path: Path
    nombre: str                      # "DJI_0063"
    numero: int                      # 63
    ancho_px: int = 0
    alto_px: int = 0
    hora: Optional[datetime] = None
    altura_m: Optional[float] = None
    focal_mm: Optional[float] = None
    f35_mm: Optional[float] = None
    modelo_camara: str = ""
    gsd_cm_px: Optional[float] = None
    cliente: str = ""
    # resultados de detección (se completan después)
    n_animales: int = 0
    pesos_kg: List[float] = field(default_factory=list)
    error: str = ""


def leer_xmp_altura(path: Path) -> Optional[float]:
    """Extrae drone-dji:RelativeAltitude del bloque XMP del JPEG."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    m = re.search(rb'RelativeAltitude\s*=\s*"([+\-]?\d+(?:\.\d+)?)"', data)
    if not m:
        # Variante como elemento XML: <drone-dji:RelativeAltitude>+30.0</...>
        m = re.search(
            rb"RelativeAltitude>\s*([+\-]?\d+(?:\.\d+)?)\s*<", data
        )
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_exif_dt(valor) -> Optional[datetime]:
    try:
        return datetime.strptime(str(valor), "%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def calcular_gsd_cm_px(
    altura_m: float,
    ancho_px: int,
    f35_mm: Optional[float],
    focal_mm: Optional[float],
    modelo: str,
) -> Optional[float]:
    """
    GSD (cm/px) = (altura * ancho_sensor) / (focal * ancho_imagen).

    Preferimos el equivalente 35 mm (ancho full-frame = 36 mm), que las
    cámaras DJI graban en EXIF y evita conocer el sensor exacto:
        GSD = altura_cm * 36 / (f35 * ancho_px)
    Fallback: focal real + ancho de sensor tabulado por modelo.
    """
    if altura_m is None or altura_m <= 0 or ancho_px <= 0:
        return None
    if f35_mm and f35_mm > 0:
        return altura_m * 100.0 * 36.0 / (f35_mm * ancho_px)
    sensor_w = SENSOR_WIDTH_MM.get(modelo.strip())
    if sensor_w and focal_mm and focal_mm > 0:
        return altura_m * 100.0 * sensor_w / (focal_mm * ancho_px)
    return None


def cliente_de(numero: int) -> str:
    if ROXDAN_DESDE <= numero <= ROXDAN_HASTA:
        return "Roxdan"
    return "La Esperanza Argentina"


def leer_metadata_foto(path: Path) -> FotoMeta:
    m = re.search(r"(\d+)", path.stem)
    numero = int(m.group(1)) if m else -1
    meta = FotoMeta(path=path, nombre=path.stem, numero=numero,
                    cliente=cliente_de(numero))
    try:
        with Image.open(path) as im:
            meta.ancho_px, meta.alto_px = im.size
            exif = im._getexif() or {}
    except Exception as e:  # foto corrupta: seguimos con el resto
        meta.error = f"No pude leer EXIF: {e}"
        return meta

    meta.hora = _parse_exif_dt(
        exif.get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME)
    )
    try:
        meta.focal_mm = float(exif.get(TAG_FOCAL_LENGTH) or 0) or None
    except (TypeError, ValueError):
        meta.focal_mm = None
    try:
        meta.f35_mm = float(exif.get(TAG_FOCAL_35MM) or 0) or None
    except (TypeError, ValueError):
        meta.f35_mm = None
    meta.modelo_camara = str(exif.get(TAG_MODEL) or "").strip("\x00 ").strip()

    meta.altura_m = leer_xmp_altura(path)
    if meta.altura_m is not None:
        meta.gsd_cm_px = calcular_gsd_cm_px(
            meta.altura_m, meta.ancho_px, meta.f35_mm,
            meta.focal_mm, meta.modelo_camara,
        )
    return meta


def listar_fotos(desde: Optional[str], hasta: Optional[str]) -> List[FotoMeta]:
    """Lista las fotos JPG de la tarjeta, con filtro opcional --desde/--hasta."""
    if not DIR_FOTOS.is_dir():
        sys.exit(f"ERROR: no existe la carpeta de fotos: {DIR_FOTOS}")

    def num_de(nombre: str) -> int:
        m = re.search(r"(\d+)", nombre)
        return int(m.group(1)) if m else -1

    n_desde = num_de(desde) if desde else None
    n_hasta = num_de(hasta) if hasta else None

    fotos = []
    for p in sorted(DIR_FOTOS.iterdir()):
        if p.suffix.upper() not in (".JPG", ".JPEG"):
            continue
        n = num_de(p.stem)
        if n_desde is not None and n < n_desde:
            continue
        if n_hasta is not None and n > n_hasta:
            continue
        fotos.append(leer_metadata_foto(p))
    return fotos


# ----------------------------------------------------------------------
# Agrupación en "corrales" (fotos contiguas del mismo vuelo)
# ----------------------------------------------------------------------

def agrupar_fotos(fotos: List[FotoMeta]) -> List[List[FotoMeta]]:
    """Agrupa fotos contiguas (mismo cliente, sin saltos grandes de hora ni
    de numeración) como aproximación de "mismo corral/vuelo"."""
    ordenadas = sorted(fotos, key=lambda f: (f.hora or datetime.min, f.numero))
    grupos: List[List[FotoMeta]] = []
    for f in ordenadas:
        if grupos:
            prev = grupos[-1][-1]
            mismo_cliente = f.cliente == prev.cliente
            gap_ok = True
            if f.hora and prev.hora:
                gap_ok = (f.hora - prev.hora).total_seconds() <= GAP_GRUPO_SEG
            if mismo_cliente and gap_ok:
                grupos[-1].append(f)
                continue
        grupos.append([f])
    return grupos


# ----------------------------------------------------------------------
# Detección + peso (imports pesados adentro: corre sin ultralytics
# mientras no se llame)
# ----------------------------------------------------------------------

def cargar_weight_model():
    """Reusa el estimador del pipeline con los coeficientes de config.yaml."""
    sys.path.insert(0, str(REPO_ROOT))
    from src.weight_estimator import WeightModel

    cfg_path = REPO_ROOT / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            return WeightModel.from_config(cfg)
        except Exception as e:
            print(f"  Aviso: no pude leer config.yaml ({e}), uso defaults.")
    return WeightModel()


def crear_detector(modelo: str, conf: float, imgsz: int):
    """Reusa src/detector.py (clases COCO aéreas, NMS iou 0.5, tiles SAHI)."""
    sys.path.insert(0, str(REPO_ROOT))
    from src.detector import CattleDetector

    model_path = str(REPO_ROOT / modelo) if (REPO_ROOT / modelo).exists() else modelo
    return CattleDetector(
        model_path=model_path,
        conf=conf,
        iou=0.5,
        imgsz=imgsz,
    )


def procesar_foto(meta, detector, weight_model, categoria, solo_conteo,
                  anotar, out_anotadas):
    """Detecta, cuenta, estima pesos y (opcional) guarda la foto anotada."""
    import cv2

    img = cv2.imread(str(meta.path))
    if img is None:
        meta.error = "No pude abrir la imagen con OpenCV"
        return

    # Fotos altas (50/100 m): tiles SAHI-style del detector existente para
    # no perder animales chicos; fotos bajas: full-frame.
    alta = meta.altura_m is not None and meta.altura_m > ALTURA_MODO_TILES_M
    detector.modo_tropa_densa = alta
    detector.tile_grid = 3 if alta else 2

    detecciones = detector.detect(img)
    meta.n_animales = len(detecciones)

    pesos = []
    if not solo_conteo and meta.gsd_cm_px:
        m_per_px = meta.gsd_cm_px / 100.0
        for det in detecciones:
            # det.area_px ya viene con máscara real o bbox*0.69 (silueta)
            area_m2 = det.area_px * (m_per_px ** 2)
            peso = weight_model.estimate(area_m2, categoria=categoria)
            pesos.append(peso)  # None si el área cae fuera de rango
    meta.pesos_kg = [p for p in pesos if p is not None]

    if anotar:
        _guardar_anotada(img, meta, detecciones, pesos, out_anotadas)


def _guardar_anotada(img, meta, detecciones, pesos, out_dir):
    import cv2

    verde = (0, 255, 0)
    for i, det in enumerate(detecciones):
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), verde, 3)
        peso = pesos[i] if i < len(pesos) else None
        etiqueta = f"{peso:.0f} kg" if peso else f"{det.confidence:.2f}"
        cv2.putText(img, etiqueta, (x1, max(30, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(img, etiqueta, (x1, max(30, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    texto = f"{meta.nombre}  |  {meta.n_animales} animales"
    if meta.altura_m is not None:
        texto += f"  |  {meta.altura_m:.0f} m"
    cv2.rectangle(img, (10, 10), (60 + 32 * len(texto), 90), (0, 0, 0), -1)
    cv2.putText(img, texto, (30, 68), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (0, 255, 255), 4)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{meta.nombre}_anotada.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------------
# Salidas
# ----------------------------------------------------------------------

def _stats(pesos: List[float]):
    if not pesos:
        return "", "", ""
    prom = sum(pesos) / len(pesos)
    return f"{prom:.0f}", f"{min(pesos):.0f}", f"{max(pesos):.0f}"


def escribir_csv(fotos: List[FotoMeta], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "archivo", "hora", "altura_m", "gsd_cm_px", "n_animales",
            "peso_prom_kg", "peso_min", "peso_max", "cliente",
        ])
        for f in fotos:
            prom, pmin, pmax = _stats(f.pesos_kg)
            w.writerow([
                f.path.name,
                f.hora.strftime("%H:%M:%S") if f.hora else "",
                f"{f.altura_m:.1f}" if f.altura_m is not None else "",
                f"{f.gsd_cm_px:.3f}" if f.gsd_cm_px else "",
                f.n_animales,
                prom, pmin, pmax,
                f.cliente,
            ])


def escribir_resumen(fotos: List[FotoMeta], path: Path, categoria: str,
                     solo_conteo: bool) -> None:
    grupos = agrupar_fotos(fotos)
    lineas = [
        "RESUMEN RECORRIDA DRONE - conteo y peso estimado",
        f"Generado: {datetime.now():%Y-%m-%d %H:%M}   "
        f"Categoria peso: {categoria}   Fotos procesadas: {len(fotos)}",
        "Conteo del corral = MAXIMO de animales detectados en una foto del",
        "grupo (la foto que mejor cubre el corral). Peso promedio = solo de",
        f"fotos a <= {ALTURA_PESO_CONFIABLE_M:.0f} m (escala mas confiable).",
        "=" * 70,
    ]
    cliente_actual = None
    for i, g in enumerate(grupos, 1):
        if g[0].cliente != cliente_actual:
            cliente_actual = g[0].cliente
            lineas += ["", f"CLIENTE: {cliente_actual}", "-" * 70]

        horas = [f.hora for f in g if f.hora]
        rango_h = (f"{min(horas):%H:%M}-{max(horas):%H:%M}" if horas else "s/hora")
        alturas = sorted({round(f.altura_m) for f in g if f.altura_m is not None})
        mejor = max(g, key=lambda f: f.n_animales)

        lineas.append(
            f"Grupo {i}: {g[0].nombre} a {g[-1].nombre} "
            f"({len(g)} fotos, {rango_h}, alturas {alturas} m)"
        )
        lineas.append(
            f"  Conteo maximo: {mejor.n_animales} animales (en {mejor.nombre})"
        )
        if not solo_conteo:
            pesos_conf = [
                p for f in g
                if f.altura_m is not None and f.altura_m <= ALTURA_PESO_CONFIABLE_M
                for p in f.pesos_kg
            ]
            if pesos_conf:
                prom = sum(pesos_conf) / len(pesos_conf)
                lineas.append(
                    f"  Peso promedio (fotos <=20 m, {len(pesos_conf)} animales):"
                    f" {prom:.0f} kg  [{min(pesos_conf):.0f}-{max(pesos_conf):.0f}]"
                )
            else:
                lineas.append("  Peso promedio: sin fotos <=20 m con pesos validos")
        conteos = ", ".join(f"{f.nombre}={f.n_animales}" for f in g)
        lineas.append(f"  Detalle conteos: {conteos}")
        lineas.append("")

    errores = [f for f in fotos if f.error]
    if errores:
        lineas += ["FOTOS CON ERROR:"] + [
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
    ap.add_argument("--conf", type=float, default=0.15,
                    help="Umbral de confianza YOLO (default 0.15)")
    ap.add_argument("--categoria", default="novillo",
                    choices=["ternero", "vaquillona", "novillo",
                             "vaca_adulta", "toro", "desconocido"],
                    help="Categoría para el factor de peso (default novillo)")
    ap.add_argument("--desde", default=None, metavar="DJI_0063",
                    help="Primera foto a procesar (inclusive)")
    ap.add_argument("--hasta", default=None, metavar="DJI_0078",
                    help="Última foto a procesar (inclusive)")
    ap.add_argument("--modelo", default="yolov8m.pt",
                    help="Modelo YOLO (default yolov8m.pt)")
    ap.add_argument("--imgsz", type=int, default=1920,
                    help="Tamaño de inferencia YOLO (default 1920)")
    ap.add_argument("--alt-max-anotar", type=float, default=30.5,
                    help="Anotar JPG solo para fotos hasta esta altura "
                         "(default 30 m; 0 = no anotar ninguna)")
    args = ap.parse_args()

    print("Leyendo metadata EXIF/XMP de las fotos...")
    fotos = listar_fotos(args.desde, args.hasta)
    if not fotos:
        sys.exit("No hay fotos JPG que procesar con ese filtro.")
    print(f"  {len(fotos)} fotos encontradas en {DIR_FOTOS}")

    print(f"Cargando YOLO ({args.modelo}, conf={args.conf}, imgsz={args.imgsz})...")
    detector = crear_detector(args.modelo, args.conf, args.imgsz)
    weight_model = None if args.solo_conteo else cargar_weight_model()

    for i, meta in enumerate(fotos, 1):
        alt = f"{meta.altura_m:.0f}m" if meta.altura_m is not None else "alt?"
        gsd = f"{meta.gsd_cm_px:.2f}cm/px" if meta.gsd_cm_px else "gsd?"
        print(f"[{i}/{len(fotos)}] {meta.path.name} ({alt}, {gsd}, "
              f"{meta.cliente})...", end=" ", flush=True)
        if meta.error:
            print(f"SALTEADA: {meta.error}")
            continue
        anotar = (args.alt_max_anotar > 0 and meta.altura_m is not None
                  and meta.altura_m <= args.alt_max_anotar)
        try:
            procesar_foto(meta, detector, weight_model, args.categoria,
                          args.solo_conteo, anotar, DIR_ANOTADAS)
        except Exception as e:
            meta.error = f"Error en detección: {e}"
            print(f"ERROR: {e}")
            continue
        prom, _, _ = _stats(meta.pesos_kg)
        extra = f", peso prom {prom} kg" if prom else ""
        print(f"{meta.n_animales} animales{extra}")

    csv_path = DIR_RESULTADOS / "resultados_fotos.csv"
    resumen_path = DIR_RESULTADOS / "resumen.txt"
    escribir_csv(fotos, csv_path)
    escribir_resumen(fotos, resumen_path, args.categoria, args.solo_conteo)

    print()
    print("=" * 60)
    print("LISTO. Resultados en:")
    print(f"  CSV por foto:    {csv_path}")
    print(f"  Resumen grupos:  {resumen_path}")
    if args.alt_max_anotar > 0:
        print(f"  Fotos anotadas:  {DIR_ANOTADAS}/")
    print("Contrastar el 'Conteo maximo' y 'Peso promedio' de resumen.txt")
    print("con los datos reales de cada corral.")


if __name__ == "__main__":
    main()
