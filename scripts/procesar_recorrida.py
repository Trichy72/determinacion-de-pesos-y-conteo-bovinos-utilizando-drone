#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procesamiento batch de la recorrida con drone del 24/7/2026.  (v3: calibración)

Recorre las fotos DJI de videos_drone/tarjeta_drone/, y para cada una:
  1. Lee altura relativa (XMP RelativeAltitude) y datos de cámara (EXIF)
     para calcular la escala GSD (cm/píxel) sin necesidad del cuadrado
     de referencia en el piso.
  2. Detecta bovinos con YOLO de SEGMENTACIÓN (default yolov8l-seg.pt,
     clases COCO de cuadrúpedos grandes, conf baja 0.10, imgsz 3840 para
     usar casi toda la resolución 4000x2250 — clave para fotos de 50/100 m
     donde un animal mide ~50 px).
  3. Estima peso por animal con el ÁREA DE LA MÁSCARA de segmentación
     (píxeles de silueta x GSD²), que EXCLUYE LA SOMBRA del animal — la
     primera corrida (sol bajo de invierno) inflaba pesos porque el bbox
     abarcaba animal + sombra. Reusa src/weight_estimator.py (modelo
     alométrico a=220, b=1.20, factor por categoría). Si el modelo elegido
     NO es de segmentación (--modelo yolov8m.pt), cae al método anterior:
     área bbox x 0.69.

Cambios v2 vs v1:
  - Peso por máscara (sin sombra) en vez de bbox x 0.69.
  - imgsz default 3840 (antes 1920) y conf default 0.10 (antes 0.15):
    mejora el recall de animales negros y de fotos altas. Se eliminó el
    modo tiles: a resolución casi nativa el modelo -seg grande encuentra
    los animales chicos directo en full-frame.
  - Filtro anti-sombra: detecciones cuya área de silueta cae fuera del
    rango válido [area_min_m2, area_max_m2] del WeightModel se DESCARTAN
    (típico: máscara que fusionó animal+sombra o dos animales pegados).
  - Anotadas: se dibuja el CONTORNO de la máscara (no solo bbox) para
    verificar visualmente que la sombra quedó excluida, y se anotan TODAS
    las fotos con detecciones (antes solo <=30 m).
  - CSV: columnas nuevas `metodo` (mask|bbox) y `modelo`.

Cambios v3 vs v2:
  - FILTRO DE BORDE: detecciones cuya bbox toca el borde de la imagen
    (margen 10 px) cuentan para el CONTEO pero NO para el peso — en la
    corrida v2 los animales cortados por el encuadre (media máscara)
    generaban outliers de 98-113 kg que bajaban el promedio. Columna
    nueva `n_completos` en el CSV.
  - MEDIA RECORTADA: el peso promedio del corral descarta el 10% inferior
    y superior de los pesos individuales (animales completos del grupo).
  - GRUPOS_REALES: mapeo hardcodeado rango de archivo -> corral con los
    datos de verdad de campo (conteo y rango de peso real). El resumen
    compara estimado vs real por corral.
  - CALIBRACIÓN AUTOMÁTICA: con los corrales que tienen peso real se
    calcula f = promedio ponderado de (peso_real_medio / peso_estimado)
    y se propone a_calibrado = a x f. Se imprime tabla ANTES/DESPUÉS y
    se guarda videos_drone/resultados/calibracion.json. config.yaml NO
    se modifica (eso se decide después con el usuario).

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
import csv
import json
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

# Umbrales
ALTURA_PESO_CONFIABLE_M = 20.5 # fotos ≤20 m: las más confiables para peso
GAP_GRUPO_SEG = 240            # >4 min entre fotos => nuevo grupo/corral
MARGEN_BORDE_PX = 10           # bbox a <=10 px del borde => animal cortado
FRAC_RECORTE = 0.10            # media recortada: descarta 10% inf y sup

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

# COCO ids de cuadrúpedos grandes con los que YOLO confunde bovinos en
# vista cenital (mismo set que src/detector.py):
# 17 dog, 18 horse, 19 cow, 20 sheep, 21 elephant, 22 bear, 23 zebra
CLASES_COCO_BOVINO = [17, 18, 19, 20, 21, 22, 23]

# Factor bbox->silueta calibrado empíricamente (solo para modelos SIN
# segmentación; con máscara la silueta es directa y no hace falta).
FACTOR_BBOX_SILUETA = 0.69


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
    n_completos: int = 0             # sin tocar el borde: los usados para peso
    pesos_kg: List[float] = field(default_factory=list)
    metodo: str = ""                 # "mask" | "bbox"
    n_descartadas: int = 0           # filtro anti-sombra (área fuera de rango)
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


def crear_yolo(modelo: str):
    """Carga el modelo YOLO (de la raíz del repo si existe ahí)."""
    from ultralytics import YOLO

    model_path = str(REPO_ROOT / modelo) if (REPO_ROOT / modelo).exists() else modelo
    return YOLO(model_path)


def procesar_foto(meta, yolo, weight_model, categoria, solo_conteo,
                  anotar, out_anotadas, conf, imgsz):
    """Detecta, filtra, cuenta, estima pesos y (opcional) anota la foto.

    Peso por MÁSCARA de segmentación (excluye la sombra):

      * `res.masks.data` de ultralytics es un tensor N x H_inf x W_inf en la
        RESOLUCIÓN DE INFERENCIA (imagen reescalada + letterbox a múltiplo
        de 32; ej. 4000x2250 con imgsz=3840 -> máscaras de 2176x3840), NO en
        la resolución original. Por eso el área en píxeles hay que escalarla
        al tamaño original:

            r = min(W_mask / W_orig, H_mask / H_orig)   # escala letterbox
            area_px_orig = mask.sum() * (1 / r)**2

        Con fotos apaisadas el letterbox rellena arriba/abajo, así que
        r = W_mask / W_orig y la fórmula equivale a
        area_px_orig = mask.sum() * (W_orig / W_mask)**2.
        (Se suma sobre el tensor sin resize a 4000x2250: mismo resultado,
        mucha menos memoria.)

      * Los contornos para anotar salen de `res.masks.xy`, que ultralytics
        YA devuelve en coordenadas de la imagen original (usa scale_coords
        internamente), igual que `res.boxes.xyxy`.

    Fallback sin segmentación (modelo detección pura): área bbox x 0.69.

    Filtro anti-sombra: si el área de silueta (máscara o bbox corregido)
    cae fuera de [area_min_m2, area_max_m2] del WeightModel, la detección
    se DESCARTA por completo (conteo y anotado): casi siempre es una
    máscara que fusionó animal+sombra o dos animales, o un falso positivo.
    """
    import cv2

    img = cv2.imread(str(meta.path))
    if img is None:
        meta.error = "No pude abrir la imagen con OpenCV"
        return

    h_orig, w_orig = img.shape[:2]
    res = yolo.predict(
        img,
        conf=conf,
        iou=0.5,
        imgsz=imgsz,
        classes=CLASES_COCO_BOVINO,
        verbose=False,
    )[0]

    # --- parsear detecciones: bbox + (opcional) máscara -----------------
    dets = []  # dicts: bbox, conf, area_px (silueta en px ORIGINALES), poly
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()   # ya en coords originales
        confs = res.boxes.conf.cpu().numpy()
        masks = getattr(res, "masks", None)
        polys = masks.xy if masks is not None else None  # coords originales
        if masks is not None:
            h_mask, w_mask = masks.data.shape[-2], masks.data.shape[-1]
            # Escala letterbox original->inferencia (ver docstring)
            r = min(w_mask / float(w_orig), h_mask / float(h_orig))
            factor_area = (1.0 / r) ** 2
        for i, (box, cf) in enumerate(zip(boxes, confs)):
            if masks is not None and i < len(masks.data):
                area_px = float(masks.data[i].sum()) * factor_area
                metodo = "mask"
                poly = polys[i] if polys is not None and i < len(polys) else None
            else:
                area_px = float((box[2] - box[0]) * (box[3] - box[1])) \
                    * FACTOR_BBOX_SILUETA
                metodo = "bbox"
                poly = None
            dets.append({"bbox": box, "conf": float(cf), "area_px": area_px,
                         "poly": poly, "metodo": metodo})

    if dets:
        meta.metodo = dets[0]["metodo"]  # si no hay dets queda el default
        # que setea main() según el tipo de modelo (mask|bbox)

    # --- filtro de borde (v3): animales cortados por el encuadre --------
    # Cuentan para n_animales pero NO para el peso: en la corrida v2 los
    # animales con media máscara daban outliers de 98-113 kg. Tampoco se
    # les aplica el filtro anti-sombra de área (un animal cortado ES chico
    # y lo queremos igual en el conteo).
    for det in dets:
        x1, y1, x2, y2 = (float(v) for v in det["bbox"][:4])
        det["borde"] = (
            x1 <= MARGEN_BORDE_PX or y1 <= MARGEN_BORDE_PX
            or x2 >= w_orig - MARGEN_BORDE_PX
            or y2 >= h_orig - MARGEN_BORDE_PX
        )

    # --- área -> m², filtro anti-sombra y peso --------------------------
    m_per_px = (meta.gsd_cm_px / 100.0) if meta.gsd_cm_px else None
    validas = []
    for det in dets:
        det["peso"] = None
        if m_per_px:
            area_m2 = det["area_px"] * (m_per_px ** 2)
            det["area_m2"] = area_m2
            if not det["borde"] and weight_model is not None and (
                area_m2 < weight_model.area_min_m2
                or area_m2 > weight_model.area_max_m2
            ):
                meta.n_descartadas += 1
                continue  # descartada: sombra fusionada / doble / FP
            if (not det["borde"] and not solo_conteo
                    and weight_model is not None):
                det["peso"] = weight_model.estimate(area_m2, categoria=categoria)
        validas.append(det)

    meta.n_animales = len(validas)
    meta.n_completos = sum(1 for d in validas if not d.get("borde"))
    meta.pesos_kg = [d["peso"] for d in validas if d["peso"] is not None]

    if anotar and validas:
        _guardar_anotada(img, meta, validas, out_anotadas)


def _guardar_anotada(img, meta, dets, out_dir):
    """Anota contorno de máscara (verde) o bbox (si no hay máscara) + peso."""
    import cv2
    import numpy as np

    verde = (0, 255, 0)
    for det in dets:
        x1, y1 = int(det["bbox"][0]), int(det["bbox"][1])
        poly = det.get("poly")
        if poly is not None and len(poly) >= 3:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, verde, 3)
        else:
            x2, y2 = int(det["bbox"][2]), int(det["bbox"][3])
            cv2.rectangle(img, (x1, y1), (x2, y2), verde, 3)
        peso = det.get("peso")
        if det.get("borde"):
            etiqueta = "borde"
        else:
            etiqueta = f"{peso:.0f} kg" if peso else f"{det['conf']:.2f}"
        cv2.putText(img, etiqueta, (x1, max(30, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(img, etiqueta, (x1, max(30, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    texto = f"{meta.nombre}  |  {meta.n_animales} animales"
    if meta.altura_m is not None:
        texto += f"  |  {meta.altura_m:.0f} m"
    if meta.n_descartadas:
        texto += f"  |  {meta.n_descartadas} descartadas"
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


def media_recortada(pesos: List[float], frac: float = FRAC_RECORTE) -> Optional[float]:
    """Media recortada: descarta la fracción `frac` inferior Y superior.

    Con pocas muestras (si el recorte dejaría la lista vacía) devuelve la
    media simple.
    """
    if not pesos:
        return None
    s = sorted(pesos)
    k = int(len(s) * frac)
    if len(s) - 2 * k >= 1:
        s = s[k:len(s) - k]
    return sum(s) / len(s)


def escribir_csv(fotos: List[FotoMeta], path: Path, modelo: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "archivo", "hora", "altura_m", "gsd_cm_px", "n_animales",
            "n_completos", "peso_prom_kg", "peso_min", "peso_max", "cliente",
            "metodo", "modelo",
        ])
        for f in fotos:
            prom, pmin, pmax = _stats(f.pesos_kg)
            w.writerow([
                f.path.name,
                f.hora.strftime("%H:%M:%S") if f.hora else "",
                f"{f.altura_m:.1f}" if f.altura_m is not None else "",
                f"{f.gsd_cm_px:.3f}" if f.gsd_cm_px else "",
                f.n_animales,
                f.n_completos,
                prom, pmin, pmax,
                f.cliente,
                f.metodo,
                modelo,
            ])


# ----------------------------------------------------------------------
# Corrales reales: estadísticas por corral y calibración automática
# ----------------------------------------------------------------------

def stats_corrales(fotos: List[FotoMeta]) -> List[dict]:
    """Agrupa las fotos según GRUPOS_REALES y calcula por corral:
    conteo máximo por foto, n animales pesados (completos), peso promedio
    estimado (media recortada 10%) y desvío % vs punto medio del rango real.

    Para el peso usa primero solo fotos <= ALTURA_PESO_CONFIABLE_M; si el
    corral no tiene ninguna (ej. Corral 6, fotografiado a 50 m), cae a los
    pesos de todas las alturas y lo marca con `alturas_altas`.
    """
    out = []
    for g in GRUPOS_REALES:
        fg = [f for f in fotos if g["desde"] <= f.numero <= g["hasta"]]
        if not fg:
            continue
        c = dict(g)
        c["fotos"] = fg
        mejor = max(fg, key=lambda f: f.n_animales)
        c["conteo_max"] = mejor.n_animales
        c["foto_max"] = mejor.nombre

        pesos = [p for f in fg
                 if f.altura_m is not None
                 and f.altura_m <= ALTURA_PESO_CONFIABLE_M
                 for p in f.pesos_kg]
        # Regla calibrada 27/7/26: el peso SOLO se mide con fotos <=20 m.
        # Con fotos altas (50-100 m) la silueta sale gruesa y sesgada
        # (corral 6 dio +22% vs balanza): mejor no informar peso.
        c["alturas_altas"] = (not pesos) and any(f.pesos_kg for f in fg)
        c["n_pesados"] = len(pesos)
        c["peso_est"] = media_recortada(pesos)

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

    es_seg = "seg" in Path(args.modelo).stem.lower()
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
