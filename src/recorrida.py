# -*- coding: utf-8 -*-
"""
Pipeline CALIBRADO de la recorrida con drone — módulo compartido.

Única fuente de verdad para el procesamiento de fotos DJI:
  - EXIF/XMP (RelativeAltitude) -> altura -> GSD (cm/píxel)
  - Detección YOLO de segmentación + peso por área de MÁSCARA (excluye
    la sombra), con fallback bbox x 0.69 para modelos sin segmentación
  - Filtro anti-sombra por área y FILTRO DE BORDE (animales cortados
    por el encuadre cuentan para el conteo pero NO para el peso)
  - Peso confiable SOLO con fotos <= 20 m (regla calibrada 27/7/26
    contra balanza de La Esperanza; coef_a=250.4 en config.yaml)
  - Agrupación de fotos contiguas en "corrales" y MEDIA RECORTADA 10%

Lo usan:
  - scripts/procesar_recorrida.py (CLI batch de la recorrida)
  - drone_app.py (app local Streamlit)

Los imports pesados (ultralytics, cv2, numpy, yaml) se hacen recién
dentro de las funciones que los necesitan, para que el parsing EXIF/GSD
y la agrupación se puedan usar/testear sin GPU ni ultralytics.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

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

# Umbrales calibrados
ALTURA_PESO_CONFIABLE_M = 20.5 # fotos ≤20 m: las más confiables para peso
GAP_GRUPO_SEG = 240            # >4 min entre fotos => nuevo grupo/corral
MARGEN_BORDE_PX = 10           # bbox a <=10 px del borde => animal cortado
FRAC_RECORTE = 0.10            # media recortada: descarta 10% inf y sup

# COCO ids de cuadrúpedos grandes con los que YOLO confunde bovinos en
# vista cenital (mismo set que src/detector.py):
# 17 dog, 18 horse, 19 cow, 20 sheep, 21 elephant, 22 bear, 23 zebra
CLASES_COCO_BOVINO = [17, 18, 19, 20, 21, 22, 23]

# Factor bbox->silueta calibrado empíricamente (solo para modelos SIN
# segmentación; con máscara la silueta es directa y no hace falta).
FACTOR_BBOX_SILUETA = 0.69

# Defaults de inferencia (v3 calibrada)
CONF_DEFAULT = 0.10
IMGSZ_DEFAULT = 3840
MODELO_DEFAULT = "yolov8l-seg.pt"

CATEGORIAS = ["ternero", "vaquillona", "novillo", "vaca_adulta", "toro",
              "desconocido"]


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


def leer_metadata_foto(path: Path,
                       cliente_fn: Optional[Callable[[int], str]] = None
                       ) -> FotoMeta:
    """Lee EXIF + XMP de un JPG DJI y arma el FotoMeta (altura, hora, GSD).

    `cliente_fn` opcional: numero de foto -> nombre de cliente (lo usa el
    script de la recorrida para el mapeo Roxdan / La Esperanza).
    """
    m = re.search(r"(\d+)", path.stem)
    numero = int(m.group(1)) if m else -1
    meta = FotoMeta(path=path, nombre=path.stem, numero=numero,
                    cliente=cliente_fn(numero) if cliente_fn else "")
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


def listar_fotos(dir_fotos: Path,
                 desde: Optional[str] = None,
                 hasta: Optional[str] = None,
                 cliente_fn: Optional[Callable[[int], str]] = None,
                 ) -> List[FotoMeta]:
    """Lista los JPG de una carpeta (con filtro opcional desde/hasta por
    número de archivo) y lee la metadata de cada uno.

    Lanza FileNotFoundError si la carpeta no existe.
    """
    dir_fotos = Path(dir_fotos)
    if not dir_fotos.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de fotos: {dir_fotos}"
        )

    def num_de(nombre: str) -> int:
        m = re.search(r"(\d+)", nombre)
        return int(m.group(1)) if m else -1

    n_desde = num_de(desde) if desde else None
    n_hasta = num_de(hasta) if hasta else None

    fotos = []
    for p in sorted(dir_fotos.iterdir()):
        if p.suffix.upper() not in (".JPG", ".JPEG"):
            continue
        n = num_de(p.stem)
        if n_desde is not None and n < n_desde:
            continue
        if n_hasta is not None and n > n_hasta:
            continue
        fotos.append(leer_metadata_foto(p, cliente_fn))
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
# Estadística: media recortada y resumen por grupo/corral
# ----------------------------------------------------------------------

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


def stats_pesos(pesos: List[float]):
    """(prom, min, max) formateados como strings; vacíos si no hay pesos."""
    if not pesos:
        return "", "", ""
    prom = sum(pesos) / len(pesos)
    return f"{prom:.0f}", f"{min(pesos):.0f}", f"{max(pesos):.0f}"


def resumen_grupo(fotos: List[FotoMeta]) -> dict:
    """Resumen calibrado de un grupo de fotos (un corral/vuelo):

      - conteo_max / foto_max: conteo del corral = MÁXIMO por foto.
      - pesos: pesos individuales de animales COMPLETOS en fotos <= 20 m
        (regla calibrada 27/7/26: con fotos altas la silueta sale gruesa
        y sesgada, mejor no informar peso).
      - peso_est: media recortada 10% de esos pesos.
      - alturas_altas: True si hay pesos pero todos de fotos > 20 m
        (=> no se informa peso confiable).
    """
    mejor = max(fotos, key=lambda f: f.n_animales)
    pesos = [p for f in fotos
             if f.altura_m is not None
             and f.altura_m <= ALTURA_PESO_CONFIABLE_M
             for p in f.pesos_kg]
    alturas_altas = (not pesos) and any(f.pesos_kg for f in fotos)
    horas = [f.hora for f in fotos if f.hora]
    return {
        "conteo_max": mejor.n_animales,
        "foto_max": mejor.nombre,
        "n_pesados": len(pesos),
        "pesos": pesos,
        "peso_est": media_recortada(pesos),
        "alturas_altas": alturas_altas,
        "hora_ini": min(horas) if horas else None,
        "hora_fin": max(horas) if horas else None,
    }


# ----------------------------------------------------------------------
# Modelos (imports pesados adentro: corre sin ultralytics mientras no
# se llame)
# ----------------------------------------------------------------------

def es_modelo_seg(modelo: str) -> bool:
    return "seg" in Path(modelo).stem.lower()


def cargar_weight_model(repo_root: Path = REPO_ROOT):
    """Reusa el estimador del pipeline con los coeficientes de config.yaml."""
    import sys
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.weight_estimator import WeightModel

    cfg_path = repo_root / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            return WeightModel.from_config(cfg)
        except Exception as e:
            print(f"  Aviso: no pude leer config.yaml ({e}), uso defaults.")
    return WeightModel()


def crear_yolo(modelo: str, repo_root: Path = REPO_ROOT):
    """Carga el modelo YOLO (de la raíz del repo si existe ahí)."""
    from ultralytics import YOLO

    model_path = str(repo_root / modelo) if (repo_root / modelo).exists() else modelo
    return YOLO(model_path)


# ----------------------------------------------------------------------
# Detección + peso por foto
# ----------------------------------------------------------------------

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

    Filtro de borde: detecciones cuya bbox toca el borde de la imagen
    (margen MARGEN_BORDE_PX) cuentan para el conteo pero NO para el peso
    (media máscara => outliers de 98-113 kg en la corrida v2).
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
        # que setea el caller según el tipo de modelo (mask|bbox)

    # --- filtro de borde: animales cortados por el encuadre --------------
    # Cuentan para n_animales pero NO para el peso. Tampoco se les aplica
    # el filtro anti-sombra de área (un animal cortado ES chico y lo
    # queremos igual en el conteo).
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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{meta.nombre}_anotada.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------------
# Salidas
# ----------------------------------------------------------------------

def filas_csv(fotos: List[FotoMeta], modelo: str) -> List[List]:
    """Filas del CSV por foto (header incluido), compartidas por el CLI y
    la descarga de la app."""
    filas = [[
        "archivo", "hora", "altura_m", "gsd_cm_px", "n_animales",
        "n_completos", "peso_prom_kg", "peso_min", "peso_max", "cliente",
        "metodo", "modelo",
    ]]
    for f in fotos:
        prom, pmin, pmax = stats_pesos(f.pesos_kg)
        filas.append([
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
    return filas


def escribir_csv(fotos: List[FotoMeta], path: Path, modelo: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(filas_csv(fotos, modelo))
