# -*- coding: utf-8 -*-
"""
Prepara el dataset de fine-tuning de bovinos en vista cenital (drone).

Corre en la Mac (venv con ultralytics). Hace TODO el trabajo pesado local:

  1. SELECCIÓN de imágenes:
       - Todas las fotos DJI de <= 20 m (animales grandes y nítidos; la
         altura sale del XMP RelativeAltitude vía src/recorrida.py).
       - Frames de los MP4: 1 frame cada 3 segundos, y SOLO si el detector
         encuentra >= 1 animal en el frame (para no juntar pasto vacío).
         Límite configurable (default 150 frames).

  2. PRE-ETIQUETADO con yolov8l-seg.pt (conf 0.20: precisión alta, sin
     inventos). Escribe labels YOLO detect (clase 0 "bovino",
     cx cy w h normalizados). El pre-etiquetado cubre ~60-70% de los
     animales: un revisor humano/IA completa lo que falte.

  3. REVISIÓN: guarda una copia visual de cada imagen con las cajas
     dibujadas y numeradas en dataset_bovinos/revision/, para que el
     revisor detecte de un vistazo los animales SIN caja.

  4. SPLIT 85/15 train/val (aleatorio, seed fija) y estructura YOLO:

        dataset_bovinos/
          dataset.yaml
          images/train  images/val
          labels/train  labels/val
          revision/               (no va al zip)

  5. Crea dataset_bovinos.zip listo para subir al notebook de Colab
     (notebooks/finetune_hms.ipynb).

Uso:
    python scripts/preparar_dataset.py
    python scripts/preparar_dataset.py --origen videos_drone/tarjeta_drone \\
        --salida dataset_bovinos --max-frames 150 --conf 0.20

Los imports pesados (ultralytics, cv2) se hacen recién dentro de las
funciones que los necesitan: la selección de fotos y el armado de la
estructura se pueden testear sin GPU ni ultralytics.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import recorrida as rec  # liviano: solo PIL/re a nivel módulo

# Defaults
ORIGEN_DEFAULT = REPO_ROOT / "videos_drone" / "tarjeta_drone"
SALIDA_DEFAULT = REPO_ROOT / "dataset_bovinos"
ALTURA_MAX_M = rec.ALTURA_PESO_CONFIABLE_M   # 20.5 m: fotos "buenas"
CONF_PREETIQUETA = 0.20                      # precisión alta, sin inventos
IMGSZ_PREETIQUETA = rec.IMGSZ_DEFAULT        # 3840 (fotos 4000x2250)
CADA_SEG_DEFAULT = 3                         # 1 frame cada 3 s de video
MAX_FRAMES_DEFAULT = 150
FRAC_VAL = 0.15
SEED = 42

# detect_fn(fuente) -> [(x1, y1, x2, y2, conf), ...] en píxeles originales.
# `fuente` puede ser una ruta (str) o un ndarray BGR (frame de video).
DetectFn = Callable[[object], List[Tuple[float, float, float, float, float]]]


# ----------------------------------------------------------------------
# 1) Selección de fotos <= 20 m (sin ultralytics: solo EXIF/XMP)
# ----------------------------------------------------------------------

def seleccionar_fotos(dir_origen: Path,
                      altura_max: float = ALTURA_MAX_M) -> List:
    """Fotos DJI con altura XMP conocida y <= altura_max metros."""
    fotos = rec.listar_fotos(Path(dir_origen))
    return [f for f in fotos
            if not f.error
            and f.altura_m is not None
            and f.altura_m <= altura_max]


# ----------------------------------------------------------------------
# Detector real (import pesado adentro)
# ----------------------------------------------------------------------

def crear_detector(modelo: str = rec.MODELO_DEFAULT,
                   conf: float = CONF_PREETIQUETA,
                   imgsz: int = IMGSZ_PREETIQUETA) -> DetectFn:
    """Devuelve una detect_fn basada en YOLO (bovinos = cuadrúpedos COCO)."""
    yolo = rec.crear_yolo(modelo)

    def detect_fn(fuente):
        res = yolo.predict(fuente, conf=conf, iou=0.5, imgsz=imgsz,
                           classes=rec.CLASES_COCO_BOVINO, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for box, cf in zip(boxes, confs):
                dets.append((float(box[0]), float(box[1]),
                             float(box[2]), float(box[3]), float(cf)))
        return dets

    return detect_fn


# ----------------------------------------------------------------------
# 2) Frames de video con >= 1 animal (import cv2 adentro)
# ----------------------------------------------------------------------

def extraer_frames_videos(dir_origen: Path, detect_fn: DetectFn,
                          dir_tmp: Path,
                          cada_seg: float = CADA_SEG_DEFAULT,
                          max_frames: int = MAX_FRAMES_DEFAULT):
    """Recorre los MP4 y guarda 1 frame cada `cada_seg` segundos, solo si
    el detector encuentra >= 1 animal. Devuelve [(nombre, ruta_jpg, dets)].

    Nombre de frame: DJI_0068_t012 = video DJI_0068, segundo 12.
    """
    import cv2

    dir_tmp.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in Path(dir_origen).iterdir()
                    if p.suffix.upper() == ".MP4")
    seleccion = []
    for video in videos:
        if len(seleccion) >= max_frames:
            break
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"  Aviso: no pude abrir {video.name}, lo salteo.")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        paso = max(1, int(round(fps * cada_seg)))
        idx = 0
        usados = 0
        while len(seleccion) < max_frames:
            ok = cap.grab()
            if not ok:
                break
            if idx % paso == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    dets = detect_fn(frame)
                    if dets:
                        seg = int(round(idx / fps))
                        nombre = f"{video.stem}_t{seg:03d}"
                        ruta = dir_tmp / f"{nombre}.jpg"
                        cv2.imwrite(str(ruta), frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        seleccion.append((nombre, ruta, dets))
                        usados += 1
            idx += 1
        cap.release()
        print(f"  {video.name}: {usados} frames con animales")
    return seleccion


# ----------------------------------------------------------------------
# 3) Labels YOLO + copia visual de revisión (PIL: corre sin ultralytics)
# ----------------------------------------------------------------------

def escribir_label(ruta_txt: Path, dets, ancho: int, alto: int) -> int:
    """Escribe el .txt YOLO detect (clase 0, cx cy w h normalizados)."""
    lineas = []
    for (x1, y1, x2, y2, _cf) in dets:
        x1 = min(max(x1, 0.0), ancho)
        x2 = min(max(x2, 0.0), ancho)
        y1 = min(max(y1, 0.0), alto)
        y2 = min(max(y2, 0.0), alto)
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            continue
        cx = (x1 + x2) / 2.0 / ancho
        cy = (y1 + y2) / 2.0 / alto
        lineas.append(f"0 {cx:.6f} {cy:.6f} {w / ancho:.6f} {h / alto:.6f}")
    ruta_txt.write_text("\n".join(lineas) + ("\n" if lineas else ""),
                        encoding="utf-8")
    return len(lineas)


def guardar_revision(ruta_img: Path, dets, ruta_out: Path) -> None:
    """Copia visual con cajas rojas numeradas, para revisar qué faltó."""
    from PIL import Image, ImageDraw

    with Image.open(ruta_img) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        grosor = max(2, im.width // 800)
        for i, (x1, y1, x2, y2, cf) in enumerate(dets, 1):
            draw.rectangle([x1, y1, x2, y2], outline=(255, 40, 40),
                           width=grosor)
            texto = f"{i} ({cf:.2f})"
            tx, ty = x1, max(0, y1 - 14 * grosor // 2)
            draw.rectangle([tx, ty, tx + 8 * len(texto) * grosor // 2,
                            ty + 12 * grosor // 2], fill=(255, 40, 40))
            draw.text((tx + 2, ty), texto, fill=(255, 255, 255))
        im.save(ruta_out, "JPEG", quality=90)


# ----------------------------------------------------------------------
# 4) Split + estructura final + yaml + zip
# ----------------------------------------------------------------------

def hacer_split(nombres: List[str], frac_val: float = FRAC_VAL,
                seed: int = SEED):
    """85/15 train/val por imagen, aleatorio con seed fija."""
    nombres = sorted(nombres)
    rng = random.Random(seed)
    rng.shuffle(nombres)
    n_val = max(1, round(len(nombres) * frac_val)) if len(nombres) > 1 else 0
    val = set(nombres[:n_val])
    return [n for n in nombres if n not in val], sorted(val)


def escribir_yaml(dataset_dir: Path) -> Path:
    """dataset.yaml para ultralytics. La ruta `path` es local: el notebook
    de Colab la reescribe a /content/dataset_bovinos al descomprimir."""
    ruta = dataset_dir / "dataset.yaml"
    ruta.write_text(
        "# Dataset bovinos vista cenital (drone) — generado por\n"
        "# scripts/preparar_dataset.py. Pre-etiquetado con yolov8l-seg\n"
        "# (COCO) y revisado a mano.\n"
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: bovino\n",
        encoding="utf-8",
    )
    return ruta


def armar_dataset(items, dataset_dir: Path, frac_val: float = FRAC_VAL,
                  seed: int = SEED) -> dict:
    """`items` = [(nombre, ruta_jpg_origen, dets)]. Copia imágenes y labels
    a la estructura YOLO final + revisión. Devuelve el resumen."""
    from PIL import Image

    for sub in ("images/train", "images/val", "labels/train", "labels/val",
                "revision"):
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    train, val = hacer_split([n for n, _, _ in items], frac_val, seed)
    en_val = set(val)

    n_inst = 0
    for nombre, ruta, dets in items:
        split = "val" if nombre in en_val else "train"
        destino = dataset_dir / "images" / split / f"{nombre}.jpg"
        shutil.copy2(ruta, destino)
        with Image.open(destino) as im:
            ancho, alto = im.size
        n_inst += escribir_label(
            dataset_dir / "labels" / split / f"{nombre}.txt",
            dets, ancho, alto)
        guardar_revision(destino, dets,
                         dataset_dir / "revision" / f"{nombre}.jpg")

    escribir_yaml(dataset_dir)
    return {"n_imagenes": len(items), "n_train": len(train),
            "n_val": len(val), "n_instancias": n_inst,
            "promedio": n_inst / len(items) if items else 0.0}


def crear_zip(dataset_dir: Path) -> Path:
    """dataset_bovinos.zip listo para Colab (sin revision/: es solo para
    el revisor humano y duplicaría el peso del zip)."""
    ruta_zip = dataset_dir.parent / f"{dataset_dir.name}.zip"
    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dataset_dir.rglob("*")):
            if p.is_dir() or "revision" in p.parts[len(dataset_dir.parts):]:
                continue
            z.write(p, p.relative_to(dataset_dir.parent))
    return ruta_zip


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None,
         detect_fn: Optional[DetectFn] = None) -> dict:
    ap = argparse.ArgumentParser(
        description="Prepara dataset_bovinos (imágenes + pre-etiquetas "
                    "YOLO + revisión + zip) para fine-tuning en Colab.")
    ap.add_argument("--origen", type=Path, default=ORIGEN_DEFAULT,
                    help="Carpeta con JPG y MP4 del drone")
    ap.add_argument("--salida", type=Path, default=SALIDA_DEFAULT,
                    help="Carpeta de salida del dataset")
    ap.add_argument("--altura-max", type=float, default=ALTURA_MAX_M,
                    help="Altura máxima (m) de las fotos a incluir")
    ap.add_argument("--conf", type=float, default=CONF_PREETIQUETA,
                    help="Confianza del pre-etiquetado (0.20 = preciso)")
    ap.add_argument("--imgsz", type=int, default=IMGSZ_PREETIQUETA)
    ap.add_argument("--modelo", default=rec.MODELO_DEFAULT)
    ap.add_argument("--cada-seg", type=float, default=CADA_SEG_DEFAULT,
                    help="Segundos entre frames de video")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES_DEFAULT,
                    help="Máximo de frames de video a incluir")
    ap.add_argument("--sin-videos", action="store_true",
                    help="Solo fotos (más rápido)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    if args.salida.exists():
        print(f"Borrando dataset anterior: {args.salida}")
        shutil.rmtree(args.salida)

    # 1) Fotos <= altura_max
    fotos = seleccionar_fotos(args.origen, args.altura_max)
    print(f"Fotos <= {args.altura_max:g} m: {len(fotos)} "
          f"(de {args.origen})")

    if detect_fn is None:
        print(f"Cargando {args.modelo} (pre-etiquetado, conf {args.conf})…")
        detect_fn = crear_detector(args.modelo, args.conf, args.imgsz)

    items = []
    for i, f in enumerate(fotos, 1):
        dets = detect_fn(str(f.path))
        print(f"  [{i}/{len(fotos)}] {f.nombre} "
              f"({f.altura_m:.0f} m): {len(dets)} animales pre-etiquetados")
        items.append((f.nombre, f.path, dets))

    # 2) Frames de video con animales
    if not args.sin_videos:
        print(f"Extrayendo frames de video (1 cada {args.cada_seg:g} s, "
              f"máx {args.max_frames}, solo con animales)…")
        dir_tmp = args.salida / "_frames_tmp"
        frames = extraer_frames_videos(args.origen, detect_fn, dir_tmp,
                                       args.cada_seg, args.max_frames)
        items.extend(frames)
        print(f"Frames seleccionados: {len(frames)}")

    if not items:
        print("No hay imágenes para el dataset. Nada que hacer.")
        return {}

    # 3-4) Estructura final + labels + revisión + yaml
    resumen = armar_dataset(items, args.salida, seed=args.seed)
    dir_tmp = args.salida / "_frames_tmp"
    if dir_tmp.exists():
        shutil.rmtree(dir_tmp)

    # 5) Zip para Colab
    ruta_zip = crear_zip(args.salida)

    print("\n" + "=" * 60)
    print("RESUMEN DEL DATASET")
    print(f"  Imágenes:            {resumen['n_imagenes']} "
          f"(train {resumen['n_train']} / val {resumen['n_val']})")
    print(f"  Instancias (cajas):  {resumen['n_instancias']}")
    print(f"  Promedio por imagen: {resumen['promedio']:.1f}")
    print(f"  Dataset:  {args.salida}")
    print(f"  Revisión: {args.salida / 'revision'}  <- mirá acá qué "
          f"animales quedaron SIN caja y completá los labels")
    print(f"  Zip Colab: {ruta_zip}")
    print("Siguiente paso: revisar/completar labels y subir el zip a "
          "notebooks/finetune_hms.ipynb en Colab.")
    return resumen


if __name__ == "__main__":
    main()
