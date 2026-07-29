# -*- coding: utf-8 -*-
"""
Compara combinaciones de parámetros de detección sobre unas pocas fotos,
para elegir la mejor ANTES de procesar el dataset entero.

El pre-etiquetado por mosaicos tiene tres perillas:
  - conf:    umbral de confianza. Más bajo = detecta más, pero también
             más falsos positivos.
  - tile:    lado del mosaico en px. Más chico = cada animal le llega más
             grande al detector, pero más mosaicos (más lento).
  - overlap: solapamiento entre mosaicos. Más alto = menos animales
             perdidos en los cortes.

Uso:
    python scripts/probar_deteccion.py
    python scripts/probar_deteccion.py --fotos DJI_0084 DJI_0114 --salida pruebas

Deja en la carpeta de salida una imagen por combinación con las cajas
dibujadas, y una tabla con cuántos animales encontró cada una.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import recorrida as rec

# (conf, tile, overlap) — de más conservador a más agresivo
COMBINACIONES = [
    (0.20, 1024, 0.25),   # la que usamos hasta ahora
    (0.10, 1024, 0.30),
    (0.05, 1024, 0.30),
    (0.10, 768, 0.35),
    (0.05, 768, 0.35),
    (0.05, 640, 0.40),
]

FOTOS_DEFAULT = ["DJI_0084", "DJI_0114", "DJI_0129"]
ORIGEN_DEFAULT = REPO_ROOT / "videos_drone" / "tarjeta_drone"


def dibujar(img_bgr, dets, ruta_out: Path) -> None:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img_bgr[:, :, ::-1]).convert("RGB")
    d = ImageDraw.Draw(im)
    grosor = max(2, im.width // 700)
    for (x1, y1, x2, y2, _cf) in dets:
        d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=grosor)
    im.thumbnail((1800, 1800))
    im.save(ruta_out, "JPEG", quality=85)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Compara parámetros de detección sobre pocas fotos.")
    ap.add_argument("--origen", type=Path, default=ORIGEN_DEFAULT)
    ap.add_argument("--fotos", nargs="*", default=FOTOS_DEFAULT,
                    help="Nombres sin extensión, ej: DJI_0084")
    ap.add_argument("--salida", type=Path, default=REPO_ROOT / "pruebas_deteccion")
    ap.add_argument("--modelo", default=rec.MODELO_DEFAULT)
    args = ap.parse_args(argv)

    import cv2
    from src import mosaico as mos

    args.salida.mkdir(parents=True, exist_ok=True)
    yolo = rec.crear_yolo(args.modelo)

    imagenes = {}
    for nombre in args.fotos:
        ruta = args.origen / f"{nombre}.JPG"
        if not ruta.exists():
            ruta = args.origen / f"{nombre}.jpg"
        if not ruta.exists():
            print(f"  No encontré {nombre} en {args.origen}, la salteo.")
            continue
        img = cv2.imread(str(ruta))
        if img is not None:
            imagenes[nombre] = img

    if not imagenes:
        print("No hay fotos para probar.")
        return

    print(f"Probando {len(COMBINACIONES)} combinaciones sobre "
          f"{len(imagenes)} fotos con {args.modelo}…\n")

    filas = []
    for (conf, tile, overlap) in COMBINACIONES:
        def _predict(recorte, _conf=conf, _tile=tile):
            res = yolo.predict(recorte, conf=_conf, iou=0.5, imgsz=_tile,
                               classes=rec.CLASES_COCO_BOVINO,
                               verbose=False)[0]
            out = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for b, cf in zip(boxes, confs):
                    out.append((float(b[0]), float(b[1]), float(b[2]),
                                float(b[3]), float(cf)))
            return out

        etiqueta = f"conf{conf}_tile{tile}_ov{overlap}"
        conteos = {}
        for nombre, img in imagenes.items():
            dets = mos.detectar_por_mosaicos(_predict, img, tile=tile,
                                             overlap=overlap)
            conteos[nombre] = len(dets)
            dibujar(img, dets, args.salida / f"{nombre}__{etiqueta}.jpg")
        filas.append((etiqueta, conteos))
        print(f"  {etiqueta}: " +
              "  ".join(f"{n}={c}" for n, c in conteos.items()))

    print("\n" + "=" * 62)
    print("TABLA COMPARATIVA (animales detectados)")
    nombres = list(imagenes)
    print(f"{'combinacion':<28}" + "".join(f"{n:>12}" for n in nombres))
    for etiqueta, conteos in filas:
        print(f"{etiqueta:<28}" +
              "".join(f"{conteos[n]:>12}" for n in nombres))
    print(f"\nImágenes en: {args.salida}")
    print("Mirá las imágenes: la mejor combinación es la que marca casi "
          "todos los animales SIN inventar cajas sobre el suelo.")


if __name__ == "__main__":
    main()
