# -*- coding: utf-8 -*-
"""
Selecciona las imágenes cuyo pre-etiquetado está probablemente COMPLETO.

Idea: un dataset chico y limpio entrena mejor que uno grande con animales
sin marcar (los no marcados le enseñan al modelo "acá no hay vaca").

El detector genérico cubre bien las escenas fáciles (animales separados,
buen contraste) y mal las manadas apretadas. Este script puntúa cada
imagen para separar unas de otras, usando dos señales que no necesitan
ojo humano:

  1. CONFIANZA MEDIA. Si el detector está seguro de lo que marcó
     (confianzas altas), es que la escena le resulta clara. Cuando
     empieza a devolver muchas detecciones de confianza 0.05-0.10, está
     dudando, y donde duda también se le escapan animales.

  2. SOLAPAMIENTO. Cajas muy superpuestas = animales pegados = escena
     donde seguro perdió individuos.

Salida: un ranking + las imágenes con las cajas dibujadas en
`candidatas/`, ordenadas por puntaje (01_, 02_, …), para revisarlas
visualmente y quedarse solo con las verificadas.

Uso:
    python scripts/seleccionar_limpias.py
    python scripts/seleccionar_limpias.py --dataset dataset_v2 --top 40
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONF_DETECCION = 0.05      # detectamos todo y después juzgamos por confianza
CONF_SEGURA = 0.25         # umbral de "el detector está cómodo"
TILE = 1024
OVERLAP = 0.30
TOP_DEFAULT = 40
MIN_ANIMALES = 2           # menos de 2 no aporta nada al entrenamiento


def iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def puntuar(dets) -> dict:
    """Puntaje 0-1: alto = pre-etiquetado probablemente completo."""
    n = len(dets)
    if n == 0:
        return {"n": 0, "n_segura": 0, "frac_segura": 0.0,
                "solape": 0.0, "puntaje": 0.0}

    n_segura = sum(1 for d in dets if d[4] >= CONF_SEGURA)
    frac_segura = n_segura / n

    # Fracción de cajas que se solapan con otra (animales pegados)
    con_solape = 0
    for i, a in enumerate(dets):
        if any(iou(a, b) > 0.15 for j, b in enumerate(dets) if i != j):
            con_solape += 1
    solape = con_solape / n

    # Escenas con muchísimos animales son las que peor cubre
    penal_densidad = min(1.0, n / 40.0)

    puntaje = frac_segura * (1.0 - 0.6 * solape) * (1.0 - 0.5 * penal_densidad)
    return {"n": n, "n_segura": n_segura, "frac_segura": frac_segura,
            "solape": solape, "puntaje": puntaje}


def dibujar(img_bgr, dets, ruta_out: Path) -> None:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img_bgr[:, :, ::-1]).convert("RGB")
    d = ImageDraw.Draw(im)
    grosor = max(2, im.width // 700)
    for (x1, y1, x2, y2, cf) in dets:
        color = (255, 40, 40) if cf >= CONF_SEGURA else (255, 190, 0)
        d.rectangle([x1, y1, x2, y2], outline=color, width=grosor)
    im.thumbnail((1600, 1600))
    im.save(ruta_out, "JPEG", quality=85)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Puntúa qué imágenes tienen el pre-etiquetado completo.")
    ap.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset_v2")
    ap.add_argument("--salida", type=Path,
                    default=REPO_ROOT / "candidatas_limpias")
    ap.add_argument("--top", type=int, default=TOP_DEFAULT)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--overlap", type=float, default=OVERLAP)
    args = ap.parse_args(argv)

    import cv2
    from scripts.preparar_dataset import crear_detector_mosaico

    if not args.dataset.exists():
        print(f"No encuentro {args.dataset}.")
        return

    imgs = []
    for split in ("train", "val"):
        imgs.extend(sorted((args.dataset / "images" / split).glob("*.jpg")))
    if not imgs:
        print("El dataset no tiene imágenes.")
        return

    print(f"Analizando {len(imgs)} imágenes (mosaicos {args.tile}px, "
          f"conf {CONF_DETECCION}). Esto tarda un rato…\n")
    detectar = crear_detector_mosaico(conf=CONF_DETECCION, tile=args.tile,
                                      overlap=args.overlap)

    filas = []
    cache = {}
    for i, p in enumerate(imgs, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        dets = detectar(img)
        m = puntuar(dets)
        m["nombre"] = p.stem
        m["ruta"] = p
        filas.append(m)
        cache[p.stem] = (img, dets)
        print(f"  [{i}/{len(imgs)}] {p.stem}: {m['n']} animales, "
              f"{m['frac_segura']:.0%} con confianza alta, "
              f"solape {m['solape']:.0%} -> puntaje {m['puntaje']:.2f}")

    utiles = [f for f in filas if f["n"] >= MIN_ANIMALES]
    utiles.sort(key=lambda f: -f["puntaje"])
    elegidas = utiles[:args.top]

    if args.salida.exists():
        shutil.rmtree(args.salida)
    args.salida.mkdir(parents=True)

    for rank, f in enumerate(elegidas, 1):
        img, dets = cache[f["nombre"]]
        dibujar(img, dets,
                args.salida / f"{rank:02d}_{f['nombre']}_p{f['puntaje']:.2f}.jpg")

    with (args.salida / "ranking.csv").open("w", newline="",
                                            encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "nombre", "animales", "conf_alta",
                    "frac_segura", "solape", "puntaje"])
        for rank, f in enumerate(elegidas, 1):
            w.writerow([rank, f["nombre"], f["n"], f["n_segura"],
                        f"{f['frac_segura']:.3f}", f"{f['solape']:.3f}",
                        f"{f['puntaje']:.4f}"])

    print("\n" + "=" * 62)
    print(f"Candidatas (mejores {len(elegidas)} de {len(utiles)} útiles):")
    for rank, f in enumerate(elegidas, 1):
        print(f"  {rank:2d}. {f['nombre']:<22} {f['n']:>3} animales   "
              f"puntaje {f['puntaje']:.2f}")
    print(f"\nImágenes en: {args.salida}")
    print("Rojo = confianza alta, amarillo = confianza baja.")
    print("Siguiente paso: revisar visualmente y quedarse con las que "
          "NO tengan animales sin caja.")


if __name__ == "__main__":
    main()
