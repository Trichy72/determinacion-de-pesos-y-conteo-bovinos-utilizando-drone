# -*- coding: utf-8 -*-
"""
Prepara un lote chico de imágenes para completar a mano en makesense.ai.

Por qué: el detector genérico ubica bien las cajas pero se le escapa ~40 %
de los animales. Un dataset incompleto le enseña al modelo "acá no hay
vaca", así que hay que completarlo con ojo humano. Con ~30 imágenes bien
etiquetadas alcanza para que el modelo aprenda este dominio; después ese
modelo re-etiqueta el resto solo.

Qué hace:
  1. Elige las N imágenes más útiles del dataset ya pre-etiquetado,
     priorizando variedad: distintas alturas, distintos corrales, y un mix
     de densidad (algunas con pocos animales, algunas con muchos). Evita
     frames casi idénticos del mismo video (que aportan poco).
  2. Las copia a una carpeta plana, redimensionadas a 1920 px de ancho
     (makesense.ai se pone lento con 4000 px, y el entrenamiento va a
     1280 igual).
  3. Copia los labels YOLO correspondientes (el pre-etiquetado, que es el
     punto de partida del trabajo manual).
  4. Escribe un INSTRUCCIONES.txt con el paso a paso de makesense.ai.

Uso:
    python scripts/preparar_etiquetado_manual.py
    python scripts/preparar_etiquetado_manual.py --dataset dataset_v2 --n 30
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ANCHO_MAX = 1920
N_DEFAULT = 30

INSTRUCCIONES = """\
CÓMO COMPLETAR LAS ETIQUETAS (makesense.ai)
===========================================

Objetivo: que TODOS los bovinos de cada foto tengan una caja. Las cajas
que ya están vienen del detector automático y son correctas: solo hay que
AGREGAR las que faltan (y borrar alguna que esté claramente sobre el
suelo o sobre una sombra, si aparece).

PASOS
-----

1. Entrá a  https://www.makesense.ai  y clic en "Get Started".

2. Arrastrá TODAS las imágenes .jpg de esta carpeta ("imagenes/") a la
   ventana. Elegí "Object Detection".

3. Te va a pedir una lista de etiquetas: escribí una sola,
   exactamente así (en minúscula):

       bovino

   Y confirmá.

4. IMPORTANTE — cargar el trabajo ya hecho:
   Arriba a la izquierda: Actions -> Import Annotations -> Single file
   in YOLO format... no: elegí "Import Annotations" y el formato "YOLO",
   y seleccioná TODOS los archivos .txt de la carpeta "labels/".
   Van a aparecer las cajas rojas que ya detectó la máquina.

   (Si makesense te pide un archivo labels.txt con los nombres de clase,
   usá el que está en esta carpeta: clases.txt)

5. Ahora el trabajo: foto por foto, dibujá una caja sobre cada bovino
   que NO tenga caja.

   Criterios:
     - La caja cubre SOLO el cuerpo del animal, NO su sombra.
       (La sombra es el error más común y arruina la estimación de peso.)
     - Animal cortado por el borde de la foto: marcalo igual, con lo que
       se ve.
     - Animales pegados unos a otros: una caja por animal, aunque se
       solapen entre sí. Solapar está bien.
     - Si no distinguís si son uno o dos animales, marcá uno solo.
     - Si una caja existente está sobre el suelo vacío o sobre una
       sombra, borrala (clic en la caja -> tecla Delete).

   Atajos que ayudan:
     - Flechas ← → : cambiar de imagen
     - Rueda del mouse: zoom
     - Barra espaciadora + arrastrar: mover la imagen

6. Cuando termines TODAS las imágenes:
   Actions -> Export Annotations -> "A .zip package containing files in
   YOLO format" -> Export.

7. Guardá ese zip en la carpeta del proyecto y avisame. Yo armo el
   dataset final y lo mandamos a entrenar.

CONSEJO: hacelo en 2 o 3 tandas. makesense.ai guarda todo en el navegador
mientras no cierres la pestaña, pero si vas a cortar, exportá el zip
antes de cerrar y avisame — así no se pierde nada.
"""


def elegir_desde_ranking(dataset: Path, ranking_csv: Path, n: int):
    """Usa el ranking de scripts/seleccionar_limpias.py.

    Ese script ya midió qué imágenes tienen el pre-etiquetado más completo
    (escenas de animales separados, donde el detector estuvo cómodo). Son
    las que menos trabajo manual necesitan: falta agregar 1 a 3 cajas, no
    empezar de cero.
    """
    import csv as _csv
    orden = []
    with ranking_csv.open(encoding="utf-8") as fh:
        for fila in _csv.DictReader(fh):
            orden.append(fila["nombre"])

    items = []
    for nombre in orden[:n]:
        for split in ("train", "val"):
            img = dataset / "images" / split / f"{nombre}.jpg"
            if img.exists():
                lab = dataset / "labels" / split / f"{nombre}.txt"
                n_cajas = 0
                if lab.exists():
                    n_cajas = sum(1 for l in lab.read_text().splitlines()
                                  if l.strip())
                items.append((nombre, img, lab, n_cajas))
                break
    return items


def elegir_imagenes(dataset: Path, n: int):
    """Elige n imágenes priorizando variedad de escenas y densidades.

    - Como máximo 2 frames por video (los frames consecutivos del mismo
      video son casi la misma escena: aportan poco y cuestan el mismo
      trabajo manual).
    - Mezcla densidades: ordena por cantidad de animales detectados y
      toma de forma pareja a lo largo de todo el rango.
    """
    items = []
    for split in ("train", "val"):
        for img in sorted((dataset / "images" / split).glob("*.jpg")):
            lab = dataset / "labels" / split / f"{img.stem}.txt"
            n_cajas = 0
            if lab.exists():
                n_cajas = sum(1 for l in lab.read_text().splitlines()
                              if l.strip())
            items.append((img.stem, img, lab, n_cajas))

    # Fuente = foto suelta (DJI_0084) o video (DJI_0080_t039 -> DJI_0080)
    def fuente(nombre: str) -> str:
        m = re.match(r"(DJI_\d+)_t\d+$", nombre)
        return m.group(1) if m else nombre

    # Descarta imágenes sin ninguna detección: son las que el detector no
    # entendió, y arrancar de cero en ellas es el trabajo más caro.
    items = [it for it in items if it[3] > 0]

    # Máximo 2 por video, quedándose con las de más animales
    por_fuente = {}
    for it in sorted(items, key=lambda t: -t[3]):
        f = fuente(it[0])
        por_fuente.setdefault(f, []).append(it)
    candidatos = []
    for f, lista in por_fuente.items():
        candidatos.extend(lista[:2])

    # Muestreo parejo a lo largo del rango de densidad
    candidatos.sort(key=lambda t: t[3])
    if len(candidatos) <= n:
        return candidatos
    paso = len(candidatos) / n
    return [candidatos[int(i * paso)] for i in range(n)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Prepara imágenes + pre-etiquetas para completar a "
                    "mano en makesense.ai")
    ap.add_argument("--dataset", type=Path,
                    default=REPO_ROOT / "dataset_v2")
    ap.add_argument("--salida", type=Path,
                    default=REPO_ROOT / "etiquetado_manual")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--ancho", type=int, default=ANCHO_MAX)
    ap.add_argument("--conf", type=float, default=0.05,
                    help="Re-detecta las elegidas con este umbral (0.05 = "
                         "la mejor combinación medida: arranca con más "
                         "cajas hechas). 0 = usar las del dataset.")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.30)
    args = ap.parse_args(argv)

    from PIL import Image

    if not args.dataset.exists():
        print(f"No encuentro {args.dataset}. Corré primero "
              f"scripts/preparar_dataset.py")
        return

    # Si ya corriste seleccionar_limpias.py, usamos su ranking: son las
    # imágenes que menos trabajo manual necesitan.
    ranking = REPO_ROOT / "candidatas_limpias" / "ranking.csv"
    if ranking.exists():
        print(f"Usando el ranking de {ranking.parent.name}/ "
              f"(las escenas que menos trabajo manual necesitan)\n")
        elegidas = elegir_desde_ranking(args.dataset, ranking, args.n)
    else:
        print("No encontré candidatas_limpias/ranking.csv — elijo por "
              "variedad. Para mejores resultados corré primero "
              "scripts/seleccionar_limpias.py\n")
        elegidas = elegir_imagenes(args.dataset, args.n)

    if not elegidas:
        print("No hay imágenes con detecciones en el dataset.")
        return

    if args.salida.exists():
        shutil.rmtree(args.salida)
    (args.salida / "imagenes").mkdir(parents=True)
    (args.salida / "labels").mkdir(parents=True)

    redetectar = None
    if args.conf > 0:
        print(f"Re-detectando con conf {args.conf} (mosaicos {args.tile}px, "
              f"solapamiento {args.overlap:.0%}) para arrancar con más "
              f"cajas hechas…\n")
        from scripts.preparar_dataset import crear_detector_mosaico
        redetectar = crear_detector_mosaico(conf=args.conf, tile=args.tile,
                                            overlap=args.overlap)

    total_cajas = 0
    print(f"Preparando {len(elegidas)} imágenes en {args.salida}\n")
    for nombre, img_path, lab_path, n_cajas in elegidas:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            ancho_orig, alto_orig = im.size
            if im.width > args.ancho:
                alto = round(im.height * args.ancho / im.width)
                im = im.resize((args.ancho, alto), Image.LANCZOS)
            im.save(args.salida / "imagenes" / f"{nombre}.jpg",
                    "JPEG", quality=90)

        destino_lab = args.salida / "labels" / f"{nombre}.txt"
        if redetectar is not None:
            dets = redetectar(str(img_path))
            lineas = []
            for (x1, y1, x2, y2, _cf) in dets:
                w, h = x2 - x1, y2 - y1
                if w <= 1 or h <= 1:
                    continue
                lineas.append(
                    f"0 {(x1 + x2) / 2 / ancho_orig:.6f} "
                    f"{(y1 + y2) / 2 / alto_orig:.6f} "
                    f"{w / ancho_orig:.6f} {h / alto_orig:.6f}")
            destino_lab.write_text("\n".join(lineas) +
                                   ("\n" if lineas else ""),
                                   encoding="utf-8")
            n_cajas = len(lineas)
        elif lab_path.exists():
            shutil.copy2(lab_path, destino_lab)

        total_cajas += n_cajas
        print(f"  {nombre}: {n_cajas} cajas ya detectadas")

    (args.salida / "clases.txt").write_text("bovino\n", encoding="utf-8")
    (args.salida / "INSTRUCCIONES.txt").write_text(INSTRUCCIONES,
                                                   encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Listo: {len(elegidas)} imágenes, {total_cajas} cajas de "
          f"arranque ({total_cajas / len(elegidas):.1f} por imagen)")
    print(f"Carpeta: {args.salida}")
    print(f"Leé:     {args.salida / 'INSTRUCCIONES.txt'}")


if __name__ == "__main__":
    main()
