"""
Extrae frames de un video (o varios) para etiquetar y armar dataset YOLO.

Uso:
    # Extraer 1 frame cada 0.5 segundos (15 frames/s a 30fps → 1 frame cada 15)
    python scripts/extract_frames.py --videos "/ruta/videos/*.MP4" \\
        --output dataset_bovinos/images_raw \\
        --cada-n 15

Después de extraer:
  1) Subí los frames a Roboflow / CVAT / LabelImg
  2) Etiquetá las cajas (clase única: 'bovino')
  3) Exportá en formato YOLO
  4) Acomodá en images/train|val + labels/train|val (80-15-5 split)
  5) Corré scripts/finetune_yolo.py
"""

import argparse
import glob
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True,
                        help="Glob de videos: '/ruta/*.MP4'")
    parser.add_argument("--output", required=True, help="Carpeta de salida")
    parser.add_argument("--cada-n", type=int, default=15,
                        help="Extrae 1 frame cada N (15 = 2 fps si el video es 30fps)")
    parser.add_argument("--max-por-video", type=int, default=200,
                        help="Tope de frames por video (evita sobrecarga)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(glob.glob(args.videos))
    if not videos:
        print(f"❌ No encontré videos con: {args.videos}")
        return

    print(f"📹 {len(videos)} video(s) encontrados")
    total = 0
    for vp in videos:
        name = Path(vp).stem.replace(" ", "_")
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"⚠️  No pude abrir {vp}")
            continue

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        for fidx in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if fidx % args.cada_n != 0:
                continue
            if saved >= args.max_por_video:
                break
            out_file = out_dir / f"{name}_f{fidx:06d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved += 1
        cap.release()
        print(f"   {name}: {saved} frames")
        total += saved

    print(f"\n✅ Total: {total} frames en {out_dir}")
    print("\nSiguientes pasos:")
    print("  1) Subí estos frames a Roboflow (gratis hasta 10k imágenes)")
    print("  2) Etiquetá cada vaca con clase 'bovino'")
    print("  3) Generá split train/val/test y bajá el ZIP")
    print("  4) Ejecutá: python scripts/finetune_yolo.py --dataset <carpeta>")


if __name__ == "__main__":
    main()
