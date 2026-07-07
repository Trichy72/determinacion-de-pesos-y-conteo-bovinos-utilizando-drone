"""
Fine-tuning de YOLOv8 con dataset propio de bovinos en vista aérea.

Este script asume que ya tenés tus videos etiquetados en formato YOLO
(carpeta con images/ y labels/). Si no, primero corré:
    python scripts/extract_frames.py  (extrae frames del video)
y después etiquetalos con un labeller como CVAT, LabelImg o Roboflow.

ESTRUCTURA ESPERADA del dataset:

    dataset_bovinos/
    ├── images/
    │   ├── train/   (80% de tus frames)
    │   ├── val/     (15%)
    │   └── test/    (5%)
    ├── labels/
    │   ├── train/   (un .txt por imagen, formato YOLO)
    │   ├── val/
    │   └── test/
    └── data.yaml    (archivo de configuración, lo genera este script)

Cada label .txt tiene una línea por animal:
    0 cx cy w h     (todas las coordenadas normalizadas 0-1)
    ↑ class_id (0 = bovino, único)

Uso:
    python scripts/finetune_yolo.py \\
        --dataset /ruta/a/dataset_bovinos \\
        --modelo yolov8m.pt \\
        --epocas 100 \\
        --output models/bovino_finetuned.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def crear_data_yaml(dataset_path: Path) -> Path:
    """Genera el archivo data.yaml requerido por ultralytics."""
    yaml_path = dataset_path / "data.yaml"
    config = {
        "path": str(dataset_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "bovino"},
        "nc": 1,
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"📝 data.yaml creado: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Carpeta del dataset (estructura YOLO)")
    parser.add_argument("--modelo", default="yolov8m.pt",
                        help="Modelo base. yolov8m.pt es buen punto de partida")
    parser.add_argument("--epocas", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4,
                        help="Reducir a 2 si tu GPU/RAM se queda corta")
    parser.add_argument("--device", default="",
                        help="'cpu', '0' (gpu 0), 'mps' (Apple Silicon), o vacío para auto")
    parser.add_argument("--output", default="models/bovino_finetuned.pt")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.is_dir():
        print(f"❌ Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    # Validar estructura
    required = ["images/train", "images/val", "labels/train", "labels/val"]
    missing = [r for r in required if not (dataset_path / r).is_dir()]
    if missing:
        print("❌ Faltan estas carpetas en el dataset:")
        for m in missing:
            print(f"   - {m}")
        print("\nEstructura mínima requerida:")
        print("   dataset/images/train/*.jpg  + labels/train/*.txt")
        print("   dataset/images/val/*.jpg    + labels/val/*.txt")
        sys.exit(1)

    yaml_path = crear_data_yaml(dataset_path)

    # Importar después de validar (ultralytics es pesado)
    from ultralytics import YOLO

    print(f"\n🧠 Cargando modelo base: {args.modelo}")
    model = YOLO(args.modelo)

    print(f"\n🚀 Entrenando {args.epocas} épocas, imgsz={args.imgsz}, batch={args.batch}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Device: {args.device or 'auto'}")
    print()

    results = model.train(
        data=str(yaml_path),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=20,             # early stopping si no mejora
        project="runs/bovino",
        name="finetune",
        exist_ok=True,
        # Augmentations clave para vista aérea
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,              # rotación pequeña (drone no rota mucho)
        translate=0.1,
        scale=0.5,               # crítico: simular distintas alturas
        fliplr=0.5,
        mosaic=1.0,
    )

    # Copiar el mejor weight al destino
    best_pt = Path("runs/bovino/finetune/weights/best.pt")
    if best_pt.exists():
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, out)
        print(f"\n✅ Modelo guardado: {out}")
        print(f"\nUsalo en la app: en sidebar → Modelo YOLO → escribí la ruta `{out}`")
    else:
        print(f"\n⚠️  No encontré best.pt en runs/bovino/finetune/weights/")
        print("    Revisá los logs de entrenamiento.")


if __name__ == "__main__":
    main()
