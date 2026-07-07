"""
Calibrador del modelo de peso usando dataset propio.

Espera un CSV con columnas:
    image_path, peso_kg, raza
Donde:
    image_path: ruta a una imagen donde aparece UN SOLO animal y la
                referencia de 1,02 m visible.
    peso_kg:    peso real medido en balanza.
    raza:       angus / hereford / brangus / braford / cruza / desconocido

Para cada fila:
  1) Calibra píxeles → metros con la referencia
  2) Detecta el animal con YOLO
  3) Calcula área proyectada en m²
  4) Acumula (area_m2, peso_kg)
Al final ajusta los coeficientes a, b del modelo Peso = a · Area^b
y los guarda en JSON listo para cargar en la app.

Uso:
    python scripts/calibrate_weight.py \\
        --dataset data/calibracion.csv \\
        --config config.yaml \\
        --output models/weight_model.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# Permitir importar src/ aunque ejecutes el script desde la raíz
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration import calibrate
from src.detector import CattleDetector
from src.weight_estimator import (
    WeightModel,
    calibrate_power_model,
    calibrate_linear_model,
    evaluate_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="CSV con image_path, peso_kg, raza")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="models/weight_model.json")
    parser.add_argument("--modelo", default="potencia", choices=["potencia", "lineal"])
    parser.add_argument("--yolo", default="yolov8m-seg.pt",
                        help="Recomendado: usar modelo de segmentación")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(args.dataset)
    required = {"image_path", "peso_kg"}
    if not required.issubset(df.columns):
        raise ValueError(f"El CSV debe tener columnas {required}")

    detector = CattleDetector(
        model_path=args.yolo,
        cow_class_id=cfg["deteccion"]["clase_cow_id"],
        conf=0.25,
        iou=0.5,
        imgsz=1280,
    )

    areas, pesos, razas = [], [], []
    for i, row in df.iterrows():
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = ROOT / img_path
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("No pude leer %s, salto.", img_path)
            continue

        cal = calibrate(img, cfg)
        if cal is None:
            log.warning("Sin referencia visible en %s, salto.", img_path)
            continue

        dets = detector.detect(img)
        if not dets:
            log.warning("Sin detecciones en %s, salto.", img_path)
            continue

        # Si hay varios, agarrar el más grande (asumimos que la imagen
        # de calibración tiene un solo animal protagonista)
        det = max(dets, key=lambda d: d.area_px)
        area_m2 = cal.pixel_area_to_m2(det.area_px)

        areas.append(area_m2)
        pesos.append(float(row["peso_kg"]))
        razas.append(str(row.get("raza", "desconocido")).lower())

        log.info("%s → area=%.2f m², peso=%.1f kg",
                 img_path.name, area_m2, row["peso_kg"])

    areas = np.array(areas)
    pesos = np.array(pesos)
    if len(areas) < 5:
        raise RuntimeError(f"Necesito al menos 5 muestras válidas (tengo {len(areas)})")

    # Ajuste
    if args.modelo == "potencia":
        a, b, rmse_pct = calibrate_power_model(areas, pesos)
        log.info("Modelo potencia ajustado: Peso = %.2f * Area^%.3f (RMSE rel: %.2f%%)",
                 a, b, rmse_pct)
        model = WeightModel(modelo="potencia", coef_a=a, coef_b=b, offset_c=0.0,
                            factores_raza=cfg["estimacion_peso"]["factores_raza"])
    else:
        a, c, rmse_pct = calibrate_linear_model(areas, pesos)
        log.info("Modelo lineal ajustado: Peso = %.2f * Area + %.2f (RMSE rel: %.2f%%)",
                 a, c, rmse_pct)
        model = WeightModel(modelo="lineal", coef_a=a, coef_b=0.0, offset_c=c,
                            factores_raza=cfg["estimacion_peso"]["factores_raza"])

    # Evaluación final
    metrics = evaluate_model(model, areas, pesos, razas)
    log.info("Métricas finales: %s", json.dumps(metrics, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.to_json(out)
    log.info("✅ Modelo guardado en %s", out)

    if metrics.get("mape_pct", 100) <= 5.0:
        log.info("🎯 ¡Objetivo cumplido! MAPE = %.2f%% (<5%%)", metrics["mape_pct"])
    else:
        log.warning("⚠️  MAPE = %.2f%% — sumá más muestras o probá modelo de segmentación.",
                    metrics["mape_pct"])


if __name__ == "__main__":
    main()
