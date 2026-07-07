"""
Pipeline de procesamiento: imagen única o video.

Para video, se hace tracking con ByteTrack para mantener IDs estables y
así contar cada animal una sola vez. El peso final por animal es la
mediana del peso estimado a lo largo de los frames donde fue visible
(robusto a frames con oclusión parcial o detección imperfecta).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration import Calibration, calibrate, calibrate_from_altitude
from .detector import CattleDetector, CattleDetection
from .weight_estimator import WeightModel

log = logging.getLogger(__name__)


@dataclass
class AnimalRecord:
    track_id: int
    weights_kg: List[float] = field(default_factory=list)
    areas_m2: List[float] = field(default_factory=list)
    n_frames: int = 0
    last_centroid: Tuple[float, float] = (0.0, 0.0)
    last_bbox: Optional[np.ndarray] = None

    @property
    def peso_kg(self) -> float:
        if not self.weights_kg:
            return 0.0
        return float(np.median(self.weights_kg))

    @property
    def area_m2(self) -> float:
        if not self.areas_m2:
            return 0.0
        return float(np.median(self.areas_m2))


@dataclass
class ProcessResult:
    n_animales: int
    peso_promedio_kg: float
    peso_total_kg: float
    desvio_kg: float
    animales: List[AnimalRecord]
    calibracion: Optional[Calibration]
    output_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    # Estadísticas de calidad de la captura (solo video)
    n_frames_total: int = 0
    n_frames_validos: int = 0
    n_frames_sin_ref: int = 0
    n_frames_tilted: int = 0

    @property
    def calidad_captura_pct(self) -> float:
        if self.n_frames_total == 0:
            return 100.0
        return 100.0 * self.n_frames_validos / self.n_frames_total


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def _draw_overlay(
    frame: np.ndarray,
    detections: List[CattleDetection],
    weights: Dict[int, float],
    cal: Optional[Calibration],
    cfg: dict,
) -> np.ndarray:
    out = frame.copy()
    sal = cfg.get("salida", {})
    color_box = tuple(sal.get("color_caja_bgr", [0, 255, 0]))
    color_txt = tuple(sal.get("color_texto_bgr", [255, 255, 255]))
    color_ref = tuple(sal.get("color_referencia_bgr", [0, 165, 255]))
    scale = float(sal.get("fuente_escala", 0.7))
    thick = int(sal.get("grosor_caja", 2))

    if cal is not None and cal.reference_corners is not None and sal.get("dibujar_referencia", True):
        pts = cal.reference_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color_ref, thick + 1)
        cv2.putText(
            out,
            f"REF {cal.reference_side_m:.2f}m  ({cal.pixels_per_meter:.0f} px/m)",
            tuple(pts[0][0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color_ref,
            thick,
        )

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color_box, thick)
        labels = []
        if sal.get("dibujar_id", True) and det.track_id is not None:
            labels.append(f"#{det.track_id}")
        peso = weights.get(det.track_id) if det.track_id is not None else None
        if peso is not None and sal.get("dibujar_peso", True):
            labels.append(f"{peso:.0f} kg")
        labels.append(f"{det.confidence:.2f}")
        text = " | ".join(labels)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color_box, -1)
        cv2.putText(
            out,
            text,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color_txt,
            thick,
        )

    return out


def _draw_summary(frame: np.ndarray, n: int, prom_kg: float, total_kg: float) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    txt = f"Animales: {n}    Peso prom.: {prom_kg:.1f} kg    Peso total: {total_kg:.0f} kg"
    cv2.putText(frame, txt, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


# ----------------------------------------------------------------------
# Procesar imagen única
# ----------------------------------------------------------------------

def process_image(
    image_bgr: np.ndarray,
    detector: CattleDetector,
    weight_model: WeightModel,
    cfg: dict,
    raza: str = "desconocido",
    categoria: str = "desconocido",
    ajuste_fino: float = 1.0,
) -> Tuple[np.ndarray, ProcessResult]:
    cal = calibrate(image_bgr, cfg)
    if cal is None:
        h = image_bgr.shape[0]
        cal = calibrate_from_altitude(
            altitude_m=cfg.get("captura", {}).get("altura_vuelo_m", 10.0),
            image_height_px=h,
            image_width_px=image_bgr.shape[1],
        )
        log.warning("Sin referencia visible. Usando fallback por altitud (precisión limitada).")

    detections = detector.detect(image_bgr)
    weights: Dict[int, float] = {}
    animales: List[AnimalRecord] = []

    for i, det in enumerate(detections):
        area_m2 = cal.pixel_area_to_m2(det.area_px)
        peso = weight_model.estimate(area_m2, raza, categoria, ajuste_fino)
        det.track_id = i + 1  # asignar id efímero
        if peso is None:
            continue
        weights[det.track_id] = peso
        animales.append(
            AnimalRecord(
                track_id=det.track_id,
                weights_kg=[peso],
                areas_m2=[area_m2],
                n_frames=1,
                last_centroid=det.centroid_px,
                last_bbox=det.bbox_xyxy,
            )
        )

    n = len(animales)
    prom = float(np.mean([a.peso_kg for a in animales])) if animales else 0.0
    total = float(np.sum([a.peso_kg for a in animales])) if animales else 0.0
    desv = float(np.std([a.peso_kg for a in animales])) if animales else 0.0

    annotated = _draw_overlay(image_bgr, detections, weights, cal, cfg)
    annotated = _draw_summary(annotated, n, prom, total)

    return annotated, ProcessResult(
        n_animales=n,
        peso_promedio_kg=prom,
        peso_total_kg=total,
        desvio_kg=desv,
        animales=animales,
        calibracion=cal,
    )


# ----------------------------------------------------------------------
# Procesar video
# ----------------------------------------------------------------------

def process_video(
    video_path: str | Path,
    output_path: str | Path,
    detector: CattleDetector,
    weight_model: WeightModel,
    cfg: dict,
    raza: str = "desconocido",
    categoria: str = "desconocido",
    ajuste_fino: float = 1.0,
    progress_cb: Optional[Callable[[float], None]] = None,
    recalibrate_every_n_frames: int = 30,
) -> ProcessResult:
    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # avc1 (H.264) es compatible con Safari/Chrome; mp4v cae a fallback si no.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        log.warning("avc1 no disponible — usando mp4v (puede no reproducirse en Safari)")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    cal: Optional[Calibration] = None
    animals: Dict[int, AnimalRecord] = defaultdict(lambda: AnimalRecord(track_id=-1))
    tracker_cfg = cfg.get("tracking", {})

    # Estadísticas de calidad (para diagnóstico)
    n_frames_sin_ref = 0
    n_frames_tilted = 0
    n_frames_validos = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- Recalibración POR FRAME (no cada 30) ----
        # Aprovechamos que la detección de la lona es barata para recalibrar
        # cada frame. Eso compensa cualquier movimiento del drone por viento.
        new_cal = calibrate(frame, cfg)
        frame_calibration_valid = new_cal is not None

        if new_cal is not None:
            # ---- Filtro de inclinación de cámara ----
            # Si el cuadrado se ve como rombo (relación de lados < 0.85), la
            # cámara está inclinada por viento → área de animales distorsionada.
            # Mejor saltear este frame que aceptar datos incorrectos.
            if new_cal.is_camera_tilted:
                n_frames_tilted += 1
                frame_calibration_valid = False
                log.debug(
                    "Frame %d: cámara inclinada (aspect=%.2f), salteando",
                    frame_idx, new_cal.aspect_ratio,
                )
            else:
                cal = new_cal  # actualizar calibración válida
        else:
            n_frames_sin_ref += 1

        # Si no tenemos calibración válida en este frame, igual seguimos
        # tracking (para no romper IDs) pero NO acumulamos pesos
        if not frame_calibration_valid or cal is None:
            # Tracking sin acumular: solo para mantener IDs estables
            detector.track(
                frame,
                tracker=tracker_cfg.get("tracker", "bytetrack.yaml"),
                persist=tracker_cfg.get("persist", True),
            )
            # Escribir frame original (sin overlay de pesos)
            writer.write(frame)
            frame_idx += 1
            if progress_cb and n_total:
                progress_cb(frame_idx / n_total)
            continue

        n_frames_validos += 1
        detections = detector.track(
            frame,
            tracker=tracker_cfg.get("tracker", "bytetrack.yaml"),
            persist=tracker_cfg.get("persist", True),
        )

        weights_now: Dict[int, float] = {}
        for det in detections:
            if det.track_id is None:
                continue
            area_m2 = cal.pixel_area_to_m2(det.area_px)
            peso = weight_model.estimate(area_m2, raza, categoria, ajuste_fino)
            if peso is None:
                continue
            rec = animals[det.track_id]
            rec.track_id = det.track_id
            rec.weights_kg.append(peso)
            rec.areas_m2.append(area_m2)
            rec.n_frames += 1
            rec.last_centroid = det.centroid_px
            rec.last_bbox = det.bbox_xyxy
            weights_now[det.track_id] = rec.peso_kg  # peso "estable" (mediana)

        annotated = _draw_overlay(frame, detections, weights_now, cal, cfg)
        n_now = len(animals)
        if animals:
            avg = float(np.mean([a.peso_kg for a in animals.values()]))
            tot = float(np.sum([a.peso_kg for a in animals.values()]))
        else:
            avg = tot = 0.0
        annotated = _draw_summary(annotated, n_now, avg, tot)
        writer.write(annotated)

        frame_idx += 1
        if progress_cb and n_total:
            progress_cb(frame_idx / n_total)

    cap.release()
    writer.release()

    # Reporte de calidad de captura
    log.info(
        "Calidad del video: %d frames totales | %d válidos (%.0f%%) | "
        "%d sin referencia (%.0f%%) | %d con cámara inclinada (%.0f%%)",
        frame_idx,
        n_frames_validos, 100.0 * n_frames_validos / max(1, frame_idx),
        n_frames_sin_ref, 100.0 * n_frames_sin_ref / max(1, frame_idx),
        n_frames_tilted, 100.0 * n_frames_tilted / max(1, frame_idx),
    )
    if n_frames_validos < frame_idx * 0.5:
        log.warning(
            "⚠️ Menos del 50%% de frames son válidos. "
            "Revisá: la lona debe estar visible y plana, gimbal cenital."
        )

    # Filtro mínimo de frames: 2 para captar animales que pasan muy rápido
    # por el FOV. Ya no hacemos dedup espacial porque animales distintos
    # pueden pasar por el mismo lugar en momentos diferentes.
    min_frames = 2
    animales_finales = [a for a in animals.values() if a.n_frames >= min_frames]

    # Renumerar 1..N por orden de aparición (más útil para el productor que
    # los track_ids internos del modelo, que pueden ser 227, 234, etc.)
    animales_finales.sort(key=lambda a: a.track_id)
    for nuevo_id, animal in enumerate(animales_finales, start=1):
        animal.track_id = nuevo_id

    log.info(
        "Tracking: %d IDs únicos detectados → %d con >=%d frames retenidos (renumerados 1..%d)",
        len(animals), len(animales_finales), min_frames, len(animales_finales),
    )
    pesos = np.array([a.peso_kg for a in animales_finales])
    n = len(animales_finales)
    prom = float(np.mean(pesos)) if n else 0.0
    total = float(np.sum(pesos)) if n else 0.0
    desv = float(np.std(pesos)) if n else 0.0

    return ProcessResult(
        n_animales=n,
        peso_promedio_kg=prom,
        peso_total_kg=total,
        desvio_kg=desv,
        animales=animales_finales,
        calibracion=cal,
        output_path=output_path,
        n_frames_total=frame_idx,
        n_frames_validos=n_frames_validos,
        n_frames_sin_ref=n_frames_sin_ref,
        n_frames_tilted=n_frames_tilted,
    )


def export_results_csv(result: ProcessResult, csv_path: str | Path) -> Path:
    """Exporta resultados en formato amigable para el productor: solo
    número de animal y peso. Al final agrega fila de promedio, desvío
    estándar, peso total y conteo."""
    import csv
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Animal", "Peso (kg)"])
        for a in result.animales:
            writer.writerow([a.track_id, f"{a.peso_kg:.1f}"])

        # Filas de resumen
        writer.writerow([])
        writer.writerow(["Cantidad de animales", result.n_animales])
        writer.writerow(["Peso promedio (kg)", f"{result.peso_promedio_kg:.1f}"])
        writer.writerow(["Desvío estándar (kg)", f"{result.desvio_kg:.1f}"])
        writer.writerow(["Peso total del lote (kg)", f"{result.peso_total_kg:.0f}"])

    result.csv_path = csv_path
    return csv_path
