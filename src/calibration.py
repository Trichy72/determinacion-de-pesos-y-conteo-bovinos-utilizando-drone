"""
Módulo de calibración de escala.

Detecta el cuadrado de referencia en el piso (de lado conocido, p. ej. 1.02 m)
y devuelve el factor de conversión píxeles -> metros.

Soporta dos métodos:
  1) ArUco (recomendado): un marcador ArUco impreso de 1.02 m en el lote.
  2) Cuadrado de color sólido: detección por umbral HSV + contornos.

Como las imágenes se toman a 90° (cenital) y a altura fija, asumimos
proyección ortogonal: 1 m = N píxeles, igual en X e Y.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Calibration:
    """Resultado de la calibración de escala."""

    pixels_per_meter: float
    method: str
    reference_corners: Optional[np.ndarray] = None  # 4x2 px
    reference_side_m: float = 1.02
    confidence: float = 1.0
    tilt_ratio: float = 1.0   # 1.0 = cámara perfectamente cenital; <0.85 = inclinada
    aspect_ratio: float = 1.0 # ratio min/max lado del cuadrado detectado

    @property
    def meters_per_pixel(self) -> float:
        return 1.0 / self.pixels_per_meter

    def pixel_area_to_m2(self, area_px: float) -> float:
        return area_px * (self.meters_per_pixel ** 2)

    @property
    def is_camera_tilted(self) -> bool:
        """True si la cámara no está cenital (>20° de inclinación aprox)."""
        return self.aspect_ratio < 0.75


# ----------------------------------------------------------------------
# Método 1: ArUco
# ----------------------------------------------------------------------

_ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def detect_aruco_reference(
    image: np.ndarray,
    side_m: float = 1.02,
    aruco_dict_name: str = "DICT_4X4_50",
    target_id: Optional[int] = 0,
) -> Optional[Calibration]:
    """Detecta un marcador ArUco y devuelve la calibración."""
    if aruco_dict_name not in _ARUCO_DICTS:
        raise ValueError(f"ArUco dict desconocido: {aruco_dict_name}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICTS[aruco_dict_name])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # Elegir el id objetivo (o el primero detectado si no se especifica)
    chosen = None
    if target_id is not None:
        for c, mid in zip(corners, ids.flatten()):
            if int(mid) == target_id:
                chosen = c[0]
                break
    if chosen is None:
        chosen = corners[0][0]

    # Lado en píxeles = promedio de los 4 lados
    sides_px = [
        np.linalg.norm(chosen[0] - chosen[1]),
        np.linalg.norm(chosen[1] - chosen[2]),
        np.linalg.norm(chosen[2] - chosen[3]),
        np.linalg.norm(chosen[3] - chosen[0]),
    ]
    side_px = float(np.mean(sides_px))
    if side_px < 5:
        return None

    aspect = float(min(sides_px) / max(sides_px))
    ppm = side_px / side_m
    return Calibration(
        pixels_per_meter=ppm,
        method="aruco",
        reference_corners=chosen,
        reference_side_m=side_m,
        confidence=1.0,
        aspect_ratio=aspect,
        tilt_ratio=aspect,
    )


# ----------------------------------------------------------------------
# Método 2: cuadrado de color sólido
# ----------------------------------------------------------------------


def detect_color_square_reference(
    image: np.ndarray,
    side_m: float = 1.02,
    hsv_min: Tuple[int, int, int] = (20, 100, 100),
    hsv_max: Tuple[int, int, int] = (35, 255, 255),
    min_area_px: int = 200,   # Más permisivo (era 300)
) -> Optional[Calibration]:
    """
    Detecta un cuadrado de color sólido en el piso, tolerante a:
      - Rotación (rombo, diagonal)
      - Arrugas o bordes irregulares
      - Símbolos impresos encima (siempre que >50% del área esté del color)

    Usa cv2.minAreaRect que devuelve el rectángulo orientado de mínima
    área que envuelve el contorno, sin exigir 4 esquinas perfectas.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_min), np.array(hsv_max))

    # Cierre morfológico fuerte para "cerrar" arrugas y símbolos internos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        # Rectángulo orientado de área mínima
        rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
        (cx, cy), (w, h), angle = rect
        if w < 5 or h < 5:
            continue

        # Squareness: lado menor / lado mayor cerca de 1
        ratio = min(w, h) / max(w, h)
        if ratio < 0.6:  # más permisivo (era 0.7) - tolera ráfagas de viento
            continue

        # ¿Cuánto del rectángulo está realmente lleno de color?
        rect_area = w * h
        fill = area / rect_area
        if fill < 0.45:  # tolera arrugas, símbolos, oclusión parcial por animal
            continue

        # Puntaje combinado: área grande, cuadrado, y bien lleno
        score = area * ratio * fill
        if score > best_score:
            box_pts = cv2.boxPoints(rect).astype(np.float32)
            best_score = score
            best = box_pts

    if best is None:
        return None

    sides = [np.linalg.norm(best[i] - best[(i + 1) % 4]) for i in range(4)]
    side_px = float(np.mean(sides))
    aspect = float(min(sides) / max(sides))   # ~1.0 = cámara cenital, <0.85 = tilted
    ppm = side_px / side_m
    return Calibration(
        pixels_per_meter=ppm,
        method="color_square",
        reference_corners=best,
        reference_side_m=side_m,
        confidence=min(1.0, best_score / (min_area_px * 10)),
        aspect_ratio=aspect,
        tilt_ratio=aspect,
    )


# ----------------------------------------------------------------------
# API principal
# ----------------------------------------------------------------------


def calibrate(image: np.ndarray, cfg: dict) -> Optional[Calibration]:
    """Calibra usando el método indicado en config.yaml."""
    ref_cfg = cfg.get("referencia", {})
    side_m = float(ref_cfg.get("lado_m", 1.02))
    metodo = ref_cfg.get("metodo", "aruco")

    if metodo == "aruco":
        cal = detect_aruco_reference(
            image,
            side_m=side_m,
            aruco_dict_name=ref_cfg.get("aruco_dict", "DICT_4X4_50"),
            target_id=ref_cfg.get("aruco_id"),
        )
        if cal is not None:
            return cal
        log.warning("No se detectó ArUco, intentando cuadrado de color…")

    cal = detect_color_square_reference(
        image,
        side_m=side_m,
        hsv_min=tuple(ref_cfg.get("color_hsv_min", [20, 100, 100])),
        hsv_max=tuple(ref_cfg.get("color_hsv_max", [35, 255, 255])),
    )
    return cal


def calibrate_from_altitude(
    altitude_m: float,
    image_height_px: int,
    sensor_width_mm: float = 13.2,
    focal_length_mm: float = 8.8,
    image_width_px: int = 3840,
) -> Calibration:
    """Calibración fallback usando datos de cámara y altitud (DJI Mavic-like)."""
    gsd = (sensor_width_mm * altitude_m * 100) / (focal_length_mm * image_width_px)  # cm/px
    ppm = 100.0 / gsd
    log.warning(
        "Calibración por altitud: GSD=%.3f cm/px → %.2f px/m (precisión limitada)",
        gsd,
        ppm,
    )
    return Calibration(
        pixels_per_meter=ppm,
        method="altitude_fallback",
        confidence=0.5,
    )
