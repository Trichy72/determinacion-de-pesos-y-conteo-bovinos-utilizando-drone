# -*- coding: utf-8 -*-
"""
Detección por mosaicos (tiled inference) para vista cenital de drone.

Problema: en una foto 4000x2250 tomada a 20 m, cada bovino ocupa ~150 px.
YOLO redimensiona la imagen entera a 1280 px antes de mirarla, así que el
animal le llega de ~50 px y se le escapan muchos (por eso contaba 2 de 3).

Solución: partir la foto en cuadrados con solapamiento, correr el detector
en cada uno a resolución nativa (el animal le llega grande), y después
juntar todas las detecciones en coordenadas de la foto original.

Detalles que importan:
  - Solapamiento del 25 %: un animal partido por el corte de un mosaico
    aparece entero en el mosaico vecino.
  - Se descartan las detecciones que tocan el borde de un mosaico (salvo
    que ese borde sea también borde de la foto): están cortadas y el
    vecino ya las tiene enteras.
  - Se suma una pasada sobre la foto completa, para los animales grandes
    (fotos a 10 m) que no entran en un mosaico.
  - NMS global por IoU para eliminar el mismo animal detectado dos veces.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Deteccion = Tuple[float, float, float, float, float]  # x1, y1, x2, y2, conf

TILE_DEFAULT = 1024
OVERLAP_DEFAULT = 0.25
IOU_NMS_DEFAULT = 0.45
MARGEN_BORDE_TILE = 4       # px: "toca el borde del mosaico"
AREA_MIN_FRAC = 1e-5        # descarta motas de polvo
AREA_MAX_FRAC = 0.25        # descarta cajas absurdas (1/4 de la foto)


def _iou_matriz(caja: np.ndarray, otras: np.ndarray) -> np.ndarray:
    """IoU de una caja contra un array de cajas (formato x1 y1 x2 y2)."""
    x1 = np.maximum(caja[0], otras[:, 0])
    y1 = np.maximum(caja[1], otras[:, 1])
    x2 = np.minimum(caja[2], otras[:, 2])
    y2 = np.minimum(caja[3], otras[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_c = (caja[2] - caja[0]) * (caja[3] - caja[1])
    area_o = (otras[:, 2] - otras[:, 0]) * (otras[:, 3] - otras[:, 1])
    union = area_c + area_o - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(dets: Sequence[Deteccion],
        iou_thr: float = IOU_NMS_DEFAULT) -> List[Deteccion]:
    """Non-Maximum Suppression clásico: se queda con la de mayor confianza
    de cada grupo de cajas superpuestas."""
    if not dets:
        return []
    arr = np.asarray(dets, dtype=np.float32)
    orden = np.argsort(-arr[:, 4])
    arr = arr[orden]
    quedan = []
    while len(arr):
        mejor = arr[0]
        quedan.append(tuple(float(v) for v in mejor))
        if len(arr) == 1:
            break
        ious = _iou_matriz(mejor, arr[1:])
        arr = arr[1:][ious < iou_thr]
    return quedan


def _grilla(largo: int, tile: int, paso: int) -> List[int]:
    """Posiciones de inicio de los mosaicos sobre un eje."""
    if largo <= tile:
        return [0]
    pos = list(range(0, largo - tile + 1, paso))
    if pos[-1] != largo - tile:
        pos.append(largo - tile)
    return pos


def detectar_por_mosaicos(predict_fn,
                          imagen: np.ndarray,
                          tile: int = TILE_DEFAULT,
                          overlap: float = OVERLAP_DEFAULT,
                          iou_nms: float = IOU_NMS_DEFAULT,
                          pasada_completa: bool = True) -> List[Deteccion]:
    """Corre `predict_fn` sobre mosaicos de `imagen` y devuelve las
    detecciones en coordenadas de la imagen original.

    predict_fn(recorte_ndarray) -> [(x1, y1, x2, y2, conf), ...]
        coordenadas relativas al recorte que recibe.
    """
    alto, ancho = imagen.shape[:2]
    paso = max(1, int(tile * (1.0 - overlap)))
    todas: List[Deteccion] = []

    for y0 in _grilla(alto, tile, paso):
        for x0 in _grilla(ancho, tile, paso):
            x1t, y1t = min(x0 + tile, ancho), min(y0 + tile, alto)
            recorte = imagen[y0:y1t, x0:x1t]
            if recorte.size == 0:
                continue
            for (bx1, by1, bx2, by2, cf) in predict_fn(recorte):
                # Caja cortada por el mosaico: la tiene entera el vecino.
                toca_izq = bx1 <= MARGEN_BORDE_TILE and x0 > 0
                toca_arr = by1 <= MARGEN_BORDE_TILE and y0 > 0
                toca_der = bx2 >= (x1t - x0) - MARGEN_BORDE_TILE and x1t < ancho
                toca_aba = by2 >= (y1t - y0) - MARGEN_BORDE_TILE and y1t < alto
                if toca_izq or toca_arr or toca_der or toca_aba:
                    continue
                todas.append((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0, cf))

    if pasada_completa:
        todas.extend(predict_fn(imagen))

    # Filtro de tamaño: fuera motas y cajas absurdas
    area_img = float(ancho * alto)
    filtradas = []
    for (x1, y1, x2, y2, cf) in todas:
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            continue
        frac = (w * h) / area_img
        if frac < AREA_MIN_FRAC or frac > AREA_MAX_FRAC:
            continue
        filtradas.append((max(0.0, x1), max(0.0, y1),
                          min(float(ancho), x2), min(float(alto), y2), cf))

    return nms(filtradas, iou_nms)
