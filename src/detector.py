"""
Detector de bovinos basado en YOLOv8.

YOLOv8 pre-entrenado en COCO ya conoce la clase 'cow' (id 19). Eso da un
buen baseline para vacunos vistos desde arriba. Para mejorar precisión y
discriminación de razas, se recomienda fine-tuning con un dataset propio
(ver scripts/finetune_yolo.py).

El detector también devuelve la máscara de segmentación si el modelo lo
soporta (yolov8*-seg.pt), lo que permite calcular el área proyectada con
mucha más fidelidad que el bounding box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CattleDetection:
    """Una detección individual."""

    track_id: Optional[int]
    bbox_xyxy: np.ndarray            # 4
    confidence: float
    mask: Optional[np.ndarray] = None  # bool HxW
    area_px: float = 0.0
    centroid_px: tuple = (0.0, 0.0)
    extras: dict = field(default_factory=dict)

    @property
    def x1(self): return float(self.bbox_xyxy[0])
    @property
    def y1(self): return float(self.bbox_xyxy[1])
    @property
    def x2(self): return float(self.bbox_xyxy[2])
    @property
    def y2(self): return float(self.bbox_xyxy[3])
    @property
    def width_px(self): return self.x2 - self.x1
    @property
    def height_px(self): return self.y2 - self.y1


class CattleDetector:
    """Wrapper sobre ultralytics.YOLO.

    NOTA importante: YOLO entrenado en COCO casi no vio vacas en vista
    cenital (drone a 10m). Por eso suele clasificar bovinos vistos desde
    arriba como 'sheep', 'horse', 'bear' o 'dog'. Para tolerarlo,
    aceptamos un set de clases COCO que en aerial-view son comunes para
    cuadrúpedos grandes:
        17 dog, 18 horse, 19 cow, 20 sheep, 22 zebra, 23 giraffe (todas grandes)
    Y descartamos por tamaño/área aquellas detecciones que no encajan
    en el rango de un bovino real (filtro en weight_estimator).
    """

    # COCO ids de cuadrúpedos grandes que un bovino aéreo puede ser
    # confundido con. Todos contribuyen como "bovino candidato".
    DEFAULT_CATTLE_CLASSES = [17, 18, 19, 20, 21, 22, 23]
    # 17 dog, 18 horse, 19 cow, 20 sheep, 21 elephant, 22 bear, 23 zebra

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        cow_class_id: int = 19,
        conf: float = 0.15,
        iou: float = 0.5,
        imgsz: int = 1280,
        device: Optional[str] = None,
        cattle_classes: Optional[list] = None,
        modo_tropa_densa: bool = False,
        tile_grid: int = 2,         # 2x2 o 3x3 tiles
    ):
        # Import perezoso para que el módulo se pueda inspeccionar sin GPU
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.cow_class_id = cow_class_id
        self.cattle_classes = cattle_classes or list(self.DEFAULT_CATTLE_CLASSES)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.has_segmentation = "seg" in model_path.lower()
        self.modo_tropa_densa = modo_tropa_densa
        self.tile_grid = tile_grid

    # ------------------------------------------------------------------
    # Inferencia simple (frame a frame, sin tracking)
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[CattleDetection]:
        if self.modo_tropa_densa:
            return self._detect_tiled(image)
        results = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.cattle_classes,
            device=self.device,
            verbose=False,
        )
        return self._parse(results[0], image.shape[:2])

    # ------------------------------------------------------------------
    # Inferencia tiled (SAHI-style) para tropas densas
    # ------------------------------------------------------------------
    def _detect_tiled(self, image: np.ndarray) -> List[CattleDetection]:
        """Divide el frame en tiles solapados, infiere en cada uno y mergea
        con NMS. Muy efectivo para tropas densas donde animales pegados se
        ven como una masa en la inferencia full-frame."""
        h, w = image.shape[:2]
        rows = cols = self.tile_grid
        overlap = 0.20  # 20% de solape entre tiles para no perder animales en bordes

        tile_w = int(w / cols * (1 + overlap))
        tile_h = int(h / rows * (1 + overlap))
        step_w = int(w / cols)
        step_h = int(h / rows)

        all_boxes = []
        all_confs = []
        all_clses = []

        for r in range(rows):
            for c in range(cols):
                x0 = c * step_w
                y0 = r * step_h
                x1 = min(x0 + tile_w, w)
                y1 = min(y0 + tile_h, h)
                tile = image[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                results = self.model.predict(
                    tile,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    classes=self.cattle_classes,
                    device=self.device,
                    verbose=False,
                )
                r0 = results[0]
                if r0.boxes is None or len(r0.boxes) == 0:
                    continue
                boxes = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                clses = r0.boxes.cls.cpu().numpy()
                # Trasladar las cajas a coordenadas globales
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                all_boxes.append(boxes)
                all_confs.append(confs)
                all_clses.append(clses)

        if not all_boxes:
            return []

        boxes = np.vstack(all_boxes)
        confs = np.concatenate(all_confs)
        clses = np.concatenate(all_clses)

        # NMS final para fusionar duplicados de tiles solapados
        keep = self._nms(boxes, confs, iou_thresh=self.iou)
        boxes = boxes[keep]
        confs = confs[keep]
        clses = clses[keep]

        # Construir CattleDetection con áreas a partir de bbox
        BBOX_TO_SILHOUETTE = 0.69
        out: List[CattleDetection] = []
        for box, conf, cls in zip(boxes, confs, clses):
            bbox_area = float((box[2] - box[0]) * (box[3] - box[1]))
            cx = float((box[0] + box[2]) / 2)
            cy = float((box[1] + box[3]) / 2)
            out.append(CattleDetection(
                track_id=None,
                bbox_xyxy=box.astype(np.float32),
                confidence=float(conf),
                area_px=bbox_area * BBOX_TO_SILHOUETTE,
                centroid_px=(cx, cy),
            ))
        return out

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> List[int]:
        """NMS simple para mergear detecciones de tiles solapados."""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    # ------------------------------------------------------------------
    # Inferencia con tracking (para video)
    # ------------------------------------------------------------------
    def track(
        self, image: np.ndarray, tracker: str = "bytetrack.yaml", persist: bool = True
    ) -> List[CattleDetection]:
        results = self.model.track(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.cattle_classes,
            device=self.device,
            tracker=tracker,
            persist=persist,
            verbose=False,
        )
        return self._parse(results[0], image.shape[:2])

    # ------------------------------------------------------------------
    def _parse(self, result, shape) -> List[CattleDetection]:
        h, w = shape
        out: List[CattleDetection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return out

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        ids = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes.id is not None
            else [None] * len(boxes_xyxy)
        )

        masks = None
        if getattr(result, "masks", None) is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()  # N x h' x w'

        # Factor de corrección bbox→silueta para vista cenital de bovinos.
        # Calibrado empíricamente con dataset real (vaquillonas Hereford/Angus
        # de ~260 kg): 0.69 da promedio coherente con balanza.
        BBOX_TO_SILHOUETTE = 0.69

        for i, (box, conf, tid) in enumerate(zip(boxes_xyxy, confs, ids)):
            mask = None
            bbox_area = float((box[2] - box[0]) * (box[3] - box[1]))

            if masks is not None and i < len(masks):
                # Modelo de segmentación: usamos máscara real (precisión máxima)
                m = masks[i]
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m_bool = m > 0.5
                mask = m_bool
                area_px = float(m_bool.sum())
            else:
                # Sin máscara: aplicamos el factor de corrección
                area_px = bbox_area * BBOX_TO_SILHOUETTE

            cx = float((box[0] + box[2]) / 2)
            cy = float((box[1] + box[3]) / 2)

            out.append(
                CattleDetection(
                    track_id=int(tid) if tid is not None else None,
                    bbox_xyxy=np.array(box, dtype=np.float32),
                    confidence=float(conf),
                    mask=mask,
                    area_px=area_px,
                    centroid_px=(cx, cy),
                )
            )
        return out
