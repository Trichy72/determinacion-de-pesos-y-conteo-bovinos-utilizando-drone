"""
Helpers para el flujo de entrenamiento del modelo de drone.

1) Extracción de frames desde uno o varios videos (con muestreo regular)
2) Calibración automática del ajuste_fino comparando pesadas reales vs
   pesadas estimadas por la app
3) Validación de modelos .pt importados
4) Generación de paquetes listos para etiquetar en Roboflow

Estos helpers son consumidos por la pestaña "🎓 Entrenamiento avanzado"
de la app, así Mauricio (o cualquier asesor) puede armar su propio modelo
sin tocar línea de comandos.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =====================================================================
# 1) EXTRACCIÓN DE FRAMES
# =====================================================================

def extraer_frames_de_video(
    video_bytes: bytes,
    fps_objetivo: float = 1.0,
    max_frames: int = 100,
) -> List[Dict]:
    """
    Extrae frames de un video tomando un cuadro cada `fps_objetivo` segundos.
    Devuelve lista de dicts {nombre, bytes_jpg, segundo}.
    """
    # Guardar bytes a archivo temporal porque OpenCV no abre desde memoria
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        path_tmp = tmp.name

    cap = cv2.VideoCapture(path_tmp)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(round(fps / fps_objetivo)))

    frames = []
    fidx = 0
    extracted = 0
    while extracted < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % interval == 0:
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok2:
                frames.append({
                    "nombre": f"frame_{fidx:06d}.jpg",
                    "bytes": buf.tobytes(),
                    "segundo": fidx / fps,
                })
                extracted += 1
        fidx += 1
    cap.release()
    Path(path_tmp).unlink(missing_ok=True)
    return frames


def crear_zip_dataset(frames: List[Dict],
                      nombre_proyecto: str = "bovinos_drone") -> bytes:
    """Empaqueta los frames extraídos en un ZIP listo para subir a Roboflow."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in frames:
            zf.writestr(f"{nombre_proyecto}/images/{f['nombre']}", f["bytes"])
        # README con instrucciones
        readme = f"""# Dataset {nombre_proyecto}

Este ZIP contiene {len(frames)} frames extraídos de videos de drone.

## Cómo usarlo en Roboflow

1. Entrá a https://roboflow.com (gratis hasta 10.000 imágenes)
2. Crear proyecto → "Object Detection" → modelo "YOLOv8"
3. Drag & drop este ZIP completo
4. Auto-Label: usá "Train Auto-Label Model" para que pre-etiquete las
   imágenes con clase "cattle" / "cow" automáticamente
5. Revisá y corregí cada frame:
   - Confirmar que cada animal tenga su caja
   - Borrar falsos positivos (sombras, manchas)
   - Usar UNA SOLA CLASE: "bovino" (renombrá si auto-labeled las puso
     como "cow", "sheep", etc.)
6. Generar dataset → exportar formato YOLOv8
7. Bajá el ZIP del dataset etiquetado
8. Subilo al notebook de Colab para entrenar

Frames extraídos: {len(frames)}
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        zf.writestr(f"{nombre_proyecto}/README.md", readme)
    buf.seek(0)
    return buf.read()


# =====================================================================
# 2) CALIBRACIÓN AUTOMÁTICA DE AJUSTE_FINO
# =====================================================================

@dataclass
class ResultadoCalibracion:
    ajuste_fino_optimo: float
    n_muestras: int
    mape_actual: float       # MAPE con ajuste_fino actual
    mape_optimo: float       # MAPE con ajuste_fino calibrado
    r2: float                # coef. determinación
    sesgo_kg: float          # diferencia promedio (positiva = app sobreestima)
    pesos_reales: List[float]
    pesos_app: List[float]
    pesos_corregidos: List[float]
    rangos_confianza: Tuple[float, float]   # límites en ±2σ del peso individual


def calibrar_ajuste_fino(
    pares: List[Tuple[float, float]],
    ajuste_actual: float = 1.0,
) -> ResultadoCalibracion:
    """
    Calibra el factor `ajuste_fino` que minimiza el error entre el peso
    real (balanza) y el peso estimado por la app.

    `pares`: lista de tuplas (peso_real_kg, peso_app_kg) — al menos 5 pares.
    `ajuste_actual`: el valor actual del slider para poder mostrar la mejora.

    Devuelve métricas + el ajuste_fino óptimo que minimiza el MAPE.
    """
    if len(pares) < 3:
        raise ValueError("Se necesitan al menos 3 pares para calibrar.")

    pr = np.array([p[0] for p in pares], dtype=float)
    pa = np.array([p[1] for p in pares], dtype=float)

    if (pa <= 0).any() or (pr <= 0).any():
        raise ValueError("Los pesos deben ser positivos.")

    # MAPE con ajuste actual
    pesos_actuales = pa * ajuste_actual
    mape_actual = float(np.mean(np.abs(pesos_actuales - pr) / pr) * 100)

    # Ajuste óptimo: minimizar MAPE en escala lineal
    # Si peso_app * k = peso_real → k = peso_real / peso_app
    # El óptimo es el ratio mediano (más robusto a outliers que el promedio)
    ratios = pr / pa
    k_optimo = float(np.median(ratios))

    pesos_corregidos = pa * k_optimo
    errores = pesos_corregidos - pr
    mape_optimo = float(np.mean(np.abs(errores) / pr) * 100)
    sesgo = float(np.mean(errores))

    # R²
    ss_res = float(np.sum(errores ** 2))
    ss_tot = float(np.sum((pr - np.mean(pr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Rango de confianza individual: 2σ
    desv = float(np.std(errores))
    rango_inf = float(np.mean(errores) - 2 * desv)
    rango_sup = float(np.mean(errores) + 2 * desv)

    return ResultadoCalibracion(
        ajuste_fino_optimo=k_optimo,
        n_muestras=len(pares),
        mape_actual=mape_actual,
        mape_optimo=mape_optimo,
        r2=r2,
        sesgo_kg=sesgo,
        pesos_reales=pr.tolist(),
        pesos_app=pa.tolist(),
        pesos_corregidos=pesos_corregidos.tolist(),
        rangos_confianza=(rango_inf, rango_sup),
    )


# =====================================================================
# 3) VALIDACIÓN DE MODELOS IMPORTADOS
# =====================================================================

def validar_modelo_yolo(path: Path) -> Dict:
    """Carga un .pt importado y reporta info básica para confirmar que es
    un modelo YOLOv8 válido."""
    info = {"valido": False, "error": None, "clases": [], "tamano_mb": 0}
    if not path.exists():
        info["error"] = "Archivo no encontrado"
        return info
    info["tamano_mb"] = path.stat().st_size / (1024 * 1024)
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
        info["valido"] = True
        info["clases"] = list(model.names.values()) if model.names else []
        info["task"] = getattr(model, "task", "detect")
    except Exception as e:
        info["error"] = str(e)
    return info


# =====================================================================
# 4) GENERAR NOTEBOOK COLAB
# =====================================================================

def generar_notebook_colab(nombre_modelo: str = "bovino_finetuned") -> bytes:
    """Genera un .ipynb listo para usar en Google Colab con GPU gratis."""
    import json
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": "Fine-tuning YOLOv8 bovinos", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🐄 Fine-tuning YOLOv8 para detección de bovinos por drone\n",
                    "\n",
                    "**Pasos**:\n",
                    "1. Asegurate que el runtime tenga GPU: Entorno de ejecución → Cambiar tipo de entorno → T4 GPU\n",
                    "2. Subí tu dataset etiquetado de Roboflow (formato YOLOv8) en la sección de Files\n",
                    "3. Ejecutá las celdas en orden\n",
                    "4. Bajá el archivo `.pt` final y subilo a la app HMS",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Verificar GPU\n",
                    "!nvidia-smi"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Instalar ultralytics\n",
                    "!pip install -q ultralytics"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Subí el ZIP del dataset etiquetado de Roboflow al panel \"Files\" (ícono carpeta)\n",
                    "# Después corré esta celda para descomprimir\n",
                    "import os, zipfile\n",
                    "for f in os.listdir('.'):\n",
                    "    if f.endswith('.zip'):\n",
                    "        print(f'Descomprimiendo {f}...')\n",
                    "        with zipfile.ZipFile(f) as z:\n",
                    "            z.extractall('dataset')\n",
                    "        break\n",
                    "!ls dataset/"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Entrenamiento — ajustá epochs e imgsz si hace falta\n",
                    "from ultralytics import YOLO\n",
                    "\n",
                    "model = YOLO('yolov8m-seg.pt')   # arranca con modelo pre-entrenado\n",
                    "\n",
                    "results = model.train(\n",
                    "    data='dataset/data.yaml',\n",
                    "    epochs=100,\n",
                    "    imgsz=1280,\n",
                    "    batch=8,                    # bajar a 4 si la GPU se queda corta\n",
                    "    patience=20,\n",
                    "    project='runs',\n",
                    "    name='bovino_finetuned',\n",
                    "    # Augmentations clave para vista aérea cenital\n",
                    "    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,\n",
                    "    degrees=10, translate=0.1, scale=0.5, fliplr=0.5,\n",
                    "    mosaic=1.0,\n",
                    ")"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Evaluar el modelo\n",
                    "metrics = model.val()\n",
                    "print(f'mAP50: {metrics.box.map50:.3f}')\n",
                    "print(f'mAP50-95: {metrics.box.map:.3f}')"
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Descargar el modelo entrenado a tu computadora\n",
                    "from google.colab import files\n",
                    f"files.download('runs/bovino_finetuned/weights/best.pt')\n",
                    "print('✅ Listo. Subilo a la app HMS en la pestaña Entrenamiento avanzado.')"
                ],
            },
        ],
    }
    return json.dumps(notebook, indent=2, ensure_ascii=False).encode("utf-8")
