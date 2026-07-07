"""
Estimador de peso bovino a partir del área proyectada (vista cenital).

Modelo principal (alométrico):

        Peso (kg) = a · Area_m2^b · factor_raza + c

Esta forma funcional sigue la ley alométrica clásica (Huxley) y ha sido
validada en literatura veterinaria para predecir peso vivo a partir de
medidas morfométricas. Para vacunos vistos desde arriba, los coeficientes
típicos son a≈1100-1300 y b≈1.10-1.25, dependiendo de la raza.

Modelos alternativos:
  - "lineal":     Peso = a · Area + c
  - "polinomial": Peso = a · Area² + b · Area + c

CRÍTICO: hay que CALIBRAR los coeficientes con tu propio dataset
(imagen + peso de balanza) para alcanzar el <5% de error que pide el
proyecto. Ver scripts/calibrate_weight.py.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class WeightModel:
    modelo: str = "potencia"
    coef_a: float = 220.0      # kg / m^(2b)
    coef_b: float = 1.20
    offset_c: float = 0.0
    factores_raza: Dict[str, float] = field(
        default_factory=lambda: {
            "angus": 1.00,
            "hereford": 0.97,
            "brangus": 1.05,
            "braford": 1.03,
            "cruza": 1.00,
            "desconocido": 1.00,
        }
    )
    # Factor por categoría/edad: ajusta el modelo según el tipo de animal
    factores_categoria: Dict[str, float] = field(
        default_factory=lambda: {
            "ternero": 0.85,       # 80-180 kg
            "vaquillona": 0.92,    # 250-380 kg, hembra joven
            "novillo": 0.98,       # 400-500 kg, macho joven
            "vaca_adulta": 1.00,   # 400-650 kg
            "toro": 1.10,          # 600-1000 kg
            "desconocido": 1.00,
        }
    )
    area_min_m2: float = 0.5
    area_max_m2: float = 3.5
    peso_min_kg: float = 80
    peso_max_kg: float = 1100

    @classmethod
    def from_config(cls, cfg: dict) -> "WeightModel":
        we = cfg.get("estimacion_peso", {})
        return cls(
            modelo=we.get("modelo", "potencia"),
            coef_a=float(we.get("coef_a", 1180.0)),
            coef_b=float(we.get("coef_b", 1.18)),
            offset_c=float(we.get("offset_c", 0.0)),
            factores_raza=we.get("factores_raza", {}),
            area_min_m2=float(we.get("area_min_m2", 0.6)),
            area_max_m2=float(we.get("area_max_m2", 4.5)),
            peso_min_kg=float(we.get("peso_min_kg", 80)),
            peso_max_kg=float(we.get("peso_max_kg", 900)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "WeightModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------------
    def estimate(
        self,
        area_m2: float,
        raza: str = "desconocido",
        categoria: str = "desconocido",
        ajuste_fino: float = 1.0,
    ) -> Optional[float]:
        """Devuelve el peso estimado en kg, o None si está fuera de rango."""
        if area_m2 <= 0 or area_m2 < self.area_min_m2 or area_m2 > self.area_max_m2:
            return None

        f_raza = self.factores_raza.get(raza.lower(), 1.0)
        f_cat = self.factores_categoria.get(categoria.lower(), 1.0)

        if self.modelo == "lineal":
            peso = self.coef_a * area_m2 + self.offset_c
        elif self.modelo == "polinomial":
            peso = self.coef_a * area_m2 ** 2 + self.coef_b * area_m2 + self.offset_c
        else:  # potencia (default)
            peso = self.coef_a * (area_m2 ** self.coef_b) + self.offset_c

        peso *= f_raza * f_cat * ajuste_fino

        if peso < self.peso_min_kg or peso > self.peso_max_kg:
            return None
        return float(peso)


# ----------------------------------------------------------------------
# Calibración con dataset propio
# ----------------------------------------------------------------------


def calibrate_power_model(
    areas_m2: np.ndarray,
    pesos_kg: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Ajusta Peso = a * Area^b por mínimos cuadrados en log-log.
    Devuelve (a, b, RMSE_relativo_pct).
    """
    if len(areas_m2) < 5:
        raise ValueError("Necesito al menos 5 muestras para calibrar.")

    log_a = np.log(np.asarray(areas_m2, dtype=float))
    log_w = np.log(np.asarray(pesos_kg, dtype=float))

    # log W = log a + b log A
    X = np.column_stack([np.ones_like(log_a), log_a])
    coefs, *_ = np.linalg.lstsq(X, log_w, rcond=None)
    log_a_coef, b = coefs
    a = math.exp(log_a_coef)

    pred = a * (np.asarray(areas_m2) ** b)
    err_rel = np.abs(pred - pesos_kg) / pesos_kg
    rmse_rel_pct = float(np.sqrt(np.mean(err_rel ** 2)) * 100)
    return float(a), float(b), rmse_rel_pct


def calibrate_linear_model(
    areas_m2: np.ndarray,
    pesos_kg: np.ndarray,
) -> Tuple[float, float, float]:
    """Ajusta Peso = a*Area + c. Devuelve (a, c, RMSE_relativo_pct)."""
    a, c = np.polyfit(areas_m2, pesos_kg, 1)
    pred = a * areas_m2 + c
    err_rel = np.abs(pred - pesos_kg) / pesos_kg
    rmse_rel_pct = float(np.sqrt(np.mean(err_rel ** 2)) * 100)
    return float(a), float(c), rmse_rel_pct


def evaluate_model(
    model: WeightModel,
    areas_m2: np.ndarray,
    pesos_kg: np.ndarray,
    razas: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evalúa el modelo con métricas estándar."""
    if razas is None:
        razas = ["desconocido"] * len(areas_m2)

    preds = np.array([
        model.estimate(a, r) or np.nan for a, r in zip(areas_m2, razas)
    ])
    valid = ~np.isnan(preds)
    if valid.sum() == 0:
        return {"n_validos": 0}

    p = preds[valid]
    w = np.asarray(pesos_kg)[valid]
    err = p - w
    err_rel = err / w

    return {
        "n_validos": int(valid.sum()),
        "n_total": int(len(areas_m2)),
        "mae_kg": float(np.mean(np.abs(err))),
        "rmse_kg": float(np.sqrt(np.mean(err ** 2))),
        "mape_pct": float(np.mean(np.abs(err_rel)) * 100),
        "max_err_pct": float(np.max(np.abs(err_rel)) * 100),
        "r2": float(1 - np.var(err) / np.var(w)),
    }
