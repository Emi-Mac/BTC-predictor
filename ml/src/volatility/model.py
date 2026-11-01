"""Modelos lineales de volatilidad (HAR / HAR-X) como artefactos JSON autocontenidos.

El mismo código ajusta (entrenamiento y evaluación walk-forward) y predice (producción), por
lo que el pronóstico servido coincide con el evaluado. Todo se calcula en float64.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.economic import PERIODS_PER_YEAR_4H
from src.volatility.features import HAR_COLUMNS, HAR_X_LIVE_EXTRA, HORIZONS

if TYPE_CHECKING:
    from pathlib import Path

    from src.training.types import FloatArray

SCHEMA_VERSION = 1
KINDS = {
    "har": tuple(HAR_COLUMNS),
    "har_x_live": (*HAR_COLUMNS, *HAR_X_LIVE_EXTRA),
}
REGIMES = ("low", "normal", "high", "extreme")
REGIME_PERCENTILES = (25.0, 75.0, 90.0)


@dataclass(frozen=True)
class VolatilityArtifact:
    kind: str
    horizon_candles: int
    feature_names: list[str]
    median: list[float]
    mean: list[float]
    scale: list[float]
    intercept: float
    coefficients: list[float]
    residual_variance: float
    regime_thresholds: dict[str, float]
    training_rows: int
    trained_through: str
    schema_version: int = SCHEMA_VERSION
    calibration_seed: dict[str, list[Any]] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def problems(self) -> list[str]:
        """Motivos por los que el artefacto no puede servirse (lista vacía si es válido)."""
        issues: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            issues.append(f"schema_version {self.schema_version} no soportado")
        if self.kind not in KINDS:
            issues.append(f"kind desconocido: {self.kind}")
        elif tuple(self.feature_names) != KINDS[self.kind]:
            issues.append("feature_names no corresponde al kind")
        if self.horizon_candles not in HORIZONS.values():
            issues.append(f"horizonte no soportado: {self.horizon_candles}")
        width = len(self.feature_names)
        if not all(
            len(v) == width for v in (self.median, self.mean, self.scale, self.coefficients)
        ):
            issues.append("las dimensiones de los parámetros no coinciden")
        numbers = [
            *self.median,
            *self.mean,
            *self.scale,
            *self.coefficients,
            self.intercept,
            self.residual_variance,
        ]
        if not all(math.isfinite(value) for value in numbers):
            issues.append("hay parámetros no finitos")
        if any(value <= 0 for value in self.scale):
            issues.append("hay escalas no positivas")
        if self.residual_variance < 0:
            issues.append("la varianza residual es negativa")
        low, high, extreme = (self.regime_thresholds.get(k) for k in ("low", "high", "extreme"))
        if low is None or high is None or extreme is None or not low < high < extreme:
            issues.append("los umbrales de régimen deben ser crecientes")
        return issues

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> VolatilityArtifact:
        return cls(**json.loads(text))

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> VolatilityArtifact:
        return cls.from_json(path.read_text(encoding="utf-8"))


def standardize(artifact: VolatilityArtifact, frame: pd.DataFrame) -> FloatArray:
    values = frame[artifact.feature_names].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Las features contienen infinitos")
    imputed = np.where(np.isnan(values), np.asarray(artifact.median), values)
    result: FloatArray = (imputed - np.asarray(artifact.mean)) / np.asarray(artifact.scale)
    return result


def predict_log_variance(artifact: VolatilityArtifact, frame: pd.DataFrame) -> FloatArray:
    result: FloatArray = standardize(artifact, frame) @ np.asarray(artifact.coefficients)
    return result + artifact.intercept


def predict_variance(artifact: VolatilityArtifact, frame: pd.DataFrame) -> FloatArray:
    """Varianza anualizada con corrección de sesgo lognormal."""
    result: FloatArray = np.exp(
        predict_log_variance(artifact, frame) + 0.5 * artifact.residual_variance
    )
    return result


def fit_artifact(
    kind: str, train: pd.DataFrame, horizon: str, regime_reference: pd.DataFrame | None = None
) -> VolatilityArtifact:
    """Ajusta el modelo lineal; los umbrales de régimen salen de `regime_reference` (o train)."""
    if kind not in KINDS:
        raise ValueError(f"kind desconocido: {kind}")
    names = list(KINDS[kind])
    values = train[names].to_numpy(dtype=float)
    if not len(values) or np.isinf(values).any() or np.isnan(values).all(axis=0).any():
        raise ValueError("Train vacío, con infinitos o alguna feature totalmente ausente")
    imputer = SimpleImputer(strategy="median").fit(values)
    imputed = imputer.transform(values)
    scaler = StandardScaler().fit(imputed)
    x = scaler.transform(imputed)
    y = train[f"log_target_{horizon}"].to_numpy(dtype=float)
    estimator = LinearRegression().fit(x, y)
    residual = y - estimator.predict(x)
    draft = VolatilityArtifact(
        kind=kind,
        horizon_candles=HORIZONS[horizon],
        feature_names=names,
        median=[float(v) for v in imputer.statistics_],
        mean=[float(v) for v in scaler.mean_],
        scale=[float(v) for v in scaler.scale_],
        intercept=float(estimator.intercept_),
        coefficients=[float(v) for v in estimator.coef_],
        residual_variance=float(np.var(residual)),
        regime_thresholds={"low": 0.0, "high": 1.0, "extreme": 2.0},
        training_rows=len(train),
        trained_through=pd.Timestamp(train["open_time"].max()).isoformat(),
    )
    reference = train if regime_reference is None else regime_reference
    thresholds = regime_thresholds(np.sqrt(predict_variance(draft, reference)))
    return VolatilityArtifact(**{**asdict(draft), "regime_thresholds": thresholds})


def regime_thresholds(vol_annual: FloatArray) -> dict[str, float]:
    """Percentiles (p25, p75, p90) de la volatilidad anualizada pronosticada."""
    low, high, extreme = (float(v) for v in np.percentile(vol_annual, REGIME_PERCENTILES))
    return {"low": low, "high": high, "extreme": extreme}


def regime(vol_annual: float, thresholds: dict[str, float]) -> str:
    """Régimen de riesgo por percentil histórico del pronóstico (descriptivo, no una señal)."""
    if vol_annual < thresholds["low"]:
        return "low"
    if vol_annual < thresholds["high"]:
        return "normal"
    if vol_annual < thresholds["extreme"]:
        return "high"
    return "extreme"


def horizon_sigma(var_annual: FloatArray, horizon_candles: int) -> FloatArray:
    """Desviación estándar del retorno log a `horizon_candles` velas de 4h."""
    result: FloatArray = np.sqrt(np.maximum(var_annual, 0) * horizon_candles / PERIODS_PER_YEAR_4H)
    return result
