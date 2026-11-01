"""Calibración conforme de las bandas de precio, sin fuga ni tautología.

Estadístico: z = |log(P[t+h] / P[t])| / σ̂_h, con σ̂_h la desviación estándar pronosticada a h
velas. Para un pronóstico emitido en `t`, el multiplicador `k_q` es el cuantil conforme `q` de z
entre los pronósticos cuya resolución ocurrió antes de `t` (`t' + h ≤ t`), dentro de una ventana
móvil. La banda es `P[t] · exp(±k_q · σ̂_h)`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.evaluation.robustness import block_bootstrap_mean
from src.volatility.features import STEP

if TYPE_CHECKING:
    from src.training.types import FloatArray

LEVELS = (0.68, 0.90, 0.95)
WINDOW = pd.Timedelta(days=730)
# Pronósticos resueltos mínimos para estimar multiplicadores (≈ 180 días de velas de 4h).
MIN_RESOLVED = 1080
GAUSSIAN_K = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96}


def horizon_log_returns(close: pd.Series, horizon: int) -> pd.Series:
    """log(close[t+h] / close[t]); NaN si falta alguna de las `h` velas siguientes."""
    result: pd.Series = np.log(close.shift(-horizon) / close)
    return result


def standardized_residual(log_return: FloatArray, sigma: FloatArray) -> FloatArray:
    result: FloatArray = np.abs(log_return) / np.maximum(sigma, 1e-12)
    return result


def conformal_quantile(values: FloatArray, level: float) -> float:
    """Cuantil conforme: el ⌈(n+1)·q⌉-ésimo menor valor (NaN si no hay datos suficientes)."""
    n = len(values)
    rank = math.ceil((n + 1) * level)
    if n == 0 or rank > n:
        return float("nan")
    return float(np.partition(values, rank - 1)[rank - 1])


def multipliers_at(
    as_of: pd.Timestamp,
    issued: pd.Series,
    resolved: pd.Series,
    z: FloatArray,
    levels: tuple[float, ...] = LEVELS,
    window: pd.Timedelta = WINDOW,
    min_count: int = MIN_RESOLVED,
) -> dict[float, float]:
    """Multiplicadores para un pronóstico emitido en `as_of` (uso en producción).

    `issued` y `resolved` son los instantes de emisión y de resolución de cada pronóstico
    pasado; solo cuentan los resueltos hasta `as_of` y emitidos dentro de la ventana.
    """
    mask = (
        (pd.to_datetime(resolved, utc=True) <= as_of)
        & (pd.to_datetime(issued, utc=True) > as_of - window)
        & np.isfinite(z)
    )
    values = np.asarray(z)[mask.to_numpy()]
    if len(values) < min_count:
        return {level: float("nan") for level in levels}
    return {level: conformal_quantile(values, level) for level in levels}


def rolling_multipliers(
    open_times: pd.Series,
    z: FloatArray,
    horizon: int,
    levels: tuple[float, ...] = LEVELS,
    window: pd.Timedelta = WINDOW,
    min_count: int = MIN_RESOLVED,
) -> pd.DataFrame:
    """Multiplicadores de cada fila usando solo pronósticos ya resueltos (serie contigua).

    Equivale a `multipliers_at` para cada fila, pero vectorizado sobre una rejilla de 4h
    contigua: el pronóstico `j` está resuelto en `i` cuando `j + h ≤ i`.
    """
    times = pd.to_datetime(open_times, utc=True).reset_index(drop=True)
    if len(times) > 1 and (times.diff().dropna() != STEP).any():
        raise ValueError("Se requiere una rejilla contigua de 4h")
    values = np.asarray(z, dtype=float)
    first_in_window = np.searchsorted(times.to_numpy(), (times - window).to_numpy(), side="right")
    out = np.full((len(times), len(levels)), np.nan)
    for i in range(horizon, len(times)):
        sample = values[first_in_window[i] : i - horizon + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) < min_count:
            continue
        for column, level in enumerate(levels):
            out[i, column] = conformal_quantile(sample, level)
    return pd.DataFrame(out, columns=[f"k{round(level * 100)}" for level in levels])


def coverage(
    z: FloatArray, multipliers: pd.DataFrame, levels: tuple[float, ...] = LEVELS
) -> pd.DataFrame:
    """Indicadores de cobertura por nivel (NaN donde no hay multiplicador)."""
    out = pd.DataFrame(index=multipliers.index)
    for level in levels:
        k = multipliers[f"k{round(level * 100)}"].to_numpy()
        inside = np.where(np.isnan(k), np.nan, (np.asarray(z) <= k).astype(float))
        out[f"inside_{round(level * 100)}"] = inside
    return out


def price_band(reference_price: float, sigma: float, k: float) -> tuple[float, float]:
    """Banda de precio `P · exp(±k·σ)`."""
    if not (reference_price > 0 and sigma >= 0 and k >= 0):
        raise ValueError("Precio, sigma y k deben ser no negativos (precio positivo)")
    return reference_price * math.exp(-k * sigma), reference_price * math.exp(k * sigma)


def summarize_levels(inside: pd.DataFrame, block: int, seed: int) -> dict[str, Any]:
    """Cobertura global con IC por bloques para cada nivel."""
    result: dict[str, Any] = {}
    for level in LEVELS:
        column = f"inside_{round(level * 100)}"
        values = inside[column].dropna().to_numpy(dtype=float)
        entry = block_bootstrap_mean(values, block, seed=seed)
        entry.update({"nominal": level, "n": len(values)})
        entry["ci_contains_nominal"] = bool(entry["ci_95"][0] <= level <= entry["ci_95"][1])
        result[column] = entry
    return result
