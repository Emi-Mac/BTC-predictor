"""Matemática de volatilidad del dominio (sin dependencias de infraestructura).

Replica las definiciones del ML worker (`ml/src/volatility`): 6·365 periodos de 4h por año,
retornos logarítmicos y bandas de precio `P · exp(±k·σ_h)`. Los tests de `test_volatility_math`
fijan los valores de referencia para que ambos lados no diverjan.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

PERIODS_PER_YEAR = 6 * 365
EPSILON = 1e-12


def horizon_sigma(vol_annual: float, horizon_candles: int) -> float:
    """Desviación estándar del retorno log a `horizon_candles` velas de 4h."""
    if vol_annual < 0 or horizon_candles <= 0:
        raise ValueError("La volatilidad no puede ser negativa y el horizonte debe ser positivo")
    return vol_annual * math.sqrt(horizon_candles / PERIODS_PER_YEAR)


def price_band(reference_price: float, sigma: float, k: float) -> tuple[float, float]:
    if reference_price <= 0 or sigma < 0 or k < 0:
        raise ValueError("Precio positivo; sigma y k no negativos")
    return reference_price * math.exp(-k * sigma), reference_price * math.exp(k * sigma)


def realized_variance(closes: Sequence[float]) -> float | None:
    """Varianza anualizada de los retornos log de cierres consecutivos (None si no hay datos)."""
    if len(closes) < 2 or any(c <= 0 or not math.isfinite(c) for c in closes):
        return None
    squares = [math.log(b / a) ** 2 for a, b in zip(closes, closes[1:], strict=False)]
    return sum(squares) / len(squares) * PERIODS_PER_YEAR


def realized_volatility(closes: Sequence[float], window: int) -> float | None:
    """Volatilidad anualizada de los últimos `window` retornos (None si faltan velas)."""
    if len(closes) < window + 1:
        return None
    variance = realized_variance(closes[-(window + 1) :])
    return None if variance is None else math.sqrt(variance)


def qlike(realized_var: float, forecast_var: float) -> float:
    ratio = max(realized_var / max(forecast_var, EPSILON), EPSILON)
    return ratio - math.log(ratio) - 1


def mincer_zarnowitz_slope(realized: Sequence[float], forecast: Sequence[float]) -> float | None:
    """Pendiente de regresar log(varianza realizada) sobre log(pronóstico); ideal ≈ 1."""
    if len(realized) != len(forecast) or len(realized) < 2:
        return None
    y = [math.log(max(r, EPSILON)) for r in realized]
    x = [math.log(max(f, EPSILON)) for f in forecast]
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    spread = sum((a - mean_x) ** 2 for a in x)
    if spread < EPSILON:
        return None
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)) / spread
