"""Entidades de dominio del pronóstico de volatilidad, el riesgo y su calibración."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Horizon = Literal["1d", "7d"]
HORIZON_CANDLES: dict[str, int] = {"1d": 6, "7d": 42}
Regime = Literal["low", "normal", "high", "extreme"]
ValidationStatus = Literal["in_validation", "validated"]
Side = Literal["long", "short"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class PriceBand(_Frozen):
    low: float
    high: float


class PriceBands(_Frozen):
    """Rangos de precio al 68%, 90% y 95% (multiplicadores conformes, no gaussianos)."""

    p68: PriceBand
    p90: PriceBand
    p95: PriceBand


class StoredForecast(_Frozen):
    """Fila de `volatility_forecasts` expuesta al dominio."""

    id: int
    candle_id: int
    candle_open_time: datetime
    horizon_candles: int
    model_version: int
    requested_model_version: int
    fallback_reason: str | None
    forecast_var_annual: float
    forecast_vol_annual: float
    k68: float | None
    k90: float | None
    k95: float | None
    regime: Regime
    reference_price: float
    realized_var_annual: float | None = None
    realized_log_return: float | None = None
    inside_68: bool | None = None
    inside_90: bool | None = None
    inside_95: bool | None = None
    resolved_at: datetime | None = None


class ResolvedForecast(_Frozen):
    """Pronóstico ya resuelto, con el pronóstico ingenuo para compararlo (QLIKE)."""

    candle_open_time: datetime
    forecast_var_annual: float
    realized_var_annual: float
    qlike: float
    naive_var_annual: float | None
    inside_68: bool | None
    inside_90: bool | None
    inside_95: bool | None


class VolatilityForecast(_Frozen):
    """Pronóstico vigente de un horizonte."""

    horizon: Horizon
    as_of: datetime
    candle_open_time: datetime
    reference_price: float
    vol_annual: float
    vol_horizon_pct: float
    regime: Regime
    bands: PriceBands | None
    dvol: float | None
    realized_vol_7d: float | None
    realized_vol_30d: float | None
    model_version: int
    fallback: str | None
    validation_status: ValidationStatus


class VolatilityHistoryPoint(_Frozen):
    """Pronóstico frente a lo realizado, por vela."""

    candle_open_time: datetime
    as_of: datetime
    forecast_vol_annual: float
    realized_vol_annual: float | None
    regime: Regime
    model_version: int
    fallback: str | None
    resolved: bool
    inside_68: bool | None
    inside_90: bool | None
    inside_95: bool | None


class WindowCoverage(_Frozen):
    weeks: int
    resolved: int
    coverage_68: float | None
    coverage_90: float | None
    coverage_95: float | None


class VolatilityMetrics(_Frozen):
    """Calidad del pronóstico y cobertura real de las bandas."""

    horizon: Horizon
    resolved: int
    coverage_68: float | None
    coverage_90: float | None
    coverage_95: float | None
    qlike: float | None
    qlike_naive: float | None
    mz_slope: float | None
    last_12w: WindowCoverage
    validation_status: ValidationStatus


class PositionSize(_Frozen):
    """Tamaño de posición con stop basado en la volatilidad (herramienta educativa)."""

    horizon: Horizon
    side: Side
    reference_price: float
    vol_horizon_pct: float
    stop_distance_pct: float
    stop_price: float
    risk_usd: float
    position_usd: float
    position_btc: float
    exposure_pct: float
    regime: Regime
    warnings: list[str]
