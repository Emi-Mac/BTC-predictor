"""Servicio de volatilidad: implementa `VolatilityPort` sobre `VolatilityRepositoryPort`."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from src.domain.entities.pagination import Page, build_pagination
from src.domain.entities.volatility import (
    HORIZON_CANDLES,
    Horizon,
    PriceBand,
    PriceBands,
    ResolvedForecast,
    ValidationStatus,
    VolatilityForecast,
    VolatilityHistoryPoint,
    VolatilityMetrics,
    WindowCoverage,
)
from src.domain.exceptions import NotFoundError
from src.domain.ports.volatility_port import VolatilityPort
from src.domain.ports.volatility_repository_port import VolatilityRepositoryPort
from src.domain.volatility_math import (
    horizon_sigma,
    mincer_zarnowitz_slope,
    price_band,
    qlike,
    realized_volatility,
)

CLOSES_FOR_CONTEXT = 181  # 180 retornos: volatilidad realizada de 30 días
MIN_MZ_SAMPLES = 30
RECENT_WEEKS = 12
DVOL_AVAILABILITY_DELAY = timedelta(hours=1)  # una vela horaria solo existe al cerrar


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _coverage(flags: Sequence[bool | None]) -> float | None:
    known = [flag for flag in flags if flag is not None]
    return sum(known) / len(known) if known else None


class VolatilityService(VolatilityPort):
    """Casos de uso de consulta del pronóstico de volatilidad."""

    def __init__(
        self,
        repository: VolatilityRepositoryPort,
        validation_status: ValidationStatus = "in_validation",
        candle_step: timedelta = timedelta(hours=4),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._validation_status = validation_status
        self._step = candle_step
        self._clock = clock

    async def get_current(self, horizon: Horizon) -> VolatilityForecast:
        candles = HORIZON_CANDLES[horizon]
        stored = await self._repository.get_latest(candles)
        if stored is None:
            msg = f"No hay pronósticos de volatilidad a {horizon} todavía"
            raise NotFoundError(msg)
        as_of = stored.candle_open_time + self._step
        closes = await self._repository.recent_closes(stored.candle_open_time, CLOSES_FOR_CONTEXT)
        dvol = await self._repository.get_dvol(as_of - DVOL_AVAILABILITY_DELAY)
        sigma = horizon_sigma(stored.forecast_vol_annual, candles)
        return VolatilityForecast(
            horizon=horizon,
            as_of=as_of,
            candle_open_time=stored.candle_open_time,
            reference_price=stored.reference_price,
            vol_annual=stored.forecast_vol_annual,
            vol_horizon_pct=sigma * 100,
            regime=stored.regime,
            bands=self._bands(stored.reference_price, sigma, stored.k68, stored.k90, stored.k95),
            dvol=None if dvol is None else dvol / 100,
            realized_vol_7d=realized_volatility(closes, 42),
            realized_vol_30d=realized_volatility(closes, 180),
            model_version=stored.model_version,
            fallback=stored.fallback_reason,
            validation_status=self._validation_status,
        )

    @staticmethod
    def _bands(
        price: float, sigma: float, k68: float | None, k90: float | None, k95: float | None
    ) -> PriceBands | None:
        if k68 is None or k90 is None or k95 is None:
            return None  # sin calibración suficiente no se muestran bandas

        def band(k: float) -> PriceBand:
            low, high = price_band(price, sigma, k)
            return PriceBand(low=low, high=high)

        return PriceBands(p68=band(k68), p90=band(k90), p95=band(k95))

    async def get_history(
        self, horizon: Horizon, limit: int, offset: int
    ) -> Page[VolatilityHistoryPoint]:
        rows, total = await self._repository.list_history(HORIZON_CANDLES[horizon], limit, offset)
        points = [
            VolatilityHistoryPoint(
                candle_open_time=row.candle_open_time,
                as_of=row.candle_open_time + self._step,
                forecast_vol_annual=row.forecast_vol_annual,
                realized_vol_annual=(
                    None if row.realized_var_annual is None else math.sqrt(row.realized_var_annual)
                ),
                regime=row.regime,
                model_version=row.model_version,
                fallback=row.fallback_reason,
                resolved=row.resolved_at is not None,
                inside_68=row.inside_68,
                inside_90=row.inside_90,
                inside_95=row.inside_95,
            )
            for row in rows
        ]
        return Page(data=points, pagination=build_pagination(limit, offset, total, len(points)))

    async def get_metrics(self, horizon: Horizon) -> VolatilityMetrics:
        resolved = await self._repository.list_resolved(HORIZON_CANDLES[horizon])
        comparable = [row for row in resolved if row.naive_var_annual is not None]
        since = self._clock() - timedelta(weeks=RECENT_WEEKS)
        recent = [row for row in resolved if row.candle_open_time >= since]
        return VolatilityMetrics(
            horizon=horizon,
            resolved=len(resolved),
            coverage_68=_coverage([row.inside_68 for row in resolved]),
            coverage_90=_coverage([row.inside_90 for row in resolved]),
            coverage_95=_coverage([row.inside_95 for row in resolved]),
            qlike=_mean([row.qlike for row in comparable]),
            qlike_naive=_mean(
                [
                    qlike(row.realized_var_annual, row.naive_var_annual or 0.0)
                    for row in comparable
                ]
            ),
            mz_slope=self._mz_slope(resolved),
            last_12w=self._window(recent),
            validation_status=self._validation_status,
        )

    @staticmethod
    def _mz_slope(rows: Sequence[ResolvedForecast]) -> float | None:
        if len(rows) < MIN_MZ_SAMPLES:
            return None
        return mincer_zarnowitz_slope(
            [row.realized_var_annual for row in rows], [row.forecast_var_annual for row in rows]
        )

    @staticmethod
    def _window(rows: Sequence[ResolvedForecast]) -> WindowCoverage:
        return WindowCoverage(
            weeks=RECENT_WEEKS,
            resolved=len(rows),
            coverage_68=_coverage([row.inside_68 for row in rows]),
            coverage_90=_coverage([row.inside_90 for row in rows]),
            coverage_95=_coverage([row.inside_95 for row in rows]),
        )
