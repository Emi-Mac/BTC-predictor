"""Calculadora de tamaño de posición con stop basado en la volatilidad pronosticada.

Herramienta educativa, sin estado: no emite señales ni órdenes. El tamaño sale de arriesgar un
porcentaje fijo del capital hasta un stop situado a `stop_k` desviaciones del horizonte.
"""
from __future__ import annotations

import math

from src.domain.entities.volatility import Horizon, PositionSize, Side
from src.domain.exceptions import InvalidParameterError
from src.domain.ports.risk_port import RiskPort
from src.domain.ports.volatility_port import VolatilityPort

MAX_RISK_PCT = 5.0
MAX_STOP_K = 10.0
AGGRESSIVE_RISK_PCT = 2.0
MIN_RECOMMENDED_STOP_K = 1.0
MIN_STOP_DISTANCE = 1e-9


def _validate(capital: float, risk_pct: float, stop_k: float) -> None:
    if not math.isfinite(capital) or capital <= 0:
        raise InvalidParameterError("El capital debe ser positivo")
    if not math.isfinite(risk_pct) or not 0 < risk_pct <= MAX_RISK_PCT:
        raise InvalidParameterError(f"El riesgo por operación debe estar en (0, {MAX_RISK_PCT}]%")
    if not math.isfinite(stop_k) or not 0 < stop_k <= MAX_STOP_K:
        raise InvalidParameterError(f"El multiplicador del stop debe estar en (0, {MAX_STOP_K}]")


class RiskService(RiskPort):
    def __init__(self, volatility: VolatilityPort) -> None:
        self._volatility = volatility

    async def position_size(
        self, capital: float, risk_pct: float, stop_k: float, horizon: Horizon, side: Side
    ) -> PositionSize:
        _validate(capital, risk_pct, stop_k)
        forecast = await self._volatility.get_current(horizon)
        sigma = forecast.vol_horizon_pct / 100
        # Stop log-simétrico: cae a P·exp(−kσ) (largo) o sube a P·exp(+kσ) (corto).
        move = stop_k * sigma
        stop_price = forecast.reference_price * math.exp(-move if side == "long" else move)
        distance = abs(stop_price / forecast.reference_price - 1)
        if distance < MIN_STOP_DISTANCE:
            raise InvalidParameterError("El stop resultante es nulo; aumenta el multiplicador")
        risk_usd = capital * risk_pct / 100
        position_usd = risk_usd / distance
        return PositionSize(
            horizon=horizon,
            side=side,
            reference_price=forecast.reference_price,
            vol_horizon_pct=forecast.vol_horizon_pct,
            stop_distance_pct=distance * 100,
            stop_price=stop_price,
            risk_usd=risk_usd,
            position_usd=position_usd,
            position_btc=position_usd / forecast.reference_price,
            exposure_pct=position_usd / capital * 100,
            regime=forecast.regime,
            warnings=self._warnings(
                position_usd / capital * 100, risk_pct, stop_k, forecast.regime,
                forecast.fallback, forecast.validation_status,
            ),
        )

    @staticmethod
    def _warnings(
        exposure_pct: float,
        risk_pct: float,
        stop_k: float,
        regime: str,
        fallback: str | None,
        validation_status: str,
    ) -> list[str]:
        warnings: list[str] = []
        if exposure_pct > 100:
            warnings.append(
                "La posición supera el capital: requeriría apalancamiento. "
                "Reduce el riesgo o amplía el stop."
            )
        if risk_pct > AGGRESSIVE_RISK_PCT:
            warnings.append("Arriesgar más de 2% del capital por operación es agresivo.")
        if stop_k < MIN_RECOMMENDED_STOP_K:
            warnings.append(
                "Un stop a menos de 1σ se activa con frecuencia por ruido normal del mercado."
            )
        if regime in ("high", "extreme"):
            warnings.append(
                f"Régimen de volatilidad {regime}: los movimientos esperados son mayores de lo normal."
            )
        if fallback is not None:
            warnings.append(f"Pronóstico de respaldo (sin variables exógenas): {fallback}.")
        if validation_status == "in_validation":
            warnings.append("El pronóstico de volatilidad sigue en validación.")
        return warnings
