"""Backtest económico simple de señales direccionales con costos de transacción."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.training.types import FloatArray

PERIODS_PER_YEAR_4H = 6 * 365
# Taker de futuros (~5 pb) + deslizamiento (~2 pb) por unidad de rotación; 0 y 15 como sensibilidad.
COST_SCENARIOS_BPS = (0.0, 7.0, 15.0)


def positions_from_probabilities(
    probabilities: FloatArray, margin: float = 0.0, allow_short: bool = True
) -> FloatArray:
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilidades fuera de [0, 1]")
    if not 0 <= margin < 0.5:
        raise ValueError("El margen debe estar en [0, 0.5)")
    long = p > 0.5 + margin
    short = p < 0.5 - margin
    result: FloatArray = np.where(long, 1.0, np.where(short, -1.0 if allow_short else 0.0, 0.0))
    return result


def overlapping_positions(signals: FloatArray, horizon: int) -> FloatArray:
    """Cartera solapada: cada señal compromete 1/h del capital durante h velas.

    La posición en t promedia las señales de t-h+1..t; antes del inicio cuentan como 0.
    """
    s = np.asarray(signals, dtype=float)
    if horizon < 1 or not len(s) or not np.isfinite(s).all():
        raise ValueError("Horizonte o señales inválidos")
    cumulative = np.concatenate([[0.0], np.cumsum(s)])
    index = np.arange(1, len(s) + 1)
    result: FloatArray = (cumulative[index] - cumulative[np.maximum(index - horizon, 0)]) / horizon
    return result


def strategy_returns(
    positions: FloatArray,
    forward_returns: FloatArray,
    cost_bps: float,
    carry: FloatArray | None = None,
) -> FloatArray:
    """Retorno neto por periodo: posición × retorno siguiente − costo por cambio de posición.

    Supone un único libro cronológico y continuo: cobra la entrada inicial y cada rebalanceo.
    `carry` es el costo por periodo y unidad de posición larga (p. ej. funding de perpetuos;
    las posiciones cortas lo reciben cuando es positivo).
    """
    pos = np.asarray(positions, dtype=float)
    ret = np.asarray(forward_returns, dtype=float)
    if pos.shape != ret.shape or not len(pos):
        raise ValueError("Posiciones y retornos deben tener la misma longitud no nula")
    if not np.isfinite(ret).all() or (ret <= -1).any():
        raise ValueError("Retornos inválidos")
    if cost_bps < 0:
        raise ValueError("El costo no puede ser negativo")
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    result: FloatArray = pos * ret - turnover * cost_bps / 10_000
    if carry is not None:
        rate = np.asarray(carry, dtype=float)
        if rate.shape != pos.shape or not np.isfinite(rate).all():
            raise ValueError("El carry debe ser finito y alinear con las posiciones")
        result = result - pos * rate
    return result


def performance(
    returns: FloatArray,
    positions: FloatArray | None = None,
    periods_per_year: int = PERIODS_PER_YEAR_4H,
) -> dict[str, Any]:
    r = np.asarray(returns, dtype=float)
    if not len(r) or not np.isfinite(r).all():
        raise ValueError("Se requieren retornos finitos")
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    gains, losses = float(r[r > 0].sum()), float(-r[r < 0].sum())
    years = len(r) / periods_per_year
    total = float(equity[-1] - 1)
    result: dict[str, Any] = {
        "periods": len(r),
        "total_return": total,
        "annualized_return": float(equity[-1] ** (1 / years) - 1) if equity[-1] > 0 else -1.0,
        "annualized_volatility": std * np.sqrt(periods_per_year),
        "sharpe": float(r.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0,
        "max_drawdown": float((equity / peak - 1).min()),
        "profit_factor": gains / losses if losses > 0 else None,
    }
    if positions is not None:
        pos = np.asarray(positions, dtype=float)
        active = pos != 0
        result["exposure"] = float(active.mean())
        result["hit_rate_active"] = float((r[active] > 0).mean()) if active.any() else None
        result["turnover_per_period"] = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).mean())
    return result


def backtest(
    probabilities: FloatArray,
    forward_returns: FloatArray,
    margins: Sequence[float] = (0.0, 0.02, 0.05),
    costs_bps: Sequence[float] = COST_SCENARIOS_BPS,
    allow_short: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for margin in margins:
        pos = positions_from_probabilities(probabilities, margin, allow_short)
        for cost in costs_bps:
            net = strategy_returns(pos, forward_returns, cost)
            rows.append({"margin": float(margin), "cost_bps": float(cost), **performance(net, pos)})
    return rows
