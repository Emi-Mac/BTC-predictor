import math

import numpy as np
import pandas as pd
import pytest

from src.volatility import calibration as cal

STEP = pd.Timedelta(hours=4)


def times(n):
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"))


def test_conformal_quantile_is_the_ceil_n_plus_one_order_statistic():
    values = np.arange(1.0, 100.0)  # n = 99
    assert cal.conformal_quantile(values, 0.9) == 90.0  # ceil(100 * 0.9)
    assert cal.conformal_quantile(values, 0.68) == 68.0
    assert math.isnan(cal.conformal_quantile(values, 0.999))
    assert math.isnan(cal.conformal_quantile(np.array([]), 0.5))


def test_horizon_returns_and_standardized_residuals():
    close = pd.Series([100.0, 110.0, 121.0, 100.0], index=times(4))
    result = cal.horizon_log_returns(close, 2)
    assert result.iloc[0] == pytest.approx(math.log(1.21))
    assert result.iloc[1] == pytest.approx(math.log(100 / 110))
    assert result.iloc[2:].isna().all()
    z = cal.standardized_residual(np.array([-0.2, 0.1]), np.array([0.1, 0.0]))
    assert z[0] == pytest.approx(2.0)
    assert np.isfinite(z[1])


def test_rolling_multipliers_equal_pointwise_multipliers():
    rng = np.random.default_rng(0)
    n, horizon = 600, 6
    index = times(n)
    z = np.abs(rng.standard_t(5, n))
    kwargs = {"window": pd.Timedelta(days=60), "min_count": 40}
    rolling = cal.rolling_multipliers(index, z, horizon, **kwargs)
    issued = index + STEP
    resolved = issued + horizon * STEP
    for i in (10, 45, 100, 333, 599):
        expected = cal.multipliers_at(issued.iloc[i], issued, resolved, z, **kwargs)
        for level in cal.LEVELS:
            value = rolling[f"k{round(level * 100)}"].iloc[i]
            assert (math.isnan(value) and math.isnan(expected[level])) or value == expected[level]
    assert rolling.iloc[:40].isna().all().all()


def test_multipliers_use_only_resolved_forecasts_and_no_future_values():
    rng = np.random.default_rng(1)
    n, horizon = 500, 42
    index = times(n)
    z = np.abs(rng.normal(size=n))
    kwargs = {"window": pd.Timedelta(days=365), "min_count": 30}
    base = cal.rolling_multipliers(index, z, horizon, **kwargs)
    i = 300
    shifted = z.copy()
    shifted[i - horizon + 1 :] = 99.0  # aún sin resolver en `i`
    assert cal.rolling_multipliers(index, shifted, horizon, **kwargs).iloc[i].equals(base.iloc[i])
    shifted = z.copy()
    shifted[i - horizon] = 99.0  # recién resuelto en `i`
    changed = cal.rolling_multipliers(index, shifted, horizon, **kwargs).iloc[i]
    assert changed["k95"] >= base["k95"].iloc[i]


def test_window_drops_old_forecasts_and_gaps_are_rejected():
    n = 400
    index = times(n)
    z = np.concatenate([np.full(200, 50.0), np.ones(200)])
    kwargs = {"window": pd.Timedelta(days=20), "min_count": 10}
    result = cal.rolling_multipliers(index, z, 6, **kwargs)
    assert result["k95"].iloc[390] == 1.0  # los z antiguos quedaron fuera de la ventana
    with pytest.raises(ValueError, match="contigua"):
        cal.rolling_multipliers(index.drop(5), z[:-1], 6, **kwargs)


def test_coverage_is_nan_without_multipliers_and_near_nominal_otherwise():
    rng = np.random.default_rng(2)
    n, horizon = 6000, 6
    z = np.abs(rng.normal(size=n))
    result = cal.rolling_multipliers(times(n), z, horizon, min_count=500)
    inside = cal.coverage(z, result)
    assert inside.iloc[:100].isna().all().all()
    evaluated = inside.notna().all(axis=1)
    for level in cal.LEVELS:
        assert inside.loc[evaluated, f"inside_{round(level * 100)}"].mean() == pytest.approx(
            level, abs=0.03
        )
    summary = cal.summarize_levels(inside[evaluated], block=42, seed=1)
    assert all(entry["ci_contains_nominal"] for entry in summary.values())


def test_price_band_is_log_symmetric_and_validates_inputs():
    low, high = cal.price_band(100.0, 0.1, 1.5)
    assert high == pytest.approx(100 * math.exp(0.15))
    assert low * high == pytest.approx(100.0**2)
    assert cal.price_band(100.0, 0.0, 2.0) == (100.0, 100.0)
    for args in ((0.0, 0.1, 1.0), (100.0, -0.1, 1.0), (100.0, 0.1, -1.0)):
        with pytest.raises(ValueError, match="no negativos"):
            cal.price_band(*args)
