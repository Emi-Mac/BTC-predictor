import numpy as np
import pandas as pd
import pytest

from src.volatility import features as vf
from src.volatility.resolution import qlike_loss, realized_statistics, resolution_values


def test_realized_statistics_match_the_training_target_definition():
    rng = np.random.default_rng(4)
    times = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(times)))), index=times)
    frame = vf.volatility_frame(close, pd.Series(times))
    for t in (10, 120, 250):
        for name, horizon in vf.HORIZONS.items():
            variance, log_return = realized_statistics(close.iloc[t : t + horizon + 1].to_numpy())
            assert variance == pytest.approx(frame[f"target_var_{name}"].iloc[t], rel=1e-12)
            assert log_return == pytest.approx(np.log(close.iloc[t + horizon] / close.iloc[t]))


def test_realized_statistics_reject_bad_prices():
    for bad in ([100.0], [100.0, np.nan], [100.0, -1.0], [100.0, np.inf]):
        with pytest.raises(ValueError, match="cierres"):
            realized_statistics(np.array(bad))


def test_resolution_flags_follow_the_band_multipliers():
    closes = np.array([100.0, 101.0, 99.0, 103.0, 104.0, 102.0, 106.0])
    horizon = 6
    var = 0.5**2
    sigma = np.sqrt(var * horizon / 2190)
    z = abs(np.log(106 / 100)) / sigma
    narrow = {"68": z - 0.01, "90": z + 0.01, "95": None}
    values = resolution_values(closes, horizon, var, narrow)
    assert values["inside_68"] is False
    assert values["inside_90"] is True
    assert values["inside_95"] is None
    assert values["realized_log_return"] == pytest.approx(np.log(1.06))
    assert values["qlike"] == pytest.approx(qlike_loss(values["realized_var_annual"], var))
    with pytest.raises(ValueError, match="h\\+1"):
        resolution_values(closes[:-1], horizon, var, narrow)


def test_qlike_loss_is_zero_only_when_forecast_equals_realized():
    assert qlike_loss(0.25, 0.25) == pytest.approx(0.0)
    assert qlike_loss(0.25, 0.5) > 0
    assert qlike_loss(0.5, 0.25) > 0
    assert np.isfinite(qlike_loss(0.0, 0.25))
