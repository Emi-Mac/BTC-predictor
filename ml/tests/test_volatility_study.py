import json

import numpy as np
import pandas as pd
import pytest

from src.research import volatility_study as vol
from tests.test_phase2_features import synthetic_inputs


def test_volatility_targets_use_only_future_returns_and_past_inputs_only_past():
    times = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(times)))), index=times)
    frame = vol.volatility_frame(close, pd.Series(times))
    squared = np.log(close).diff() ** 2
    t = 200
    expected_future = squared.iloc[t + 1 : t + 43].mean() * vol.PERIODS_PER_YEAR_4H
    expected_past = squared.iloc[t - 41 : t + 1].mean() * vol.PERIODS_PER_YEAR_4H
    assert frame["target_var_7d"].iloc[t] == pytest.approx(expected_future)
    assert frame["var_7d"].iloc[t] == pytest.approx(expected_past)
    changed = close.copy()
    changed.iloc[t + 1 :] *= 1.5
    other = vol.volatility_frame(changed, pd.Series(times))
    past_columns = ["var_1d", "var_7d", "var_30d", "var_ewma", "abs_return_1d"]
    pd.testing.assert_frame_equal(frame.loc[:t, past_columns], other.loc[:t, past_columns])


def test_qlike_is_zero_only_for_perfect_forecast():
    realized = np.array([0.2, 0.5, 1.0])
    assert vol.qlike(realized, realized) == pytest.approx(np.zeros(3))
    assert (vol.qlike(realized, realized * 2) > 0).all()
    assert (vol.qlike(realized, realized / 2) > 0).all()


def test_target_exposure_caps_at_one_and_respects_band():
    variance = np.array([0.04, 0.25, 0.26, 1.0, 1.0])
    position = vol.target_exposure(variance)
    assert position[0] == 1.0
    assert position[1] == pytest.approx(1.0)
    assert position[2] == pytest.approx(1.0)
    assert position[3] == pytest.approx(0.5)


def _development():
    features, sources, metrics, aux = synthetic_inputs(days=900)
    rng = np.random.default_rng(7)
    columns = {
        name
        for model in vol.models()
        for name in model.columns
        if name not in vol.HAR_COLUMNS
        and name not in vol.HAR_X_EXTRA
        and name not in features.columns
    }
    for name in sorted(columns - {c for f in vol.FAMILIES.values() for c in f}):
        features[name] = rng.normal(size=len(features))
    features["funding_rate"] = 1e-4
    features["vix_close"] = rng.uniform(12, 30, len(features))
    features["volume_sma_ratio"] = rng.uniform(0.5, 1.5, len(features))
    enriched = vol.build_long_horizon_features(features, sources, metrics, aux)
    close = vol.price_history(sources["candles"], aux["btc_spot_klines"])
    data = vol.model_inputs(enriched, close)
    return vol.development_rows(data, data["open_time"].iloc[-1])


def test_walk_forward_purges_by_horizon_and_report_is_serializable():
    development = _development()
    seen = []

    def spy(model, train, test, horizon):
        seen.append((train["open_time"].max(), test["open_time"].min(), vol.HORIZONS[horizon]))
        return vol.fit_predict_variance(model, train, test, horizon)

    report, predictions = vol.build_report(development, n_folds=3, predictor=spy)
    json.dumps(report, allow_nan=False)
    for train_end, test_start, horizon in seen:
        assert train_end + (horizon + 1) * vol.STEP <= test_start
    assert set(report["forecasts"]) == set(vol.HORIZONS)
    assert len(report["economics"]["strategies"]) == len(vol.models()) * len(vol.HORIZONS)
    assert isinstance(report["gate"]["volatility_predictable"], bool)
    assert (predictions.filter(like="_7d").drop(columns="target_var_7d") > 0).all().all()
