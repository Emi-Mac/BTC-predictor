from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.funding_context import FUNDING_FEATURE_COLUMNS, add_funding_context
from tests.conftest import make_candles

STEP = timedelta(hours=4)
MAX_AGE = timedelta(hours=36)


def _funding():
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "funding_time": pd.date_range("2023-12-31", periods=20, freq="8h", tz="UTC"),
            "funding_rate": np.arange(20) / 10000,
            "mark_price": 40000.0,
        }
    )


def test_funding_becomes_available_only_after_full_candle_delay():
    candles = make_candles(30)
    funding = _funding()
    result = add_funding_context(candles[["open_time"]], funding, STEP, MAX_AGE)
    # La vela 04:00-08:00 aún no puede usar el evento de las 08:00.
    assert result.loc[1, "funding_time"] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    # A las 12:00 ya han transcurrido cuatro horas desde el evento de las 08:00.
    assert result.loc[2, "funding_time"] == pd.Timestamp("2024-01-01 08:00", tz="UTC")
    event = funding.loc[funding["funding_time"] == pd.Timestamp("2024-01-01 08:00", tz="UTC")]
    assert result.loc[2, "funding_rate"] == pytest.approx(event["funding_rate"].iloc[0])


def test_funding_vectorized_matches_each_incremental_prefix():
    candles = make_candles(30)
    funding = _funding()
    full = add_funding_context(candles[["open_time"]], funding, STEP, MAX_AGE)
    for length in range(1, len(candles) + 1):
        incremental = add_funding_context(
            candles[["open_time"]].iloc[:length], funding, STEP, MAX_AGE
        )
        pd.testing.assert_series_equal(
            full.loc[length - 1, FUNDING_FEATURE_COLUMNS],
            incremental.loc[length - 1, FUNDING_FEATURE_COLUMNS],
            check_names=False,
        )


def test_future_funding_cannot_change_past_features():
    candles = make_candles(30)
    funding = _funding()
    before = add_funding_context(candles[["open_time"]], funding, STEP, MAX_AGE)
    changed = funding.copy()
    changed.loc[changed["funding_time"] >= pd.Timestamp("2024-01-02", tz="UTC"), "funding_rate"] = (
        0.9
    )
    after = add_funding_context(candles[["open_time"]], changed, STEP, MAX_AGE)
    pd.testing.assert_frame_equal(
        before.loc[:5, FUNDING_FEATURE_COLUMNS], after.loc[:5, FUNDING_FEATURE_COLUMNS]
    )


def test_old_funding_and_missing_event_windows_remain_unavailable():
    candles = make_candles(60)
    old = _funding().iloc[:1]
    result = add_funding_context(candles[["open_time"]], old, STEP, MAX_AGE)
    assert pd.notna(result.loc[0, "funding_rate"])
    assert pd.isna(result.loc[10, "funding_rate"])

    funding = _funding().drop(index=3)
    result = add_funding_context(candles[["open_time"]], funding, STEP, MAX_AGE)
    t = pd.Timestamp("2024-01-01 08:00", tz="UTC")
    first_after_gap = result[result["funding_time"] == t].iloc[0]
    assert pd.isna(first_after_gap["funding_change_1"])
    assert pd.isna(first_after_gap["funding_mean_3"])


def test_invalid_funding_is_rejected():
    candles = make_candles(3)
    funding = _funding()
    funding.loc[0, "funding_rate"] = float("inf")
    with pytest.raises(ValueError, match="no finitos"):
        add_funding_context(candles[["open_time"]], funding, STEP, MAX_AGE)
    with pytest.raises(ValueError, match="requieren"):
        add_funding_context(candles[["open_time"]], funding.iloc[:0], STEP, MAX_AGE)
