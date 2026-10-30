from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.research.metrics_archive import METRICS_COLUMNS
from src.research.metrics_features import (
    OI_FEATURE_COLUMNS,
    POSITION_FEATURE_COLUMNS,
    add_metrics_context,
)


def metrics_data():
    times = pd.date_range("2023-12-31", periods=6 * 288, freq="5min", tz="UTC")
    values = np.arange(len(times), dtype=float) + 1000
    frame = pd.DataFrame({"create_time": times})
    for column in METRICS_COLUMNS:
        frame[column] = values
    return frame


def candles_data():
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-03", periods=12, freq="4h", tz="UTC"),
            "return_1": 0.02,
        }
    )


def test_metrics_obey_five_minute_interval_end_and_48h_delay():
    metrics, candles = metrics_data(), candles_data()
    result = add_metrics_context(
        candles, metrics, timedelta(hours=4), timedelta(hours=48), timedelta(minutes=10)
    )
    row = result.iloc[0]
    decision = candles.loc[0, "open_time"] + timedelta(hours=4)
    assert row["oi_source_time"] == decision - timedelta(hours=48)
    assert row["oi_available_at"] == decision
    source_start = row["oi_source_time"] - timedelta(minutes=5)
    i = metrics.index[metrics["create_time"] == source_start][0]
    expected = np.log(metrics.loc[i, "sum_open_interest"]) - np.log(
        metrics.loc[i - 48, "sum_open_interest"]
    )
    assert row["oi_log_change_4h"] == pytest.approx(expected)
    assert row["oi_price_interaction"] == pytest.approx(expected * 0.02)


def test_future_metrics_cannot_change_past_features():
    metrics, candles = metrics_data(), candles_data()
    baseline = add_metrics_context(
        candles, metrics, timedelta(hours=4), timedelta(hours=48), timedelta(minutes=10)
    )
    changed = metrics.copy()
    changed.loc[
        changed["create_time"] >= pd.Timestamp("2024-01-02", tz="UTC"), METRICS_COLUMNS
    ] *= 3
    after = add_metrics_context(
        candles, changed, timedelta(hours=4), timedelta(hours=48), timedelta(minutes=10)
    )
    pd.testing.assert_frame_equal(
        baseline.loc[:4, OI_FEATURE_COLUMNS + POSITION_FEATURE_COLUMNS],
        after.loc[:4, OI_FEATURE_COLUMNS + POSITION_FEATURE_COLUMNS],
    )


def test_missing_intervals_are_not_fabricated():
    metrics, candles = metrics_data(), candles_data()
    decision = candles.loc[0, "open_time"] + timedelta(hours=4)
    current_source = decision - timedelta(hours=48, minutes=5)
    missing_4h_ago = current_source - timedelta(hours=4)
    metrics = metrics[metrics["create_time"] != missing_4h_ago]
    result = add_metrics_context(
        candles, metrics, timedelta(hours=4), timedelta(hours=48), timedelta(minutes=10)
    )
    assert pd.isna(result.loc[0, "oi_log_change_4h"])


def test_duplicate_or_naive_metrics_are_rejected():
    metrics, candles = metrics_data(), candles_data()
    with pytest.raises(ValueError, match="duplicados"):
        add_metrics_context(
            candles,
            pd.concat([metrics, metrics.iloc[:1]]),
            timedelta(hours=4),
            timedelta(hours=48),
            timedelta(minutes=10),
        )
    metrics["create_time"] = metrics["create_time"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="zona horaria"):
        add_metrics_context(
            candles, metrics, timedelta(hours=4), timedelta(hours=48), timedelta(minutes=10)
        )
