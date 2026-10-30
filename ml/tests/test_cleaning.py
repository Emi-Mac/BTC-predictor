import pandas as pd

from src.ingestion.cleaning import (
    clean_candles,
    detect_gaps,
    invalid_ohlcv_mask,
    invalid_taker_volume_mask,
)
from src.ingestion.integrity import check_candle_integrity
from tests.conftest import make_candles


def test_detect_gaps_reports_start_and_size(step):
    df = make_candles(10, skip={3, 4, 7})
    gaps = detect_gaps(df["open_time"], step)
    assert [(g.start, g.missing) for g in gaps] == [
        (pd.Timestamp("2024-01-01 12:00", tz="UTC"), 2),
        (pd.Timestamp("2024-01-02 04:00", tz="UTC"), 1),
    ]


def test_single_missing_candle_is_forward_filled_with_previous_close(step):
    df = make_candles(6, skip={3})
    out, report = clean_candles(df, step)
    assert report.filled == 1 and not report.unfilled_gaps
    filled = out.iloc[3]
    prev_close = out.iloc[2]["close"]
    assert (filled[["open", "high", "low", "close"]] == prev_close).all()
    assert filled["volume"] == 0 and filled["num_trades"] == 0


def test_gaps_longer_than_one_period_are_not_invented(step):
    df = make_candles(8, skip={3, 4})
    out, report = clean_candles(df, step)
    assert report.filled == 0
    assert len(out) == 6
    assert report.unfilled_gaps[0].missing == 2
    assert report.completeness < 1.0


def test_invalid_rows_are_dropped(step):
    df = make_candles(6)
    df.loc[1, "close"] = 0.0  # close > 0
    df.loc[2, "volume"] = -1.0  # volume >= 0
    df.loc[3, "high"] = df.loc[3, "low"] - 1  # high < low
    assert invalid_ohlcv_mask(df).tolist() == [False, True, True, True, False, False]
    _, report = clean_candles(df, step)
    assert report.invalid_removed == 3


def test_duplicates_are_removed_keeping_last(step):
    df = make_candles(4)
    dup = df.iloc[[1]].copy()
    dup["close"] = dup["close"] + 0.5
    out, report = clean_candles(pd.concat([df, dup], ignore_index=True), step)
    assert report.duplicates_removed == 1
    assert out["open_time"].is_unique
    kept = out.loc[out["open_time"] == df.loc[1, "open_time"], "close"].iloc[0]
    assert kept == dup["close"].iloc[0]


def test_empty_input(step):
    out, report = clean_candles(make_candles(0), step)
    assert out.empty and report.completeness == 1.0


def test_integrity_passes_on_clean_data(step):
    report = check_candle_integrity(make_candles(500), step)
    assert report.ok
    assert report.completeness == 1.0


def test_integrity_flags_duplicates_gaps_and_short_history(step):
    df = make_candles(500, skip={100, 101})
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    report = check_candle_integrity(df, step, min_span_days=365)
    assert not report.ok
    joined = " ".join(report.problems)
    assert "duplicadas" in joined and "gaps" in joined and "histórico" in joined


def test_integrity_completeness_threshold_counts_synthetic_rows(step):
    df = make_candles(200)
    df.loc[10:11, ["volume", "num_trades"]] = 0
    report = check_candle_integrity(df, step)
    assert report.synthetic_rows == 2
    assert report.completeness == 198 / 200
    assert not report.ok


def test_integrity_empty_dataset(step):
    assert not check_candle_integrity(make_candles(0), step).ok


def test_integrity_detects_misaligned_candles(step):
    df = make_candles(10)
    df.loc[3, "open_time"] += pd.Timedelta(minutes=7)
    assert check_candle_integrity(df, step).misaligned_rows == 1


def test_taker_volume_validation_and_unknown_synthetic_rows(step):
    df = make_candles(6, skip={3})
    df["taker_buy_base_volume"] = 20.0
    df["taker_buy_quote_volume"] = 500_000.0
    cleaned, report = clean_candles(df, step)
    assert report.filled == 1
    assert pd.isna(cleaned.loc[3, "taker_buy_base_volume"])
    assert pd.isna(cleaned.loc[3, "taker_buy_quote_volume"])
    assert not invalid_taker_volume_mask(cleaned).any()

    df.loc[1, "taker_buy_base_volume"] = df.loc[1, "volume"] + 1
    df.loc[2, "taker_buy_quote_volume"] = -1
    df.loc[3, "taker_buy_base_volume"] = float("inf")
    _, report = clean_candles(df, step)
    assert report.invalid_removed == 3
