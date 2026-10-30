from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features import pipeline
from src.features.compute import FEATURE_COLUMNS, compute_features
from src.features.selection import select_features

STEP = timedelta(hours=4)


@pytest.fixture
def market_data():
    n = 650
    i = np.arange(n)
    times = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = 40000 + i * 5 + 150 * np.sin(i / 8) + 35 * np.cos(i / 3)
    candles = pd.DataFrame(
        {
            "id": i + 1,
            "open_time": times,
            "open": close - 2,
            "high": close + 15,
            "low": close - 20,
            "close": close,
            "volume": 100 + i % 23,
        }
    )
    dates = pd.date_range("2023-11-01", periods=230, freq="D", tz="UTC")
    macro = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "trade_date": dates.date,
                    "close": base
                    + np.arange(len(dates)) * slope
                    + 2 * np.sin(np.arange(len(dates)) / 3),
                }
            )
            for symbol, base, slope in [
                ("DX-Y.NYB", 100, 0.02),
                ("^GSPC", 4000, 0.7),
                ("^VIX", 20, 0.01),
                ("GC=F", 2000, 0.3),
            ]
        ],
        ignore_index=True,
    )
    fgi = pd.DataFrame(
        {
            "obs_date": dates.date,
            "value": 40 + np.arange(len(dates)) % 35,
            "classification": "Neutral",
        }
    )
    return candles, macro, fgi


def test_technical_features_targets_and_calendar(market_data):
    candles, macro, fgi = market_data
    out = compute_features(candles, macro, fgi, STEP)
    row = out.iloc[250]
    close = candles["close"]
    assert row["sma_200"] == pytest.approx(close.iloc[51:251].mean())
    assert row["bb_middle"] == pytest.approx(close.iloc[231:251].mean())
    assert row["bb_upper"] == pytest.approx(
        close.iloc[231:251].mean() + 2 * close.iloc[231:251].std(ddof=0)
    )
    assert row["roc_5"] == pytest.approx(close.iloc[250] / close.iloc[245] - 1)
    assert row["target_label"] == int(close.iloc[251] > close.iloc[250])
    assert pd.isna(out["target_label"].iloc[-1])
    assert out[FEATURE_COLUMNS].iloc[199:].notna().all().all()
    assert all(
        out[[f"weekday_{d}" for d in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")]].sum(axis=1)
        == 1
    )
    decision = candles["open_time"].iloc[250] + STEP
    assert row["hour_of_day"] == decision.hour
    assert row["day_of_week"] == decision.day_of_week


def test_daily_macro_sentiment_correlation_and_vwap(market_data):
    candles, macro, fgi = market_data
    result = compute_features(candles, macro, fgi, STEP)
    i = 250
    decision = candles.loc[i, "open_time"] + STEP
    day = decision.floor("D")
    dxy = macro[macro["symbol"] == "DX-Y.NYB"].set_index("trade_date")["close"]
    assert result.loc[i, "dxy_change_pct"] == pytest.approx(
        dxy[(day - pd.Timedelta(days=1)).date()] / dxy[(day - pd.Timedelta(days=2)).date()] - 1
    )
    sentiment = fgi.set_index("obs_date")["value"]
    assert result.loc[i, "fear_greed_sma_7"] == pytest.approx(
        np.mean([sentiment[(day - pd.Timedelta(days=k)).date()] for k in range(7)])
    )
    assert result.loc[i, "fear_greed_momentum_7"] == (
        sentiment[day.date()] - sentiment[(day - pd.Timedelta(days=7)).date()]
    )

    daily_btc = candles.iloc[: i + 1].copy()
    closed = daily_btc["open_time"] + STEP
    daily_btc = daily_btc.loc[closed.dt.hour == 0].copy()
    daily_btc["trade_date"] = (closed.loc[daily_btc.index] - pd.Timedelta(days=1)).dt.date
    sp = macro[macro["symbol"] == "^GSPC"][["trade_date", "close"]]
    paired = daily_btc.merge(sp, on="trade_date", suffixes=("_btc", "_sp"))
    btc_returns = paired["close_btc"].pct_change(fill_method=None)
    sp_returns = paired["close_sp"].pct_change(fill_method=None)
    assert result.loc[i, "btc_sp500_corr_20"] == pytest.approx(
        btc_returns.iloc[-20:].corr(sp_returns.iloc[-20:])
    )
    intraday = candles.iloc[: i + 1]
    intraday = intraday[
        intraday["open_time"].dt.floor("D") == candles.loc[i, "open_time"].floor("D")
    ]
    typical = (intraday["high"] + intraday["low"] + intraday["close"]) / 3
    assert result.loc[i, "vwap_daily"] == pytest.approx(
        (typical * intraday["volume"]).sum() / intraday["volume"].sum()
    )


def test_vectorized_matches_incremental_calculation(market_data):
    candles, macro, fgi = market_data
    candles = candles.iloc[:225]
    full = compute_features(candles, macro, fgi, STEP)
    for length in range(1, len(candles) + 1):
        incremental = compute_features(candles.iloc[:length], macro, fgi, STEP)
        pd.testing.assert_series_equal(
            full.loc[length - 1, FEATURE_COLUMNS],
            incremental.loc[length - 1, FEATURE_COLUMNS],
            check_names=False,
            rtol=1e-10,
            atol=1e-10,
        )


def test_future_candle_and_external_changes_do_not_change_past_features(market_data):
    candles, macro, fgi = market_data
    cutoff = candles["open_time"].iloc[250] + STEP
    original = compute_features(candles, macro, fgi, STEP)
    altered_candles = candles.copy()
    altered_candles.loc[251:, ["open", "high", "low", "close"]] *= 2
    altered_macro = macro.copy()
    altered_macro.loc[
        pd.to_datetime(macro["trade_date"], utc=True) >= cutoff.floor("D"), "close"
    ] *= 2
    altered_fgi = fgi.copy()
    altered_fgi.loc[pd.to_datetime(fgi["obs_date"], utc=True) > cutoff, "value"] = 1
    changed = compute_features(altered_candles, altered_macro, altered_fgi, STEP)
    pd.testing.assert_series_equal(
        original.loc[250, FEATURE_COLUMNS], changed.loc[250, FEATURE_COLUMNS], check_names=False
    )


def test_zero_range_and_flat_prices_have_defined_neutral_indicators(market_data):
    candles, macro, fgi = market_data
    candles[["open", "high", "low", "close"]] = 100.0
    result = compute_features(candles, macro, fgi, STEP)
    assert result.loc[250, "rsi_14"] == 50
    assert result.loc[250, "williams_r"] == -50
    assert result.loc[250, "stoch_rsi"] == 50
    assert result.loc[250, "accumulation_distribution"] == 0


def test_missing_external_source_and_temporal_gaps_are_errors(market_data):
    candles, macro, fgi = market_data
    with pytest.raises(ValueError, match="Faltan series externas"):
        compute_features(candles, macro[macro["symbol"] != "^VIX"], fgi, STEP)
    with pytest.raises(ValueError, match="consecutivas"):
        compute_features(candles.drop(index=20), macro, fgi, STEP)
    candles["volume"] = candles["volume"].astype(float)
    candles.loc[4, "volume"] = np.inf
    with pytest.raises(ValueError, match="inválidas"):
        compute_features(candles, macro, fgi, STEP)


def _selection_data():
    rng = np.random.default_rng(42)
    count = 300
    target = rng.integers(0, 2, count)
    data = pd.DataFrame({name: np.zeros(count) for name in FEATURE_COLUMNS})
    data["rsi_14"] = target + rng.normal(0, 0.05, count)
    data["ema_9"] = data["rsi_14"] * 3
    data["open_time"] = pd.date_range("2024-01-01", periods=count, freq="4h", tz="UTC")
    data["target_label"] = target
    return data


def test_feature_selection_uses_only_labels_known_at_cutoff():
    data = _selection_data()
    end = datetime(2024, 2, 1, tzinfo=UTC)
    result = select_features(data, end, STEP)
    assert result.train_rows == 185
    assert len(set(result.selected) & {"rsi_14", "ema_9"}) == 1
    assert len(result.removed_collinear) == 1
    assert result.correlations["macd"] == 0
    altered = data.copy()
    future = altered["open_time"] + 2 * STEP > end
    altered.loc[future, "target_label"] = 1 - altered.loc[future, "target_label"]
    altered.loc[future, "rsi_14"] *= 50
    assert select_features(altered, end, STEP) == result


def test_selection_rejects_naive_cutoff_and_single_class():
    data = _selection_data()
    with pytest.raises(ValueError, match="zona horaria"):
        select_features(data, datetime(2024, 2, 1), STEP)
    data["target_label"] = 1
    with pytest.raises(ValueError, match="ambas clases"):
        select_features(data, datetime(2024, 2, 1, tzinfo=UTC), STEP)


async def test_cycle_calculates_features_after_ingestion(monkeypatch):
    calls = []

    async def ingest(repo, settings):
        calls.append(("ingestion", repo, settings))

    async def features(engine, settings):
        calls.append(("features", engine, settings))

    class Repo:
        engine = "engine"

    repo = Repo()
    settings = object()
    monkeypatch.setattr(pipeline, "run_ingestion", ingest)
    monkeypatch.setattr(pipeline, "run_features", features)
    await pipeline.run_cycle(repo, settings)
    assert calls == [("ingestion", repo, settings), ("features", "engine", settings)]
