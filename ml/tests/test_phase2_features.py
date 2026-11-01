import io
import json
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.features import long_horizon as lh
from src.research import aux_sources, feature_study
from src.research import target_study as study
from tests.test_target_study import _study_frame

STEP = pd.Timedelta(hours=4)

# ─── Fuentes auxiliares ──────────────────────────────────────────────────────


def _kline_csv(times_ms, header=False, micro=False):
    lines = (
        ["open_time,open,high,low,close,volume,close_time,qv,count,tb,tq,ignore"] if header else []
    )
    for index, value in enumerate(times_ms):
        stamp = value * 1000 if micro else value
        lines.append(f"{stamp},{index},{index + 1},{index},{index + 0.5},10,0,0,0,0,0,0")
    return "\n".join(lines).encode()


def test_parse_kline_csv_handles_header_and_microseconds():
    start = int(pd.Timestamp("2025-06-01", tz="UTC").timestamp() * 1000)
    times = [start, start + 4 * 3_600_000]
    plain = aux_sources.parse_kline_csv(_kline_csv(times))
    with_header = aux_sources.parse_kline_csv(_kline_csv(times, header=True, micro=True))
    pd.testing.assert_frame_equal(plain, with_header)
    assert plain["open_time"].iloc[1] == pd.Timestamp("2025-06-01 04:00", tz="UTC")
    assert plain["close"].tolist() == [0.5, 1.5]


CALENDAR_HTML = """
<a id="1">2025 FOMC Meetings</a>
<div class="fomc-meeting__month col"><strong>January</strong></div>
<div class="fomc-meeting__date col">28-29</div>
<div class="fomc-meeting__month col"><strong>Apr/May</strong></div>
<div class="fomc-meeting__date col">30-1*</div>
<div class="fomc-meeting__month col"><strong>August</strong></div>
<div class="fomc-meeting__date col">22 (notation vote)</div>
"""
HISTORICAL_HTML = """
<h5 class="x">January 28-29 Meeting - 2020</h5>
<h5 class="x">March 15 (unscheduled) Meeting - 2020</h5>
<h5 class="x">March 17-18 (cancelled) Meeting - 2020</h5>
<h5 class="x">March 19 (notation vote) - 2020</h5>
<h5 class="x">July 28-29 Meeting - 2020</h5>
"""


def test_fomc_parsers_keep_only_scheduled_meetings_and_convert_new_york_time():
    assert aux_sources.parse_fomc_calendar(CALENDAR_HTML) == [date(2025, 1, 29), date(2025, 5, 1)]
    assert aux_sources.parse_fomc_historical(HISTORICAL_HTML) == [
        date(2020, 1, 29),
        date(2020, 7, 29),
    ]
    decisions = aux_sources.fomc_decisions([date(2020, 1, 29), date(2020, 7, 29)])
    assert decisions["decision_time"].dt.hour.tolist() == [19, 18]


def _zip(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", content)
    return buffer.getvalue()


def test_capture_is_immutable_verifiable_and_checks_checksums(tmp_path, monkeypatch):
    import hashlib

    start_ms = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
    payload = _zip(_kline_csv([start_ms + i * 4 * 3_600_000 for i in range(3)]))
    page = {
        "data": [
            {
                "asset": "btc",
                "time": "2021-01-01T00:00:00Z",
                **dict.fromkeys(aux_sources.BTC_METRICS, "1"),
            }
        ]
    }
    stable = {
        "data": [
            {"asset": a, "time": "2021-01-01T00:00:00Z", "SplyCur": "5"}
            for a in aux_sources.STABLECOINS
        ]
    }
    corrupt = {"value": False}

    def fake_get(client, url, params=None):
        if url.endswith(".CHECKSUM"):
            digest = "0" * 64 if corrupt["value"] else hashlib.sha256(payload).hexdigest()
            return f"{digest}  file.zip".encode()
        if url.endswith(".zip"):
            return payload
        if url == aux_sources.COINMETRICS_URL:
            return json.dumps(stable if "usdt" in params["assets"] else page).encode()
        return (CALENDAR_HTML if "calendars" in url else HISTORICAL_HTML).encode()

    monkeypatch.setattr(aux_sources, "_get", fake_get)
    target = tmp_path / "aux"
    manifest = aux_sources.capture(target, "test", date(2021, 1, 1), date(2021, 2, 1))
    assert aux_sources.verify(target)["manifest_sha256"] == manifest["manifest_sha256"]
    assert set(aux_sources.load(target)) == {
        *aux_sources.KLINE_SERIES,
        "coinmetrics_btc",
        "coinmetrics_stablecoins",
        "fomc",
    }
    with pytest.raises(ValueError, match="finalizadas"):
        aux_sources.capture(target, "test", date(2021, 1, 1), date(2021, 2, 1))
    (target / "fomc.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="no coincide"):
        aux_sources.verify(target)
    corrupt["value"] = True
    with pytest.raises(ValueError, match="Checksum"):
        aux_sources.capture(tmp_path / "other", "x", date(2021, 1, 1), date(2021, 2, 1))


# ─── Features de horizonte largo ─────────────────────────────────────────────


def synthetic_inputs(days=520, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 6
    times = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.uniform(50, 150, n)
    candles = pd.DataFrame({"open_time": times, "close": close, "volume": volume})
    series = pd.Series(close)
    features = (
        pd.DataFrame(
            {
                "open_time": times,
                "sma_200": series.rolling(200, min_periods=1).mean(),
                "obv": np.cumsum(np.sign(series.diff().fillna(0)) * volume),
                "accumulation_distribution": np.cumsum(volume * 0.1),
                "macd_signal": series.ewm(span=9).mean() - series,
                "macd_histogram": series.diff().fillna(0),
            }
        )
        .iloc[1300:]
        .reset_index(drop=True)
    )
    history = pd.DataFrame(
        {
            "open_time": pd.date_range(times[0] - 300 * STEP, periods=300 + 10, freq="4h"),
            "close": np.concatenate([100 * np.exp(rng.normal(0, 0.01, 300).cumsum()), close[:10]]),
        }
    )
    funding_times = pd.date_range(pd.Timestamp("2019-12-01", tz="UTC"), times[-1], freq="8h")
    funding = pd.DataFrame(
        {"funding_time": funding_times, "funding_rate": rng.normal(1e-4, 1e-4, len(funding_times))}
    )
    metric_times = pd.date_range(times[0], times[-1], freq="5min")
    metrics = pd.DataFrame(
        {
            "create_time": metric_times,
            "sum_open_interest": np.exp(rng.normal(0, 0.001, len(metric_times)).cumsum()) * 1e5,
            "sum_open_interest_value": 1.0,
            "count_toptrader_long_short_ratio": 1.0,
            "sum_toptrader_long_short_ratio": rng.uniform(0.8, 1.2, len(metric_times)),
            "count_long_short_ratio": rng.uniform(0.8, 1.2, len(metric_times)),
            "sum_taker_long_short_vol_ratio": rng.uniform(0.8, 1.2, len(metric_times)),
        }
    )
    days_index = pd.date_range(pd.Timestamp("2019-11-01", tz="UTC"), times[-1].floor("D"), freq="D")
    onchain = pd.DataFrame(
        {
            "time": days_index,
            **{m: rng.uniform(1, 2, len(days_index)) for m in aux_sources.BTC_METRICS},
        }
    )
    stable = pd.concat(
        [
            pd.DataFrame(
                {"time": days_index, "asset": a, "SplyCur": rng.uniform(1, 2, len(days_index))}
            )
            for a in aux_sources.STABLECOINS
        ]
    )
    business = pd.bdate_range("2019-10-01", times[-1].tz_localize(None))
    macro = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": s,
                    "trade_date": business.date,
                    "close": rng.uniform(10, 20, len(business)),
                }
            )
            for s in ("^GSPC", "DX-Y.NYB", "GC=F", "^VIX")
        ]
    )
    perp = pd.DataFrame({"open_time": times, "close": close * (1 + rng.normal(0, 1e-4, n))})
    premium = pd.DataFrame({"open_time": times, "close": rng.normal(0, 1e-4, n)})
    ethbtc = pd.DataFrame({"open_time": times, "close": np.exp(rng.normal(0, 0.01, n).cumsum())})
    fomc = aux_sources.fomc_decisions(
        [d.date() for d in pd.date_range("2019-12-01", "2022-01-31", freq="45D")]
    )
    sources = {"candles": candles, "funding": funding, "macro": macro}
    aux = {
        "btc_spot_klines": history,
        "perp_klines": perp,
        "premium_index": premium,
        "ethbtc_klines": ethbtc,
        "coinmetrics_btc": onchain,
        "coinmetrics_stablecoins": stable,
        "fomc": fomc,
    }
    return features, sources, metrics, aux


def test_long_horizon_features_declare_all_columns():
    features, sources, metrics, aux = synthetic_inputs()
    result = lh.build_long_horizon_features(features, sources, metrics, aux)
    expected = [c for family in lh.FAMILIES.values() for c in family]
    assert set(expected) <= set(result.columns)
    late = result.iloc[-100:][expected]
    assert late.notna().all().all(), late.columns[late.isna().any()].tolist()


def test_long_horizon_features_never_use_unavailable_data():
    features, sources, metrics, aux = synthetic_inputs()
    first = lh.build_long_horizon_features(features, sources, metrics, aux)
    cutoff = features["open_time"].iloc[len(features) // 2] + STEP

    def later(times, delay=None):
        return pd.to_datetime(times, utc=True) + (delay or pd.Timedelta(0)) > cutoff

    rng = np.random.default_rng(99)

    def scramble(frame, mask, columns):
        frame = frame.copy()
        for column in columns:
            frame.loc[mask, column] = frame.loc[mask, column] * rng.uniform(
                0.5, 1.5, int(mask.sum())
            )
        return frame

    changed_features = scramble(
        features,
        later(features["open_time"], STEP),
        ["sma_200", "obv", "accumulation_distribution", "macd_signal", "macd_histogram"],
    )
    changed_sources = {
        "candles": scramble(
            sources["candles"], later(sources["candles"]["open_time"], STEP), ["close", "volume"]
        ),
        "funding": scramble(
            sources["funding"], later(sources["funding"]["funding_time"], STEP), ["funding_rate"]
        ),
        "macro": scramble(
            sources["macro"], later(sources["macro"]["trade_date"], pd.Timedelta(days=1)), ["close"]
        ),
    }
    metric_cols = [c for c in metrics.columns if c != "create_time"]
    changed_metrics = scramble(
        metrics, later(metrics["create_time"], lh.METRICS_DELAY), metric_cols
    )
    changed_aux = dict(aux)
    for name in ("perp_klines", "premium_index", "ethbtc_klines"):
        changed_aux[name] = scramble(aux[name], later(aux[name]["open_time"], STEP), ["close"])
    changed_aux["coinmetrics_btc"] = scramble(
        aux["coinmetrics_btc"],
        later(aux["coinmetrics_btc"]["time"], lh.ONCHAIN_DELAY),
        list(aux_sources.BTC_METRICS),
    )
    stable = aux["coinmetrics_stablecoins"]
    changed_aux["coinmetrics_stablecoins"] = scramble(
        stable, later(stable["time"], lh.ONCHAIN_DELAY), ["SplyCur"]
    )
    second = lh.build_long_horizon_features(
        changed_features, changed_sources, changed_metrics, changed_aux
    )
    known = (first["open_time"] + STEP <= cutoff).to_numpy()
    columns = [c for family in lh.FAMILIES.values() for c in family]
    pd.testing.assert_frame_equal(first.loc[known, columns], second.loc[known, columns])
    assert not first.loc[~known, columns].equals(second.loc[~known, columns])


def test_price_history_rejects_inconsistent_auxiliary_history():
    features, sources, metrics, aux = synthetic_inputs(days=10)
    bad = aux["btc_spot_klines"].copy()
    bad["close"] *= 1.01
    with pytest.raises(ValueError, match="no coincide"):
        lh.price_history(sources["candles"], bad)


def test_event_features_count_days_and_expiries():
    fomc = aux_sources.fomc_decisions([date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1)])
    times = pd.Series(pd.to_datetime(["2024-03-16 20:00", "2024-03-21 00:00"], utc=True))
    out = lh.event_features(times, fomc)
    assert out["days_to_next_fomc"].iloc[0] == pytest.approx(3.75)
    assert out["fomc_last_72h"].tolist() == [0.0, 1.0]
    assert out["days_to_monthly_expiry"].iloc[0] == pytest.approx(12 + 8 / 24)
    assert out["quarterly_expiry_week"].tolist() == [0.0, 0.0]


# ─── Estudio de features ─────────────────────────────────────────────────────


def _development(n=5200):
    features, candles = _study_frame(n)
    data = study.attach_horizons(features, candles)
    rng = np.random.default_rng(5)
    for name in {c for columns in feature_study.feature_sets().values() for c in columns}:
        if name not in data:
            data[name] = rng.normal(size=len(data))
    return study.development_rows(data, data["open_time"].iloc[-1])


def test_feature_sets_are_nested_on_the_stationary_reference():
    sets = feature_study.feature_sets()
    assert not set(lh.NONSTATIONARY_COLUMNS) & set(sets["stationary"])
    for family in feature_study.EXTENSIONS:
        assert set(sets["stationary"]) < set(sets[f"stationary_{family}"])
    assert len(sets["all"]) == len(set(sets["all"]))


def test_fit_config_returns_coefficients_for_linear_models():
    data = _development(3000)
    train, test = data.iloc[:2000].copy(), data.iloc[2000:2100]
    train["target_label"] = study.horizon_label(train, 6)
    p, coef = feature_study.fit_config(
        feature_study.FeatureConfig(6, "all", "logistic_l1"), train, test
    )
    assert (
        len(p) == 100 and coef is not None and len(coef) == len(feature_study.feature_sets()["all"])
    )
    p, coef = feature_study.fit_config(
        feature_study.FeatureConfig(6, "all", "lightgbm"), train, test
    )
    assert coef is None and ((p > 0) & (p < 1)).all()


def test_feature_study_report_is_serializable_and_ablation_is_paired(monkeypatch):
    data = _development()
    monkeypatch.setattr(study, "MIN_SELECTION_HISTORY", 300)
    monkeypatch.setattr(feature_study, "PBO_PARTITIONS", 4)
    validation_seen = []

    def fake(config, train, test):
        validation_seen.append((train["open_time"].max(), test["open_time"].min(), config.horizon))
        rng = np.random.default_rng(len(train) + len(config.name))
        coef = {"a": 1.0, "b": -0.5} if config.model != "lightgbm" else None
        return np.clip(0.5 + rng.normal(0, 0.05, len(test)), 0.01, 0.99), coef

    report, predictions = feature_study.build_report(data, n_folds=4, fitter=fake)
    json.dumps(report, allow_nan=False)
    for train_end, test_start, horizon in validation_seen:
        assert train_end + (horizon + 1) * STEP <= test_start
    assert len(report["configs"]) == len(feature_study.configs())
    rows = report["ablation"]["comparisons"]
    assert len(rows) == len(feature_study.HORIZONS) * (1 + len(feature_study.EXTENSIONS) + 1)
    assert all(isinstance(row["kept"], bool) for row in rows)
    assert set(report["coefficient_stability"]) == {
        "h6_all_logistic_l2",
        "h6_all_logistic_l1",
        "h18_all_logistic_l2",
        "h18_all_logistic_l1",
    }
    assert isinstance(report["gate"]["passed"], bool)
    assert predictions["open_time"].is_unique
