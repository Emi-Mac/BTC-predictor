import hashlib
import io
import json
import zipfile
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research import source_snapshot
from src.research.dataset import finalize_dataset, verify_dataset
from src.research.metrics_archive import (
    METRICS_COLUMNS,
    MetricsArchiveClient,
    download_dataset_archives,
    parse_daily_archive,
)
from src.research.metrics_features import OI_FEATURE_COLUMNS
from src.research.oi_experiment import variant_features
from src.research.oi_holdout import history_hash
from src.research.source_snapshot import capture_sources, file_sha256
from tests.test_features import STEP

pytest_plugins = ["tests.test_features"]


def _funding():
    times = pd.date_range("2023-12-24", periods=150, freq="8h", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "funding_time": times,
            "funding_rate": ["0.0001"] * len(times),
            "mark_price": ["40000"] * len(times),
        }
    )


def _ohlcv(market_data, n=210):
    candles, macro, sentiment = market_data
    candles = candles.iloc[:n].copy()
    candles["close_time"] = candles["open_time"] + STEP - pd.Timedelta(milliseconds=1)
    candles["quote_volume"] = candles["volume"] * candles["close"]
    candles["num_trades"] = 100
    candles["taker_buy_base_volume"] = candles["volume"] * 0.4
    candles["taker_buy_quote_volume"] = candles["quote_volume"] * 0.4
    return candles, macro, sentiment


def _write_spec(directory):
    directory.mkdir(exist_ok=True)
    (directory / "build-spec.json").write_text(
        json.dumps(
            {
                "dataset_id": "test-version",
                "symbol": "BTCUSDT",
                "start": "2024-01-01",
                "end_exclusive": "2024-02-05",
                "assumed_availability_delay_hours": 48,
            }
        )
    )


async def test_source_capture_is_read_only_resumable_and_detects_tampering(
    tmp_path, market_data, monkeypatch
):
    _write_spec(tmp_path)
    candles, macro, sentiment = _ohlcv(market_data)
    requests = []

    class Spot:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def fetch_klines(self, symbol, interval, start, end):
            requests.append("spot")
            return candles

    class Futures(Spot):
        async def fetch(self, symbol, start, end):
            requests.append("funding")
            return _funding()

    class Repo:
        def __init__(self, engine):
            pass

        async def load_fear_greed(self):
            requests.append("sentiment")
            return sentiment

    async def fetch_macro(*args):
        requests.append("macro")
        return macro

    settings = SimpleNamespace(
        btc_symbol="BTCUSDT",
        default_interval="4h",
        binance_base_url="https://example.test",
        binance_rate_limit_per_min=1200,
        binance_futures_base_url="https://example.test",
        funding_rate_limit_per_min=90,
        macro_symbols=["DX-Y.NYB", "^GSPC", "^VIX", "GC=F"],
    )
    monkeypatch.setattr(source_snapshot, "BinanceKlinesClient", Spot)
    monkeypatch.setattr(source_snapshot, "BinanceFundingClient", Futures)
    monkeypatch.setattr(source_snapshot, "IngestionRepository", Repo)
    monkeypatch.setattr(source_snapshot, "fetch_macro_series", fetch_macro)
    first = await capture_sources(object(), settings, tmp_path)
    assert set(first["files"]) == {"candles", "macro", "sentiment", "funding"}
    assert set(requests) == {"spot", "funding", "macro", "sentiment"}
    requests.clear()
    assert await capture_sources(object(), settings, tmp_path) == first
    assert not requests
    (tmp_path / "sources" / "candles.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="huella"):
        await capture_sources(object(), settings, tmp_path)
    gap_directory = tmp_path / "gap"
    _write_spec(gap_directory)
    candles = candles.drop(index=10).reset_index(drop=True)
    with pytest.raises(ValueError, match="integridad o cobertura"):
        await capture_sources(object(), settings, gap_directory)
    assert not (gap_directory / "source-manifest.json").exists()


def _archive(day):
    times = pd.date_range(day.isoformat(), periods=288, freq="5min")
    frame = pd.DataFrame(
        {
            "create_time": times.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": "BTCUSDT",
        }
    )
    for index, name in enumerate(METRICS_COLUMNS):
        frame[name] = np.arange(len(times)) + 1000.0 + index
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"BTCUSDT-metrics-{day}.csv", frame.to_csv(index=False))
    return content.getvalue()


def _finalized_dataset(directory, market_data):
    _write_spec(directory)
    candles, macro, sentiment = _ohlcv(market_data)
    funding = _funding()
    funding["funding_rate"] = funding["funding_rate"].astype(float)
    funding["mark_price"] = funding["mark_price"].astype(float)
    sources = directory / "sources"
    sources.mkdir()
    files = {}
    for name, frame in {
        "candles": candles,
        "macro": macro,
        "sentiment": sentiment,
        "funding": funding,
    }.items():
        path = sources / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        files[name] = {
            "path": f"sources/{name}.parquet",
            "sha256": file_sha256(path),
            "rows": len(frame),
        }
    (directory / "source-manifest.json").write_text(
        json.dumps(
            {
                "files": files,
                "candle_synthetic_rows": 0,
            }
        )
    )
    archives = directory / "archives"
    archives.mkdir()
    entries = []
    start = date(2024, 1, 1)
    for i in range(35):
        day = start + timedelta(days=i)
        data = _archive(day)
        filename = f"BTCUSDT-metrics-{day}.zip"
        (archives / filename).write_bytes(data)
        _, audit = parse_daily_archive(data, "BTCUSDT", day)
        entries.append({**audit, "filename": filename, "sha256": hashlib.sha256(data).hexdigest()})
    (directory / "archive-manifest.json").write_text(json.dumps({"files": entries}))

    return finalize_dataset(directory), entries


def test_finalize_immutable_dataset_checks_asof_and_every_archive(tmp_path, market_data):
    manifest, entries = _finalized_dataset(tmp_path, market_data)
    assert manifest["point_in_time_verified"] is False
    assert manifest["quality"]["archive_days"] == 35
    assert manifest["quality"]["missing_5m_intervals"] == 0
    assert manifest["files"]["features"]["rows"] == 10
    assert "research/dataset.py" in manifest["implementation"]["files"]
    assert verify_dataset(tmp_path)["dataset_id"] == "test-version"
    frame = pd.read_parquet(tmp_path / "features.parquet")
    assert frame[OI_FEATURE_COLUMNS].notna().all().all()
    assert (frame["oi_available_at"] <= frame["open_time"] + STEP).all()
    assert (frame["open_time"] + 2 * STEP <= pd.Timestamp("2024-02-05", tz="UTC")).all()
    with pytest.raises(FileExistsError):
        finalize_dataset(tmp_path)
    (tmp_path / "archives" / entries[0]["filename"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="huella"):
        verify_dataset(tmp_path)


async def test_extending_dataset_preserves_old_sources_and_archive_versions(
    tmp_path, market_data, monkeypatch
):
    root = tmp_path / "datasets"
    root.mkdir()
    seed = root / "base"
    seed_manifest, entries = _finalized_dataset(seed, market_data)
    old_features = pd.read_parquet(seed / "features.parquet")
    requested_days = []

    async def fetch_day(self, archive_root, symbol, day):
        requested_days.append(day)
        filename = f"{symbol}-metrics-{day}.zip"
        metadata_path = archive_root / f"{filename}.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
        content = _archive(day)
        _, audit = parse_daily_archive(content, symbol, day)
        metadata = {**audit, "filename": filename, "sha256": hashlib.sha256(content).hexdigest()}
        (archive_root / filename).write_bytes(content)
        metadata_path.write_text(json.dumps(metadata))
        return metadata

    monkeypatch.setattr(MetricsArchiveClient, "fetch_day", fetch_day)
    start, end = date(2024, 1, 1), date(2024, 2, 6)
    target = await download_dataset_archives(root, "next", "BTCUSDT", start, end, seed_dataset=seed)
    assert len(requested_days) == 36
    assert (target / "archives" / entries[0]["filename"]).read_bytes() == (
        seed / "archives" / entries[0]["filename"]
    ).read_bytes()
    assert (
        json.loads((target / "build-spec.json").read_text())["seed_manifest_sha256"]
        == (seed_manifest["manifest_sha256"])
    )
    await download_dataset_archives(root, "next", "BTCUSDT", start, end, True, seed)

    candles, macro, sentiment = _ohlcv(market_data, 216)
    candles.loc[0, "close"] += 1
    macro.loc[0, "close"] += 1
    sentiment.loc[0, "value"] += 1
    funding = _funding()
    funding.loc[0, "funding_rate"] = "0.001"

    class Spot:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def fetch_klines(self, symbol, interval, start_time, end_time):
            assert start_time == pd.Timestamp("2024-02-05T00:00:00Z")
            return candles

    class Futures(Spot):
        async def fetch(self, symbol, start_time, end_time):
            return funding

    class Repo:
        def __init__(self, engine):
            pass

        async def load_fear_greed(self):
            return sentiment

    async def fetch_macro(*args):
        return macro

    settings = SimpleNamespace(
        btc_symbol="BTCUSDT",
        default_interval="4h",
        binance_base_url="https://example.test",
        binance_rate_limit_per_min=1200,
        binance_futures_base_url="https://example.test",
        funding_rate_limit_per_min=90,
        macro_symbols=["DX-Y.NYB", "^GSPC", "^VIX", "GC=F"],
    )
    monkeypatch.setattr(source_snapshot, "BinanceKlinesClient", Spot)
    monkeypatch.setattr(source_snapshot, "BinanceFundingClient", Futures)
    monkeypatch.setattr(source_snapshot, "IngestionRepository", Repo)
    monkeypatch.setattr(source_snapshot, "fetch_macro_series", fetch_macro)
    with pytest.raises(ValueError, match="requiere el dataset base"):
        await capture_sources(object(), settings, target)
    captured = await capture_sources(object(), settings, target, seed)
    assert captured["seed_manifest_sha256"] == seed_manifest["manifest_sha256"]
    finalize_dataset(target)
    extended_features = pd.read_parquet(target / "features.parquet")
    cutoff = pd.Timestamp("2024-02-05T00:00:00Z")
    columns = [
        "open_time",
        "target_label",
        "previous_direction",
        *variant_features()["base_oi_position"],
    ]
    pd.testing.assert_frame_equal(
        old_features[columns],
        extended_features.iloc[: len(old_features)][columns],
        check_exact=True,
        check_dtype=False,
    )
    assert history_hash(old_features, cutoff) == history_hash(extended_features, cutoff)
    assert verify_dataset(target)["point_in_time_verified"] is False

    (target / "archives" / entries[0]["filename"]).write_bytes(b"changed")
    assert verify_dataset(seed)["manifest_sha256"] == seed_manifest["manifest_sha256"]
    with pytest.raises(ValueError, match="huella"):
        verify_dataset(target)
