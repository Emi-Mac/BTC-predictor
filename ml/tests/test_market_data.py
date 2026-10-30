from datetime import date

import httpx
import pandas as pd
import pytest

from src.config import Settings
from src.features.labels import make_target_label
from src.ingestion.market_data import fetch_fear_greed, normalize_macro_frame, parse_fear_greed
from src.ingestion.scheduler import build_trigger

FNG_PAYLOAD = {
    "data": [
        {"value": "51", "value_classification": "Neutral", "timestamp": "1793232000"},
        {"value": "20", "value_classification": "Extreme Fear", "timestamp": "1793145600"},
        {"value": "abc", "value_classification": "Bad", "timestamp": "1793059200"},
        {"value": "150", "value_classification": "OutOfRange", "timestamp": "1792972800"},
    ]
}


def test_parse_fear_greed_maps_utc_dates_and_skips_bad_rows():
    df = parse_fear_greed(FNG_PAYLOAD)
    assert df["obs_date"].tolist() == [date(2026, 10, 28), date(2026, 10, 29)]
    assert df["value"].tolist() == [20, 51]


def test_parse_fear_greed_empty():
    assert parse_fear_greed({"data": []}).empty


async def test_fetch_fear_greed_retries_then_parses():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        assert request.url.params["limit"] == "0"
        if attempts["n"] == 1:
            return httpx.Response(502)
        return httpx.Response(200, json=FNG_PAYLOAD)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        df = await fetch_fear_greed(
            "https://api.alternative.me/fng/", client=http, backoff_multiplier=0.001
        )
    assert len(df) == 2 and attempts["n"] == 2


def test_normalize_macro_frame_handles_multiindex_tz_and_nan():
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], tz="America/New_York")
    raw = pd.DataFrame({"Close": [1.0, None, 3.0], "Volume": [1, 2, 3]}, index=index)
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["^VIX"]])
    out = normalize_macro_frame(raw, "^VIX")
    assert out["trade_date"].tolist() == [date(2024, 1, 2), date(2024, 1, 4)]
    assert out["close"].tolist() == [1.0, 3.0]


def test_normalize_macro_frame_empty():
    assert normalize_macro_frame(pd.DataFrame(), "^VIX").empty


def test_target_label_uses_next_close_only():
    close = pd.Series([10.0, 11.0, 9.0, 9.0, 12.0])
    label = make_target_label(close)
    assert label.iloc[:-1].tolist() == [1, 0, 0, 1]
    assert pd.isna(label.iloc[-1])  # la última vela no tiene target


def test_target_label_not_affected_by_changes_after_t_plus_1():
    close = pd.Series([10.0, 11.0, 9.0, 12.0, 13.0])
    changed = close.copy()
    changed.iloc[3:] = [1.0, 1.0]
    assert make_target_label(close).iloc[0] == make_target_label(changed).iloc[0]
    assert make_target_label(close).iloc[1] == make_target_label(changed).iloc[1]


def test_scheduler_trigger_fires_at_each_4h_close():
    trigger = build_trigger("4h")
    fields = {f.name: str(f) for f in trigger.fields}
    assert fields["hour"] == "*/4" and fields["minute"] == "1"
    assert build_trigger("1h") is not None
    with pytest.raises(ValueError):
        build_trigger("7h")


def test_settings_parse_symbols_and_driver(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
    monkeypatch.setenv("MACRO_SYMBOLS", "DX-Y.NYB, ^GSPC,^VIX,GC=F")
    settings = Settings(_env_file=None)
    assert settings.macro_symbols == ["DX-Y.NYB", "^GSPC", "^VIX", "GC=F"]
    assert settings.database_url.startswith("postgresql+asyncpg://")
