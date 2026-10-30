from datetime import UTC, datetime, timedelta

import httpx
import pytest

from src.ingestion.binance import BinanceKlinesClient, interval_to_timedelta, parse_klines

STEP_MS = 4 * 3600 * 1000


def kline(open_ms: int, price: float = 100.0) -> list:
    return [
        open_ms,
        str(price),
        str(price + 1),
        str(price - 1),
        str(price + 0.5),
        "10.5",
        open_ms + STEP_MS - 1,
        "1000.0",
        42,
        "6.0",
        "600.0",
        "0",
    ]


def make_client(handler, **kwargs) -> BinanceKlinesClient:
    http = httpx.AsyncClient(
        base_url="https://api.binance.com", transport=httpx.MockTransport(handler)
    )
    return BinanceKlinesClient(
        client=http, rate_limit_per_min=1_000_000, backoff_multiplier=0.001, **kwargs
    )


def test_interval_to_timedelta():
    assert interval_to_timedelta("4h") == timedelta(hours=4)
    assert interval_to_timedelta("15m") == timedelta(minutes=15)
    assert interval_to_timedelta("1d") == timedelta(days=1)
    with pytest.raises(ValueError):
        interval_to_timedelta("4x")


async def test_pagination_walks_forward_until_exhausted():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    base_ms = int(start.timestamp() * 1000)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert request.url.path == "/api/v3/klines"
        assert params["limit"] == "1000"
        first = int(params["startTime"])
        calls.append(first)
        total = 1500
        idx = (first - base_ms) // STEP_MS
        batch = [kline(base_ms + i * STEP_MS) for i in range(idx, min(idx + 1000, total))]
        return httpx.Response(200, json=batch)

    client = make_client(handler)
    end = start + timedelta(hours=4 * 1500)
    df = await client.fetch_klines("BTCUSDT", "4h", start, end)
    await client.aclose()

    assert len(df) == 1500
    assert df["open_time"].is_monotonic_increasing and df["open_time"].is_unique
    assert len(calls) == 2 and calls[1] == base_ms + 1000 * STEP_MS
    assert df["close_time"].iloc[0] == df["open_time"].iloc[0] + timedelta(hours=4, milliseconds=-1)


async def test_retries_on_429_and_5xx_then_succeeds():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    base_ms = int(start.timestamp() * 1000)
    statuses = iter([429, 503])
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        status = next(statuses, 200)
        if status != 200:
            return httpx.Response(status, headers={"Retry-After": "0"})
        if int(request.url.params["startTime"]) > base_ms:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[kline(base_ms)])

    client = make_client(handler)
    df = await client.fetch_klines("BTCUSDT", "4h", start, start + timedelta(hours=8))
    await client.aclose()
    assert len(df) == 1
    assert attempts["n"] >= 3


async def test_gives_up_after_max_attempts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = make_client(handler, max_attempts=3)
    with pytest.raises(httpx.HTTPStatusError):
        await client.fetch_klines(
            "BTCUSDT", "4h", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
        )
    await client.aclose()


async def test_does_not_retry_client_errors():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})

    client = make_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.fetch_klines(
            "NOPE", "4h", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
        )
    await client.aclose()
    assert attempts["n"] == 1


def test_parse_klines_drops_the_candle_still_open():
    step = timedelta(hours=4)
    now = datetime(2024, 1, 1, 5, 0, tzinfo=UTC)
    base_ms = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    rows = [kline(base_ms), kline(base_ms + STEP_MS)]  # la segunda cierra a las 08:00 > now
    df = parse_klines(rows, step, closed_before=now)
    assert len(df) == 1
    assert df["open_time"].iloc[0] == datetime(2024, 1, 1, tzinfo=UTC)


def test_parse_klines_empty():
    assert parse_klines([], timedelta(hours=4)).empty


def test_parse_klines_preserves_taker_buy_volumes():
    df = parse_klines([kline(0)], timedelta(hours=4))
    assert df.loc[0, "taker_buy_base_volume"] == 6.0
    assert df.loc[0, "taker_buy_quote_volume"] == 600.0


def test_parse_klines_rejects_missing_taker_fields():
    with pytest.raises(ValueError, match="volumen agresor"):
        parse_klines([kline(0)[:9]], timedelta(hours=4))
