from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from src.ingestion.funding_client import BinanceFundingClient

HOUR_MS = 3_600_000


def make_client(handler, **kwargs):
    http = httpx.AsyncClient(
        base_url="https://fapi.binance.com", transport=httpx.MockTransport(handler)
    )
    client = BinanceFundingClient(client=http, backoff_multiplier=0.001, **kwargs)
    client._limiter._min_interval = 0
    return client, http


def funding_event(timestamp):
    return {
        "symbol": "BTCUSDT",
        "fundingTime": timestamp,
        "fundingRate": "0.00012345",
        "markPrice": "50000.12345678",
    }


async def test_funding_paginates_and_preserves_decimal_precision():
    start = datetime(2023, 1, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    calls = []

    def handler(request):
        cursor = int(request.url.params["startTime"])
        calls.append(cursor)
        i = max(0, -(-(cursor - start_ms) // (8 * HOUR_MS)))
        return httpx.Response(
            200,
            json=[funding_event(start_ms + n * 8 * HOUR_MS) for n in range(i, min(i + 1000, 1500))],
        )

    client, http = make_client(handler)
    frame = await client.fetch("BTCUSDT", start, start + timedelta(hours=8 * 1500))
    await http.aclose()
    assert len(frame) == 1500
    assert len(calls) == 3 and calls[1] == start_ms + 999 * 8 * HOUR_MS + 1
    assert frame["funding_time"].is_monotonic_increasing
    assert frame["funding_time"].iloc[0].tzinfo is not None
    assert frame["funding_rate"].iloc[0] == Decimal("0.00012345")
    assert frame["mark_price"].iloc[0] == Decimal("50000.12345678")


async def test_funding_retries_rate_limits_and_server_errors():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamp = int(start.timestamp() * 1000)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return httpx.Response([429, 503][calls - 1], headers={"Retry-After": "0"})
        return httpx.Response(200, json=[funding_event(timestamp)] if calls == 3 else [])

    client, http = make_client(handler)
    frame = await client.fetch("BTCUSDT", start, start + timedelta(hours=1))
    await http.aclose()
    assert len(frame) == 1 and calls == 4


async def test_funding_rejects_bad_payloads_and_unsupported_client_errors():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamp = int(start.timestamp() * 1000)
    client, http = make_client(
        lambda _: httpx.Response(200, json={"code": -1, "msg": "not a list"})
    )
    with pytest.raises(ValueError, match="formato"):
        await client.fetch("BTCUSDT", start, start + timedelta(hours=1))
    await http.aclose()

    bad = funding_event(timestamp)
    bad["fundingRate"] = "NaN"
    client, http = make_client(lambda _: httpx.Response(200, json=[bad]))
    with pytest.raises(ValueError, match="finito"):
        await client.fetch("BTCUSDT", start, start + timedelta(hours=1))
    await http.aclose()

    bad_time = funding_event(timestamp)
    bad_time["fundingTime"] = 1.5
    client, http = make_client(lambda _: httpx.Response(200, json=[bad_time]))
    with pytest.raises(ValueError, match="entero"):
        await client.fetch("BTCUSDT", start, start + timedelta(hours=1))
    await http.aclose()

    attempts = 0

    def reject(_):
        nonlocal attempts
        attempts += 1
        return httpx.Response(400)

    client, http = make_client(reject)
    with pytest.raises(httpx.HTTPStatusError):
        await client.fetch("BTCUSDT", start, start + timedelta(hours=1))
    await http.aclose()
    assert attempts == 1


async def test_funding_rejects_repeated_timestamps_and_naive_time():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    event = funding_event(int(start.timestamp() * 1000))
    client, http = make_client(lambda _: httpx.Response(200, json=[event, event]))
    with pytest.raises(ValueError, match="orden"):
        await client.fetch("BTCUSDT", start, start + timedelta(hours=1))
    with pytest.raises(ValueError, match="zona horaria"):
        await client.fetch("BTCUSDT", datetime(2024, 1, 1))
    await http.aclose()


async def test_historical_empty_mark_price_is_kept_as_missing():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    event = funding_event(int(start.timestamp() * 1000))
    event["markPrice"] = ""
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[event] if calls == 1 else [])

    client, http = make_client(handler)
    frame = await client.fetch("BTCUSDT", start, start + timedelta(hours=1))
    await http.aclose()
    assert frame["mark_price"].isna().all()
