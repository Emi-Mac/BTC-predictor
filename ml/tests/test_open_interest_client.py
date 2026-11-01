from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from src.ingestion.open_interest_client import BinanceOpenInterestClient
from src.ingestion.public_api import PublicApiClient

STEP_MS = 300_000
NOW = datetime(2026, 11, 1, 12, 30, tzinfo=UTC)


def make_client(handler):
    http = httpx.AsyncClient(
        base_url="https://fapi.binance.com", transport=httpx.MockTransport(handler)
    )
    api = PublicApiClient("https://fapi.binance.com", "oi", client=http, backoff_multiplier=0.001)
    api._limiter._min_interval = 0
    return BinanceOpenInterestClient(client=api), http


def row(timestamp, amount="100.5", value="5000000.25"):
    return {
        "symbol": "BTCUSDT",
        "sumOpenInterest": amount,
        "sumOpenInterestValue": value,
        "timestamp": timestamp,
    }


def ms(moment):
    return int(moment.timestamp() * 1000)


def windowed_handler(requests):
    """Replica el comportamiento de Binance: devuelve las últimas `limit` filas del rango."""

    def handler(request):
        params = request.url.params
        requests.append(dict(params))
        start, end, limit = int(params["startTime"]), int(params["endTime"]), int(params["limit"])
        first = -(-start // STEP_MS) * STEP_MS
        stamps = list(range(first, end + 1, STEP_MS))[-limit:]
        return httpx.Response(200, json=[row(stamp) for stamp in stamps])

    return handler


async def test_open_interest_pages_in_windows_without_gaps_or_duplicates():
    requests = []
    client, http = make_client(windowed_handler(requests))
    start = datetime(2026, 10, 30, tzinfo=UTC)
    frame = await client.fetch("BTCUSDT", start, now=NOW)
    await http.aclose()
    assert len(requests) > 1
    assert all(int(r["endTime"]) - int(r["startTime"]) < 500 * STEP_MS for r in requests)
    assert frame["create_time"].is_monotonic_increasing
    assert frame["create_time"].is_unique
    steps = frame["create_time"].diff().dropna().unique()
    assert list(steps) == [timedelta(minutes=5)]
    assert frame["create_time"].iloc[0] == start
    assert frame["create_time"].iloc[-1] == datetime(2026, 11, 1, 12, 30, tzinfo=UTC)
    assert frame["sum_open_interest"].iloc[0] == Decimal("100.5")


async def test_open_interest_start_is_clamped_to_available_history():
    requests = []
    client, http = make_client(windowed_handler(requests))
    await client.fetch("BTCUSDT", NOW - timedelta(days=90), now=NOW)
    await http.aclose()
    assert int(requests[0]["startTime"]) == ms(NOW - timedelta(days=29))


async def test_open_interest_rejects_invalid_payloads():
    start = datetime(2026, 11, 1, 12, tzinfo=UTC)
    stamp = ms(start)
    cases = [
        {"not": "a list"},
        ["not a dict"],
        [row(stamp + 1)],
        [row(float(stamp))],
        [row(True)],
        [row(stamp), row(stamp)],
        [row(stamp - STEP_MS)],
        [row(stamp, amount="NaN")],
        [row(stamp, amount="-1")],
        [row(stamp, value="-1")],
        [{**row(stamp), "symbol": "ETHUSDT"}],
        [row(ms(NOW) + STEP_MS)],
    ]
    for payload in cases:
        client, http = make_client(
            lambda _request, payload=payload: httpx.Response(200, json=payload)
        )
        with pytest.raises(ValueError):
            await client.fetch("BTCUSDT", start, now=NOW)
        await http.aclose()


async def test_open_interest_validates_time_arguments():
    client, http = make_client(lambda _request: httpx.Response(200, json=[]))
    with pytest.raises(ValueError, match="zona horaria"):
        await client.fetch("BTCUSDT", datetime(2026, 1, 1), now=NOW)
    with pytest.raises(ValueError, match="zona horaria"):
        await client.fetch("BTCUSDT", NOW, end=datetime(2026, 1, 1), now=NOW)
    with pytest.raises(ValueError, match="posterior"):
        await client.fetch("BTCUSDT", NOW + timedelta(hours=1), now=NOW)
    assert (await client.fetch("BTCUSDT", NOW - timedelta(hours=1), now=NOW)).empty
    await http.aclose()
    await BinanceOpenInterestClient().aclose()
