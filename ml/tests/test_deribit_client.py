from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from src.ingestion.deribit_client import DeribitDvolClient
from src.ingestion.public_api import PublicApiClient

HOUR_MS = 3_600_000
NOW = datetime(2026, 11, 1, 12, 30, tzinfo=UTC)
T0 = datetime(2026, 11, 1, 6, tzinfo=UTC)


def make_client(handler):
    http = httpx.AsyncClient(
        base_url="https://www.deribit.com", transport=httpx.MockTransport(handler)
    )
    api = PublicApiClient(
        "https://www.deribit.com", "deribit", client=http, backoff_multiplier=0.001
    )
    api._limiter._min_interval = 0
    return DeribitDvolClient(client=api), http


def candle(ms, close=40.0):
    return [ms, close, close, close, close]


def ms(moment):
    return int(moment.timestamp() * 1000)


def reply(data, continuation=None):
    return httpx.Response(200, json={"result": {"data": data, "continuation": continuation}})


async def test_dvol_paginates_backwards_and_drops_the_open_hour():
    requests = []

    def handler(request):
        requests.append(dict(request.url.params))
        if len(requests) == 1:
            data = [candle(ms(T0 + timedelta(hours=h)), 40 + h) for h in (3, 4, 5, 6)]
            return reply(data, continuation=ms(T0 + timedelta(hours=2)))
        return reply([candle(ms(T0 + timedelta(hours=h)), 40 + h) for h in (0, 1, 2)])

    client, http = make_client(handler)
    frame = await client.fetch("BTC", T0, now=NOW)
    await http.aclose()
    # 12:00 (T0+6h) cierra a las 13:00 > ahora: es parcial y se descarta
    assert frame["open_time"].iloc[0] == T0
    assert frame["open_time"].iloc[-1] == T0 + timedelta(hours=5)
    assert len(frame) == 6
    assert frame["open_time"].is_monotonic_increasing
    assert frame["close"].iloc[0] == Decimal("40.0")
    assert requests[1]["end_timestamp"] == str(ms(T0 + timedelta(hours=2)))
    assert requests[0]["resolution"] == "3600"
    assert requests[0]["currency"] == "BTC"


async def test_dvol_rejects_invalid_payloads():
    bad = {
        "error": {"error": {"code": 1}},
        "list": [],
        "shape": {"result": {"data": [[1, 2]], "continuation": None}},
    }
    cases = [
        httpx.Response(200, json=bad["error"]),
        httpx.Response(200, json=bad["list"]),
        httpx.Response(200, json=bad["shape"]),
        reply([candle(ms(T0) + 1)]),
        reply([candle(ms(T0), 0)]),
        reply([candle(ms(T0), -1.0)]),
        reply([candle(ms(T0), "40")]),
        reply([candle(ms(T0))], continuation=ms(NOW) + 1),
        reply([candle(ms(T0))], continuation=True),
        reply([[True, 1, 1, 1, 1]]),
    ]
    for response in cases:
        client, http = make_client(lambda _request, response=response: response)
        with pytest.raises(ValueError):
            await client.fetch("BTC", T0, now=NOW)
        await http.aclose()


async def test_dvol_validates_time_arguments_and_page_limit():
    client, http = make_client(lambda _request: reply([]))
    with pytest.raises(ValueError, match="zona horaria"):
        await client.fetch("BTC", datetime(2026, 1, 1), now=NOW)
    with pytest.raises(ValueError, match="zona horaria"):
        await client.fetch("BTC", T0, end=datetime(2026, 1, 1), now=NOW)
    with pytest.raises(ValueError, match="posterior"):
        await client.fetch("BTC", NOW + timedelta(hours=1), now=NOW)
    assert (await client.fetch("BTC", T0, now=NOW)).empty
    await http.aclose()

    def endless(request):
        end = int(request.url.params["end_timestamp"])
        return reply([candle(ms(T0))], continuation=end - 1)

    client, http = make_client(endless)
    with pytest.raises(ValueError, match="máximo de páginas"):
        await client.fetch("BTC", NOW - timedelta(days=400), now=NOW)
    await http.aclose()


async def test_dvol_retries_server_errors():
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return reply([candle(ms(T0))])

    client, http = make_client(handler)
    frame = await client.fetch("BTC", T0, now=NOW)
    await http.aclose()
    assert calls == 3
    assert len(frame) == 1


async def test_owned_client_is_closed():
    client = DeribitDvolClient()
    await client.aclose()
    api = PublicApiClient("https://example.com", "x")
    async with api as entered:
        assert entered is api
    with pytest.raises(ValueError, match="límite"):
        PublicApiClient("https://example.com", "x", rate_limit_per_min=0)
