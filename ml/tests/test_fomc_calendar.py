from datetime import UTC, datetime, timedelta

import httpx
import pytest

from src.ingestion.fomc_calendar import FomcCalendarClient, events_from_pages
from src.ingestion.public_api import PublicApiClient

CALENDAR_HTML = """
<a id="1">2025 FOMC Meetings</a>
<div class="fomc-meeting__month col"><strong>January</strong></div>
<div class="fomc-meeting__date col">28-29</div>
<div class="fomc-meeting__month col"><strong>Apr/May</strong></div>
<div class="fomc-meeting__date col">30-1*</div>
"""
HISTORICAL_HTML = """
<h5 class="x">January 28-29 Meeting - 2020</h5>
<h5 class="x">March 15 (unscheduled) Meeting - 2020</h5>
"""


def test_events_use_new_york_decision_time_and_remember_the_source():
    events = events_from_pages(
        {
            "https://fed/current": ("current", CALENDAR_HTML),
            "https://fed/history": ("historical", HISTORICAL_HTML),
        }
    )
    assert events["event_type"].unique().tolist() == ["fomc"]
    assert events["event_time"].is_monotonic_increasing
    assert events["event_time"].iloc[0] == datetime(2020, 1, 29, 19, tzinfo=UTC)
    assert events["event_time"].iloc[1] == datetime(2025, 1, 29, 19, tzinfo=UTC)
    assert events["event_time"].iloc[2] == datetime(2025, 5, 1, 18, tzinfo=UTC)
    assert events["source_url"].tolist() == [
        "https://fed/history",
        "https://fed/current",
        "https://fed/current",
    ]


def test_events_reject_pages_without_meetings():
    with pytest.raises(ValueError, match="No se reconocieron reuniones"):
        events_from_pages({"https://fed/current": ("current", "<html></html>")})


async def test_client_fetches_both_pages():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        html = HISTORICAL_HTML if "historical" in request.url.path else CALENDAR_HTML
        return httpx.Response(200, text=html)

    http = httpx.AsyncClient(base_url="https://fed", transport=httpx.MockTransport(handler))
    api = PublicApiClient("https://fed", "fomc", client=http)
    api._limiter._min_interval = 0
    client = FomcCalendarClient(base_url="https://fed", client=api)
    events = await client.fetch()
    await http.aclose()
    assert len(paths) == 2
    assert len(events) == 3
    assert events["event_time"].max() < datetime.now(UTC) + timedelta(days=365 * 10)
    await FomcCalendarClient().aclose()
