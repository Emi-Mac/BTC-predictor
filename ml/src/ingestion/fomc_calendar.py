"""Calendario del FOMC: reuniones programadas, publicadas con antelación por la Reserva Federal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import structlog

from src.ingestion.public_api import PublicApiClient

if TYPE_CHECKING:
    from datetime import date

log = structlog.get_logger(__name__)
EVENT_TYPE = "fomc"
FOMC_BASE_URL = "https://www.federalreserve.gov"
FOMC_PAGES = {
    "/monetarypolicy/fomccalendars.htm": "current",
    "/monetarypolicy/fomchistorical2020.htm": "historical",
}
USER_AGENT = "Mozilla/5.0 (btc-predictor ingestion)"
EVENT_COLUMNS = ["event_type", "event_time", "source_url"]


def events_from_pages(pages: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """`pages` mapea la URL a (tipo de página, HTML); usa los parsers de la investigación."""
    # Importación diferida: `research.aux_sources` importa `ingestion.run`, que importa este módulo.
    from src.research.aux_sources import fomc_decisions, parse_fomc_calendar, parse_fomc_historical

    frames: list[pd.DataFrame] = []
    for url, (kind, html) in pages.items():
        meetings: list[date] = (
            parse_fomc_calendar(html) if kind == "current" else parse_fomc_historical(html)
        )
        if not meetings:
            raise ValueError(f"No se reconocieron reuniones en {url}")
        decisions = fomc_decisions(meetings)
        frames.append(
            pd.DataFrame(
                {
                    "event_type": EVENT_TYPE,
                    "event_time": decisions["decision_time"],
                    "source_url": url,
                }
            )
        )
    events = pd.concat(frames, ignore_index=True)
    events = events.drop_duplicates("event_time").sort_values("event_time")
    return events[EVENT_COLUMNS].reset_index(drop=True)


class FomcCalendarClient:
    def __init__(
        self,
        base_url: str = FOMC_BASE_URL,
        client: PublicApiClient | None = None,
    ) -> None:
        self._api = client or PublicApiClient(
            base_url,
            "fomc_calendar",
            rate_limit_per_min=10,
            max_attempts=4,
            headers={"User-Agent": USER_AGENT},
        )
        self._owns_client = client is None
        self._base_url = base_url

    async def aclose(self) -> None:
        if self._owns_client:
            await self._api.aclose()

    async def fetch(self) -> pd.DataFrame:
        pages: dict[str, tuple[str, str]] = {}
        for path, kind in FOMC_PAGES.items():
            response = await self._api.get(path)
            pages[f"{self._base_url}{path}"] = (kind, response.text)
        events = events_from_pages(pages)
        log.info("fomc_calendar_fetched", rows=len(events))
        return events
