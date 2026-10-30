"""Comprueba la última vela cerrada esperada, incluyendo una demora de publicación."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.ingestion.time_utils import floor_to_step


@dataclass(frozen=True)
class Freshness:
    expected_open_time: datetime
    latest_open_time: datetime | None
    missing_recent: int | None
    premature_candle: bool

    @property
    def stale(self) -> bool:
        return self.latest_open_time is None or bool(self.missing_recent) or self.premature_candle


def check_freshness(
    latest_open_time: datetime | None,
    now: datetime,
    step: timedelta,
    publish_delay: timedelta | None = None,
) -> Freshness:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("La fecha actual debe incluir zona horaria")
    if publish_delay is None:
        publish_delay = min(timedelta(minutes=1), step / 2)
    if step <= timedelta(0) or publish_delay < timedelta(0) or publish_delay >= step:
        raise ValueError("El intervalo y la demora de publicación son inválidos")
    now = now.astimezone(UTC)
    expected = floor_to_step(now - publish_delay, step) - step
    if latest_open_time is None:
        return Freshness(expected, None, None, False)
    if latest_open_time.tzinfo is None or latest_open_time.utcoffset() is None:
        raise ValueError("La última vela debe incluir zona horaria")
    latest = latest_open_time.astimezone(UTC)
    if floor_to_step(latest, step) != latest:
        raise ValueError("La última vela no está alineada con el intervalo")
    latest_closed = floor_to_step(now, step) - step
    return Freshness(
        expected_open_time=expected,
        latest_open_time=latest,
        missing_recent=max(0, (expected - latest) // step),
        premature_candle=latest > latest_closed,
    )


def funding_is_stale(latest: datetime | None, now: datetime, max_lag: timedelta) -> bool:
    if now.tzinfo is None or now.utcoffset() is None or max_lag <= timedelta(0):
        raise ValueError("Se requiere fecha UTC y tolerancia de funding positiva")
    if latest is None:
        return True
    if latest.tzinfo is None or latest.utcoffset() is None:
        raise ValueError("El evento de funding debe incluir zona horaria")
    if latest > now:
        raise ValueError("El evento de funding está en el futuro")
    return now - latest > max_lag
