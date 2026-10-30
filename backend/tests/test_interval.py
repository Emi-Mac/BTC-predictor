"""Tests de la utilidad `interval_to_timedelta` del dominio."""
from __future__ import annotations

from datetime import timedelta

import pytest

from src.domain.interval import interval_to_timedelta


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("15m", timedelta(minutes=15)),
        ("1h", timedelta(hours=1)),
        ("4h", timedelta(hours=4)),
        ("1d", timedelta(days=1)),
        ("1w", timedelta(weeks=1)),
    ],
)
def test_interval_to_timedelta_parses_known_units(interval, expected):
    assert interval_to_timedelta(interval) == expected


@pytest.mark.parametrize("interval", ["", "4", "4x", "h4", "4H"])
def test_interval_to_timedelta_rejects_invalid_format(interval):
    with pytest.raises(ValueError, match="Intervalo no soportado"):
        interval_to_timedelta(interval)
