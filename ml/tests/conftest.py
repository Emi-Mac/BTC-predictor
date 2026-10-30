import os
from datetime import timedelta

import pandas as pd
import pytest

STEP = timedelta(hours=4)


def make_candles(n: int, start: str = "2024-01-01", skip: set[int] | None = None) -> pd.DataFrame:
    """Velas 4h sintéticas y válidas; `skip` omite posiciones para simular gaps."""
    skip = skip or set()
    times = [
        pd.Timestamp(start, tz="UTC") + i * pd.Timedelta(STEP) for i in range(n) if i not in skip
    ]
    rows = []
    for i, t in enumerate(times):
        price = 40_000.0 + i
        rows.append(
            {
                "open_time": t,
                "close_time": t + pd.Timedelta(STEP) - pd.Timedelta(milliseconds=1),
                "open": price,
                "high": price + 10,
                "low": price - 10,
                "close": price + 1,
                "volume": 100.0 + i,
                "quote_volume": 4_000_000.0,
                "num_trades": 1000 + i,
                "taker_buy_base_volume": None,
                "taker_buy_quote_volume": None,
            }
        )
    columns = ["open_time", "close_time", "open", "high", "low", "close"]
    columns += ["volume", "quote_volume", "num_trades"]
    columns += ["taker_buy_base_volume", "taker_buy_quote_volume"]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def step() -> timedelta:
    return STEP


@pytest.fixture
def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL no definido (tests de integración con PostgreSQL)")
    return url
