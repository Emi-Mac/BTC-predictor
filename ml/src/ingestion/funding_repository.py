"""Persistencia idempotente y lectura de eventos históricos de funding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import sqlalchemy as sa
import structlog
from sqlalchemy.dialects.postgresql import insert

from src.ingestion.funding_client import FUNDING_COLUMNS, parse_decimal
from src.ingestion.tables import funding_rates

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger(__name__)
CHUNK_SIZE = 1000


class FundingRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def upsert(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        rows: list[dict[str, Any]] = []
        for item in data[FUNDING_COLUMNS].to_dict("records"):
            timestamp = pd.Timestamp(item["funding_time"])
            if timestamp.tzinfo is None:
                raise ValueError("funding_time debe incluir zona horaria")
            symbol = item["symbol"]
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("El símbolo de funding es inválido")
            rate = parse_decimal(item["funding_rate"], "funding_rate")
            raw_mark = item["mark_price"]
            mark = parse_decimal(raw_mark, "mark_price") if pd.notna(raw_mark) else None
            if mark is not None and mark <= 0:
                raise ValueError("mark_price debe ser positivo")
            rows.append(
                {
                    "symbol": symbol,
                    "funding_time": timestamp.to_pydatetime(),
                    "funding_rate": rate,
                    "mark_price": mark,
                }
            )
        inserted = 0
        async with self._engine.begin() as conn:
            for offset in range(0, len(rows), CHUNK_SIZE):
                stmt = insert(funding_rates).values(rows[offset : offset + CHUNK_SIZE])
                stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "funding_time"])
                result = await conn.execute(stmt)
                inserted += result.rowcount or 0
        log.info("funding_upserted", received=len(rows), inserted=inserted)
        return inserted

    async def last_funding_time(self, symbol: str) -> datetime | None:
        query = sa.select(sa.func.max(funding_rates.c.funding_time)).where(
            funding_rates.c.symbol == symbol
        )
        async with self._engine.connect() as conn:
            latest: datetime | None = (await conn.execute(query)).scalar()
        return latest

    async def load(self, symbol: str) -> pd.DataFrame:
        query = (
            sa.select(*[funding_rates.c[name] for name in FUNDING_COLUMNS])
            .where(funding_rates.c.symbol == symbol)
            .order_by(funding_rates.c.funding_time)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        frame = pd.DataFrame(rows, columns=FUNDING_COLUMNS)
        if not frame.empty:
            frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True)
            frame["funding_rate"] = frame["funding_rate"].astype(float)
            frame["mark_price"] = frame["mark_price"].astype(float)
        return frame
