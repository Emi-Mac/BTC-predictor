"""Rellena el historial de pronósticos en una base de DESARROLLO reproduciendo el ciclo vela a vela.

Para ver el panel de riesgo con historial y calibración sin esperar semanas: por cada una de las
últimas N velas emite el pronóstico que el worker habría emitido en ese momento (con los datos
disponibles hasta entonces) y al final resuelve los que ya vencieron.

**Nunca contra la sombra ni producción:** los pronósticos reproducidos no son prospectivos. Por
eso se rechaza cualquier base cuyo nombre no contenga «dev».

    python -m src.volatility.replay --candles 330
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.ingestion.repository import IngestionRepository
from src.ingestion.run import configure_logging
from src.volatility.features import STEP
from src.volatility.inference import run_inference
from src.volatility.resolution import resolve_forecasts
from src.volatility_config import get_volatility_settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings
    from src.volatility_config import VolatilitySettings

log = structlog.get_logger(__name__)
ISSUE_DELAY = timedelta(minutes=2)  # el worker emite poco después del cierre de la vela


def assert_development_database(name: str | None) -> None:
    if not name or "dev" not in name.lower():
        msg = f"Solo se permite en bases de desarrollo ('dev' en el nombre): {name!r}"
        raise ValueError(msg)


async def replay(
    engine: AsyncEngine,
    settings: Settings,
    live: VolatilitySettings,
    candles: int,
    resolve_at: datetime | None = None,
    enforce_dev_database: bool = True,
) -> dict[str, int]:
    if enforce_dev_database:
        assert_development_database(engine.url.database)
    if candles < 1:
        raise ValueError("candles debe ser positivo")
    frame = await IngestionRepository(engine).load_candles(
        settings.btc_symbol, settings.default_interval
    )
    # La última vela se deja al ciclo real del worker; aquí solo las anteriores.
    opened = list(frame["open_time"])[-(candles + 1) : -1]
    created = 0
    for open_time in opened:
        now = (open_time + STEP + ISSUE_DELAY).to_pydatetime()
        created += await run_inference(engine, settings, live, now)
    resolved = await resolve_forecasts(engine, settings, resolve_at or datetime.now(UTC))
    log.info("volatility_replay_done", candles=len(opened), created=created, resolved=resolved)
    return {"candles": len(opened), "created": created, "resolved": resolved}


async def _main(candles: int) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        result = await replay(engine, settings, get_volatility_settings(), candles)
        print(result)  # noqa: T201
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce pronósticos pasados (solo desarrollo)")
    parser.add_argument("--candles", type=int, default=330, help="velas a reproducir (≈55 días)")
    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    asyncio.run(_main(args.candles))


if __name__ == "__main__":
    main()
