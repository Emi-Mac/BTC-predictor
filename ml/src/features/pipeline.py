"""Pipeline offline ejecutado después de la ingesta."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.features.compute import FEATURE_COLUMNS, compute_features
from src.features.context_store import save_contextual_features
from src.features.repository import FeatureRepository
from src.ingestion.binance import interval_to_timedelta
from src.ingestion.pipeline import run_ingestion
from src.ingestion.repository import IngestionRepository
from src.volatility.inference import run_volatility_cycle
from src.volatility_config import get_volatility_settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings

log = structlog.get_logger(__name__)


async def run_features(engine: AsyncEngine, settings: Settings) -> int:
    source = IngestionRepository(engine)
    target = FeatureRepository(engine)
    candles = await target.load_candles(settings.btc_symbol, settings.default_interval)
    computed = compute_features(
        candles,
        await source.load_macro(),
        await source.load_fear_greed(),
        interval_to_timedelta(settings.default_interval),
        settings.macro_max_ffill_days,
    )
    complete = computed.dropna(subset=FEATURE_COLUMNS)
    if complete.empty:
        raise ValueError("No hay filas con todas las features disponibles")
    log.info("features_computed", total=len(computed), incomplete=len(computed) - len(complete))
    saved = await target.save(settings.btc_symbol, settings.default_interval, complete)
    # Contexto de precio requerido por los modelos `expanded` servidos por el backend.
    await save_contextual_features(engine, settings)
    live = get_volatility_settings()
    if live.volatility_sources_enabled:
        await run_volatility_cycle(engine, settings, live)
    return saved


async def run_cycle(repo: IngestionRepository, settings: Settings) -> None:
    await run_ingestion(repo, settings)
    await run_features(repo.engine, settings)
