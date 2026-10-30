"""Carga offline de las features persistidas y el contexto adicional del precio."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from src.features.funding_context import add_funding_context
from src.features.price_context import add_price_context
from src.features.repository import FeatureRepository
from src.features.taker_flow import add_taker_flow
from src.ingestion.binance import interval_to_timedelta
from src.ingestion.funding_repository import FundingRepository
from src.ingestion.repository import IngestionRepository

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings


async def load_contextual_features(
    engine: AsyncEngine,
    settings: Settings,
    include_taker: bool = False,
    include_funding: bool = False,
) -> pd.DataFrame:
    features = await FeatureRepository(engine).load_for_selection(
        settings.btc_symbol, settings.default_interval
    )
    candles = await IngestionRepository(engine).load_candles(
        settings.btc_symbol, settings.default_interval
    )
    step = interval_to_timedelta(settings.default_interval)
    features = add_price_context(features, candles, step)
    if include_taker:
        features = add_taker_flow(features, candles, step)
    if include_funding:
        funding = await FundingRepository(engine).load(settings.btc_symbol)
        features = add_funding_context(
            features, funding, step, timedelta(hours=settings.funding_feature_max_age_hours)
        )
    return features
