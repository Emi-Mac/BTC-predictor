"""Router REST para transparencia de features: features actuales del modelo."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.config.dependencies import get_feature_service
from src.config.settings import get_settings
from src.domain.entities.feature import FeatureSnapshot
from src.domain.ports.feature_port import FeaturePort

router = APIRouter(prefix="/api/v1/features", tags=["Features"])


@router.get("/latest", summary="Features actuales (transparencia)")
async def get_latest_features(
    service: FeaturePort = Depends(get_feature_service),
) -> FeatureSnapshot:
    settings = get_settings()
    return await service.get_latest_features(settings.btc_symbol, settings.default_interval)
