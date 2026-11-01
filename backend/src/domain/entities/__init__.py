"""Entidades de dominio (Pydantic inmutables) del backend."""
from src.domain.entities.candle import Candle, CandleWithChange
from src.domain.entities.feature import FeatureSnapshot
from src.domain.entities.metrics import AccuracyMetrics, MonthlyAccuracy
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.entities.pagination import Page, PaginationMeta, build_pagination
from src.domain.entities.prediction import Direction, Prediction

__all__ = [
    "AccuracyMetrics",
    "Candle",
    "CandleWithChange",
    "Direction",
    "FeatureSnapshot",
    "ModelMetadata",
    "MonthlyAccuracy",
    "Page",
    "PaginationMeta",
    "Prediction",
    "build_pagination",
]
