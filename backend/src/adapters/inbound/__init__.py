"""Adaptadores inbound: routers FastAPI, WebSocket y manejo de errores."""
from src.adapters.inbound.candle_router import router as candle_router
from src.adapters.inbound.error_handlers import register_error_handlers
from src.adapters.inbound.feature_router import router as feature_router
from src.adapters.inbound.metrics_router import router as metrics_router
from src.adapters.inbound.prediction_router import router as prediction_router
from src.adapters.inbound.volatility_router import risk_router
from src.adapters.inbound.volatility_router import router as volatility_router
from src.adapters.inbound.websocket_router import router as websocket_router

__all__ = [
    "candle_router",
    "feature_router",
    "metrics_router",
    "prediction_router",
    "register_error_handlers",
    "risk_router",
    "volatility_router",
    "websocket_router",
]
