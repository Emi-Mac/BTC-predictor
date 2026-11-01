"""Manejo centralizado de errores de dominio → respuestas HTTP (task 4.8)."""
from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.domain.exceptions import (
    InvalidFeatureVectorError,
    InvalidParameterError,
    ModelUnavailableError,
    NotFoundError,
)

log = structlog.get_logger(__name__)


async def _not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def _model_unavailable_handler(
    request: Request, exc: ModelUnavailableError
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


async def _invalid_feature_vector_handler(
    request: Request, exc: InvalidFeatureVectorError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def _invalid_parameter_handler(
    request: Request, exc: InvalidParameterError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_exception", path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def register_error_handlers(app: FastAPI) -> None:
    """Traduce las excepciones de dominio a códigos HTTP y evita filtrar stack traces."""
    app.add_exception_handler(NotFoundError, _not_found_handler)
    app.add_exception_handler(ModelUnavailableError, _model_unavailable_handler)
    app.add_exception_handler(InvalidFeatureVectorError, _invalid_feature_vector_handler)
    app.add_exception_handler(InvalidParameterError, _invalid_parameter_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
