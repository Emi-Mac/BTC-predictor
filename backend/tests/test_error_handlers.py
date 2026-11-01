"""Tests de `register_error_handlers`: excepciones de dominio → códigos HTTP."""
from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from src.adapters.inbound.error_handlers import register_error_handlers
from src.domain.exceptions import InvalidFeatureVectorError, ModelUnavailableError, NotFoundError


def _build_app() -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/not-found")
    async def _not_found():
        msg = "missing"
        raise NotFoundError(msg)

    @app.get("/unavailable")
    async def _unavailable():
        msg = "no model"
        raise ModelUnavailableError(msg)

    @app.get("/invalid")
    async def _invalid():
        msg = "bad vector"
        raise InvalidFeatureVectorError(msg)

    @app.get("/boom")
    async def _boom():
        msg = "unexpected"
        raise RuntimeError(msg)

    return app


@pytest.mark.asyncio
async def test_not_found_error_returns_404_with_message():
    app = _build_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/not-found")

    assert response.status_code == 404
    assert response.json() == {"detail": "missing"}


@pytest.mark.asyncio
async def test_model_unavailable_error_returns_503():
    app = _build_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/unavailable")

    assert response.status_code == 503
    assert response.json() == {"detail": "no model"}


@pytest.mark.asyncio
async def test_invalid_feature_vector_error_returns_422():
    app = _build_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/invalid")

    assert response.status_code == 422
    assert response.json() == {"detail": "bad vector"}


@pytest.mark.asyncio
async def test_unhandled_exception_returns_500_without_leaking_details():
    app = _build_app()
    # Starlette siempre re-lanza las excepciones genéricas tras el `ServerErrorMiddleware`
    # (para que un servidor ASGI real pueda loguearlas); `raise_app_exceptions=False` simula
    # el comportamiento real de un servidor y deja que el cliente sólo vea la respuesta 500.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
