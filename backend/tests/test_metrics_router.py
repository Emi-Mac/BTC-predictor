"""Tests de `metrics_router` con `MetricsPort` sobrescrito (sin DB real)."""
from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from httpx import ASGITransport

from src.config.dependencies import get_metrics_service
from src.domain.entities.metrics import AccuracyMetrics
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.exceptions import NotFoundError
from src.domain.ports.metrics_port import MetricsPort
from src.main import app


class FakeMetricsService(MetricsPort):
    def __init__(self, accuracy=None, model=None, raise_not_found=False):
        self._accuracy = accuracy
        self._model = model
        self._raise_not_found = raise_not_found

    async def get_accuracy(self):
        return self._accuracy

    async def get_active_model(self):
        if self._raise_not_found:
            msg = "no active model"
            raise NotFoundError(msg)
        return self._model


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_accuracy_returns_payload():
    metrics = AccuracyMetrics(
        overall=0.56, last_30d=0.55, last_7d=0.6, total_resolved=100, by_month=[]
    )
    app.dependency_overrides[get_metrics_service] = lambda: FakeMetricsService(accuracy=metrics)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/metrics/accuracy")

    assert response.status_code == 200
    assert response.json()["overall"] == 0.56


@pytest.mark.asyncio
async def test_get_model_metadata_returns_payload():
    model = ModelMetadata(
        version=1,
        model_type="xgboost",
        val_accuracy=0.56,
        val_f1=0.55,
        val_precision=0.57,
        val_recall=0.54,
        baseline_accuracy=0.52,
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        is_active=True,
        feature_names=["rsi_14"],
        artifact_path="/artifacts/1/model.onnx",
        scaler_path="/artifacts/1/preprocessor.json",
    )
    app.dependency_overrides[get_metrics_service] = lambda: FakeMetricsService(model=model)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/metrics/model")

    assert response.status_code == 200
    assert response.json()["model_type"] == "xgboost"


@pytest.mark.asyncio
async def test_get_model_metadata_returns_404_when_no_active_model():
    app.dependency_overrides[get_metrics_service] = lambda: FakeMetricsService(
        raise_not_found=True
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/metrics/model")

    assert response.status_code == 404
