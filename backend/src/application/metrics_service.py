"""Servicio de métricas: implementa `MetricsPort` sobre predicciones y el registro de modelos."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.domain.entities.metrics import AccuracyMetrics
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.entities.prediction import DirectionStatus
from src.domain.exceptions import NotFoundError
from src.domain.ports.metrics_port import MetricsPort
from src.domain.ports.model_registry_port import ModelRegistryPort
from src.domain.ports.prediction_repository_port import PredictionRepositoryPort


EVIDENCE_PATH = "docs/research/experiment-log.md"


def _ratio(correct: int, total: int) -> float | None:
    return (correct / total) if total else None


class MetricsService(MetricsPort):
    """Caso de uso de métricas históricas y metadata del modelo activo."""

    def __init__(
        self,
        prediction_repo: PredictionRepositoryPort,
        model_registry: ModelRegistryPort,
        model_status: DirectionStatus = "experimental",
    ) -> None:
        self._prediction_repo = prediction_repo
        self._model_registry = model_registry
        self._model_status = model_status

    async def get_accuracy(self) -> AccuracyMetrics:
        now = datetime.now(UTC)
        overall_correct, overall_total = await self._prediction_repo.accuracy_since(None)
        correct_30d, total_30d = await self._prediction_repo.accuracy_since(
            now - timedelta(days=30)
        )
        correct_7d, total_7d = await self._prediction_repo.accuracy_since(now - timedelta(days=7))
        by_month = await self._prediction_repo.accuracy_by_month()
        return AccuracyMetrics(
            overall=_ratio(overall_correct, overall_total),
            last_30d=_ratio(correct_30d, total_30d),
            last_7d=_ratio(correct_7d, total_7d),
            total_resolved=overall_total,
            by_month=by_month,
            baseline_overall=await self._prediction_repo.majority_direction_share(),
        )

    async def get_active_model(self) -> ModelMetadata:
        model = await self._model_registry.get_active()
        if model is None:
            msg = "No hay ningún modelo activo en model_registry"
            raise NotFoundError(msg)
        beats = (
            None
            if model.val_accuracy is None or model.baseline_accuracy is None
            else model.val_accuracy > model.baseline_accuracy
        )
        return model.model_copy(
            update={
                "best_baseline_accuracy": model.baseline_accuracy,
                "beats_baseline": beats,
                "status": self._model_status,
                "evidence_url": EVIDENCE_PATH,
            }
        )
