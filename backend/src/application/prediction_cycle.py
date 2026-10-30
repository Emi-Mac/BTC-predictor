"""Ciclo periódico de predicción: genera y resuelve predicciones sobre velas ya existentes.

La ingesta de velas y el cálculo de features son responsabilidad del servicio `ml-worker`
(ver `ml/src/ingestion` y `ml/src/features`). Este ciclo sólo consume datos ya persistidos
para producir y resolver predicciones usando el modelo ONNX activo. Se ejecuta desde el
scheduler de APScheduler configurado en el lifespan de FastAPI (ver `src.config.scheduler`).
"""
from __future__ import annotations

from pathlib import Path

import structlog

from src.domain.ports.candle_repository_port import CandleRepositoryPort
from src.domain.ports.feature_repository_port import FeatureRepositoryPort
from src.domain.ports.model_inference_port import ModelInferencePort
from src.domain.ports.model_registry_port import ModelRegistryPort
from src.domain.ports.prediction_repository_port import PredictionRepositoryPort

log = structlog.get_logger(__name__)


async def run_prediction_cycle(
    *,
    candle_repo: CandleRepositoryPort,
    feature_repo: FeatureRepositoryPort,
    model_registry: ModelRegistryPort,
    inference: ModelInferencePort,
    prediction_repo: PredictionRepositoryPort,
    symbol: str,
    interval: str,
) -> None:
    """Emite la predicción pendiente de la última vela y resuelve la de la anterior."""
    model = await model_registry.get_active()
    if model is None:
        log.warning("prediction_cycle_skipped_no_active_model")
        return
    if model.artifact_path is None:
        log.error("prediction_cycle_active_model_missing_artifact", version=model.version)
        return
    if not inference.is_ready or inference.loaded_version != model.version:
        inference.load(Path(model.artifact_path).parent, model.version)

    candles, _ = await candle_repo.list_candles(symbol, interval, limit=2, offset=0)
    if not candles:
        return
    latest, *rest = candles

    # 1) Emitir una nueva predicción para la vela más reciente, si no existe aún.
    if not await prediction_repo.exists_for_candle(latest.id, model.version):
        vector = await feature_repo.get_vector_for_candle(latest.id, model.feature_names)
        if vector is None:
            log.warning("prediction_cycle_missing_features", candle_id=latest.id)
        else:
            outcome = inference.predict(vector)
            await prediction_repo.create(
                candle_id=latest.id,
                model_version=model.version,
                direction=outcome.direction,
                confidence=outcome.confidence,
                probability_up=outcome.probability_up,
            )
            log.info(
                "prediction_emitted",
                candle_id=latest.id,
                direction=outcome.direction,
                confidence=outcome.confidence,
            )

    # 2) Resolver la predicción de la vela anterior, ahora que ya conocemos su resultado real.
    if rest:
        previous = rest[0]
        actual_direction = "UP" if latest.close > previous.close else "DOWN"
        resolved = await prediction_repo.resolve_for_candle(
            previous.id, actual_direction, latest.close_time
        )
        if resolved:
            log.info("prediction_resolved", candle_id=previous.id, actual=actual_direction)
