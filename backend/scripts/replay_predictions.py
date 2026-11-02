"""Reproduce las predicciones direccionales de las últimas velas en una base de DESARROLLO.

Para ver el gráfico, el historial y las métricas del modelo direccional en un entorno nuevo sin
esperar a que pasen las velas. Recorre las últimas N velas de la más antigua a la más reciente,
como lo haría el ciclo real (`run_prediction_cycle`): emite la predicción de cada vela con las
features guardadas y resuelve la de la vela anterior. Ajusta `predicted_at` al cierre de cada vela.

Solo para bases con «dev» en el nombre (las predicciones reproducidas no son prospectivas).

    docker exec btc_dev_backend python /app/scripts/replay_predictions.py --candles 330
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import sqlalchemy as sa

from src.config.database import AsyncSessionLocal, engine
from src.config.dependencies import (
    get_candle_repository,
    get_feature_repository,
    get_model_inference_adapter,
    get_model_registry_repository,
    get_prediction_repository,
)
from src.config.settings import get_settings


async def replay(count: int) -> dict[str, int]:
    database = engine.url.database or ""
    if "dev" not in database.lower():
        raise SystemExit(f"Solo se permite en bases de desarrollo ('dev' en el nombre): {database!r}")
    settings = get_settings()
    model = await get_model_registry_repository().get_active()
    if model is None or model.artifact_path is None:
        raise SystemExit("No hay modelo direccional activo en model_registry")
    inference = get_model_inference_adapter()
    inference.load(Path(model.artifact_path).parent, model.version)
    candles, _ = await get_candle_repository().list_candles(
        settings.btc_symbol, settings.default_interval, limit=count + 1, offset=0
    )
    predictions = get_prediction_repository()
    features = get_feature_repository()
    created = resolved = 0
    ordered = list(reversed(candles))  # de la más antigua a la más reciente
    for index, candle in enumerate(ordered):
        if not await predictions.exists_for_candle(candle.id, model.version):
            vector = await features.get_vector_for_candle(candle.id, model.feature_names)
            if vector is not None:
                outcome = inference.predict(vector)
                await predictions.create(
                    candle_id=candle.id,
                    model_version=model.version,
                    direction=outcome.direction,
                    confidence=outcome.confidence,
                    probability_up=outcome.probability_up,
                )
                created += 1
        if index > 0:
            previous = ordered[index - 1]
            actual = "UP" if candle.close > previous.close else "DOWN"
            resolved += await predictions.resolve_for_candle(previous.id, actual, candle.close_time)
    async with AsyncSessionLocal() as session:
        await session.execute(
            sa.text(
                "update predictions_log p set predicted_at = c.close_time "
                "from ohlcv_candles c where c.id = p.candle_id"
            )
        )
        await session.commit()
    return {"candles": len(ordered), "created": created, "resolved": resolved}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce predicciones direccionales (solo dev)")
    parser.add_argument("--candles", type=int, default=330)
    args = parser.parse_args()
    print(asyncio.run(replay(args.candles)))  # noqa: T201


if __name__ == "__main__":
    main()
