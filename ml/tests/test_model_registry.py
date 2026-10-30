from uuid import uuid4

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from src.evaluation.calibration import PlattCalibration
from src.training.models import fit_xgb
from src.training.preprocess import Preprocessor
from src.training.registry import register_model
from src.training.walk_forward import ModelBundle


async def test_registers_inactive_model_and_final_fold_metrics(test_database_url, tmp_path):
    url = test_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    schema = f"registry_test_{uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as conn:
        await conn.execute(CreateSchema(schema))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    meta = sa.MetaData()
    table = sa.Table(
        "model_registry",
        meta,
        sa.Column("version", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model_type", sa.String, nullable=False),
        sa.Column("hyperparameters", sa.JSON, nullable=False),
        sa.Column("val_accuracy", sa.Numeric(6, 4)),
        sa.Column("val_f1", sa.Numeric(6, 4)),
        sa.Column("val_precision", sa.Numeric(6, 4)),
        sa.Column("val_recall", sa.Numeric(6, 4)),
        sa.Column("baseline_accuracy", sa.Numeric(6, 4)),
        sa.Column("artifact_path", sa.String),
        sa.Column("scaler_path", sa.String),
        sa.Column("feature_names", sa.JSON),
        sa.Column("is_active", sa.Boolean, nullable=False),
        sa.Column("notes", sa.String),
    )
    try:
        async with engine.begin() as conn:
            await conn.run_sync(meta.create_all)
        x = np.arange(60, dtype=np.float32).reshape(30, 2)
        model = fit_xgb(
            x,
            np.arange(30) % 2,
            {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 3, "subsample": 1.0},
            42,
        )
        pre = Preprocessor(["a", "b"], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
        bundle = ModelBundle("xgboost", model, pre, PlattCalibration(1, 0), {}, x)
        report = {
            "seed": 42,
            "criteria": {"passed": False},
            "final_fold_best_baseline": 0.52,
            "folds": [
                {
                    "calibration_start": "2025-01-01",
                    "validation_start": "2025-02-01",
                    "validation_end_exclusive": "2025-03-01",
                    "models": {
                        "xgboost": {
                            "calibrated": {
                                "accuracy": 0.50,
                                "f1_macro": 0.48,
                                "precision_up": 0.49,
                                "recall_up": 0.51,
                            }
                        }
                    },
                }
            ],
        }
        version = await register_model(engine, bundle, report, tmp_path)
        async with engine.connect() as conn:
            row = (
                (await conn.execute(sa.select(table).where(table.c.version == version)))
                .mappings()
                .one()
            )
        assert row["is_active"] is False
        assert float(row["val_accuracy"]) == pytest.approx(0.50)
        assert float(row["baseline_accuracy"]) == pytest.approx(0.52)
        assert row["feature_names"] == ["a", "b"]
        assert row["hyperparameters"]["quality_criteria"]["passed"] is False
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(DropSchema(schema, cascade=True))
        await admin.dispose()
