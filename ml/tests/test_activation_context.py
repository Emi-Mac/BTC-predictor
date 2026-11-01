import json
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from src.features.compute import FEATURE_COLUMNS, compute_features
from src.features.context_store import context_records
from src.features.price_context import PRICE_CONTEXT_COLUMNS, add_price_context
from src.training.activation import activate, activation_problems
from tests.test_features import STEP

pytest_plugins = ["tests.test_features"]

FEATURES = ["rsi_14", "return_1"]


def _artifact(directory, feature_names=FEATURES, family="logistic"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.onnx").write_bytes(b"onnx")
    (directory / "calibration.json").write_text(json.dumps({"slope": 1, "intercept": 0}))
    (directory / "inference.json").write_text(json.dumps({"family": family}))
    (directory / "preprocessor.json").write_text(json.dumps({"feature_names": feature_names}))
    return str(directory / "model.onnx")


def test_activation_problems_detects_unservable_models(tmp_path):
    path = _artifact(tmp_path / "ok")
    assert activation_problems(FEATURES, path) == []
    assert "no persistidas" in activation_problems(["rsi_14", "oi_change_7d"], path)[0]
    assert "Faltan archivos" in activation_problems(FEATURES, str(tmp_path / "x/model.onnx"))[0]
    lstm = _artifact(tmp_path / "lstm", family="lstm")
    assert any("lstm" in p for p in activation_problems(FEATURES, lstm))
    other = _artifact(tmp_path / "other", feature_names=["return_1", "rsi_14"])
    assert any("no coinciden" in p for p in activation_problems(FEATURES, other))
    assert activation_problems([], None)[0] == "El modelo no registra feature_names"


def test_context_records_match_price_context_and_skip_incomplete_rows(market_data):
    candles, macro, fgi = market_data
    computed = compute_features(candles, macro, fgi, STEP).dropna(subset=FEATURE_COLUMNS)
    features = computed.rename(columns={"id": "candle_id"})
    records = context_records(features, candles, STEP)
    expected = add_price_context(features, candles, STEP).dropna(subset=PRICE_CONTEXT_COLUMNS)
    assert [r["candle_id"] for r in records] == expected["candle_id"].astype(int).tolist()
    first = expected.iloc[0]
    for name in PRICE_CONTEXT_COLUMNS:
        assert records[0]["feature_values"][name] == pytest.approx(first[name])
    assert context_records(features.iloc[0:0], candles, STEP) == []


@pytest.fixture
async def registry_engine(test_database_url):
    url = test_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    schema = f"activation_test_{uuid4().hex}"
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
        sa.Column("val_accuracy", sa.Numeric(6, 4)),
        sa.Column("artifact_path", sa.String),
        sa.Column("feature_names", sa.JSON),
        sa.Column("is_active", sa.Boolean, nullable=False),
        sa.Column("notes", sa.String),
    )
    try:
        async with engine.begin() as conn:
            await conn.run_sync(meta.create_all)
        yield engine, table
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(DropSchema(schema, cascade=True))
        await admin.dispose()


async def test_activate_leaves_a_single_active_version_with_reason(registry_engine, tmp_path):
    engine, table = registry_engine
    good = _artifact(tmp_path / "v2")
    async with engine.begin() as conn:
        await conn.execute(
            table.insert(),
            [
                {
                    "model_type": "lstm",
                    "val_accuracy": None,
                    "artifact_path": "/nope/model.onnx",
                    "feature_names": FEATURES,
                    "is_active": True,
                    "notes": "v1",
                },
                {
                    "model_type": "logistic",
                    "val_accuracy": 0.5536,
                    "artifact_path": good,
                    "feature_names": FEATURES,
                    "is_active": False,
                    "notes": "No activado",
                },
            ],
        )
    preview = await activate(engine, 2, "prueba", dry_run=True)
    assert preview["previously_active"] == [1] and preview["dry_run"] is True
    summary = await activate(engine, 2, "plan original")
    async with engine.connect() as conn:
        rows = (await conn.execute(sa.select(table).order_by(table.c.version))).mappings().all()
    assert [row["is_active"] for row in rows] == [False, True]
    assert "Activado manualmente" in rows[1]["notes"] and "plan original" in rows[1]["notes"]
    assert summary["val_accuracy"] == pytest.approx(0.5536)
    with pytest.raises(ValueError, match="No se puede activar"):
        await activate(engine, 1, "lstm")
    with pytest.raises(ValueError, match="No existe"):
        await activate(engine, 99, "x")
    with pytest.raises(ValueError, match="motivo"):
        await activate(engine, 2, "  ")
