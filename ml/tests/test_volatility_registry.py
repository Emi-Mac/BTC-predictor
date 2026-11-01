"""Registro y activación de modelos de volatilidad (requiere TEST_DATABASE_URL)."""

from dataclasses import replace

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from src.ingestion.tables import metadata, volatility_models
from src.research import volatility_study as e12
from src.volatility import activation, registry
from src.volatility.model import VolatilityArtifact, fit_artifact
from tests.test_volatility_live_model import synthetic_bundle


@pytest.fixture
async def engine(test_database_url):
    url = test_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await eng.dispose()


@pytest.fixture(scope="module")
def development():
    data = synthetic_bundle()[4]
    return e12.development_rows(data, data["open_time"].iloc[-1])


@pytest.fixture
def artifacts(development, tmp_path):
    result = {}
    for horizon, candles in (("1d", 6), ("7d", 42)):
        for role, kind in (("primary", "har_x_live"), ("fallback", "har")):
            artifact = replace(fit_artifact(kind, development, horizon), validation={"qlike": 0.25})
            path = tmp_path / f"h{candles}-{role}.json"
            artifact.save(path)
            result[(candles, role)] = path
    return result


async def rows(engine):
    async with engine.connect() as conn:
        found = await conn.execute(sa.select(volatility_models).order_by("version"))
        return [dict(row) for row in found.mappings()]


async def test_register_inserts_inactive_models_with_hash_and_metadata(engine, artifacts):
    version = await registry.register(engine, artifacts[(42, "primary")], "primary", "nota")
    (row,) = await rows(engine)
    assert row["version"] == version
    assert (row["kind"], row["role"], row["horizon_candles"]) == ("har_x_live", "primary", 42)
    assert row["is_active"] is False
    assert row["artifact_sha256"] == registry.file_sha256(artifacts[(42, "primary")])
    assert row["validation_metrics"] == {"qlike": 0.25}
    assert "log_dvol_var" in row["feature_names"]
    assert row["notes"] == "nota"


async def test_register_rejects_unknown_roles_and_invalid_artifacts(engine, artifacts):
    with pytest.raises(ValueError, match="Rol desconocido"):
        await registry.register(engine, artifacts[(42, "primary")], "main")
    broken = replace(VolatilityArtifact.load(artifacts[(42, "fallback")]), scale=[0.0, 1.0, 1.0])
    broken.save(artifacts[(42, "fallback")])
    with pytest.raises(ValueError, match="Artefacto inválido"):
        await registry.register(engine, artifacts[(42, "fallback")], "fallback")
    assert await rows(engine) == []


async def test_activation_replaces_only_the_same_horizon_and_role(engine, artifacts):
    ids = {key: await registry.register(engine, path, key[1]) for key, path in artifacts.items()}
    first = await activation.activate(engine, ids[(42, "primary")], "E-12b aceptado")
    assert first["previously_active"] == []
    await activation.activate(engine, ids[(42, "fallback")], "respaldo")
    await activation.activate(engine, ids[(6, "primary")], "1d")
    state = {row["version"]: row["is_active"] for row in await rows(engine)}
    assert {v for v, active in state.items() if active} == {
        ids[(42, "primary")],
        ids[(42, "fallback")],
        ids[(6, "primary")],
    }
    # Mismo horizonte y rol: la versión anterior se desactiva
    newer = await registry.register(engine, artifacts[(42, "primary")], "primary")
    replaced = await activation.activate(engine, newer, "reentrenamiento mensual")
    assert replaced["previously_active"] == [ids[(42, "primary")]]
    state = {row["version"]: row["is_active"] for row in await rows(engine)}
    assert state[newer] is True
    assert state[ids[(42, "primary")]] is False
    assert state[ids[(42, "fallback")]] is True
    notes = {row["version"]: row["notes"] for row in await rows(engine)}
    assert "reentrenamiento mensual" in notes[newer]


async def test_dry_run_changes_nothing_and_reason_is_required(engine, artifacts):
    version = await registry.register(engine, artifacts[(6, "primary")], "primary")
    summary = await activation.activate(engine, version, "prueba", dry_run=True)
    assert summary["dry_run"] is True
    assert (await rows(engine))[0]["is_active"] is False
    with pytest.raises(ValueError, match="motivo"):
        await activation.activate(engine, version, "  ")
    with pytest.raises(ValueError, match="No existe la versión"):
        await activation.activate(engine, 999, "x")


async def test_activation_refuses_tampered_or_missing_artifacts(engine, artifacts):
    version = await registry.register(engine, artifacts[(6, "primary")], "primary")
    path = artifacts[(6, "primary")]
    original = path.read_text()
    path.write_text(original.replace('"kind": "har_x_live"', '"kind": "har"'))
    with pytest.raises(ValueError, match="huella"):
        await activation.activate(engine, version, "x")
    path.unlink()
    with pytest.raises(ValueError, match="No existe el artefacto"):
        await activation.activate(engine, version, "x")
    assert (await rows(engine))[0]["is_active"] is False


async def test_activation_detects_registry_and_artifact_mismatch(engine, artifacts):
    version = await registry.register(engine, artifacts[(6, "primary")], "primary")
    async with engine.begin() as conn:
        await conn.execute(
            volatility_models.update().values(kind="har", horizon_candles=42, feature_names=["a"])
        )
    with pytest.raises(ValueError, match="no coinciden"):
        await activation.activate(engine, version, "x")


async def test_database_enforces_one_active_model_per_horizon_and_role(engine, artifacts):
    first = await registry.register(engine, artifacts[(6, "primary")], "primary")
    second = await registry.register(engine, artifacts[(6, "primary")], "primary")
    async with engine.begin() as conn:
        await conn.execute(
            volatility_models.update()
            .where(volatility_models.c.version == first)
            .values(is_active=True)
        )
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                volatility_models.update()
                .where(volatility_models.c.version == second)
                .values(is_active=True)
            )
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(volatility_models.update().values(horizon_candles=7))
