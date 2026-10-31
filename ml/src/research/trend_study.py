"""E-11: reglas clásicas de tendencia como referencia sin aprendizaje automático.

Reglas preespecificadas (sin ajuste): momentum de series de tiempo a 7, 30, 90 y 180 días y
cruce de medias de 50/200 días. Variantes largo/plano y largo/corto, con y sin dimensionamiento
por volatilidad. Contabilidad idéntica a E-08/E-09: posición decidida al cierre de t, aplicada
al retorno t→t+1, 7 pb por rotación y funding del perpetuo.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog

from src.config import get_settings
from src.evaluation.economic import PERIODS_PER_YEAR_4H, performance, strategy_returns
from src.evaluation.robustness import (
    probability_of_backtest_overfitting,
    reality_check,
    sharpe_difference_bootstrap,
)
from src.features.long_horizon import (
    FUNDING_MAX_AGE,
    asof_align,
    price_history,
    realized_volatility,
)
from src.ingestion.run import configure_logging
from src.research import aux_sources
from src.research import evaluation_harness as harness
from src.research.dataset import verify_dataset
from src.research.source_snapshot import file_sha256

log = structlog.get_logger(__name__)
SEED = harness.SEED
STEP = pd.Timedelta(hours=4)
MOMENTUM_LOOKBACKS = {"7d": 42, "30d": 180, "90d": 540, "180d": 1080}
MA_FAST, MA_SLOW = 300, 1200
TARGET_VOLATILITY = 0.5
VOLATILITY_WINDOW = 180
COST_BPS = 7.0
FUNDING_FRACTION_PER_CANDLE = 0.5
BLOCK = 180
PBO_PARTITIONS = 16
WINDOWS = {
    "oos_48m": ("2022-09-01", "2026-09-01"),
    "evaluable_e08_e09": ("2023-03-01", "2026-09-01"),
}
PRIMARY_WINDOW = "oos_48m"
IMPLEMENTATION_FILES = (
    "research/trend_study.py",
    "features/long_horizon.py",
    "evaluation/economic.py",
    "evaluation/robustness.py",
)


def raw_signals(close: pd.Series) -> pd.DataFrame:
    """Señales ±1 conocidas al cierre de cada vela (NaN mientras falte historia)."""
    log_price = np.log(close)
    signals = pd.DataFrame(index=close.index)
    for name, lookback in MOMENTUM_LOOKBACKS.items():
        change = log_price - log_price.shift(lookback)
        signals[f"tsmom_{name}"] = np.sign(change).where(change.notna())
    fast = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    slow = close.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    signals["ma_cross_50d_200d"] = np.where(fast > slow, 1.0, -1.0)
    signals.loc[slow.isna(), "ma_cross_50d_200d"] = np.nan
    return signals.replace(0.0, -1.0)


def strategy_positions(close: pd.Series) -> pd.DataFrame:
    signals = raw_signals(close)
    scale = (TARGET_VOLATILITY / realized_volatility(close, VOLATILITY_WINDOW)).clip(upper=1.0)
    positions = {}
    for rule in signals:
        for side in ("long_flat", "long_short"):
            base = signals[rule].clip(lower=0.0) if side == "long_flat" else signals[rule]
            positions[f"{rule}_{side}"] = base
            positions[f"{rule}_{side}_voltarget"] = base * scale
    return pd.DataFrame(positions)


def evaluate_window(
    positions: pd.DataFrame,
    forward_return: pd.Series,
    carry: pd.Series,
    hit_label: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    window = (positions.index >= start) & (positions.index < end)
    frame = positions.loc[window]
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Faltan señales en la ventana para {missing}")
    r1 = forward_return.loc[window].to_numpy(dtype=float)
    funding = carry.loc[window].to_numpy(dtype=float)
    label = hit_label.loc[window].to_numpy()
    spot = strategy_returns(np.ones(len(r1)), r1, 0.0)
    perpetual = strategy_returns(np.ones(len(r1)), r1, 0.0, funding)
    returns = {
        name: strategy_returns(frame[name].to_numpy(dtype=float), r1, COST_BPS, funding)
        for name in frame
    }
    names = list(returns)
    matrix = np.column_stack([returns[name] for name in names])
    check = reality_check(matrix - spot[:, None], BLOCK, seed=SEED)
    check["best_strategy"] = names[check.pop("best_column")]
    strategies = {}
    for name in names:
        position = frame[name].to_numpy(dtype=float)
        known = ~np.isnan(label)
        direction = np.sign(position)
        active = known & (direction != 0)
        strategies[name] = {
            **performance(returns[name], position),
            "sharpe_minus_buy_hold": sharpe_difference_bootstrap(
                returns[name], spot, BLOCK, PERIODS_PER_YEAR_4H, seed=SEED
            ),
            "hit_rate_3d_when_active": (
                float(((direction[active] > 0) == (label[active] > 0)).mean())
                if active.any()
                else None
            ),
        }
    return {
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "samples": len(r1),
        "buy_and_hold_spot": performance(spot, np.ones(len(r1))),
        "buy_and_hold_perpetual": performance(perpetual, np.ones(len(r1))),
        "reality_check_vs_buy_and_hold": check,
        "pbo_sharpe": probability_of_backtest_overfitting(matrix, PBO_PARTITIONS, "sharpe"),
        "strategies": strategies,
    }


def build_report(
    candles: pd.DataFrame, history: pd.DataFrame | None, funding: pd.DataFrame
) -> dict[str, Any]:
    close = price_history(candles, history)
    positions = strategy_positions(close)
    forward = close.shift(-1) / close - 1
    funding_rate = asof_align(
        pd.Series(close.index),
        funding[["funding_rate"]].reset_index(drop=True),
        pd.to_datetime(funding["funding_time"], utc=True).reset_index(drop=True) + STEP,
        FUNDING_MAX_AGE,
    )["funding_rate"]
    carry = pd.Series(
        funding_rate.to_numpy(dtype=float) * FUNDING_FRACTION_PER_CANDLE, index=close.index
    )
    label_3d = (close.shift(-18) > close).astype(float).where(close.shift(-18).notna())
    windows = {
        name: evaluate_window(
            positions,
            forward,
            carry.fillna(0.0),
            label_3d,
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"),
        )
        for name, (start, end) in WINDOWS.items()
    }
    primary = windows[PRIMARY_WINDOW]
    best = max(primary["strategies"], key=lambda n: primary["strategies"][n]["sharpe"])
    gate = {
        "best_strategy_by_sharpe": best,
        "best_sharpe": primary["strategies"][best]["sharpe"],
        "best_max_drawdown": primary["strategies"][best]["max_drawdown"],
        "buy_and_hold_spot_sharpe": primary["buy_and_hold_spot"]["sharpe"],
        "buy_and_hold_spot_max_drawdown": primary["buy_and_hold_spot"]["max_drawdown"],
        "best_sharpe_difference_ci_95": primary["strategies"][best]["sharpe_minus_buy_hold"][
            "ci_95"
        ],
        "reality_check_p_value": primary["reality_check_vs_buy_and_hold"]["p_value"],
        "pbo_sharpe": primary["pbo_sharpe"]["pbo"],
    }
    gate["beats_buy_and_hold_returns"] = bool(gate["reality_check_p_value"] <= 0.05)
    return {
        "evaluation": "trend_rules_reference",
        "model_registered": False,
        "rules": {
            "momentum_lookbacks_candles": MOMENTUM_LOOKBACKS,
            "ma_fast_slow_candles": [MA_FAST, MA_SLOW],
            "target_volatility": TARGET_VOLATILITY,
            "volatility_window_candles": VOLATILITY_WINDOW,
            "cost_bps": COST_BPS,
            "instrument": "perpetuo con funding (tasa/2 por vela)",
        },
        "pre_specified": {
            "primary_window": PRIMARY_WINDOW,
            "block": BLOCK,
            "gate": "White p <= 0.05 frente a buy & hold spot sobre retornos netos",
        },
        "windows": windows,
        "gate": gate,
    }


def execute(dataset: Path, aux_directory: Path, artifact_root: Path) -> Path:
    manifest = verify_dataset(dataset)
    aux_manifest = aux_sources.verify(aux_directory)
    root = Path(__file__).resolve().parents[1]
    implementation = {name: file_sha256(root / name) for name in IMPLEMENTATION_FILES}
    sources = json.loads((dataset / "source-manifest.json").read_text(encoding="utf-8"))
    candles = pd.read_parquet(dataset / sources["files"]["candles"]["path"])
    funding = pd.read_parquet(dataset / sources["files"]["funding"]["path"])
    aux = aux_sources.load(aux_directory)
    report = build_report(candles, aux["btc_spot_klines"], funding)
    name = f"trend-study-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    report.update(
        {
            "run_id": name,
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "aux_id": aux_manifest["aux_id"],
            "aux_manifest_sha256": aux_manifest["manifest_sha256"],
            "implementation": implementation,
        }
    )
    target = artifact_root / name
    target.mkdir(parents=True, exist_ok=False)
    path = target / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    harness.append_registry(
        artifact_root,
        {
            "run_id": name,
            "kind": report["evaluation"],
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": manifest["dataset_id"],
            "aux_id": aux_manifest["aux_id"],
            "strategies": len(report["windows"][PRIMARY_WINDOW]["strategies"]),
            "gate": report["gate"],
            "report_sha256": file_sha256(path),
        },
    )
    log.info("trend_study_complete", report=str(path), gate=report["gate"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reglas clásicas de tendencia (E-11)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--aux", type=Path, required=True)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    execute(args.dataset, args.aux, Path(settings.model_artifact_dir))


if __name__ == "__main__":
    main()
