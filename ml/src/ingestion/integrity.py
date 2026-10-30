"""Chequeos de integridad del dataset de velas almacenado (tarea 1.7 / DoD Fase 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from src.ingestion.cleaning import Gap, detect_gaps, invalid_ohlcv_mask
from src.ingestion.freshness import check_freshness

if TYPE_CHECKING:
    from datetime import datetime, timedelta

MIN_COMPLETENESS = 0.995


@dataclass
class IntegrityReport:
    rows: int
    duplicates: int
    invalid_rows: int
    misaligned_rows: int
    gaps: list[Gap]
    synthetic_rows: int
    completeness: float
    span_days: float
    problems: list[str] = field(default_factory=list)
    last_open_time: datetime | None = None
    expected_open_time: datetime | None = None
    missing_recent: int | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


def check_candle_integrity(
    df: pd.DataFrame,
    step: timedelta,
    min_completeness: float = MIN_COMPLETENESS,
    min_span_days: float = 0.0,
    as_of: datetime | None = None,
    publish_delay: timedelta | None = None,
) -> IntegrityReport:
    """
    Verifica: sin duplicados, sin velas inválidas, alineación al intervalo, continuidad temporal
    y completitud. Las velas sintéticas (relleno de 1 período) se identifican por volumen y
    num_trades = 0 y NO cuentan como datos reales.
    """
    if df.empty:
        return IntegrityReport(0, 0, 0, 0, [], 0, 0.0, 0.0, ["dataset vacío"])

    ordered = df.sort_values("open_time").reset_index(drop=True)
    duplicates = int(ordered["open_time"].duplicated().sum())
    unique = ordered.drop_duplicates(subset="open_time").reset_index(drop=True)
    step_ts = pd.Timedelta(step)

    invalid = int(invalid_ohlcv_mask(unique).sum())
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    offset = (unique["open_time"] - epoch).dt.total_seconds() % step_ts.total_seconds()
    misaligned = int((offset != 0).sum())
    gaps = detect_gaps(unique["open_time"], step)
    synthetic = int(((unique["volume"] == 0) & (unique["num_trades"] == 0)).sum())

    missing = sum(g.missing for g in gaps)
    expected = len(unique) + missing
    completeness = (len(unique) - synthetic) / expected if expected else 1.0
    span_days = (unique["open_time"].iloc[-1] - unique["open_time"].iloc[0]) / pd.Timedelta(days=1)

    problems: list[str] = []
    if duplicates:
        problems.append(f"{duplicates} velas duplicadas")
    if invalid:
        problems.append(f"{invalid} velas con valores inválidos")
    if misaligned:
        problems.append(f"{misaligned} velas desalineadas del intervalo")
    if gaps:
        problems.append(f"{len(gaps)} gaps temporales ({missing} velas faltantes)")
    if completeness < min_completeness:
        problems.append(f"completitud {completeness:.4%} < {min_completeness:.2%}")
    if span_days < min_span_days:
        problems.append(f"histórico de {span_days:.0f} días < {min_span_days:.0f} requeridos")

    last_open_time = unique["open_time"].iloc[-1].to_pydatetime()
    expected_open_time = None
    missing_recent = None
    if as_of is not None:
        freshness = check_freshness(last_open_time, as_of, step, publish_delay)
        expected_open_time = freshness.expected_open_time
        missing_recent = freshness.missing_recent
        if missing_recent:
            problems.append(f"{missing_recent} velas recientes faltantes")
        if freshness.premature_candle:
            problems.append("hay una vela futura o aún sin cerrar")

    return IntegrityReport(
        rows=len(unique),
        duplicates=duplicates,
        invalid_rows=invalid,
        misaligned_rows=misaligned,
        gaps=gaps,
        synthetic_rows=synthetic,
        completeness=completeness,
        span_days=float(span_days),
        problems=problems,
        last_open_time=last_open_time,
        expected_open_time=expected_open_time,
        missing_recent=missing_recent,
    )
