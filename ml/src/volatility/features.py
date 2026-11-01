"""Features de volatilidad compartidas por la investigación (E-12/E-12b) y la producción.

Una sola definición de las varianzas pasadas/futuras y de las entradas del modelo, para que
entrenamiento, evaluación e inferencia no puedan divergir. `live_inputs` arma las entradas de
`har_x_live` desde las tablas crudas (PostgreSQL en producción, dataset congelado en la
investigación) reutilizando las mismas funciones de `features.long_horizon` y, por tanto, las
mismas reglas de disponibilidad y frescura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.economic import PERIODS_PER_YEAR_4H
from src.features.long_horizon import (
    event_features,
    funding_features,
    options_features,
    regime_features,
)

STEP = pd.Timedelta(hours=4)
HORIZONS = {"1d": 6, "7d": 42}
EWMA_HALFLIFE = 42
EPSILON = 1e-12
HAR_COLUMNS = ["log_var_1d", "log_var_7d", "log_var_30d"]
HAR_X_EXTRA = [
    "log_dvol_var",
    "log_vix",
    "funding_sum_7d",
    "oi_change_7d",
    "drawdown_90d",
    "abs_return_1d",
    "volume_sma_ratio",
    "fomc_next_72h",
]
# `har_x_live` excluye el interés abierto: el histórico de archivos solo es utilizable con 48 h
# de demora simulada y en vivo estaría disponible al instante (sesgo entre entrenamiento y
# servicio). Ver docs/plans/volatility-integration-plan.md, sección 6.
HAR_X_LIVE_EXTRA = [name for name in HAR_X_EXTRA if name != "oi_change_7d"]
LIVE_BASE_COLUMNS = ("vix_close", "volume_sma_ratio")
# Velas hacia atrás que necesitan las entradas: 540 (drawdown_90d) más margen.
LIVE_WINDOW_CANDLES = 600


def volatility_frame(close: pd.Series, open_times: pd.Series) -> pd.DataFrame:
    """Varianzas anualizadas pasadas (conocidas en t) y futuras (objetivo) por fila."""
    squared = np.log(close).diff() ** 2
    out = pd.DataFrame(index=close.index)
    for name, window in (("1d", 6), ("7d", 42), ("30d", 180)):
        out[f"var_{name}"] = (
            squared.rolling(window, min_periods=window).mean() * PERIODS_PER_YEAR_4H
        )
    out["var_ewma"] = (
        squared.ewm(halflife=EWMA_HALFLIFE, adjust=False, min_periods=EWMA_HALFLIFE).mean()
        * PERIODS_PER_YEAR_4H
    )
    for name, horizon in HORIZONS.items():
        future = squared[::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-1)
        out[f"target_var_{name}"] = future * PERIODS_PER_YEAR_4H
    out["abs_return_1d"] = np.log(close).diff(6).abs()
    out["forward_return"] = close.shift(-1) / close - 1
    times = pd.to_datetime(open_times, utc=True)
    return out.reindex(times).reset_index(drop=True)


def model_inputs(enriched: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    data = enriched.reset_index(drop=True).copy()
    vol = volatility_frame(close, data["open_time"])
    for column in vol:
        data[column] = vol[column].to_numpy()
    for name in ("1d", "7d", "30d"):
        data[f"log_var_{name}"] = np.log(data[f"var_{name}"].clip(lower=EPSILON))
    data["log_dvol_var"] = np.log((data["dvol"] ** 2).clip(lower=EPSILON))
    data["log_vix"] = np.log(data["vix_close"].clip(lower=EPSILON))
    for name in HORIZONS:
        data[f"log_target_{name}"] = np.log(data[f"target_var_{name}"].clip(lower=EPSILON))
    return data


def live_inputs(
    base: pd.DataFrame,
    close: pd.Series,
    funding: pd.DataFrame,
    dvol: pd.DataFrame,
    fomc: pd.DataFrame,
) -> pd.DataFrame:
    """Entradas de `har_x_live` por vela, desde tablas crudas.

    `base`: `open_time`, `vix_close` y `volume_sma_ratio` por vela (columnas de
    `computed_features`, ya alineadas con la regla de disponibilidad del macro y del volumen).
    `close`: cierre spot 4h en rejilla completa. `funding`: `funding_time`, `funding_rate`.
    `dvol`: `open_time`, `close` (puntos de volatilidad). `fomc`: `decision_time`.
    """
    times = pd.to_datetime(base["open_time"], utc=True).reset_index(drop=True)
    enriched = pd.DataFrame({"open_time": times})
    for column in LIVE_BASE_COLUMNS:
        enriched[column] = base[column].to_numpy(dtype=float)
    enriched["dvol"] = options_features(times, close, dvol)["dvol"].to_numpy()
    enriched["funding_sum_7d"] = funding_features(times, funding)["funding_sum_7d"].to_numpy()
    enriched["drawdown_90d"] = regime_features(times, close)["drawdown_90d"].to_numpy()
    enriched["fomc_next_72h"] = event_features(times, fomc)["fomc_next_72h"].to_numpy()
    return model_inputs(enriched, close)
