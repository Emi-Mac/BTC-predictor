"""Huella verificable de los datos utilizados en un experimento."""

import hashlib

import pandas as pd

from src.features.compute import FEATURE_COLUMNS
from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.features.taker_flow import TAKER_FEATURE_COLUMNS


def snapshot_hash(
    data: pd.DataFrame, include_taker: bool = False, include_funding: bool = False
) -> str:
    columns = [
        "open_time",
        "target_label",
        "previous_direction",
        *FEATURE_COLUMNS,
        *PRICE_CONTEXT_COLUMNS,
    ]
    if include_taker:
        columns.extend(TAKER_FEATURE_COLUMNS)
    if include_funding:
        columns.extend(FUNDING_FEATURE_COLUMNS)
    return hashlib.sha256(
        pd.util.hash_pandas_object(data[columns], index=False).to_numpy().tobytes()
    ).hexdigest()
