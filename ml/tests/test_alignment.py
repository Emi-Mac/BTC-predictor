from datetime import date

import pandas as pd

from src.ingestion.alignment import align_external_to_candles, macro_column_name
from tests.conftest import make_candles


def macro_frame(symbol: str, rows: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": [date.fromisoformat(d) for d in rows],
            "close": list(rows.values()),
        }
    )


def value_at(aligned: pd.DataFrame, ts: str, column: str) -> float:
    row = aligned[aligned["open_time"] == pd.Timestamp(ts, tz="UTC")]
    return row[column].iloc[0]


def test_macro_value_is_never_taken_from_the_future(step):
    # 2024-01-02 es martes. Su cierre solo es utilizable desde 2024-01-03 00:00 UTC.
    candles = make_candles(24, start="2024-01-02")  # velas 4h desde el martes 00:00
    macro = macro_frame("^VIX", {"2024-01-01": 10.0, "2024-01-02": 99.0})
    aligned = align_external_to_candles(candles, step, macro=macro)

    # Vela 16:00-20:00 del martes decide a las 20:00 -> solo conoce el cierre del lunes
    assert value_at(aligned, "2024-01-02 16:00", "vix_close") == 10.0
    # Vela 20:00-24:00 decide a las 00:00 del miércoles -> ya conoce el cierre del martes
    assert value_at(aligned, "2024-01-02 20:00", "vix_close") == 99.0


def test_weekend_is_forward_filled_from_friday(step):
    candles = make_candles(6 * 4, start="2024-01-06")  # sábado 6 a lunes 8 (4 días de velas 4h)
    macro = macro_frame("^GSPC", {"2024-01-04": 4700.0, "2024-01-05": 4750.0})  # jue, vie
    aligned = align_external_to_candles(candles, step, macro=macro)
    assert value_at(aligned, "2024-01-06 00:00", "sp500_close") == 4750.0
    assert value_at(aligned, "2024-01-07 12:00", "sp500_close") == 4750.0


def test_forward_fill_is_bounded(step):
    candles = make_candles(6 * 20, start="2024-01-02")
    macro = macro_frame("^VIX", {"2024-01-01": 10.0})
    aligned = align_external_to_candles(candles, step, macro=macro, max_ffill_days=3)
    assert value_at(aligned, "2024-01-03 00:00", "vix_close") == 10.0
    assert pd.isna(value_at(aligned, "2024-01-10 00:00", "vix_close"))


def test_fear_greed_available_from_its_utc_midnight(step):
    candles = make_candles(12, start="2024-01-01")
    fng = pd.DataFrame(
        {
            "obs_date": [date(2023, 12, 31), date(2024, 1, 1)],
            "value": [30, 80],
            "classification": ["Fear", "Extreme Greed"],
        }
    )
    aligned = align_external_to_candles(candles, step, fear_greed=fng)
    # Vela 00:00-04:00 decide a las 04:00 del día 1 -> ya conoce el valor del día 1
    assert value_at(aligned, "2024-01-01 00:00", "fear_greed_value") == 80
    assert value_at(aligned, "2024-01-01 20:00", "fear_greed_value") == 80


def test_fear_greed_not_leaked_before_publication(step):
    candles = make_candles(12, start="2023-12-31")
    fng = pd.DataFrame({"obs_date": [date(2024, 1, 1)], "value": [80], "classification": ["x"]})
    aligned = align_external_to_candles(candles, step, fear_greed=fng)
    # Velas que cierran antes de 2024-01-01 00:00 no pueden ver el valor del día 1
    assert pd.isna(value_at(aligned, "2023-12-31 16:00", "fear_greed_value"))
    assert value_at(aligned, "2023-12-31 20:00", "fear_greed_value") == 80


def test_output_keeps_one_row_per_candle_without_mutating_input(step):
    candles = make_candles(30)
    before = candles.copy()
    macro = macro_frame("GC=F", {"2023-12-31": 2000.0})
    aligned = align_external_to_candles(candles, step, macro=macro)
    assert len(aligned) == len(candles)
    assert aligned["open_time"].tolist() == candles["open_time"].tolist()
    pd.testing.assert_frame_equal(candles, before)


def test_empty_inputs(step):
    out = align_external_to_candles(make_candles(0), step)
    assert out.empty
    out = align_external_to_candles(make_candles(3), step)
    assert list(out.columns) == ["open_time"]


def test_macro_column_names():
    assert macro_column_name("DX-Y.NYB") == "dxy_close"
    assert macro_column_name("^TNX") == "tnx_close"
