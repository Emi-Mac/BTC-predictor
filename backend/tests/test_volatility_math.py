import math

import pytest

from src.domain import volatility_math as vm


def test_horizon_sigma_scales_with_the_square_root_of_time():
    assert vm.horizon_sigma(0.5, 42) == pytest.approx(0.5 * math.sqrt(42 / 2190))
    assert vm.horizon_sigma(0.5, 42) == pytest.approx(vm.horizon_sigma(0.5, 6) * math.sqrt(7))
    assert vm.horizon_sigma(0.0, 6) == 0.0
    for args in ((-0.1, 6), (0.5, 0)):
        with pytest.raises(ValueError, match="no puede ser negativa"):
            vm.horizon_sigma(*args)


def test_price_band_is_log_symmetric_and_validates_inputs():
    low, high = vm.price_band(100.0, 0.1, 1.5)
    assert high == pytest.approx(100 * math.exp(0.15))
    assert low * high == pytest.approx(100.0**2)
    assert vm.price_band(100.0, 0.0, 2.0) == (100.0, 100.0)
    for args in ((0.0, 0.1, 1.0), (100.0, -0.1, 1.0), (100.0, 0.1, -1.0)):
        with pytest.raises(ValueError, match="positivo"):
            vm.price_band(*args)


def test_realized_variance_matches_the_training_definition():
    closes = [100.0, 101.0, 99.0, 103.0]
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:], strict=False)]
    expected = sum(r * r for r in returns) / 3 * 2190
    assert vm.realized_variance(closes) == pytest.approx(expected)
    assert vm.realized_volatility(closes, 3) == pytest.approx(math.sqrt(expected))
    assert vm.realized_volatility(closes, 2) == pytest.approx(
        math.sqrt(vm.realized_variance(closes[-3:]))
    )
    assert vm.realized_volatility(closes, 4) is None  # faltan velas


def test_realized_variance_rejects_unusable_prices():
    for closes in ([100.0], [], [100.0, 0.0], [100.0, -5.0], [100.0, math.nan]):
        assert vm.realized_variance(closes) is None


def test_qlike_is_zero_only_for_a_perfect_forecast():
    assert vm.qlike(0.25, 0.25) == pytest.approx(0.0)
    assert vm.qlike(0.25, 0.5) > 0 and vm.qlike(0.5, 0.25) > 0
    assert math.isfinite(vm.qlike(0.0, 0.25))


def test_mincer_zarnowitz_slope():
    forecast = [0.1, 0.2, 0.4, 0.8]
    assert vm.mincer_zarnowitz_slope(forecast, forecast) == pytest.approx(1.0)
    squared = [f**2 for f in forecast]
    assert vm.mincer_zarnowitz_slope(squared, forecast) == pytest.approx(2.0)
    assert vm.mincer_zarnowitz_slope([0.1], [0.1]) is None
    assert vm.mincer_zarnowitz_slope([0.1, 0.2], [0.3, 0.3]) is None  # sin dispersión
    assert vm.mincer_zarnowitz_slope([0.1, 0.2], [0.3]) is None
