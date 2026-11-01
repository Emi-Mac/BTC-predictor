"""Diagnósticos de robustez estadística para series temporales financieras.

Referencias: López de Prado (2018), *Advances in Financial Machine Learning* (CPCV, purga y
embargo); Bailey et al. (2017), *The Probability of Backtest Overfitting* (CSCV/PBO); Bailey y
López de Prado (2014), *The Deflated Sharpe Ratio*; White (2000), *A Reality Check for Data
Snooping*.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.stats import kurtosis, norm, skew

if TYPE_CHECKING:
    from src.training.types import FloatArray, IntArray

EULER_GAMMA = 0.5772156649015329


# ─── Bootstrap por bloques ───────────────────────────────────────────────────


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> IntArray:
    if n < 1 or not 1 <= block <= n:
        raise ValueError("El bloque debe estar entre 1 y la longitud de la serie")
    starts = rng.integers(0, n - block + 1, size=-(-n // block))
    result: IntArray = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return result


def block_bootstrap_mean(
    values: FloatArray, block: int, draws: int = 2000, seed: int = 42
) -> dict[str, Any]:
    """IC 95% de la media preservando autocorrelación de corto plazo."""
    series = np.asarray(values, dtype=float)
    if not len(series) or not np.isfinite(series).all() or draws < 1:
        raise ValueError("Se requieren valores finitos y remuestreos positivos")
    rng = np.random.default_rng(seed)
    means = np.array(
        [series[moving_block_indices(len(series), block, rng)].mean() for _ in range(draws)]
    )
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(series.mean()),
        "ci_95": [float(lower), float(upper)],
        "fraction_draws_above_zero": float((means > 0).mean()),
        "block": block,
        "draws": draws,
        "seed": seed,
    }


def reality_check(
    differences: FloatArray, block: int, draws: int = 2000, seed: int = 42
) -> dict[str, Any]:
    """Prueba de White sobre el máximo de T×N diferencias de desempeño contra un benchmark.

    H0: ningún candidato supera al benchmark. Corrige la selección del mejor entre N.
    """
    matrix = np.asarray(differences, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all() or draws < 1:
        raise ValueError("Se requiere una matriz T×N finita")
    observed = matrix.mean(axis=0)
    statistic = float(np.sqrt(len(matrix)) * observed.max())
    rng = np.random.default_rng(seed)
    maxima = np.empty(draws)
    for index in range(draws):
        resampled = matrix[moving_block_indices(len(matrix), block, rng)].mean(axis=0)
        maxima[index] = np.sqrt(len(matrix)) * (resampled - observed).max()
    return {
        "best_column": int(observed.argmax()),
        "best_mean_difference": float(observed.max()),
        "p_value": float((maxima >= statistic).mean()),
        "candidates": int(matrix.shape[1]),
        "block": block,
        "draws": draws,
        "seed": seed,
    }


def sharpe_difference_bootstrap(
    first: FloatArray,
    second: FloatArray,
    block: int,
    periods_per_year: int,
    draws: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """IC por bloques (remuestreo conjunto) de Sharpe(first) − Sharpe(second), anualizado."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape or len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Se requieren dos series finitas alineadas")

    def sharpe(values: FloatArray) -> float:
        std = float(values.std(ddof=1))
        return float(values.mean()) / std * np.sqrt(periods_per_year) if std > 0 else 0.0

    rng = np.random.default_rng(seed)
    differences = np.empty(draws)
    for index in range(draws):
        rows = moving_block_indices(len(a), block, rng)
        differences[index] = sharpe(a[rows]) - sharpe(b[rows])
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": sharpe(a) - sharpe(b),
        "ci_95": [float(lower), float(upper)],
        "fraction_draws_above_zero": float((differences > 0).mean()),
        "block": block,
        "draws": draws,
        "seed": seed,
    }


def simultaneous_lower_bounds(
    differences: FloatArray, block: int, draws: int = 2000, seed: int = 42, level: float = 0.95
) -> dict[str, Any]:
    """Límites inferiores simultáneos para las medias de K series de diferencias (T×K).

    Usa el cuantil bootstrap por bloques del máximo desvío centrado; todos los límites se
    cumplen a la vez con probabilidad ≈ `level`, lo que permite seleccionar a posteriori.
    """
    matrix = np.asarray(differences, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all() or draws < 1:
        raise ValueError("Se requiere una matriz T×K finita")
    if not 0 < level < 1:
        raise ValueError("El nivel debe estar en (0, 1)")
    observed = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    maxima = np.empty(draws)
    for index in range(draws):
        resampled = matrix[moving_block_indices(len(matrix), block, rng)].mean(axis=0)
        maxima[index] = (resampled - observed).max()
    quantile = float(np.quantile(maxima, level))
    return {
        "means": observed,
        "lower_bounds": observed - quantile,
        "quantile": quantile,
        "level": level,
        "block": block,
        "draws": draws,
        "seed": seed,
    }


# ─── Combinatorial Purged Cross-Validation ───────────────────────────────────


@dataclass(frozen=True)
class CpcvSplit:
    test_groups: tuple[int, ...]
    train: IntArray
    test: IntArray


def cpcv_splits(
    n_samples: int, n_groups: int, n_test_groups: int, purge: int, embargo: int
) -> tuple[list[IntArray], list[CpcvSplit]]:
    """Grupos contiguos; elimina del train las filas cuya etiqueta alcanza un bloque de test
    (`purge` filas antes) y las inmediatamente posteriores (`embargo` filas después)."""
    if not 2 <= n_groups <= n_samples or not 1 <= n_test_groups < n_groups:
        raise ValueError("Configuración CPCV inválida")
    if purge < 0 or embargo < 0:
        raise ValueError("Purga y embargo no pueden ser negativos")
    groups = np.array_split(np.arange(n_samples), n_groups)
    splits = []
    for chosen in combinations(range(n_groups), n_test_groups):
        test = np.concatenate([groups[g] for g in chosen])
        excluded = np.zeros(n_samples, dtype=bool)
        for g in chosen:
            start, end = groups[g][0], groups[g][-1]
            excluded[max(0, start - purge) : min(n_samples, end + embargo + 1)] = True
        splits.append(CpcvSplit(tuple(chosen), np.flatnonzero(~excluded), test))
    return groups, splits


def cpcv_paths(n_groups: int, n_test_groups: int) -> list[dict[int, int]]:
    """Asigna a cada trayectoria, para cada grupo, el índice del split que lo predice.

    Hay C(N-1, k-1) trayectorias completas; cada split aporta a k de ellas.
    """
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("Configuración CPCV inválida")
    combos = list(combinations(range(n_groups), n_test_groups))
    per_group: dict[int, list[int]] = {g: [] for g in range(n_groups)}
    for index, chosen in enumerate(combos):
        for g in chosen:
            per_group[g].append(index)
    count = comb(n_groups - 1, n_test_groups - 1)
    return [{g: per_group[g][path] for g in range(n_groups)} for path in range(count)]


# ─── Probability of Backtest Overfitting (CSCV) ──────────────────────────────


def _block_metric(
    sums: FloatArray, squares: FloatArray, counts: FloatArray, metric: str
) -> FloatArray:
    mean = sums / counts
    if metric == "mean":
        return mean
    variance = np.maximum(squares / counts - mean**2, 0) * counts / np.maximum(counts - 1, 1)
    std = np.sqrt(variance)
    result: FloatArray = np.where(std > 0, mean / np.where(std > 0, std, 1), 0.0)
    return result


def probability_of_backtest_overfitting(
    performance: FloatArray, n_partitions: int = 16, metric: Literal["mean", "sharpe"] = "mean"
) -> dict[str, Any]:
    """CSCV: elige el mejor candidato en la mitad in-sample y mide su rango fuera de muestra.

    PBO es la fracción de combinaciones donde el elegido queda en la mitad inferior OOS. Solo
    cuantifica sobreajuste por selección entre los candidatos dados; no es un p-valor.
    """
    matrix = np.asarray(performance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("Se requiere una matriz T×N finita con al menos dos candidatos")
    if n_partitions < 2 or n_partitions % 2 or n_partitions > len(matrix):
        raise ValueError("El número de particiones debe ser par y no mayor que T")
    blocks = np.array_split(np.arange(len(matrix)), n_partitions)
    sums = np.array([matrix[b].sum(axis=0) for b in blocks])
    squares = np.array([(matrix[b] ** 2).sum(axis=0) for b in blocks])
    counts = np.array([len(b) for b in blocks], dtype=float)[:, None]
    total_s, total_q, total_c = sums.sum(axis=0), squares.sum(axis=0), counts.sum()
    candidates = matrix.shape[1]
    logits = []
    degradation = []
    for chosen in combinations(range(n_partitions), n_partitions // 2):
        index = list(chosen)
        in_s, in_q, in_c = sums[index].sum(axis=0), squares[index].sum(axis=0), counts[index].sum()
        is_metric = _block_metric(in_s, in_q, np.full(candidates, in_c), metric)
        oos_metric = _block_metric(
            total_s - in_s, total_q - in_q, np.full(candidates, total_c - in_c), metric
        )
        best = int(is_metric.argmax())
        rank = float(
            (oos_metric < oos_metric[best]).sum()
            + 1
            + 0.5 * ((oos_metric == oos_metric[best]).sum() - 1)
        )
        omega = rank / (candidates + 1)
        logits.append(np.log(omega / (1 - omega)))
        degradation.append((float(is_metric[best]), float(oos_metric[best])))
    values = np.asarray(logits)
    pairs = np.asarray(degradation)
    return {
        "pbo": float((values <= 0).mean()),
        "combinations": len(values),
        "partitions": n_partitions,
        "candidates": candidates,
        "metric": metric,
        "median_logit": float(np.median(values)),
        "mean_in_sample_best": float(pairs[:, 0].mean()),
        "mean_out_of_sample_of_best": float(pairs[:, 1].mean()),
    }


def maximin_pbo(stack: FloatArray, n_partitions: int = 16) -> dict[str, Any]:
    """PBO (CSCV) cuando se elige el candidato con mayor mínimo entre B criterios.

    `stack` tiene forma B×T×N (p. ej. diferencias de acierto contra B baselines). En cada
    combinación se selecciona por max_n min_b de la media in-sample y se ordena fuera de muestra
    con el mismo criterio.
    """
    values = np.asarray(stack, dtype=float)
    if values.ndim != 3 or values.shape[2] < 2 or not np.isfinite(values).all():
        raise ValueError("Se requiere un arreglo B×T×N finito con al menos dos candidatos")
    periods = values.shape[1]
    if n_partitions < 2 or n_partitions % 2 or n_partitions > periods:
        raise ValueError("El número de particiones debe ser par y no mayor que T")
    blocks = np.array_split(np.arange(periods), n_partitions)
    sums = np.stack([values[:, b, :].sum(axis=1) for b in blocks])
    counts = np.array([len(b) for b in blocks], dtype=float)
    total_sums, total_count = sums.sum(axis=0), counts.sum()
    candidates = values.shape[2]
    logits = []
    for chosen in combinations(range(n_partitions), n_partitions // 2):
        index = list(chosen)
        in_count = counts[index].sum()
        in_metric = (sums[index].sum(axis=0) / in_count).min(axis=0)
        out_metric = ((total_sums - sums[index].sum(axis=0)) / (total_count - in_count)).min(axis=0)
        best = int(in_metric.argmax())
        ties = (out_metric == out_metric[best]).sum() - 1
        rank = float((out_metric < out_metric[best]).sum() + 1 + 0.5 * ties)
        omega = rank / (candidates + 1)
        logits.append(np.log(omega / (1 - omega)))
    result = np.asarray(logits)
    return {
        "pbo": float((result <= 0).mean()),
        "combinations": len(result),
        "partitions": n_partitions,
        "candidates": candidates,
        "criteria": int(values.shape[0]),
        "median_logit": float(np.median(result)),
    }


# ─── Deflated Sharpe Ratio ───────────────────────────────────────────────────


def effective_trials(returns: FloatArray) -> float:
    """N efectivo ≈ ρ̄ + (1 − ρ̄)·N según la correlación media entre estrategias."""
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("Se requiere una matriz T×N")
    n = matrix.shape[1]
    if n == 1:
        return 1.0
    std = matrix.std(axis=0)
    usable = matrix[:, std > 0]
    if usable.shape[1] < 2:
        return 1.0
    corr = np.corrcoef(usable, rowvar=False)
    mean_corr = float(np.clip(corr[~np.eye(len(corr), dtype=bool)].mean(), 0, 1))
    return float(mean_corr + (1 - mean_corr) * n)


def deflated_sharpe_ratio(
    returns: FloatArray, trial_sharpes: FloatArray, n_trials: float
) -> dict[str, Any]:
    """Probabilidad de que el Sharpe verdadero supere al máximo esperado bajo H0 (por periodo).

    Usa la asimetría y curtosis de la serie elegida; la varianza de Sharpe entre los ensayos
    define el umbral. No corrige la dependencia serial de la serie.
    """
    r = np.asarray(returns, dtype=float)
    trials = np.asarray(trial_sharpes, dtype=float)
    if len(r) < 3 or not np.isfinite(r).all() or not len(trials) or n_trials < 1:
        raise ValueError("Datos insuficientes para el Sharpe deflactado")
    std = r.std(ddof=1)
    sharpe = float(r.mean() / std) if std > 0 else 0.0
    gamma3 = float(skew(r))
    gamma4 = float(kurtosis(r, fisher=False))
    if n_trials > 1 and len(trials) > 1:
        sigma = float(trials.std(ddof=1))
        threshold = sigma * (
            (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        threshold = 0.0
    denominator = np.sqrt(max(1 - gamma3 * sharpe + (gamma4 - 1) / 4 * sharpe**2, 1e-12))
    z = (sharpe - threshold) * np.sqrt(len(r) - 1) / denominator
    return {
        "sharpe_per_period": sharpe,
        "expected_max_sharpe_null": float(threshold),
        "skewness": gamma3,
        "kurtosis": gamma4,
        "n_trials": float(n_trials),
        "periods": len(r),
        "deflated_sharpe_ratio": float(norm.cdf(z)),
    }
