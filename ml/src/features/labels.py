"""Etiquetas objetivo. Anti-leakage: y[t] usa close[t+1] y close[t]; las features X[t] solo <= t."""
import pandas as pd


def make_target_label(close: pd.Series) -> pd.Series:
    """
    y[t] = 1 si close[t+1] > close[t], 0 en caso contrario.
    La última vela no tiene siguiente cierre: su etiqueta es NA (nunca se entrena con ella).
    """
    future = close.shift(-1)
    label = (future > close).astype("Int64")
    return label.mask(future.isna())
