"""Metricas de avaliacao para contagem de objetos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def absolute_error(y_true, y_pred):
    """Erro absoluto por amostra."""
    return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))


def mean_absolute_error(y_true, y_pred):
    """Media dos erros absolutos."""
    errors = absolute_error(y_true, y_pred)
    return float(np.mean(errors)) if len(errors) else 0.0


def exact_match_accuracy(y_true, y_pred):
    """Proporcao de contagens exatamente corretas."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean(yt == yp)) if len(yt) else 0.0


def mean_bias_error(y_true, y_pred):
    """Erro medio assinado. Positivo indica supercontagem."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(yp - yt)) if len(yt) else 0.0


def count_overestimates(y_true, y_pred):
    """Quantidade de casos em que a predicao passou da contagem real."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return int(np.sum(yp > yt))


def count_underestimates(y_true, y_pred):
    """Quantidade de casos em que a predicao ficou abaixo da contagem real."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return int(np.sum(yp < yt))


def summary_metrics(results_df: pd.DataFrame) -> dict:
    """Resume um DataFrame com colunas y_true e y_pred."""
    if "y_true" not in results_df or results_df["y_true"].isna().all():
        return {}
    df = results_df.dropna(subset=["y_true", "y_pred"]).copy()
    if df.empty:
        return {}
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    return {
        "n": int(len(df)),
        "mae": mean_absolute_error(y_true, y_pred),
        "exact_match_accuracy": exact_match_accuracy(y_true, y_pred),
        "mean_bias_error": mean_bias_error(y_true, y_pred),
        "overestimates": count_overestimates(y_true, y_pred),
        "underestimates": count_underestimates(y_true, y_pred),
    }

