"""A small, honest forecasting study: 12-month CPI inflation.

Target: ``y_t = 100 * (CPI_t / CPI_{t-12} - 1)`` (annual CPI inflation, %).
Models: seasonal naive (the most recently observed annual inflation, which for
a 12-month-ahead annual target is exactly the seasonal-naive forecast),
OLS and ridge on three lags-known-at-origin features: current inflation,
unemployment and the 10-year yield.

Evaluation: rolling origins every 3 months (Tashman 2000). For an origin ``t``
the training pool contains only rows whose 12-month-ahead target was already
observed at ``t`` (rows dated ``t' <= t - 12``), so there is no look-ahead.
Model comparison uses the Diebold-Mariano (1995) test with a Newey-West HAC
variance at lag ``h - 1`` via statsmodels; with more than one comparison,
marginal p-values are read with an explicit multiple-testing caution
(Holm/Bonferroni in spirit), never as standalone discoveries.

This is a methods evaluation, not a market forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.regression.linear_model import OLS

from finscope.config import (
    FORECAST_HORIZON,
    FORECAST_MIN_TRAIN,
    FORECAST_STEP,
    RIDGE_ALPHA,
)

FEATURES = ["inflation", "unemployment_rate", "yield_10y"]


# --------------------------------------------------------------------- targets
def annual_inflation(cpi: pd.Series) -> pd.Series:
    """12-month CPI inflation in percent from the monthly CPI index."""
    return ((cpi / cpi.shift(12) - 1.0) * 100.0).dropna()


def feature_frame(infl: pd.Series, macro: pd.DataFrame) -> pd.DataFrame:
    """Origin-dated features (all known at the origin, no future information)."""
    frame = pd.DataFrame(
        {
            "inflation": infl,
            "unemployment_rate": macro["unemployment_rate"],
            "yield_10y": macro["yield_10y"],
        }
    ).dropna()
    return frame


# ---------------------------------------------------------------------- models
def _design(matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(matrix)), matrix])


def ols_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """OLS with intercept via least squares (deterministic closed form)."""
    beta, *_ = np.linalg.lstsq(_design(x_train), y_train, rcond=None)
    return _design(x_new) @ beta


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_new: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    """Ridge on standardized features (closed-form solver, deterministic)."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
    model.fit((x_train - mean) / std, y_train)
    return model.predict((x_new - mean) / std)


def seasonal_naive_forecast(infl: pd.Series, origin) -> float:
    """Seasonal-naive 12-month-ahead forecast: the inflation observed at origin.

    For an annual (12-month) target the seasonal-naive prediction is the value
    from 12 months before the target month, which equals the last observation
    at the forecast origin.
    """
    return float(infl.loc[origin])


# ------------------------------------------------------------------- metrics
def rmse(errors) -> float:
    values = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def mae(errors) -> float:
    return float(np.mean(np.abs(np.asarray(errors, dtype=float))))


def mase(errors, benchmark_errors) -> float:
    """Mean absolute error scaled by the benchmark's MAE on the SAME origins.

    MASE < 1 means the model beats the benchmark in absolute-error terms
    (Hyndman & Koehler 2006 introduced MASE; this variant scales by the
    benchmark's test MAE so the comparison is out-of-sample on both sides).
    """
    scale = mae(benchmark_errors)
    if scale == 0:
        raise ValueError("benchmark MAE is zero; MASE is undefined")
    return float(mae(errors) / scale)


def diebold_mariano(loss_diff, horizon: int = 1) -> dict:
    """Diebold-Mariano test on per-origin loss differentials.

    Uses statsmodels OLS with a Newey-West HAC covariance (Bartlett kernel,
    lag ``horizon - 1``, no small-sample correction) and normal p-values, the
    standard DM setup. Positive statistic = first model has HIGHER loss.
    """
    diff = np.asarray(loss_diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = diff.size
    if n < 3:
        raise ValueError("need at least 3 loss differentials for the DM test")
    lag = max(horizon - 1, 0)
    fit = OLS(diff, np.ones((n, 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lag, "use_correction": False},
        use_t=False,
    )
    return {
        "dm_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "mean_loss_diff": float(np.mean(diff)),
        "n": int(n),
        "hac_lag": int(lag),
    }


# ------------------------------------------------------------------- the study
def rolling_origin_study(
    macro: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
    step: int = FORECAST_STEP,
    min_train: int = FORECAST_MIN_TRAIN,
    ridge_alpha: float = RIDGE_ALPHA,
) -> dict:
    """Run the full rolling-origin evaluation and return predictions + metrics."""
    infl = annual_inflation(macro["cpi_index"])
    feats = feature_frame(infl, macro)
    target = infl.shift(-horizon)  # target[origin] = inflation `horizon` months later

    # candidate origin positions: enough training pairs and an observable target
    n = len(feats)
    first_pos = min_train + horizon - 1  # pool size at position i is i + 1 - horizon
    target_known = target.notna().to_numpy()
    positions = [i for i in range(first_pos, n) if target_known[i]][::step]

    rows = []
    for pos in positions:
        origin_date = feats.index[pos]
        cutoff = origin_date - pd.DateOffset(months=horizon)
        pool_mask = (feats.index <= cutoff) & target.notna()
        train = feats.loc[pool_mask]
        y_train = target.loc[train.index].to_numpy()
        x_train = train[FEATURES].to_numpy()
        x_new = feats[FEATURES].to_numpy()[pos : pos + 1]
        rows.append(
            {
                "origin": origin_date,
                "actual": float(target.iloc[pos]),
                "seasonal_naive": seasonal_naive_forecast(infl, origin_date),
                "ols": float(ols_fit_predict(x_train, y_train, x_new)[0]),
                "ridge": float(ridge_fit_predict(x_train, y_train, x_new, ridge_alpha)[0]),
                "n_train_pairs": int(len(train)),
            }
        )
    predictions = pd.DataFrame(rows).set_index("origin")
    metrics = _score_predictions(predictions, horizon)
    return {"predictions": predictions, "metrics": metrics}


def _score_predictions(predictions: pd.DataFrame, horizon: int) -> dict:
    """RMSE/MAE/MASE per model + direction-aware DM tests vs seasonal naive."""
    actual = predictions["actual"].to_numpy()
    model_metrics: dict[str, dict] = {}
    naive_errors = predictions["seasonal_naive"].to_numpy() - actual
    model_metrics["seasonal_naive"] = {
        "rmse": rmse(naive_errors),
        "mae": mae(naive_errors),
        "mase": 1.0,
    }
    for model in ("ols", "ridge"):
        errors = predictions[model].to_numpy() - actual
        loss_diff = np.square(errors) - np.square(naive_errors)
        dm = diebold_mariano(loss_diff, horizon=horizon)
        entry = {
            "rmse": rmse(errors),
            "mae": mae(errors),
            "mase": mase(errors, naive_errors),
            "dm_stat": dm["dm_stat"],
            "dm_p": dm["p_value"],
            "mean_loss_diff": dm["mean_loss_diff"],
            "n_origins": int(len(errors)),
        }
        # direction-aware significance: lower loss AND p < 0.05
        entry["beats_naive"] = bool(entry["rmse"] < model_metrics["seasonal_naive"]["rmse"])
        entry["dm_significant"] = bool(entry["dm_p"] < 0.05 and entry["beats_naive"])
        model_metrics[model] = entry
    return model_metrics


def build_verdict(metrics: dict, alpha: float = 0.05) -> str:
    """Honest, direction-aware verdict text assembled from actual numbers."""
    naive = metrics["seasonal_naive"]
    lines = [
        f"Seasonal-naive benchmark: RMSE {naive['rmse']:.2f} pp (MASE 1.00 by definition).",
    ]
    survivors = []
    for name in ("ols", "ridge"):
        entry = metrics[name]
        direction = "lower" if entry["beats_naive"] else "HIGHER"
        lines.append(
            f"{name.upper()}: RMSE {entry['rmse']:.2f} pp, MASE {entry['mase']:.2f} "
            f"({direction} error than naive), DM stat {entry['dm_stat']:.2f}, "
            f"p {entry['dm_p']:.3f}."
        )
        if entry["dm_p"] < alpha and entry["beats_naive"]:
            survivors.append(name.upper())
    if survivors:
        lines.append(
            "Only " + ", ".join(survivors) + f" beats the naive benchmark at p < {alpha} "
            "in the improving-error direction; with two comparisons this is read with an "
            "explicit multiple-testing caution and should not be treated as a discovery."
        )
    else:
        lines.append(
            "No model delivers a multiple-testing-robust improvement over the naive "
            "benchmark; the honest conclusion is that simple benchmarks remain hard "
            "to beat at this horizon with this feature set."
        )
    return " ".join(lines)
