"""PHASE-2 part B forecast uncertainty (pre-registered, RQ5).

Paired circular block bootstrap over the 35 rolling origins for RMSE
differences (OLS - naive, ridge - naive), block length 4 origins (the
12-month-ahead targets of adjacent origins overlap, so blocks are required to
carry the dependence), B = 2000, seed 42, percentile CIs 95%.

The existing Diebold-Mariano tests remain the PRIMARY inferential tool; this
bootstrap is a distributional cross-check. The registered decision rule: CIs
consistent with the DM conclusions are reported as agreement; any
inconsistency is reported prominently. Input is the per-origin SQUARED error
of each model (RMSE = sqrt(mean(se))), and every comparison resamples BOTH
models' errors with the SAME origin index stream (paired).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finscope.analytics.bootstrap import block_indices, percentile_ci
from finscope.forecast.inflation import diebold_mariano

SEED_BOOTSTRAP = 42
DEFAULT_BLOCK_LEN = 4  # registered: origins every 3 months; h = 12 overlap
BASELINE = "seasonal_naive"
COMPARISON_MODELS = ("ols", "ridge")


def squared_errors(predictions: pd.DataFrame, model: str) -> np.ndarray:
    """Per-origin squared error of one model against the realized target."""
    errors = predictions[model].to_numpy(dtype=float) - predictions["actual"].to_numpy(
        dtype=float
    )
    return np.square(errors)


def rmse_diff_cis(
    squared_errors_by_model: dict[str, pd.Series | np.ndarray],
    block_len: int = DEFAULT_BLOCK_LEN,
    n_boot: int = 2000,
    seed: int = SEED_BOOTSTRAP,
    confidence_level: float = 0.95,
    baseline: str = BASELINE,
    comparison_models: tuple[str, ...] = COMPARISON_MODELS,
) -> dict:
    """RQ5: paired block-bootstrap CIs for RMSE(model) - RMSE(baseline).

    ``squared_errors_by_model`` maps model name -> per-origin squared error.
    For each comparison model a FRESH ``default_rng(seed)`` stream is drawn
    and the SAME origin-index matrix is applied to both models' errors
    (paired resampling). Percentile CIs at ``confidence_level``.
    """
    missing = [m for m in (baseline, *comparison_models) if m not in squared_errors_by_model]
    if missing:
        raise KeyError(f"missing squared-error series for: {missing}")
    arrays = {
        m: np.asarray(squared_errors_by_model[m], dtype=float)
        for m in (baseline, *comparison_models)
    }
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"squared-error length mismatch across models: {lengths}")
    n = lengths.pop()
    if n < 2 * block_len:
        raise ValueError(
            f"n={n} origins is too short for block_len={block_len} block bootstrap"
        )

    pairs_out = {}
    for model in comparison_models:
        se_model = arrays[model]
        se_base = arrays[baseline]
        # same operations as the committed forecast stage's rmse():
        # sqrt(mean(square(errors))), so the point diff matches forecast.json
        point = float(np.sqrt(np.mean(se_model)) - np.sqrt(np.mean(se_base)))
        rng = np.random.default_rng(seed)
        idx = block_indices(n, block_len, n_boot, rng)
        rmse_model = np.sqrt(se_model[idx].mean(axis=1))
        rmse_base = np.sqrt(se_base[idx].mean(axis=1))
        diffs = rmse_model - rmse_base
        lo, hi = percentile_ci(diffs, confidence_level)
        pairs_out[model] = {
            "rmse": round(float(np.sqrt(np.mean(se_model))), 6),
            "rmse_baseline": round(float(np.sqrt(np.mean(se_base))), 6),
            "diff": round(point, 6),
            "ci_low": round(lo, 6),
            "ci_high": round(hi, 6),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "direction": "higher" if point > 0 else ("lower" if point < 0 else "zero"),
        }

    return {
        "B": int(n_boot),
        "block_len": int(block_len),
        "n_origins": int(n),
        "seed": int(seed),
        "baseline": baseline,
        "confidence_level": confidence_level,
        "statistic": (
            "RMSE(model) - RMSE(seasonal_naive) over the rolling origins; paired "
            "circular block bootstrap (same origin-index stream for both models), "
            "block length 4 origins, percentile intervals"
        ),
        "pairs": pairs_out,
    }


def dm_agreement(
    predictions: pd.DataFrame,
    bootstrap: dict,
    horizon: int = 12,
    alpha: float = 0.05,
) -> dict:
    """Registered RQ5 decision rule: bootstrap CIs vs the primary DM verdicts.

    For each comparison model the DM test is recomputed on the squared-error
    loss differential (identical arithmetic to the committed forecast stage)
    and the bootstrap interval is compared with the DM conclusion:
    agreement = the CI excludes zero on the SAME side as the DM verdict, or
    both are non-significant; anything else is a DISCREPANCY that must be
    reported prominently.
    """
    actual = predictions["actual"].to_numpy(dtype=float)
    naive_errors = predictions[BASELINE].to_numpy(dtype=float) - actual
    out = {}
    for model in bootstrap["pairs"]:
        errors = predictions[model].to_numpy(dtype=float) - actual
        loss_diff = np.square(errors) - np.square(naive_errors)
        dm = diebold_mariano(loss_diff, horizon=horizon)
        pair = bootstrap["pairs"][model]
        ci_side = None
        if pair["ci_low"] > 0:
            ci_side = "higher"
        elif pair["ci_high"] < 0:
            ci_side = "lower"
        dm_side = None
        if dm["p_value"] < alpha:
            dm_side = "higher" if dm["mean_loss_diff"] > 0 else "lower"
        if ci_side is None and dm_side is None:
            agreement = "agree (neither finds a significant difference)"
        elif ci_side == dm_side and ci_side is not None:
            agreement = f"agree (both flag the model as significantly {ci_side}-error)"
        else:
            agreement = (
                "DISCREPANCY (bootstrap CI and DM verdict disagree) — reported "
                "prominently per the registered decision rule"
            )
        out[model] = {
            "rmse": pair["rmse"],
            "rmse_baseline": pair["rmse_baseline"],
            "diff": pair["diff"],
            "ci_low": pair["ci_low"],
            "ci_high": pair["ci_high"],
            "ci_excludes_zero": pair["ci_excludes_zero"],
            "dm_stat": round(dm["dm_stat"], 6),
            "dm_p": round(dm["p_value"], 6),
            "dm_mean_loss_diff": round(dm["mean_loss_diff"], 6),
            "dm_side": dm_side,
            "ci_side": ci_side,
            "agreement": agreement,
        }
    return out
