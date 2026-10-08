"""Offline tests for the forecasting study: MASE/DM math, models, study wiring."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

from finscope.forecast.inflation import (
    annual_inflation,
    build_verdict,
    diebold_mariano,
    mase,
    ols_fit_predict,
    ridge_fit_predict,
    rolling_origin_study,
    seasonal_naive_forecast,
)


def _monthly_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="ME")


def test_annual_inflation_hand_computed():
    cpi = pd.Series([100.0] * 13 + [110.0, 121.0], index=_monthly_index("2015-01-31", 15))
    infl = annual_inflation(cpi)
    assert len(infl) == 3
    # position 12 vs 0 -> 0%; position 13 vs 1 (100) -> 10%; position 14 vs 2 -> 21%
    assert np.allclose(infl.to_numpy(), [0.0, 10.0, 21.0])


def test_seasonal_naive_contract():
    infl = pd.Series([1.0, 2.0, 3.0], index=_monthly_index("2020-01-31", 3))
    origin = infl.index[1]
    assert seasonal_naive_forecast(infl, origin) == 2.0  # last observed value at origin


def test_mase_hand_computed_and_undefined_case():
    assert mase([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == pytest.approx(2.0)
    assert mase([0.5, 0.5], [1.0, 1.0]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mase([1.0, 2.0], [0.0, 0.0])


def test_dm_lag_zero_equals_hand_computed_t_stat():
    d = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    res = diebold_mariano(d, horizon=1)
    # HAC lag 0 without correction: se = sqrt(pop_var / n) = sqrt(0.96/5)
    expected_stat = 0.2 / np.sqrt(0.96 / 5)
    assert res["dm_stat"] == pytest.approx(expected_stat, rel=1e-8)
    assert res["p_value"] == pytest.approx(2 * (1 - sstats.norm.cdf(expected_stat)), rel=1e-6)


def test_dm_detects_improving_direction():
    d = np.array([2.0, 1.5, 2.2, 1.8, 2.1, 1.9, 2.0, 1.7])  # model loses (positive diff)
    res = diebold_mariano(d, horizon=1)
    assert res["dm_stat"] > 2.0
    assert res["p_value"] < 0.05
    assert res["mean_loss_diff"] > 0  # positive = first model has HIGHER loss


def test_ols_ridge_deterministic_and_ridge_shrinks():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.1, 3.9, 6.2, 7.8])
    p1 = ols_fit_predict(x, y, x)
    p2 = ols_fit_predict(x, y, x)
    assert np.array_equal(p1, p2)  # exact determinism
    r1 = ridge_fit_predict(x, y, x, alpha=10.0)
    r2 = ridge_fit_predict(x, y, x, alpha=10.0)
    assert np.array_equal(r1, r2)
    # slope implied by ridge predictions is flatter than the OLS slope (1.94)
    ols_slope = (p1[-1] - p1[0]) / 3.0
    ridge_slope = (r1[-1] - r1[0]) / 3.0
    assert abs(ridge_slope) < abs(ols_slope)


def _synthetic_macro(n_months: int = 120) -> pd.DataFrame:
    """CPI with a regime change at month 60 (1%/mo then 2%/mo) + mild features."""
    index = _monthly_index("2013-01-31", n_months)
    cpi = np.empty(n_months)
    level = 100.0
    for t in range(n_months):
        level *= 1.01 if t < 60 else 1.02
        cpi[t] = level
    t_grid = np.arange(n_months)
    return pd.DataFrame(
        {
            "cpi_index": cpi,
            "unemployment_rate": 5.0 + 0.01 * np.sin(t_grid / 6.0),
            "yield_10y": 2.0 + 0.001 * t_grid,
        },
        index=index,
    )


def test_rolling_origin_study_structure_and_no_lookahead():
    study = rolling_origin_study(_synthetic_macro(), horizon=12, step=3, min_train=24)
    preds = study["predictions"]
    # n=120 monthly rows -> inflation series has 108 rows (2014-01 .. 2022-12);
    # first origin at position 35, target known through position 95 -> 21 origins
    assert len(preds) == 21
    assert list(preds.columns) == [
        "actual",
        "seasonal_naive",
        "ols",
        "ridge",
        "n_train_pairs",
    ]
    assert preds.notna().all().all()
    assert (preds["n_train_pairs"] >= 24).all()
    macro = _synthetic_macro()
    infl = annual_inflation(macro["cpi_index"])
    # no look-ahead by construction: naive forecast at origin == inflation at origin
    assert np.allclose(preds["seasonal_naive"], infl.loc[preds.index].to_numpy())
    # and the actual equals inflation 12 months after each origin
    shifted = infl.shift(-12).loc[preds.index]
    assert np.allclose(preds["actual"], shifted.to_numpy())
    metrics = study["metrics"]
    assert set(metrics) == {"seasonal_naive", "ols", "ridge"}
    assert metrics["seasonal_naive"]["mase"] == 1.0
    # the regime change at month 60 makes the naive benchmark err (it cannot adapt)
    assert metrics["seasonal_naive"]["rmse"] > 0.0


def test_build_verdict_mentions_multiple_testing():
    metrics = {
        "seasonal_naive": {"rmse": 2.0, "mae": 1.5, "mase": 1.0},
        "ols": {"rmse": 1.0, "mae": 0.8, "mase": 0.53, "dm_stat": 2.5, "dm_p": 0.01,
                "beats_naive": True, "dm_significant": True},
        "ridge": {"rmse": 2.4, "mae": 2.0, "mase": 1.33, "dm_stat": -1.1, "dm_p": 0.27,
                  "beats_naive": False, "dm_significant": False},
    }
    verdict = build_verdict(metrics)
    assert "multiple-testing" in verdict
    assert "OLS" in verdict and "RIDGE" in verdict
