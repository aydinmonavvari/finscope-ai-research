"""Offline tests for return statistics and diagnostics (hand-computed values)."""

import numpy as np
import pandas as pd
import pytest

from finscope.analytics import stats as st


def test_log_returns_hand_computed():
    prices = pd.Series([100.0, 110.0, 99.0])
    rets = st.log_returns(prices)
    assert np.allclose(rets.to_numpy(), [np.log(1.1), np.log(0.9)])


def test_annualized_vol_hand_computed():
    # 8 alternating +/-1% returns: std(ddof=1) = sqrt(8e-4/7) = 0.01069045
    r = [0.01, -0.01] * 4
    assert st.annualized_vol(r) == pytest.approx(0.01069045 * np.sqrt(252), rel=1e-4)


def test_annualized_sharpe_hand_computed():
    r = [0.01, 0.02, 0.03, 0.04]
    # mean 0.025, std(ddof=1) = sqrt(0.0005/3) = 0.012909944
    expected = 0.025 / 0.012909944 * np.sqrt(252)
    assert st.annualized_sharpe(r) == pytest.approx(expected, rel=1e-6)


def test_sharpe_zero_vol_guard():
    assert st.annualized_sharpe([0.01] * 10) == 0.0


def test_max_drawdown_hand_computed():
    r = pd.Series([0.2, -0.25, 0.1])
    # wealth 1.2 -> 0.9 -> 0.99; drawdown trough 0.9/1.2 - 1 = -0.25
    assert st.max_drawdown(r) == pytest.approx(-0.25)


def test_drawdown_series_shape_and_sign():
    r = pd.Series([0.1, -0.2, 0.3])
    dd = st.drawdown_series(r)
    assert (dd <= 1e-12).all()
    assert dd.iloc[0] == pytest.approx(0.0)


def test_jarque_bera_normal_vs_skewed():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=2000)
    skewed = rng.chisquare(df=3, size=2000)
    normal_res = st.jarque_bera_test(normal)
    skewed_res = st.jarque_bera_test(skewed)
    assert normal_res["p_value"] > 0.05
    assert skewed_res["p_value"] < 0.05
    assert skewed_res["excess_kurtosis"] > normal_res["excess_kurtosis"]


def test_adf_stationary_vs_random_walk():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=500)
    walk = np.cumsum(rng.normal(size=500))
    assert st.adf_test(noise)["p_value"] < 0.05
    assert st.adf_test(walk)["p_value"] > 0.05


def test_ljung_box_detects_autocorrelation():
    rng = np.random.default_rng(2)
    eps = rng.normal(size=500)
    ar1 = np.empty(500)
    ar1[0] = 0.0
    for t in range(1, 500):
        ar1[t] = 0.9 * ar1[t - 1] + eps[t]
    assert st.ljung_box_test(ar1)["p_value"] < 0.05


def test_rolling_vol_window_structure():
    series = pd.Series([0.01] * 63 + [-0.02] * 37)
    rolling = st.rolling_vol(series, window=63)
    assert rolling.iloc[:62].isna().all()
    assert rolling.iloc[62] == pytest.approx(0.0)  # constant first window
    assert rolling.iloc[63] > 0


def test_cagr_one_year_double():
    prices = pd.Series([100.0] * 252 + [200.0])
    assert st.cagr(prices) == pytest.approx(1.0)
