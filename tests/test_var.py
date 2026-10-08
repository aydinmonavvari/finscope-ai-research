"""Offline tests for VaR/ES definitions and the drawdown table."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

from finscope.risk.var import (
    VAR_LIMITATIONS,
    drawdown_table,
    gaussian_var_es,
    historical_var_es,
    var_es_table,
)


def test_historical_var_es_hand_computed():
    returns = [-0.10, -0.03, -0.01, 0.00, 0.02]
    res = historical_var_es(returns, level=0.95)
    # 5% quantile (linear interpolation): -0.10 + 0.2 * 0.07 = -0.086
    assert res["var"] == pytest.approx(0.086)
    # tail = returns <= -0.086 -> only -0.10 -> ES = 0.10
    assert res["es"] == pytest.approx(0.10)
    assert res["n"] == 5


def test_gaussian_var_uses_z_value():
    returns = [0.01, -0.01]  # mean 0, std(ddof=1) = 0.01 * sqrt(2)
    sigma = 0.01 * np.sqrt(2)
    z95 = 1.6448536269514722
    res = gaussian_var_es(returns, level=0.95)
    assert res["var"] == pytest.approx(z95 * sigma, rel=1e-8)
    # ES for the Gaussian: sigma * phi(z) / (1 - alpha) - mu, mu = 0
    assert res["es"] == pytest.approx(sigma * sstats.norm.pdf(z95) / 0.05, rel=1e-8)


def test_var_es_monotonicity():
    rng = np.random.default_rng(3)
    heavy = rng.standard_t(df=3, size=400)
    hist95 = historical_var_es(heavy, 0.95)
    hist99 = historical_var_es(heavy, 0.99)
    gauss95 = gaussian_var_es(heavy, 0.95)
    gauss99 = gaussian_var_es(heavy, 0.99)
    for res in (hist95, hist99, gauss95, gauss99):
        assert res["es"] >= res["var"] - 1e-12  # ES >= VaR always
    assert hist99["var"] >= hist95["var"]
    assert hist99["es"] >= hist95["es"]
    assert gauss99["var"] >= gauss95["var"]
    assert gauss99["es"] >= gauss95["es"]


def test_var_es_table_levels():
    table = var_es_table([0.01, -0.02, 0.005, -0.005, 0.0])
    assert set(table) == {"95", "99"}
    assert table["99"]["hist_var"] >= table["95"]["hist_var"]


def test_drawdown_table_dates_and_recovery():
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    frame = pd.DataFrame(
        {
            "a": [0.10, -0.20, 0.00, 0.25],  # wealth 1.1, 0.88, 0.88, 1.10 -> recovers
            "b": [0.10, -0.50, 0.00, 0.00],  # wealth 1.1, 0.55 -> never recovers
        },
        index=index,
    )
    table = drawdown_table(frame)
    row_a = table.loc["a"]
    assert row_a["max_dd"] == pytest.approx(-0.20)
    assert row_a["peak_date"] == "2020-01-31"
    assert row_a["trough_date"] == "2020-02-29"
    assert row_a["recovery_date"] == "2020-04-30"
    row_b = table.loc["b"]
    assert row_b["max_dd"] == pytest.approx(-0.50)
    assert row_b["recovery_date"] == "not recovered"


def test_var_limitations_note_is_present():
    assert "estimates, not guarantees" in VAR_LIMITATIONS
    assert "estimation risk" in VAR_LIMITATIONS
