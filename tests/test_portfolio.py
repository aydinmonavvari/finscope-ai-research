"""Offline tests for portfolio construction, caps, fallbacks and cost math."""

import numpy as np
import pandas as pd
import pytest

from finscope.portfolio.construct import (
    Scheme,
    build_schemes,
    equal_weights,
    inverse_vol_weights,
    min_variance_weights,
    performance_summary,
    rebalance_cost,
    tangency_weights,
    walk_forward_backtest,
    walk_forward_schedule,
)

COV2 = np.array([[0.04, 0.0], [0.0, 0.09]])
COV3 = np.array([[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.05]])


def test_equal_weights():
    assert np.allclose(equal_weights(4), [0.25, 0.25, 0.25, 0.25])


def test_inverse_vol_ordering_and_normalization():
    frame = pd.DataFrame(
        {"a": [0.01, -0.01] * 5, "b": [0.02, -0.02] * 5}  # vol 0.01  # vol 0.02
    )
    w = inverse_vol_weights(frame)
    assert np.allclose(w, [2 / 3, 1 / 3])
    assert w.sum() == pytest.approx(1.0)


def test_inverse_vol_degenerate_vol_falls_back_to_equal():
    frame = pd.DataFrame({"a": [0.01] * 10, "b": [0.02, -0.02] * 5})
    w = inverse_vol_weights(frame)
    assert np.allclose(w, [0.5, 0.5])


def test_min_variance_two_asset_hand_computed():
    # uncorrelated: w1 = sigma2^2 / (sigma1^2 + sigma2^2) = 0.09/0.13
    w = min_variance_weights(COV2, cap=1.0)
    assert np.allclose(w, [0.09 / 0.13, 0.04 / 0.13], atol=1e-5)
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_respects_cap():
    # uncapped: w ∝ 1/sigma^2 = [0.4455, 0.198, 0.3564] -> asset 1 binds at 0.40;
    # residual 0.6 split to minimize 0.09 w2^2 + 0.05 w3^2 -> [0.4, 0.214286, 0.385714]
    w = min_variance_weights(COV3, cap=0.40)
    assert np.allclose(w, [0.40, 0.214286, 0.385714], atol=1e-4)
    assert np.all(w <= 0.40 + 1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_infeasible_cap_raises():
    with pytest.raises(ValueError):
        min_variance_weights(COV2, cap=0.40, allow_fallback=True)


def test_min_variance_fallback_paths():
    bad_cov = np.array([[np.nan, 0.0], [0.0, 0.09]])
    assert np.allclose(min_variance_weights(bad_cov, cap=1.0, allow_fallback=True), [0.5, 0.5])
    with pytest.raises(ValueError):
        min_variance_weights(bad_cov, cap=1.0, allow_fallback=False)


def test_tangency_hand_computed_when_feasible():
    mu = np.array([0.10, 0.05])
    w, fell_back = tangency_weights(mu, COV2, cap=0.90, rf=0.0)
    assert not fell_back
    # z = [0.10/0.04, 0.05/0.09] -> normalized [0.818182, 0.181818]
    assert np.allclose(w, [0.8181818, 0.1818182], atol=1e-5)


def test_tangency_fallback_on_cap_violation():
    mu = np.array([0.10, 0.05, 0.07])
    w, fell_back = tangency_weights(mu, COV3, cap=0.40, rf=0.0)
    assert fell_back  # tangency puts 56% on asset 1 -> cap violated
    assert np.allclose(w, [0.40, 0.214286, 0.385714], atol=1e-4)  # capped min-variance


def test_tangency_fallback_on_nonpositive_excess():
    mu = np.array([0.05, 0.05])
    w, fell_back = tangency_weights(mu, COV2, cap=0.90, rf=0.10)
    assert fell_back  # excess return is negative -> tangency infeasible
    assert np.allclose(w, [0.09 / 0.13, 0.04 / 0.13], atol=1e-5)


def test_rebalance_cost_math():
    assert rebalance_cost(np.array([0.5, 0.5]), np.array([1.0, 0.0]), 10.0) == pytest.approx(0.001)
    assert rebalance_cost(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 10.0) == pytest.approx(0.002)


def test_walk_forward_schedule_no_overlap():
    folds = walk_forward_schedule(30, train_months=12, hold_months=3)
    assert len(folds) == 6
    for train_start, train_end, hold_start, hold_end in folds:
        assert hold_start == train_end  # train ends exactly when the hold begins
        assert train_end - train_start == 12
        assert hold_end - hold_start == 3
    for (_, _, _, hold_end), (_, _, next_hold_start, _) in zip(folds, folds[1:], strict=False):
        assert next_hold_start >= hold_end  # holding periods never overlap
    assert folds[-1][3] == 30


def test_backtest_cost_math_on_flat_returns():
    index = pd.date_range("2020-01-31", periods=15, freq="ME")
    frame = pd.DataFrame({"a": [0.0] * 15, "b": [0.0] * 15}, index=index)
    scheme = Scheme("fixed_a", lambda window: np.array([1.0, 0.0]))
    result = walk_forward_backtest(frame, scheme, train_months=12, hold_months=3, cost_bps=10.0)
    # one fold: initial turnover 1.0 -> single charge of 0.001; no other costs
    assert result["n_folds"] == 1
    assert result["total_cost_return"] == pytest.approx(0.001)
    assert result["net"].iloc[0] == pytest.approx(-0.001)
    assert result["net"].iloc[1:].abs().sum() == pytest.approx(0.0)
    assert result["gross"].abs().sum() == pytest.approx(0.0)


def test_performance_summary_hand_computed():
    returns = pd.Series([0.01, 0.03])
    summary = performance_summary(returns)
    # wealth 1.0403 over 2/12 years; mean 0.02; std(ddof=1) 0.0141421
    assert summary["sharpe"] == pytest.approx(0.02 / 0.0141421356 * np.sqrt(12), rel=1e-4)
    assert summary["cagr"] == pytest.approx(1.0403**6 - 1.0, rel=1e-6)
    assert summary["max_dd"] == pytest.approx(0.0)
    assert summary["n_months"] == 2


def test_build_schemes_fallback_counter():
    schemes = build_schemes(cap=0.40)
    assert set(schemes) == {"equal", "inv_vol", "min_var", "max_sharpe"}
    frame = pd.DataFrame(
        {
            "a": [0.01, -0.01] * 6,
            "b": [0.02, -0.02] * 6,
            "c": [0.005, 0.015, -0.005, 0.01] * 3,
        }
    )
    schemes["max_sharpe"].fn(frame)  # exercises the counting closure
    assert np.isclose(schemes["max_sharpe"].fn(frame).sum(), 1.0)
    assert np.isclose(schemes["min_var"].fn(frame).sum(), 1.0)
    assert np.isclose(schemes["inv_vol"].fn(frame).sum(), 1.0)


def test_max_sharpe_fallback_counted_on_scheme_field():
    """The Scheme field (what reports read) must count fallbacks — regression test.

    The high-mean asset dominates the tangency solution (weight ~72% > 40% cap),
    so the scheme MUST fall back; the dataclass field must reflect every call.
    """
    schemes = build_schemes(cap=0.40)
    frame = pd.DataFrame(
        {
            "a": [0.031, 0.019, 0.030, 0.020],  # mean 0.025, tiny variance
            "b": [0.010, 0.001, 0.009, 0.000],
            "c": [0.005, 0.015, -0.005, 0.005],
        }
    )
    _, fell_back = tangency_weights(
        frame.mean().to_numpy(), frame.cov().to_numpy(), cap=0.40, rf=0.0
    )
    assert fell_back  # the unconstrained tangency violates the cap
    schemes["max_sharpe"].fn(frame)
    assert schemes["max_sharpe"].fallbacks == 1
    schemes["max_sharpe"].fn(frame)
    assert schemes["max_sharpe"].fallbacks == 2
    # and the returned weights are the capped min-variance fallback
    assert np.allclose(
        schemes["max_sharpe"].fn(frame),
        min_variance_weights(frame.cov().to_numpy(), cap=0.40),
    )
