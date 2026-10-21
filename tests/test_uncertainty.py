"""Offline tests for the PHASE-2 part B uncertainty suite.

All synthetic, deterministic, offline. Registered pins: level-CI sanity on a
known-Sharpe DGP; the Holm known-example; the registered exact cost claims
(affine mean net return, bitwise net == gross at 0 bps, exact recomputation
identity vs the backtest); window-stability accounting; strict-JSON safety;
and the paired paired-bootstrap + structural-exclusion behavior.
"""

import json

import numpy as np
import pandas as pd
import pytest

from finscope.analytics.bootstrap import block_indices, monthly_sharpe
from finscope.forecast.uncertainty import (
    dm_agreement,
    rmse_diff_cis,
    squared_errors,
)
from finscope.portfolio.construct import performance_summary, walk_forward_backtest
from finscope.portfolio.uncertainty import (
    DISTINCT_SCHEMES,
    cost_grid,
    holm_adjust,
    level_cis,
    net_at_cost,
    pairwise_holm,
    structural_exclusion_record,
    window_stability,
)
from finscope.reporting.jsonio import strict_json_dumps, write_strict_json


def _synthetic_net(n_months: int = 93, seed: int = 7) -> dict[str, pd.Series]:
    """Three distinct synthetic OOS series + max_sharpe == min_var (structural)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-02-28", periods=n_months, freq="ME")
    equal = rng.normal(0.010, 0.045, n_months)
    inv_vol = rng.normal(0.009, 0.040, n_months)
    min_var = rng.normal(0.011, 0.038, n_months)
    return {
        "equal": pd.Series(equal, index=index),
        "inv_vol": pd.Series(inv_vol, index=index),
        "min_var": pd.Series(min_var, index=index),
        "max_sharpe": pd.Series(min_var.copy(), index=index),
    }


# ------------------------------------------------------------------- RQ1
def test_level_cis_sanity_on_known_sharpe_dgp():
    n = 93
    rng = np.random.default_rng(123)
    values = rng.normal(0.01, 0.04, n)  # DGP annualized Sharpe = 0.01/0.04*sqrt(12)
    truth = 0.01 / 0.04 * np.sqrt(12.0)
    net = {"equal": pd.Series(values, index=pd.date_range("2019-02-28", periods=n, freq="ME"))}
    out = level_cis(net, n_boot=2000, seed=42)
    point = out["point_estimates"]["equal"]
    summary = performance_summary(pd.Series(values))["sharpe"]
    assert monthly_sharpe(values) == summary  # bitwise (artifact stores 6dp)
    assert point == round(summary, 6)
    assert abs(point - truth) < 0.6  # one bootstrap SE at n=93 is ~0.5
    for block in ("3", "6", "12"):
        ci = out["levels"]["equal"][block]
        assert ci["ci_low"] < ci["ci_high"]
        # wide intervals are the EXPECTED outcome at n=93 (registered rule)
        assert ci["ci_high"] - ci["ci_low"] > 0.8
    ci6 = out["levels"]["equal"]["6"]
    assert ci6["ci_low"] <= point <= ci6["ci_high"]
    assert ci6["ci_low"] <= truth <= ci6["ci_high"]  # the DGP truth is covered
    assert out["B"] == 2000 and out["seed"] == 42


def test_level_cis_deterministic_and_multiblock():
    net = _synthetic_net()
    first = level_cis(net, n_boot=500, seed=42)
    second = level_cis(net, n_boot=500, seed=42)
    assert first == second
    # max_sharpe is structurally identical to min_var: identical CIs bitwise
    assert first["levels"]["max_sharpe"] == first["levels"]["min_var"]
    assert first["point_estimates"]["max_sharpe"] == first["point_estimates"]["min_var"]


# ------------------------------------------------------------------- RQ2
def test_holm_known_example():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04}, alpha=0.05)
    assert adjusted["a"] == {"p_raw": 0.01, "p_holm": 0.03, "reject_holm_05": True}
    assert adjusted["b"]["p_holm"] == 0.06 and not adjusted["b"]["reject_holm_05"]
    assert adjusted["c"]["p_holm"] == 0.06 and not adjusted["c"]["reject_holm_05"]
    # monotonicity of adjusted p-values in the raw ones
    raws = {"x": 0.20, "y": 0.04, "z": 0.30, "w": 0.004}
    out = holm_adjust(raws)
    values = [out[k]["p_holm"] for k in sorted(raws, key=raws.get)]
    assert values == sorted(values)


def test_structural_exclusion_verified_and_enforced():
    net = _synthetic_net()
    record = structural_exclusion_record(net)
    assert record["excluded_pair"] == ["min_var", "max_sharpe"]
    assert record["verified_bitwise_identical"] is True
    assert record["n_pairs_tested"] == 3
    # if the structural fact broke, the suite must fail loudly, not silently test
    broken = _synthetic_net()
    broken["max_sharpe"] = broken["max_sharpe"] + 1e-9
    with pytest.raises(AssertionError):
        structural_exclusion_record(broken)


def test_pairwise_holm_end_to_end_and_determinism():
    net = _synthetic_net()
    first = pairwise_holm(net, block_len=6, n_boot=800, seed=42)
    second = pairwise_holm(net, block_len=6, n_boot=800, seed=42)
    assert first == second  # fresh seed-42 stream per pair, deterministic
    assert first["family_size"] == 3
    pairs = first["pairs"]
    assert {p["a"] for p in pairs} | {p["b"] for p in pairs} == set(DISTINCT_SCHEMES)
    assert all("max_sharpe" not in p["pair"] for p in pairs)
    assert first["structural_exclusion"]["verified_bitwise_identical"] is True
    for pair in first["pairs"]:
        assert pair["ci_low"] < pair["ci_high"]
        assert 0.0 <= pair["p_raw"] <= 1.0
        assert pair["p_holm"] >= pair["p_raw"] - 1e-9


def test_paired_streams_bitwise_identity_in_pairwise():
    """The pairwise engine must use the SAME index matrix for both legs."""
    net = _synthetic_net()
    a = net["equal"].to_numpy(dtype=float)
    b = net["inv_vol"].to_numpy(dtype=float)
    idx = block_indices(93, 6, 300, np.random.default_rng(42))
    diff_direct = (
        np.sqrt(12.0) * (a[idx].mean(axis=1) / a[idx].std(axis=1, ddof=1))
        - np.sqrt(12.0) * (b[idx].mean(axis=1) / b[idx].std(axis=1, ddof=1))
    )
    # the engine's CI for this pair must equal the direct paired computation
    out = pairwise_holm(net, block_len=6, n_boot=300, seed=42)
    pair = next(p for p in out["pairs"] if p["pair"] == "equal__vs__inv_vol")
    lo = float(np.percentile(diff_direct, 2.5))
    hi = float(np.percentile(diff_direct, 97.5))
    assert pair["ci_low"] == round(lo, 6)
    assert pair["ci_high"] == round(hi, 6)


# ------------------------------------------------------------------- RQ3
def _backtest_fixture():
    rng = np.random.default_rng(19)
    index = pd.date_range("2020-01-31", periods=30, freq="ME")
    frame = pd.DataFrame(
        {"a": rng.normal(0.01, 0.03, 30), "b": rng.normal(0.008, 0.02, 30)},
        index=index,
    )
    run = walk_forward_backtest(
        frame, lambda window: np.array([0.5, 0.5]), train_months=12, hold_months=3,
        cost_bps=10.0,
    )
    return run


def test_exact_recomputation_identity_vs_backtest():
    """net(reference bps) recomputed from turnover must equal the backtest net."""
    run = _backtest_fixture()
    index = run["net"].index
    positions = np.array([index.get_loc(d) for d in run["charges"].index], dtype=int)
    recomputed = net_at_cost(
        run["gross"].to_numpy(), run["fold_turnover"].to_numpy(), positions, 10.0
    )
    assert np.array_equal(recomputed, run["net"].to_numpy())  # bitwise


def test_net_zero_bps_is_bitwise_gross():
    run = _backtest_fixture()
    index = run["net"].index
    positions = np.array([index.get_loc(d) for d in run["charges"].index], dtype=int)
    net0 = net_at_cost(
        run["gross"].to_numpy(), run["fold_turnover"].to_numpy(), positions, 0.0
    )
    assert np.array_equal(net0, run["gross"].to_numpy())  # bitwise, incl. -0.0


def test_mean_net_return_affine_in_bps():
    run = _backtest_fixture()
    index = run["net"].index
    positions = np.array([index.get_loc(d) for d in run["charges"].index], dtype=int)
    gross_mean = float(np.mean(run["gross"].to_numpy()))
    n = len(run["gross"])
    total_turnover = float(np.sum(run["fold_turnover"].to_numpy()))
    grid = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0)
    means = []
    for bps in grid:
        net = net_at_cost(
            run["gross"].to_numpy(), run["fold_turnover"].to_numpy(), positions, bps
        )
        means.append(float(np.mean(net)))
        expected = gross_mean - (bps / 1e4) * total_turnover / n
        assert abs(means[-1] - expected) <= 1e-12
    slope = -total_turnover / (1e4 * n)
    for i in range(len(grid) - 1):
        assert abs((means[i + 1] - means[i]) - slope * (grid[i + 1] - grid[i])) <= 1e-12


def test_cost_grid_reports_crossings_on_designed_data():
    """Scheme A leads gross but pays far more turnover -> ordering must cross."""
    rng = np.random.default_rng(23)
    n = 24
    index = pd.date_range("2021-01-31", periods=n, freq="ME")
    a_gross = pd.Series(rng.normal(0.010, 0.02, n), index=index)
    b_gross = pd.Series(rng.normal(0.006, 0.02, n), index=index)
    fold_dates = index[::3]
    gross = {"equal": a_gross, "inv_vol": b_gross}
    turnover = {
        "equal": pd.Series([2.0] * len(fold_dates), index=fold_dates),
        "inv_vol": pd.Series([0.05] * len(fold_dates), index=fold_dates),
    }
    out = cost_grid(gross, turnover, fold_dates)
    assert out["any_ordering_crossing"] is True
    assert len(out["ordering_crossings"]) >= 1
    # the expensive-but-stronger scheme leads at 0 bps and trails by 100 bps
    first_order = out["ordering_at_grid_points"]["0"].split(" > ")[0]
    last_order = out["ordering_at_grid_points"]["100"].split(" > ")[0]
    assert first_order == "equal" and last_order == "inv_vol"


def test_cost_grid_rows_are_12dp_and_match_direct_sharpe():
    run = _backtest_fixture()
    gross = {"equal": run["gross"]}
    turnover = {"equal": run["fold_turnover"]}
    out = cost_grid(gross, turnover, run["charges"].index)
    assert len(out["rows_12dp"]) == 6
    for row in out["rows_12dp"]:
        assert row["mean_net_return"] == round(row["mean_net_return"], 12)
        assert row["sharpe_net"] == round(row["sharpe_net"], 12)
    slopes = out["mean_net_return_slope_per_bps"]["equal"]
    total_turnover = float(np.sum(run["fold_turnover"].to_numpy()))
    assert slopes == round(-total_turnover / (1e4 * len(run["gross"])), 12)


# ------------------------------------------------------------------- RQ4
def test_window_stability_accounting():
    net = _synthetic_net()
    out = window_stability(net, n_windows=3)
    assert out["n_windows"] == 3 and out["window_months"] == 31
    assert len(out["window_sharpes"]) == 3
    assert len(out["adjacent_kendall_tau"]) == 2
    for tau in out["adjacent_kendall_tau"]:
        assert set(tau) == {"windows", "tau", "degenerate"}
        if tau["tau"] is None:
            assert tau["degenerate"] is True
        else:
            assert tau["degenerate"] is False
            assert -1.0 <= tau["tau"] <= 1.0
    assert sum(out["rank1_counts"].values()) == 3
    assert set(out["rank1_counts"]) == set(DISTINCT_SCHEMES)
    # structural identity shows up bitwise in the window sharpes too
    for row in out["window_sharpes"]:
        assert row["max_sharpe"] == row["min_var"]
    assert out["label"].startswith("EXPLORATORY")
    again = window_stability(net, n_windows=3)
    assert out == again
    with pytest.raises(ValueError):
        window_stability(net, n_windows=4)  # 93 does not divide into 4


# --------------------------------------------------------- forecast (RQ5)
def _predictions_fixture(n=35, seed=5):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2016-12-31", periods=n, freq="3ME")
    actual = rng.normal(2.5, 1.5, n)
    return pd.DataFrame(
        {
            "actual": actual,
            "seasonal_naive": actual + rng.normal(0.0, 0.5, n),
            "ols": actual + rng.normal(0.0, 1.2, n),
            "ridge": actual + rng.normal(0.0, 1.1, n),
        },
        index=index,
    )


def test_squared_errors_match_forecast_stage_arithmetic():
    from finscope.forecast.inflation import rmse

    predictions = _predictions_fixture()
    se = squared_errors(predictions, "ols")
    errors = predictions["ols"].to_numpy() - predictions["actual"].to_numpy()
    assert np.array_equal(se, np.square(errors))
    assert float(np.sqrt(np.mean(se))) == rmse(errors)  # bitwise


def test_rmse_diff_cis_deterministic_and_self_pair_zero():
    predictions = _predictions_fixture()
    se = {m: squared_errors(predictions, m) for m in ("seasonal_naive", "ols", "ridge")}
    first = rmse_diff_cis(se)
    second = rmse_diff_cis(se)
    assert first == second
    assert first["n_origins"] == 35 and first["block_len"] == 4 and first["B"] == 2000
    for model in ("ols", "ridge"):
        pair = first["pairs"][model]
        assert pair["ci_low"] < pair["ci_high"]
        expected_point = float(
            np.sqrt(np.mean(se[model])) - np.sqrt(np.mean(se["seasonal_naive"]))
        )
        assert pair["diff"] == round(expected_point, 6)
    # a model against ITSELF must give a degenerate zero interval
    se_self = {"seasonal_naive": se["seasonal_naive"], "ols": se["seasonal_naive"]}
    self_out = rmse_diff_cis(se_self, comparison_models=("ols",))
    pair = self_out["pairs"]["ols"]
    assert pair["diff"] == 0.0 and pair["ci_low"] == 0.0 and pair["ci_high"] == 0.0


def test_rmse_diff_cis_validates_block_len():
    se_short = {
        "seasonal_naive": pd.Series(np.arange(6.0)),
        "ols": pd.Series(np.arange(6.0)),
    }
    with pytest.raises(ValueError):
        rmse_diff_cis(se_short, block_len=4, comparison_models=("ols",))


def test_dm_agreement_structure():
    predictions = _predictions_fixture()
    se = {m: squared_errors(predictions, m) for m in ("seasonal_naive", "ols", "ridge")}
    boot = rmse_diff_cis(se)
    agreement = dm_agreement(predictions, boot)
    assert set(agreement) == {"ols", "ridge"}
    for model, entry in agreement.items():
        assert 0.0 <= entry["dm_p"] <= 1.0
        assert isinstance(entry["agreement"], str) and entry["agreement"]
        assert entry["ci_low"] == boot["pairs"][model]["ci_low"]
        assert entry["ci_high"] == boot["pairs"][model]["ci_high"]


# ------------------------------------------------------------- JSON (A8)
def test_strict_json_rejects_non_finite():
    for bad in (float("nan"), float("inf"), float("-inf"), np.nan, np.inf):
        with pytest.raises(ValueError):
            strict_json_dumps({"value": bad})
    with pytest.raises(ValueError):
        strict_json_dumps({"nested": {"deep": [1.0, float("nan")]}})


def test_strict_json_round_trip_and_numpy_types(tmp_path):
    def _reject(value):
        raise ValueError(f"non-finite constant in JSON text: {value}")

    payload = {
        "int": np.int64(3),
        "float": np.float64(0.25),
        "bool": np.bool_(True),
        "array": np.array([1.0, 2.0]),
        "timestamp": pd.Timestamp("2026-10-08"),
        "series": pd.Series([1.0, 2.0]),
    }
    path = tmp_path / "artifact.json"
    text = write_strict_json(payload, path)
    parsed = json.loads(text, parse_constant=_reject)  # strict parse: no NaN/Infinity
    assert parsed["int"] == 3 and parsed["float"] == 0.25 and parsed["bool"] is True
    assert parsed["array"] == [1.0, 2.0] and parsed["series"] == [1.0, 2.0]
    assert parsed["timestamp"] == "2026-10-08T00:00:00"
    assert "NaN" not in text and "Infinity" not in text


def test_stage_uncertainty_not_run_branch(tmp_path, monkeypatch):
    """Macro ingestion failure -> RQ5 recorded as machine-readable not_run.

    Registered deviation path (PHASE-2 plan section 10): when macro ingestion
    is impossible in the execution environment, the portfolio half (RQ1-RQ4)
    must still complete and forecast_uncertainty.json must carry an explicit
    not_run status with the exact reason — never a silent skip or a crash.
    """
    import importlib.util

    import finscope.config as config_mod
    import finscope.data.macro as macro_mod
    from finscope.config import PROJECT_ROOT

    spec = importlib.util.spec_from_file_location(
        "run_workbench_ph2b", PROJECT_ROOT / "scripts" / "run_workbench.py"
    )
    wb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wb)

    monkeypatch.setattr(config_mod, "REPORTS_DIR", tmp_path)

    def _blocked(refresh: bool = False):
        raise RuntimeError("FRED unreachable (synthetic blockage for test)")

    monkeypatch.setattr(macro_mod, "load_macro_monthly", _blocked)

    run = _backtest_fixture()
    # Distinct per-scheme series (the real pipeline has 4 different schemes;
    # an all-identical fixture would produce fully-tied window ranks and a
    # NaN Kendall tau, which strict JSON correctly rejects). min_var and
    # max_sharpe stay BITWISE identical: the registered structural exclusion
    # depends on it and is re-verified by the stage.
    net = run["net"]
    gross = run["gross"]
    nets = {
        "equal": net,
        "inv_vol": net * 0.9 + 0.0005,
        "min_var": net + 0.0002,
        "max_sharpe": net + 0.0002,
    }
    grosses = {
        "equal": gross,
        "inv_vol": gross * 0.9 + 0.0005,
        "min_var": gross + 0.0002,
        "max_sharpe": gross + 0.0002,
    }
    optimization = {
        "results": {
            name: {**run, "net": nets[name], "gross": grosses[name]}
            for name in nets
        },
        "net_series": nets,
    }
    result = wb.stage_uncertainty(optimization=optimization)

    fu = json.loads((tmp_path / "forecast_uncertainty.json").read_text())
    assert fu["status"] == "not_run"
    assert "FRED unreachable" in fu["reason"]
    assert "DM tests" in fu["primary_inferential_tool"]
    assert result["forecast_uncertainty"]["status"] == "not_run"

    unc = json.loads((tmp_path / "uncertainty.json").read_text())
    assert "level_cis" in unc and "pairwise_holm" in unc
    assert "cost_grid" in unc and "window_stability" in unc
    assert (tmp_path / "cost_sensitivity.csv").exists()


def test_stage_uncertainty_standalone_path(tmp_path, monkeypatch):
    """Round-trip through reports/oos_returns.json must reproduce the in-memory run.

    Regression guard for the standalone path: JSON serialisation loses the
    DatetimeIndex, and the fold-start dates must still align with the gross
    index after reload (PHASE-2 part B validation-report finding).
    """
    import importlib.util

    import finscope.config as config_mod
    import finscope.data.macro as macro_mod
    from finscope.config import PROJECT_ROOT
    from finscope.reporting.jsonio import write_strict_json

    spec = importlib.util.spec_from_file_location(
        "run_workbench_ph2b_sa", PROJECT_ROOT / "scripts" / "run_workbench.py"
    )
    wb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wb)

    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(config_mod, "REPORTS_DIR", reports)

    def _blocked(refresh: bool = False):
        raise RuntimeError("FRED unreachable (synthetic blockage for test)")

    monkeypatch.setattr(macro_mod, "load_macro_monthly", _blocked)

    run = _backtest_fixture()
    net, gross = run["net"], run["gross"]
    payload = {
        "metadata": {"as_of": "2026-10-08", "note": "test fixture"},
        "net": {
            name: {str(idx.date()): float(v) for idx, v in net.items()}
            for name in ("equal", "inv_vol", "min_var", "max_sharpe")
        },
        "gross": {
            name: {str(idx.date()): float(v) for idx, v in gross.items()}
            for name in ("equal", "inv_vol", "min_var", "max_sharpe")
        },
        "fold_turnover": {
            name: {
                str(idx.date()): float(v)
                for idx, v in run["fold_turnover"].items()
            }
            for name in ("equal", "inv_vol", "min_var", "max_sharpe")
        },
    }
    write_strict_json(payload, reports / "oos_returns.json")

    result = wb.stage_uncertainty()  # standalone: loads the artifact

    assert result["uncertainty"]["source"]["portfolio"].startswith(
        "reports/oos_returns.json"
    )
    assert (reports / "uncertainty.json").exists()
    assert (reports / "cost_sensitivity.csv").exists()

    # determinism: identical artifact on a second identical standalone run
    first = (reports / "uncertainty.json").read_text()
    wb.stage_uncertainty()
    assert (reports / "uncertainty.json").read_text() == first
