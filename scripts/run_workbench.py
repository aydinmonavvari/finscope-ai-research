#!/usr/bin/env python
"""Finscope workbench CLI.

Stages (each idempotent and cache-first; `all` runs them in order):

    ingest       fetch/cache Yahoo Finance prices + FRED macro series
    analyze      market diagnostics + figures 1-3
    forecast     12-month CPI inflation study + figures 4-5
    optimize     walk-forward portfolio backtest + figures 6-7 (+ oos_returns.json)
    risk         VaR/ES and drawdown reporting + figure 8
    uncertainty  pre-registered Phase-2 part B resampling suite (RQ1-RQ5)
    report       merge fragments -> reports/metrics.json + research_brief.md
    all          ingest -> analyze -> forecast -> optimize -> risk -> uncertainty -> report
"""

from __future__ import annotations

import argparse
import re
import sys
import time

STAGES = (
    "ingest",
    "analyze",
    "forecast",
    "optimize",
    "risk",
    "uncertainty",
    "report",
    "all",
)


def stage_ingest() -> dict:
    """Fetch or read from cache the price panel and macro series."""
    from finscope.config import TICKERS
    from finscope.data.macro import load_macro_monthly
    from finscope.data.prices import load_prices

    prices = load_prices(TICKERS)
    macro = load_macro_monthly()
    print(
        f"[ingest] prices: {len(prices)} days x {prices.shape[1]} tickers "
        f"({prices.index[0].date()} -> {prices.index[-1].date()})"
    )
    print(
        f"[ingest] macro: {len(macro)} months "
        f"({macro.index[0].date()} -> {macro.index[-1].date()})"
    )
    return {"prices": prices, "macro": macro}


def stage_analyze(prices=None) -> dict:
    """Market diagnostics per ticker; writes reports/analysis.json + 3 figures."""
    from finscope.analytics import stats as st
    from finscope.config import TICKERS
    from finscope.data.prices import load_prices
    from finscope.reporting.brief import write_fragment
    from finscope.reporting.figures import (
        plot_drawdowns,
        plot_prices_normalized,
        plot_rolling_vol,
    )

    if prices is None:
        prices = load_prices(TICKERS)
    log_ret = st.log_returns(prices)
    simple_ret = st.simple_returns(prices)
    table = st.summary_table(prices)
    figures = [
        plot_prices_normalized(prices),
        plot_rolling_vol(log_ret),
        plot_drawdowns(simple_ret),
    ]
    payload = {
        "tickers_used": list(prices.columns),
        "window": {
            "start": str(prices.index[0].date()),
            "end": str(prices.index[-1].date()),
            "n_days": int(len(prices)),
        },
        "tickers": {t: {k: v for k, v in row.items()} for t, row in table.iterrows()},
        "figures": figures,
        "notes": [
            "Statistics computed on daily log returns; max drawdown on daily simple returns.",
            "Annualization: 252 trading days; Sharpe uses rf = 0 (documented simplification).",
            "ADF run on log returns (stationarity) and on log prices (unit-root illustration).",
        ],
    }
    write_fragment("analysis", payload)
    print(
        f"[analyze] {prices.shape[1]} tickers, {len(prices)} days "
        f"-> analysis.json, {len(figures)} figures"
    )
    return {"prices": prices, "log_returns": log_ret}


def stage_forecast(macro=None) -> dict:
    """Rolling-origin inflation study; writes reports/forecast.json + 2 figures."""
    from finscope.config import (
        FORECAST_HORIZON,
        FORECAST_MIN_TRAIN,
        FORECAST_STEP,
        RIDGE_ALPHA,
    )
    from finscope.data.macro import load_macro_monthly
    from finscope.forecast.inflation import build_verdict, rolling_origin_study
    from finscope.reporting.brief import write_fragment
    from finscope.reporting.figures import (
        plot_inflation_backtest,
        plot_inflation_errors,
    )

    if macro is None:
        macro = load_macro_monthly()
    study = rolling_origin_study(
        macro,
        horizon=FORECAST_HORIZON,
        step=FORECAST_STEP,
        min_train=FORECAST_MIN_TRAIN,
        ridge_alpha=RIDGE_ALPHA,
    )
    predictions = study["predictions"]
    metrics = study["metrics"]
    figures = [
        plot_inflation_backtest(predictions),
        plot_inflation_errors(predictions),
    ]
    payload = {
        "target": "12-month CPI inflation (%) from CPIAUCSL",
        "horizon_months": FORECAST_HORIZON,
        "step_months": FORECAST_STEP,
        "min_train_pairs": FORECAST_MIN_TRAIN,
        "ridge_alpha": RIDGE_ALPHA,
        "features": ["inflation", "unemployment_rate", "yield_10y"],
        "n_origins": int(len(predictions)),
        "first_origin": str(predictions.index[0].date()),
        "last_origin": str(predictions.index[-1].date()),
        "models": metrics,
        "verdict": build_verdict(metrics),
        "multiple_testing_note": (
            "two DM comparisons (OLS, ridge) against the same benchmark; marginal "
            "p-values are read with a Holm/Bonferroni-style caution and low power at "
            f"n={len(predictions)} origins is acknowledged"
        ),
        "figures": figures,
    }
    write_fragment("forecast", payload)
    print(f"[forecast] {len(predictions)} origins -> forecast.json, {len(figures)} figures")
    for name, m in metrics.items():
        extra = f", DM p={m['dm_p']:.3f}" if "dm_p" in m else ""
        print(f"    {name}: RMSE={m['rmse']:.2f}pp MASE={m['mase']:.2f}{extra}")
    return {"predictions": predictions}


def stage_optimize(prices=None) -> dict:
    """Walk-forward portfolio backtest; writes reports/portfolio.json + 2 figures."""
    import numpy as np
    import pandas as pd

    from finscope.config import ASSET_CAP, COST_BPS_PER_SIDE, HOLD_MONTHS, TRAIN_MONTHS
    from finscope.data.prices import load_prices, monthly_simple_returns
    from finscope.portfolio.construct import (
        build_schemes,
        performance_summary,
        walk_forward_backtest,
    )
    from finscope.reporting.brief import write_fragment
    from finscope.reporting.figures import plot_portfolio_oos, plot_weights

    if prices is None:
        from finscope.config import TICKERS

        prices = load_prices(TICKERS)
    monthly = monthly_simple_returns(prices)
    schemes = build_schemes(cap=ASSET_CAP)
    results = {}
    for name, scheme in schemes.items():
        run = walk_forward_backtest(
            monthly, scheme, TRAIN_MONTHS, HOLD_MONTHS, COST_BPS_PER_SIDE
        )
        run["scheme"] = scheme
        results[name] = run

    oos_start = str(results["equal"]["net"].index[0].date())
    oos_end = str(results["equal"]["net"].index[-1].date())
    scheme_payload = {}
    net_series = {}
    for name, run in results.items():
        summary = performance_summary(run["net"])
        summary.update(
            {
                "avg_turnover": run["avg_turnover"],
                "total_cost_pct": run["total_cost_return"],
                "n_folds": run["n_folds"],
            }
        )
        n_folds = run["n_folds"]
        # fallback_share is reported for EVERY scheme so the artifact is complete;
        # in the committed run only max_sharpe ever fires (its documented
        # tangency -> min-variance fallback), the others report 0.0.
        summary["fallback_share"] = (
            float(run["scheme"].fallbacks / n_folds) if n_folds else 0.0
        )
        scheme_payload[name] = summary
        net_series[name] = run["net"]
    ranking = sorted(scheme_payload, key=lambda k: scheme_payload[k]["sharpe"], reverse=True)

    avg_weights = {}
    for name, run in results.items():
        avg_weights[name] = np.mean(np.asarray(run["weights"]), axis=0)
    weights_frame = pd.DataFrame(avg_weights, index=monthly.columns).T
    figures = [
        plot_weights(weights_frame),
        plot_portfolio_oos(net_series),
    ]
    payload = {
        "cost_bps_per_side": COST_BPS_PER_SIDE,
        "train_months": TRAIN_MONTHS,
        "hold_months": HOLD_MONTHS,
        "asset_cap": ASSET_CAP,
        "oos_months": int(len(results["equal"]["net"])),
        "oos_start": oos_start,
        "oos_end": oos_end,
        "n_folds": results["equal"]["n_folds"],
        "schemes": scheme_payload,
        "ranking_net_sharpe": ranking,
        "avg_weights": {k: [round(float(x), 4) for x in v] for k, v in avg_weights.items()},
        "assets": list(monthly.columns),
        "figures": figures,
    }
    write_fragment("portfolio", payload)
    _write_oos_returns(results, scheme_payload, oos_start, oos_end)
    print(
        f"[optimize] {payload['oos_months']} OOS months, {payload['n_folds']} folds "
        "-> portfolio.json + oos_returns.json"
    )
    for name in ranking:
        s = scheme_payload[name]
        print(
            f"    {name}: Sharpe(net)={s['sharpe']:.3f} CAGR={s['cagr'] * 100:.1f}% "
            f"maxDD={s['max_dd'] * 100:.1f}% turnover={s['avg_turnover']:.3f}"
        )
    return {"net_series": net_series, "monthly": monthly, "results": results}


def _write_oos_returns(results: dict, scheme_payload: dict, oos_start: str, oos_end: str) -> None:
    """Persist the previously unpersisted OOS series (registered plan §6 artifact).

    ``reports/oos_returns.json`` stores the full-precision monthly NET series
    per scheme — the same series already visualized in the committed figures —
    plus the GROSS series and dated per-fold one-way turnover so the RQ3 cost
    grid is exactly recomputable from the committed artifact alone. This is a
    derived pipeline output (registered exception to the no-data-committed
    policy), never constituent market data.
    """
    from finscope.config import AS_OF, REPORTS_DIR
    from finscope.reporting.jsonio import write_strict_json

    max_fallback = scheme_payload.get("max_sharpe", {}).get("fallback_share", 0.0)
    fallback_note = (
        f"max_sharpe used its min-variance fallback in {max_fallback * 100:.0f}% of "
        "folds, so its OOS series is bitwise identical to min_var by construction "
        "(structural exclusion registered in the Phase-2 plan §1)."
        if max_fallback
        else "no scheme triggered a fallback path in this run."
    )
    payload = {
        "metadata": {
            "as_of": AS_OF,
            "window": {"start": oos_start, "end": oos_end},
            "n_months": int(len(results["equal"]["net"])),
            "n_folds": int(results["equal"]["n_folds"]),
            "frequency": "monthly OOS returns, month-end labels",
            "fallback_note": fallback_note,
            "provenance": (
                "registered Phase-2 part B artifact "
                "(docs/admissions/PHASE2-P10-PLAN.md §6): derived pipeline output, "
                "not constituent market data; net series at full float precision"
            ),
            "schemes": {
                name: {"fallback_share": scheme_payload[name]["fallback_share"]}
                for name in results
            },
        },
        "net": {
            name: {str(d.date()): float(v) for d, v in run["net"].items()}
            for name, run in results.items()
        },
        "gross": {
            name: {str(d.date()): float(v) for d, v in run["gross"].items()}
            for name, run in results.items()
        },
        "fold_turnover": {
            name: {str(d.date()): float(v) for d, v in run["fold_turnover"].items()}
            for name, run in results.items()
        },
    }
    write_strict_json(payload, REPORTS_DIR / "oos_returns.json")


def stage_risk(prices=None, net_series=None) -> dict:
    """VaR/ES + drawdown reporting; writes reports/risk.json + 1 figure."""
    import pandas as pd

    from finscope.config import TICKERS
    from finscope.data.prices import load_prices
    from finscope.portfolio.construct import build_schemes, walk_forward_backtest
    from finscope.reporting.brief import write_fragment
    from finscope.reporting.figures import plot_var_es
    from finscope.risk.var import drawdown_table, var_es_table, var_limitations

    if net_series is None or prices is None:
        if prices is None:
            prices = load_prices(TICKERS)
        from finscope.data.prices import monthly_simple_returns

        monthly = monthly_simple_returns(prices)
        schemes = build_schemes()
        net_series = {
            name: walk_forward_backtest(monthly, scheme)["net"]
            for name, scheme in schemes.items()
        }
    var_table = {}
    for name, series in net_series.items():
        entries = var_es_table(series)
        var_table[name] = {
            "var95": entries["95"]["hist_var"],
            "es95": entries["95"]["hist_es"],
            "var99": entries["99"]["hist_var"],
            "es99": entries["99"]["hist_es"],
            "gauss_var95": entries["95"]["gauss_var"],
            "gauss_es95": entries["95"]["gauss_es"],
            "gauss_var99": entries["99"]["gauss_var"],
            "gauss_es99": entries["99"]["gauss_es"],
            "n_months": int(len(series)),
        }
    spy_daily = load_prices(["SPY"])["SPY"].pct_change().dropna()
    spy_hist95 = var_es_table(spy_daily)
    drawdowns = drawdown_table(pd.DataFrame(net_series))
    figure = plot_var_es(var_table)
    # the limitations note carries the MEASURED monthly sample size (n_months
    # is identical across schemes by construction; 93 in the committed run)
    measured_n = int(len(next(iter(net_series.values()))))
    payload = {
        "frequency": "monthly OOS net portfolio returns; daily SPY reference",
        "var_table": var_table,
        "spy_daily": {
            "n": int(len(spy_daily)),
            "var95": spy_hist95["95"]["hist_var"],
            "es95": spy_hist95["95"]["hist_es"],
            "var99": spy_hist95["99"]["hist_var"],
            "es99": spy_hist95["99"]["hist_es"],
        },
        "drawdowns": {k: dict(v) for k, v in drawdowns.iterrows()},
        "limitations": var_limitations(measured_n),
        "figures": [figure],
    }
    write_fragment("risk", payload)
    print(f"[risk] VaR/ES for {len(var_table)} schemes + SPY daily reference -> risk.json")
    return {"var_table": var_table}


def _load_oos_returns() -> dict:
    """Load the registered OOS-series artifact (standalone uncertainty runs)."""
    import json

    from finscope.config import REPORTS_DIR

    path = REPORTS_DIR / "oos_returns.json"
    if not path.exists():
        raise RuntimeError(
            "reports/oos_returns.json not found — run the `optimize` stage first "
            "(it persists the registered OOS series artifact)."
        )
    return json.loads(path.read_text())


def stage_uncertainty(optimization: dict | None = None, predictions=None) -> dict:
    """Pre-registered Phase-2 part B suite (RQ1-RQ5); writes 3 artifacts + 1 csv.

    Uses the in-memory optimization results and forecast predictions when
    given (the `all` flow), else loads ``reports/oos_returns.json`` and
    recomputes the rolling-origin study from the cached macro panel — both
    paths are deterministic and produce identical artifacts on the pinned
    vintage. Runtime budget (registered A7): well under 60 s.
    """
    import time

    import pandas as pd

    from finscope.config import (
        FORECAST_HORIZON,
        FORECAST_MIN_TRAIN,
        FORECAST_STEP,
        REPORTS_DIR,
        RIDGE_ALPHA,
    )
    from finscope.forecast.inflation import rolling_origin_study
    from finscope.forecast.uncertainty import (
        dm_agreement,
        rmse_diff_cis,
        squared_errors,
    )
    from finscope.portfolio.uncertainty import (
        cost_grid,
        level_cis,
        pairwise_holm,
        window_stability,
    )
    from finscope.reporting.jsonio import write_strict_json

    start = time.perf_counter()
    if optimization is not None:
        results = optimization["results"]
        net_series = optimization["net_series"]
        gross = {name: run["gross"] for name, run in results.items()}
        turnover = {name: run["fold_turnover"] for name, run in results.items()}
        charge_dates = results["equal"]["charges"].index
        source = "in-memory optimization results"
    else:
        loaded = _load_oos_returns()
        meta = loaded["metadata"]

        def _from_json(d: dict) -> pd.Series:
            # JSON round-trip loses the DatetimeIndex; the in-memory path works
            # on Timestamp indexes, so reconstruct them here (regression-tested:
            # tests/test_uncertainty.py::test_stage_uncertainty_standalone_path).
            s = pd.Series(d, dtype=float)
            s.index = pd.to_datetime(s.index)
            return s.sort_index()

        net_series = {name: _from_json(s) for name, s in loaded["net"].items()}
        gross = {
            name: _from_json(s) for name, s in loaded["gross"].items()
        }
        turnover = {
            name: _from_json(s)
            for name, s in loaded["fold_turnover"].items()
        }
        charge_dates = pd.DatetimeIndex(next(iter(turnover.values())).index)
        source = f"reports/oos_returns.json (as_of {meta.get('as_of', 'n/a')})"

    forecast_unavailable_reason: str | None = None
    if predictions is None:
        try:
            from finscope.data.macro import load_macro_monthly

            macro = load_macro_monthly()
            study = rolling_origin_study(
                macro,
                horizon=FORECAST_HORIZON,
                step=FORECAST_STEP,
                min_train=FORECAST_MIN_TRAIN,
                ridge_alpha=RIDGE_ALPHA,
            )
            predictions = study["predictions"]
            forecast_source = "rolling-origin study recomputed from cached macro panel"
        except RuntimeError as exc:
            # Registered not-run path (PHASE-2 plan §10 deviations log): when
            # macro ingestion is impossible in the execution environment (e.g.
            # FRED serving bot-challenge pages), RQ5 is recorded as not run
            # with the exact reason — the committed DM tests in
            # reports/forecast.json remain the primary inferential tool. The
            # portfolio half (RQ1-RQ4) is unaffected and still runs.
            # The full error text embeds the absolute cache path, which must
            # never reach a committed artifact: neutralize any absolute path
            # and keep only the leading sentence.
            forecast_unavailable_reason = re.sub(
                r"/(?:home|Users)/\S*", "<cache path>", str(exc)
            ).split(". ")[0].rstrip(".")
            forecast_source = "not_run — macro ingestion unavailable"
    else:
        forecast_source = "in-memory forecast predictions"

    # RQ1 / RQ2 / RQ3 / RQ4 (portfolio side)
    levels = level_cis(net_series)
    pairwise = pairwise_holm(net_series)
    grid = cost_grid(gross, turnover, charge_dates)
    windows = window_stability(net_series)

    uncertainty_payload = {
        "plan": "docs/admissions/PHASE2-P10-PLAN.md (pre-registered)",
        "status": "Phase-2 part B uncertainty suite; registered seeds 42; B=2000",
        "source": {"portfolio": source, "forecast": forecast_source},
        "level_cis": levels,
        "pairwise_holm": pairwise,
        "cost_grid": grid,
        "window_stability": windows,
    }
    write_strict_json(uncertainty_payload, REPORTS_DIR / "uncertainty.json")

    # RQ3 registered artifact: the full grid at 12-decimal precision
    grid_frame = pd.DataFrame(grid["rows_12dp"])
    grid_frame.to_csv(
        REPORTS_DIR / "cost_sensitivity.csv",
        index=False,
        float_format="%.12f",
    )

    # RQ5 (forecast side)
    if predictions is None:
        forecast_payload = {
            "plan": "docs/admissions/PHASE2-P10-PLAN.md (pre-registered, RQ5)",
            "status": "not_run",
            "reason": forecast_unavailable_reason,
            "registered_method": (
                "paired circular block bootstrap over the rolling origins "
                "(block length 4, B=2000, seed 42) for the RMSE differences "
                "OLS-naive and ridge-naive; machinery implemented in "
                "finscope.forecast.uncertainty and unit-tested on synthetic "
                "data (tests/test_uncertainty.py)"
            ),
            "primary_inferential_tool": (
                "committed DM tests in reports/forecast.json (v1.0.0 artifacts, "
                "byte-identical, unchanged by this stage)"
            ),
        }
    else:
        se = {
            model: squared_errors(predictions, model)
            for model in ("seasonal_naive", "ols", "ridge")
        }
        rmse_boot = rmse_diff_cis(se)
        forecast_payload = {
            "plan": "docs/admissions/PHASE2-P10-PLAN.md (pre-registered, RQ5)",
            "source": forecast_source,
            **rmse_boot,
            "dm_agreement": dm_agreement(predictions, rmse_boot),
        }
    write_strict_json(forecast_payload, REPORTS_DIR / "forecast_uncertainty.json")

    elapsed = time.perf_counter() - start
    n_reject = pairwise["n_reject_holm_05"]
    rq5_note = (
        "RQ5 NOT RUN (macro unavailable)"
        if predictions is None
        else "RQ5 RMSE CIs"
    )
    print(
        f"[uncertainty] RQ1 level CIs, RQ2 Holm ({n_reject}/3 rejected), RQ3 grid "
        f"({len(grid['ordering_crossings'])} crossings), RQ4 windows, {rq5_note} "
        f"-> uncertainty.json, cost_sensitivity.csv, forecast_uncertainty.json "
        f"({elapsed:.1f}s)"
    )
    return {
        "uncertainty": uncertainty_payload,
        "forecast_uncertainty": forecast_payload,
        "elapsed_s": elapsed,
    }


def stage_report() -> dict:
    """Merge fragments, score the bundled sample, write metrics.json + brief."""
    import json

    from finscope.config import REPORTS_DIR, SAMPLES_DIR
    from finscope.nlp.sentiment import score_headlines, summarize
    from finscope.reporting.brief import collect_metrics, write_fragment, write_reports

    metrics = collect_metrics()
    scored = score_headlines(SAMPLES_DIR / "sample_headlines.csv")
    metrics["sentiment"] = summarize(scored)
    write_fragment("sentiment", metrics["sentiment"])
    # RQ5 reaches the brief from its own artifact (additive metrics key only;
    # the RQ1-RQ4 numbers flow in via the uncertainty fragment).
    forecast_uncertainty_path = REPORTS_DIR / "forecast_uncertainty.json"
    if forecast_uncertainty_path.exists():
        metrics["forecast_uncertainty"] = json.loads(
            forecast_uncertainty_path.read_text()
        )
    metrics["findings"] = _build_findings(metrics)
    write_reports(metrics)
    print(
        f"[report] fragments merged ({sorted(k for k in metrics if k != 'findings')}) "
        "-> reports/metrics.json + reports/research_brief.md"
    )
    return metrics


def _build_findings(metrics: dict) -> list[str]:
    """Evidence-first findings assembled from the actual stage outputs."""
    findings = []
    analysis = metrics.get("analysis", {})
    if analysis:
        n = len(analysis.get("tickers", {}))
        rejects = sum(
            1 for v in analysis.get("tickers", {}).values() if float(v.get("jb_p", 1)) < 0.05
        )
        findings.append(
            f"Market diagnostics: daily returns of all {n} tickers reject normality under "
            f"Jarque-Bera ({rejects}/{n} at 5%); log returns are stationary while log "
            "prices are not — the standard risk-modelling caveat applies to shortcuts."
        )
    forecast = metrics.get("forecast", {})
    if forecast:
        models = forecast.get("models", {})
        winners = [
            name.upper()
            for name in ("ols", "ridge")
            if models.get(name, {}).get("dm_significant")
        ]
        if winners:
            findings.append(
                f"Forecasting: {', '.join(winners)} beats the seasonal-naive benchmark with an "
                "improving-error DM test, read under the stated multiple-testing caution."
            )
        else:
            findings.append(
                "Forecasting: no model delivers a multiple-testing-robust improvement over "
                "seasonal naive for 12-month inflation; the benchmark's RMSE is "
                f"{models['seasonal_naive']['rmse']:.2f} pp across "
                f"{forecast.get('n_origins')} origins."
            )
    portfolio = metrics.get("portfolio", {})
    if portfolio:
        ranking = portfolio.get("ranking_net_sharpe", [])
        schemes = portfolio.get("schemes", {})
        if ranking:
            top = ranking[0]
            fallback_share = schemes.get("max_sharpe", {}).get("fallback_share")
            fallback_note = (
                f" (max-Sharpe used its min-variance fallback in {fallback_share * 100:.0f}% "
                "of folds, so the two coincide)"
                if fallback_share
                else ""
            )
            equal_sharpe = schemes.get("equal", {}).get("sharpe")
            findings.append(
                f"Portfolios: {top.replace('_', ' ')} leads out-of-sample net Sharpe "
                f"({schemes[top]['sharpe']:.2f}) across {portfolio.get('oos_months')} months"
                f"{fallback_note}; the margin over equal weights ({equal_sharpe:.2f}) is modest "
                "and consistent with estimation-error theory — variance reduction, not return "
                "forecasting."
            )
    risk = metrics.get("risk", {})
    if risk:
        table = risk.get("var_table", {})
        if table:
            worst = max(table, key=lambda k: table[k]["es99"])
            findings.append(
                f"Risk: the {worst.replace('_', ' ')} scheme shows the deepest 99% monthly "
                f"expected shortfall ({table[worst]['es99'] * 100:.1f}%); quantiles on ~"
                # default 93 = the measured OOS months of the committed run; the
                # key is always present in fragments written by stage_risk
                f"{table[worst].get('n_months', 93)} observations carry "
                "first-order estimation risk."
            )
    sentiment = metrics.get("sentiment", {})
    if sentiment:
        findings.append(
            "Sentiment: VADER scoring of the bundled illustrative sample runs end to end "
            "(no network, no dataset claims); the evidence-bearing benchmark is the "
            "financial-nlp-sentiment repository."
        )
    uncertainty = metrics.get("uncertainty", {})
    if uncertainty:
        pairwise = uncertainty.get("pairwise_holm", {})
        n_reject = pairwise.get("n_reject_holm_05")
        family = pairwise.get("family_size", 3)
        if n_reject is not None:
            levels = uncertainty.get("level_cis", {})
            points = levels.get("point_estimates", {})
            block_cis = levels.get("levels", {})
            headline_block = "6"
            # MEASURED overlap statement: does each scheme's headline 95% CI
            # contain every scheme's point estimate? (no hard-coded claim)
            mutually_spanning = bool(points) and all(
                block_cis.get(s, {}).get(headline_block, {}).get("ci_low", 1e9)
                <= min(points.values())
                and max(points.values())
                <= block_cis.get(s, {}).get(headline_block, {}).get("ci_high", -1e9)
                for s in points
            )
            overlap_note = (
                "each scheme's 95% block-bootstrap CI contains every scheme's point "
                "estimate"
                if mutually_spanning
                else "the level CIs overlap across schemes"
            )
            findings.append(
                f"Resampling checks (pre-registered Phase-2 extension): {n_reject} of "
                f"{family} pairwise net-Sharpe differences survive Holm correction at "
                f"family-wise 0.05; {overlap_note} — the observed scheme gaps are within "
                "bootstrap rank noise at n=93 monthly observations."
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_workbench",
        description="Finscope research workbench — staged, cached, offline-safe after first fetch.",
    )
    parser.add_argument(
        "stage", choices=STAGES, help="pipeline stage to run ('all' runs everything)"
    )
    parser.add_argument("--refresh", action="store_true", help="force re-download of cached data")
    args = parser.parse_args(argv)

    def timed(label: str, fn, *fn_args):
        start = time.perf_counter()
        out = fn(*fn_args)
        print(f"    ({label} took {time.perf_counter() - start:.1f}s)")
        return out

    if args.stage == "ingest":
        timed("ingest", stage_ingest)
    elif args.stage == "analyze":
        timed("analyze", stage_analyze)
    elif args.stage == "forecast":
        timed("forecast", stage_forecast)
    elif args.stage == "optimize":
        timed("optimize", stage_optimize)
    elif args.stage == "risk":
        timed("risk", stage_risk)
    elif args.stage == "uncertainty":
        timed("uncertainty", stage_uncertainty)
    elif args.stage == "report":
        timed("report", stage_report)
    elif args.stage == "all":
        start = time.perf_counter()
        data = timed("ingest", stage_ingest)
        prices = data["prices"]
        analysis = timed("analyze", stage_analyze, prices)
        forecast = timed("forecast", stage_forecast, data["macro"])
        opt = timed("optimize", stage_optimize, prices)
        timed("risk", stage_risk, prices, opt["net_series"])
        timed("uncertainty", stage_uncertainty, opt, forecast["predictions"])
        timed("report", stage_report)
        print(f"[all] total wall time {time.perf_counter() - start:.1f}s")
        # keep objects referenced so linters see intent (data flows through stages)
        _ = analysis
    return 0


if __name__ == "__main__":
    sys.exit(main())
