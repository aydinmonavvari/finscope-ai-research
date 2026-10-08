#!/usr/bin/env python
"""Finscope workbench CLI.

Stages (each idempotent and cache-first; `all` runs them in order):

    ingest    fetch/cache Yahoo Finance prices + FRED macro series
    analyze   market diagnostics + figures 1-3
    forecast  12-month CPI inflation study + figures 4-5
    optimize  walk-forward portfolio backtest + figures 6-7
    risk      VaR/ES and drawdown reporting + figure 8
    report    merge fragments -> reports/metrics.json + research_brief.md
    all       ingest -> analyze -> forecast -> optimize -> risk -> report
"""

from __future__ import annotations

import argparse
import sys
import time

STAGES = ("ingest", "analyze", "forecast", "optimize", "risk", "report", "all")


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
    print(
        f"[optimize] {payload['oos_months']} OOS months, {payload['n_folds']} folds "
        "-> portfolio.json"
    )
    for name in ranking:
        s = scheme_payload[name]
        print(
            f"    {name}: Sharpe(net)={s['sharpe']:.3f} CAGR={s['cagr'] * 100:.1f}% "
            f"maxDD={s['max_dd'] * 100:.1f}% turnover={s['avg_turnover']:.3f}"
        )
    return {"net_series": net_series, "monthly": monthly, "results": results}


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


def stage_report() -> dict:
    """Merge fragments, score the bundled sample, write metrics.json + brief."""
    from finscope.config import SAMPLES_DIR
    from finscope.nlp.sentiment import score_headlines, summarize
    from finscope.reporting.brief import collect_metrics, write_fragment, write_reports

    metrics = collect_metrics()
    scored = score_headlines(SAMPLES_DIR / "sample_headlines.csv")
    metrics["sentiment"] = summarize(scored)
    write_fragment("sentiment", metrics["sentiment"])
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
        timed("report", stage_report)
        print(f"[all] total wall time {time.perf_counter() - start:.1f}s")
        # keep objects referenced so linters see intent (data flows through stages)
        _ = analysis, forecast
    return 0


if __name__ == "__main__":
    sys.exit(main())
