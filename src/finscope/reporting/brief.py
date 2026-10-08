"""Evidence-first reporting: metrics fragments, the research brief, metrics.json.

Every stage writes one JSON fragment under ``reports/``; the report stage
merges them into ``reports/metrics.json`` and renders
``reports/research_brief.md``. All paths written into outputs are RELATIVE to
the project root so the reports stay portable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from finscope.config import REPORTS_DIR

STAGE_FRAGMENTS = ("analysis", "forecast", "portfolio", "risk", "sentiment")
DISCLAIMER = (
    "Finscope is a research workbench for evaluating methods on public data. "
    "Educational research output; not investment advice, not a market prediction, "
    "and not a trading system."
)


def write_fragment(name: str, payload: dict, reports_dir: Path | None = None) -> Path:
    """Persist one stage fragment as ``reports/{name}.json``."""
    reports_dir = reports_dir if reports_dir is not None else REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def collect_metrics(reports_dir: Path | None = None) -> dict:
    """Merge all stage fragments into one metrics dict (missing stages skipped)."""
    reports_dir = reports_dir if reports_dir is not None else REPORTS_DIR
    metrics: dict[str, dict] = {}
    for name in STAGE_FRAGMENTS:
        path = reports_dir / f"{name}.json"
        if path.exists():
            metrics[name] = json.loads(path.read_text())
    return metrics


def _fmt(value, digits: int = 2, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{float(value) * 100:.{digits}f}%"
    return f"{float(value):.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_brief(metrics: dict) -> str:
    """Render the full research brief from a metrics dict (defensive: any stage
    may be absent, in which case the section says so explicitly)."""
    analysis = metrics.get("analysis", {})
    forecast = metrics.get("forecast", {})
    portfolio = metrics.get("portfolio", {})
    risk = metrics.get("risk", {})
    sentiment = metrics.get("sentiment", {})
    window = analysis.get("window", {})
    lines: list[str] = []
    lines.append("# Finscope Research Brief")
    lines.append("")
    lines.append(
        f"*Generated {metrics.get('generated', 'n/a')} — end-to-end research cycle on "
        "public data (Yahoo Finance, FRED). Educational research output; "
        f"{DISCLAIMER.lower().rstrip('.')}.*"
    )
    lines.append("")

    # ---------------------------------------------------------------- scope
    lines.append("## 1. Scope and data")
    lines.append("")
    if window:
        lines.append(
            f"- Universe: {', '.join(analysis.get('tickers_used', []))} "
            f"({len(analysis.get('tickers_used', []))} tickers)."
        )
        lines.append(
            f"- Daily price window: {window.get('start')} → {window.get('end')} "
            f"({window.get('n_days')} trading days; monthly returns used for portfolios)."
        )
    else:
        lines.append("- Data window not available: run the `ingest` and `analyze` stages.")
    lines.append(
        "- Sources: Yahoo Finance (adjusted OHLC, cached to `data/raw/`, never committed) "
        "and the FRED public CSV endpoint (CPIAUCSL, UNRATE, DGS10)."
    )
    lines.append(
        "- Cost and method assumptions: 10 bps per side on turnover, rf = 0, "
        "252 trading days, 12-month forecast horizon evaluated every 3 months."
    )
    lines.append("")

    # ----------------------------------------------------------- diagnostics
    lines.append("## 2. Market diagnostics (daily log returns, 2018 → present)")
    lines.append("")
    tickers = analysis.get("tickers", {})
    if tickers:
        rows = [
            [
                t,
                _fmt(v.get("cagr"), 1, pct=True),
                _fmt(v.get("ann_vol"), 1, pct=True),
                _fmt(v.get("sharpe")),
                _fmt(v.get("max_dd"), 1, pct=True),
                _fmt(v.get("excess_kurtosis")),
                "reject" if float(v.get("jb_p", 1.0)) < 0.05 else "fail to reject",
                "stationary" if float(v.get("adf_returns_p", 1.0)) < 0.05 else "non-stationary",
                "yes" if float(v.get("ljungbox_p", 1.0)) < 0.05 else "no",
            ]
            for t, v in tickers.items()
        ]
        lines.append(
            _md_table(
                [
                    "Ticker", "CAGR", "Ann. vol", "Sharpe", "Max DD", "Ex. kurt",
                    "JB(5%)", "ADF returns", "LB(20) sig.",
                ],
                rows,
            )
        )
        lines.append("")
        lines.append(
            "Diagnostics are descriptive: heavy tails and non-normality motivate the "
            "use of drawdown- and quantile-based risk reporting below."
        )
    else:
        lines.append("Diagnostics not available: run the `analyze` stage.")
    lines.append("")

    # -------------------------------------------------------------- forecast
    target_label = forecast.get("target", "12-month CPI inflation (%)")
    lines.append(f"## 3. Forecasting study — {target_label}")
    lines.append("")
    if forecast:
        lines.append(
            f"Design: rolling origins every {forecast.get('step_months')} months "
            f"(Tashman 2000), {forecast.get('n_origins')} evaluation origins "
            f"({forecast.get('first_origin')} → {forecast.get('last_origin')}), "
            f"expanding training pool with no look-ahead."
        )
        lines.append("")
        models = forecast.get("models", {})
        rows = []
        for name, v in models.items():
            row = [
                name.replace("_", " "),
                _fmt(v.get("rmse")),
                _fmt(v.get("mase")),
            ]
            if "dm_p" in v:
                row += [_fmt(v.get("dm_stat")), _fmt(v.get("dm_p"), 3)]
            else:
                row += ["—", "—"]
            rows.append(row)
        lines.append(_md_table(["Model", "RMSE (pp)", "MASE", "DM stat", "DM p"], rows))
        lines.append("")
        lines.append(f"**Verdict.** {forecast.get('verdict', 'n/a')}")
        lines.append("")
        lines.append(
            f"Multiple-testing caution: {forecast.get('multiple_testing_note', 'n/a')}"
        )
    else:
        lines.append("Forecast study not available: run the `forecast` stage.")
    lines.append("")

    # ------------------------------------------------------------- portfolio
    lines.append("## 4. Portfolio construction (walk-forward, net of costs)")
    lines.append("")
    if portfolio:
        lines.append(
            f"Design: 12-month training window → 3-month hold, rebalanced with "
            f"{portfolio.get('cost_bps_per_side')} bps per side on turnover; "
            f"{portfolio.get('oos_months')} out-of-sample months "
            f"({portfolio.get('oos_start')} → {portfolio.get('oos_end')})."
        )
        lines.append("")
        schemes = portfolio.get("schemes", {})
        rows = []
        for name, v in schemes.items():
            rows.append(
                [
                    name.replace("_", " "),
                    _fmt(v.get("sharpe")),
                    _fmt(v.get("cagr"), 1, pct=True),
                    _fmt(v.get("ann_vol"), 1, pct=True),
                    _fmt(v.get("max_dd"), 1, pct=True),
                    _fmt(v.get("avg_turnover"), 3),
                    _fmt(v.get("total_cost_pct"), 2, pct=True),
                ]
            )
        lines.append(
            _md_table(
                [
                    "Scheme", "Sharpe (net)", "CAGR", "Ann. vol", "Max DD",
                    "Avg turnover", "Total costs",
                ],
                rows,
            )
        )
        lines.append("")
        ranking = portfolio.get("ranking_net_sharpe", [])
        if ranking:
            lines.append(
                f"OOS Sharpe ranking (net of costs): "
                f"{' > '.join(r.replace('_', ' ') for r in ranking)}."
            )
        lines.append("")
        fallback_share = schemes.get("max_sharpe", {}).get("fallback_share")
        if fallback_share:
            lines.append(
                f"Reading: the max-Sharpe scheme triggered its documented minimum-variance "
                f"fallback in {fallback_share * 100:.0f}% of folds — with 12-month mean/covariance "
                "estimates the unconstrained tangency violates the per-asset cap or assigns "
                "negative weights — so it coincides with min-variance. When optimized schemes "
                "lead equal weights, margins are modest and the edge is variance reduction, "
                "not return forecasting (DeMiguel et al. 2009 estimation-error result)."
            )
        else:
            lines.append(
                "Reading: equal weights leading or matching optimized schemes replicates the "
                "well-documented estimation-error result (DeMiguel et al. 2009); short training "
                "windows make covariance estimates noisy by construction."
            )
    else:
        lines.append("Portfolio study not available: run the `optimize` stage.")
    lines.append("")

    # ------------------------------------------------------------------ risk
    lines.append("## 5. Risk report — VaR/ES and drawdowns")
    lines.append("")
    if risk:
        table = risk.get("var_table", {})
        if table:
            rows = []
            for scheme, v in table.items():
                rows.append(
                    [
                        scheme.replace("_", " "),
                        _fmt(v.get("var95"), 1, pct=True),
                        _fmt(v.get("es95"), 1, pct=True),
                        _fmt(v.get("var99"), 1, pct=True),
                        _fmt(v.get("es99"), 1, pct=True),
                    ]
                )
            lines.append("Monthly OOS net returns, historical VaR/ES (positive = loss):")
            lines.append("")
            lines.append(_md_table(["Scheme", "VaR 95", "ES 95", "VaR 99", "ES 99"], rows))
            lines.append("")
            spy = risk.get("spy_daily", {})
            if spy:
                lines.append(
                    f"Daily SPY reference (n={spy.get('n')}): hist VaR95 "
                    f"{_fmt(spy.get('var95'), 2, pct=True)}, "
                    f"ES95 {_fmt(spy.get('es95'), 2, pct=True)}, "
                    f"VaR99 {_fmt(spy.get('var99'), 2, pct=True)}, "
                    f"ES99 {_fmt(spy.get('es99'), 2, pct=True)} — a far "
                    "better-estimated quantile than the monthly figures above."
                )
                lines.append("")
        dd = risk.get("drawdowns", {})
        if dd:
            rows = [
                [
                    name.replace("_", " "),
                    _fmt(v.get("max_dd"), 1, pct=True),
                    v.get("peak_date", "n/a"),
                    v.get("trough_date", "n/a"),
                    v.get("recovery_date", "n/a"),
                ]
                for name, v in dd.items()
            ]
            lines.append(_md_table(["Scheme", "Max DD", "Peak", "Trough", "Recovered"], rows))
            lines.append("")
        lines.append(f"**Limitations.** {risk.get('limitations', 'n/a')}")
    else:
        lines.append("Risk report not available: run the `risk` stage.")
    lines.append("")

    # ------------------------------------------------------------- sentiment
    lines.append("## 6. Sentiment screen (illustrative bundled sample)")
    lines.append("")
    if sentiment:
        lines.append(
            f"- {sentiment.get('n_headlines')} headlines scored with VADER "
            f"(mean compound {sentiment.get('mean_compound', 0):.3f}); "
            f"labels: {sentiment.get('label_counts', {})}."
        )
        lines.append(
            f"- **{sentiment.get('note', '')}** No market dataset is claimed here; the full"
        )
        lines.append(
            "  sentiment benchmark lives in the sibling repository "
            f"{sentiment.get('benchmark_repo', 'financial-nlp-sentiment')}."
        )
    else:
        lines.append(
            "Sentiment screen not available: run the `report` stage "
            "(data/samples/sample_headlines.csv is bundled and tracked)."
        )
    lines.append("")

    # -------------------------------------------------------------- findings
    lines.append("## 7. Findings & limitations")
    lines.append("")
    lines.append("**Findings (evidence-first, all from the actual run above):**")
    lines.append("")
    findings = metrics.get("findings", [])
    for item in findings:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**Limitations (per module and global):**")
    lines.append("")
    limitations = metrics.get(
        "limitations",
        [
            "Data: seven US tickers and three FRED series; no survivorship-free universe, "
            "no transaction-cost microstructure, dividend/tax treatment simplified.",
            "Forecast: ~3 feature set, expanding-window OLS/ridge on modest training pools; "
            "DM comparisons are multiple and low-powered at n≈36 origins — nothing here is "
            "a forecast of future inflation.",
            "Portfolio: 12-month covariance estimates are noisy; max-Sharpe falls back to "
            "min-variance whenever the tangency solution violates caps; results are one "
            "historical path, not a distribution.",
            "Risk: quantile estimates on ~92 monthly observations carry first-order "
            "estimation risk; VaR/ES are descriptive, not guarantees.",
            "Sentiment: the bundled headlines are an illustrative authored sample, not evidence.",
            "Global: single data window, single seed, no hyperparameter search; every number "
            "is an in-context description of one historical period.",
        ],
    )
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"**Disclaimer.** {DISCLAIMER}")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    metrics: dict, reports_dir: Path | None = None
) -> tuple[Path, Path]:
    """Write ``metrics.json`` and ``research_brief.md`` (relative paths only)."""
    reports_dir = reports_dir if reports_dir is not None else REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    payload["generated"] = datetime.now(UTC).isoformat(timespec="seconds")
    payload["paths"] = {
        "brief": "reports/research_brief.md",
        "metrics": "reports/metrics.json",
        "figures": "figures/",
        "stage_fragments": [f"reports/{name}.json" for name in STAGE_FRAGMENTS],
    }
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    brief_path = reports_dir / "research_brief.md"
    brief_path.write_text(render_brief(payload) + "\n")
    return metrics_path, brief_path
