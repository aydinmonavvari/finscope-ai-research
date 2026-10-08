"""Offline tests for reporting, portability, CLI smoke and the bundled sample."""

import json
import subprocess
import sys

import pandas as pd

from finscope.config import PROJECT_ROOT, SAMPLES_DIR
from finscope.nlp.sentiment import label_from_compound, score_headlines, summarize
from finscope.reporting.brief import render_brief, write_reports


def test_brief_includes_mandatory_markers():
    metrics = {"generated": "2026-10-08T00:00:00+00:00"}
    text = render_brief(metrics)
    assert "Findings & limitations" in text
    assert "not investment advice" in text
    assert "## 7." in text


def test_brief_limitations_use_measured_scope_numbers():
    """Limitations must render the MEASURED scope numbers from the fragments.

    Regression guard for the hard-coded 'n≈36 origins' / '~92 monthly
    observations' strings: the brief generator reads n_origins (forecast) and
    n_months (risk) from the stage fragments instead.
    """
    metrics = {
        "forecast": {"n_origins": 35},
        "risk": {"var_table": {"equal": {"n_months": 93}}},
    }
    text = render_brief(metrics)
    assert "low-powered at n=35 origins" in text
    assert "quantile estimates on 93 monthly observations" in text
    assert "≈36" not in text and "~92" not in text and "36 origins" not in text
    # without the fragments the text stays honest (no invented numbers)
    fallback_text = render_brief({"generated": "2026-10-08T00:00:00+00:00"})
    assert "low-powered at the available number of rolling origins" in fallback_text
    assert "quantile estimates on small monthly samples" in fallback_text


def test_brief_renders_stage_tables():
    metrics = {
        "analysis": {
            "tickers_used": ["SPY"],
            "window": {"start": "2018-01-02", "end": "2026-10-07", "n_days": 2200},
            "tickers": {
                "SPY": {
                    "cagr": 0.11, "ann_vol": 0.18, "sharpe": 0.6, "max_dd": -0.34,
                    "excess_kurtosis": 5.0, "jb_p": 1e-8, "adf_returns_p": 1e-9,
                    "ljungbox_p": 0.02,
                }
            },
        },
        "forecast": {
            "target": "12-month CPI inflation (%)",
            "step_months": 3, "n_origins": 30, "first_origin": "2016-12-31",
            "last_origin": "2025-09-30",
            "models": {
                "seasonal_naive": {"rmse": 2.0, "mase": 1.0},
                "ols": {"rmse": 1.5, "mase": 0.75, "dm_stat": 1.2, "dm_p": 0.23},
            },
            "verdict": "benchmark holds.",
        },
    }
    text = render_brief(metrics)
    assert "seasonal naive" in text
    assert "Rolling origins" in text or "rolling origins" in text
    assert "0.75" in text


def test_metrics_and_brief_contain_no_absolute_paths(tmp_path):
    metrics = {"analysis": {"window": {"start": "2018-01-02", "end": "2026-10-07", "n_days": 2200}}}
    metrics_path, brief_path = write_reports(metrics, reports_dir=tmp_path)
    metrics_text = metrics_path.read_text()
    brief_text = brief_path.read_text()
    for text in (metrics_text, brief_text):
        assert "/home/" not in text
        assert str(tmp_path) not in text
        assert str(PROJECT_ROOT) not in text
    payload = json.loads(metrics_text)
    assert payload["paths"]["brief"] == "reports/research_brief.md"


def test_cli_help_smoke():
    script = PROJECT_ROOT / "scripts" / "run_workbench.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    for stage in ("ingest", "analyze", "forecast", "optimize", "risk", "report", "all"):
        assert stage in result.stdout


def test_vader_label_mapping_boundaries():
    assert label_from_compound(0.05) == "positive"
    assert label_from_compound(-0.05) == "negative"
    assert label_from_compound(0.0) == "neutral"
    assert label_from_compound(0.049) == "neutral"
    assert label_from_compound(-0.049) == "neutral"


def test_bundled_sample_scores_end_to_end():
    path = SAMPLES_DIR / "sample_headlines.csv"
    scored = score_headlines(path)
    assert len(scored) == 10
    assert scored["compound"].between(-1.0, 1.0).all()
    # labels must be consistent with the documented mapping
    relabeled = scored["compound"].map(label_from_compound)
    assert (relabeled == scored["label"]).all()
    assert set(scored["ticker"]).issubset({"SPY", "AAPL", "MSFT", "JPM", "XOM", "PG", "AMZN"})
    summary = summarize(scored)
    assert summary["n_headlines"] == 10
    assert "authored by the researcher" in summary["note"]
    assert sum(summary["label_counts"].values()) == 10


def test_sample_csv_is_valid_and_tracked():
    path = SAMPLES_DIR / "sample_headlines.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["date", "ticker", "headline"]
    assert frame["headline"].str.len().min() > 10
