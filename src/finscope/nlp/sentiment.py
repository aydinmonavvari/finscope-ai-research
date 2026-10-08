"""Optional sentiment utility (VADER) over a bundled, clearly-labeled sample.

The bundled CSV (``data/samples/sample_headlines.csv``) is an *illustrative
sample authored by the researcher* to exercise the reporting integration
end to end. It is NOT a market dataset and carries no evidential weight; the
rows describe the same seven tickers as the rest of the workbench but were
written for this repository. The full sentiment benchmark (Financial
PhraseBank; VADER vs TF-IDF vs MiniLM vs FinBERT) lives in the sibling
repository ``financial-nlp-sentiment``.

This module is purely local: no network access is used or required.
"""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SAMPLE_NOTE = "Illustrative sample authored by the researcher; not a market dataset."


def label_from_compound(compound: float, pos: float = 0.05, neg: float = -0.05) -> str:
    """Map a VADER compound score to a label (VADER's documented convention).

    ``compound >= 0.05`` → positive, ``compound <= -0.05`` → negative,
    otherwise neutral. Thresholds are inclusive at the boundary.
    """
    if compound >= pos:
        return "positive"
    if compound <= neg:
        return "negative"
    return "neutral"


def score_headlines(csv_path) -> pd.DataFrame:
    """Score the bundled headlines with VADER; adds ``compound`` and ``label``."""
    frame = pd.read_csv(csv_path)
    required = {"date", "ticker", "headline"}
    if not required.issubset(frame.columns):
        raise ValueError(f"headline sample must contain columns {sorted(required)}")
    analyzer = SentimentIntensityAnalyzer()
    compounds = [analyzer.polarity_scores(str(h))["compound"] for h in frame["headline"]]
    frame["compound"] = compounds
    frame["label"] = [label_from_compound(c) for c in compounds]
    return frame


def summarize(scored: pd.DataFrame) -> dict:
    """Compact summary of a scored sample (used in the research brief)."""
    counts = scored["label"].value_counts().to_dict()
    return {
        "n_headlines": int(len(scored)),
        "mean_compound": float(scored["compound"].mean()),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "tickers": sorted(scored["ticker"].unique().tolist()),
        "note": SAMPLE_NOTE,
        "benchmark_repo": "financial-nlp-sentiment",
    }
