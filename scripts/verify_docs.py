#!/usr/bin/env python
"""Verify that the docs' numbers ARE the artifacts' numbers (A4/A9).

Extracts artifact-derived numbers (level CIs, pairwise/Holm p-values, cost
grid values, Kendall taus, RMSE-difference CIs, point estimates) from
``reports/`` and checks that README.md, docs/research_report.md and
reports/research_brief.md quote them exactly (same rounding), that the new
artifacts are strict JSON, and that no forbidden claim pattern (unqualified
superiority/skill language) appears in any doc. Exit 0 = all checks pass;
exit 1 with a listing on any mismatch. Wired into pytest via
``tests/test_docs_match_artifacts.py`` so CI enforces it.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
DOC_PATHS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "research_report.md",
    REPORTS / "research_brief.md",
)

# Numbers quoted from artifacts must appear in these docs (per check).
ALL_DOCS = ("README", "research_report", "research_brief")

# Forbidden-claim patterns. Each entry: (regex, requires_negation_nearby).
# Patterns with requires_negation_nearby=True are tolerated ONLY when a
# negation/qualification cue appears within +/- 80 characters (e.g. "no model
# beats"); the hard patterns are never tolerated in our own docs.
NEGATION_CUES = (
    "no ",
    "not ",
    "n't",
    "never",
    "nothing",
    "none ",
    "cannot",
    "can't",
    "without",
    "fails to",
    "rather than",
    "instead of",
    "hard to",
    "does not",
    "did not",
    "do not",
    "could not",
    "would not",
    "zero ",
    "no scheme",
    "no model",
    "no pairwise",
    "unqualified",
)
FORBIDDEN_PATTERNS: tuple[tuple[re.Pattern, bool], ...] = (
    (re.compile(r"significantly\s+(better|outperforms?)", re.IGNORECASE), False),
    (re.compile(r"\boutperform(s|ed)?\b", re.IGNORECASE), True),
    (re.compile(r"\bbeats?\b", re.IGNORECASE), True),
    (re.compile(r"\bsuperior (to|scheme|strategy|model)\b", re.IGNORECASE), True),
    (
        re.compile(r"statistically\s+significant\s+(edge|alpha|advantage|skill)", re.IGNORECASE),
        False,
    ),
    (re.compile(r"\b(undeniable|guaranteed)\s+(alpha|edge|skill|profit)", re.IGNORECASE), False),
)


def _load_json(path: Path):
    def _reject(value):
        raise ValueError(f"non-finite constant {value!r} in {path.name}")

    return json.loads(path.read_text(), parse_constant=_reject)


def _doc_texts() -> dict[str, str]:
    texts = {}
    for path in DOC_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"required doc missing: {path}")
        texts[path.name] = path.read_text()
    return texts


def _check_value_in_docs(
    checks: list[str], failures: list[str], needle: str, docs: dict[str, str], label: str
) -> None:
    checks.append(label)
    for doc_name, text in docs.items():
        if needle not in text:
            failures.append(f"{label}: {needle!r} not found in {doc_name}")


def _scan_forbidden_claims(docs: dict[str, str], failures: list[str], checks: list[str]) -> None:
    for doc_name, text in docs.items():
        for pattern, needs_negation in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                if needs_negation:
                    window = text[max(0, match.start() - 80) : match.end() + 80]
                    lowered = window.lower()
                    if any(cue in lowered for cue in NEGATION_CUES):
                        continue
                failures.append(
                    f"forbidden claim pattern {pattern.pattern!r} in {doc_name} "
                    f"near: ...{text[max(0, match.start() - 40) : match.end() + 40]}..."
                )
        checks.append(f"forbidden-claim scan: {doc_name}")


def build_checks() -> tuple[int, list[str]]:
    """Run every artifact-derived check; return (n_checks, failures)."""
    checks: list[str] = []
    failures: list[str] = []
    docs = _doc_texts()

    uncertainty = _load_json(REPORTS / "uncertainty.json")
    forecast_unc = _load_json(REPORTS / "forecast_uncertainty.json")
    portfolio = _load_json(REPORTS / "portfolio.json")
    oos = _load_json(REPORTS / "oos_returns.json")

    # ------------------------------------------------ artifact-artifact pins
    # point estimates in uncertainty.json must BE the portfolio.json Sharpes
    points = uncertainty["level_cis"]["point_estimates"]
    for scheme, value in points.items():
        checks.append(f"artifact pin: point_sharpe[{scheme}] == portfolio.json sharpe")
        artifact_sharpe = portfolio["schemes"][scheme]["sharpe"]
        if round(artifact_sharpe, 6) != value:
            failures.append(
                f"point_sharpe[{scheme}] {value} != round(portfolio.json {artifact_sharpe}, 6)"
            )
    checks.append("artifact pin: oos_returns has 4 schemes x 93 months")
    for scheme, series in oos["net"].items():
        if len(series) != oos["metadata"]["n_months"]:
            failures.append(f"oos_returns net[{scheme}] length {len(series)} != n_months")
    if oos["net"]["min_var"] != oos["net"]["max_sharpe"]:
        failures.append("oos_returns: min_var and max_sharpe net series differ (structural fact)")

    # -------------------------------------------------- RQ1: level CIs
    levels = uncertainty["level_cis"]["levels"]
    for scheme in ("equal", "inv_vol", "min_var", "max_sharpe"):
        ci = levels[scheme]["6"]
        needle = f"[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]"
        _check_value_in_docs(
            checks, failures, needle, docs, f"RQ1 CI block6 {scheme} '{needle}'"
        )
        point_needle = f"{points[scheme]:.2f}"
        _check_value_in_docs(
            checks, failures, point_needle, docs, f"RQ1 point {scheme} '{point_needle}'"
        )

    # ------------------------------------- RQ2: pairwise diffs + Holm p-values
    for pair in uncertainty["pairwise_holm"]["pairs"]:
        diff_needle = f"{pair['diff']:.2f}"
        _check_value_in_docs(
            checks, failures, diff_needle, docs, f"RQ2 diff {pair['pair']} '{diff_needle}'"
        )
        ci_needle = f"[{pair['ci_low']:.2f}, {pair['ci_high']:.2f}]"
        _check_value_in_docs(
            checks, failures, ci_needle, docs, f"RQ2 CI {pair['pair']} '{ci_needle}'"
        )
        raw_needle = f"{pair['p_raw']:.3f}"
        _check_value_in_docs(
            checks, failures, raw_needle, docs, f"RQ2 raw p {pair['pair']} '{raw_needle}'"
        )
        holm_needle = f"{pair['p_holm']:.3f}"
        _check_value_in_docs(
            checks, failures, holm_needle, docs, f"RQ2 Holm p {pair['pair']} '{holm_needle}'"
        )
    n_reject = uncertainty["pairwise_holm"]["n_reject_holm_05"]
    family = uncertainty["pairwise_holm"]["family_size"]
    _check_value_in_docs(
        checks,
        failures,
        f"{n_reject} of {family}",
        docs,
        f"RQ2 outcome '{n_reject} of {family}'",
    )

    # -------------------------------------------------- RQ3: cost grid
    grid_rows = {}
    with (REPORTS / "cost_sensitivity.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            grid_rows[(row["scheme"], float(row["cost_bps"]))] = float(row["sharpe_net"])
    for scheme in ("equal", "min_var"):
        for bps in (0.0, 100.0):
            needle = f"{grid_rows[(scheme, bps)]:.2f}"
            _check_value_in_docs(
                checks, failures, needle, docs, f"RQ3 sharpe {scheme}@{bps:.0f}bps '{needle}'"
            )

    # -------------------------------------------------- RQ4: Kendall taus
    windows = uncertainty["window_stability"]
    for i, tau in enumerate(windows["adjacent_kendall_tau"], start=1):
        tval = tau["tau"] if isinstance(tau, dict) else tau
        if tval is None:
            _check_value_in_docs(
                checks, failures, "undefined (tied)", docs,
                f"RQ4 tau W{i}->W{i+1} degenerate marker",
            )
        else:
            needle = f"{tval:.2f}"
            _check_value_in_docs(
                checks, failures, needle, docs, f"RQ4 tau W{i}->W{i+1} '{needle}'"
            )
    mean_tau = windows["mean_adjacent_kendall_tau"]
    if mean_tau is not None:
        _check_value_in_docs(
            checks, failures, f"{mean_tau:.2f}", docs, "RQ4 mean tau"
        )

    # ------------------------------------------ RQ5: RMSE-difference CIs
    if forecast_unc.get("status") == "not_run":
        # Registered not-run path (PHASE-2 plan section 10): the RQ5 numbers
        # do not exist in this environment; docs must carry the explicit
        # not_run disclosure instead of any RMSE-difference figures.
        _check_value_in_docs(
            checks, failures, "not_run", docs, "RQ5 not_run disclosure marker"
        )
    else:
        for model in ("ols", "ridge"):
            pair = forecast_unc["pairs"][model]
            diff_needle = f"{pair['diff']:.2f}"
            _check_value_in_docs(
                checks,
                failures,
                diff_needle,
                docs,
                f"RQ5 diff {model} '{diff_needle}'",
            )
            ci_needle = f"[{pair['ci_low']:.2f}, {pair['ci_high']:.2f}]"
            _check_value_in_docs(
                checks,
                failures,
                ci_needle,
                docs,
                f"RQ5 CI {model} '{ci_needle}'",
            )
        # DM agreement statements must reach the brief
        for model, entry in forecast_unc["dm_agreement"].items():
            needle = entry["agreement"]
            brief_text = docs["research_brief.md"]
            checks.append(f"RQ5 agreement {model} in brief")
            if needle not in brief_text:
                failures.append(
                    f"RQ5 agreement {model}: {needle!r} not found in the brief"
                )

    #
    # ------------------------------------------------ forbidden claims (A4)
    _scan_forbidden_claims(docs, failures, checks)

    # ------------------------------------------------ strict JSON scan (A8)
    for name in (
        "uncertainty.json",
        "forecast_uncertainty.json",
        "oos_returns.json",
    ):
        text = (REPORTS / name).read_text()
        checks.append(f"strict JSON scan: {name}")
        if "NaN" in text or "Infinity" in text:
            failures.append(f"{name} contains NaN/Infinity tokens")

    return len(checks), failures


def main() -> int:
    try:
        n_checks, failures = build_checks()
    except FileNotFoundError as exc:
        print(f"verify_docs: MISSING INPUT — {exc}")
        return 1
    print(f"verify_docs: {n_checks - len(failures)}/{n_checks} checks passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"verify_docs: {len(failures)} FAILURE(S)")
        return 1
    print("verify_docs: OK — docs match artifacts; no forbidden claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
