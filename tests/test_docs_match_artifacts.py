"""CI-enforced doc-artifact verification (A4/A9) + forbidden-claim scanner tests.

The heavy lifting lives in ``scripts/verify_docs.py``; this test wires it into
pytest so a doc whose numbers drift from the committed artifacts fails CI,
and pins the scanner's own behavior on synthetic good/bad text.
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "verify_docs.py"

_spec = importlib.util.spec_from_file_location("verify_docs", SCRIPT)
verify_docs = importlib.util.module_from_spec(_spec)
sys.modules["verify_docs"] = verify_docs
_spec.loader.exec_module(verify_docs)


def test_docs_match_artifacts_end_to_end():
    """The registered >= 40 artifact-derived checks must all pass."""
    n_checks, failures = verify_docs.build_checks()
    assert n_checks >= 40, f"registered minimum is 40 artifact-derived checks, got {n_checks}"
    assert not failures, "\n".join(failures)


def test_scanner_flags_unqualified_superiority():
    bad_texts = [
        "The min-variance scheme beats every other scheme out of sample.",
        "Ridge significantly better than the naive benchmark.",
        "The strategy outperforms in all windows.",
        "This delivers a statistically significant edge over equal weights.",
    ]
    for text in bad_texts:
        failures: list[str] = []
        checks: list[str] = []
        verify_docs._scan_forbidden_claims({"doc.md": text}, failures, checks)
        assert failures, f"scanner missed a forbidden claim: {text}"


def test_scanner_allows_qualified_negated_statements():
    good_texts = [
        "No model beats the naive benchmark.",
        "No scheme beats another once the Holm correction is applied.",
        "nothing here outperforms a coin flip; the gaps are within rank noise",
        "The naive benchmark wins; OLS is significantly worse, not better.",
        "no scheme outperforms the others after multiplicity control",
    ]
    for text in good_texts:
        failures: list[str] = []
        checks: list[str] = []
        verify_docs._scan_forbidden_claims({"doc.md": text}, failures, checks)
        assert not failures, f"scanner flagged qualified statement: {text} -> {failures}"


def test_verify_docs_cli_exit_zero():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checks passed" in result.stdout
