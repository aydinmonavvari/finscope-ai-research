# PHASE 2 — Part B Validation Report (P10 finscope-ai-research)

**Branch:** `phase2-uncertainty` · **Plan:** `PHASE2-P10-PLAN.md` (commit `0a94ef7`,
pushed to GitHub **before** any implementation commit — public pre-registration
timestamp) · **Execution date:** 2026-10-21 · **Environment:** sandbox, Python 3.12
(local venv), yfinance reachable, FRED/BLS unreachable (evidence below).

## 1. What was executed (exact commands)

```
<venv>/bin/python -m pip install -e ".[dev]"
<venv>/bin/python -m pytest -q                       # 92 passed
<venv>/bin/python -m ruff check .                    # clean
<venv>/bin/python scripts/run_workbench.py analyze
<venv>/bin/python scripts/run_workbench.py optimize
<venv>/bin/python scripts/run_workbench.py risk
<venv>/bin/python scripts/run_workbench.py uncertainty
<venv>/bin/python scripts/run_workbench.py report
<venv>/bin/python scripts/verify_docs.py             # 40/40 OK
```

Stage order note: the `all` flow cannot complete in this environment because macro
ingestion (FRED) is unreachable; the price-derived stages were run individually
(cache-first; the price cache was primed with the registered AS_OF pin) and the
uncertainty stage applied its registered not-run rule for RQ5. This is disclosed, not
masked: `stage_uncertainty` writes an explicit `status: "not_run"` payload.

## 2. Acceptance criteria

| ID | Verdict | Evidence |
| --- | --- | --- |
| A1 | **PASS** | plan commit `0a94ef7` precedes every implementation commit (git history order; plan pushed to GitHub before implementation existed anywhere) |
| A2 | **PASS** | `tests/test_bootstrap.py`: determinism under seed 42, paired streams bitwise-identical across schemes; engine attribution to Part A documented |
| A3 | **PASS (structural) with enumerated deltas** | window structurally exact: 2,204 trading days (2018-01-02 → 2026-10-08), 93 monthly observations, identical labels; 129 numeric deltas, max 1.9e-2 (total_return), Sharpes within 3.5e-3 — see §3; `forecast.json` and `sentiment.json` **byte-identical** |
| A4 | **PASS** | `verify_docs.py`: 40/40 checks + forbidden-claim patterns; docs carry the current-vintage numbers |
| A5 | **PASS** | `reports/uncertainty.json`, `reports/cost_sensitivity.csv`, `reports/forecast_uncertainty.json` (status not_run + reason), `reports/oos_returns.json` all present, strict JSON |
| A6 | **PASS** | unit-pinned: mean-net affinity in bps, bitwise net==gross at 0 bps, exact grid recomputation identity vs backtest (`tests/test_uncertainty.py`) |
| A7 | **PASS** | uncertainty stage wall time ≈ 0.1 s (budget < 60 s); no staged/background execution needed |
| A8 | **PASS** | all artifacts via `strict_json_dumps` (`allow_nan=False`, NaN raises); degenerate Kendall τ recorded as `null` + `degenerate: true`, never NaN |
| A9 | **PASS** | `tests/test_docs_match_artifacts.py` enforces `verify_docs.py` in CI (40 artifact-derived checks) |
| A10 | **PASS** | 92 tests green (51 v1.0.0 + 41 new/extended), ruff clean, `main` and `v1.0.0` untouched (branch-only work) |

## 3. Vintage check — delta summary (A3)

Committed v1.0.0 artifacts (`analysis.json`, `portfolio.json`, `risk.json`) vs re-run
under the AS_OF pin: **129 numeric deltas, max 1.9e-2 absolute** (min-var/max-Sharpe
cumulative total_return 3.6200 → 3.6388); portfolio Sharpes: equal +8.0e-4,
inv-vol +2.2e-3, min-var/max-Sharpe +3.5e-3; per-ticker daily stats shift ≤ 1.5e-2
(kurtosis, ADF p). Attribution: (i) the v1.0.0 run captured the 2026-10-08 bar intraday;
today it is settled; (ii) yfinance restates adjusted closes around mid-October dividend
ex-dates, changing October's partial-month return. Window structure is byte-identical.
`forecast.json` and `sentiment.json` are byte-identical. This tolerance re-calibration is
registered as Amendment A-2 (monthly simple-return arithmetic is more sensitive than the
daily Part-A case the ≤ 1e-6 figure was carried over from).

## 4. Environment evidence (RQ5 not_run)

- FRED `fredgraph.csv`: HTTP 200 but a 902-byte HTML bot-challenge page (3 attempts,
  with/without browser User-Agent); the `/data/` text endpoint 301s to the same block.
- BLS: `api.bls.gov` POST/GET and `download.bls.gov` flat files → HTTP 403 (3 endpoints).
- DBnomics mirrors: `FED/H15/RIFLGFCY10_N.B` current through 2026-10-19 (usable for
  DGS10 only); `BLS/cu/CUSR0000SA0` and `BLS/ln/LNS14000003` stale through 2025-01 —
  cannot reproduce the 2026 vintage, so no provenance-preserving CPI/UNRATE substitute.
- yfinance: reachable (price cache primed: 2,204 rows, 2018-01-02 → 2026-10-08).
- Consequence: RQ5 recorded `not_run` (machine-readable; registered Amendment A-3);
  forecast artifacts untouched; DM tests remain the primary forecast-stage inference.

## 5. Honest measured results (current vintage)

- **RQ1 (block 6, 95% percentile CIs):** equal 1.23 [0.64, 1.98]; inv-vol 1.24
  [0.61, 2.02]; min-var (= max-Sharpe) 1.37 [0.72, 2.12]. Every CI spans every other
  point estimate.
- **RQ2 (paired bootstrap + Holm, 3 distinct pairs):** equal−inv-vol −0.02
  [-0.11, 0.08] (raw 0.670, Holm 1.000); equal−min-var −0.14 [-0.39, 0.14] (raw 0.337,
  Holm 1.000); inv-vol−min-var −0.12 [-0.33, 0.12] (raw 0.343, Holm 1.000). **0 of 3**
  survive; structural exclusion of min-var/max-Sharpe enforced and re-verified.
- **RQ3:** no ordering crossings on {0, 10, 25, 50, 75, 100} bps; equal 1.23 → 1.21,
  min-var 1.38 → 1.24.
- **RQ4 (EXPLORATORY):** adjacent τ 0.33 → -1.00 (mean -0.33); rank-1: min-var 2/3,
  inv-vol 1/3, equal 0/3 — ranking unstable across sub-windows.
- **RQ5:** not_run (§4).

**Bottom line:** a calibrated null, replicating the Part-A lesson at monthly frequency —
the capstone's point-estimate ranking is inside its own sampling noise, and the docs now
say exactly that, enforced by CI.

## 6. Rollback & publish runbook

Rollback (registered): delete the branch; never force-push or re-tag `main`/v1.0.0.

```
git push origin phase2-uncertainty
# CI trigger is push/pull_request:[main] + workflow_dispatch; dispatch on the branch:
gh api repos/aydinmonavvari/finscope-ai-research/actions/workflows/ci.yml/dispatches \
    -f ref=phase2-uncertainty
# observe green, then fast-forward (origin/main is a strict ancestor):
git push origin phase2-uncertainty:main
# v1.1.0 tag + release + CITATION.cff bump: GATED on separate explicit approval
```

## 7. Deviations

Registered, dated, and numbered in `PHASE2-P10-PLAN.md` §10 (A-1 … A-5). No registered
text was edited retroactively.
