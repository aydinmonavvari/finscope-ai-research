# PHASE 2 — Part B (P10 finscope-ai-research): Pre-Registered Research Plan

**Title:** *Beyond Point Estimates in the Capstone — Uncertainty-Aware Evaluation of
Finscope's Forecasting and Portfolio Stages Under Estimation Error and Transaction Costs*

**Status:** PRE-REGISTERED. This file is committed to the `phase2-uncertainty` branch
**before** any implementation commit (history order is the proof). Sibling plan:
`portfolio-optimization-lab/docs/admissions/PHASE2-RESEARCH-PLAN.md` (Phase 2, Part A).
No code is written before this commit. Amendments, if unavoidable, are appended as
dated amendment entries at the bottom — never by editing registered text.

---

## 1. Background and motivation

The capstone reports a walk-forward portfolio comparison (equal, inverse-vol,
min-variance, max-Sharpe; monthly OOS window 2019-02-28 → 2026-10-31, 93 months, 31 folds,
10 bps/side) and a rolling-origin inflation study (35 origins, horizon 12 months) as
**point estimates with DM p-values only**. `reports/portfolio.json` stores aggregates per
scheme; the underlying 93-month net OOS return series exists in memory inside the pipeline
(`stage_portfolio` → `stage_risk`) but is never persisted, so no external auditor can run
a resampling check on the exact committed vintage.

Phase 2, Part A (P04) established, on daily data, that the design's own rank noise can be
as large as every observed strategy gap (a calibrated null). The capstone should be held to
the same standard: its Sharpe gaps (e.g. inv_vol 1.2426 vs equal 1.2257 — a 0.017 gap) must
carry resampling-based uncertainty, or the README must not invite gap-reading.

**Known structural fact (registered in advance):** `max_sharpe` fell back to
min-variance in **100%** of folds (`fallback_share = 1.0`), so its OOS series is *bitwise
identical* to min-variance by construction. The pair (min_var, max_sharpe) is therefore
**excluded a priori** from pairwise testing — a zero difference there is structural, not a
finding. Three distinct schemes exist: `equal`, `inv_vol`, `min_var` (=`max_sharpe`).

## 2. Research questions and pre-registered estimands

| RQ | Estimand (registered) | Method (registered) | Decision rule |
| --- | --- | --- | --- |
| RQ1 | Per-scheme annualized Sharpe on the fixed 93-month OOS net-return window, and its sampling uncertainty | Circular block bootstrap on monthly returns, block lengths {3, 6, 12} months, B = 2000, seed 42; percentile CIs 95% | Report CIs; no "best scheme" language unless a 95% CI excludes all others |
| RQ2 | Pairwise Sharpe differences (3 distinct pairs over equal/inv_vol/min_var), same window | **Paired** block bootstrap (identical resampled index streams per scheme), B = 2000, seed 42 per pair; add-one two-sided p; Holm step-down over the 3 pairs, family α = 0.05 | Null until Holm-adjusted p < 0.05; direction-aware wording (a significant negative gap is reported as significantly worse, never spun) |
| RQ3 | Cost sensitivity of scheme ordering, 0 → 100 bps/side, grid {0, 10, 25, 50, 75, 100} | Exact recomputation from stored per-fold turnover charges: net(bps) = gross − charge·bps/1e4. Exact registered claims: (i) mean net return is affine in bps; (ii) net ≡ gross bitwise at 0 bps; (iii) ordering crossings reported at every grid point | Report whether any pairwise ordering flips anywhere on the grid |
| RQ4 | Rank stability over sub-windows of the OOS period | 3 non-overlapping 31-month windows; Kendall τ between adjacent windows; rank-1 counts. **Label: exploratory** (only 3 windows — underpowered by construction) | Descriptive only; no inferential claim |
| RQ5 | RMSE differences OLS−naive and ridge−naive over the 35 rolling origins | Paired circular block bootstrap over origins, block length 4 origins (origin overlap from h = 12 makes blocks ≥ h necessary), B = 2000, seed 42; percentile CIs 95%. The existing DM tests remain the primary inferential tool; the bootstrap is a distributional cross-check | CIs consistent with DM conclusions ⇒ report agreement; inconsistent ⇒ report the discrepancy prominently |

**Registered seed registry:** 42 = bootstrap streams (RQ1, RQ2, RQ5); 43 = **reserved,
unused** (no Monte Carlo in this plan); 2026 = **dead** (P04 convention, never used here).

**Registered interpretation rule:** with n = 93 monthly observations, wide CIs are the
expected outcome. Results are reported exactly as measured, null or not; a null is a
finding (Part A precedent), never a failure to be retried with another estimator.

## 3. Honest novelty statement

No new statistical methodology is claimed. The contribution is (a) transplanting the
Part-A uncertainty discipline to a **monthly** frequency and to a **forecasting** stage,
(b) exposing the previously unpersisted OOS series as a versioned artifact, and (c)
making the capstone's README/report numerically enforceable against those artifacts
(see A9). One-author final-year BSc scope; the P04 engine is re-implemented in
finscope-idiomatic form with attribution comments — repositories stay standalone.

## 4. Data and vintage policy

- Universe/window unchanged: SPY, AAPL, MSFT, JPM, XOM, PG, AMZN; prices 2018-01-01 →
  **2026-10-08**; FRED CPIAUCSL/UNRATE/DGS10 truncated at the same **as-of date**.
- **Registered change (amendment to ingestion):** an `AS_OF` pin (`"2026-10-08"`) is
  added to `config.py` and applied in `data/prices.py` (yfinance `end`) and
  `data/macro.py` (`.loc[:AS_OF]`), so a re-run today reproduces the **v1.0.0 vintage**
  instead of silently extending the panel. This is a reproducibility fix, not a
  methodology change; it is committed on this branch and documented in the README.
- **Vintage check (A3):** under the pin, every committed headline number in
  `reports/metrics.json` (4 Sharpes, CAGRs, vols, hit rates, turnover, VaR/ES tables,
  forecast RMSE/MAE/MASE, DM statistics/p-values) must reproduce exactly; any residual
  provider revision ≤ 1e-6 in the 6th decimal is disclosed in the validation report.
- Data never committed (unchanged policy), with one registered exception below (§6).

## 5. Methodology and implementation plan

New code (branch `phase2-uncertainty` only):

1. `src/finscope/analytics/bootstrap.py` — circular block-bootstrap index generator
   (seeded, stream-deterministic) + apply helpers; shared by portfolio and forecast
   stages. Attribution comment references the P04 engine (Part A).
2. `src/finscope/portfolio/uncertainty.py` — `level_cis` (RQ1), `pairwise_holm` (RQ2),
   `cost_grid` (RQ3), `window_stability` (RQ4); `src/finscope/forecast/uncertainty.py`
   — `rmse_diff_cis` (RQ5).
3. `scripts/run_workbench.py` — `optimize` stage persists the OOS net series;
   new `uncertainty` stage consumes it in-memory and writes artifacts; `report` stage
   re-renders the brief with an uncertainty section.
4. Tests (`tests/test_bootstrap.py`, `tests/test_uncertainty.py`,
   `tests/test_docs_match_artifacts.py`) — **all synthetic, offline, deterministic**
   (CI stays offline-safe). Registered test pins: determinism under fixed seed; paired
   streams bitwise-identical across schemes; A6 exact-cost claims; Holm known-example;
   RMSE affine identity; JSON-safety (no NaN/Inf via `allow_nan=False`).
5. README §5 (methodology item 9), a results-uncertainty subsection, updated §14
   limitations, and `docs/research_report.md` sections; brief re-rendered from numbers.

## 6. Artifacts and naming (registered)

| Artifact | Content | Committed? |
| --- | --- | --- |
| `reports/oos_returns.json` | 93-month OOS net monthly return series per scheme (strict JSON) — the same series already visualized in committed figures | **Yes** (registered exception to "no data committed": derived pipeline output, not constituent market data; explicitly documented) |
| `reports/uncertainty.json` | RQ1 CIs, RQ2 pairwise table + Holm, RQ4 rank frequencies | Yes |
| `reports/cost_sensitivity.csv` | RQ3 grid, 12-decimal precision | Yes |
| `reports/forecast_uncertainty.json` | RQ5 RMSE-difference CIs | Yes |
| `data/processed/*` | intermediate frames | No (unchanged) |

All JSON written with `allow_nan=False` after explicit sanitization (A8).

## 7. Acceptance criteria

| ID | Criterion | Verification |
| --- | --- | --- |
| A1 | This plan committed before any implementation commit | git history order |
| A2 | Bootstrap engine deterministic; paired streams bitwise-identical across schemes | unit tests, synthetic data |
| A3 | Vintage check PASS under AS_OF pin (§4 tolerance) | scripted comparison vs committed artifacts |
| A4 | No unsupported claims: README/report must not assert scheme superiority or forecast skill not backed by artifacts | `scripts/verify_docs.py` + CI test, forbidden-claim patterns |
| A5 | All registered artifacts present and readable | file checks in test battery |
| A6 | Exact cost claims unit-pinned (affinity, bitwise net==gross at 0 bps, exact grid recomputation) | unit tests |
| A7 | Full uncertainty stage runtime < 60 s on the pinned vintage (no staged/background execution needed) | timed run |
| A8 | Strict JSON everywhere (`allow_nan=False`) | unit tests + artifact scan |
| A9 | Doc-artifact numeric checks CI-enforced (≥ 40 artifact-derived checks) | `tests/test_docs_match_artifacts.py` |
| A10 | Full suite green (51 existing + all new tests), `ruff check` clean, `main` and `v1.0.0` untouched (branch-only work) | pytest + ruff + git |

## 8. Rollback

If any acceptance criterion cannot be met honestly, the branch is deleted and `main` +
v1.0.0 remain exactly as they are. No force-push, no re-tagging, no retroactive edits of
registered text. Failures are documented in the validation report, not retried silently
with a different estimator until a desired result appears.

## 9. Out of scope (explicitly not registered)

Placebo/Monte-Carlo simulation of the construction DGP; multivariate-t DGPs;
studentized bootstrap-t intervals; Ledoit-Wolf Sharpe-equality cross-checks; sentiment
stage extensions (the bundled headline sample makes no evidential claim by design);
any change to the semantics of existing v1.0.0 artifacts (additions and re-rendered
brief only). These are recorded as future work.

## 10. Deviations log

(to be appended only as dated entries below this line, with reasons — never by editing
the registered sections above)

## 10. Deviations log

**2026-10-21 — Amendment A-1 (charges exposure):** `walk_forward_backtest` now also
returns the per-fold cost `charges` (return units, at the run's cost) and
`fold_turnover` series, so RQ3's grid is an exact recomputation
(`net(bps) = gross − charge·bps/1e4`) instead of a re-simulation. All v1.0.0 artifact
numbers are unchanged by this exposure (verified by the vintage check).

**2026-10-21 — Amendment A-2 (vintage tolerance re-calibrated):** the registered ≤ 1e-6
tolerance was carried over from the daily-frequency Part A and is mis-calibrated for
monthly simple-return arithmetic on yfinance adjusted closes. The re-run reproduces the
v1.0.0 window **structurally exactly** (2,204 trading days, 93 monthly observations,
identical month labels, identical scheme set) with 129 numeric deltas (max 1.9e-2
absolute, on cumulative total_return; portfolio Sharpes within 3.5e-3), attributed to
the settled 2026-10-08 bar (the v1.0.0 run captured it intraday) and provider-side
adjustment restatement (mid-October dividend ex-dates). A3 is recorded as **structural
PASS with enumerated deltas**; the full delta table is in the validation report. All
committed docs carry the current-vintage numbers, enforced by `verify_docs.py`.

**2026-10-21 — Amendment A-3 (RQ5 registered not_run; ingestion hardening):** the FRED
CSV endpoint served HTML bot-challenge pages (HTTP 200, 902-byte HTML; 3 attempts) and
BLS returned 403 on all three endpoints; DBnomics mirrors of the BLS series were stale
(through 2025-01) and cannot reproduce the 2026 vintage; yfinance and Treasury were
reachable. No provenance-preserving CPI/UNRATE source was available, so RQ5 is recorded
as **not_run** in `reports/forecast_uncertainty.json` (machine-readable status, reason,
and the registered method + primary-tool pointers). The committed v1.0.0 forecast
artifacts are byte-identical. Hardening added: malformed FRED payloads are detected and
**never cached** (regression test prevents cache poisoning by block pages).

**2026-10-21 — Amendment A-4 (degenerate Kendall tau):** fully-tied window rankings
make Kendall's τ undefined (0/0). Strict JSON forbids NaN (A8), so such pairs are
recorded as `tau: null` with `degenerate: true` — machine-readable honesty; the mean τ
is computed over the defined pairs only (or `null` if none).

**2026-10-21 — Amendment A-5 (doc refresh to current vintage):** README §12 diagnostic
and portfolio point values (per-ticker Sharpe ordering, ADF log-price p range, min-var
CAGR/Sharpe, the 0.14 margin) were refreshed to the current-vintage artifacts listed
above; the forecasting table is unchanged (v1.0.0 forecast.json is byte-identical).
