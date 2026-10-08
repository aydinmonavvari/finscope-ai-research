# finscope-ai-research

![CI](https://github.com/aydinmonavvari/finscope-ai-research/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1 · Short description

**Finscope is the integrating capstone of a ten-repository research portfolio**: a modular,
reproducible workbench that runs one complete financial-research cycle on real public data —
ingestion → diagnostic analytics → a forecasting study → portfolio construction → risk
reporting → an evidence-first *research brief* — and documents the epistemic limits of every
stage in the same artifacts that carry the numbers.

What Finscope **is**:

- a unified orchestration of methods demonstrated across the nine sibling repositories,
  with shared configuration, one CLI, and a single honest reporting layer;
- a methods-evaluation harness: every model is compared against a defensible baseline,
  out of sample, net of stated costs, with the comparison test reported alongside.

What Finscope **is NOT**:

- **not a trading system** and **not investment advice** — nothing here predicts markets,
  selects live positions, or times anything;
- not a copy of any sibling repository — its value is the unification, the shared config,
  and the reporting discipline, not any single model;
- not a data vendor — market and macro data are fetched from public endpoints, cached
  locally, and never committed.

*Educational research project (BSc finance portfolio). Run date 2026-10-08; every number in
this README is an actual output of the committed pipeline (`reports/metrics.json`).*

## 2 · Research question

> Can a full research cycle — diagnostics, forecasting, portfolio construction, risk
> measurement — be executed as one reproducible pipeline whose every headline claim is
> backed by an out-of-sample comparison against a naive benchmark, and whose limitations
> are documented as rigorously as its results?

## 3 · Motivation

Individual studies answer narrow questions; real research fails at the seams between them.
Data silently drifts between notebooks, one stage's assumptions contradict another's, and
conclusions are summarized more boldly than their evidence. Finscope is built against those
failure modes: one configuration module feeds every stage, each stage writes a JSON fragment
that the reporting layer merges, and the final brief is rendered *from* the measured numbers —
the honest framing is code, not prose. It is also the portfolio's demonstration of
integration skill: the same author who built nine focused studies shows they compose into a
disciplined whole.

## 4 · Why this matters

- **For finance:** the full cycle mirrors how a small research desk actually works — clean
  ingestion, descriptive diagnostics, a forecasting question, portfolio implications, risk
  reporting, and a written brief a reviewer could audit end to end.
- **For machine learning / statistics:** baseline-first evaluation (seasonal naive,
  equal weights), rolling-origin design, direction-aware significance tests, and cost-aware
  backtesting are the transferable core; the specific models are deliberately simple.
- **For research integrity:** the capstone's headline finding is a *negative* result (no
  forecasting model beats the seasonal-naive benchmark), reported with the same rigor as
  the portfolio table — evidence-first reporting, stated assumptions, explicit limits.

## 5 · Methodology

Per module, in pipeline order:

1. **`data/prices.py` — ingestion.** Daily adjusted closes for SPY, AAPL, MSFT, JPM, XOM,
   PG, AMZN (2018-01-01 → present) via `yfinance`; cache-first: reads `data/raw/` when
   present (offline-safe), downloads only on a cold cache or `--refresh`. Month-end
   resampling produces the monthly panel used downstream.
2. **`data/macro.py` — macro ingestion.** FRED public CSV endpoint (no API key): CPIAUCSL,
   UNRATE, DGS10; the daily 10-year yield is aggregated to monthly means; cache-first,
   offline after the first fetch.
3. **`analytics/stats.py` — diagnostics.** Log/simple returns, annualized vol and Sharpe
   (252 days, rf = 0 — documented simplification), max drawdown, Jarque-Bera normality,
   ADF stationarity (on log returns *and* log prices), Ljung-Box(20), rolling 63-day vol.
4. **`forecast/inflation.py` — forecasting study.** Target: 12-month CPI inflation
   (y/y %). Models: seasonal naive (the last observed annual inflation — the correct
   seasonal-naive forecast for an annual target), OLS and ridge (α = 1, standardized) on
   three origin-known features (inflation, unemployment, 10-y yield). Rolling origins
   every 3 months (Tashman 2000); the training pool at each origin contains only rows
   whose 12-month-ahead target was already observed — no look-ahead by construction.
   Metrics: RMSE, MAE, MASE (Hyndman & Koehler 2006), Diebold-Mariano (1995) with
   Newey-West HAC at lag h−1, **direction-aware** (a "significant" p-value in the
   worsening direction is reported as significantly *worse*, never as a win).
5. **`portfolio/construct.py` — construction + backtest.** Four schemes on monthly
   returns: equal weight, inverse-vol, minimum variance (scipy SLSQP, objective scaled
   ×1e3 for conditioning, per-asset cap 40%) and max-Sharpe via closed-form tangency
   w ∝ Σ⁻¹μ with a documented min-variance fallback whenever the tangency violates the
   cap or has negative weights. Walk-forward backtest: 12-month train → 3-month hold,
   31 non-overlapping folds; 10 bps per side charged on turnover (drifted weights vs
   fresh target; entry from cash counts as full turnover).
6. **`risk/var.py` — risk reporting.** Historical and Gaussian VaR/ES at 95/99% on the
   OOS net monthly returns (positive = loss), drawdown table with peak/trough/recovery
   dates, and a fixed limitations note on quantile estimation risk.
7. **`nlp/sentiment.py` — sentiment utility.** VADER (Hutto & Gilbert 2014) over ten
   bundled headlines (`data/samples/sample_headlines.csv`) that describe the same seven
   tickers. The CSV is an **illustrative sample authored by the researcher** — not a
   market dataset, no evidential claim; the real sentiment benchmark lives in
   [`financial-nlp-sentiment`](https://github.com/aydinmonavvari/financial-nlp-sentiment).
8. **`reporting/brief.py` — the brief.** Merges per-stage JSON fragments into
   `reports/metrics.json` and renders `reports/research_brief.md` *from those numbers*,
   including a mandatory "Findings & limitations" section and the disclaimer. All paths
   in outputs are relative — no absolute sandbox paths leak into reports.

## 6 · Dataset

| Property | Value |
| --- | --- |
| Price universe | SPY, AAPL, MSFT, JPM, XOM, PG, AMZN (7 tickers, daily adjusted close) |
| Price window (actual) | 2018-01-02 → 2026-10-08 — 2,204 trading days; 2,203 daily return observations |
| Monthly panel | 104 month-end observations; portfolio OOS: 93 months (2019-02-28 → 2026-10-31) |
| Macro series | CPIAUCSL (CPI index), UNRATE (unemployment), DGS10 (10-y yield, monthly mean) |
| Macro window | 2013-01 → 2026-10 (166 months; extra history gives the lagged features depth) |
| Forecast origins | 35 rolling origins, 2016-12-31 → 2025-06-30 (target observed through 2026-06) |
| Bundled sample | 10 researcher-authored headlines (tracked in git; the only committed data file) |

Note: the final monthly return (labelled 2026-10-31) is a *partial* month — October 2026
contributes 6 trading days at the run date. Documented in §14.

## 7 · Data sources

- **Yahoo Finance** via `yfinance` (daily adjusted OHLC) — fetched once, cached to
  `data/raw/prices_daily.csv` (git-ignored; the repository never redistributes market data).
- **FRED public CSV endpoint** (`fred.stlouisfed.org/graph/fredgraph.csv?id=…`) — no API
  key; cached to `data/raw/fred_*.csv`.
- No SEC/Wayback/proxy sources are used anywhere in this project.
- The bundled headline sample is authored, versioned in git, and labeled as illustrative.

## 8 · Architecture

```
              ┌────────────────────────── config.py (single source of truth) ─────────┐
              │  tickers · dates · FRED ids · seeds · 10 bps/side · caps · horizons   │
              └───────────────────────────────────────────────────────────────────────┘
                                            │
   Yahoo Finance ─┐                  ┌──────┴───────┐
   (cache-first)  ├─► data/prices.py ─┤   ANALYZE    │ analytics/stats.py
   FRED CSV  ─────┴─► data/macro.py ──┤ (diagnostics: JB · ADF · Ljung-Box · drawdowns)
                                      └──────┬───────┘
                         ┌────────────────────┼───────────────────────┐
                         ▼                    ▼                       ▼
                 ┌───────────────┐   ┌───────────────────┐   ┌──────────────────┐
                 │   FORECAST    │   │     OPTIMIZE      │   │       RISK       │
                 │ 12m CPI infl. │   │ 4 schemes · 12m→3m│   │ hist+Gauss VaR/ES│
                 │ naive/OLS/ridge│  │ walk-forward·10bps│   │ drawdown table   │
                 │ RMSE·MASE·DM  │   │ SLSQP·tangency    │   │ 95/99%           │
                 └───────┬───────┘   └─────────┬─────────┘   └────────┬─────────┘
                         └──────────── reports/*.json fragments ───────┘
                                            │
                          nlp/sentiment.py (VADER on bundled sample)
                                            │
                          reporting/brief.py ─► metrics.json + research_brief.md
                                        (findings & limitations, relative paths)
```

**The ten-repository portfolio map** (Finscope unifies their methodologies):

| # | Repository | One-line role |
| --- | --- | --- |
| 01 | [`finance-data-analysis-lab`](https://github.com/aydinmonavvari/finance-data-analysis-lab) | Market diagnostics on real price data: returns, vol, drawdowns, normality/stationarity tests |
| 02 | [`macro-forecasting-lab`](https://github.com/aydinmonavvari/macro-forecasting-lab) | Rolling-origin macro forecasting (SARIMA/OLS/GBM) vs naive benchmarks, DM tests |
| 03 | [`ml-market-prediction-study`](https://github.com/aydinmonavvari/ml-market-prediction-study) | Honest negative result: daily direction prediction at chance level, leakage-controlled |
| 04 | [`portfolio-optimization-lab`](https://github.com/aydinmonavvari/portfolio-optimization-lab) | Markowitz schemes, risk parity, tangency robustness, cost-aware OOS backtests |
| 05 | [`credit-risk-modeling`](https://github.com/aydinmonavvari/credit-risk-modeling) | PD modeling with calibration, cost-based thresholds, fairness diagnostics |
| 06 | [`fraud-anomaly-detection`](https://github.com/aydinmonavvari/fraud-anomaly-detection) | Fraud/anomaly detection under extreme imbalance; ROC vs PR lesson |
| 07 | [`dl-financial-time-series`](https://github.com/aydinmonavvari/dl-financial-time-series) | Tiny deep-learning architectures on SPY direction — another honest null |
| 08 | [`financial-nlp-sentiment`](https://github.com/aydinmonavvari/financial-nlp-sentiment) | Financial PhraseBank benchmark: VADER → TF-IDF → MiniLM → FinBERT |
| 09 | [`fin-rag-research-assistant`](https://github.com/aydinmonavvari/fin-rag-research-assistant) | Retrieval-augmented research assistant over SEC filings |
| 10 | **`finscope-ai-research`** (this repo) | **Integrating capstone: one reproducible workbench running the full cycle with an evidence-first reporting layer** |

## 9 · Experimental design

**Baseline-first everywhere.** The forecasting study scores every model against seasonal
naive on identical origins (MASE is 1.00 for the benchmark by construction); the portfolio
study always includes equal weights. Improvements must be demonstrated, out of sample,
against the boring option.

**No look-ahead, by construction.** At each forecast origin the training pool holds only
rows whose 12-month-ahead target was already observed (dated ≤ origin − 12 months); at each
backtest fold the training window ends exactly when the holding period begins
(`hold_start == train_end`, asserted in tests).

**Direction-aware significance.** The DM test is two-sided but the verdict logic is
direction-aware: a significant statistic is only a "win" when the model's loss is *lower*.
This repository inherits the correction lesson from sibling repo 03 (where a significant
test was initially misread in the wrong direction).

**Multiple-testing caution.** Two DM comparisons (OLS, ridge) share one benchmark; marginal
p-values are read with a Holm/Bonferroni-style caution and never presented as standalone
discoveries.

**Costs are explicit.** 10 bps per side on turnover — a full one-way rebalance costs
10 bps, a full switch 20 bps; the entry charge from cash is full turnover. Turnover is
computed against drift-adjusted weights, not last period's target.

**Idempotent stages.** Each CLI stage writes its fragment and can be re-run alone; `all`
runs ingest → analyze → forecast → optimize → risk → report in one pass (6.3 s wall on
cached data; the first network fetch adds seconds, then everything is cached).

## 10 · Models

| Module | Model / scheme | Configuration |
| --- | --- | --- |
| Forecast | Seasonal naive | last observed annual inflation at the origin (benchmark) |
| Forecast | OLS | intercept + 3 features (inflation, unemployment, 10-y yield), `np.linalg.lstsq` |
| Forecast | Ridge | standardized features, α = 1.0, closed-form `cholesky` solver |
| Portfolio | Equal weight | 1/7 each |
| Portfolio | Inverse volatility | w ∝ 1/σ on trailing 12-month window, normalized |
| Portfolio | Min variance | SLSQP, objective ×1e3, long-only, cap 40%, ftol 1e-12 |
| Portfolio | Max Sharpe | closed-form tangency w ∝ Σ⁻¹μ; min-variance fallback (flagged + counted) |
| Risk | Historical VaR/ES | empirical quantile, linear interpolation; ES = mean of tail beyond VaR |
| Risk | Gaussian VaR/ES | closed form from sample μ, σ (z·σ − μ; σ·φ(z)/(1−α) − μ) |
| NLP | VADER | `vaderSentiment`, compound thresholds ±0.05 (documented convention) |

Seed = 42 (unused by the deterministic core; pinned for future stochastic extensions).

## 11 · Evaluation metrics

- **Diagnostics:** CAGR, annualized vol (252d), Sharpe (rf = 0 — simplification stated),
  max drawdown, skewness, excess kurtosis, JB p, ADF p (returns and log prices),
  Ljung-Box(20) p, rolling 63-day vol.
- **Forecasting:** RMSE and MAE (percentage points), MASE vs the naive benchmark on the
  same origins (Hyndman & Koehler 2006), Diebold-Mariano statistic and p-value with
  Newey-West HAC variance at lag h−1 = 11.
- **Portfolios:** OOS net Sharpe, CAGR, annualized vol, max DD, hit rate, average
  turnover per rebalance, cumulative cost drag, fallback share for max-Sharpe.
- **Risk:** historical and Gaussian VaR/ES at 95/99% (positive = loss), drawdown table
  with peak/trough/recovery dates.

## 12 · Results

Actual outputs of the committed run — `reports/metrics.json`, rendered in
[`reports/research_brief.md`](reports/research_brief.md). Run 2026-10-08, data window
2018-01-02 → 2026-10-08 (2,204 trading days).

**Market diagnostics (daily log returns):** every ticker rejects normality under
Jarque-Bera (7/7 at 5%; excess kurtosis 4.6–13.7). ADF: log returns stationary for all
seven (p ≤ 2.6e-18) while log prices are non-stationary (p 0.37–0.95). Ljung-Box(20):
significant autocorrelation in 6/7 series (AMZN p = 0.14 the exception). Sharpe (rf = 0):
AAPL 0.80 (CAGR 27.6%) > MSFT 0.75 > SPY 0.72 > JPM 0.54 > AMZN 0.49 > XOM 0.41 ≈ PG 0.42;
PG has the lowest vol (20.0%), XOM the deepest drawdown (−61.0%).

**Forecasting study — 12-month CPI inflation (35 origins, 2016-12-31 → 2025-06-30):**

| Model | RMSE (pp) | MASE | DM stat | DM p | Verdict |
| --- | --- | --- | --- | --- | --- |
| Seasonal naive | **2.51** | 1.00 | — | — | benchmark |
| OLS | 3.30 | 1.53 | +2.87 | 0.004 | significantly **WORSE** |
| Ridge | 3.28 | 1.52 | +2.81 | 0.005 | significantly **WORSE** |

**No model beats the naive benchmark.** Both regressions are significantly worse in the
direction-aware DM sense — the 2021–2022 inflation surge punished models anchored on
lagged levels. This is the honest headline, consistent with sibling repo 02's finding that
naive benchmarks are hard to beat at long horizons.

**Portfolios — walk-forward OOS, net of 10 bps/side (93 months, 2019-02-28 → 2026-10-31):**

| Scheme | Sharpe (net) | CAGR | Vol | Max DD | Avg turnover | Cost drag |
| --- | --- | --- | --- | --- | --- | --- |
| Min variance | **1.36** | 21.8% | 15.5% | −18.4% | 0.555 | 1.72% |
| Max Sharpe | **1.36** | 21.8% | 15.5% | −18.4% | 0.555 | 1.72% |
| Inverse vol | 1.24 | 19.9% | 15.7% | −21.3% | 0.152 | 0.47% |
| Equal weight | 1.23 | 20.6% | 16.5% | −18.4% | 0.100 | 0.31% |

The max-Sharpe scheme triggered its documented min-variance fallback in **31/31 folds**
(the unconstrained tangency violates the 40% cap or assigns negative weights on noisy
12-month estimates), so it coincides with min-variance — reported, not hidden. The
optimizer's margin over equal weights is 0.13 Sharpe: modest, and attributable to variance
reduction, not return forecasting.

**Risk (monthly OOS net returns, n = 93; positive = loss):**

| Scheme | VaR 95 | ES 95 | VaR 99 | ES 99 |
| --- | --- | --- | --- | --- |
| Equal | 7.9% | 9.2% | 9.7% | 9.8% |
| Inverse vol | 7.4% | 8.8% | 9.1% | 9.6% |
| Min variance (= max Sharpe) | 5.9% | 7.8% | 8.7% | 8.8% |

Daily SPY reference (n = 2,203): VaR95 1.76%, ES95 2.87%, VaR99 3.32%, ES99 4.93% — a far
better-estimated quantile than the monthly figures. All schemes peaked 2021-12-31 and
troughed 2022-09-30; min-variance recovered first (2022-11-30), inverse-vol last (2023-06-30).

**Sentiment screen (illustrative):** 10 bundled headlines scored with VADER — mean compound
+0.209, labels 6 positive / 4 neutral / 0 negative. No evidential claim; benchmark lives in
repo 08.

Figures: [`fig01_prices_normalized.png`](figures/fig01_prices_normalized.png),
[`fig02_rolling_vol.png`](figures/fig02_rolling_vol.png),
[`fig03_drawdowns.png`](figures/fig03_drawdowns.png),
[`fig04_inflation_backtest.png`](figures/fig04_inflation_backtest.png),
[`fig05_inflation_errors.png`](figures/fig05_inflation_errors.png),
[`fig06_weights.png`](figures/fig06_weights.png),
[`fig07_portfolio_oos.png`](figures/fig07_portfolio_oos.png),
[`fig08_var_es.png`](figures/fig08_var_es.png).

## 13 · Interpretation

1. **The naive benchmark won, and that is the finding.** With only three origin-known
   features and 24+ training pairs, OLS/ridge could not out-predict "inflation stays where
   it is" over 12 months — and the DM test says the gap is real, not noise (both models
   chase level shifts and miss the 2021–2022 surge's turning point). Any portfolio or
   policy narrative built on "our model forecasts inflation" would have been fabricated.
2. **The optimizer added variance reduction, not foresight.** Min-variance leads net Sharpe
   by 0.13 over equal weights with 3.4× the turnover — the entire edge is a lower vol
   (15.5% vs 16.5%) during the 2022 drawdown, where it also recovered two months earlier.
   The tangency scheme's 100% fallback rate is itself evidence of how unstable mean-variance
   optimization is on short windows (estimation error, DeMiguel et al. 2009).
3. **Costs are visible and material at high turnover.** Min-variance paid 1.72% cumulative
   (vs 0.31% for equal weight) — roughly a tenth of its CAGR advantage; cost assumptions
   belong in the table, not in a footnote.
4. **Monthly VaR/ES numbers are fragile by construction.** With 93 observations the 99%
   quantile is essentially the worst observed month; ES ≈ VaR at 99% because the tail
   beyond it contains ~1 observation. The daily SPY reference (n = 2,203) shows what a
   well-populated quantile looks like — the comparison is the lesson.
5. **The reporting layer is the deliverable.** Every claim above is generated from the
   stage fragments by `reporting/brief.py`; the "Findings & limitations" section is not
   hand-written prose that can drift from the numbers.

## 14 · Limitations

**Per module:**

- *Ingestion:* seven US-listed tickers (survivorship-prone universe by design — these firms
  were chosen ex post), adjusted-close abstraction, no dividend/tax microstructure; the
  final monthly return is a partial month (run date 2026-10-08 inside October).
- *Analytics:* rf = 0 simplifies Sharpe; 252-day annualization; ADF/JB/LB are
  as-yesterday diagnostics, not forecasts of regime change.
- *Forecast:* three features only; expanding-window OLS/ridge on modest pools; two DM
  comparisons read under a multiple-testing caution; n = 35 origins gives low power — and
  the study period contains an extraordinary inflation episode. **Nothing here forecasts
  future inflation.**
- *Portfolio:* 12-month covariance/mean estimates are noisy; SLSQP can stall on unscaled
  objectives (mitigated, documented); max-Sharpe is only defined through its fallback in
  this sample; one historical path, not a distribution; no rebalance-timing sensitivity.
- *Risk:* quantile estimates on 93 monthly points have first-order estimation error; VaR
  says nothing about losses beyond the quantile; Gaussian VaR understates fat tails (see
  fig08 — historical ≥ Gaussian in the tail, as expected).
- *Sentiment:* the bundled headlines are authored by the researcher for integration
  testing; zero evidential weight.

**Global:** single data window, single seed, no hyperparameter search, no alternative
cost/cap scenarios; all results are in-context descriptions of 2018–2026, not out-of-sample
claims about any future period. Finscope evaluates methods; it does not predict markets,
and it is not a trading system.

## 15 · Reproducibility

```bash
# 1) environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# 2) data (network on the first run only, then cached; datasets are NOT committed)
python scripts/run_workbench.py ingest        # Yahoo Finance + FRED -> data/raw/

# 3) full cycle (6.3 s on cached data; well under the 8-min budget)
python scripts/run_workbench.py all

# 4) verify: lint + 48 offline unit tests (no network)
ruff check .
pytest -q
```

Determinism: the pipeline core is deterministic (closed-form OLS/ridge/tangency, seeded
scikit-learn); repeated runs on the same cache produce identical numbers. Stages are
idempotent — rerun any stage alone (`python scripts/run_workbench.py forecast`).

## 16 · Installation

```bash
git clone https://github.com/aydinmonavvari/finscope-ai-research.git
cd finscope-ai-research
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## 17 · Usage

```bash
python scripts/run_workbench.py ingest     # fetch/cache prices + macro (once)
python scripts/run_workbench.py analyze    # diagnostics + figures 1-3
python scripts/run_workbench.py forecast   # inflation study + figures 4-5
python scripts/run_workbench.py optimize   # walk-forward portfolios + figures 6-7
python scripts/run_workbench.py risk       # VaR/ES + figure 8
python scripts/run_workbench.py report     # merge -> metrics.json + research_brief.md
python scripts/run_workbench.py all        # the whole cycle
```

As a library:

```python
from finscope.config import TICKERS
from finscope.data.prices import load_prices, monthly_simple_returns
from finscope.portfolio.construct import build_schemes, walk_forward_backtest

prices = load_prices(TICKERS)                      # cache-first, offline-safe
monthly = monthly_simple_returns(prices)
schemes = build_schemes()
run = walk_forward_backtest(monthly, schemes["min_var"])
print(run["n_folds"], run["avg_turnover"])         # 31 folds, ~0.555 avg turnover
```

## 18 · Example

The capstone's reporting discipline in one artifact: open
[`reports/research_brief.md`](reports/research_brief.md) and read the forecasting section.
It does not say "our model predicts inflation". It says — *from the measured numbers* —
that the seasonal-naive benchmark's RMSE is 2.51 pp, that OLS (3.30 pp) and ridge
(3.28 pp) are *significantly worse* under direction-aware Diebold-Mariano tests
(p = 0.004 / 0.005), and that with two comparisons and 35 origins no model earns a
multiple-testing-robust win. The same brief then shows the portfolio table net of costs,
the VaR/ES table with its estimation-risk warning, and a findings section whose every
bullet is assembled from the stage fragments. That is what "evidence-first" means here.

## 19 · Project structure

```
finscope-ai-research/
├── README.md · LICENSE · CITATION.cff · pyproject.toml · .gitignore
├── src/finscope/
│   ├── config.py              # single source of truth: paths, tickers, FRED ids, costs, seeds
│   ├── data/prices.py         # yfinance OHLCV, cache-first, monthly resampling
│   ├── data/macro.py          # FRED CSV loaders (CPIAUCSL, UNRATE, DGS10), cache-first
│   ├── analytics/stats.py     # returns, vol/Sharpe, drawdown, JB, ADF, Ljung-Box
│   ├── forecast/inflation.py  # rolling-origin study: naive/OLS/ridge, RMSE/MASE/DM
│   ├── portfolio/construct.py # 4 schemes, SLSQP + tangency fallback, walk-forward backtest
│   ├── risk/var.py            # hist+Gaussian VaR/ES, drawdown table, limitations note
│   ├── nlp/sentiment.py       # VADER utility on the bundled illustrative sample
│   └── reporting/             # brief.py (metrics.json + research brief), figures.py
├── scripts/run_workbench.py   # CLI: ingest|analyze|forecast|optimize|risk|report|all
├── tests/                     # 48 offline tests (math vs hand-computed, no network)
├── data/raw/                  # git-ignored caches (.gitkeep tracked)
├── data/samples/              # sample_headlines.csv — the ONLY tracked data file
├── reports/                   # metrics.json, research_brief.md, stage fragments (committed)
├── figures/                   # 8 generated PNGs (committed)
├── docs/research_report.md
└── .github/workflows/ci.yml
```

## 20 · Future work

- Confidence intervals on the OOS Sharpe gap (stationary block bootstrap) before any
  scheme preference is claimed.
- More forecast feature families (money supply, term spreads, survey expectations) — with
  the expectation, inherited from this run, that the naive benchmark stays hard to beat.
- Regime-conditional covariance estimation (EWMA/DCC) and turnover-penalized objectives.
- Monte-Carlo VaR with fat-tailed (Student-t) simulations and backtests of the VaR
  estimates themselves (Kupiec/Christoffersen).
- A second data window (e.g., 2005–2015) to show every headline number is window-conditional.
- Scenario walk-throughs linking the risk layer to the credit/EDF thinking of repo 05.

## 21 · Citation

If you use this work, please cite (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{monavvari2026finscope,
  author  = {Monavvari, Aydin},
  title   = {finscope-ai-research: a modular, reproducible financial-research workbench with an evidence-first reporting layer},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/aydinmonavvari/finscope-ai-research}
}
```

## 22 · License

MIT — see [`LICENSE`](LICENSE).

## 23 · Acknowledgments

Built with pandas, NumPy, scikit-learn, SciPy, statsmodels, matplotlib, yfinance and
vaderSentiment. Data courtesy of Yahoo Finance and the St. Louis Fed's FRED service
(public CSV endpoint). This is the capstone of a ten-repository undergraduate research
portfolio — see the map in §8 — and it borrows its discipline from each sibling: baseline
comparisons from 02/03, cost-aware backtesting from 04, evaluation honesty from 05/06,
and reporting restraint from 07/08. Not investment advice.
