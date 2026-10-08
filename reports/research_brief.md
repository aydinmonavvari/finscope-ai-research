# Finscope Research Brief

*Generated 2026-10-08T16:58:19+00:00 — end-to-end research cycle on public data (Yahoo Finance, FRED). Educational research output; finscope is a research workbench for evaluating methods on public data. educational research output; not investment advice, not a market prediction, and not a trading system.*

## 1. Scope and data

- Universe: SPY, AAPL, MSFT, JPM, XOM, PG, AMZN (7 tickers).
- Daily price window: 2018-01-02 → 2026-10-08 (2204 trading days; monthly returns used for portfolios).
- Sources: Yahoo Finance (adjusted OHLC, cached to `data/raw/`, never committed) and the FRED public CSV endpoint (CPIAUCSL, UNRATE, DGS10).
- Cost and method assumptions: 10 bps per side on turnover, rf = 0, 252 trading days, 12-month forecast horizon evaluated every 3 months.

## 2. Market diagnostics (daily log returns, 2018 → present)

| Ticker | CAGR | Ann. vol | Sharpe | Max DD | Ex. kurt | JB(5%) | ADF returns | LB(20) sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | 14.6% | 19.0% | 0.72 | -33.7% | 13.68 | reject | stationary | yes |
| AAPL | 27.6% | 30.4% | 0.80 | -38.5% | 6.10 | reject | stationary | yes |
| MSFT | 24.3% | 29.0% | 0.75 | -37.1% | 8.12 | reject | stationary | yes |
| JPM | 16.7% | 28.5% | 0.54 | -43.6% | 12.92 | reject | stationary | yes |
| XOM | 13.0% | 30.0% | 0.41 | -61.0% | 5.24 | reject | stationary | yes |
| PG | 8.7% | 20.0% | 0.42 | -23.8% | 9.39 | reject | stationary | yes |
| AMZN | 18.3% | 34.4% | 0.49 | -56.1% | 4.64 | reject | stationary | no |

Diagnostics are descriptive: heavy tails and non-normality motivate the use of drawdown- and quantile-based risk reporting below.

## 3. Forecasting study — 12-month CPI inflation (%) from CPIAUCSL

Design: rolling origins every 3 months (Tashman 2000), 35 evaluation origins (2016-12-31 → 2025-06-30), expanding training pool with no look-ahead.

| Model | RMSE (pp) | MASE | DM stat | DM p |
| --- | --- | --- | --- | --- |
| seasonal naive | 2.51 | 1.00 | — | — |
| ols | 3.30 | 1.53 | 2.87 | 0.004 |
| ridge | 3.28 | 1.52 | 2.81 | 0.005 |

**Verdict.** Seasonal-naive benchmark: RMSE 2.51 pp (MASE 1.00 by definition). OLS: RMSE 3.30 pp, MASE 1.53 (HIGHER error than naive), DM stat 2.87, p 0.004. RIDGE: RMSE 3.28 pp, MASE 1.52 (HIGHER error than naive), DM stat 2.81, p 0.005. No model delivers a multiple-testing-robust improvement over the naive benchmark; the honest conclusion is that simple benchmarks remain hard to beat at this horizon with this feature set.

Multiple-testing caution: two DM comparisons (OLS, ridge) against the same benchmark; marginal p-values are read with a Holm/Bonferroni-style caution and low power at n=35 origins is acknowledged

## 4. Portfolio construction (walk-forward, net of costs)

Design: 12-month training window → 3-month hold, rebalanced with 10.0 bps per side on turnover; 93 out-of-sample months (2019-02-28 → 2026-10-31).

| Scheme | Sharpe (net) | CAGR | Ann. vol | Max DD | Avg turnover | Total costs |
| --- | --- | --- | --- | --- | --- | --- |
| equal | 1.23 | 20.6% | 16.5% | -18.4% | 0.100 | 0.31% |
| inv vol | 1.24 | 19.9% | 15.7% | -21.3% | 0.152 | 0.47% |
| min var | 1.36 | 21.8% | 15.5% | -18.4% | 0.555 | 1.72% |
| max sharpe | 1.36 | 21.8% | 15.5% | -18.4% | 0.555 | 1.72% |

OOS Sharpe ranking (net of costs): min var > max sharpe > inv vol > equal.

Reading: the max-Sharpe scheme triggered its documented minimum-variance fallback in 100% of folds — with 12-month mean/covariance estimates the unconstrained tangency violates the per-asset cap or assigns negative weights — so it coincides with min-variance. When optimized schemes lead equal weights, margins are modest and the edge is variance reduction, not return forecasting (DeMiguel et al. 2009 estimation-error result).

## 5. Risk report — VaR/ES and drawdowns

Monthly OOS net returns, historical VaR/ES (positive = loss):

| Scheme | VaR 95 | ES 95 | VaR 99 | ES 99 |
| --- | --- | --- | --- | --- |
| equal | 7.9% | 9.2% | 9.7% | 9.7% |
| inv vol | 7.4% | 8.8% | 9.1% | 9.6% |
| min var | 5.9% | 7.8% | 8.7% | 8.8% |
| max sharpe | 5.9% | 7.8% | 8.7% | 8.8% |

Daily SPY reference (n=2203): hist VaR95 1.76%, ES95 2.87%, VaR99 3.32%, ES99 4.93% — a far better-estimated quantile than the monthly figures above.

| Scheme | Max DD | Peak | Trough | Recovered |
| --- | --- | --- | --- | --- |
| equal | -18.4% | 2021-12-31 | 2022-09-30 | 2023-04-30 |
| inv vol | -21.3% | 2021-12-31 | 2022-09-30 | 2023-06-30 |
| min var | -18.4% | 2021-12-31 | 2022-09-30 | 2022-11-30 |
| max sharpe | -18.4% | 2021-12-31 | 2022-09-30 | 2022-11-30 |

**Limitations.** VaR and ES are estimates, not guarantees. A VaR quantile says nothing about how bad losses beyond it can be; ES summarizes the tail but is itself an estimate with substantial sampling error; both are window- and distribution-dependent. With only ~92 monthly out-of-sample observations, the 99% historical quantile is close to the single worst observed month — estimation risk is first-order, and these figures should be read as descriptive statistics, not risk certificates.

## 6. Sentiment screen (illustrative bundled sample)

- 10 headlines scored with VADER (mean compound 0.209); labels: {'positive': 6, 'neutral': 4}.
- **Illustrative sample authored by the researcher; not a market dataset.** No market dataset is claimed here; the full
  sentiment benchmark lives in the sibling repository financial-nlp-sentiment.

## 7. Findings & limitations

**Findings (evidence-first, all from the actual run above):**

- Market diagnostics: daily returns of all 7 tickers reject normality under Jarque-Bera (7/7 at 5%); log returns are stationary while log prices are not — the standard risk-modelling caveat applies to shortcuts.
- Forecasting: no model delivers a multiple-testing-robust improvement over seasonal naive for 12-month inflation; the benchmark's RMSE is 2.51 pp across 35 origins.
- Portfolios: min var leads out-of-sample net Sharpe (1.36) across 93 months (max-Sharpe used its min-variance fallback in 100% of folds, so the two coincide); the margin over equal weights (1.23) is modest and consistent with estimation-error theory — variance reduction, not return forecasting.
- Risk: the equal scheme shows the deepest 99% monthly expected shortfall (9.7%); quantiles on ~93 observations carry first-order estimation risk.
- Sentiment: VADER scoring of the bundled illustrative sample runs end to end (no network, no dataset claims); the evidence-bearing benchmark is the financial-nlp-sentiment repository.

**Limitations (per module and global):**

- Data: seven US tickers and three FRED series; no survivorship-free universe, no transaction-cost microstructure, dividend/tax treatment simplified.
- Forecast: ~3 feature set, expanding-window OLS/ridge on modest training pools; DM comparisons are multiple and low-powered at n≈36 origins — nothing here is a forecast of future inflation.
- Portfolio: 12-month covariance estimates are noisy; max-Sharpe falls back to min-variance whenever the tangency solution violates caps; results are one historical path, not a distribution.
- Risk: quantile estimates on ~92 monthly observations carry first-order estimation risk; VaR/ES are descriptive, not guarantees.
- Sentiment: the bundled headlines are an illustrative authored sample, not evidence.
- Global: single data window, single seed, no hyperparameter search; every number is an in-context description of one historical period.

**Disclaimer.** Finscope is a research workbench for evaluating methods on public data. Educational research output; not investment advice, not a market prediction, and not a trading system.

