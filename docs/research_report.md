# Research Report — Finscope: An Evidence-First Financial-Research Workbench

**A modular capstone unifying ingestion, diagnostics, forecasting evaluation, portfolio
construction, risk reporting and honest summary across one reproducible pipeline**

*Author: Aydin Monavvari — educational research project (not investment advice, not a
trading system). Run date: 2026-10-08. All numbers are actual outputs of the committed
pipeline (`reports/metrics.json`).*

---

## Abstract

Finscope is the integrating capstone of a ten-repository research portfolio: a modular
workbench that executes a complete financial-research cycle on real public data — Yahoo
Finance daily adjusted closes for seven US large caps (2018-01-02 → 2026-10-08, 2,204
trading days) and three FRED series (CPIAUCSL, UNRATE, DGS10) — through cache-first
ingestion, diagnostic analytics, a rolling-origin forecasting study, cost-aware walk-forward
portfolio construction, VaR/ES risk reporting, and a research brief rendered directly from
the measured numbers. The forecasting study evaluates 12-month CPI inflation at 35 rolling
origins (2016-12-31 → 2025-06-30) against a seasonal-naive benchmark: the benchmark wins
(RMSE 2.51 pp vs 3.30/3.28 pp for OLS/ridge, MASE 1.53/1.52), with direction-aware
Diebold-Mariano tests labeling both regressions significantly *worse* (p = 0.004/0.005) —
an honestly reported negative result read under a multiple-testing caution. Out-of-sample
portfolio construction (93 months net of 10 bps per side) ranks minimum variance first
(Sharpe 1.36) ahead of inverse volatility (1.24) and equal weights (1.23), while the
closed-form max-Sharpe scheme triggers its documented min-variance fallback in 31 of 31
folds — direct evidence of estimation error in short-window mean–covariance inputs.
Historical monthly VaR/ES are reported with an explicit estimation-risk warning (n = 93
makes the 99% quantile nearly the worst observed month). The contribution is not any single
model but the unified orchestration, shared configuration, and a reporting layer whose
findings and limitations are generated from the same fragments that carry the numbers.

## Introduction

A research cycle fails at its seams more often than at its models. Data is re-downloaded
inconsistently between notebooks; one stage's assumptions contradict another's; conclusions
are summarized more boldly than their evidence supports. This project attacks those failure
modes directly: one configuration module defines every path, ticker, series, cost and seed
used anywhere in the pipeline; each stage writes a JSON fragment; a reporting layer merges
the fragments and renders the brief *from* them, so prose cannot drift from numbers; and
every model is compared against a defensible naive baseline, out of sample, with the
comparison test printed beside the metric. Finscope therefore demonstrates integration as a
skill in its own right: nine sibling repositories each study one method; this repository
shows the methods compose into a disciplined, auditable whole.

## Research Question

> Can a complete research cycle — diagnostics, forecasting, portfolio construction, risk
> measurement — be run as one reproducible pipeline in which every headline claim is an
> out-of-sample comparison against a naive benchmark, and the limitations are documented
> with the same rigor as the results?

## Related Work

- **Markowitz, H. (1952).** "Portfolio Selection." *Journal of Finance*, 7(1), 77–91.
  Mean–variance framing behind the optimization schemes (min-variance; tangency portfolio).
- **Sharpe, W. F. (1966).** "Mutual Fund Performance." *Journal of Business*, 39(1), 119–138.
  The reward-to-variability ratio used throughout as the net Sharpe ratio.
- **Fama, E. F. (1970).** "Efficient Capital Markets: A Review of Theory and Empirical
  Work." *Journal of Finance*, 25(2), 383–417. The reason baseline-first evaluation is the
  default posture: claims of outperformance must clear a high, boring bar.
- **Tashman, L. J. (2000).** "Out-of-sample tests of forecasting accuracy: an analysis and
  review." *International Journal of Forecasting*, 16(4), 437–450. Rolling-origin design
  (origins every 3 months) used for the forecasting study.
- **Diebold, F. X., & Mariano, R. S. (1995).** "Comparing Predictive Accuracy."
  *Journal of Business & Economic Statistics*, 13(3), 253–263. The loss-differential test
  with Newey-West HAC variance at lag h−1 used to compare models against seasonal naive.
- **Hyndman, R. J., & Koehler, A. B. (2006).** "Another look at measures of forecast
  accuracy." *International Journal of Forecasting*, 22(4), 679–688. MASE scaling logic.
- **Dickey, D. A., & Fuller, W. A. (1979).** "Distribution of the Estimators for
  Autoregressive Time Series with a Unit Root." *JASA*, 74(366), 427–431. ADF stationarity
  diagnostics on returns and log prices.
- **Jarque, C. M., & Bera, A. K. (1980).** "Efficient tests for normality, homoscedasticity
  and serial independence of regression residuals." *Economics Letters*, 6(3), 255–259.
  Normality diagnostics motivating quantile-based risk reporting.
- **Ljung, G. M., & Box, G. E. P. (1978).** "On a measure of lack of fit in time series
  models." *Biometrika*, 65(2), 297–303. Serial-dependence diagnostic on daily returns.
- **Hutto, C. J., & Gilbert, E. (2014).** "VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text." *ICWSM-8*. The sentiment utility's scorer.
- **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** "Optimal Versus Naive Diversification:
  How Inefficient is the 1/N Portfolio Strategy?" *Review of Financial Studies*, 22(5),
  1915–1953. The estimation-error lens through which the portfolio results are read.
- **Breiman, L. (2001).** "Statistical Modeling: The Two Cultures." *Statistical Science*,
  16(3), 199–231. The data-modeling posture behind evaluating methods rather than asserting
  mechanisms.

## Data

| Property | Value |
| --- | --- |
| Prices | SPY, AAPL, MSFT, JPM, XOM, PG, AMZN — daily adjusted close (yfinance) |
| Price window | 2018-01-02 → 2026-10-08: 2,204 trading days; 2,203 daily return observations |
| Monthly panel | 104 month-end observations; OOS backtest window 2019-02-28 → 2026-10-31 (93 months) |
| Macro | CPIAUCSL (CPI index), UNRATE (unemployment), DGS10 (10-y yield → monthly mean) |
| Macro window | 2013-01 → 2026-10 (166 months, giving lagged features depth) |
| Caching | `data/raw/` (git-ignored, never redistributed); cache-first, offline-safe reads |
| Bundled sample | 10 researcher-authored headlines (tracked in git; labeled illustrative) |

The final monthly observation (labelled 2026-10-31) is a partial month — October 2026
contributes six trading days at the run date. This is disclosed rather than dropped,
because dropping the current month would silently condition results on the window's end.

## Methodology

**Configuration as governance.** `config.py` defines tickers, dates, FRED series, the
10 bps-per-side cost assumption, the 40% per-asset cap, horizons and seeds. Stages read
config; nothing hard-codes a number twice.

**Forecasting study.** Target: y/y CPI inflation. Origins every 3 months (Tashman 2000);
horizon h = 12. At origin *t*, the training pool contains only rows dated ≤ t − 12 months
whose targets were already observed — no look-ahead by construction. Models: seasonal
naive (last observed annual inflation — the correct seasonal-naive forecast for an annual
target), OLS and standardized ridge (α = 1) on three origin-known features. Metrics: RMSE,
MAE, MASE (Hyndman & Koehler 2006), and the Diebold-Mariano test on squared-error loss
differentials with HAC variance at lag h−1 = 11. The verdict logic is direction-aware: a
significant DM statistic only counts as a win when the model's loss is lower; otherwise it
is reported as significantly worse.

**Portfolio construction.** Four schemes on monthly returns: equal weight; inverse
volatility; long-only capped minimum variance via SLSQP (annualized-variance objective
scaled ×1e3 to fix the tiny-gradient stall; cap 40% enforced as smooth box bounds); and
max-Sharpe via the closed-form tangency w ∝ Σ⁻¹μ with a documented fallback to capped
min-variance whenever the tangency solution is infeasible (singular covariance,
non-positive excess return, or cap/negativity violation). Fallbacks are counted per fold
and reported. Walk-forward backtest: 12-month training window → 3-month hold, folds
contiguous and non-overlapping (`hold_start == train_end`); 10 bps per side charged on
turnover against drift-adjusted weights; entry from cash counts as full turnover.

**Risk reporting.** Historical VaR/ES (empirical quantile with linear interpolation; ES =
mean of returns at or beyond VaR) and closed-form Gaussian VaR/ES at 95/99%, on OOS net
monthly returns and on a daily SPY reference series; drawdown table with peak, trough and
recovery dates; a fixed limitations paragraph shipped with the numbers.

**Sentiment utility.** VADER (Hutto & Gilbert 2014) over the ten bundled headlines, with
VADER's documented ±0.05 compound thresholds. The sample is labeled as researcher-authored;
the evidence-bearing benchmark lives in the `financial-nlp-sentiment` repository.

## Experimental Design

Every evaluation is out-of-sample and baseline-referenced. The forecasting study scores
identical origins across models; the portfolio backtest always includes equal weights.
Two DM comparisons share one benchmark, so marginal p-values are read under an explicit
Holm/Bonferroni-style caution with low power at n = 35 acknowledged. Cost sensitivity is
structural, not incidental: turnover and cumulative cost drag are reported per scheme.
Determinism: closed-form OLS/ridge/tangency and seeded scikit-learn make repeated runs on
the same cache numerically identical. The offline test suite (48 tests) covers the math
against hand-computed values — log returns, Sharpe, drawdown, MASE, DM lag-0 equivalence
to a t-test, SLSQP cap behavior, tangency fallback paths, rebalance-cost arithmetic,
walk-forward non-overlap, VaR/ES monotonicity and z-value correctness, VADER label
boundaries, and brief rendering invariants (limitation marker present; no absolute paths
in outputs).

## Results

Actual outputs (`reports/metrics.json`).

**Diagnostics.** Jarque-Bera rejects normality for all seven tickers (excess kurtosis
4.6–13.7). ADF: daily log returns stationary for all (p ≤ 2.6e-18); log prices
non-stationary (p 0.37–0.95). Ljung-Box(20): significant serial dependence in 6/7 series
(AMZN p = 0.14). Net-of-nothing Sharpe (rf = 0): AAPL 0.80 (CAGR 27.6%), MSFT 0.75,
SPY 0.72, JPM 0.54, AMZN 0.49, PG 0.42, XOM 0.41; XOM's max drawdown −61.0% is the
deepest, PG's vol (20.0%) the lowest.

**Forecasting (12-month CPI inflation, 35 origins 2016-12-31 → 2025-06-30).**

| Model | RMSE (pp) | MASE | DM stat | DM p |
| --- | --- | --- | --- | --- |
| Seasonal naive | **2.51** | 1.00 | — | — |
| OLS | 3.30 | 1.53 | +2.87 | 0.004 |
| Ridge | 3.28 | 1.52 | +2.81 | 0.005 |

Both regressions are significantly **worse** than the naive benchmark in the
direction-aware sense. No multiple-testing-robust improvement exists. The failures
concentrate in the 2021–2022 inflation surge, where lagged-level models undershot the
turning point by wide margins (fig05).

**Portfolios (93 OOS months, net of 10 bps per side).**

| Scheme | Sharpe (net) | CAGR | Vol | Max DD | Avg turnover | Cost drag |
| --- | --- | --- | --- | --- | --- | --- |
| Min variance | **1.36** | 21.8% | 15.5% | −18.4% | 0.555 | 1.72% |
| Max Sharpe | **1.36** | 21.8% | 15.5% | −18.4% | 0.555 | 1.72% |
| Inverse vol | 1.24 | 19.9% | 15.7% | −21.3% | 0.152 | 0.47% |
| Equal weight | 1.23 | 20.6% | 16.5% | −18.4% | 0.100 | 0.31% |

Max-Sharpe used its min-variance fallback in **31/31 folds** (the unconstrained tangency
violates the 40% cap or assigns negative weights on 12-month estimates), so its row
coincides with min-variance. Average min-variance weights: PG 28.3%, MSFT 20.0%, SPY 16.6%,
XOM 15.3%, JPM 7.9%, AAPL 7.4%, AMZN 4.7%. All schemes peaked 2021-12-31 and troughed
2022-09-30; min-variance recovered first (2022-11-30), inverse-vol last (2023-06-30).

**Risk (monthly OOS net returns, n = 93; positive = loss).**

| Scheme | VaR 95 | ES 95 | VaR 99 | ES 99 |
| --- | --- | --- | --- | --- |
| Equal | 7.9% | 9.2% | 9.7% | 9.8% |
| Inverse vol | 7.4% | 8.8% | 9.1% | 9.6% |
| Min variance (= max Sharpe) | 5.9% | 7.8% | 8.7% | 8.8% |

Daily SPY reference (n = 2,203): VaR95 1.76%, ES95 2.87%, VaR99 3.32%, ES99 4.93%.
Historical tail magnitudes exceed Gaussian estimates at the same levels, as fat tails
predict (fig08).

## Discussion

1. **The negative forecasting result is the study's most useful output.** With three
   origin-known features and modest training pools, nothing beats "inflation stays where
   it is" at a 12-month horizon — and the DM tests indicate the regressions were reliably
   *worse*, chasing level shifts through the 2021–2022 surge. Under the efficient-markets
   posture (Fama 1970) and the two-cultures humility of Breiman (2001), the correct
   response is to report the benchmark's win, not to search harder for a specification
   that flatters the method.
2. **The optimizer's edge is variance reduction, not foresight.** Min-variance's 0.13
   net-Sharpe margin over equal weights comes entirely from lower realized volatility
   (15.5% vs 16.5%) through the 2022 drawdown — and it paid 1.72% cumulative costs
   (5.5× equal weight's 0.31%) to get there. The 100% tangency fallback rate is the
   DeMiguel et al. (2009) estimation-error result made visible in a single number: on
   12-month windows, Σ⁻¹μ routinely assigns negative or cap-violating weights.
3. **Monthly risk quantiles are descriptive, not protective.** With 93 observations, the
   99% historical quantile is essentially the worst observed month, and ES ≈ VaR at 99%
   because the tail beyond contains roughly one observation. The daily SPY series shows
   the contrast a well-populated quantile makes. Both are reported with a fixed
   limitations paragraph so the numbers cannot be quoted without the caveat.
4. **Diagnostics shaped the risk layer.** The universal JB rejections and the ADF
   contrast (returns stationary, prices not) are why risk is reported with drawdowns and
   quantiles rather than Gaussian shortcuts alone — and why the Gaussian VaR row is shown
   next to the historical one, underestimating the tail by construction.
5. **The reporting layer closed the integrity loop.** Because the brief is rendered from
   the stage fragments, the direction-aware verdict logic ("significantly worse") is code,
   not judgment — the same correction applied in sibling repo 03, where a significant
   test had been misread in the wrong direction.

## Limitations

- **Universe selection.** Seven current US large caps chosen ex post are a
  survivorship-prone universe; no claim of investability is made or should be inferred.
- **Window conditionality.** All results describe one path through 2018–2026, a period
  containing a unique inflation episode and a rate-hiking cycle; a second window would
  change every number. The final monthly return is a partial month (run-date artifact).
- **Simplifications.** rf = 0 for Sharpe; 252-day annualization; adjusted-close
  abstraction; linear cost model (no market impact, no slippage beyond 10 bps per side).
- **Forecasting scope.** Three features; two model families; n = 35 origins gives low
  power; the DM comparisons are multiple and are read under a caution. Nothing here
  forecasts future inflation.
- **Optimization scope.** One cap (40%), one training length (12 months), one cost level;
  SLSQP conditioning mitigated by objective scaling rather than reparameterization;
  max-Sharpe is defined here only through its fallback.
- **Risk estimation.** Quantile estimates on ~93 monthly points carry first-order sampling
  error; no VaR backtest (Kupiec/Christoffersen) is included; Gaussian VaR understates
  fat tails by construction.
- **Sentiment.** The bundled headlines are authored for integration testing and carry no
  evidential weight.

## Conclusion

Finscope demonstrates that a complete research cycle can be executed as one reproducible,
baseline-referenced pipeline whose reporting cannot overstate its evidence: the forecasting
study ends with a seasonal-naive win (RMSE 2.51 pp) and significantly worse regressions;
the portfolio layer shows a modest, cost-aware optimizer edge (net Sharpe 1.36 vs 1.23 for
equal weights) earned through variance reduction with a fully-triggered tangency fallback;
the risk layer reports fat-tailed quantiles with their fragility attached. Every headline
number is generated into the brief by code, every assumption lives in one config module,
and every limitation ships beside the result it qualifies.

## Future Research

- Block-bootstrap confidence intervals on the net-Sharpe gaps between schemes.
- Richer forecast feature families (money aggregates, term spreads, survey expectations),
  with the prior — informed by this run and by sibling repo 02 — that naive benchmarks
  remain hard to beat at long horizons.
- EWMA/DCC-style conditional covariance estimation and turnover-penalized objectives.
- Student-t and Monte-Carlo VaR with formal backtests (Kupiec POF, Christoffersen
  independence).
- A second, non-overlapping data window to demonstrate the window-conditionality of every
  headline number.

## Portfolio Integration

Finscope is the tenth and final repository of the portfolio, and its integration role is
explicit. From **finance-data-analysis-lab** (01) it inherits the diagnostic layer — the
same JB/ADF/Ljung-Box discipline, here fed by a cache-first loader. From
**macro-forecasting-lab** (02) it inherits the rolling-origin machinery and the finding
that naive benchmarks dominate long-horizon macro forecasting — replicated here on CPI
inflation with a different, smaller model set. From **ml-market-prediction-study** (03) it
inherits the two integrity rules: direction-aware significance verdicts and the refusal to
present a significant statistic as a win in the wrong direction. From
**portfolio-optimization-lab** (04) it inherits the SLSQP conditioning fixes, the smooth
cap handling, and the closed-form tangency with a counted, documented fallback. From
**credit-risk-modeling** (05) and **fraud-anomaly-detection** (06) it inherits the
reporting ethic: metrics beside their caveats, thresholds beside their costs. From
**dl-financial-time-series** (07) and **financial-nlp-sentiment** (08) it inherits the
capability ladder framing — simple baselines first, complex models only where they earn
their complexity — which is why the sentiment layer here is a labeled utility pointing to
repo 08's benchmark rather than a redundant study. From **fin-rag-research-assistant** (09)
it inherits the sourcing discipline (documented provenance, no unattributed claims). What
the capstone adds is the connective tissue: one config, one CLI, stage fragments that merge
into a single brief, and the demonstration that the portfolio's methods survive being run
together — including the moments where they honestly lose to a naive benchmark.

## References

- Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science*, 16(3), 199–231.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915–1953.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427–431.
- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383–417.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media (ICWSM-8)*.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. *Economics Letters*, 6(3), 255–259.
- Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297–303.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Sharpe, W. F. (1966). Mutual fund performance. *Journal of Business*, 39(1), 119–138.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437–450.
