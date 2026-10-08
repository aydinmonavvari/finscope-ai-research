"""Value-at-Risk, expected shortfall and drawdown reporting.

Conventions: VaR/ES are reported as POSITIVE loss magnitudes (e.g. 0.08 =
"8% loss"). Historical VaR is the empirical quantile of returns with linear
interpolation; Gaussian VaR/ES use the sample mean and standard deviation.

Honest limitations (see ``var_limitations``/``VAR_LIMITATIONS``): VaR is a
quantile estimate, not a tail guarantee — it says nothing about the size of
losses beyond the quantile (ES summarizes them but is itself an estimate),
both are highly sensitive to the sample and window, and small samples (like
the 93 monthly OOS observations in the committed run) make the 99% quantile
essentially the worst observed month. None of this is a substitute for
judgment or stress testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from finscope.config import VAR_LEVELS

# Measured, not assumed: the committed run's out-of-sample net-return series
# has 93 monthly observations (reports/risk.json, n_months). The limitation
# note is rendered from the MEASURED count via ``var_limitations(n)``; the
# constant below keeps the default text accurate when no count is supplied.
MEASURED_N_MONTHLY_OBS = 93


def var_limitations(n_months: int | None = MEASURED_N_MONTHLY_OBS) -> str:
    """Limitations note with the measured monthly sample size baked in."""
    sample = (
        f"With only {n_months} monthly out-of-sample observations, "
        if n_months is not None
        else "With small monthly out-of-sample samples, "
    )
    return (
        "VaR and ES are estimates, not guarantees. A VaR quantile says nothing about "
        "how bad losses beyond it can be; ES summarizes the tail but is itself an "
        "estimate with substantial sampling error; both are window- and "
        "distribution-dependent. "
        + sample
        + (
            "the 99% historical quantile is close to the single worst observed month — "
            "estimation risk is first-order, and these figures should be read as "
            "descriptive statistics, not risk certificates."
        )
    )


VAR_LIMITATIONS = var_limitations()  # default text uses the measured n = 93


def historical_var_es(returns, level: float = 0.95) -> dict:
    """Historical (empirical) VaR and ES as positive loss magnitudes."""
    values = np.sort(np.asarray(returns, dtype=float))
    values = values[~np.isnan(values)]
    var = float(-np.quantile(values, 1.0 - level))
    tail = values[values <= -var]
    es = float(-tail.mean()) if tail.size else var
    return {"var": var, "es": es, "level": float(level), "n": int(values.size)}


def gaussian_var_es(returns, level: float = 0.95) -> dict:
    """Parametric Gaussian VaR and ES (closed form) as positive loss magnitudes."""
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    z = stats.norm.ppf(level)  # positive z at level, e.g. 1.645 at 95%
    var = float(z * sigma - mu)
    es = float(sigma * stats.norm.pdf(z) / (1.0 - level) - mu)
    return {"var": var, "es": es, "level": float(level), "n": int(values.size)}


def var_es_table(returns, levels=VAR_LEVELS) -> dict:
    """Historical + Gaussian VaR/ES at each requested level."""
    table: dict[str, dict] = {}
    for level in levels:
        hist = historical_var_es(returns, level)
        gauss = gaussian_var_es(returns, level)
        table[f"{int(level * 100)}"] = {
            "hist_var": hist["var"],
            "hist_es": hist["es"],
            "gauss_var": gauss["var"],
            "gauss_es": gauss["es"],
        }
    return table


def drawdown_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-column drawdown facts: depth, peak date, trough date, recovery date."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    rows = {}
    for column in returns.columns:
        series_wealth = wealth[column].dropna()
        series_dd = dd[column].dropna()
        trough_date = series_dd.idxmin()
        max_dd = float(series_dd.min())
        peak_date = series_wealth.loc[:trough_date].idxmax()
        after = series_dd.loc[trough_date:]
        recovered = after[after >= -1e-12]
        recovery_date = recovered.index.min() if len(recovered) else None
        rows[column] = {
            "max_dd": max_dd,
            "peak_date": str(peak_date.date()),
            "trough_date": str(trough_date.date()),
            "recovery_date": (
                str(recovery_date.date()) if recovery_date is not None else "not recovered"
            ),
        }
    return pd.DataFrame(rows).T
