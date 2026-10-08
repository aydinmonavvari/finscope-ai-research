"""Return statistics and diagnostic tests with explicit, testable definitions.

Pure functions on pandas/numpy objects — no filesystem, no network. All
annualization uses 252 trading days and the risk-free assumption from
``config`` (0%, documented in the README limitations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from finscope.config import RISK_FREE_ANNUAL, ROLLING_VOL_WINDOW, TRADING_DAYS


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log returns of a price series/frame, with leading NaN dropped."""
    rets = np.log(prices / prices.shift(1))
    return rets.dropna(how="all") if isinstance(prices, pd.DataFrame) else rets.dropna()


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple (arithmetic) returns — exact for portfolio arithmetic ``w @ r``."""
    rets = prices.pct_change()
    return rets.dropna(how="all") if isinstance(prices, pd.DataFrame) else rets.dropna()


def annualized_vol(returns, periods: int = TRADING_DAYS) -> float:
    """Sample standard deviation of per-period returns, scaled by sqrt(periods)."""
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.std(values, ddof=1) * np.sqrt(periods))


def annualized_sharpe(
    returns,
    rf_annual: float = RISK_FREE_ANNUAL,
    periods: int = TRADING_DAYS,
) -> float:
    """Mean excess return over standard deviation, annualized (rf = 0 by default)."""
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    excess = values - rf_annual / periods
    sd = np.std(excess, ddof=1)
    if sd <= 1e-12:  # constant (or float-noise-constant) series have no Sharpe
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods))


def drawdown_series(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Drawdown of a simple-return series: wealth / running peak - 1 (<= 0)."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns) -> float:
    """Most negative point of :func:`drawdown_series` (a negative fraction)."""
    dd = drawdown_series(pd.Series(np.asarray(returns, dtype=float)))
    return float(dd.min())


def jarque_bera_test(returns) -> dict:
    """Jarque-Bera normality test with skewness and excess kurtosis (scipy)."""
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    result = stats.jarque_bera(values)
    return {
        "stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "skew": float(stats.skew(values)),
        "excess_kurtosis": float(stats.kurtosis(values, fisher=True)),
    }


def adf_test(series, regression: str = "c") -> dict:
    """Augmented Dickey-Fuller stationarity test (statsmodels, AIC lag choice)."""
    values = pd.Series(series).dropna()
    result = adfuller(values, regression=regression, autolag="AIC")
    return {
        "stat": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": int(result[2]),
        "nobs": int(result[3]),
    }


def ljung_box_test(returns, lags: int = 20) -> dict:
    """Ljung-Box autocorrelation test on returns (statsmodels)."""
    values = pd.Series(returns).dropna()
    result = acorr_ljungbox(values, lags=[lags], return_df=True)
    return {
        "stat": float(result["lb_stat"].iloc[0]),
        "p_value": float(result["lb_pvalue"].iloc[0]),
        "lags": int(lags),
    }


def rolling_vol(
    returns: pd.Series | pd.DataFrame,
    window: int = ROLLING_VOL_WINDOW,
    periods: int = TRADING_DAYS,
) -> pd.Series | pd.DataFrame:
    """Rolling annualized volatility (63 trading days by default)."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods)


def cagr(prices: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by a price series."""
    values = prices.dropna()
    n = len(values) - 1
    if n <= 0:
        return 0.0
    years = n / periods_per_year
    return float((values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0)


def summary_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset diagnostics computed from daily prices (real outputs only)."""
    rows: dict[str, dict] = {}
    for column in prices.columns:
        series = prices[column].dropna()
        lr = log_returns(series)
        sr = simple_returns(series)
        rows[column] = {
            "obs": int(len(lr)),
            "cagr": cagr(series),
            "ann_vol": annualized_vol(lr),
            "sharpe": annualized_sharpe(lr),
            "max_dd": max_drawdown(sr),
            "skew": float(stats.skew(lr)),
            "excess_kurtosis": float(stats.kurtosis(lr, fisher=True)),
            "jb_p": jarque_bera_test(lr)["p_value"],
            "adf_returns_p": adf_test(lr)["p_value"],
            "adf_logprice_p": adf_test(np.log(series))["p_value"],
            "ljungbox_p": ljung_box_test(lr)["p_value"],
        }
    return pd.DataFrame(rows).T
