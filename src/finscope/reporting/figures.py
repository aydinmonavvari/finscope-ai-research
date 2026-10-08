"""Matplotlib figure builders for the workbench (Agg backend, PNG output).

Each function takes in-memory data and saves one PNG under ``figures/``.
Import order note: ``matplotlib.use("Agg")`` must precede pyplot import
(E402 is per-file ignored for this reason).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from finscope.config import FIGURES_DIR

DPI = 150


def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"figures/{name}"


def plot_prices_normalized(prices: pd.DataFrame) -> str:
    """Growth of $1 across the universe (adjusted close, normalized)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    wealth = prices.dropna(how="all")
    wealth = wealth / wealth.iloc[0]
    for column in wealth.columns:
        ax.plot(wealth.index, wealth[column], linewidth=1.2, label=column)
    ax.set_title("Growth of $1 — adjusted close, normalized to first observation")
    ax.set_ylabel("Multiple of initial price")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, "fig01_prices_normalized.png")


def plot_rolling_vol(log_returns: pd.DataFrame, window: int = 63) -> str:
    """Rolling annualized volatility (63-day window) per ticker."""
    fig, ax = plt.subplots(figsize=(9, 5))
    rolling = log_returns.rolling(window).std(ddof=1) * np.sqrt(252)
    for column in rolling.columns:
        ax.plot(rolling.index, rolling[column], linewidth=1.0, label=column)
    ax.set_title(f"Rolling {window}-day annualized volatility (daily log returns)")
    ax.set_ylabel("Annualized volatility")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, "fig02_rolling_vol.png")


def plot_drawdowns(simple_returns: pd.DataFrame) -> str:
    """Drawdown series per ticker (daily simple returns)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    wealth = (1 + simple_returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    for column in dd.columns:
        ax.plot(dd.index, dd[column], linewidth=1.0, label=column)
    ax.set_title("Drawdowns from running peak (daily simple returns)")
    ax.set_ylabel("Drawdown")
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    return _save(fig, "fig03_drawdowns.png")


def plot_inflation_backtest(predictions: pd.DataFrame) -> str:
    """Actual vs modeled 12-month inflation at each rolling origin."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(predictions.index, predictions["actual"], color="black", linewidth=2.0, label="actual")
    for column, style in (
        ("seasonal_naive", "--"),
        ("ols", "-."),
        ("ridge", ":"),
    ):
        ax.plot(
            predictions.index, predictions[column], linestyle=style, linewidth=1.3, label=column
        )
    ax.set_title("12-month CPI inflation: rolling-origin backtest (%)")
    ax.set_ylabel("Annual CPI inflation (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, "fig04_inflation_backtest.png")


def plot_inflation_errors(predictions: pd.DataFrame) -> str:
    """Forecast errors by origin (positive = model under-predicted)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    actual = predictions["actual"]
    for column, style in (("seasonal_naive", "--"), ("ols", "-."), ("ridge", ":")):
        errors = predictions[column] - actual
        ax.plot(errors.index, errors, linestyle=style, linewidth=1.2, label=column)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Forecast errors by origin (forecast − actual, pp)")
    ax.set_ylabel("Error (percentage points)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, "fig05_inflation_errors.png")


def plot_weights(avg_weights: pd.DataFrame) -> str:
    """Average out-of-sample weights per scheme (grouped bars)."""
    schemes = list(avg_weights.index)
    assets = list(avg_weights.columns)
    x = np.arange(len(assets))
    width = 0.8 / max(len(schemes), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, scheme in enumerate(schemes):
        ax.bar(x + i * width, avg_weights.loc[scheme].to_numpy(), width, label=scheme)
    ax.axhline(1.0 / len(assets), color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + width * (len(schemes) - 1) / 2)
    ax.set_xticklabels(assets)
    ax.set_title("Average out-of-sample weights by scheme (cap 40%)")
    ax.set_ylabel("Average weight")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "fig06_weights.png")


def plot_portfolio_oos(net_returns: dict[str, pd.Series]) -> str:
    """Growth of $1 from OOS net monthly returns per scheme."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, series in net_returns.items():
        wealth = (1 + series.sort_index()).cumprod()
        ax.plot(wealth.index, wealth, linewidth=1.4, label=name.replace("_", " "))
    ax.set_title("Out-of-sample growth of $1 (net of 10 bps per side on turnover)")
    ax.set_ylabel("Wealth multiple")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, "fig07_portfolio_oos.png")


def plot_var_es(var_table: dict) -> str:
    """Historical vs Gaussian VaR/ES per scheme (95% and 99%)."""
    schemes = list(var_table.keys())
    metrics = ["var95", "es95", "var99", "es99"]
    hist = [[var_table[s].get(m, np.nan) for m in metrics] for s in schemes]
    gauss = [
        [var_table[s].get(m.replace("var", "gauss_var"), np.nan) for m in metrics]
        for s in schemes
    ]
    x = np.arange(len(schemes))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [row[i] for row in hist], width, label=f"hist {metric}")
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [row[i] for row in gauss], width, hatch="//", fill=False,
               edgecolor=f"C{i}", label=f"gauss {metric}", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in schemes])
    ax.set_title("VaR / ES at 95% and 99% — historical vs Gaussian (monthly, positive = loss)")
    ax.set_ylabel("Loss magnitude")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "fig08_var_es.png")
