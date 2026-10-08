"""Portfolio construction and a walk-forward backtest (research, not advice).

Schemes: equal weight, inverse volatility, minimum variance (scipy SLSQP with
the annualized-variance objective scaled by 1e3 — SLSQP stalls on the tiny
gradients of unscaled variance objectives — and per-asset caps enforced as
smooth box bounds) and a closed-form tangency portfolio with a documented
fallback to minimum variance whenever the tangency solution is infeasible
(zero/negative excess return or a per-asset cap violation).

Every scheme with a documented fallback path has that fallback COUNTED on its
``Scheme.fallbacks`` field and reported as ``fallback_share``: max-Sharpe
(tangency infeasible → capped min-variance), min-variance (non-finite
covariance or SLSQP failure → equal weights) and inverse-volatility
(degenerate trailing vol → equal weights). Equal weights has no fallback path
by construction.

Backtest: rolling 12-month training window → 3-month holding period. Costs of
10 bps per side are applied on turnover, where the sum of absolute weight
changes counts buys and sells separately (a full one-way rebalance costs
10 bps, a full switch costs 20 bps).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from finscope.config import ASSET_CAP, COST_BPS_PER_SIDE, HOLD_MONTHS, TRAIN_MONTHS


# ---------------------------------------------------------------- weight rules
def equal_weights(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def inverse_vol_weights_flagged(returns: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """Inverse-vol weights plus a fallback flag (degenerate vol → equal weights)."""
    vol = returns.std(ddof=1).to_numpy(dtype=float)
    vol = np.where(vol > 1e-12, vol, np.nan)  # degenerate/tiny vol cannot be trusted
    inv = 1.0 / vol
    if not np.all(np.isfinite(inv)):
        return equal_weights(returns.shape[1]), True
    return inv / inv.sum(), False


def inverse_vol_weights(returns: pd.DataFrame) -> np.ndarray:
    """Weight proportional to 1 / trailing volatility (long-only, normalized)."""
    weights, _ = inverse_vol_weights_flagged(returns)
    return weights


def min_variance_weights_flagged(
    cov: np.ndarray,
    cap: float = ASSET_CAP,
    allow_fallback: bool = True,
    maxiter: int = 1000,
) -> tuple[np.ndarray, bool]:
    """Long-only capped minimum-variance weights plus a fallback flag.

    Returns ``(weights, used_fallback)``. Falls back to equal weights
    (documented) when the covariance is not usable or the optimizer fails;
    raise instead with ``allow_fallback=False``.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cap * n < 1.0:
        raise ValueError(
            f"cap {cap} with {n} assets makes the simplex infeasible (cap * n < 1)"
        )
    if not np.all(np.isfinite(cov)):
        if allow_fallback:
            return equal_weights(n), True
        raise ValueError("covariance contains non-finite values")
    objective = lambda w: 1e3 * float(w @ cov @ w)  # noqa: E731 - scaled for conditioning
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, cap)] * n
    result = minimize(
        objective,
        equal_weights(n),  # feasible start whenever cap * n >= 1
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    weights_ok = result.success and np.all(np.isfinite(result.x))
    if weights_ok:
        candidate = np.asarray(result.x, dtype=float)
        weights_ok = (
            abs(candidate.sum() - 1.0) < 1e-6
            and np.all(candidate >= -1e-9)
            and np.all(candidate <= cap + 1e-6)
        )
    if weights_ok:
        return np.asarray(result.x, dtype=float), False
    if allow_fallback:
        return equal_weights(n), True
    raise RuntimeError(f"SLSQP did not converge to a feasible solution: {result.message}")


def min_variance_weights(
    cov: np.ndarray,
    cap: float = ASSET_CAP,
    allow_fallback: bool = True,
    maxiter: int = 1000,
) -> np.ndarray:
    """Long-only capped minimum-variance weights via SLSQP (objective x 1e3).

    Falls back to equal weights (documented) when the optimizer fails or the
    covariance is not usable; raise instead with ``allow_fallback=False``.
    The flagged variant ``min_variance_weights_flagged`` reports whether the
    fallback was used, so schemes can count it.
    """
    weights, _ = min_variance_weights_flagged(
        cov, cap=cap, allow_fallback=allow_fallback, maxiter=maxiter
    )
    return weights


def tangency_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    cap: float = ASSET_CAP,
    rf: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """Closed-form tangency portfolio with documented min-variance fallback.

    Returns ``(weights, used_fallback)``. The unconstrained tangency solution
    ``w ∝ Σ⁻¹ (μ - rf)`` requires a positive total excess return and respects
    the per-asset cap; otherwise the capped minimum-variance portfolio is
    returned and the fallback is flagged so it can be reported honestly.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    excess = mu - rf
    try:
        z = np.linalg.solve(cov, excess)
    except np.linalg.LinAlgError:
        return min_variance_weights(cov, cap=cap), True
    if not np.all(np.isfinite(z)) or z.sum() <= 0:
        return min_variance_weights(cov, cap=cap), True
    weights = z / z.sum()
    if np.any(weights < 0) or np.any(weights > cap):
        return min_variance_weights(cov, cap=cap), True
    return weights, False


@dataclass
class Scheme:
    """A named weighting rule; ``fallbacks`` counts documented fallback uses."""

    name: str
    fn: Callable[[pd.DataFrame], np.ndarray]
    fallbacks: int = field(default=0)


def build_schemes(cap: float = ASSET_CAP) -> dict[str, Scheme]:
    """The four study schemes over a trailing monthly-return window.

    Each ``Scheme`` instance is the single source of truth for its own
    fallback count: the closures increment the dataclass field, which the
    reporting stage reads and reports as ``fallback_share``.
    """

    def equal_fn(window: pd.DataFrame) -> np.ndarray:
        return equal_weights(window.shape[1])

    inv_vol_scheme = Scheme("inv_vol", equal_fn)  # fn replaced below

    def inv_vol_fn(window: pd.DataFrame) -> np.ndarray:
        weights, fell_back = inverse_vol_weights_flagged(window)
        if fell_back:
            inv_vol_scheme.fallbacks += 1
        return weights

    inv_vol_scheme.fn = inv_vol_fn

    min_var_scheme = Scheme("min_var", equal_fn)  # fn replaced below

    def min_var_fn(window: pd.DataFrame) -> np.ndarray:
        weights, fell_back = min_variance_weights_flagged(window.cov().to_numpy(), cap=cap)
        if fell_back:
            min_var_scheme.fallbacks += 1
        return weights

    min_var_scheme.fn = min_var_fn

    max_sharpe_scheme = Scheme("max_sharpe", equal_fn)  # fn replaced below

    def max_sharpe_fn(window: pd.DataFrame) -> np.ndarray:
        weights, fell_back = tangency_weights(
            window.mean().to_numpy(), window.cov().to_numpy(), cap=cap, rf=0.0
        )
        if fell_back:
            max_sharpe_scheme.fallbacks += 1
        return weights

    max_sharpe_scheme.fn = max_sharpe_fn
    return {
        "equal": Scheme("equal", equal_fn),
        "inv_vol": inv_vol_scheme,
        "min_var": min_var_scheme,
        "max_sharpe": max_sharpe_scheme,
    }


# ------------------------------------------------------------------- backtest
def walk_forward_schedule(
    n_months: int, train_months: int = TRAIN_MONTHS, hold_months: int = HOLD_MONTHS
) -> list[tuple[int, int, int, int]]:
    """Folds as ``(train_start, train_end, hold_start, hold_end)`` half-open ints.

    ``hold_start == train_end``: every training window ends exactly one month
    before its holding period begins (no overlap, no gap). Holding periods
    themselves are contiguous and non-overlapping.
    """
    folds = []
    cursor = train_months
    while cursor + hold_months <= n_months:
        folds.append((cursor - train_months, cursor, cursor, cursor + hold_months))
        cursor += hold_months
    return folds


def rebalance_cost(previous: np.ndarray, target: np.ndarray, cost_bps: float) -> float:
    """Cost in return units: bps-per-side x sum of absolute weight changes."""
    return (cost_bps / 1e4) * float(np.abs(np.asarray(target) - np.asarray(previous)).sum())


def walk_forward_backtest(
    monthly_returns: pd.DataFrame,
    scheme: Scheme | Callable[[pd.DataFrame], np.ndarray],
    train_months: int = TRAIN_MONTHS,
    hold_months: int = HOLD_MONTHS,
    cost_bps: float = COST_BPS_PER_SIDE,
) -> dict:
    """Walk-forward backtest net of per-side costs on turnover.

    Weights drift with returns inside each holding period; the turnover charge
    at each rebalance compares the fresh target with the drifted weights (the
    initial charge is the cost of entering from cash, i.e. full turnover).
    """
    if isinstance(scheme, Scheme):
        scheme_fn = scheme.fn
    else:
        scheme_fn = scheme
    folds = walk_forward_schedule(len(monthly_returns), train_months, hold_months)
    net_rows: list[tuple[pd.Timestamp, float]] = []
    gross_rows: list[tuple[pd.Timestamp, float]] = []
    turnover: list[float] = []
    weight_snapshots: list[np.ndarray] = []
    weights = np.zeros(monthly_returns.shape[1])
    for train_start, train_end, hold_start, hold_end in folds:
        window = monthly_returns.iloc[train_start:train_end]
        target = np.asarray(scheme_fn(window), dtype=float)
        charge = rebalance_cost(weights, target, cost_bps)
        turnover.append(float(np.abs(target - weights).sum()))
        weights = target.copy()
        weight_snapshots.append(weights.copy())
        for i in range(hold_start, hold_end):
            r = monthly_returns.iloc[i].to_numpy(dtype=float)
            gross = float(weights @ r)
            net = gross - (charge if i == hold_start else 0.0)
            net_rows.append((monthly_returns.index[i], net))
            gross_rows.append((monthly_returns.index[i], gross))
            # drift weights through the month (start-of-period weights are exact)
            weights = weights * (1.0 + r) / (1.0 + gross)
    net_series = pd.Series(dict(net_rows)).sort_index()
    gross_series = pd.Series(dict(gross_rows)).sort_index()
    return {
        "net": net_series,
        "gross": gross_series,
        "turnover": turnover,
        "schedule": folds,
        "weights": weight_snapshots,
        "total_cost_return": float(sum(t * cost_bps / 1e4 for t in turnover)),
        "avg_turnover": float(np.mean(turnover)) if turnover else 0.0,
        "n_folds": len(folds),
    }


def performance_summary(net_returns: pd.Series) -> dict:
    """Annualized summary of a monthly net return series (rf = 0)."""
    values = net_returns.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("need at least 2 monthly observations")
    mean_monthly = float(np.mean(values))
    vol_monthly = float(np.std(values, ddof=1))
    wealth = float(np.prod(1.0 + values))
    years = len(values) / 12.0
    dd = (np.cumprod(1.0 + values) / np.maximum.accumulate(np.cumprod(1.0 + values))) - 1.0
    return {
        "cagr": float(wealth ** (1.0 / years) - 1.0),
        "ann_vol": float(vol_monthly * np.sqrt(12.0)),
        "sharpe": float(mean_monthly / vol_monthly * np.sqrt(12.0)) if vol_monthly > 0 else 0.0,
        "max_dd": float(dd.min()),
        "total_return": float(wealth - 1.0),
        "n_months": int(len(values)),
        "hit_rate": float(np.mean(values > 0)),
    }
