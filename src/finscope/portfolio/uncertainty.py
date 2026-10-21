"""PHASE-2 part B portfolio uncertainty suite (pre-registered, RQ1-RQ4).

Implements exactly the registered methodology of
``docs/admissions/PHASE2-P10-PLAN.md`` (committed before implementation):

- RQ1 ``level_cis``: per-scheme annualized net-Sharpe percentile CIs, circular
  block bootstrap on monthly returns, block lengths {3, 6, 12} months,
  B = 2000, seed 42 (one fresh seed-42 stream per block length);
- RQ2 ``pairwise_holm``: paired block bootstrap over the 3 DISTINCT scheme
  pairs (equal / inv_vol / min_var), identical resampled index stream applied
  to both legs of a pair, add-one two-sided p-values, Holm step-down over the
  3-pair family at alpha = 0.05. The pair (min_var, max_sharpe) is EXCLUDED a
  priori: max_sharpe fell back to min-variance in 100% of folds, so its OOS
  series is bitwise identical by construction — the exclusion is verified in
  code and reported as a machine-readable ``structural_exclusion`` field;
- RQ3 ``cost_grid``: exact recomputation net(bps) = gross - turnover*bps/1e4
  per fold from the stored per-fold turnover (no re-estimation), with the
  registered exact claims (affine mean net return; bitwise net == gross at
  0 bps) pinned by tests;
- RQ4 ``window_stability``: 3 non-overlapping 31-month windows, Kendall tau
  between adjacent windows, rank-1 counts — registered EXPLORATORY.

Registered seeds: 42 for every bootstrap stream; 43 reserved-unused; 2026
dead (P04 convention, never used here).

Honesty contract: with n = 93 monthly observations wide CIs are the expected
outcome; results are reported exactly as measured, null or not. No "best
scheme" language is generated anywhere in this module.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

from finscope.analytics.bootstrap import (
    block_indices,
    monthly_sharpe,
    percentile_ci,
    resampled_monthly_sharpes,
)

SEED_BOOTSTRAP = 42
SEED_43_RESERVED_UNUSED = 43  # registered reserved: no Monte Carlo in this plan
SEED_2026_DEAD = 2026  # registered DEAD: must never be repurposed

DEFAULT_BLOCK_LENS = (3, 6, 12)
DEFAULT_COST_GRID = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0)
DISTINCT_SCHEMES = ("equal", "inv_vol", "min_var")
ALL_SCHEMES = ("equal", "inv_vol", "min_var", "max_sharpe")
STRUCTURAL_PAIR = ("min_var", "max_sharpe")


def _validate_series_map(net: dict[str, pd.Series], names) -> dict[str, np.ndarray]:
    arrays = {}
    for name in names:
        if name not in net:
            raise KeyError(f"missing OOS net-return series for scheme {name!r}")
        series = net[name].dropna()
        arrays[name] = series.to_numpy(dtype=float)
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"scheme series length mismatch: {lengths}")
    n = lengths.pop()
    if n < 2:
        raise ValueError("need at least 2 monthly observations")
    return arrays


def structural_exclusion_record(net: dict[str, pd.Series]) -> dict:
    """Verify and document the registered a priori exclusion of (min_var, max_sharpe).

    max_sharpe's tangency solution is infeasible in every fold of the
    committed run (fallback_share = 1.0), so its OOS series IS min-variance's
    series. The identity is asserted here bitwise; the returned record is
    embedded in the pairwise artifact so the exclusion is machine-readable.
    """
    a = net[STRUCTURAL_PAIR[0]].dropna().to_numpy(dtype=float)
    b = net[STRUCTURAL_PAIR[1]].dropna().to_numpy(dtype=float)
    identical = a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=False))
    if not identical:
        raise AssertionError(
            "registered structural fact violated: max_sharpe's OOS series is not "
            "bitwise identical to min_var's — the a priori exclusion of the pair "
            "(min_var, max_sharpe) no longer applies and the plan must be revisited"
        )
    return {
        "excluded_pair": list(STRUCTURAL_PAIR),
        "reason": (
            "max_sharpe fell back to min-variance in 100% of folds "
            "(fallback_share = 1.0); its OOS net series is bitwise identical to "
            "min_var's by construction, so a zero pairwise difference there is "
            "structural, not a finding"
        ),
        "verified_bitwise_identical": True,
        "distinct_schemes_tested": list(DISTINCT_SCHEMES),
        "n_pairs_tested": 3,
    }


def level_cis(
    net: dict[str, pd.Series],
    block_lens: tuple[int, ...] = DEFAULT_BLOCK_LENS,
    n_boot: int = 2000,
    seed: int = SEED_BOOTSTRAP,
    confidence_level: float = 0.95,
) -> dict:
    """RQ1: per-scheme Sharpe percentile CIs per block length.

    One fresh ``default_rng(seed)`` stream per block length; the index matrix
    is drawn once and every scheme is scored on those same resamples. The
    point estimate reproduces ``performance_summary(series)["sharpe"]``
    (same arithmetic, pinned bitwise by tests).
    """
    schemes = [s for s in ALL_SCHEMES if s in net]
    arrays = _validate_series_map(net, schemes)
    n = len(arrays[schemes[0]])
    point = {s: monthly_sharpe(arrays[s]) for s in schemes}

    levels: dict[str, dict] = {s: {} for s in schemes}
    for block_len in block_lens:
        rng = np.random.default_rng(seed)
        idx = block_indices(n, block_len, n_boot, rng)
        for s in schemes:
            sharpes = resampled_monthly_sharpes(arrays[s], idx)
            lo, hi = percentile_ci(sharpes, confidence_level)
            levels[s][str(block_len)] = {
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
            }
    return {
        "B": int(n_boot),
        "block_lens": [int(b) for b in block_lens],
        "seed": int(seed),
        "confidence_level": confidence_level,
        "statistic": (
            "annualized monthly Sharpe sqrt(12)*mean(r)/std(r, ddof=1), rf = 0, "
            "on the 93-month OOS net-return window; circular block bootstrap, "
            "percentile intervals; one fresh seed-42 stream per block length"
        ),
        "n_months": int(n),
        "point_estimates": {s: round(v, 6) for s, v in point.items()},
        "levels": levels,
        "reading_rule": (
            "no scheme preference is supported unless a 95% CI excludes every "
            "other scheme's point estimate (registered decision rule RQ1)"
        ),
    }


def holm_adjust(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down adjustment over a named family of p-values.

    The adjusted value for the k-th smallest p is the running maximum of
    ``(m - k) * p_k`` capped at 1; monotone by construction. Rejection follows
    the step-down rule: reject while adjusted p <= alpha, stopping at the
    first non-rejection.
    """
    names = sorted(p_values, key=lambda k: (p_values[k], k))
    m = len(names)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for k, name in enumerate(names):
        raw = float(p_values[name])
        running_max = max(running_max, (m - k) * raw)
        adjusted[name] = float(min(1.0, running_max))
    rejected_so_far = True
    out: dict[str, dict] = {}
    for name in p_values:
        adj = adjusted[name]
        if rejected_so_far and adj <= alpha:
            reject = True
        else:
            reject = False
            rejected_so_far = False
        out[name] = {
            "p_raw": round(float(p_values[name]), 6),
            "p_holm": round(adj, 6),
            "reject_holm_05": bool(reject),
        }
    return out


def pairwise_holm(
    net: dict[str, pd.Series],
    block_len: int = 6,
    n_boot: int = 2000,
    seed: int = SEED_BOOTSTRAP,
    sensitivity_blocks: tuple[int, ...] = (3, 12),
    alpha: float = 0.05,
    confidence_level: float = 0.95,
) -> dict:
    """RQ2: paired bootstrap Sharpe differences + Holm step-down over 3 pairs.

    Per pair a FRESH ``default_rng(seed)`` stream is drawn (registered "seed
    42 per pair"); the SAME index matrix is applied to both legs (paired).
    p-values are two-sided with the add-one correction
    ``p = 2 * min((#{d <= 0} + 1) / (B + 1), (#{d >= 0} + 1) / (B + 1))``.
    The headline block (default 6 months) carries the p-values; the
    registered block-length sensitivity (3, 12) is reported as UNADJUSTED
    percentile intervals only and must always be read next to the Holm column.
    """
    exclusion = structural_exclusion_record(net)
    arrays = _validate_series_map(net, DISTINCT_SCHEMES)
    n = len(arrays[DISTINCT_SCHEMES[0]])

    def pair_statistics(a: str, b: str, blk: int, rng: np.random.Generator) -> np.ndarray:
        idx = block_indices(n, blk, n_boot, rng)
        sa = resampled_monthly_sharpes(arrays[a], idx)
        sb = resampled_monthly_sharpes(arrays[b], idx)
        return sa - sb

    pairs_out = []
    raw_p: dict[str, float] = {}
    for a, b in combinations(DISTINCT_SCHEMES, 2):
        point = monthly_sharpe(arrays[a]) - monthly_sharpe(arrays[b])
        boot = pair_statistics(a, b, block_len, np.random.default_rng(seed))
        lo, hi = percentile_ci(boot, confidence_level)
        n_le = int(np.sum(boot <= 0))
        n_ge = int(np.sum(boot >= 0))
        p_raw = 2.0 * min((n_le + 1) / (n_boot + 1), (n_ge + 1) / (n_boot + 1))
        p_raw = float(min(1.0, p_raw))
        sensitivity = {}
        for blk in sensitivity_blocks:
            if blk == block_len:
                continue
            boot_s = pair_statistics(a, b, blk, np.random.default_rng(seed))
            lo_s, hi_s = percentile_ci(boot_s, confidence_level)
            sensitivity[str(blk)] = {"ci_low": round(lo_s, 6), "ci_high": round(hi_s, 6)}
        key = f"{a}__vs__{b}"
        raw_p[key] = p_raw
        pairs_out.append(
            {
                "pair": key,
                "a": a,
                "b": b,
                "diff": round(point, 6),
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "ci_excludes_zero_unadjusted": bool(lo > 0 or hi < 0),
                "sensitivity": sensitivity,
            }
        )

    holm = holm_adjust(raw_p, alpha=alpha)
    for pair in pairs_out:
        pair.update(holm[pair["pair"]])

    n_reject = sum(1 for p in pairs_out if p["reject_holm_05"])
    return {
        "B": int(n_boot),
        "block_len": int(block_len),
        "sensitivity_blocks": [int(b) for b in sensitivity_blocks if b != block_len],
        "seed": int(seed),
        "alpha": alpha,
        "method": (
            "paired circular-block bootstrap of Sharpe(a) - Sharpe(b) on monthly "
            "OOS net returns (identical resampled indices for both legs, fresh "
            "seed-42 stream per pair); two-sided add-one p-values; Holm "
            "step-down over the 3 distinct pairs (family-wise alpha 0.05)"
        ),
        "family_size": len(pairs_out),
        "n_reject_holm_05": n_reject,
        "structural_exclusion": exclusion,
        "pairs": pairs_out,
    }


def net_at_cost(
    gross: np.ndarray,
    fold_turnover: np.ndarray,
    fold_start_positions: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    """Exact recomputation: ``net(bps) = gross - turnover * bps / 1e4`` per fold.

    The charge lands on each fold-start month and scales LINEARLY in bps; the
    ``bps / 1e4`` factor is formed first to match ``rebalance_cost``'s
    arithmetic bitwise (so net(reference bps) reproduces the backtest's own
    net series exactly, and net(0) is bitwise gross).
    """
    net = np.asarray(gross, dtype=float).copy()
    charge = np.asarray(fold_turnover, dtype=float) * (float(cost_bps) / 1e4)
    net[np.asarray(fold_start_positions, dtype=int)] -= charge
    return net


def cost_grid(
    gross: dict[str, pd.Series],
    fold_turnover: dict[str, pd.Series],
    fold_start_dates: pd.DatetimeIndex | list,
    grid: tuple[float, ...] = DEFAULT_COST_GRID,
    confidence_level: float = 0.95,
) -> dict:
    """RQ3: cost sensitivity of the scheme ordering on the registered grid.

    For every scheme and every bps level the net series is recomputed EXACTLY
    from the stored gross series and per-fold one-way turnover
    (``net(bps) = gross - turnover * bps / 1e4`` on fold-start months) and the
    monthly-convention Sharpe is evaluated on it. No optimizer is re-run, so
    the weight and turnover paths are identical across the grid by
    construction. Registered exact claims (pinned by unit tests): the mean
    net return is affine in bps with slope ``-sum(turnover) / (1e4 * n)``;
    net is bitwise gross at 0 bps.

    Returns the per-point table, the empirical slopes, and every pairwise
    ordering crossing observed between consecutive grid points.
    """
    schemes = [s for s in DISTINCT_SCHEMES if s in gross]
    if not schemes:
        raise ValueError("no distinct schemes present in the gross series map")
    gross_series = {s: gross[s].dropna() for s in schemes}
    index = gross_series[schemes[0]].index
    for s in schemes:
        if not gross_series[s].index.equals(index):
            raise ValueError(f"gross series for {s!r} is not aligned with the others")
    gross_arr = {s: gross_series[s].to_numpy(dtype=float) for s in schemes}
    turnover_arr = {s: fold_turnover[s].to_numpy(dtype=float) for s in schemes}
    n = len(gross_arr[schemes[0]])
    dates = pd.DatetimeIndex(fold_start_dates)
    missing = [str(d.date()) for d in dates if d not in index]
    if missing:
        raise ValueError(f"fold-start dates missing from the gross index: {missing}")
    positions = np.array([index.get_loc(d) for d in dates], dtype=int)

    rows = []
    sharpes_at = {bps: {} for bps in grid}
    for bps in grid:
        for s in schemes:
            net = net_at_cost(gross_arr[s], turnover_arr[s], positions, bps)
            sharpe = monthly_sharpe(net)
            sharpes_at[bps][s] = sharpe
            rows.append(
                {
                    "cost_bps": float(bps),
                    "scheme": s,
                    "mean_net_return": round(float(np.mean(net)), 12),
                    "sharpe_net": round(float(sharpe), 12),
                }
            )

    slopes = {}
    for s in schemes:
        total_turnover = float(np.sum(turnover_arr[s]))
        slopes[s] = round(-total_turnover / (1e4 * n), 12)

    crossings = []
    for a, b in combinations(schemes, 2):
        signs = [np.sign(sharpes_at[bps][a] - sharpes_at[bps][b]) for bps in grid]
        for i in range(len(grid) - 1):
            if signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1]:
                crossings.append(
                    {
                        "pair": f"{a}__vs__{b}",
                        "between_bps": [float(grid[i]), float(grid[i + 1])],
                        "from": f"{a} ahead" if signs[i] > 0 else f"{b} ahead",
                        "to": f"{a} ahead" if signs[i + 1] > 0 else f"{b} ahead",
                    }
                )
    any_crossing = bool(crossings)
    orderings = {}
    for bps in grid:
        order = sorted(schemes, key=lambda s: sharpes_at[bps][s], reverse=True)
        key = str(int(bps)) if float(bps).is_integer() else str(bps)
        orderings[key] = " > ".join(order)
    return {
        "grid": [float(g) for g in grid],
        "statistic": (
            "monthly-convention Sharpe of the EXACTLY recomputed net series "
            "net(bps) = gross - fold_turnover*bps/1e4 (no re-estimation); "
            "registered exact claims: mean net return affine in bps, net "
            "bitwise gross at 0 bps"
        ),
        "rows_12dp": rows,
        "mean_net_return_slope_per_bps": slopes,
        "ordering_at_grid_points": orderings,
        "ordering_crossings": crossings,
        "any_ordering_crossing": any_crossing,
        "reading_rule": (
            "report whether any pairwise ordering flips anywhere on the grid "
            "(registered decision rule RQ3); a monotone no-crossing result is a "
            "finding, not a failure"
        ),
    }


def window_stability(
    net: dict[str, pd.Series], n_windows: int = 3
) -> dict:
    """RQ4 (EXPLORATORY): rank stability over non-overlapping sub-windows.

    Splits the OOS window into ``n_windows`` equal, contiguous, non-overlapping
    monthly windows (93 months -> 3 x 31), scores each scheme's monthly-
    convention Sharpe per window, and reports Kendall tau between adjacent
    windows' rankings plus rank-1 counts over the 3 DISTINCT schemes
    (max_sharpe is structurally identical to min_var and is excluded from the
    ranking, mirroring the registered RQ2 exclusion). Registered label:
    EXPLORATORY — with 3 windows this is underpowered by construction and
    supports no inferential claim.
    """
    exclusion = structural_exclusion_record(net)
    all_schemes = [s for s in ALL_SCHEMES if s in net]
    arrays = _validate_series_map(net, all_schemes)
    n = len(arrays[all_schemes[0]])
    if n % n_windows != 0:
        raise ValueError(
            f"n={n} does not divide into {n_windows} equal non-overlapping windows"
        )
    window_len = n // n_windows

    window_sharpes = []
    for w in range(n_windows):
        seg = slice(w * window_len, (w + 1) * window_len)
        window_sharpes.append(
            {s: monthly_sharpe(arrays[s][seg]) for s in all_schemes}
        )
    taus = []
    for w in range(n_windows - 1):
        a = [window_sharpes[w][s] for s in DISTINCT_SCHEMES]
        b = [window_sharpes[w + 1][s] for s in DISTINCT_SCHEMES]
        raw = sps.kendalltau(a, b).statistic
        if not np.isfinite(raw):
            # Fully-tied rankings in a window (e.g. identical window Sharpes)
            # make Kendall's tau undefined (0/0). Strict JSON forbids NaN
            # (plan A8), so the pair is recorded as degenerate with tau=None —
            # machine-readable honesty, never a silent or crash-inducing value.
            taus.append({"windows": [w + 1, w + 2], "tau": None, "degenerate": True})
        else:
            taus.append(
                {"windows": [w + 1, w + 2], "tau": round(float(raw), 6), "degenerate": False}
            )
    rank1_counts = {
        s: sum(
            1
            for row in window_sharpes
            if s == max(DISTINCT_SCHEMES, key=lambda x: row[x])
        )
        for s in DISTINCT_SCHEMES
    }
    ranks_per_window = {
        str(w + 1): {
            "ranking": sorted(
                DISTINCT_SCHEMES, key=lambda s: window_sharpes[w][s], reverse=True
            )
        }
        for w in range(n_windows)
    }
    return {
        "n_windows": int(n_windows),
        "window_months": int(window_len),
        "label": (
            "EXPLORATORY (registered): 3 windows — underpowered by construction; "
            "descriptive only"
        ),
        "statistic": (
            "annualized monthly Sharpe per contiguous non-overlapping window; "
            "Kendall tau between adjacent-window rankings over the 3 distinct "
            "schemes (max_sharpe structurally = min_var, excluded from ranks)"
        ),
        "window_sharpes": [
            {s: round(v, 6) for s, v in row.items()} for row in window_sharpes
        ],
        "window_rankings": ranks_per_window,
        "adjacent_kendall_tau": taus,
        "mean_adjacent_kendall_tau": (
            round(float(np.mean([t["tau"] for t in taus if t["tau"] is not None])), 6)
            if any(t["tau"] is not None for t in taus)
            else None
        ),
        "rank1_counts": rank1_counts,
        "structural_exclusion": exclusion,
    }
