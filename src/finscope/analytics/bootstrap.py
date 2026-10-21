"""Seeded circular block bootstrap shared by the portfolio and forecast stages.

Why a block bootstrap (and not IID)
-----------------------------------
Out-of-sample strategy returns and rolling-origin forecast losses are serially
dependent (the walk-forward design holds one weight vector for a whole
3-month holding period; the 12-month-ahead inflation targets of adjacent
origins overlap by 9 months). An IID bootstrap destroys exactly this
dependence and understates the sampling variability of statistics such as the
Sharpe ratio. The circular block bootstrap (Politis & Romano 1994) resamples
contiguous blocks with wrap-around: within-block dependence is preserved, the
wrap-around avoids edge loss, and every resample has the same length as the
original series.

Methodological origin
---------------------
This module re-implements, in finscope-idiomatic vectorized form, the engine
introduced in ``portfolio-optimization-lab`` Phase 2 Part A
(``docs/admissions/PHASE2-RESEARCH-PLAN.md`` and its
``src/portfolio_optimization_lab/bootstrap.py``, pre-registered before
implementation). The streams are NOT stream-compatible with the P04 engine
(different draw order); the guarantee here is internal: for a fixed seed the
index stream is deterministic, and the SAME index matrix is applied to both
legs of any paired comparison.

Determinism contract: for a fixed ``seed`` and identical ``(n, block_len,
n_boot)`` the returned index matrix is bitwise identical across calls, Python
versions and platforms (numpy's PCG64 generator is stream-stable).
"""

from __future__ import annotations

import numpy as np

# Monthly annualization convention of finscope.portfolio.construct.
#   sharpe = mean(r) / std(r, ddof=1) * sqrt(12)   (rf = 0, documented)
# This mirrors performance_summary() EXACTLY (same operations, same order);
# a regression test pins the bitwise agreement.
MONTHLY_PERIODS_PER_YEAR = 12


def block_indices(
    n: int, block_len: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """A ``(n_boot, n)`` integer index matrix built from circular blocks.

    For each of the ``n_boot`` resamples, ``ceil(n / block_len)`` block start
    positions are drawn uniformly from ``[0, n)`` and laid out as ``block_len``
    consecutive (wrap-around) indices, trimmed to length ``n``. Advances the
    given generator; determinism is the caller's responsibility (pass a fresh
    ``np.random.default_rng(seed)`` for a registered stream).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 1 <= block_len <= n:
        raise ValueError(f"block_len must be in [1, n={n}], got {block_len}")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_len)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def resample_series(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Apply an index matrix to a 1-D series -> ``(n_boot, n)`` resamples.

    ``resample_series(v, idx)[b] == v[idx[b]]`` elementwise; passing the SAME
    ``idx`` to two series is what makes a bootstrap *paired*.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {values.shape}")
    idx = np.asarray(idx)
    if idx.ndim != 2:
        raise ValueError(f"expected a 2-D index matrix, got shape {idx.shape}")
    if idx.min() < 0 or idx.max() >= len(values):
        raise ValueError("index matrix out of bounds for the given series")
    return values[idx]


def monthly_sharpe(values: np.ndarray) -> float:
    """Annualized monthly Sharpe (rf = 0) — the ``performance_summary`` arithmetic.

    ``mean(r) / std(r, ddof=1) * sqrt(12)``; a zero-volatility resample scores
    0.0 by the same convention ``performance_summary`` uses. Do NOT change the
    operation order: a regression test pins bitwise agreement with
    ``performance_summary`` on the full sample.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("need at least 2 monthly observations")
    mean = float(np.mean(values))
    vol = float(np.std(values, ddof=1))
    if not vol > 0:
        return 0.0
    return mean / vol * np.sqrt(12.0)


def resampled_monthly_sharpes(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Monthly-convention Sharpe of every resample (vectorized over rows)."""
    sampled = resample_series(values, idx)
    mean = sampled.mean(axis=1)
    vol = sampled.std(axis=1, ddof=1)
    ratio = np.divide(mean, vol, out=np.zeros_like(mean), where=vol > 0)
    return ratio * np.sqrt(12.0)


def percentile_ci(statistics: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of a bootstrap statistic vector."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = np.percentile(statistics, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)
