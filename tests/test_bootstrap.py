"""Offline tests for the shared circular block bootstrap (PHASE-2 part B).

All synthetic, deterministic, offline. Registered pins: determinism under a
fixed seed, index validity, block structure, paired-stream bitwise identity,
and bitwise agreement of the monthly Sharpe helper with
``performance_summary``.
"""

import numpy as np
import pandas as pd

from finscope.analytics.bootstrap import (
    block_indices,
    monthly_sharpe,
    percentile_ci,
    resample_series,
    resampled_monthly_sharpes,
)
from finscope.portfolio.construct import performance_summary


def test_block_indices_deterministic_under_fixed_seed():
    rng1 = np.random.default_rng(42)
    a = block_indices(93, 6, 50, rng1)
    rng2 = np.random.default_rng(42)
    b = block_indices(93, 6, 50, rng2)
    assert np.array_equal(a, b)
    rng3 = np.random.default_rng(43)
    c = block_indices(93, 6, 50, rng3)
    assert not np.array_equal(a, c)


def test_block_indices_shape_and_validity():
    idx = block_indices(93, 6, 200, np.random.default_rng(42))
    assert idx.shape == (200, 93)
    assert idx.dtype.kind == "i"
    assert idx.min() >= 0
    assert idx.max() <= 92


def test_block_indices_preserve_block_structure():
    """Within a row, indices come in runs of block_len consecutive (mod n)."""
    n, block_len, n_boot = 20, 5, 100
    idx = block_indices(n, block_len, n_boot, np.random.default_rng(7))
    for row in idx:
        for start in range(0, n, block_len):
            run = row[start : start + block_len]
            expected = (row[start] + np.arange(len(run))) % n
            assert np.array_equal(run, expected)


def test_block_indices_trims_to_length_n():
    n, block_len = 10, 4  # ceil(10/4)=3 blocks -> 12 draws trimmed to 10
    idx = block_indices(n, block_len, 1, np.random.default_rng(0))
    assert idx.shape == (1, 10)


def test_block_indices_rejects_invalid_arguments():
    import pytest

    with pytest.raises(ValueError):
        block_indices(0, 3, 10, np.random.default_rng(1))
    with pytest.raises(ValueError):
        block_indices(5, 0, 10, np.random.default_rng(1))
    with pytest.raises(ValueError):
        block_indices(5, 6, 10, np.random.default_rng(1))
    with pytest.raises(ValueError):
        block_indices(5, 2, 0, np.random.default_rng(1))


def test_resample_series_matches_indexing_and_bounds_check():
    values = np.arange(10, dtype=float)
    idx = np.array([[0, 3, 3, 9], [9, 9, 1, 4]])
    out = resample_series(values, idx)
    assert np.array_equal(out, values[idx])
    assert out.shape == (2, 4)
    import pytest

    bad = np.array([[0, 10]])
    with pytest.raises(ValueError):
        resample_series(values, bad)


def test_paired_streams_bitwise_identical_across_series():
    """A2: the SAME index matrix must resample both legs of a pair."""
    rng = np.random.default_rng(42)
    idx_a = block_indices(93, 6, 25, rng)
    a = np.random.default_rng(1).normal(0.01, 0.04, 93)
    b = np.random.default_rng(2).normal(0.008, 0.03, 93)
    ra = resample_series(a, idx_a)
    rb = resample_series(b, idx_a)
    # paired: the difference of resamples equals the resample of differences
    assert np.array_equal(ra - rb, resample_series(a - b, idx_a))


def test_monthly_sharpe_bitwise_matches_performance_summary():
    values = np.random.default_rng(11).normal(0.01, 0.045, 93)
    via_helper = monthly_sharpe(values)
    via_summary = performance_summary(pd.Series(values))["sharpe"]
    assert via_helper == via_summary  # bitwise, not approx


def test_monthly_sharpe_zero_volatility_returns_zero():
    assert monthly_sharpe(np.array([0.01, 0.01, 0.01])) == 0.0


def test_resampled_sharpes_match_per_row_helper():
    values = np.random.default_rng(5).normal(0.0, 0.03, 40)
    idx = block_indices(40, 5, 30, np.random.default_rng(3))
    vectorized = resampled_monthly_sharpes(values, idx)
    for row in range(30):
        assert vectorized[row] == monthly_sharpe(values[idx[row]])


def test_percentile_ci_basic():
    import pytest

    stats = np.arange(1000, dtype=float)
    lo, hi = percentile_ci(stats, 0.95)
    assert lo == pytest.approx(24.975)  # numpy linear interpolation, rank 2.5%
    assert hi == pytest.approx(974.025)
    assert lo < hi
    with pytest.raises(ValueError):
        percentile_ci(stats, 1.5)
