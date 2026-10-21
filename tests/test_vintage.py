"""Offline tests for the registered AS_OF vintage pin (PHASE-2 plan §4).

The pin guarantees that a re-run reproduces the v1.0.0 data window instead of
silently extending the panel: yfinance downloads are windowed at AS_OF (with
the documented exclusive-end workaround) and every cached read is truncated.
All tests are synthetic and offline (tmp-path caches, no network).
"""

import pandas as pd
import pytest

from finscope.config import AS_OF
from finscope.data.macro import load_fred_series
from finscope.data.prices import FETCH_END_EXCLUSIVE, load_prices


def test_as_of_pin_constants():
    assert AS_OF == "2026-10-08"
    # yfinance `end` is EXCLUSIVE: one extra calendar day includes the AS_OF bar
    assert FETCH_END_EXCLUSIVE == "2026-10-09"


def test_macro_series_truncated_at_as_of(tmp_path):
    frame = pd.DataFrame(
        {
            "date": [
                "2026-06-01",
                "2026-07-01",
                "2026-08-01",
                "2026-09-01",
                "2026-10-01",
                "2026-11-01",
            ],
            "TEST": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )
    cache = tmp_path / "fred_TEST.csv"
    frame.to_csv(cache, index=False)
    series = load_fred_series("TEST", cache_path=cache, start="2020-01-01")
    assert series.index.max() == pd.Timestamp("2026-10-01")  # last month <= the pin
    assert len(series) == 5
    assert 105.0 not in series.to_numpy()  # the post-pin observation is dropped
    # the cache itself is untouched (cache-first behavior preserved)
    reloaded = pd.read_csv(cache)
    assert len(reloaded) == 6


def test_macro_series_dgs10_daily_truncated_at_as_of(tmp_path):
    frame = pd.DataFrame(
        {
            "date": ["2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10"],
            "TEST2": [4.0, 4.1, 4.2, 4.3],
        }
    )
    cache = tmp_path / "fred_TEST2.csv"
    frame.to_csv(cache, index=False)
    series = load_fred_series("TEST2", cache_path=cache, start="2020-01-01")
    assert series.index.max() == pd.Timestamp(AS_OF)
    assert list(series.to_numpy()) == [4.0, 4.1]


def test_prices_cache_truncated_at_as_of(tmp_path):
    dates = pd.date_range("2026-10-01", periods=12, freq="B")  # includes days past AS_OF
    frame = pd.DataFrame(
        {"SPY": range(100, 112), "AAPL": range(200, 212)}, index=dates
    )
    cache = tmp_path / "prices_daily.csv"
    frame.to_csv(cache, index_label="date")
    prices = load_prices(["SPY", "AAPL"], cache_path=cache)
    assert prices.index.max() == pd.Timestamp(AS_OF)
    assert (prices.index <= pd.Timestamp(AS_OF)).all()


def test_prices_within_pin_window_unchanged(tmp_path):
    dates = pd.date_range("2026-09-28", periods=8, freq="B")  # all <= AS_OF
    frame = pd.DataFrame({"SPY": range(1, 9)}, index=dates)
    cache = tmp_path / "prices_daily.csv"
    frame.to_csv(cache, index_label="date")
    prices = load_prices(["SPY"], cache_path=cache)
    assert len(prices) == 8
    assert prices.index.equals(dates)


def test_fred_malformed_payload_never_cached(tmp_path, monkeypatch):
    """An HTML bot-challenge page (HTTP 200) must raise, never poison the cache.

    Regression guard for the environment documented in the PHASE-2 part B
    validation report: FRED serving an HTML block page must not be cached as
    a 'successful' fetch (cache poisoning would corrupt every later offline
    run silently).
    """
    import pandas as pd

    import finscope.data.macro as macro_mod

    junk = pd.DataFrame({"search": ["<!doctype html>", "<html>", "</html>"]})

    def fake_read_csv(*args, **kwargs):
        return junk.copy()

    monkeypatch.setattr(macro_mod.pd, "read_csv", fake_read_csv)
    cache = tmp_path / "fred_CPIAUCSL.csv"
    with pytest.raises(RuntimeError, match="malformed"):
        macro_mod.load_fred_series("CPIAUCSL", cache_path=cache)
    assert not cache.exists()
