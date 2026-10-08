"""Cache-first ingestion of daily adjusted close prices from Yahoo Finance.

``load_prices`` reads the local cache when it exists (offline-safe, no network
call) and only downloads through ``yfinance`` when the cache is missing or
``refresh=True``. Raw datasets are cached under ``data/raw`` which is
git-ignored — the repository never redistributes market data.
"""

from __future__ import annotations

import pandas as pd

from finscope.config import PRICE_START, PRICES_CACHE, TICKERS


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by a naive, normalized DatetimeIndex."""
    frame = frame.sort_index()
    index = pd.to_datetime(frame.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame.index.name = "date"
    return frame


def fetch_prices(tickers: list[str], start: str = PRICE_START) -> pd.DataFrame:
    """Download daily adjusted close prices (wide frame, one column per ticker)."""
    import yfinance as yf  # lazy import: keeps CLI --help and tests import-light

    raw = yf.download(
        tickers=list(tickers),
        start=start,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty or "Close" not in raw:
        raise RuntimeError("yfinance returned no data")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise RuntimeError(f"yfinance did not return columns for: {missing}")
    return _clean_frame(close[list(tickers)])


def load_prices(
    tickers: list[str] | None = None,
    start: str = PRICE_START,
    cache_path=PRICES_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Cache-first daily price loader (offline-safe reads).

    Raises a clear error if the cache is absent and the network is unusable.
    """
    tickers = list(tickers if tickers is not None else TICKERS)
    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in frame.columns]
        if missing:
            raise ValueError(f"cache {cache_path} is missing columns: {missing}")
        return _clean_frame(frame[tickers])
    try:
        frame = fetch_prices(tickers, start)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download prices for {tickers} and no usable cache exists "
            f"at {cache_path}. Run once with network access to populate the cache."
        ) from exc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index_label="date")
    return frame


def monthly_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end closing prices (last observation of each calendar month)."""
    return prices.resample("ME").last().dropna(how="all")


def monthly_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly simple returns used for portfolio arithmetic (w @ r is exact)."""
    return monthly_close(prices).pct_change().dropna(how="all")
