"""Cache-first loaders for FRED macro series (public CSV endpoint, no API key).

Series used by the workbench: CPIAUCSL (CPI index), UNRATE (unemployment) and
DGS10 (10-year Treasury yield). The CSV endpoint is verified reachable and is
the only FRED access path used; responses are cached under ``data/raw`` so
every subsequent run — and CI — is fully offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finscope.config import AS_OF, FRED_BASE_URL, FRED_SERIES, MACRO_START, RAW_DIR


def _cache_path(series_id: str) -> Path:
    return RAW_DIR / f"fred_{series_id}.csv"


def load_fred_series(
    series_id: str,
    cache_path: Path | None = None,
    refresh: bool = False,
    start: str = MACRO_START,
) -> pd.Series:
    """Load one FRED series as a daily/weekly/monthly ``pd.Series`` (cache-first)."""
    cache_path = cache_path if cache_path is not None else _cache_path(series_id)
    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        url = FRED_BASE_URL.format(series_id=series_id)
        try:
            payload = pd.read_csv(url)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download FRED series {series_id} and no usable cache "
                f"exists at {cache_path}. Run once with network access."
            ) from exc
        payload = payload.rename(columns={payload.columns[0]: "date"})
        if series_id not in payload.columns:
            # Never cache a malformed response. The FRED CSV endpoint is
            # documented (PHASE-2 part B validation report) to serve an HTML
            # bot-challenge page with HTTP 200 when the client is blocked; that
            # page can parse as a junk frame whose first column is renamed to
            # "date". Caching it would silently poison every later offline run.
            raise RuntimeError(
                f"FRED endpoint returned a malformed payload for {series_id} "
                f"(missing column {series_id!r}); nothing was cached. The "
                f"endpoint may be serving an HTML block page — retry later or "
                f"populate {cache_path} from a trusted session."
            )
        payload["date"] = pd.to_datetime(payload["date"])
        frame = payload.set_index("date").sort_index()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index_label="date")
    if series_id not in frame.columns:
        raise ValueError(f"FRED payload for {series_id} lacks column {series_id}")
    series = pd.to_numeric(frame[series_id], errors="coerce").dropna()
    series.name = FRED_SERIES.get(series_id, series_id)
    # Registered AS_OF vintage pin (PHASE-2 plan §4): truncate at the as-of
    # date so a re-run cannot silently extend the panel past 2026-10-08,
    # regardless of what the endpoint serves on the day of the re-run.
    return series.loc[start:AS_OF]


def load_macro_monthly(refresh: bool = False) -> pd.DataFrame:
    """Monthly macro panel: CPI index, unemployment rate, 10-year yield.

    CPI and unemployment are already monthly; the daily 10-year yield is
    aggregated to the monthly mean (documented in the research report).
    """
    cpi = load_fred_series("CPIAUCSL", refresh=refresh)
    unrate = load_fred_series("UNRATE", refresh=refresh)
    dgs10 = load_fred_series("DGS10", refresh=refresh)
    panel = pd.concat(
        {
            "cpi_index": cpi.resample("ME").last(),
            "unemployment_rate": unrate.resample("ME").last(),
            "yield_10y": dgs10.resample("ME").mean(),
        },
        axis=1,
    )
    return panel.dropna(how="all")
