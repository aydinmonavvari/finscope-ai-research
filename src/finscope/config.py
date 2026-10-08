"""Central configuration for the Finscope workbench.

All paths are derived from this file's location, so the project is portable
and no absolute paths are ever written into reports. Datasets themselves are
never committed (``data/raw`` and ``data/processed`` are git-ignored); only
the bundled headline sample under ``data/samples`` is tracked.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------- market data
TICKERS = ["SPY", "AAPL", "MSFT", "JPM", "XOM", "PG", "AMZN"]
PRICE_START = "2018-01-01"
PRICES_CACHE = RAW_DIR / "prices_daily.csv"

# ---------------------------------------------------------------- macro (FRED)
# Public CSV endpoint; no API key required. Verified reachable; cache-first.
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_SERIES = {
    "CPIAUCSL": "cpi_index",  # CPI: All Urban Consumers (index, monthly)
    "UNRATE": "unemployment_rate",  # civilian unemployment rate (monthly)
    "DGS10": "yield_10y",  # 10-year Treasury constant maturity (daily)
}
MACRO_START = "2013-01-01"  # extra history so lagged features have depth

# ------------------------------------------------------------------ analytics
TRADING_DAYS = 252
ROLLING_VOL_WINDOW = 63  # ~ one quarter of daily observations
RISK_FREE_ANNUAL = 0.0  # documented simplification for Sharpe ratios

# ---------------------------------------------------------------- forecasting
FORECAST_HORIZON = 12  # months ahead
FORECAST_STEP = 3  # rolling-origin step in months (Tashman 2000)
FORECAST_MIN_TRAIN = 24  # minimum training pairs before the first origin
RIDGE_ALPHA = 1.0

# ------------------------------------------------------------------ portfolio
TRAIN_MONTHS = 12
HOLD_MONTHS = 3
ASSET_CAP = 0.40
COST_BPS_PER_SIDE = 10.0

# ----------------------------------------------------------------------- risk
VAR_LEVELS = (0.95, 0.99)

# ----------------------------------------------------------------------- misc
SEED = 42


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for path in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
