from __future__ import annotations
import pandas as pd
from pathlib import Path

def cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("^", "")
    return cache_dir / f"cache_{safe}_{start}_{end}.csv"

def load_from_stooq(ticker: str) -> pd.DataFrame:
    stooq_symbol = "spy.us" if ticker.upper() == "SPY" else ticker.lower()
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def load_from_openbb(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Optional: requires OpenBB installed + provider configured.
    We keep this isolated so the default run doesn't depend on it.
    """
    try:
        from openbb import obb
    except Exception as e:
        raise RuntimeError("OpenBB not installed. Install openbb to use data_source='openbb'.") from e

    # OpenBB API can change across versions/providers
    # We keep a minimal attempt and fall back if not available.
    try:
        # Example (may vary by OpenBB version):
        df = obb.equity.price.historical(symbol=ticker, start_date=start, end_date=end).to_df()
    except Exception as e:
        raise RuntimeError(
            "OpenBB fetch failed. Check provider setup / OpenBB version. "
            "Use data_source='stooq' for the default reproducible path."
        ) from e

    # Normalize columns
    # Try common fields: date/close
    colmap = {}
    for c in df.columns:
        if c.lower() in ("date", "datetime", "time"):
            colmap[c] = "Date"
        if c.lower() in ("close", "adj_close", "adjclose"):
            colmap[c] = "Close"
    df = df.rename(columns=colmap)

    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError("OpenBB output missing required columns Date/Close.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close"]].copy()

def load_prices(ticker: str, start: str, end: str, cache_dir: Path, source: str) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, ticker, start, end)

    # Always prefer cache if it exists
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"])

    if source == "cache":
        raise RuntimeError("Cache requested but cache file not found. Run once with stooq/openbb to create it.")

    if source == "stooq":
        df = load_from_stooq(ticker)
        df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] < pd.to_datetime(end))].copy()
        df.to_csv(path, index=False)
        return df

    if source == "openbb":
        df = load_from_openbb(ticker, start, end)
        df.to_csv(path, index=False)
        return df

    raise ValueError("Unknown data_source. Use 'cache', 'stooq', or 'openbb'.")
