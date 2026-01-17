import numpy as np
import pandas as pd

def compute_log_returns(price_df: pd.DataFrame) -> pd.Series:
    if "Date" not in price_df.columns:
        raise ValueError("Expected a 'Date' column.")
    if "Close" not in price_df.columns:
        raise ValueError("Expected a 'Close' column (auto_adjust=True provides Close).")

    close = price_df["Close"].astype(float)
    r = np.log(close).diff()
    r.index = pd.to_datetime(price_df["Date"])
    return r.dropna()

def rolling_features(r: pd.Series, h: int) -> pd.DataFrame:
    mu = r.rolling(h).mean()
    vol = r.rolling(h).std()
    out = pd.DataFrame({"mu": mu, "vol": vol}).dropna()
    return out
