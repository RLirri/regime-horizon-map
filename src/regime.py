import pandas as pd

def infer_regime(features: pd.DataFrame) -> pd.Series:
    """
    Regime labels:
      +1: positive drift, lower volatility
      -1: negative drift, lower volatility
       0: high uncertainty (high volatility)
    """
    vol_thr = features["vol"].median()
    high_unc = features["vol"] >= vol_thr

    regime = pd.Series(index=features.index, dtype=int)
    regime[high_unc] = 0
    regime[~high_unc & (features["mu"] >= 0)] = 1
    regime[~high_unc & (features["mu"] < 0)] = -1
    return regime
