from src.config import CFG
from src.data_loader import load_prices
from src.features import compute_log_returns, rolling_features
from src.regime import infer_regime
from src.plot import plot_regime_horizon_map

def main():
    prices = load_prices(
        CFG.ticker,
        CFG.start,
        CFG.end,
        CFG.cache_dir,
        CFG.data_source
    )
    r = compute_log_returns(prices)

    panels = []
    for h in CFG.horizons:
        f = rolling_features(r, h)
        regime = infer_regime(f)
        panels.append({
            "label": f"{h}D",
            "index": f.index,
            "mu": f["mu"],
            "vol": f["vol"],
            "regime": regime
        })

    plot_regime_horizon_map(panels, out_path=CFG.figure_path)
    print(f"[OK] Saved: {CFG.figure_path}")

if __name__ == "__main__":
    main()
