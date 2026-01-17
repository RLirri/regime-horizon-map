from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    ticker: str = "SPY"
    start: str = "2018-01-01"
    end: str = "2026-01-01"
    horizons: tuple = (5, 20, 60)
    data_source: str = "stooq"

    cache_dir: Path = Path("data")
    figure_path: Path = Path("outputs/regime_horizon_map_output1.png")

CFG = Config()
