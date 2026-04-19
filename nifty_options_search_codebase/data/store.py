"""
data/store.py – Versioned local data store backed by Parquet files.

Provides a simple API to persist and load the frozen dataset:
  • save_dataset(index_df, vix_df, options_df)
  • load_index(), load_vix(), load_options()
"""

import pathlib
import pandas as pd
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config


def _ensure_dir():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_dataset(index_df: pd.DataFrame, vix_df: pd.DataFrame,
                 options_df: pd.DataFrame) -> None:
    """Persist all three DataFrames to Parquet."""
    _ensure_dir()
    index_df.to_parquet(config.DATA_DIR / "index.parquet", index=False)
    vix_df.to_parquet(config.DATA_DIR / "vix.parquet", index=False)
    options_df.to_parquet(config.DATA_DIR / "options.parquet", index=False)
    print(f"[store] Dataset saved to {config.DATA_DIR}")


def load_index() -> pd.DataFrame:
    path = config.DATA_DIR / "index.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Index data not found at {path}. Run data generation first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_vix() -> pd.DataFrame:
    path = config.DATA_DIR / "vix.parquet"
    if not path.exists():
        raise FileNotFoundError(f"VIX data not found at {path}. Run data generation first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_options() -> pd.DataFrame:
    path = config.DATA_DIR / "options.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Options data not found at {path}. Run data generation first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def dataset_exists() -> bool:
    """Check whether the frozen dataset has already been generated."""
    return all(
        (config.DATA_DIR / f).exists()
        for f in ["index.parquet", "vix.parquet", "options.parquet"]
    )


def dataset_summary() -> dict:
    """Return quick statistics about the stored dataset."""
    idx = load_index()
    vix = load_vix()
    opts = load_options()
    return {
        "index_rows": len(idx),
        "index_date_range": (str(idx["date"].min().date()), str(idx["date"].max().date())),
        "vix_rows": len(vix),
        "option_rows": len(opts),
        "unique_expiries": opts["expiry"].nunique(),
        "unique_strikes": opts["strike"].nunique(),
    }
