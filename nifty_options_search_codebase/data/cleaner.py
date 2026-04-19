"""
data/cleaner.py – Data cleaning and contract mapping utilities.

Handles:
  • Expiry date normalisation
  • Missing-data imputation / forward-fill
  • Outlier filtering (option price sanity checks)
  • Contract metadata extraction
"""

import pandas as pd
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config


def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Clean index price data: forward-fill gaps, drop NaN tails."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["close"] = df["close"].ffill()
    df = df.dropna(subset=["close"])
    df["ret1"] = df["close"].pct_change()
    return df


def clean_vix(df: pd.DataFrame) -> pd.DataFrame:
    """Clean VIX data: clamp extremes, forward-fill."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["vix"] = df["vix"].ffill()
    df["vix"] = df["vix"].clip(lower=5.0, upper=80.0)
    return df


def clean_options(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean option contract data.

    Steps:
      1. Remove rows with non-positive prices.
      2. Remove contracts with volume < MIN_VOLUME_FILTER (optional, kept for later).
      3. Re-compute DTE from date & expiry columns.
      4. Remove contracts already expired (DTE <= 0).
      5. Basic intrinsic-value sanity: call price >= max(0, S-K·e^{-rT}).
    """
    df = df.copy()

    # basic price filter
    df = df[df["close"] > 0].copy()

    # recompute DTE
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    df = df[df["dte"] > 0].copy()

    # merge spot for intrinsic-value check
    spot_map = index_df.set_index("date")["close"].to_dict()
    df["spot"] = df["date"].map(spot_map)
    df = df.dropna(subset=["spot"])

    # intrinsic-value sanity (allow small tolerance)
    # Compute intrinsic values aligned to the full dataframe index
    intrinsic_call = np.maximum(df["spot"] - df["strike"], 0)
    bad_calls = (df["option_type"] == "CE") & (df["close"] < intrinsic_call * 0.8)

    intrinsic_put = np.maximum(df["strike"] - df["spot"], 0)
    bad_puts = (df["option_type"] == "PE") & (df["close"] < intrinsic_put * 0.8)

    df = df[~(bad_calls | bad_puts)].copy()
    df = df.drop(columns=["spot"], errors="ignore")

    return df.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)


def extract_contract_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a contract metadata table with unique (expiry, strike, option_type)
    combos and their first/last trading dates + average volume.
    """
    meta = (
        df.groupby(["expiry", "strike", "option_type"])
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            avg_volume=("volume", "mean"),
            avg_oi=("oi", "mean"),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )
    return meta
