"""
data/downloader.py – Synthetic NIFTY option data generator.

Generates realistic end-of-day data for:
  • NIFTY index prices  (GBM simulation)
  • NIFTY option contracts  (BS-priced, with realistic OI / volume)
  • India VIX  (mean-reverting process)

Also provides a hook for loading real NSE bhavcopy CSV data.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config


# ── helpers ──────────────────────────────────────────────────────────────────

def _trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Return business-day calendar between *start* and *end*."""
    return pd.bdate_range(start, end, freq="B")


def _bs_price(S, K, T, r, sigma, option_type="CE"):
    """Black-Scholes European option price (vectorised)."""
    from scipy.stats import norm

    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ── synthetic data generation ────────────────────────────────────────────────

def generate_index_data(seed: int = 42) -> pd.DataFrame:
    """
    Simulate NIFTY 50 daily close prices via Geometric Brownian Motion.

    Returns DataFrame with columns: [date, close, ret1]
    """
    rng = np.random.default_rng(seed)
    dates = _trading_days(config.DATA_START, config.DATA_END)
    n = len(dates)

    dt = 1 / 252
    log_returns = (
        (config.SYNTH_ANNUAL_DRIFT - 0.5 * config.SYNTH_ANNUAL_VOL**2) * dt
        + config.SYNTH_ANNUAL_VOL * np.sqrt(dt) * rng.standard_normal(n)
    )
    log_returns[0] = 0.0
    prices = config.SYNTH_INITIAL_SPOT * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame({"date": dates, "close": np.round(prices, 2)})
    df["ret1"] = df["close"].pct_change()
    return df


def generate_vix_data(seed: int = 43) -> pd.DataFrame:
    """
    Simulate India VIX as a mean-reverting (Ornstein-Uhlenbeck) process.

    Returns DataFrame with columns: [date, vix]
    """
    rng = np.random.default_rng(seed)
    dates = _trading_days(config.DATA_START, config.DATA_END)
    n = len(dates)

    kappa = 5.0        # mean-reversion speed
    theta = config.SYNTH_VIX_MEAN
    sigma = config.SYNTH_VIX_STD
    dt = 1 / 252

    vix = np.empty(n)
    vix[0] = theta
    for i in range(1, n):
        dW = rng.standard_normal()
        vix[i] = vix[i - 1] + kappa * (theta - vix[i - 1]) * dt + sigma * np.sqrt(dt) * dW
        vix[i] = max(vix[i], 5.0)  # VIX floor

    return pd.DataFrame({"date": dates, "vix": np.round(vix, 2)})


def _weekly_expiries(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Generate weekly Thursday expiries covering the date range.
    NSE NIFTY weekly options expire every Thursday.
    """
    start = dates.min() - timedelta(days=7)
    end = dates.max() + timedelta(days=14)
    all_days = pd.date_range(start, end, freq="B")
    thursdays = all_days[all_days.weekday == 3]
    return thursdays


def generate_option_data(index_df: pd.DataFrame, vix_df: pd.DataFrame,
                         seed: int = 44) -> pd.DataFrame:
    """
    Generate synthetic NIFTY option contract EOD data.

    For each trading day, creates option contracts for the nearest 2-3 weekly
    expiries with strikes around ATM (±5 strikes = ±250 points).

    Returns DataFrame with columns:
        [date, expiry, strike, option_type, close, volume, oi, dte]
    """
    rng = np.random.default_rng(seed)
    expiries = _weekly_expiries(index_df["date"])

    rows = []
    for _, row in index_df.iterrows():
        dt_date = row["date"]
        spot = row["close"]
        vix_row = vix_df.loc[vix_df["date"] == dt_date]
        iv_base = (vix_row["vix"].values[0] / 100) if len(vix_row) > 0 else 0.15

        # ATM strike (round to nearest STRIKE_STEP)
        atm = int(round(spot / config.STRIKE_STEP) * config.STRIKE_STEP)

        # strikes: ATM ± 5 steps
        strikes = [atm + i * config.STRIKE_STEP for i in range(-5, 6)]

        # relevant expiries: those with 1-15 DTE
        future_expiries = expiries[expiries > dt_date]
        relevant = future_expiries[(future_expiries - dt_date).days <= 15]

        for exp in relevant:
            dte_days = (exp - dt_date).days
            if dte_days <= 0:
                continue
            T = dte_days / 365.0

            for strike in strikes:
                for otype in ["CE", "PE"]:
                    # add IV smile: OTM options have higher IV
                    moneyness = np.log(spot / strike)
                    smile_adj = 0.02 * (moneyness**2) * 100
                    iv = max(iv_base + smile_adj + rng.normal(0, 0.005), 0.05)

                    price = _bs_price(spot, strike, T, config.RISK_FREE_RATE, iv, otype)
                    price = max(round(float(price), 2), 0.05)

                    # synthetic volume & OI – higher near ATM
                    dist = abs(strike - atm) / config.STRIKE_STEP
                    base_vol = max(5000 - dist * 800, 200)
                    vol = int(base_vol * (1 + rng.uniform(-0.3, 0.3)))
                    oi = int(vol * rng.uniform(3, 10))

                    rows.append({
                        "date": dt_date,
                        "expiry": exp,
                        "strike": strike,
                        "option_type": otype,
                        "close": price,
                        "volume": vol,
                        "oi": oi,
                        "dte": dte_days,
                    })

    df = pd.DataFrame(rows)
    return df


# ── real data loading hooks ──────────────────────────────────────────────────

def load_real_index_csv(path: str) -> pd.DataFrame:
    """Load real NIFTY index data from a CSV.  Expected columns: Date, Close."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", "Close": "close"})
    df["ret1"] = df["close"].pct_change()
    return df.sort_values("date").reset_index(drop=True)


def load_real_vix_csv(path: str) -> pd.DataFrame:
    """Load real India VIX data from a CSV.  Expected columns: Date, Close."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", "Close": "vix"})
    return df.sort_values("date").reset_index(drop=True)


def load_real_options_csv(path: str) -> pd.DataFrame:
    """
    Load real NSE bhavcopy option data from a CSV.
    Expected columns: date, expiry, strike, option_type, close, volume, oi
    """
    df = pd.read_csv(path, parse_dates=["date", "expiry"])
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    return df.sort_values(["date", "expiry", "strike"]).reset_index(drop=True)
