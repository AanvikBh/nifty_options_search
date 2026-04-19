"""
features/engine.py – Feature computation pipeline.

Computes all approved features from the frozen dataset and returns a single
wide DataFrame indexed by trading date.
"""

import numpy as np
import pandas as pd
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config
from features.black_scholes import implied_volatility


def compute_features(index_df: pd.DataFrame,
                     vix_df: pd.DataFrame,
                     options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the daily feature matrix.

    Parameters
    ----------
    index_df  : columns [date, close, ret1]
    vix_df    : columns [date, vix]
    options_df: columns [date, expiry, strike, option_type, close, volume, oi, dte]

    Returns
    -------
    DataFrame with columns [date, <all approved features>]
    """
    idx = index_df.sort_values("date").copy()
    idx = idx.set_index("date")

    # ── Underlying-move features ─────────────────────────────────────────
    idx["nifty_ret1"] = idx["close"].pct_change(1)
    idx["nifty_ret5"] = idx["close"].pct_change(5)

    # ── Realized-risk features ───────────────────────────────────────────
    log_ret = np.log(idx["close"] / idx["close"].shift(1))
    idx["rv5"]  = log_ret.rolling(5).std()  * np.sqrt(252)
    idx["rv20"] = log_ret.rolling(20).std() * np.sqrt(252)

    # ── Option-surface features (per day) ────────────────────────────────
    atm_iv_series = []
    skew_series = []
    term_slope_series = []
    oi_change_series = []
    volume_series = []

    dates = idx.index.sort_values()
    prev_oi = {}

    for dt in dates:
        spot = idx.loc[dt, "close"]
        day_opts = options_df[options_df["date"] == dt].copy()

        if len(day_opts) == 0:
            atm_iv_series.append(np.nan)
            skew_series.append(np.nan)
            term_slope_series.append(np.nan)
            oi_change_series.append(np.nan)
            volume_series.append(np.nan)
            continue

        # nearest weekly expiry with 3-10 DTE
        valid_exp = day_opts[(day_opts["dte"] >= config.DTE_MIN) &
                             (day_opts["dte"] <= config.DTE_MAX)]
        if len(valid_exp) == 0:
            # fall back to nearest expiry
            valid_exp = day_opts[day_opts["dte"] > 0]

        if len(valid_exp) == 0:
            atm_iv_series.append(np.nan)
            skew_series.append(np.nan)
            term_slope_series.append(np.nan)
            oi_change_series.append(np.nan)
            volume_series.append(np.nan)
            continue

        near_exp = valid_exp["expiry"].min()
        near = valid_exp[valid_exp["expiry"] == near_exp]

        # ATM strike
        atm_strike = near.iloc[(near["strike"] - spot).abs().argsort()[:1]]["strike"].values[0]

        # ATM call IV
        atm_call = near[(near["strike"] == atm_strike) & (near["option_type"] == "CE")]
        if len(atm_call) > 0:
            row = atm_call.iloc[0]
            T = row["dte"] / 365.0
            iv_val = implied_volatility(
                row["close"], spot, atm_strike, T, config.RISK_FREE_RATE, "CE"
            )
            atm_iv_series.append(iv_val)
        else:
            atm_iv_series.append(np.nan)

        # Skew proxy: IV(1% OTM put) – IV(ATM call)
        otm_put_strike = atm_strike - config.STRIKE_STEP
        otm_put = near[(near["strike"] == otm_put_strike) & (near["option_type"] == "PE")]
        if len(otm_put) > 0 and len(atm_call) > 0:
            row_put = otm_put.iloc[0]
            T_put = row_put["dte"] / 365.0
            iv_put = implied_volatility(
                row_put["close"], spot, otm_put_strike, T_put,
                config.RISK_FREE_RATE, "PE"
            )
            if not np.isnan(iv_put) and not np.isnan(atm_iv_series[-1]):
                skew_series.append(iv_put - atm_iv_series[-1])
            else:
                skew_series.append(np.nan)
        else:
            skew_series.append(np.nan)

        # Term slope: next expiry ATM IV – near expiry ATM IV
        next_exps = valid_exp[valid_exp["expiry"] > near_exp]
        if len(next_exps) > 0:
            next_exp = next_exps["expiry"].min()
            next_atm = next_exps[
                (next_exps["expiry"] == next_exp) &
                (next_exps["option_type"] == "CE")
            ]
            if len(next_atm) > 0:
                next_row = next_atm.iloc[
                    (next_atm["strike"] - spot).abs().argsort()[:1]
                ].iloc[0]
                T_next = next_row["dte"] / 365.0
                iv_next = implied_volatility(
                    next_row["close"], spot, next_row["strike"], T_next,
                    config.RISK_FREE_RATE, "CE"
                )
                if not np.isnan(iv_next) and not np.isnan(atm_iv_series[-1]):
                    term_slope_series.append(iv_next - atm_iv_series[-1])
                else:
                    term_slope_series.append(np.nan)
            else:
                term_slope_series.append(np.nan)
        else:
            term_slope_series.append(np.nan)

        # OI change & volume for ATM contracts
        atm_contracts = near[near["strike"] == atm_strike]
        total_oi = atm_contracts["oi"].sum()
        total_vol = atm_contracts["volume"].sum()

        dt_key = str(dt)
        if dt_key in prev_oi:
            oi_change_series.append(total_oi - prev_oi[dt_key])
        else:
            oi_change_series.append(0)
        prev_oi[dt_key] = total_oi

        volume_series.append(total_vol)

    idx["atm_iv"] = atm_iv_series
    idx["skew_proxy"] = skew_series
    idx["term_slope"] = term_slope_series
    idx["oi_change"] = oi_change_series
    idx["volume"] = volume_series

    # ── IV minus RV20 ────────────────────────────────────────────────────
    idx["iv_minus_rv20"] = idx["atm_iv"] - idx["rv20"]

    # ── Regime features (VIX) ────────────────────────────────────────────
    vix_s = vix_df.set_index("date")["vix"]
    idx["india_vix"] = vix_s
    idx["india_vix_change"] = vix_s.pct_change()

    # ── Final cleanup ────────────────────────────────────────────────────
    feature_cols = [
        "nifty_ret1", "nifty_ret5", "rv5", "rv20",
        "atm_iv", "iv_minus_rv20", "skew_proxy", "term_slope",
        "oi_change", "volume", "india_vix", "india_vix_change",
    ]
    result = idx[feature_cols].copy()
    result = result.reset_index().rename(columns={"index": "date"})
    result = result.ffill()  # forward-fill any remaining NaNs

    return result
