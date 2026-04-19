"""
backtest/engine.py – Core backtesting engine.

Walk-forward backtest that:
  1. Iterates day-by-day through the dataset.
  2. Evaluates strategy entry conditions using data available up to day t.
  3. On signal, selects contracts for the nearest weekly expiry.
  4. Enters at day t+1 EOD prices.
  5. Exits after the hold period at EOD prices.
  6. Applies transaction costs and liquidity penalties.
  7. Returns daily PnL and per-trade records.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from strategy.grammar import StrategyRule
from strategy.templates import get_template, resolve_strikes
from backtest.costs import total_cost, passes_liquidity_filter
from backtest.metrics import compute_metrics, BacktestMetrics


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: str
    exit_date: str
    template: str
    legs: List[dict]
    entry_cost: float        # total premium paid/received
    exit_value: float        # total premium at exit
    transaction_cost: float
    pnl: float               # net PnL after costs
    hold_days: int


@dataclass
class BacktestResult:
    """Full backtest result for one strategy candidate."""
    rule: StrategyRule
    metrics: BacktestMetrics
    daily_pnl: List[float]
    trades: List[Trade]
    n_signals: int = 0
    n_traded: int = 0
    n_skipped_liquidity: int = 0


def _find_contracts(options_df: pd.DataFrame, date, spot: float,
                    template_name: str, target_dte: int) -> Optional[pd.DataFrame]:
    """
    Find the option contracts needed for a template on a given date.

    Returns a DataFrame with one row per leg, or None if contracts unavailable.
    """
    day_opts = options_df[options_df["date"] == date]
    if len(day_opts) == 0:
        return None

    # Find expiry closest to target DTE
    day_opts = day_opts.copy()
    day_opts["dte_diff"] = abs(day_opts["dte"] - target_dte)
    eligible = day_opts[(day_opts["dte"] >= config.DTE_MIN) &
                        (day_opts["dte"] <= config.DTE_MAX)]

    if len(eligible) == 0:
        return None

    best_expiry = eligible.loc[eligible["dte_diff"].idxmin(), "expiry"]
    exp_opts = eligible[eligible["expiry"] == best_expiry]

    # ATM strike
    atm_strike = exp_opts.iloc[
        (exp_opts["strike"] - spot).abs().argsort()[:1]
    ]["strike"].values[0]

    # Resolve template legs
    template = get_template(template_name)
    legs = resolve_strikes(template, atm_strike, config.STRIKE_STEP)

    # Find matching contracts
    leg_rows = []
    for option_type, strike, direction in legs:
        match = exp_opts[
            (exp_opts["strike"] == strike) &
            (exp_opts["option_type"] == option_type)
        ]
        if len(match) == 0:
            return None
        row = match.iloc[0].to_dict()
        row["direction"] = direction
        leg_rows.append(row)

    return pd.DataFrame(leg_rows)


def _get_exit_prices(options_df: pd.DataFrame, entry_legs: pd.DataFrame,
                     exit_date) -> Optional[List[float]]:
    """Get closing prices for the same contracts on the exit date."""
    exit_day = options_df[options_df["date"] == exit_date]
    if len(exit_day) == 0:
        return None

    prices = []
    for _, leg in entry_legs.iterrows():
        match = exit_day[
            (exit_day["expiry"] == leg["expiry"]) &
            (exit_day["strike"] == leg["strike"]) &
            (exit_day["option_type"] == leg["option_type"])
        ]
        if len(match) == 0:
            # Contract may have expired – use intrinsic or 0
            prices.append(0.0)
        else:
            prices.append(float(match.iloc[0]["close"]))

    return prices


def run_backtest(rule: StrategyRule,
                 features_df: pd.DataFrame,
                 index_df: pd.DataFrame,
                 options_df: pd.DataFrame,
                 start_date=None,
                 end_date=None) -> BacktestResult:
    """
    Run a walk-forward backtest for a single strategy rule.

    Parameters
    ----------
    rule : StrategyRule – the candidate to evaluate
    features_df : DataFrame – daily feature matrix [date, feature1, ...]
    index_df : DataFrame – [date, close, ret1]
    options_df : DataFrame – full option contract data
    start_date, end_date : optional date bounds

    Returns
    -------
    BacktestResult
    """
    # Prepare date-indexed lookups
    feat = features_df.copy()
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.set_index("date").sort_index()

    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date").sort_index()

    opts = options_df.copy()
    opts["date"] = pd.to_datetime(opts["date"])
    opts["expiry"] = pd.to_datetime(opts["expiry"])

    dates = feat.index.sort_values()
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]

    daily_pnl = []
    trades = []
    n_signals = 0
    n_traded = 0
    n_skipped = 0

    # Track open positions to avoid overlapping trades
    position_exit_date = None
    i = 0

    while i < len(dates):
        dt = dates[i]
        daily_pnl_val = 0.0

        # Skip if we're already in a position
        if position_exit_date is not None and dt <= position_exit_date:
            daily_pnl.append(0.0)
            i += 1
            continue

        position_exit_date = None

        # Evaluate entry conditions
        if dt not in feat.index:
            daily_pnl.append(0.0)
            i += 1
            continue

        feature_row = feat.loc[dt].to_dict()
        signal = rule.evaluate_conditions(feature_row)

        if not signal:
            daily_pnl.append(0.0)
            i += 1
            continue

        n_signals += 1

        # Entry is on t+1
        entry_idx = i + 1
        if entry_idx >= len(dates):
            daily_pnl.append(0.0)
            i += 1
            continue

        entry_date = dates[entry_idx]

        if entry_date not in idx.index:
            daily_pnl.append(0.0)
            i += 1
            continue

        spot = float(idx.loc[entry_date, "close"])

        # Find contracts
        entry_legs = _find_contracts(opts, entry_date, spot,
                                     rule.template, rule.dte)
        if entry_legs is None:
            daily_pnl.append(0.0)
            i += 1
            continue

        # Liquidity filter
        all_liquid = all(
            passes_liquidity_filter(
                float(entry_legs.iloc[j]["volume"]),
                float(entry_legs.iloc[j]["oi"])
            )
            for j in range(len(entry_legs))
        )
        if not all_liquid:
            n_skipped += 1
            daily_pnl.append(0.0)
            i += 1
            continue

        # Determine exit date
        exit_idx = min(entry_idx + rule.hold_period, len(dates) - 1)
        exit_date = dates[exit_idx]

        # Get exit prices
        exit_prices = _get_exit_prices(opts, entry_legs, exit_date)
        if exit_prices is None:
            daily_pnl.append(0.0)
            i += 1
            continue

        # Calculate PnL
        entry_premiums = [float(entry_legs.iloc[j]["close"]) for j in range(len(entry_legs))]
        directions = [int(entry_legs.iloc[j]["direction"]) for j in range(len(entry_legs))]
        volumes = [float(entry_legs.iloc[j]["volume"]) for j in range(len(entry_legs))]
        ois = [float(entry_legs.iloc[j]["oi"]) for j in range(len(entry_legs))]

        # Entry cost: sum of direction × premium × lot_size
        entry_cost = sum(
            d * p * config.NIFTY_LOT_SIZE
            for d, p in zip(directions, entry_premiums)
        )

        # Exit value: sum of direction × exit_price × lot_size
        exit_value = sum(
            d * p * config.NIFTY_LOT_SIZE
            for d, p in zip(directions, exit_prices)
        )

        # PnL = exit_value - entry_cost (for long positions)
        # For a bought option: PnL = (exit_price - entry_price) × lot_size
        # For a sold option:  PnL = (entry_price - exit_price) × lot_size
        raw_pnl = exit_value - entry_cost

        # Transaction costs (entry + exit)
        tc = total_cost(entry_premiums, volumes, ois)
        tc += total_cost(exit_prices, volumes, ois)

        net_pnl = raw_pnl - tc

        trade = Trade(
            entry_date=str(entry_date.date()),
            exit_date=str(exit_date.date()),
            template=rule.template,
            legs=[{
                "type": entry_legs.iloc[j]["option_type"],
                "strike": float(entry_legs.iloc[j]["strike"]),
                "direction": directions[j],
                "entry_price": entry_premiums[j],
                "exit_price": exit_prices[j],
            } for j in range(len(entry_legs))],
            entry_cost=round(entry_cost, 2),
            exit_value=round(exit_value, 2),
            transaction_cost=round(tc, 2),
            pnl=round(net_pnl, 2),
            hold_days=rule.hold_period,
        )
        trades.append(trade)
        n_traded += 1

        # Record PnL on exit date
        daily_pnl.append(net_pnl)

        # Mark position until exit
        position_exit_date = exit_date
        i += 1
        continue

    # Compute metrics
    trade_returns = [t.pnl for t in trades]
    metrics = compute_metrics(daily_pnl, trade_returns)

    return BacktestResult(
        rule=rule,
        metrics=metrics,
        daily_pnl=daily_pnl,
        trades=trades,
        n_signals=n_signals,
        n_traded=n_traded,
        n_skipped_liquidity=n_skipped,
    )
