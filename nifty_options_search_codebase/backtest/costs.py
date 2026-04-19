"""
backtest/costs.py – Transaction cost and liquidity model.

Provides:
  • Premium-based transaction cost (configurable bps)
  • Liquidity penalty based on volume / open interest
  • Combined cost function
"""

import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config


def premium_cost(premium: float, lots: int = 1) -> float:
    """
    Transaction cost as a fraction of the premium.

    Cost = premium × lots × lot_size × (cost_bps / 10000)
    """
    return abs(premium) * lots * config.NIFTY_LOT_SIZE * (config.TRANSACTION_COST_BPS / 10000)


def liquidity_penalty(volume: float, oi: float, premium: float,
                      lots: int = 1) -> float:
    """
    Extra cost for low-liquidity contracts.

    Penalty is proportional to the inverse of the liquidity ratio and
    applied per lot.
    """
    if volume <= 0 or oi <= 0:
        return abs(premium) * lots * config.NIFTY_LOT_SIZE * 0.01  # 1% penalty

    # Liquidity ratio: higher is better
    liq_ratio = min(volume / config.MIN_VOLUME_FILTER, 1.0)
    penalty_factor = (1.0 - liq_ratio) * (config.LIQUIDITY_PENALTY_BPS / 10000)

    return abs(premium) * lots * config.NIFTY_LOT_SIZE * penalty_factor


def total_cost(premiums: list, volumes: list, ois: list,
               lots: int = 1) -> float:
    """
    Total transaction + liquidity cost for a multi-leg position.

    Parameters
    ----------
    premiums : list of float – per-leg premiums
    volumes  : list of float – per-leg daily volumes
    ois      : list of float – per-leg open interests
    lots     : int – number of lots

    Returns
    -------
    float – total cost in index points × lot_size
    """
    cost = 0.0
    for prem, vol, oi in zip(premiums, volumes, ois):
        cost += premium_cost(prem, lots)
        cost += liquidity_penalty(vol, oi, prem, lots)
    return cost


def passes_liquidity_filter(volume: float, oi: float) -> bool:
    """Check if a contract passes the minimum liquidity filter."""
    return volume >= config.MIN_VOLUME_FILTER and oi >= config.MIN_OI_FILTER
