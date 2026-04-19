"""
backtest/metrics.py – Performance metrics for backtested strategies.

Computes:
  • Total and average return
  • Sharpe ratio (annualised)
  • Maximum drawdown
  • Hit rate (fraction of profitable trades)
  • Turnover
  • Stability across sub-periods (rolling Sharpe)
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class BacktestMetrics:
    """Container for all backtest performance metrics."""
    total_return: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    n_trades: int = 0
    turnover: float = 0.0
    stability: float = 0.0           # rolling Sharpe stability
    avg_holding_days: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(pnl_series: List[float],
                    trade_returns: List[float] = None) -> BacktestMetrics:
    """
    Compute backtest metrics from PnL data.

    Parameters
    ----------
    pnl_series : list of float – daily portfolio PnL
    trade_returns : list of float – per-trade returns (optional, for hit rate)

    Returns
    -------
    BacktestMetrics
    """
    pnl = np.array(pnl_series, dtype=float)

    if len(pnl) == 0:
        return BacktestMetrics()

    # Total return
    total_return = float(np.sum(pnl))

    # Average daily return
    avg_return = float(np.mean(pnl))

    # Sharpe ratio (annualised, assume 252 trading days)
    std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 1e-10
    sharpe = (avg_return / max(std, 1e-10)) * np.sqrt(252)

    # Maximum drawdown
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Hit rate (from trade returns)
    if trade_returns and len(trade_returns) > 0:
        tr = np.array(trade_returns)
        hit_rate = float(np.mean(tr > 0))
        n_trades = len(trade_returns)
    else:
        hit_rate = float(np.mean(pnl > 0)) if len(pnl) > 0 else 0.0
        n_trades = int(np.sum(pnl != 0))

    # Turnover (fraction of days with trades)
    turnover = float(np.mean(pnl != 0))

    # Stability: std of rolling Sharpe (lower = more stable)
    window = min(63, len(pnl) // 2)  # ~quarter
    if window > 5:
        rolling_means = np.convolve(pnl, np.ones(window)/window, mode='valid')
        rolling_stds = np.array([
            np.std(pnl[i:i+window], ddof=1)
            for i in range(len(pnl) - window + 1)
        ])
        rolling_stds = np.maximum(rolling_stds, 1e-10)
        rolling_sharpes = (rolling_means / rolling_stds) * np.sqrt(252)
        stability = 1.0 / (1.0 + float(np.std(rolling_sharpes)))
    else:
        stability = 0.5

    return BacktestMetrics(
        total_return=round(total_return, 2),
        avg_return=round(avg_return, 4),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd, 2),
        hit_rate=round(hit_rate, 4),
        n_trades=n_trades,
        turnover=round(turnover, 4),
        stability=round(stability, 4),
    )
