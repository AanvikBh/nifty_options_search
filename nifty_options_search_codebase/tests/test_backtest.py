"""
tests/test_backtest.py – Unit tests for the backtesting engine and metrics.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from backtest.metrics import compute_metrics, BacktestMetrics
from backtest.costs import premium_cost, liquidity_penalty, total_cost, passes_liquidity_filter


class TestMetrics:
    def test_empty_pnl(self):
        m = compute_metrics([])
        assert m.total_return == 0.0
        assert m.sharpe_ratio == 0.0

    def test_all_positive(self):
        pnl = [10.0] * 100
        m = compute_metrics(pnl)
        assert m.total_return == 1000.0
        assert m.avg_return == 10.0
        assert m.max_drawdown == 0.0
        assert m.hit_rate == 1.0

    def test_all_negative(self):
        pnl = [-5.0] * 50
        m = compute_metrics(pnl)
        assert m.total_return == -250.0
        assert m.hit_rate == 0.0
        assert m.max_drawdown > 0

    def test_mixed_pnl(self):
        pnl = [10, -5, 10, -5, 10, -5]
        m = compute_metrics(pnl)
        assert m.total_return == 15.0
        assert 0 < m.hit_rate < 1

    def test_sharpe_positive_for_positive_expectation(self):
        rng = np.random.default_rng(42)
        pnl = list(rng.normal(1.0, 2.0, 252))
        m = compute_metrics(pnl)
        assert m.sharpe_ratio > 0

    def test_trade_returns_hit_rate(self):
        pnl = [0.0] * 10
        trade_returns = [10, -5, 20, -3, 15]
        m = compute_metrics(pnl, trade_returns)
        assert m.hit_rate == 0.6
        assert m.n_trades == 5

    def test_to_dict(self):
        m = compute_metrics([1.0, -0.5, 2.0])
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d


class TestCosts:
    def test_premium_cost(self):
        cost = premium_cost(100.0, lots=1)
        assert cost > 0
        # 100 * 1 * 25 * 5/10000 = 1.25
        assert abs(cost - 1.25) < 0.01

    def test_zero_premium(self):
        cost = premium_cost(0.0, lots=1)
        assert cost == 0.0

    def test_liquidity_penalty_high_volume(self):
        # High volume should give low penalty
        penalty = liquidity_penalty(10000, 50000, 100.0)
        assert penalty >= 0
        assert penalty < 1.0  # should be very small

    def test_liquidity_penalty_zero_volume(self):
        # Zero volume should give maximum penalty
        penalty = liquidity_penalty(0, 0, 100.0)
        assert penalty > 0

    def test_total_cost(self):
        cost = total_cost([100.0, 50.0], [5000, 3000], [10000, 8000])
        assert cost > 0

    def test_liquidity_filter(self):
        assert passes_liquidity_filter(200, 1000) is True
        assert passes_liquidity_filter(50, 1000) is False
        assert passes_liquidity_filter(200, 100) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
