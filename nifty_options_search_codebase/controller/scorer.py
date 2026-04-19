"""
controller/scorer.py – Composite strategy scoring.

Implements the robust performance score from the proposal:
    R = Sharpe_OOS − λ_dd × Drawdown − λ_to × Turnover − λ_c × Complexity

Also supports multi-objective ranking.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from backtest.metrics import BacktestMetrics
from strategy.grammar import StrategyRule


def composite_score(metrics: BacktestMetrics, rule: StrategyRule,
                    weights: config.ScoringWeights = None) -> float:
    """
    Compute the composite robust performance score.

    R = Sharpe − λ_dd × MaxDrawdown/1000 − λ_to × Turnover − λ_c × Complexity

    Drawdown is normalised by 1000 to be on a comparable scale.
    """
    if weights is None:
        weights = config.SCORING_WEIGHTS

    # Normalise drawdown (divide by a scale factor)
    dd_normalised = metrics.max_drawdown / 1000.0

    score = (
        metrics.sharpe_ratio
        - weights.lambda_dd * dd_normalised
        - weights.lambda_to * metrics.turnover
        - weights.lambda_c * rule.complexity()
    )
    return round(score, 6)


def rank_candidates(results: list) -> list:
    """
    Rank a list of (BacktestResult, score) tuples by score descending.

    Parameters
    ----------
    results : list of (BacktestResult, float)

    Returns
    -------
    list of (BacktestResult, float) sorted by score
    """
    return sorted(results, key=lambda x: x[1], reverse=True)


def multi_objective_dominance(a_metrics: BacktestMetrics,
                               b_metrics: BacktestMetrics) -> int:
    """
    Pareto dominance check.

    Returns:
      +1 if a dominates b
      -1 if b dominates a
       0 if neither dominates
    """
    objectives = [
        (a_metrics.sharpe_ratio, b_metrics.sharpe_ratio, True),      # higher better
        (a_metrics.max_drawdown, b_metrics.max_drawdown, False),     # lower better
        (a_metrics.hit_rate, b_metrics.hit_rate, True),              # higher better
        (a_metrics.stability, b_metrics.stability, True),            # higher better
    ]

    a_better = 0
    b_better = 0

    for a_val, b_val, higher_is_better in objectives:
        if higher_is_better:
            if a_val > b_val:
                a_better += 1
            elif b_val > a_val:
                b_better += 1
        else:
            if a_val < b_val:
                a_better += 1
            elif b_val < a_val:
                b_better += 1

    if a_better > 0 and b_better == 0:
        return 1
    elif b_better > 0 and a_better == 0:
        return -1
    return 0
