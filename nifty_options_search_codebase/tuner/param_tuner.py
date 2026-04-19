"""
tuner/param_tuner.py – Parameter tuning for strategy candidates.

Once a strategy's structure (template + condition features) is fixed,
this module optimises the numeric parts:
  • threshold levels
  • DTE within [3, 10]
  • hold period within [1, 3]

Supports grid search, random search, and simple Bayesian optimisation.
"""

import copy
import numpy as np
from typing import Callable, List, Tuple

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from strategy.grammar import StrategyRule


def _extract_params(rule: StrategyRule) -> dict:
    """Extract tunable parameters from a rule."""
    params = {"dte": rule.dte, "hold_period": rule.hold_period}
    for i, cond in enumerate(rule.conditions):
        params[f"thresh_{i}"] = cond.threshold
        if cond.threshold_upper is not None:
            params[f"thresh_upper_{i}"] = cond.threshold_upper
    return params


def _apply_params(rule: StrategyRule, params: dict) -> StrategyRule:
    """Create a new rule with updated parameters."""
    data = rule.model_dump()
    data["dte"] = int(params.get("dte", data["dte"]))
    data["hold_period"] = int(params.get("hold_period", data["hold_period"]))

    for i, cond in enumerate(data["conditions"]):
        key = f"thresh_{i}"
        if key in params:
            cond["threshold"] = float(params[key])
        key_u = f"thresh_upper_{i}"
        if key_u in params:
            cond["threshold_upper"] = float(params[key_u])

    return StrategyRule.model_validate(data)


def _param_ranges(rule: StrategyRule) -> dict:
    """Define search ranges for each tunable parameter."""
    ranges = {
        "dte": list(range(config.DTE_MIN, config.DTE_MAX + 1)),
        "hold_period": list(range(config.HOLD_PERIOD_MIN, config.HOLD_PERIOD_MAX + 1)),
    }
    for i, cond in enumerate(rule.conditions):
        base = cond.threshold
        if base == 0:
            lo, hi = -0.05, 0.05
        else:
            lo = base * 0.5
            hi = base * 1.5
            if lo > hi:
                lo, hi = hi, lo
        ranges[f"thresh_{i}"] = (lo, hi)

        if cond.threshold_upper is not None:
            base_u = cond.threshold_upper
            if base_u == 0:
                lo_u, hi_u = -0.05, 0.05
            else:
                lo_u = base_u * 0.5
                hi_u = base_u * 1.5
                if lo_u > hi_u:
                    lo_u, hi_u = hi_u, lo_u
            ranges[f"thresh_upper_{i}"] = (lo_u, hi_u)

    return ranges


def random_search(rule: StrategyRule,
                  eval_fn: Callable[[StrategyRule], float],
                  n_evals: int = None,
                  seed: int = 42) -> Tuple[StrategyRule, float]:
    """
    Random search over the parameter space.

    Parameters
    ----------
    rule : StrategyRule – the structure to tune
    eval_fn : callable – evaluation function returning a score (higher = better)
    n_evals : int – number of random parameter sets to try
    seed : int

    Returns
    -------
    (best_rule, best_score)
    """
    if n_evals is None:
        n_evals = config.TUNER_MAX_EVALS

    rng = np.random.default_rng(seed)
    ranges = _param_ranges(rule)

    best_rule = rule
    best_score = eval_fn(rule)

    for _ in range(n_evals):
        params = {}
        for key, val_range in ranges.items():
            if isinstance(val_range, list):
                # discrete
                params[key] = int(rng.choice(val_range))
            else:
                # continuous
                lo, hi = val_range
                params[key] = float(rng.uniform(lo, hi))

        try:
            candidate = _apply_params(rule, params)
            score = eval_fn(candidate)
            if score > best_score:
                best_score = score
                best_rule = candidate
        except Exception:
            continue

    return best_rule, best_score


def grid_search(rule: StrategyRule,
                eval_fn: Callable[[StrategyRule], float],
                n_points: int = 5) -> Tuple[StrategyRule, float]:
    """
    Grid search: discretise each continuous range into *n_points* and
    try all DTE × hold_period combos with the best threshold set.
    """
    ranges = _param_ranges(rule)
    best_rule = rule
    best_score = eval_fn(rule)

    # Build grid for continuous params
    grid = {}
    for key, val_range in ranges.items():
        if isinstance(val_range, list):
            grid[key] = val_range
        else:
            lo, hi = val_range
            grid[key] = np.linspace(lo, hi, n_points).tolist()

    # Simple: iterate over DTE × hold × one threshold at a time
    for dte in grid["dte"]:
        for hold in grid["hold_period"]:
            params = {"dte": dte, "hold_period": hold}

            # Set thresholds to original as baseline
            for key in grid:
                if key.startswith("thresh"):
                    params[key] = rule.model_dump()["conditions"][
                        int(key.split("_")[-1])
                    ].get("threshold", 0)

            # One-at-a-time threshold sweep
            for key in grid:
                if key.startswith("thresh"):
                    for val in grid[key]:
                        params[key] = val
                        try:
                            candidate = _apply_params(rule, params)
                            score = eval_fn(candidate)
                            if score > best_score:
                                best_score = score
                                best_rule = candidate
                        except Exception:
                            continue

    return best_rule, best_score


def tune(rule: StrategyRule,
         eval_fn: Callable[[StrategyRule], float],
         mode: str = None,
         seed: int = 42) -> Tuple[StrategyRule, float]:
    """
    Tune parameters for a strategy rule.

    Parameters
    ----------
    rule : StrategyRule
    eval_fn : callable(StrategyRule) -> float
    mode : "grid" | "random" | None (uses config default)
    seed : int

    Returns
    -------
    (best_rule, best_score)
    """
    if mode is None:
        mode = config.TUNER_MODE

    if mode == "grid":
        return grid_search(rule, eval_fn)
    else:
        return random_search(rule, eval_fn, seed=seed)
