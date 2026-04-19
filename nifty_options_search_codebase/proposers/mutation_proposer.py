"""
proposers/mutation_proposer.py – Mutation-based strategy proposer.

Takes the top-k strategies from previous rounds and generates variants by
making small, local changes:
  • Nudge thresholds ± 10-20%
  • Swap one feature for another in the same family
  • Change DTE or hold_period by ±1
  • Add or remove one condition
"""

import copy
import numpy as np
from typing import List

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from strategy.grammar import StrategyRule, Condition
from features.registry import FEATURE_REGISTRY, get_family


def _nudge_threshold(value: float, rng: np.random.Generator,
                     scale: float = 0.15) -> float:
    """Randomly adjust a threshold value by ±scale proportion."""
    factor = 1.0 + rng.uniform(-scale, scale)
    return round(value * factor, 4)


def _swap_feature(cond: Condition, rng: np.random.Generator) -> Condition:
    """Replace the feature with another from the same family."""
    family = get_family(cond.feature)
    same_family = [
        name for name, meta in FEATURE_REGISTRY.items()
        if meta.family == family and name != cond.feature
    ]
    if not same_family:
        return cond
    new_feature = rng.choice(same_family)
    return Condition(
        feature=new_feature,
        operator=cond.operator,
        threshold=cond.threshold,
        threshold_upper=cond.threshold_upper,
    )


def _mutate_one(rule: StrategyRule, rng: np.random.Generator) -> StrategyRule:
    """Apply one random mutation to a strategy rule."""
    data = rule.model_dump()
    mutation_type = rng.choice([
        "nudge_threshold", "swap_feature", "change_dte",
        "change_hold", "toggle_condition"
    ])

    if mutation_type == "nudge_threshold" and data["conditions"]:
        idx = int(rng.integers(0, len(data["conditions"])))
        data["conditions"][idx]["threshold"] = _nudge_threshold(
            data["conditions"][idx]["threshold"], rng
        )
        if data["conditions"][idx].get("threshold_upper") is not None:
            data["conditions"][idx]["threshold_upper"] = _nudge_threshold(
                data["conditions"][idx]["threshold_upper"], rng
            )

    elif mutation_type == "swap_feature" and data["conditions"]:
        idx = int(rng.integers(0, len(data["conditions"])))
        old_cond = Condition(**data["conditions"][idx])
        new_cond = _swap_feature(old_cond, rng)
        data["conditions"][idx] = new_cond.model_dump()

    elif mutation_type == "change_dte":
        delta = int(rng.choice([-1, 1]))
        new_dte = max(config.DTE_MIN, min(config.DTE_MAX, data["dte"] + delta))
        data["dte"] = new_dte

    elif mutation_type == "change_hold":
        delta = int(rng.choice([-1, 1]))
        new_hold = max(config.HOLD_PERIOD_MIN,
                       min(config.HOLD_PERIOD_MAX, data["hold_period"] + delta))
        data["hold_period"] = new_hold

    elif mutation_type == "toggle_condition":
        if len(data["conditions"]) > 1 and rng.random() < 0.5:
            # Remove a random condition
            idx = int(rng.integers(0, len(data["conditions"])))
            data["conditions"].pop(idx)
        elif len(data["conditions"]) < config.MAX_CONDITIONS:
            # Add a new random condition
            feature = rng.choice(config.APPROVED_FEATURES)
            op = rng.choice([">", "<"])
            thresh = round(float(rng.uniform(-0.1, 0.3)), 4)
            data["conditions"].append({
                "feature": feature,
                "operator": op,
                "threshold": thresh,
                "threshold_upper": None,
            })

    try:
        return StrategyRule.model_validate(data)
    except Exception:
        return rule  # return original if mutation produced invalid state


def propose(n: int,
            top_strategies: List[StrategyRule],
            seed: int = 42) -> List[StrategyRule]:
    """
    Generate *n* mutated strategy candidates from top past strategies.

    Parameters
    ----------
    n : int – number of candidates
    top_strategies : list of StrategyRule – best strategies to mutate
    seed : int

    Returns
    -------
    list of StrategyRule
    """
    rng = np.random.default_rng(seed)
    results = []

    if not top_strategies:
        return results

    for i in range(n):
        parent = top_strategies[int(rng.integers(0, len(top_strategies)))]
        # Apply 1-2 mutations
        child = copy.deepcopy(parent)
        n_mutations = int(rng.integers(1, 3))
        for _ in range(n_mutations):
            child = _mutate_one(child, rng)
        results.append(child)

    return results
