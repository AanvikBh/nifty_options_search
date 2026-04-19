"""
proposers/diversity_proposer.py – Random diversity strategy proposer.

Generates valid but intentionally different candidates to prevent the
search from becoming too narrow.  Ensures coverage of all templates and
avoids near-duplicates.
"""

import numpy as np
from typing import List, Set

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from strategy.grammar import StrategyRule, Condition


def _random_condition(rng: np.random.Generator) -> Condition:
    """Generate one random valid condition."""
    feature = rng.choice(config.APPROVED_FEATURES)
    op = rng.choice([">", "<", ">=", "<="])

    # Sensible threshold ranges per feature type
    if "ret" in feature:
        thresh = round(float(rng.uniform(-0.03, 0.03)), 4)
    elif "rv" in feature or "iv" in feature or "atm_iv" in feature:
        thresh = round(float(rng.uniform(0.05, 0.40)), 4)
    elif "vix" in feature and "change" not in feature:
        thresh = round(float(rng.uniform(10.0, 30.0)), 2)
    elif "change" in feature:
        thresh = round(float(rng.uniform(-0.15, 0.15)), 4)
    elif "volume" in feature:
        thresh = round(float(rng.uniform(500, 8000)), 0)
    elif "oi" in feature:
        thresh = round(float(rng.uniform(-5000, 5000)), 0)
    elif "skew" in feature or "slope" in feature:
        thresh = round(float(rng.uniform(-0.05, 0.05)), 4)
    else:
        thresh = round(float(rng.uniform(-1, 1)), 4)

    return Condition(feature=feature, operator=op, threshold=thresh)


def _random_rule(rng: np.random.Generator,
                 template: str = None) -> StrategyRule:
    """Generate one random valid strategy rule."""
    if template is None:
        template = rng.choice(config.ALLOWED_TEMPLATES)

    n_conds = int(rng.integers(1, config.MAX_CONDITIONS + 1))
    conditions = [_random_condition(rng) for _ in range(n_conds)]

    # Ensure no duplicate features
    seen = set()
    unique_conds = []
    for c in conditions:
        if c.feature not in seen:
            seen.add(c.feature)
            unique_conds.append(c)
    if not unique_conds:
        unique_conds = [_random_condition(rng)]

    return StrategyRule(
        template=template,
        conditions=unique_conds,
        dte=int(rng.integers(config.DTE_MIN, config.DTE_MAX + 1)),
        hold_period=int(rng.integers(config.HOLD_PERIOD_MIN, config.HOLD_PERIOD_MAX + 1)),
    )


def propose(n: int,
            existing_fingerprints: Set[str] = None,
            seed: int = 42) -> List[StrategyRule]:
    """
    Generate *n* diverse strategy candidates.

    Ensures:
      • Coverage of all templates (round-robin)
      • No near-duplicates (by fingerprint)

    Parameters
    ----------
    n : int – number of candidates
    existing_fingerprints : set of str – fingerprints to avoid
    seed : int

    Returns
    -------
    list of StrategyRule
    """
    rng = np.random.default_rng(seed)
    if existing_fingerprints is None:
        existing_fingerprints = set()

    results = []
    templates_cycle = config.ALLOWED_TEMPLATES.copy()
    attempts = 0
    max_attempts = n * 10

    while len(results) < n and attempts < max_attempts:
        # Round-robin through templates
        template = templates_cycle[len(results) % len(templates_cycle)]
        rule = _random_rule(rng, template)

        # Check for near-duplicate
        fp = rule.fingerprint()
        if fp not in existing_fingerprints:
            existing_fingerprints.add(fp)
            results.append(rule)

        attempts += 1

    return results
