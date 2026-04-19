"""
controller/trust.py – Bandit-style trust and budget allocation.

Implements the trust mechanism from the proposal:
  • UCB1 bandit for budget allocation across proposal sources
  • Minimum exploration floor (no source gets < 10%)
  • Similarity-aware success discounting
  • Thompson Sampling alternative

The controller learns which proposal source (LLM, mutation, diversity)
produces better strategies and allocates more budget accordingly.
"""

import math
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass, field

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config


SOURCES = ["llm", "mutation", "diversity"]


@dataclass
class SourceStats:
    """Statistics for one proposal source."""
    n_total: int = 0
    n_success: int = 0         # positive composite score
    total_score: float = 0.0
    best_score: float = -999.0
    n_unique: int = 0           # unique fingerprints (for similarity discount)

    @property
    def hit_rate(self) -> float:
        return self.n_success / max(self.n_total, 1)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.n_total, 1)


class TrustController:
    """
    Bandit-style budget allocation across proposal sources.

    Uses UCB1 to balance exploitation (sources with high hit rates)
    and exploration (under-tried sources).
    """

    def __init__(self, trust_config: config.TrustConfig = None):
        if trust_config is None:
            trust_config = config.TRUST_CONFIG

        self.config = trust_config
        self.stats: Dict[str, SourceStats] = {s: SourceStats() for s in SOURCES}
        self.round_num = 0

        # Initial weights
        self.weights = dict(self.config.initial_weights)

    def update(self, source: str, score: float, fingerprint: str,
               existing_fingerprints: set = None):
        """
        Update stats after evaluating one candidate.

        Parameters
        ----------
        source : str – "llm", "mutation", or "diversity"
        score : float – composite score
        fingerprint : str – rule fingerprint
        existing_fingerprints : set – all fingerprints seen so far
        """
        if source not in self.stats:
            self.stats[source] = SourceStats()

        s = self.stats[source]
        s.n_total += 1
        s.total_score += score
        s.best_score = max(s.best_score, score)

        # Similarity-aware success counting
        is_unique = True
        if existing_fingerprints and fingerprint in existing_fingerprints:
            is_unique = False

        if score > 0:
            if is_unique:
                s.n_success += 1
            else:
                # Discounted success for near-duplicate
                s.n_success += self.config.similarity_discount
            s.n_unique += 1

    def allocate_budget(self, total_budget: int) -> Dict[str, int]:
        """
        Allocate the next round's budget across sources using UCB1.

        Parameters
        ----------
        total_budget : int – total candidates for this round

        Returns
        -------
        dict mapping source -> number of candidates
        """
        self.round_num += 1
        total_pulls = sum(s.n_total for s in self.stats.values())

        if total_pulls == 0:
            # First round: use initial weights
            alloc = {}
            for src in SOURCES:
                alloc[src] = max(1, int(total_budget * self.weights[src]))
            # Distribute remainder
            remainder = total_budget - sum(alloc.values())
            if remainder > 0:
                alloc[SOURCES[0]] += remainder
            return alloc

        # UCB1 scores
        ucb_scores = {}
        for src in SOURCES:
            s = self.stats[src]
            if s.n_total == 0:
                ucb_scores[src] = float('inf')  # force exploration
            else:
                exploit = s.avg_score
                explore = self.config.ucb_exploration_c * math.sqrt(
                    math.log(total_pulls) / s.n_total
                )
                ucb_scores[src] = exploit + explore

        # Normalise UCB scores to weights, handling inf values
        inf_sources = [s for s in SOURCES if math.isinf(ucb_scores[s])]
        finite_sources = [s for s in SOURCES if not math.isinf(ucb_scores[s])]

        if inf_sources:
            # Give untried sources equal large share to force exploration
            inf_weight = 0.8 / len(inf_sources) if len(inf_sources) < len(SOURCES) else 1.0 / len(SOURCES)
            finite_budget = 1.0 - inf_weight * len(inf_sources)
            weights = {}
            for src in inf_sources:
                weights[src] = inf_weight
            if finite_sources:
                finite_total = sum(max(ucb_scores[s], 0.01) for s in finite_sources)
                for src in finite_sources:
                    weights[src] = finite_budget * max(ucb_scores[src], 0.01) / max(finite_total, 0.01)
        else:
            total_ucb = sum(max(v, 0.01) for v in ucb_scores.values())
            weights = {
                src: max(ucb_scores[src], 0.01) / total_ucb
                for src in SOURCES
            }

        # Apply exploration floor
        floor = self.config.min_exploration_floor
        for src in SOURCES:
            if weights[src] < floor:
                # Redistribute from others
                deficit = floor - weights[src]
                weights[src] = floor
                # Proportionally reduce others
                others = [s for s in SOURCES if s != src]
                other_total = sum(weights[s] for s in others)
                for o in others:
                    weights[o] -= deficit * (weights[o] / max(other_total, 0.01))

        # Allocate
        alloc = {}
        for src in SOURCES:
            alloc[src] = max(1, int(total_budget * weights[src]))

        # Fix rounding
        remainder = total_budget - sum(alloc.values())
        if remainder > 0:
            # Give to the highest-weighted source
            best_src = max(weights, key=weights.get)
            alloc[best_src] += remainder
        elif remainder < 0:
            # Take from the lowest-weighted source
            worst_src = min(weights, key=weights.get)
            alloc[worst_src] = max(1, alloc[worst_src] + remainder)

        self.weights = weights
        return alloc

    def get_weights(self) -> Dict[str, float]:
        """Return current normalised weights."""
        return dict(self.weights)

    def get_stats(self) -> Dict[str, dict]:
        """Return per-source statistics."""
        return {
            src: {
                "n_total": s.n_total,
                "n_success": s.n_success,
                "hit_rate": round(s.hit_rate, 4),
                "avg_score": round(s.avg_score, 4),
                "best_score": round(s.best_score, 4),
            }
            for src, s in self.stats.items()
        }

    def summary(self) -> str:
        """Pretty-print trust state."""
        lines = [f"[Trust Controller – Round {self.round_num}]"]
        for src in SOURCES:
            s = self.stats[src]
            w = self.weights.get(src, 0)
            lines.append(
                f"  {src:12s}: weight={w:.2%}  "
                f"hit_rate={s.hit_rate:.2%}  "
                f"avg_score={s.avg_score:.4f}  "
                f"n={s.n_total}"
            )
        return "\n".join(lines)
