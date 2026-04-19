"""
proposers/llm_proposer.py – LLM-based strategy proposer using Ollama (Mistral 7B).

Generates structured strategy candidates by prompting a local LLM with:
  • The allowed grammar (Pydantic JSON schema)
  • The approved features and templates
  • Top strategies from previous rounds (for context)

The LLM outputs a JSON object that conforms to StrategyRule.
"""

import json
import logging
from typing import List, Optional

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from strategy.grammar import StrategyRule, Condition

logger = logging.getLogger(__name__)


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quantitative finance assistant that proposes option trading strategy rules.

You MUST return a single JSON object matching this exact schema:
{schema}

CONSTRAINTS:
- template MUST be one of: {templates}
- All features MUST be from: {features}
- operator MUST be one of: {operators}
- dte MUST be an integer between {dte_min} and {dte_max}
- hold_period MUST be an integer between {hold_min} and {hold_max}
- Maximum {max_conds} conditions
- Conditions use AND logic (all must be true to trigger the trade)

RETURN ONLY VALID JSON. No explanation, no markdown, no code blocks.
"""


def _build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        schema=json.dumps(StrategyRule.model_json_schema(), indent=2),
        templates=", ".join(config.ALLOWED_TEMPLATES),
        features=", ".join(config.APPROVED_FEATURES),
        operators=", ".join(config.ALLOWED_OPERATORS),
        dte_min=config.DTE_MIN,
        dte_max=config.DTE_MAX,
        hold_min=config.HOLD_PERIOD_MIN,
        hold_max=config.HOLD_PERIOD_MAX,
        max_conds=config.MAX_CONDITIONS,
    )


def _build_user_prompt(top_strategies: Optional[List[dict]] = None,
                       hint: str = "") -> str:
    parts = []
    if top_strategies:
        parts.append("Here are the best strategies found so far:")
        for i, s in enumerate(top_strategies[:5], 1):
            parts.append(f"  {i}. {json.dumps(s)}")
        parts.append("\nPropose a NEW and DIFFERENT strategy that might perform better.")
    else:
        parts.append(
            "Propose a trading strategy for NIFTY weekly options. "
            "Think about what market conditions (IV, momentum, VIX) would "
            "favour each template."
        )
    if hint:
        parts.append(f"\nHint: {hint}")
    parts.append("\nReturn ONLY the JSON object.")
    return "\n".join(parts)


# ── Ollama integration ───────────────────────────────────────────────────────

def _call_ollama(system: str, user: str) -> Optional[str]:
    """Call local Ollama with Mistral model.  Returns raw response text."""
    try:
        import ollama
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=StrategyRule.model_json_schema(),
            options={"temperature": config.OLLAMA_TEMPERATURE},
        )
        return response["message"]["content"]
    except ImportError:
        logger.warning("ollama package not installed – using fallback proposer")
        return None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e} – using fallback proposer")
        return None


def _parse_response(raw: str) -> Optional[StrategyRule]:
    """Parse LLM response into a StrategyRule."""
    try:
        # Try direct parse
        data = json.loads(raw)
        rule = StrategyRule.model_validate(data)
        return rule
    except (json.JSONDecodeError, Exception) as e:
        # Try extracting JSON from response
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return StrategyRule.model_validate(data)
            except Exception:
                pass
        logger.warning(f"Failed to parse LLM response: {e}")
        return None


# ── Fallback proposer (no LLM required) ─────────────────────────────────────

def _fallback_propose(rng) -> StrategyRule:
    """Generate a reasonable heuristic-based strategy when LLM is unavailable."""
    import random

    template = rng.choice(config.ALLOWED_TEMPLATES)
    n_conds = rng.integers(1, config.MAX_CONDITIONS + 1)

    # Heuristic: pick conditions that make financial sense
    heuristic_conditions = [
        Condition(feature="iv_minus_rv20", operator="<", threshold=round(rng.uniform(-0.05, 0.02), 4)),
        Condition(feature="india_vix_change", operator="<", threshold=round(rng.uniform(-0.1, 0.0), 4)),
        Condition(feature="nifty_ret1", operator=">", threshold=round(rng.uniform(-0.02, 0.01), 4)),
        Condition(feature="atm_iv", operator=">", threshold=round(rng.uniform(0.10, 0.25), 4)),
        Condition(feature="rv20", operator="<", threshold=round(rng.uniform(0.15, 0.30), 4)),
        Condition(feature="india_vix", operator="<", threshold=round(rng.uniform(15.0, 25.0), 4)),
        Condition(feature="skew_proxy", operator=">", threshold=round(rng.uniform(-0.02, 0.02), 4)),
        Condition(feature="volume", operator=">", threshold=round(rng.uniform(1000, 5000), 0)),
    ]

    chosen = list(rng.choice(heuristic_conditions, size=int(n_conds), replace=False))

    return StrategyRule(
        template=template,
        conditions=chosen,
        dte=int(rng.integers(config.DTE_MIN, config.DTE_MAX + 1)),
        hold_period=int(rng.integers(config.HOLD_PERIOD_MIN, config.HOLD_PERIOD_MAX + 1)),
    )


# ── Public API ───────────────────────────────────────────────────────────────

def propose(n: int = 1,
            top_strategies: Optional[List[dict]] = None,
            seed: int = 42) -> List[StrategyRule]:
    """
    Propose *n* strategy candidates using the LLM (or fallback).

    Parameters
    ----------
    n : int – number of candidates to generate
    top_strategies : list of dicts – best strategies from previous rounds
    seed : int – random seed for fallback

    Returns
    -------
    list of StrategyRule
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    system = _build_system_prompt()
    results = []

    for i in range(n):
        user = _build_user_prompt(top_strategies, hint=f"Attempt {i+1}/{n}")
        raw = _call_ollama(system, user)

        if raw is not None:
            rule = _parse_response(raw)
            if rule is not None:
                results.append(rule)
                continue

        # Fallback
        rule = _fallback_propose(rng)
        results.append(rule)

    return results
