"""
strategy/grammar.py – Pydantic models for structured strategy rules.

A strategy rule is a JSON-serialisable object with:
  • template  – one of the allowed trade templates
  • conditions – list of Condition objects (feature, operator, thresholds)
  • dte       – days to expiry to target
  • hold_period – how many trading days to hold

This grammar is what the LLM proposer must output and what the validator checks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import json
import hashlib

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config


class Condition(BaseModel):
    """A single entry condition: feature op threshold."""
    feature: str = Field(..., description="One of the approved feature names")
    operator: str = Field(..., description="Comparison operator: >, <, >=, <=, between")
    threshold: float = Field(..., description="Primary threshold value")
    threshold_upper: Optional[float] = Field(
        None, description="Upper bound for 'between' operator"
    )

    @field_validator("operator")
    @classmethod
    def check_operator(cls, v):
        if v not in config.ALLOWED_OPERATORS:
            raise ValueError(f"Operator {v!r} not in {config.ALLOWED_OPERATORS}")
        return v

    def evaluate(self, value: float) -> bool:
        """Evaluate this condition against a feature value."""
        if self.operator == ">":
            return value > self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "between":
            upper = self.threshold_upper if self.threshold_upper is not None else self.threshold
            return self.threshold <= value <= upper
        return False

    def to_readable(self) -> str:
        if self.operator == "between":
            return f"{self.threshold:.4f} <= {self.feature} <= {self.threshold_upper:.4f}"
        return f"{self.feature} {self.operator} {self.threshold:.4f}"


class StrategyRule(BaseModel):
    """A complete candidate strategy rule."""
    template: str = Field(..., description="Trade template name")
    conditions: List[Condition] = Field(
        ..., description="List of entry conditions (AND logic)"
    )
    dte: int = Field(..., description="Target DTE within [3, 10]")
    hold_period: int = Field(..., description="Hold period in trading days [1, 3]")

    @field_validator("template")
    @classmethod
    def check_template(cls, v):
        if v not in config.ALLOWED_TEMPLATES:
            raise ValueError(f"Template {v!r} not in {config.ALLOWED_TEMPLATES}")
        return v

    @field_validator("dte")
    @classmethod
    def check_dte(cls, v):
        if not (config.DTE_MIN <= v <= config.DTE_MAX):
            raise ValueError(f"DTE {v} not in [{config.DTE_MIN}, {config.DTE_MAX}]")
        return v

    @field_validator("hold_period")
    @classmethod
    def check_hold(cls, v):
        if not (config.HOLD_PERIOD_MIN <= v <= config.HOLD_PERIOD_MAX):
            raise ValueError(
                f"Hold {v} not in [{config.HOLD_PERIOD_MIN}, {config.HOLD_PERIOD_MAX}]"
            )
        return v

    def fingerprint(self) -> str:
        """Deterministic hash for de-duplication."""
        canonical = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.md5(canonical.encode()).hexdigest()[:12]

    def complexity(self) -> int:
        """Rule complexity = number of conditions."""
        return len(self.conditions)

    def to_readable(self) -> str:
        conds = " AND ".join(c.to_readable() for c in self.conditions)
        return (
            f"[{self.template}] DTE={self.dte} hold={self.hold_period}d "
            f"IF {conds}"
        )

    def evaluate_conditions(self, feature_row: dict) -> bool:
        """Return True if all conditions pass for the given feature values."""
        for cond in self.conditions:
            val = feature_row.get(cond.feature)
            if val is None or (isinstance(val, float) and __import__('math').isnan(val)):
                return False
            if not cond.evaluate(val):
                return False
        return True
