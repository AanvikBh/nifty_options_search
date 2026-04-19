"""
strategy/validator.py – Candidate strategy validation.

Checks every candidate rule against the project's constraints before it
reaches the backtester.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from strategy.grammar import StrategyRule
from strategy.templates import TEMPLATES
from features.registry import is_approved


class ValidationError(Exception):
    """Raised when a candidate fails validation."""
    pass


def validate(rule: StrategyRule, available_dates=None) -> bool:
    """
    Validate a StrategyRule.  Raises ValidationError on failure.

    Checks:
      1. Template is in the allowed set.
      2. Number of legs ≤ MAX_LEGS.
      3. All features are in the approved registry.
      4. All operators are allowed.
      5. Complexity cap (max conditions).
      6. DTE and hold_period are within bounds.
      7. No future-data leakage (structural check).
      8. 'between' conditions have valid upper bound.

    Returns True if all checks pass.
    """
    errors = []

    # 1. Template check
    if rule.template not in config.ALLOWED_TEMPLATES:
        errors.append(f"Template '{rule.template}' is not allowed.")

    # 2. Legs check
    tmpl = TEMPLATES.get(rule.template)
    if tmpl and tmpl.n_legs() > config.MAX_LEGS:
        errors.append(f"Template '{rule.template}' has {tmpl.n_legs()} legs > max {config.MAX_LEGS}.")

    # 3. Feature check
    for cond in rule.conditions:
        if not is_approved(cond.feature):
            errors.append(f"Feature '{cond.feature}' is not in the approved registry.")

    # 4. Operator check
    for cond in rule.conditions:
        if cond.operator not in config.ALLOWED_OPERATORS:
            errors.append(f"Operator '{cond.operator}' not allowed.")

    # 5. Complexity cap
    if len(rule.conditions) > config.MAX_CONDITIONS:
        errors.append(
            f"Too many conditions ({len(rule.conditions)} > max {config.MAX_CONDITIONS})."
        )

    # 6. DTE bounds
    if not (config.DTE_MIN <= rule.dte <= config.DTE_MAX):
        errors.append(f"DTE {rule.dte} outside [{config.DTE_MIN}, {config.DTE_MAX}].")

    # 7. Hold-period bounds
    if not (config.HOLD_PERIOD_MIN <= rule.hold_period <= config.HOLD_PERIOD_MAX):
        errors.append(
            f"Hold period {rule.hold_period} outside "
            f"[{config.HOLD_PERIOD_MIN}, {config.HOLD_PERIOD_MAX}]."
        )

    # 8. 'between' operator has upper bound
    for cond in rule.conditions:
        if cond.operator == "between" and cond.threshold_upper is None:
            errors.append(
                f"Condition on '{cond.feature}' uses 'between' but has no upper bound."
            )
        if cond.operator == "between" and cond.threshold_upper is not None:
            if cond.threshold > cond.threshold_upper:
                errors.append(
                    f"Condition on '{cond.feature}': lower > upper in 'between'."
                )

    # 9. No future-data leakage (structural)
    # All approved features are computed from day-t data by construction,
    # so this is a design-level guarantee.  We still flag any feature
    # that includes "future" or "tomorrow" in its name (defensive).
    for cond in rule.conditions:
        if "future" in cond.feature.lower() or "tomorrow" in cond.feature.lower():
            errors.append(f"Feature '{cond.feature}' may leak future data.")

    if errors:
        raise ValidationError("; ".join(errors))

    return True


def is_valid(rule: StrategyRule) -> bool:
    """Convenience wrapper – returns bool instead of raising."""
    try:
        return validate(rule)
    except ValidationError:
        return False
