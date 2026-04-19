"""
tests/test_validator.py – Unit tests for strategy validation.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from strategy.grammar import Condition, StrategyRule
from strategy.validator import validate, is_valid, ValidationError


def _make_rule(**overrides):
    defaults = {
        "template": "atm_straddle",
        "conditions": [
            Condition(feature="atm_iv", operator=">", threshold=0.15),
        ],
        "dte": 5,
        "hold_period": 2,
    }
    defaults.update(overrides)
    return StrategyRule(**defaults)


class TestValidator:
    def test_valid_rule_passes(self):
        rule = _make_rule()
        assert validate(rule) is True
        assert is_valid(rule) is True

    def test_unapproved_feature_fails(self):
        rule = _make_rule(conditions=[
            Condition(feature="bitcoin_price", operator=">", threshold=50000),
        ])
        with pytest.raises(ValidationError, match="approved"):
            validate(rule)
        assert is_valid(rule) is False

    def test_too_many_conditions_fails(self):
        import config
        conds = [
            Condition(feature="atm_iv", operator=">", threshold=0.1 + i*0.01)
            for i in range(config.MAX_CONDITIONS + 1)
        ]
        rule = _make_rule(conditions=conds)
        assert is_valid(rule) is False

    def test_between_without_upper_fails(self):
        rule = _make_rule(conditions=[
            Condition(feature="rv20", operator="between", threshold=0.10),
        ])
        assert is_valid(rule) is False

    def test_between_with_valid_bounds_passes(self):
        rule = _make_rule(conditions=[
            Condition(feature="rv20", operator="between",
                      threshold=0.10, threshold_upper=0.25),
        ])
        assert is_valid(rule) is True

    def test_between_lower_greater_than_upper_fails(self):
        rule = _make_rule(conditions=[
            Condition(feature="rv20", operator="between",
                      threshold=0.30, threshold_upper=0.10),
        ])
        assert is_valid(rule) is False

    def test_future_leakage_feature_fails(self):
        # Register a fake feature for leakage test
        rule = _make_rule(conditions=[
            Condition(feature="future_return", operator=">", threshold=0.01),
        ])
        assert is_valid(rule) is False

    def test_multi_condition_valid(self):
        rule = _make_rule(conditions=[
            Condition(feature="atm_iv", operator=">", threshold=0.15),
            Condition(feature="nifty_ret1", operator="<", threshold=0.01),
            Condition(feature="india_vix", operator="<", threshold=20.0),
        ])
        assert is_valid(rule) is True

    def test_all_templates_valid(self):
        import config
        for tmpl in config.ALLOWED_TEMPLATES:
            rule = _make_rule(template=tmpl)
            assert is_valid(rule), f"Template {tmpl} should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
