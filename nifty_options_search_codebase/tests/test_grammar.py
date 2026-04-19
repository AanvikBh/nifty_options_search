"""
tests/test_grammar.py – Unit tests for strategy grammar models.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import pytest
from strategy.grammar import Condition, StrategyRule


class TestCondition:
    def test_evaluate_greater(self):
        c = Condition(feature="atm_iv", operator=">", threshold=0.15)
        assert c.evaluate(0.20) is True
        assert c.evaluate(0.10) is False

    def test_evaluate_less(self):
        c = Condition(feature="nifty_ret1", operator="<", threshold=0.01)
        assert c.evaluate(0.005) is True
        assert c.evaluate(0.02) is False

    def test_evaluate_between(self):
        c = Condition(feature="rv20", operator="between",
                      threshold=0.10, threshold_upper=0.25)
        assert c.evaluate(0.15) is True
        assert c.evaluate(0.05) is False
        assert c.evaluate(0.30) is False

    def test_to_readable(self):
        c = Condition(feature="atm_iv", operator=">", threshold=0.15)
        assert "atm_iv" in c.to_readable()
        assert ">" in c.to_readable()

    def test_invalid_operator_raises(self):
        with pytest.raises(Exception):
            Condition(feature="atm_iv", operator="!=", threshold=0.1)


class TestStrategyRule:
    def _make_rule(self, **overrides):
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

    def test_basic_creation(self):
        rule = self._make_rule()
        assert rule.template == "atm_straddle"
        assert rule.dte == 5
        assert rule.hold_period == 2

    def test_json_roundtrip(self):
        rule = self._make_rule()
        json_str = rule.model_dump_json()
        restored = StrategyRule.model_validate_json(json_str)
        assert restored.template == rule.template
        assert restored.fingerprint() == rule.fingerprint()

    def test_fingerprint_deterministic(self):
        r1 = self._make_rule()
        r2 = self._make_rule()
        assert r1.fingerprint() == r2.fingerprint()

    def test_fingerprint_changes_with_content(self):
        r1 = self._make_rule(dte=5)
        r2 = self._make_rule(dte=7)
        assert r1.fingerprint() != r2.fingerprint()

    def test_complexity(self):
        rule = self._make_rule(conditions=[
            Condition(feature="atm_iv", operator=">", threshold=0.15),
            Condition(feature="rv20", operator="<", threshold=0.30),
        ])
        assert rule.complexity() == 2

    def test_evaluate_conditions_pass(self):
        rule = self._make_rule()
        assert rule.evaluate_conditions({"atm_iv": 0.20})

    def test_evaluate_conditions_fail(self):
        rule = self._make_rule()
        assert not rule.evaluate_conditions({"atm_iv": 0.10})

    def test_evaluate_conditions_missing_feature(self):
        rule = self._make_rule()
        assert not rule.evaluate_conditions({})

    def test_invalid_template_raises(self):
        with pytest.raises(Exception):
            self._make_rule(template="exotic_butterfly")

    def test_invalid_dte_raises(self):
        with pytest.raises(Exception):
            self._make_rule(dte=30)

    def test_invalid_hold_raises(self):
        with pytest.raises(Exception):
            self._make_rule(hold_period=10)

    def test_to_readable(self):
        rule = self._make_rule()
        text = rule.to_readable()
        assert "atm_straddle" in text
        assert "DTE=5" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
