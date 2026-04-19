"""
tests/test_trust.py – Unit tests for trust controller / budget allocation.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from controller.trust import TrustController, SourceStats
import config


class TestTrustController:
    def _make_controller(self):
        return TrustController()

    def test_initial_allocation_uses_initial_weights(self):
        tc = self._make_controller()
        alloc = tc.allocate_budget(40)
        assert sum(alloc.values()) == 40
        assert alloc["llm"] >= 1
        assert alloc["mutation"] >= 1
        assert alloc["diversity"] >= 1

    def test_budget_sums_correctly(self):
        tc = self._make_controller()
        for budget in [10, 20, 50, 100]:
            alloc = tc.allocate_budget(budget)
            assert sum(alloc.values()) == budget

    def test_exploration_floor(self):
        tc = self._make_controller()
        # Simulate LLM being very good
        for _ in range(50):
            tc.update("llm", 2.0, f"fp_{_}", set())
        for _ in range(50):
            tc.update("mutation", -1.0, f"mfp_{_}", set())
        for _ in range(50):
            tc.update("diversity", -1.0, f"dfp_{_}", set())

        alloc = tc.allocate_budget(100)
        # Even the worst sources should get at least floor
        for src in ["mutation", "diversity"]:
            assert alloc[src] >= 1, f"{src} should get at least 1"

    def test_update_stats(self):
        tc = self._make_controller()
        tc.update("llm", 1.5, "fp1", set())
        tc.update("llm", -0.5, "fp2", set())
        tc.update("mutation", 2.0, "fp3", set())

        stats = tc.get_stats()
        assert stats["llm"]["n_total"] == 2
        assert stats["mutation"]["n_total"] == 1

    def test_weights_sum_to_one(self):
        tc = self._make_controller()
        # After some updates
        tc.update("llm", 1.0, "fp1", set())
        tc.update("mutation", 0.5, "fp2", set())
        tc.update("diversity", -0.5, "fp3", set())
        tc.allocate_budget(30)

        weights = tc.get_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.1, f"Weights should sum to ~1.0, got {total}"

    def test_good_source_gets_more_budget(self):
        tc = self._make_controller()
        # LLM is consistently good, others are bad
        for i in range(20):
            tc.update("llm", 2.0, f"lfp_{i}", set())
            tc.update("mutation", -1.0, f"mfp_{i}", set())
            tc.update("diversity", -1.0, f"dfp_{i}", set())

        alloc = tc.allocate_budget(100)
        assert alloc["llm"] > alloc["mutation"]
        assert alloc["llm"] > alloc["diversity"]

    def test_summary_output(self):
        tc = self._make_controller()
        tc.update("llm", 1.0, "fp1", set())
        summary = tc.summary()
        assert "llm" in summary
        assert "mutation" in summary
        assert "diversity" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
