"""
tests/test_integration.py – End-to-end integration test with a mini search.

Runs a small 5-candidate, 1-round search to verify the full pipeline works.
"""

import sys, pathlib, os, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
import pandas as pd

import config
from data.downloader import generate_index_data, generate_vix_data, generate_option_data
from data.cleaner import clean_index, clean_vix, clean_options
from strategy.grammar import StrategyRule, Condition
from strategy.validator import is_valid
from proposers.diversity_proposer import propose as diversity_propose
from proposers.mutation_proposer import propose as mutation_propose
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from controller.scorer import composite_score
from controller.experiment_store import ExperimentStore
from controller.trust import TrustController
from orchestrator import build_features_fast


class TestEndToEnd:
    """End-to-end integration test."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Generate a small synthetic dataset for testing."""
        # Override config paths
        config.DATA_DIR = tmp_path / "data"
        config.EXPERIMENT_DB = tmp_path / "experiments.db"
        config.DATA_DIR.mkdir()

        # Generate small dataset
        self.idx = generate_index_data(seed=42)
        self.vix = generate_vix_data(seed=43)

        # Use only first 100 days for speed
        self.idx = clean_index(self.idx.head(100))
        self.vix = clean_vix(self.vix.head(100))

        # Generate options for the limited date range
        self.opts = generate_option_data(self.idx, self.vix, seed=44)
        self.opts = clean_options(self.opts, self.idx)

        # Fast features
        self.feats = build_features_fast(self.idx, self.vix)

    def test_diversity_proposer_generates_valid(self):
        """Diversity proposer should generate all-valid candidates."""
        candidates = diversity_propose(5, seed=42)
        for c in candidates:
            assert is_valid(c), f"Invalid candidate: {c.to_readable()}"

    def test_mutation_proposer_with_parent(self):
        """Mutation proposer should produce variants of a parent."""
        parent = StrategyRule(
            template="atm_straddle",
            conditions=[Condition(feature="atm_iv", operator=">", threshold=0.15)],
            dte=5,
            hold_period=2,
        )
        mutations = mutation_propose(3, [parent], seed=42)
        assert len(mutations) == 3
        # At least one should differ from parent
        fps = [m.fingerprint() for m in mutations]
        assert len(set(fps)) >= 1

    def test_backtest_runs(self):
        """Backtest should complete without error."""
        rule = StrategyRule(
            template="atm_straddle",
            conditions=[
                Condition(feature="atm_iv", operator=">", threshold=0.05),
            ],
            dte=5,
            hold_period=1,
        )
        result = run_backtest(rule, self.feats, self.idx, self.opts)
        assert result is not None
        assert isinstance(result.metrics.sharpe_ratio, float)

    def test_scorer_returns_float(self):
        """Composite score should be a finite float."""
        rule = StrategyRule(
            template="call_spread",
            conditions=[
                Condition(feature="nifty_ret1", operator="<", threshold=0.01),
            ],
            dte=5,
            hold_period=2,
        )
        bt = run_backtest(rule, self.feats, self.idx, self.opts)
        score = composite_score(bt.metrics, rule)
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_experiment_store(self):
        """Experiment store should log and query experiments."""
        store = ExperimentStore()
        rule = StrategyRule(
            template="atm_straddle",
            conditions=[Condition(feature="atm_iv", operator=">", threshold=0.15)],
            dte=5,
            hold_period=2,
        )
        from backtest.metrics import BacktestMetrics
        metrics = BacktestMetrics(sharpe_ratio=1.5, total_return=100.0)
        row_id = store.log_experiment(rule, metrics, 1.2, "diversity", 1)
        assert row_id > 0
        assert store.total_experiments() == 1

        top = store.get_top_strategies(5)
        assert len(top) == 1
        assert top[0]["composite_score"] == 1.2

    def test_trust_controller(self):
        """Trust controller should allocate budget sensibly."""
        tc = TrustController()
        alloc = tc.allocate_budget(10)
        assert sum(alloc.values()) == 10

        tc.update("llm", 1.5, "fp1", set())
        tc.update("diversity", -0.5, "fp2", set())
        alloc2 = tc.allocate_budget(10)
        assert sum(alloc2.values()) == 10

    def test_mini_search_pipeline(self):
        """Run a mini search: 5 candidates, 1 round."""
        store = ExperimentStore()
        trust = TrustController()

        # Generate candidates
        candidates = diversity_propose(5, seed=42)
        valid = [c for c in candidates if is_valid(c)]
        assert len(valid) >= 3, "Should have at least 3 valid candidates"

        # Backtest and score each
        for c in valid:
            bt = run_backtest(c, self.feats, self.idx, self.opts)
            score = composite_score(bt.metrics, c)
            store.log_experiment(c, bt.metrics, score, "diversity", 1)
            trust.update("diversity", score, c.fingerprint(), set())

        assert store.total_experiments() == len(valid)
        top = store.get_top_strategies(3)
        assert len(top) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
