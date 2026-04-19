# Walkthrough: Trust-Calibrated Strategy Search for NIFTY Weekly Options

## Summary

Built a complete end-to-end system implementing the project proposal. The codebase spans **28 files** across **10 modules** with **69 unit and integration tests**, all passing.

## What Was Built

### Files Created

| Module | Files | Purpose |
|--------|-------|---------|
| **config** | `config.py` | All project constants, scoring weights, trust config, LLM settings |
| **data/** | `downloader.py`, `cleaner.py`, `store.py` | GBM+BS synthetic data generation, cleaning, Parquet store |
| **features/** | `black_scholes.py`, `engine.py`, `registry.py` | BS pricing, IV solver, Greeks, 12-feature pipeline |
| **strategy/** | `grammar.py`, `templates.py`, `validator.py` | Pydantic strategy models, 4 trade templates, 9-check validator |
| **proposers/** | `llm_proposer.py`, `mutation_proposer.py`, `diversity_proposer.py` | Ollama/Mistral LLM, 5 mutation types, round-robin diversity |
| **tuner/** | `param_tuner.py` | Random/grid search over thresholds, DTE, hold period |
| **backtest/** | `engine.py`, `costs.py`, `metrics.py` | Walk-forward backtester, cost model, Sharpe/DD/hit rate |
| **controller/** | `scorer.py`, `experiment_store.py`, `trust.py` | Composite scoring, SQLite log, UCB1 bandit trust |
| **orchestrator** | `orchestrator.py` | CLI entry point, multi-round search loop |
| **tests/** | 6 test files | 69 tests covering all modules |

### Key Design Decisions

1. **LLM**: Mistral 7B via Ollama with Pydantic schema enforcement for structured JSON output. Falls back to heuristic proposers when Ollama is unavailable.

2. **Trust System**: UCB1 bandit with 10% exploration floor and similarity-aware success discounting. Handles edge cases (inf scores for untried sources).

3. **Data**: Fully synthetic (GBM index + OU VIX + BS-priced options) so the system is immediately runnable. Real NSE data hooks provided.

4. **Backtester**: Walk-forward, day-by-day simulation with no future leakage, contract-level entry/exit, and realistic cost model.

## Bugs Fixed During Development

1. **Pandas index alignment** in `clean_options()` — intrinsic value comparison across filtered subsets caused `ValueError`. Fixed by computing intrinsic values on the full DataFrame.

2. **NaN in trust allocation** — when sources had 0 evaluations, UCB score was `inf`, causing NaN during normalisation. Fixed by special-casing `inf` sources.

## Test Results

```
69 passed in 6.40s
```

**Test coverage:**
- Black-Scholes pricing, put-call parity, IV round-trip, Greeks ranges (16 tests)
- Strategy grammar serialisation, fingerprinting, condition evaluation (12 tests)
- Validator: approved features, complexity cap, leakage detection (9 tests)
- Metrics: Sharpe, drawdown, hit rate; Cost model (13 tests)
- Trust: budget allocation, exploration floor, weight convergence (7 tests)
- Integration: mini search pipeline end-to-end (7 tests)

## How to Run

```bash
cd "/nifty_options_search_codebase"
source venv/bin/activate

# Quick demo (20 candidates, 2 rounds)
python orchestrator.py --budget 20 --rounds 2

# Full search
python orchestrator.py --budget 400 --rounds 10 --tune

# Tests
python -m pytest tests/ -v
```
