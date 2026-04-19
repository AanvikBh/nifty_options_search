# Trust-Calibrated Strategy Search for NIFTY Weekly Options

An end-to-end system that discovers, validates, backtests, and ranks short-dated NIFTY option strategies using an LLM-guided search with adaptive trust allocation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [What's Implemented](#whats-implemented)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Running Tests](#running-tests)
8. [Module Reference](#module-reference)
9. [How the Trust System Works](#how-the-trust-system-works)

---

## Project Overview

This system implements the full proposal for a **trust-calibrated strategy search** over NIFTY 50 weekly index options. The search loop:

1. **Proposes** candidate strategy rules via three sources:
   - **LLM Proposer** – Mistral 7B (via Ollama) generates structured JSON strategies
   - **Mutation Proposer** – mutates top past strategies (nudge thresholds, swap features)
   - **Diversity Proposer** – random valid strategies for exploration breadth
2. **Validates** every candidate against strict constraints (max 2 legs, approved features only, no future leakage, complexity cap)
3. **Tunes** numeric parameters (thresholds, DTE, hold period) via random/grid search
4. **Backtests** on a frozen end-of-day dataset with realistic transaction costs and liquidity filters
5. **Scores** using a composite metric: `R = Sharpe − λ·Drawdown − λ·Turnover − λ·Complexity`
6. **Stores** every experiment in SQLite with full lineage
7. **Updates trust** using a UCB1 bandit to allocate more search budget to the proposal source that produces better strategies

---

## Architecture

```
nifty_options_search_codebase/
├── config.py                  # All project constants
├── orchestrator.py            # Main CLI entry point
│
├── data/                      # Data pipeline
│   ├── downloader.py          # Synthetic data generator + real data hooks
│   ├── cleaner.py             # Cleaning, outlier removal
│   └── store.py               # Parquet-based versioned store
│
├── features/                  # Feature engineering
│   ├── black_scholes.py       # BS pricing, IV solver, Greeks (Δ, Γ, ν, Θ)
│   ├── engine.py              # Full feature computation pipeline
│   └── registry.py            # Approved feature whitelist
│
├── strategy/                  # Strategy grammar & validation
│   ├── grammar.py             # Pydantic models (Condition, StrategyRule)
│   ├── templates.py           # Trade templates (straddle, spreads, reversal)
│   └── validator.py           # Rule validation (legs, features, leakage)
│
├── proposers/                 # Candidate generation
│   ├── llm_proposer.py        # Mistral 7B via Ollama
│   ├── mutation_proposer.py   # Mutate top strategies
│   └── diversity_proposer.py  # Random exploration
│
├── tuner/
│   └── param_tuner.py         # Grid / random search over parameters
│
├── backtest/                  # Backtesting engine
│   ├── engine.py              # Walk-forward day-by-day simulation
│   ├── costs.py               # Transaction cost + liquidity model
│   └── metrics.py             # Sharpe, drawdown, hit rate, stability
│
├── controller/                # Search controller
│   ├── scorer.py              # Composite scoring + Pareto dominance
│   ├── experiment_store.py    # SQLite experiment log
│   └── trust.py               # UCB1 bandit budget allocation
│
└── tests/                     # Test suite
    ├── test_black_scholes.py  # BS pricing & IV solver tests
    ├── test_grammar.py        # Strategy model serialisation tests
    ├── test_validator.py      # Validation logic tests
    ├── test_backtest.py       # Metrics & cost model tests
    ├── test_trust.py          # Trust allocation tests
    └── test_integration.py    # End-to-end mini search test
```

---

## What's Implemented

### Data Pipeline
- **Synthetic Data Generator**: GBM index simulation, Ornstein-Uhlenbeck VIX, BS-priced option contracts with realistic volume/OI
- **Real Data Hooks**: `load_real_index_csv()`, `load_real_vix_csv()`, `load_real_options_csv()` for plugging in NSE bhavcopy data
- **Data Cleaner**: Forward-fill, outlier removal, intrinsic-value sanity checks
- **Parquet Store**: Version-controlled frozen dataset

### Feature Engineering
- **Black-Scholes Module**: Pricing, Newton-Raphson IV solver (Brent's method), full Greeks suite (delta, gamma, vega, theta)
- **12 Approved Features**: nifty_ret1, nifty_ret5, rv5, rv20, atm_iv, iv_minus_rv20, skew_proxy, term_slope, oi_change, volume, india_vix, india_vix_change
- **Fast Mode**: VIX-proxy features for rapid iteration

### Strategy Grammar
- **Pydantic Models**: `Condition` and `StrategyRule` with built-in validation, JSON serialisation, fingerprinting
- **4 Trade Templates**: ATM straddle, bull call spread, bear put spread, risk reversal
- **Validator**: 9 constraint checks including feature whitelist, complexity cap, leakage detection

### Proposer System
- **LLM Proposer**: Ollama integration with structured JSON output (Pydantic schema enforcement), heuristic fallback when Ollama is unavailable
- **Mutation Proposer**: 5 mutation types (threshold nudge, feature swap, DTE change, hold change, condition add/remove)
- **Diversity Proposer**: Round-robin template coverage with duplicate avoidance

### Parameter Tuning
- **Random Search**: Samples from parameter ranges (thresholds, DTE, hold period)
- **Grid Search**: Systematic sweep with one-at-a-time threshold optimisation

### Backtesting Engine
- **Walk-Forward Simulation**: Day-by-day with no future data leakage
- **Contract Selection**: Nearest weekly expiry, ATM/OTM strike resolution
- **Entry/Exit**: Signal on day t → enter day t+1 → exit after hold period (all EOD)
- **Cost Model**: Premium-based transaction costs + volume-based liquidity penalties
- **Metrics**: Total return, Sharpe ratio, max drawdown, hit rate, turnover, rolling stability

### Controller
- **Composite Scorer**: `R = Sharpe_OOS − λ_dd · MaxDD − λ_to · Turnover − λ_c · Complexity`
- **Pareto Dominance**: Multi-objective comparison
- **Experiment Store**: SQLite with full experiment log (structure, params, metrics, lineage, source, timestamp)
- **Trust Controller**: UCB1 bandit with 10% exploration floor, similarity-aware success discounting

### Orchestrator
- **CLI Interface**: Configurable budget, rounds, tuning, holdout evaluation
- **Multi-Round Search Loop**: Propose → validate → tune → backtest → score → store → update trust
- **Holdout Evaluation**: Final untouched 2025 period test
- **CSV Export**: Full experiment log export

---

## Installation

### Prerequisites
- Python 3.9+
- (Optional) [Ollama](https://ollama.ai/) for LLM-based strategy proposals

### Steps

```bash
# 1. Navigate to the project
cd nifty_options_search_codebase

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install Ollama and pull Mistral
# Visit https://ollama.ai/ to install Ollama, then:
ollama pull mistral
```

> **Note**: The system works perfectly without Ollama — it falls back to heuristic-based proposers. Ollama just adds LLM-generated strategies to the mix.

---

## Quick Start

### Generate data and run a quick search (20 candidates, 2 rounds)

```bash
python orchestrator.py --budget 20 --rounds 2
```

### Full search (400 candidates, 10 rounds, with parameter tuning)

```bash
python orchestrator.py --budget 400 --rounds 10 --tune
```

### Regenerate the synthetic dataset

```bash
python orchestrator.py --budget 20 --rounds 2 --regenerate-data
```

### Export experiment log

```bash
python orchestrator.py --budget 100 --rounds 5 --export-csv results.csv
```

### What to expect

```
12:00:00 [INFO] Trust-Calibrated Strategy Search for NIFTY Weekly Options
12:00:00 [INFO] Budget: 20 candidates over 2 rounds
12:00:01 [INFO] Generating synthetic NIFTY dataset ...
12:00:25 [INFO] Dataset generated in 24.3s
12:00:26 [INFO] Feature matrix: 1245 rows × 13 cols
12:00:26 [INFO] ROUND 1/2
12:00:26 [INFO] Budget allocation: {'llm': 8, 'mutation': 7, 'diversity': 5}
...
12:01:30 [INFO] TOP 10 STRATEGIES (by composite score)
12:01:30 [INFO]   #1 [diversity] Score=0.8523 Sharpe=1.2341 ...
```

---

## Configuration

All settings are in `config.py`. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATA_START` / `DATA_END` | 2021-01-01 / 2025-12-31 | Dataset date range |
| `TRAIN_END` | 2024-12-31 | In-sample training cutoff |
| `HOLDOUT_START` | 2025-01-01 | Out-of-sample holdout start |
| `DTE_MIN` / `DTE_MAX` | 3 / 10 | Days-to-expiry band |
| `HOLD_PERIOD_MIN` / `MAX` | 1 / 3 | Holding period (trading days) |
| `MAX_LEGS` | 2 | Maximum option legs |
| `MAX_CONDITIONS` | 4 | Max conditions per rule |
| `DEFAULT_BUDGET` | 400 | Total candidate evaluations |
| `TRANSACTION_COST_BPS` | 5 | Transaction cost (bps of premium) |
| `OLLAMA_MODEL` | `"mistral"` | LLM model name |

### Scoring Weights

```python
SCORING_WEIGHTS = ScoringWeights(
    lambda_dd=0.5,    # Drawdown penalty
    lambda_to=0.1,    # Turnover penalty
    lambda_c=0.05,    # Complexity penalty
)
```

### Trust Configuration

```python
TRUST_CONFIG = TrustConfig(
    min_exploration_floor=0.10,   # No source gets < 10%
    ucb_exploration_c=1.41,       # UCB1 exploration constant
    similarity_discount=0.5,      # Discount for duplicate successes
)
```

---

## Running Tests

```bash
# Run all tests
cd nifty_options_search
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_black_scholes.py -v
python -m pytest tests/test_grammar.py -v
python -m pytest tests/test_validator.py -v
python -m pytest tests/test_backtest.py -v
python -m pytest tests/test_trust.py -v
python -m pytest tests/test_integration.py -v
```

---

## Module Reference

### `features/black_scholes.py`
- `bs_price(S, K, T, r, sigma, type)` – Black-Scholes option pricing
- `implied_volatility(price, S, K, T, r, type)` – IV solver (Brent's method)
- `delta()`, `gamma()`, `vega()`, `theta()` – Greeks

### `strategy/grammar.py`
- `Condition(feature, operator, threshold)` – Single entry condition
- `StrategyRule(template, conditions, dte, hold_period)` – Full strategy rule
- JSON serialisable via Pydantic with `model_dump_json()` / `model_validate_json()`

### `backtest/engine.py`
- `run_backtest(rule, features_df, index_df, options_df)` → `BacktestResult`

### `controller/trust.py`
- `TrustController.allocate_budget(n)` → `{"llm": x, "mutation": y, "diversity": z}`
- `TrustController.update(source, score, fingerprint)` – update after each evaluation

---

## How the Trust System Works

The trust controller solves the **budget allocation problem** from the proposal:

```
B_llm + B_mut + B_div = B_total
```

It uses **UCB1 (Upper Confidence Bound)** to balance exploitation and exploration:

```
UCB(source) = avg_score(source) + c · √(ln(N) / n_source)
```

Where:
- `avg_score` = average composite score for that source
- `c = √2` = exploration constant
- `N` = total evaluations across all sources
- `n_source` = evaluations for this specific source

**Key features:**
- **Exploration floor**: Every source gets at least 10% of the budget
- **Similarity discount**: Near-duplicate successes count as 0.5 instead of 1.0
- **Adaptive**: Budget shifts towards better-performing sources across rounds

---

## Using Real NSE Data

To use real data instead of synthetic:

1. Download NSE bhavcopy data and save as CSVs
2. Use the data loading hooks:

```python
from data.downloader import load_real_index_csv, load_real_vix_csv, load_real_options_csv

index_df = load_real_index_csv("path/to/nifty_index.csv")      # Columns: Date, Close
vix_df = load_real_vix_csv("path/to/india_vix.csv")            # Columns: Date, Close
options_df = load_real_options_csv("path/to/options.csv")       # Columns: date, expiry, strike, option_type, close, volume, oi

from data.store import save_dataset
save_dataset(index_df, vix_df, options_df)
```

Then run the orchestrator normally — it will use the stored dataset.

---

## Authors

Hardik Kalia, Manasa Kalaimalai, Aanvik Bhatnagar, Keval Jain
