"""
config.py – Central configuration for the NIFTY Weekly Options Strategy Search project.

All project-fixed constants from the proposal live here so that every module
imports a single source of truth.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict


# ── Date boundaries ──────────────────────────────────────────────────────────
DATA_START = date(2021, 1, 1)
DATA_END   = date(2025, 12, 31)

TRAIN_START = date(2021, 1, 1)
TRAIN_END   = date(2024, 12, 31)

HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END   = date(2025, 12, 31)


# ── Contract universe ────────────────────────────────────────────────────────
DTE_MIN = 3
DTE_MAX = 10
MAX_LEGS = 2
HOLD_PERIOD_MIN = 1
HOLD_PERIOD_MAX = 3

# OTM offset as a fraction of the underlying price
OTM_STEP_PCT = 0.01  # 1 %


# ── Allowed trade templates ──────────────────────────────────────────────────
ALLOWED_TEMPLATES = [
    "atm_straddle",
    "call_spread",
    "put_spread",
    "risk_reversal",
]


# ── Approved features ────────────────────────────────────────────────────────
APPROVED_FEATURES = [
    "atm_iv",
    "iv_minus_rv20",
    "skew_proxy",
    "term_slope",
    "nifty_ret1",
    "nifty_ret5",
    "rv5",
    "rv20",
    "india_vix",
    "india_vix_change",
    "oi_change",
    "volume",
]


# ── Allowed operators for rule conditions ────────────────────────────────────
ALLOWED_OPERATORS = [">", "<", ">=", "<=", "between"]


# ── Cost model ───────────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 5          # basis points of premium
LIQUIDITY_PENALTY_BPS = 3         # extra penalty for low-liquidity contracts
MIN_VOLUME_FILTER = 100           # minimum daily volume to consider a contract
MIN_OI_FILTER = 500               # minimum open interest

# ── Risk-free rate proxy (annualized) ────────────────────────────────────────
RISK_FREE_RATE = 0.06             # 6 %

# ── Lot size for NIFTY options (NSE standard) ────────────────────────────────
NIFTY_LOT_SIZE = 25               # as of 2024-25, NIFTY lot = 25

# ── Synthetic data parameters ────────────────────────────────────────────────
SYNTH_INITIAL_SPOT = 15000.0      # NIFTY spot on 2021-01-01
SYNTH_ANNUAL_DRIFT = 0.10         # 10 % annual drift
SYNTH_ANNUAL_VOL = 0.18           # 18 % annual volatility
SYNTH_VIX_MEAN = 15.0
SYNTH_VIX_STD = 4.0
STRIKE_STEP = 50                  # NIFTY strike interval


# ── Search defaults ──────────────────────────────────────────────────────────
DEFAULT_BUDGET = 400              # total candidate evaluations
DEFAULT_ROUNDS = 10               # number of search rounds
CANDIDATES_PER_ROUND = 40         # budget / rounds

# ── Complexity cap ───────────────────────────────────────────────────────────
MAX_CONDITIONS = 4                # max number of conditions in a rule


# ── Scoring weights ──────────────────────────────────────────────────────────
@dataclass
class ScoringWeights:
    lambda_dd: float = 0.5        # drawdown penalty
    lambda_to: float = 0.1        # turnover penalty
    lambda_c:  float = 0.05       # complexity penalty

SCORING_WEIGHTS = ScoringWeights()


# ── Trust / budget-allocation defaults ───────────────────────────────────────
@dataclass
class TrustConfig:
    min_exploration_floor: float = 0.10   # no source gets < 10 %
    initial_weights: Dict[str, float] = field(default_factory=lambda: {
        "llm": 0.40,
        "mutation": 0.35,
        "diversity": 0.25,
    })
    ucb_exploration_c: float = 1.41       # sqrt(2) for UCB1
    similarity_discount: float = 0.5      # discount for near-duplicate successes

TRUST_CONFIG = TrustConfig()


# ── LLM (Ollama) settings ────────────────────────────────────────────────────
OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.3
OLLAMA_TIMEOUT = 30               # seconds


# ── Parameter tuning ─────────────────────────────────────────────────────────
TUNER_MODE = "random"             # "grid" | "random" | "bayesian"
TUNER_MAX_EVALS = 20              # per candidate structure


# ── Paths ────────────────────────────────────────────────────────────────────
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data_store"
EXPERIMENT_DB = PROJECT_ROOT / "experiments.db"
