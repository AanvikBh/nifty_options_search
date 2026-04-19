"""
features/registry.py – Approved feature whitelist with metadata.

Each feature has:
  • name   – identifier used in strategy rules
  • family – grouping (underlying_move, realized_risk, option_surface, flow, regime)
  • dtype  – "float" (all numeric for now)
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FeatureMeta:
    name: str
    family: str
    description: str


FEATURE_REGISTRY: Dict[str, FeatureMeta] = {
    "nifty_ret1": FeatureMeta(
        "nifty_ret1", "underlying_move", "NIFTY 1-day return"
    ),
    "nifty_ret5": FeatureMeta(
        "nifty_ret5", "underlying_move", "NIFTY 5-day return"
    ),
    "rv5": FeatureMeta(
        "rv5", "realized_risk", "5-day realized volatility (annualised)"
    ),
    "rv20": FeatureMeta(
        "rv20", "realized_risk", "20-day realized volatility (annualised)"
    ),
    "atm_iv": FeatureMeta(
        "atm_iv", "option_surface", "ATM implied volatility"
    ),
    "iv_minus_rv20": FeatureMeta(
        "iv_minus_rv20", "option_surface", "ATM IV minus 20-day RV"
    ),
    "skew_proxy": FeatureMeta(
        "skew_proxy", "option_surface",
        "Near-strike skew: IV(1% OTM put) - IV(ATM call)"
    ),
    "term_slope": FeatureMeta(
        "term_slope", "option_surface",
        "Near-vs-next expiry ATM IV difference"
    ),
    "oi_change": FeatureMeta(
        "oi_change", "flow", "1-day change in ATM total open interest"
    ),
    "volume": FeatureMeta(
        "volume", "flow", "ATM total volume"
    ),
    "india_vix": FeatureMeta(
        "india_vix", "regime", "India VIX level"
    ),
    "india_vix_change": FeatureMeta(
        "india_vix_change", "regime", "India VIX 1-day change"
    ),
}


def approved_names() -> List[str]:
    """Return sorted list of approved feature names."""
    return sorted(FEATURE_REGISTRY.keys())


def is_approved(name: str) -> bool:
    return name in FEATURE_REGISTRY


def get_family(name: str) -> str:
    return FEATURE_REGISTRY[name].family if name in FEATURE_REGISTRY else "unknown"
