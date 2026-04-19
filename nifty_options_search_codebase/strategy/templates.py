"""
strategy/templates.py – Trade template definitions.

Each template specifies how to build a 1- or 2-leg option position from
the contract universe on a given day.

Templates:
  • atm_straddle     – buy ATM call + buy ATM put
  • call_spread      – buy ATM call + sell 1-step OTM call
  • put_spread       – buy ATM put  + sell 1-step OTM put
  • risk_reversal    – sell 1-step OTM put + buy 1-step OTM call
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Leg:
    """One leg of an option position."""
    option_type: str          # "CE" or "PE"
    strike_offset: int        # 0 = ATM, +1 = 1-step OTM call side, -1 = 1-step OTM put side
    direction: int            # +1 = buy, -1 = sell
    label: str                # human-readable label

    def describe(self) -> str:
        side = "BUY" if self.direction == 1 else "SELL"
        offset = {0: "ATM", 1: "1-step OTM", -1: "1-step OTM"}
        return f"{side} {offset.get(self.strike_offset, '?')} {self.option_type}"


@dataclass
class TradeTemplate:
    """A multi-leg option trade structure."""
    name: str
    legs: List[Leg]
    description: str

    def n_legs(self) -> int:
        return len(self.legs)


# ── Template definitions ─────────────────────────────────────────────────────

TEMPLATES = {
    "atm_straddle": TradeTemplate(
        name="atm_straddle",
        legs=[
            Leg("CE", 0, +1, "Buy ATM Call"),
            Leg("PE", 0, +1, "Buy ATM Put"),
        ],
        description="Long ATM straddle: buy ATM call + buy ATM put",
    ),
    "call_spread": TradeTemplate(
        name="call_spread",
        legs=[
            Leg("CE", 0,  +1, "Buy ATM Call"),
            Leg("CE", 1,  -1, "Sell 1-step OTM Call"),
        ],
        description="Bull call spread: buy ATM call + sell 1-step OTM call",
    ),
    "put_spread": TradeTemplate(
        name="put_spread",
        legs=[
            Leg("PE", 0,  +1, "Buy ATM Put"),
            Leg("PE", -1, -1, "Sell 1-step OTM Put"),
        ],
        description="Bear put spread: buy ATM put + sell 1-step OTM put",
    ),
    "risk_reversal": TradeTemplate(
        name="risk_reversal",
        legs=[
            Leg("PE", -1, -1, "Sell 1-step OTM Put"),
            Leg("CE",  1, +1, "Buy 1-step OTM Call"),
        ],
        description="Risk reversal: sell OTM put + buy OTM call",
    ),
}


def get_template(name: str) -> TradeTemplate:
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template: {name!r}")
    return TEMPLATES[name]


def resolve_strikes(template: TradeTemplate, atm_strike: float,
                    strike_step: float) -> List[Tuple[str, float, int]]:
    """
    Resolve a template's legs into concrete (option_type, strike, direction) tuples.

    Parameters
    ----------
    template : TradeTemplate
    atm_strike : float – the ATM strike price
    strike_step : float – the strike interval (e.g., 50)

    Returns
    -------
    list of (option_type, strike, direction)
    """
    resolved = []
    for leg in template.legs:
        if leg.option_type == "CE":
            strike = atm_strike + leg.strike_offset * strike_step
        else:
            strike = atm_strike + leg.strike_offset * strike_step
        resolved.append((leg.option_type, strike, leg.direction))
    return resolved
