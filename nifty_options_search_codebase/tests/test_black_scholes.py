"""
tests/test_black_scholes.py – Unit tests for Black-Scholes pricing and Greeks.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from features.black_scholes import (
    bs_price, implied_volatility, delta, gamma, vega, theta,
)


class TestBSPricing:
    """Tests for BS option pricing."""

    def test_call_at_the_money(self):
        """ATM call should be roughly S * sigma * sqrt(T) * 0.4."""
        price = bs_price(100, 100, 1.0, 0.05, 0.20, "CE")
        assert 5 < price < 15, f"ATM call price {price} seems wrong"

    def test_put_at_the_money(self):
        """ATM put should be close to ATM call minus forward discount."""
        call_price = bs_price(100, 100, 1.0, 0.05, 0.20, "CE")
        put_price = bs_price(100, 100, 1.0, 0.05, 0.20, "PE")
        # Put-call parity: C - P = S - K*e^{-rT}
        parity_diff = call_price - put_price
        expected = 100 - 100 * np.exp(-0.05)
        assert abs(parity_diff - expected) < 0.01

    def test_deep_itm_call(self):
        """Deep ITM call should be close to intrinsic value."""
        price = bs_price(150, 100, 0.1, 0.05, 0.20, "CE")
        intrinsic = 150 - 100 * np.exp(-0.05 * 0.1)
        assert price >= intrinsic * 0.99

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should be near zero."""
        price = bs_price(50, 100, 0.05, 0.05, 0.20, "CE")
        assert price < 0.01

    def test_zero_time_call(self):
        """At expiry, call = max(S-K, 0)."""
        price = bs_price(110, 100, 1e-10, 0.05, 0.20, "CE")
        assert abs(price - 10) < 0.1

    def test_vectorised(self):
        """BS price should work on numpy arrays."""
        S = np.array([100, 110, 90])
        K = np.array([100, 100, 100])
        prices = bs_price(S, K, 0.5, 0.05, 0.20, "CE")
        assert len(prices) == 3
        assert all(p >= 0 for p in prices)


class TestImpliedVolatility:
    """Tests for IV solver."""

    def test_round_trip(self):
        """Compute price from vol, then solve back to vol."""
        true_vol = 0.25
        price = bs_price(100, 100, 0.5, 0.05, true_vol, "CE")
        solved_vol = implied_volatility(price, 100, 100, 0.5, 0.05, "CE")
        assert abs(solved_vol - true_vol) < 1e-4

    def test_round_trip_put(self):
        """Same round-trip for a put."""
        true_vol = 0.30
        price = bs_price(100, 105, 0.25, 0.05, true_vol, "PE")
        solved_vol = implied_volatility(price, 100, 105, 0.25, 0.05, "PE")
        assert abs(solved_vol - true_vol) < 1e-4

    def test_invalid_price(self):
        """Negative price should return NaN."""
        iv = implied_volatility(-1.0, 100, 100, 0.5, 0.05, "CE")
        assert np.isnan(iv)

    def test_zero_time(self):
        """Zero time to expiry should return NaN."""
        iv = implied_volatility(5.0, 100, 100, 0.0, 0.05, "CE")
        assert np.isnan(iv)


class TestGreeks:
    """Tests for BS Greeks."""

    def test_call_delta_range(self):
        """Call delta should be in [0, 1]."""
        d = delta(100, 100, 0.5, 0.05, 0.20, "CE")
        assert 0 <= d <= 1

    def test_put_delta_range(self):
        """Put delta should be in [-1, 0]."""
        d = delta(100, 100, 0.5, 0.05, 0.20, "PE")
        assert -1 <= d <= 0

    def test_atm_delta_near_half(self):
        """ATM call delta should be near 0.5."""
        d = delta(100, 100, 0.5, 0.05, 0.20, "CE")
        assert abs(d - 0.55) < 0.1  # slightly > 0.5 due to drift

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        g = gamma(100, 100, 0.5, 0.05, 0.20)
        assert g > 0

    def test_vega_positive(self):
        """Vega should be positive (per 1% vol)."""
        v = vega(100, 100, 0.5, 0.05, 0.20)
        assert v > 0

    def test_call_theta_negative(self):
        """Call theta should be negative (time decay)."""
        t = theta(100, 100, 0.5, 0.05, 0.20, "CE")
        assert t < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
