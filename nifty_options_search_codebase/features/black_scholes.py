"""
features/black_scholes.py – Black-Scholes pricing, implied-volatility solver, and Greeks.

All functions are vectorised (accept numpy arrays) for speed.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ── Pricing ──────────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="CE"):
    """
    Black-Scholes European option price.

    Parameters
    ----------
    S : float or ndarray – spot price
    K : float or ndarray – strike price
    T : float or ndarray – time to expiry in years (must be > 0)
    r : float – risk-free rate (annualised)
    sigma : float or ndarray – volatility (annualised)
    option_type : str – "CE" for call, "PE" for put

    Returns
    -------
    float or ndarray – option price
    """
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return np.maximum(price, 0.0)


# ── Implied Volatility ──────────────────────────────────────────────────────

def implied_volatility(market_price, S, K, T, r, option_type="CE",
                       tol=1e-6, max_iter=100):
    """
    Solve for implied volatility using Brent's method.

    Parameters
    ----------
    market_price : float – observed option price
    S, K, T, r : float – BS inputs
    option_type : str – "CE" or "PE"

    Returns
    -------
    float – implied volatility, or NaN if solver fails
    """
    if T <= 0 or market_price <= 0:
        return np.nan

    # Intrinsic value check
    if option_type == "CE":
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if market_price < intrinsic * 0.95:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 0.001, 5.0, xtol=tol, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def implied_volatility_vec(prices, S_arr, K_arr, T_arr, r, option_types):
    """Vectorised IV solver – loops internally, returns ndarray."""
    n = len(prices)
    ivs = np.full(n, np.nan)
    for i in range(n):
        ivs[i] = implied_volatility(
            prices[i], S_arr[i], K_arr[i], T_arr[i], r, option_types[i]
        )
    return ivs


# ── Greeks ───────────────────────────────────────────────────────────────────

def delta(S, K, T, r, sigma, option_type="CE"):
    """BS delta."""
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "CE":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """BS gamma (same for call and put)."""
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """BS vega (same for call and put), per 1% vol move."""
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) * 0.01


def theta(S, K, T, r, sigma, option_type="CE"):
    """BS theta (per calendar day)."""
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "CE":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)

    return (term1 + term2) / 365  # per calendar day
