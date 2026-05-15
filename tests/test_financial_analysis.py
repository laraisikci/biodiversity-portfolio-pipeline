"""Tests for the financial analysis utility."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.financial_analysis import (
    bloomberg_to_yahoo_ticker,
    compute_daily_returns,
    compute_portfolio_returns,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
    beta_vs_benchmark,
    best_worst_year,
    compute_full_metrics,
    BLOOMBERG_TO_YAHOO_SUFFIX,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)
from agents.decision_log import read_log, LOG_PATH


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def _build_synthetic_prices(n_days: int = 750, seed: int = 42) -> pd.DataFrame:
    """Build synthetic daily price series for 3 tickers."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")
    # Three tickers with different return/vol profiles
    returns = pd.DataFrame({
        "ASML.AS": np.random.normal(0.0008, 0.018, n_days),  # high return, high vol
        "SAN.MC":  np.random.normal(0.0004, 0.014, n_days),  # mid return, mid vol
        "ENEL.MI": np.random.normal(0.0002, 0.012, n_days),  # low return, low vol
    }, index=dates)
    prices = (1 + returns).cumprod() * 100
    return prices


# === Ticker mapping ===

def test_bloomberg_to_yahoo_basic():
    """ABI BB Equity -> ABI.BR (Brussels)."""
    assert bloomberg_to_yahoo_ticker("ABI BB Equity") == "ABI.BR"


def test_bloomberg_to_yahoo_amsterdam():
    """ASML NA Equity -> ASML.AS."""
    assert bloomberg_to_yahoo_ticker("ASML NA Equity") == "ASML.AS"


def test_bloomberg_to_yahoo_xetra():
    """BAS GY Equity -> BAS.DE."""
    assert bloomberg_to_yahoo_ticker("BAS GY Equity") == "BAS.DE"


def test_bloomberg_to_yahoo_unknown_exchange():
    """Unknown exchange -> None."""
    assert bloomberg_to_yahoo_ticker("XYZ ZZ Equity") is None


def test_bloomberg_to_yahoo_malformed():
    """Empty or malformed ticker -> None."""
    assert bloomberg_to_yahoo_ticker("") is None
    assert bloomberg_to_yahoo_ticker(None) is None
    assert bloomberg_to_yahoo_ticker("ABI") is None


def test_all_eurostoxx_exchanges_covered():
    """All exchange codes in the EURO STOXX 50 should be in our mapping."""
    # These are the exchange codes appearing in the 50 constituents
    needed = {"BB", "NA", "GY", "FP", "SQ", "IM", "FH"}
    for code in needed:
        assert code in BLOOMBERG_TO_YAHOO_SUFFIX, (
            f"Exchange {code} not in mapping but appears in EURO STOXX 50"
        )


# === Return calculations ===

def test_daily_returns_basic():
    """Daily returns should be (price_t / price_t-1) - 1."""
    prices = _build_synthetic_prices(50)
    returns = compute_daily_returns(prices)
    # First row is dropped (no prior price); rest preserved
    assert len(returns) == 49
    # Check the math on a single point
    expected = prices.iloc[1, 0] / prices.iloc[0, 0] - 1
    assert abs(returns.iloc[0, 0] - expected) < 1e-9


def test_portfolio_returns_weighted():
    """Portfolio return = weighted sum of constituent returns."""
    prices = _build_synthetic_prices(100)
    returns = compute_daily_returns(prices)
    weights = {"ASML.AS": 0.5, "SAN.MC": 0.3, "ENEL.MI": 0.2}
    port_returns = compute_portfolio_returns(returns, weights)
    # Spot check the first day
    first_day = (
        0.5 * returns.iloc[0]["ASML.AS"]
        + 0.3 * returns.iloc[0]["SAN.MC"]
        + 0.2 * returns.iloc[0]["ENEL.MI"]
    )
    assert abs(port_returns.iloc[0] - first_day) < 1e-9


def test_portfolio_returns_missing_tickers():
    """If some weights reference tickers not in returns, renormalise."""
    prices = _build_synthetic_prices(50)
    returns = compute_daily_returns(prices)
    # Include a ticker we don't have prices for
    weights = {"ASML.AS": 0.5, "MISSING.XX": 0.3, "SAN.MC": 0.2}
    port_returns = compute_portfolio_returns(returns, weights)
    # Should still compute (ASML at 0.5/0.7, SAN at 0.2/0.7 after renormalisation)
    assert len(port_returns) > 0


# === Risk metrics ===

def test_annualised_return_zero():
    """Empty series returns 0."""
    assert annualised_return(pd.Series([], dtype=float)) == 0.0


def test_annualised_volatility_positive():
    """Positive vol from random returns."""
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    vol = annualised_volatility(returns)
    assert 0.10 < vol < 0.20  # roughly 10-20% annualised


def test_sharpe_zero_vol():
    """Zero volatility returns Sharpe of 0 (avoid div by zero)."""
    returns = pd.Series([0.0001] * 252)
    assert sharpe_ratio(returns) == 0.0


def test_max_drawdown_known_case():
    """Known pattern: 10% gain, 30% loss = ~30% drawdown."""
    # Build a price path: starts at 100, peaks at 110, drops to 77
    returns = pd.Series([0.10, -0.30])
    mdd = max_drawdown(returns)
    assert -0.31 < mdd < -0.29  # ~-30%


def test_max_drawdown_no_loss():
    """All positive returns = 0 drawdown."""
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert max_drawdown(returns) == 0.0


def test_beta_perfectly_correlated():
    """Identical series should have beta of 1.0."""
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0, 0.01, 252))
    b = beta_vs_benchmark(returns, returns)
    assert abs(b - 1.0) < 1e-6


def test_beta_double_volatility():
    """A series that's exactly 2x another should have beta 2.0."""
    np.random.seed(0)
    bench = pd.Series(np.random.normal(0, 0.01, 252))
    port = bench * 2
    b = beta_vs_benchmark(port, bench)
    assert abs(b - 2.0) < 0.01


def test_best_worst_year():
    """Best year > worst year."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=750, freq="B")
    returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
    best, worst = best_worst_year(returns)
    assert best > worst


# === Full metrics orchestration ===

def test_compute_full_metrics_all_fields_present():
    """Full metrics should return all expected fields."""
    np.random.seed(0)
    dates = pd.date_range("2022-01-01", periods=750, freq="B")
    returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
    bench = pd.Series(np.random.normal(0.0003, 0.012, len(dates)), index=dates)

    metrics = compute_full_metrics(returns, bench)

    expected = {
        "total_return", "annualised_return", "annualised_volatility",
        "sharpe_ratio", "max_drawdown", "best_year_return",
        "worst_year_return", "n_trading_days", "beta_vs_benchmark",
        "risk_free_rate_used",
    }
    assert expected.issubset(set(metrics.keys()))
    assert metrics["risk_free_rate_used"] == RISK_FREE_RATE_ANNUAL
    assert metrics["n_trading_days"] == len(returns)


def test_compute_full_metrics_without_benchmark():
    """When no benchmark provided, beta is omitted (no error)."""
    returns = pd.Series([0.001] * 100)
    metrics = compute_full_metrics(returns)
    assert "beta_vs_benchmark" not in metrics
    # All other metrics still present
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
