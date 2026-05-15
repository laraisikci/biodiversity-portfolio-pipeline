"""
Financial Analysis utility.

Fetches historical prices via yfinance and computes standard portfolio
risk/return metrics:
  - Total return (cumulative)
  - Annualised return (CAGR)
  - Annualised volatility
  - Sharpe ratio
  - Max drawdown
  - Beta vs benchmark
  - Best year / worst year

Methodology note (for the report):
  This is a HISTORICAL BACKTEST of the May 2026 portfolio applied
  retroactively. It is NOT predictive of future returns. The selection
  methodology relies on EU Taxonomy data (introduced 2022) and ENCORE
  assessments (updated 2024), which were not fully available throughout
  the backtest period. Results are illustrative of risk characteristics,
  not historical performance.

Owner: Role A (with Analytics Advisor)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from agents.decision_log import log_decision


# === Configuration ===
# 3-month Euribor average — stable approximation for Sharpe calc
# In production this would be fetched live from ECB
RISK_FREE_RATE_ANNUAL = 0.025  # 2.5% — current 3-month Euribor area

# Benchmark ticker (Yahoo Finance symbol for EURO STOXX 50 index)
EURO_STOXX_50_TICKER = "^STOXX50E"

# Backtest window
BACKTEST_YEARS = 3
TRADING_DAYS_PER_YEAR = 252


# === Ticker mapping ===
# Bloomberg-style tickers don't match Yahoo Finance. We map common European
# exchange suffixes.
#
# Bloomberg "ABI BB Equity" -> Yahoo "ABI.BR"  (Brussels)
# Bloomberg "ASML NA Equity" -> Yahoo "ASML.AS" (Amsterdam)
# etc.

BLOOMBERG_TO_YAHOO_SUFFIX = {
    "BB": ".BR",   # Brussels (Euronext Brussels)
    "NA": ".AS",   # Amsterdam (Euronext Amsterdam)
    "GY": ".DE",   # Germany (XETRA)
    "FP": ".PA",   # Paris (Euronext Paris)
    "SQ": ".MC",   # Spain (Madrid)
    "IM": ".MI",   # Italy (Milan)
    "FH": ".HE",   # Finland (Helsinki)
    "LN": ".L",    # London Stock Exchange
    "SS": ".ST",   # Stockholm
    "SW": ".SW",   # Switzerland (SIX)
    "DC": ".CO",   # Denmark (Copenhagen)
    "NO": ".OL",   # Norway (Oslo)
}


def bloomberg_to_yahoo_ticker(bloomberg_ticker: str) -> Optional[str]:
    """Convert 'ABI BB Equity' -> 'ABI.BR' or 'ASML NA Equity' -> 'ASML.AS'.

    Returns None if format unrecognised.
    """
    if not bloomberg_ticker or not isinstance(bloomberg_ticker, str):
        return None
    parts = bloomberg_ticker.strip().split()
    if len(parts) < 2:
        return None
    symbol = parts[0]
    exchange = parts[1]
    suffix = BLOOMBERG_TO_YAHOO_SUFFIX.get(exchange)
    if suffix is None:
        return None
    return f"{symbol}{suffix}"


# === Price fetching ===

def fetch_price_history(
    yahoo_tickers: List[str],
    years: int = BACKTEST_YEARS,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Fetch daily Adjusted Close prices for a list of Yahoo tickers.

    Returns a DataFrame with dates as index, tickers as columns.
    Missing tickers are omitted with a warning logged.
    """
    import yfinance as yf

    end_date = end_date or datetime.now()
    start_date = end_date - timedelta(days=years * 366)

    log_decision(
        agent="financial_analysis",
        decision_type="price_fetch_start",
        details={
            "n_tickers": len(yahoo_tickers),
            "start": str(start_date.date()),
            "end": str(end_date.date()),
            "years": years,
        },
        confidence="reported",
        notes=(
            f"Fetching {years}y daily Adjusted Close from Yahoo Finance "
            f"for {len(yahoo_tickers)} tickers."
        ),
    )

    # yfinance has a multi-ticker download function — much faster than fetching one at a time
    try:
        data = yf.download(
            tickers=yahoo_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,  # adjusts for splits and dividends
        )
    except Exception as e:
        log_decision(
            agent="financial_analysis",
            decision_type="price_fetch_failed",
            details={"error": str(e)},
            confidence="observed",
        )
        return pd.DataFrame()

    # When fetching multiple tickers, yfinance returns a multi-index DataFrame.
    # We want just 'Close' (auto_adjust handles splits/dividends).
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        # Single ticker case
        prices = data[["Close"]] if "Close" in data.columns else data

    # Drop tickers with all NaN
    prices = prices.dropna(axis=1, how="all")

    log_decision(
        agent="financial_analysis",
        decision_type="price_fetch_complete",
        details={
            "n_tickers_returned": len(prices.columns),
            "n_trading_days": len(prices),
            "first_date": str(prices.index[0].date()) if len(prices) > 0 else None,
            "last_date": str(prices.index[-1].date()) if len(prices) > 0 else None,
        },
        confidence="reported",
    )

    return prices


# === Return calculations ===

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from price series."""
    return prices.pct_change().dropna(how="all")


def compute_portfolio_returns(
    daily_returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """Compute weighted daily portfolio returns.

    Args:
        daily_returns: DataFrame of per-ticker daily returns.
        weights: Dict of yahoo_ticker -> portfolio weight (sums to 1).

    Returns:
        Series of daily portfolio returns.
    """
    # Only use tickers we have both prices and weights for
    matched_tickers = [t for t in weights if t in daily_returns.columns]
    if not matched_tickers:
        return pd.Series(dtype=float)

    # Renormalise weights to sum to 1 across matched tickers
    matched_weights = pd.Series({t: weights[t] for t in matched_tickers})
    matched_weights = matched_weights / matched_weights.sum()

    # Weighted sum of returns per day
    portfolio_daily = (daily_returns[matched_tickers] * matched_weights).sum(axis=1)
    return portfolio_daily


# === Metric calculations ===

def annualised_return(daily_returns: pd.Series) -> float:
    """Annualised return (CAGR) from daily return series."""
    if len(daily_returns) == 0:
        return 0.0
    total_return = (1 + daily_returns).prod() - 1
    n_years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / n_years) - 1)


def annualised_volatility(daily_returns: pd.Series) -> float:
    """Annualised volatility from daily return series."""
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
) -> float:
    """Sharpe ratio = (annualised return - risk-free rate) / annualised volatility."""
    ann_ret = annualised_return(daily_returns)
    ann_vol = annualised_volatility(daily_returns)
    if ann_vol < 1e-9:  # effectively zero, avoid div by ~0
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown.

    Returns a negative number (e.g. -0.25 for a 25% drawdown).
    """
    if len(daily_returns) == 0:
        return 0.0
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def beta_vs_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Portfolio beta = Cov(portfolio, benchmark) / Var(benchmark)."""
    # Align dates
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 1.0
    aligned.columns = ["port", "bench"]
    var_bench = aligned["bench"].var()
    if var_bench == 0:
        return 1.0
    cov = aligned[["port", "bench"]].cov().iloc[0, 1]
    return float(cov / var_bench)


def best_worst_year(daily_returns: pd.Series) -> Tuple[float, float]:
    """Best and worst calendar-year returns."""
    if len(daily_returns) == 0:
        return 0.0, 0.0
    # Need a datetime index to group by year
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        # Fallback: treat entire series as one period
        total = float((1 + daily_returns).prod() - 1)
        return total, total
    yearly = (1 + daily_returns).groupby(daily_returns.index.year).prod() - 1
    return float(yearly.max()), float(yearly.min())


# === Top-level orchestration ===

def compute_full_metrics(
    daily_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
) -> Dict:
    """Compute all standard metrics for a return series.

    Args:
        daily_returns: Series of daily returns (portfolio or single stock).
        benchmark_returns: Optional benchmark return series (for beta).
        risk_free_rate: Annual risk-free rate for Sharpe.

    Returns:
        Dict with all metrics.
    """
    total_ret = float((1 + daily_returns).prod() - 1) if len(daily_returns) > 0 else 0.0
    ann_ret = annualised_return(daily_returns)
    ann_vol = annualised_volatility(daily_returns)
    sharpe = sharpe_ratio(daily_returns, risk_free_rate)
    mdd = max_drawdown(daily_returns)
    best, worst = best_worst_year(daily_returns)

    metrics = {
        "total_return": round(total_ret, 4),
        "annualised_return": round(ann_ret, 4),
        "annualised_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(mdd, 4),
        "best_year_return": round(best, 4),
        "worst_year_return": round(worst, 4),
        "n_trading_days": len(daily_returns),
        "risk_free_rate_used": risk_free_rate,
    }

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        metrics["beta_vs_benchmark"] = round(
            beta_vs_benchmark(daily_returns, benchmark_returns), 3
        )

    return metrics
