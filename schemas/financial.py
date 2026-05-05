"""
Financial metrics produced by Role A's financial analysis agent.

Inputs: market data from yfinance
Output: per-company financial profile
"""

from typing import Optional
from pydantic import BaseModel, Field
from schemas.confidence import DataPoint


class FinancialMetrics(BaseModel):
    """Per-company financial profile produced by Role A."""

    company_id: str

    # Returns
    annualised_return_3y: Optional[DataPoint] = None
    annualised_return_5y: Optional[DataPoint] = None

    # Risk
    annualised_volatility: DataPoint
    max_drawdown: DataPoint
    beta_vs_benchmark: Optional[DataPoint] = None
    sharpe_ratio: Optional[DataPoint] = None

    # Liquidity
    avg_daily_volume_eur_m: Optional[DataPoint] = None

    # Data quality flags
    price_data_complete: bool = Field(
        ..., description="True if 252+ trading days of clean price data"
    )
    data_gaps_flag: Optional[str] = Field(
        None, description="Description of any data quality issue"
    )
