"""
Final portfolio schemas — produced by Role E (Master Portfolio Agent).

These schemas hold the synthesised result: which companies got included,
at what weights, and why. The audit trail per holding is what makes the
portfolio defensible in the demo and Q&A.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class HumanOverride(BaseModel):
    """A documented case where the team overrode an AI output."""

    decision_point: str = Field(
        ..., description="Where the override happened, e.g. 'inclusion of Iberdrola'"
    )
    ai_recommendation: str
    human_decision: str
    rationale: str = Field(..., description="Why the human overrode")
    decided_by: str = Field(..., description="Team member name or role")
    timestamp: datetime


class PortfolioHolding(BaseModel):
    """A single holding in the final portfolio with full audit trail."""

    company_id: str
    company_name: str
    weight: float = Field(..., ge=0, le=1, description="Portfolio weight, 0-1")

    # Why this company is in the portfolio
    inclusion_rationale: str

    # All upstream scores (for the audit trail dashboard)
    composite_esg_score: float
    composite_biodiversity_score: float
    carbon_intensity: float
    greenwashing_probability: float

    # Constraint compliance
    sector_allocation: str
    country: str

    # Any human override on this holding
    overrides: List[HumanOverride] = Field(default_factory=list)


class FinalPortfolio(BaseModel):
    """The end deliverable. Role E's master output."""

    portfolio_name: str
    mandate_summary: str
    benchmark: str
    construction_date: datetime

    holdings: List[PortfolioHolding]
    excluded_companies: List[str] = Field(
        default_factory=list, description="Companies considered then excluded"
    )
    exclusion_reasons: dict = Field(
        default_factory=dict, description="Map company_id -> reason"
    )
    watchlist: List[str] = Field(
        default_factory=list, description="Companies on watchlist (not held but monitored)"
    )

    # Portfolio-level metrics vs benchmark
    portfolio_carbon_intensity: float
    benchmark_carbon_intensity: float = Field(
        ..., description="Primary benchmark (e.g. EBA reference)"
    )
    benchmark_source: str = Field(
        default="reference",
        description="Source of the benchmark intensity (e.g. 'EBA 2023 stress test')",
    )
    empirical_universe_median: Optional[float] = Field(
        default=None,
        description=(
            "Median carbon intensity of the empirical disclosing universe. "
            "Reported alongside the reference benchmark for transparency."
        ),
    )
    empirical_universe_size: Optional[int] = Field(
        default=None,
        description="Number of companies in the universe used for the empirical median.",
    )
    portfolio_esg_score: float
    portfolio_biodiversity_score: float

    # Risk/return
    expected_volatility: Optional[float] = None
    backtested_return_3y: Optional[float] = None
    backtested_max_drawdown: Optional[float] = None

    # Construction methodology
    optimisation_method: Literal["equal_weight", "ranked", "cvxpy_optimised"]
    constraints_applied: List[str] = Field(
        default_factory=list,
        description="e.g. ['max single-name 8%', 'min 5 sectors', 'carbon < benchmark']",
    )

    # Adversarial review
    red_team_findings: Optional[List[str]] = Field(
        None, description="Output of Red Team agent's challenge of the portfolio"
    )

    # All overrides logged
    all_overrides: List[HumanOverride] = Field(default_factory=list)
