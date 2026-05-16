"""
Agent 1: Mandate Agent.

Defines and documents the investment mandate before any scoring or
portfolio construction happens. This is what the lecturer's slide
called out: "forces the group to define what 'sustainable' means
before scoring anything."

The mandate is encoded as a structured Pydantic object so that:
1. Downstream agents (portfolio construction, exclusions) can read it
2. The audit trail captures what we committed to up front
3. The report can render the mandate cleanly

Owner: Role B
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from agents.base import BaseAgent


class Mandate(BaseModel):
    """The investment mandate. Set once, referenced everywhere downstream."""

    # Client
    client_name: str
    client_type: str = Field(..., description="e.g. 'foundation', 'pension fund'")
    client_mission: str

    # Universe
    universe_description: str
    geographic_scope: List[str] = Field(..., description="Country codes, e.g. ['ES', 'FR', 'DE']")
    universe_size_target: str = Field(..., description="e.g. '50-80 candidates'")

    # Portfolio structure
    portfolio_type: str = Field(default="long_only_equity")
    n_holdings_min: int = 15
    n_holdings_max: int = 25
    time_horizon: str

    # Benchmark
    benchmark_name: str
    benchmark_weighting: str = Field(..., description="e.g. 'equal_weighted', 'cap_weighted'")
    benchmark_rationale: str

    # Sustainability ambition
    sustainability_objectives: List[str] = Field(
        ..., description="High-level objectives, e.g. ['minimise biodiversity risk']"
    )
    sustainability_frameworks: List[str] = Field(
        ..., description="Frameworks we anchor to, e.g. ['TNFD LEAP', 'EU Taxonomy', 'ENCORE']"
    )

    # Risk constraints
    max_single_name_weight: float = Field(..., ge=0, le=1, description="e.g. 0.08 for 8%")
    max_sector_weight: float = Field(..., ge=0, le=1, description="e.g. 0.20 for 20%")
    max_holdings_per_sector: int = Field(
        default=4,
        ge=1,
        description=(
            "Maximum number of holdings per BICS Level 1 sector at the "
            "selection stage. Derived from max_sector_weight at equal weighting: "
            "max_sector_weight / per_holding_weight."
        ),
    )
    min_sector_count: int = Field(5, description="Minimum sectors represented")
    carbon_intensity_cap_vs_benchmark: float = Field(
        ..., description="e.g. 0.8 = 80% of benchmark carbon intensity"
    )

    # Exclusions (hard exclusions before scoring)
    sector_exclusions: List[str] = Field(
        default_factory=list,
        description="BICS sectors to exclude entirely, e.g. ['Mining', 'Oil & Gas']"
    )
    activity_exclusions: List[str] = Field(
        default_factory=list,
        description="Activities, e.g. ['thermal coal', 'industrial agriculture']"
    )

    # Disclosure framing — critical per assignment brief
    disclosure_framing: str = Field(
        default=(
            "Academic prototype for ESADE Sustainable Finance coursework. "
            "Not a regulated investment product, financial advice, "
            "or compliant Article 8/9 fund."
        )
    )


class MandateAgent(BaseAgent):
    """Encodes the investment mandate as a structured Mandate object."""

    name = "mandate"

    def run(self, mandate_config: Optional[dict] = None) -> Mandate:
        """Build the mandate. If no config passed, returns the canonical mandate
        from our 8 May proposal.

        Args:
            mandate_config: Optional dict to override defaults. Useful for
                sensitivity analysis (e.g. tighter exclusions).

        Returns:
            A validated Mandate object.
        """
        # Canonical mandate from the proposal — Prince Albert II of Monaco Foundation
        defaults = {
            "client_name": "Prince Albert II of Monaco Foundation",
            "client_type": "environmental foundation",
            "client_mission": (
                "Environmental protection and sustainable development with a "
                "particular focus on biodiversity preservation, climate change, "
                "and water resource management."
            ),
            "universe_description": (
                "STOXX Europe 600 subset, screened to companies with material "
                "ESG disclosure and excluding hard-line sector exclusions."
            ),
            "geographic_scope": [
                "GB", "FR", "DE", "ES", "IT", "NL", "CH", "SE", "BE", "IE",
                "DK", "FI", "NO", "AT", "PT", "PL", "GR", "LU",
            ],
            "universe_size_target": "50-80 candidates analysed in full pipeline, 15-25 final",
            "portfolio_type": "long_only_equity",
            "n_holdings_min": 15,
            "n_holdings_max": 25,
            "time_horizon": "5-10 years (medium to long term)",
            "benchmark_name": "STOXX Europe 600",
            "benchmark_weighting": "equal_weighted",
            "benchmark_rationale": (
                "Equal weighting avoids large-cap bias and allows higher-growth "
                "mid-cap sustainability leaders to contribute meaningfully, "
                "though it comes with higher tracking error versus a "
                "cap-weighted index."
            ),
            "sustainability_objectives": [
                "Minimise biodiversity and nature-related risk exposure",
                "Manage portfolio carbon intensity below benchmark",
                "Avoid high-impact extractive and industrial agriculture sectors",
                "Maintain financial credibility and diversification",
                "Resist greenwashing through structured claim-evidence testing",
            ],
            "sustainability_frameworks": [
                "TNFD LEAP (Locate, Evaluate, Assess, Prepare)",
                "ENCORE (sector dependencies and impacts)",
                "WWF Biodiversity Risk Filter",
                "EU Taxonomy (eligibility, potential alignment, reported alignment)",
                "WACI (Weighted Average Carbon Intensity)",
                "SBTi-validated climate targets",
            ],
            "max_single_name_weight": 0.08,
            "max_sector_weight": 0.20,
            "max_holdings_per_sector": 4,
            "min_sector_count": 5,
            "carbon_intensity_cap_vs_benchmark": 0.80,
            "sector_exclusions": [
                "Energy",  # Maps to oil & gas at BICS Level 1
                "Materials/Mining",  # Capture mining via BICS Level 2 in practice
            ],
            "activity_exclusions": [
                "thermal coal",
                "oil & gas extraction",
                "industrial agriculture",
                "controversial weapons",
            ],
        }

        # Override with caller's config if provided
        if mandate_config:
            defaults.update(mandate_config)

        mandate = Mandate(**defaults)

        # Log this decision — the audit trail starts here
        self.log(
            decision_type="mandate_defined",
            details={
                "client": mandate.client_name,
                "n_holdings_range": f"{mandate.n_holdings_min}-{mandate.n_holdings_max}",
                "benchmark": mandate.benchmark_name,
                "max_single_name": mandate.max_single_name_weight,
                "max_sector": mandate.max_sector_weight,
                "max_holdings_per_sector": mandate.max_holdings_per_sector,
                "carbon_cap_pct_of_benchmark": mandate.carbon_intensity_cap_vs_benchmark,
                "sector_exclusions_count": len(mandate.sector_exclusions),
                "frameworks_count": len(mandate.sustainability_frameworks),
            },
            confidence="judgement_based",
            notes="Mandate encoded from 8 May proposal. Approved by team.",
        )

        return mandate

    def summarise(self, mandate: Mandate) -> str:
        """Produce a human-readable summary of the mandate.

        Useful for the report and for sanity-checking the inputs.
        """
        lines = [
            f"=== Investment Mandate ===",
            f"Client: {mandate.client_name} ({mandate.client_type})",
            f"Universe: {mandate.universe_description}",
            f"Benchmark: {mandate.benchmark_name} ({mandate.benchmark_weighting})",
            f"Holdings: {mandate.n_holdings_min}-{mandate.n_holdings_max} long-only equities",
            f"Time horizon: {mandate.time_horizon}",
            f"",
            f"Sustainability objectives:",
        ]
        for obj in mandate.sustainability_objectives:
            lines.append(f"  • {obj}")
        lines.append(f"")
        lines.append(f"Frameworks anchored:")
        for fw in mandate.sustainability_frameworks:
            lines.append(f"  • {fw}")
        lines.append(f"")
        lines.append(f"Constraints:")
        lines.append(f"  • Max {mandate.max_single_name_weight*100:.0f}% single name")
        lines.append(f"  • Max {mandate.max_sector_weight*100:.0f}% per sector")
        lines.append(f"  • Max {mandate.max_holdings_per_sector} holdings per sector")
        lines.append(f"  • Min {mandate.min_sector_count} sectors represented")
        lines.append(
            f"  • Carbon intensity ≤ {mandate.carbon_intensity_cap_vs_benchmark*100:.0f}% of benchmark"
        )
        lines.append(f"")
        lines.append(f"Hard exclusions:")
        for s in mandate.sector_exclusions:
            lines.append(f"  • Sector: {s}")
        for a in mandate.activity_exclusions:
            lines.append(f"  • Activity: {a}")
        lines.append(f"")
        lines.append(f"Framing: {mandate.disclosure_framing}")
        return "\n".join(lines)
