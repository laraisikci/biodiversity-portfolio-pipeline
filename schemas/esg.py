"""
ESG and governance scoring schemas — produced by Role B.

The composite ESG score must be transparent and explainable. We do NOT
produce a single black-box number. Every E, S, G sub-score has documented
weights and inputs.
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from schemas.confidence import DataPoint


class GovernanceScore(BaseModel):
    """Governance sub-score detail for the data dictionary."""

    company_id: str
    board_independence_pct: Optional[DataPoint] = None
    executive_compensation_score: Optional[DataPoint] = None
    audit_quality_score: Optional[DataPoint] = None
    bribery_corruption_flags: int = Field(0, description="Count of incidents")
    composite_g_score: DataPoint = Field(
        ..., description="Aggregated governance score, 0-10 scale"
    )


class ESGScore(BaseModel):
    """Full ESG score per company — Role B's primary output."""

    company_id: str

    # Sub-scores (all on 0-10 scale, sector-normalised)
    e_score: DataPoint
    s_score: DataPoint
    g_score: DataPoint

    # Composite
    composite_esg_score: DataPoint = Field(
        ..., description="Weighted combination of E, S, G (weights documented)"
    )

    # Methodology fields — populated by Role B for the data dictionary
    weighting_method: str = Field(
        ..., description="e.g. 'sector-conditional via SASB materiality'"
    )
    normalisation_method: str = Field(
        ..., description="e.g. 'z-score per BICS L2 sector'"
    )
    sub_indicators_used: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Per-pillar list of underlying indicators, e.g. {'E': ['ghg_intensity', 'water_usage', ...]}",
    )

    # Exclusion flags — Role B can also flag for exclusion based on red lines
    exclusion_flag: bool = Field(
        False, description="True if company should be excluded based on ESG screen"
    )
    exclusion_reason: Optional[str] = None
