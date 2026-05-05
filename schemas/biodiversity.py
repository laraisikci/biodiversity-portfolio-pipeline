"""
Biodiversity, nature-risk, and climate schemas — produced by Role C.

Aligned with TNFD's LEAP framework (Locate, Evaluate, Assess, Prepare).
The biodiversity score is multi-layered, not a single number — this is
deliberate, because biodiversity data is inherently multi-source.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from schemas.confidence import DataPoint


class ClimateMetrics(BaseModel):
    """Climate metrics per company — Role C's climate output."""

    company_id: str

    # Carbon footprint
    scope_1_emissions: Optional[DataPoint] = Field(
        None, description="tCO2e, direct emissions"
    )
    scope_2_emissions: Optional[DataPoint] = Field(
        None, description="tCO2e, indirect from purchased energy"
    )
    scope_3_emissions: Optional[DataPoint] = Field(
        None, description="tCO2e, value chain — often imputed"
    )
    carbon_intensity_per_revenue: DataPoint = Field(
        ..., description="tCO2e per EUR million revenue"
    )

    # Transition readiness
    sbti_validated: bool = Field(
        False, description="True if company has SBTi-validated targets"
    )
    sbti_target_year: Optional[int] = None
    transition_capex_share: Optional[DataPoint] = Field(
        None, description="Share of capex aligned with low-carbon transition"
    )

    # Coverage flag
    scope_3_imputed: bool = Field(
        False, description="True if Scope 3 was imputed by ML model"
    )


class BiodiversityRiskScore(BaseModel):
    """Multi-layered biodiversity risk score — Role C's primary output.

    Following the TNFD LEAP framework, we do NOT collapse biodiversity into
    a single number. We report multiple dimensions with their own confidence
    levels, then provide an aggregate for portfolio construction.
    """

    company_id: str

    # Layer 1 — Sector dependency (always available, from ENCORE)
    encore_dependency_score: DataPoint = Field(
        ..., description="Sector-based nature dependency, 0-1 scale"
    )
    encore_impact_score: DataPoint = Field(
        ..., description="Sector-based nature impact, 0-1 scale"
    )

    # Layer 2 — Geographic / location risk
    water_stress_score: Optional[DataPoint] = Field(
        None, description="From WRI Aqueduct, weighted by operations geography"
    )
    biodiversity_sensitive_areas_overlap: Optional[DataPoint] = Field(
        None, description="From WWF Biodiversity Risk Filter or IBAT"
    )

    # Layer 3 — Disclosure quality
    has_biodiversity_policy: Optional[bool] = None
    tnfd_adopter: bool = Field(False, description="On TNFD adopter list")
    cdp_water_score: Optional[DataPoint] = None
    cdp_forests_score: Optional[DataPoint] = None

    # Layer 4 — Forest-risk commodity exposure
    forest_risk_commodity_exposure: List[str] = Field(
        default_factory=list,
        description="Commodities sourced, e.g. ['palm_oil', 'soy', 'beef']",
    )
    forest_500_score: Optional[DataPoint] = Field(
        None, description="From Global Canopy Forest 500 if listed"
    )

    # Aggregate
    composite_biodiversity_score: DataPoint = Field(
        ...,
        description="Weighted aggregate of layers 1-4, 0-10 scale (10 = best)",
    )
    aggregation_method: str = Field(
        ..., description="Documented method, e.g. 'weighted average with layer 1 floor'"
    )

    # Constraint flag for portfolio construction
    biodiversity_exclusion_flag: bool = Field(
        False, description="True if biodiversity risk warrants exclusion"
    )
    exclusion_reason: Optional[str] = None
