"""
Greenwashing detection and document evidence schemas — produced by Role D.

Built around the 8-test claim-evidence framework from lecture slide 74:
specificity, metric, baseline, target, time horizon, scope, verification, consistency.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from schemas.confidence import DataPoint


class DocumentEvidence(BaseModel):
    """A single piece of evidence extracted from a document."""

    company_id: str
    document_type: Literal[
        "annual_report",
        "sustainability_report",
        "tcfd_disclosure",
        "tnfd_disclosure",
        "press_release",
        "regulatory_filing",
        "ngo_report",
        "news_article",
        "other",
    ]
    document_source: str = Field(..., description="URL or filename")
    document_year: Optional[int] = None

    # The claim itself
    claim_text: str = Field(..., description="Verbatim quote from document")
    claim_topic: Literal[
        "carbon",
        "biodiversity",
        "water",
        "deforestation",
        "human_rights",
        "governance",
        "general_sustainability",
        "other",
    ]

    # The 8-test framework from slide 74
    has_specific_metric: bool
    has_baseline: bool
    has_target_year: bool
    has_third_party_verification: bool
    scope_clarity: Literal["clear", "ambiguous", "absent"]
    consistency_with_capex: Optional[Literal["consistent", "inconsistent", "unknown"]] = None

    # Extraction provenance
    extraction_method: str = Field(
        ..., description="e.g. 'Claude prompt v3', 'manual review'"
    )


class GreenwashingFlag(BaseModel):
    """Greenwashing assessment per company — Role D's primary output.

    Implements Option D methodology (Lecture 5, slide 43):
    - 6 rule-based signals derived from claim-evidence gaps
    - Logistic Regression calibration of signal count -> probability
    - Output: low/med/high flag per data dictionary slide 31
    """

    company_id: str

    # === Risk flag (matches data dictionary slide 31) ===
    risk_flag: Literal["low", "medium", "high"] = Field(
        ..., description="Categorical risk level per slide 31 data dictionary"
    )

    # === Signal-level diagnostics (Lecture 5, slide 43 red flags) ===
    signal_net_zero_without_sbti: bool = Field(
        False, description="Claims net-zero but no SBTi validation"
    )
    signal_nature_claim_without_disclosure: bool = Field(
        False, description="TNFD/biodiversity claim but no specific metrics"
    )
    signal_taxonomy_eligibility_only: bool = Field(
        False, description="Claims taxonomy alignment but only has eligibility data"
    )
    signal_rating_divergence: bool = Field(
        False, description="ESG rating divergence > 1 standard deviation across MSCI/Sustainalytics/S&P/RepRisk"
    )
    signal_forest_commodity_gap: bool = Field(
        False, description="Forest-risk commodity exposure without specific commodity-level targets"
    )
    signal_transition_capex_gap: bool = Field(
        False, description="Claims transition leadership but carbon intensity >= sector median"
    )
    signal_vague_leadership_claim: bool = Field(
        False,
        description=(
            "Slide 27 pattern: aspirational leadership language ('leader', '#1', "
            "'most sustainable') without underlying commitment infrastructure "
            "(no SBTi validation, no quantitative biodiversity targets)"
        ),
    )
    signals_fired: int = Field(0, description="Count of signals triggered (0-7)")

    # === Calibrated probability from Logistic Regression ===
    greenwashing_probability: DataPoint = Field(
        ..., description="0-1 calibrated probability from Logistic Regression"
    )
    classifier_confidence: Literal["low", "medium", "high"] = Field(
        ..., description="Based on input feature completeness"
    )

    # === Existing fields preserved for backwards compatibility ===
    claim_evidence_gap_score: Optional[DataPoint] = Field(
        None, description="Legacy: aggregate of 8-test failures (kept for backwards compat)"
    )
    vague_language_count: int = Field(0)
    quantitative_targets_count: int = Field(0)
    third_party_verifications_count: int = Field(0)

    # Evidence supporting the flag
    supporting_evidence: List[DocumentEvidence] = Field(
        default_factory=list,
        description="The specific claims that triggered concerns",
    )

    # Cross-check with structured data
    structured_data_inconsistencies: List[str] = Field(
        default_factory=list,
        description="e.g. ['Scope 3 not disclosed despite net-zero claim']",
    )

    # === Action recommendation ===
    flag_for_review: bool = Field(
        False, description="True if greenwashing probability above 0.5 (recommends human review)"
    )
    recommended_action: Literal[
        "include",
        "include_with_engagement",
        "watchlist",
        "exclude",
    ]
