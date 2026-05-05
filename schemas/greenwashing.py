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
    """Greenwashing assessment per company — Role D's primary output."""

    company_id: str

    # Probability score from the trained classifier
    greenwashing_probability: DataPoint = Field(
        ..., description="0-1 score from Logistic Regression classifier"
    )
    classifier_confidence: Literal["low", "medium", "high"] = Field(
        ..., description="Based on input feature completeness"
    )

    # Feature-level breakdown (interpretability)
    claim_evidence_gap_score: DataPoint = Field(
        ..., description="Aggregate of 8-test failures"
    )
    vague_language_count: int = Field(
        0, description="Count of vague terms like 'committed to', 'aiming for'"
    )
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

    # Flag for portfolio construction
    flag_for_review: bool = Field(
        False, description="True if greenwashing probability above threshold"
    )
    recommended_action: Literal[
        "include",
        "include_with_engagement",
        "watchlist",
        "exclude",
    ]
