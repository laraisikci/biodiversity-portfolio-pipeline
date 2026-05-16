"""
Schema for human review overrides — produced by Agent 10.

Implements Lecture 5 slide 44 ("Human Review Layer") and slide 45 
("AI governance: what must be documented").

Key principle from the lecture: "The model proposes; the analyst disposes."
Every override must be logged with reviewer, timestamp, target, action, and
explicit justification. This is what makes the AI pipeline auditable.

Override types supported:
  - force_include: include a company despite low score / exclusion signal
  - force_exclude: exclude a company despite passing screens
  - change_weight: adjust position size
  - override_greenwashing_flag: reclassify the risk_flag
  - add_to_watchlist: flag for engagement instead of exclusion
  - remove_from_watchlist: clear engagement flag
"""

from typing import Optional, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field


OverrideAction = Literal[
    "force_include",
    "force_exclude",
    "change_weight",
    "override_greenwashing_flag",
    "add_to_watchlist",
    "remove_from_watchlist",
]


class OverrideDecision(BaseModel):
    """A single human override applied to an AI recommendation.

    Per slide 45 ("AI governance: what must be documented"):
      - tools used: implicit (the pipeline)
      - prompts: captured in decision_log
      - sources: captured in agent outputs
      - verification: this object
      - overrides: this object
      - limitations: captured in the report

    Per slide 44 ("Human Review Layer"):
      - Is the source correct?  →  reviewer must check
      - Do weights make sense?  →  reviewer must judge
      - Is the controversy material or noise?  →  reviewer must classify
      - Does it fit the mandate?  →  reviewer must validate

    Each OverrideDecision answers ONE of those questions for ONE company.
    """

    # === Who, when ===
    reviewer_id: str = Field(
        ..., description="Initials or name of the analyst making the override"
    )
    reviewer_role: Optional[str] = Field(
        None, description="e.g. 'Portfolio Manager', 'Sustainability Analyst'"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # === What is being overridden ===
    company_id: str = Field(..., description="Company affected by the override")
    company_name: Optional[str] = Field(None, description="Human-readable name")
    overrides_agent: Optional[str] = Field(
        None,
        description=(
            "Which agent's recommendation is being overridden "
            "(e.g. 'portfolio_construction', 'greenwashing', 'biodiversity')"
        ),
    )

    # === Action ===
    action: OverrideAction = Field(
        ..., description="The type of override being applied"
    )
    
    # === Action-specific parameters ===
    new_weight: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="For change_weight: the new portfolio weight (0.0-1.0)",
    )
    new_risk_flag: Optional[Literal["low", "medium", "high"]] = Field(
        None,
        description="For override_greenwashing_flag: the new flag value",
    )

    # === Required justification ===
    justification: str = Field(
        ...,
        min_length=20,
        description=(
            "Required prose explaining WHY the override was made. "
            "Minimum 20 chars to discourage 'because I said so' overrides. "
            "Should reference evidence, mandate, or methodology consideration."
        ),
    )

    # === Evidence pointers (optional but recommended) ===
    evidence_sources: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "URLs or document references supporting the override "
            "(e.g. CSR report page, news article, regulator notice)"
        ),
    )

    class Config:
        arbitrary_types_allowed = True


class HumanReviewSummary(BaseModel):
    """Summary of all overrides applied to the final portfolio.
    
    This is what gets included in the report appendix and the audit trail.
    """

    n_overrides_total: int = Field(0, description="Total overrides applied")
    n_force_include: int = Field(0)
    n_force_exclude: int = Field(0)
    n_change_weight: int = Field(0)
    n_override_flag: int = Field(0)
    n_watchlist_changes: int = Field(0)

    reviewers: list[str] = Field(
        default_factory=list,
        description="Unique reviewer IDs who applied overrides",
    )
    overrides: list[OverrideDecision] = Field(
        default_factory=list,
        description="All override decisions in chronological order",
    )

    summary_text: str = Field(
        "",
        description="Human-readable summary for the report appendix",
    )
