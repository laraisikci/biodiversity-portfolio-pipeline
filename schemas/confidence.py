"""
Confidence levels for every data point in the pipeline.

The brief (section 15) requires each variable in the data dictionary to be
classified as: reported / observed / estimated / AI-extracted / judgement-based.

This module enforces that taxonomy. Every data point flowing through the
pipeline carries its confidence level. This is what makes our audit trail
defensible.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ConfidenceLevel(str, Enum):
    """The five confidence levels from the assignment brief, section 15."""

    REPORTED = "reported"           # Disclosed by the company itself
    OBSERVED = "observed"           # Measured externally (e.g. satellite, market price)
    ESTIMATED = "estimated"         # Modelled / imputed (e.g. RF imputation)
    AI_EXTRACTED = "ai_extracted"   # Pulled from documents by an LLM
    JUDGEMENT = "judgement_based"   # Analyst or team override


class DataPoint(BaseModel):
    """A single data point with its provenance and confidence.

    Every meaningful number in the pipeline should be wrapped in this.
    Yes, it's verbose. Yes, it's worth it. The audit trail is the grade.
    """

    value: Any = Field(..., description="The actual value")
    unit: Optional[str] = Field(None, description="e.g. 'tCO2e', '%', 'EUR'")
    confidence: ConfidenceLevel
    source: str = Field(..., description="Where this came from, e.g. 'CDP 2024'")
    extraction_method: str = Field(
        ...,
        description="How it was obtained, e.g. 'Bloomberg API', 'Claude extraction', 'KNN imputation'",
    )
    vintage: Optional[datetime] = Field(
        None, description="When the underlying data was reported/measured"
    )
    notes: Optional[str] = Field(
        None, description="Any caveats or methodology notes"
    )

    class Config:
        use_enum_values = True
