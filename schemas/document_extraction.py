"""
Schema for document extraction outputs from Agent 4 (Document Intelligence).

Each extracted document produces a DocumentExtraction object containing
structured information that downstream agents (especially Agent 8 — 
Greenwashing) can compare against disclosed numerical data.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ExtractedClaim(BaseModel):
    """A single sustainability claim extracted from a document.

    Used for claim-evidence comparison in greenwashing detection.
    """
    text: str = Field(..., description="The claim, in the company's own words")
    category: str = Field(..., description="biodiversity / climate / water / supply_chain / other")
    target_year: Optional[int] = Field(
        None, description="Target completion year if a deadline is stated"
    )
    is_quantitative: bool = Field(
        False, description="True if the claim includes a specific numeric target"
    )
    source_page: Optional[int] = Field(
        None, description="Page number where the claim was found"
    )


class DocumentExtraction(BaseModel):
    """Structured information extracted from one company's sustainability document.

    Inputs: PDF document (or text)
    Outputs: Structured fields ready for cross-comparison with disclosed data
    """
    company_id: str = Field(..., description="Internal company ID, e.g. C00071")
    company_name: str = Field(..., description="Company name from the doc")
    document_path: str = Field(..., description="Path to source document")
    document_year: Optional[int] = Field(
        None, description="Fiscal year the document describes"
    )
    extraction_date: datetime = Field(default_factory=datetime.utcnow)

    # === Biodiversity ===
    biodiversity_commitments: List[ExtractedClaim] = Field(default_factory=list)
    tnfd_adopter: bool = Field(False, description="Has the company adopted TNFD?")
    no_deforestation_pledge: bool = Field(False)
    biodiversity_target_year: Optional[int] = Field(
        None, description="Year by which biodiversity targets must be met"
    )

    # === Climate ===
    climate_targets: List[ExtractedClaim] = Field(default_factory=list)
    net_zero_year: Optional[int] = Field(None)
    sbti_status: Optional[str] = Field(
        None, description="validated / committed / none / unknown"
    )

    # === Water ===
    water_disclosures: List[ExtractedClaim] = Field(default_factory=list)
    water_stress_disclosed: bool = Field(False)

    # === Supply chain ===
    forest_risk_commodities_mentioned: List[str] = Field(
        default_factory=list,
        description="Commodities mentioned: palm oil, soy, beef, etc.",
    )
    supply_chain_claims: List[ExtractedClaim] = Field(default_factory=list)

    # === Overall narrative ===
    top_sustainability_claims: List[ExtractedClaim] = Field(
        default_factory=list,
        description="3-5 main claims the company emphasises",
    )
    document_summary: str = Field(
        "", description="2-3 sentence summary of the document's key themes"
    )

    # === Quality flags ===
    extraction_confidence: str = Field(
        "medium", description="high / medium / low based on extraction quality"
    )
    extraction_notes: Optional[str] = Field(
        None, description="Any caveats from the extraction"
    )

    class Config:
        arbitrary_types_allowed = True
