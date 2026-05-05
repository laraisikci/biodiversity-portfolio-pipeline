"""
Company identification schema.

Role A is responsible for producing the master CompanyUniverse from
equityBicsV2.csv. Every other role consumes this as input.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class CompanyBase(BaseModel):
    """A single company. The minimal identifier set."""

    company_id: str = Field(..., description="Internal canonical ID, e.g. 'C00001'")
    name: str
    isin: Optional[str] = Field(None, description="ISIN where available")
    ticker: Optional[str] = Field(None, description="Primary exchange ticker")
    yahoo_ticker: Optional[str] = Field(
        None, description="Yahoo Finance symbol with suffix, e.g. 'IBE.MC'"
    )
    country: str
    bics_level_1: str = Field(..., description="BICS top-level sector")
    bics_level_2: Optional[str] = None
    bics_level_3: Optional[str] = None
    market_cap_eur_m: Optional[float] = Field(
        None, description="Market cap in EUR millions"
    )

    class Config:
        # Allow slight schema flexibility for messy upstream CSV data
        extra = "allow"


class CompanyUniverse(BaseModel):
    """The full investable universe — produced by Role A's data ingestion agent."""

    companies: List[CompanyBase]
    universe_size: int
    geographic_filter: str = Field(..., description="e.g. 'Europe', 'STOXX 600'")
    excluded_count: int = Field(0, description="Count removed by initial filters")
    timestamp_built: str

    def get_company(self, company_id: str) -> Optional[CompanyBase]:
        for c in self.companies:
            if c.company_id == company_id:
                return c
        return None
