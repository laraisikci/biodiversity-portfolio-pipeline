"""
Pydantic schemas defining the data contracts between agents.

Every agent MUST validate its outputs against these schemas before passing
data downstream. This is what makes the pipeline auditable and reproducible.

If you need a new field, add it here and tell the team in Slack/WhatsApp.
Do NOT silently invent fields in your own agent.
"""

from schemas.company import CompanyBase, CompanyUniverse
from schemas.financial import FinancialMetrics
from schemas.esg import ESGScore, GovernanceScore
from schemas.biodiversity import BiodiversityRiskScore, ClimateMetrics
from schemas.greenwashing import GreenwashingFlag, DocumentEvidence
from schemas.portfolio import PortfolioHolding, FinalPortfolio
from schemas.confidence import ConfidenceLevel, DataPoint

__all__ = [
    "CompanyBase",
    "CompanyUniverse",
    "FinancialMetrics",
    "ESGScore",
    "GovernanceScore",
    "BiodiversityRiskScore",
    "ClimateMetrics",
    "GreenwashingFlag",
    "DocumentEvidence",
    "PortfolioHolding",
    "FinalPortfolio",
    "ConfidenceLevel",
    "DataPoint",
]
