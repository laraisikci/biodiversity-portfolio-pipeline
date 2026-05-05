"""
Agent stubs — one per role.

These are placeholders. Each role owner replaces the body of their
stub with real implementation in Phase 2 (8-15 May).

The signatures and schemas are agreed contracts. Don't change them
without telling the team.
"""

from typing import List, Dict, Any
from agents.base import BaseAgent


# ============================================================
# ROLE A — Data ingestion + financial analysis
# ============================================================

class DataIngestionAgent(BaseAgent):
    """Loads CSV data pack + fetches market data via yfinance.

    Owner: Role A
    Inputs: paths to CSV files
    Outputs: CompanyUniverse + raw market data DataFrame
    """
    name = "data_ingestion"

    def run(self, csv_paths: Dict[str, str]) -> Any:
        # TODO Role A: implement CSV loading, ticker mapping, yfinance fetch
        # Remember to log every decision and cache the yfinance results.
        raise NotImplementedError("Role A — implement me")


class FinancialAnalysisAgent(BaseAgent):
    """Computes returns, volatility, drawdown, Sharpe per company.

    Owner: Role A
    Inputs: market data DataFrame
    Outputs: List[FinancialMetrics]
    """
    name = "financial_analysis"

    def run(self, market_data) -> Any:
        # TODO Role A
        raise NotImplementedError("Role A — implement me")


# ============================================================
# ROLE B — ESG scoring
# ============================================================

class MandateAgent(BaseAgent):
    """Defines the investment mandate (client, benchmark, constraints).

    Owner: Role B
    Inputs: human-defined mandate config
    Outputs: structured mandate object passed to Role E
    """
    name = "mandate"

    def run(self, mandate_config: Dict) -> Any:
        # TODO Role B
        raise NotImplementedError("Role B — implement me")


class ESGScoringAgent(BaseAgent):
    """Builds transparent E, S, G and composite scores per company.

    Owner: Role B
    Inputs: ESG raw data + sector classifications
    Outputs: List[ESGScore]
    """
    name = "esg_scoring"

    def run(self, esg_data, sector_data) -> Any:
        # TODO Role B
        raise NotImplementedError("Role B — implement me")


# ============================================================
# ROLE C — Biodiversity + climate
# ============================================================

class BiodiversityAgent(BaseAgent):
    """Multi-layered biodiversity risk scoring.

    Layers:
        1. ENCORE sector dependency/impact (always available)
        2. WRI Aqueduct water risk (location-based)
        3. WWF Biodiversity Risk Filter (location-based)
        4. Forest 500 / commodity exposure

    Owner: Role C
    Outputs: List[BiodiversityRiskScore]
    """
    name = "biodiversity"

    def run(self, companies, sector_data) -> Any:
        # TODO Role C
        raise NotImplementedError("Role C — implement me")


class ClimateAgent(BaseAgent):
    """Carbon intensity, WACI, transition readiness.

    Owner: Role C
    Outputs: List[ClimateMetrics]
    """
    name = "climate"

    def run(self, emissions_data, company_data) -> Any:
        # TODO Role C
        raise NotImplementedError("Role C — implement me")


# ============================================================
# ROLE D — Document intelligence + greenwashing
# ============================================================

class DocumentIntelligenceAgent(BaseAgent):
    """Extracts structured claims from sustainability reports.

    Owner: Role D
    Tools: Claude (with Pydantic schema), HF zero-shot classifier as fallback
    Outputs: List[DocumentEvidence]
    """
    name = "document_intelligence"

    def run(self, document_paths: List[str]) -> Any:
        # TODO Role D
        raise NotImplementedError("Role D — implement me")


class GreenwashingAgent(BaseAgent):
    """Greenwashing classifier — trained ML model.

    Owner: Role D + Analytics Advisor (modelling support)
    Tools: scikit-learn LogReg + features from claim-evidence framework
    Outputs: List[GreenwashingFlag]
    """
    name = "greenwashing"

    def run(self, evidence: List, structured_data) -> Any:
        # TODO Role D + Analytics Advisor
        raise NotImplementedError("Role D — implement me")


# ============================================================
# ROLE E — Master portfolio + human review + reporting
# ============================================================

class PortfolioConstructionAgent(BaseAgent):
    """Synthesises all scores into final portfolio with weights.

    Owner: Role E
    Tools: cvxpy for constrained optimisation
    Inputs: ESGScore + BiodiversityRiskScore + ClimateMetrics + GreenwashingFlag + FinancialMetrics
    Outputs: FinalPortfolio
    """
    name = "portfolio_construction"

    def run(
        self,
        esg_scores,
        biodiversity_scores,
        climate_metrics,
        greenwashing_flags,
        financial_metrics,
        mandate,
    ) -> Any:
        # TODO Role E
        raise NotImplementedError("Role E — implement me")


class RedTeamAgent(BaseAgent):
    """Adversarially challenges the constructed portfolio.

    Owner: Analytics Advisor (cross-cutting differentiator)
    Outputs: List of weaknesses, stress-test results, alternative recommendations
    """
    name = "red_team"

    def run(self, portfolio) -> List[str]:
        # TODO Analytics Advisor
        raise NotImplementedError("Analytics Advisor — implement me")


class ReportingAgent(BaseAgent):
    """Assembles factsheet, methodology report, AI Use Statement.

    Owner: Role E
    Tools: Claude for narrative drafting, Python for tables/charts
    """
    name = "reporting"

    def run(self, portfolio, decision_log) -> Any:
        # TODO Role E
        raise NotImplementedError("Role E — implement me")
