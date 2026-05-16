"""
Mock data generators for integration testing.

These functions simulate what each role's agent will produce when fully
implemented. They return schema-compliant Pydantic objects so we can test
the pipeline end-to-end before any real agent is built.

Each role can use these to:
1. Understand exactly what their agent should output
2. Test their downstream agent without waiting for upstream agents to be done

If a real agent's output doesn't match what these mocks produce, the
schemas reject it and we know there's a contract mismatch.
"""

from datetime import datetime, timezone
from typing import List
import random

from schemas.confidence import DataPoint, ConfidenceLevel
from schemas.company import CompanyBase, CompanyUniverse
from schemas.financial import FinancialMetrics
from schemas.esg import ESGScore, GovernanceScore
from schemas.biodiversity import BiodiversityRiskScore, ClimateMetrics
from schemas.greenwashing import GreenwashingFlag, DocumentEvidence
from schemas.portfolio import PortfolioHolding, FinalPortfolio, HumanOverride


# Deterministic random for reproducible tests
rng = random.Random(42)


def mock_universe(n: int = 10) -> CompanyUniverse:
    """Role A produces this — a small European universe for testing."""
    sectors = ["Industrials", "Utilities", "Consumer Staples", "Health Care", "Financials"]
    countries = ["ES", "FR", "DE", "NL", "GB"]

    companies = []
    for i in range(n):
        companies.append(
            CompanyBase(
                company_id=f"C{i:05d}",
                name=f"TestCompany_{i}",
                isin=f"ES000000{i:04d}",
                ticker=f"TC{i}",
                yahoo_ticker=f"TC{i}.MC",
                country=rng.choice(countries),
                bics_level_1=rng.choice(sectors),
                market_cap_eur_m=round(rng.uniform(500, 50000), 2),
            )
        )

    return CompanyUniverse(
        companies=companies,
        universe_size=len(companies),
        geographic_filter="Europe",
        excluded_count=0,
        timestamp_built=datetime.now(timezone.utc).isoformat(),
    )


def mock_financial_metrics(company_id: str) -> FinancialMetrics:
    """Role A produces one of these per company."""
    return FinancialMetrics(
        company_id=company_id,
        annualised_volatility=DataPoint(
            value=round(rng.uniform(0.12, 0.35), 4),
            unit="annualised",
            confidence=ConfidenceLevel.OBSERVED,
            source="yfinance",
            extraction_method="returns_std_252d",
            vintage=datetime.now(timezone.utc),
        ),
        max_drawdown=DataPoint(
            value=round(rng.uniform(-0.45, -0.10), 4),
            unit="fraction",
            confidence=ConfidenceLevel.OBSERVED,
            source="yfinance",
            extraction_method="rolling_max_drawdown",
            vintage=datetime.now(timezone.utc),
        ),
        sharpe_ratio=DataPoint(
            value=round(rng.uniform(0.2, 1.5), 3),
            unit="ratio",
            confidence=ConfidenceLevel.OBSERVED,
            source="yfinance",
            extraction_method="sharpe_calc",
            vintage=datetime.now(timezone.utc),
        ),
        price_data_complete=True,
    )


def mock_esg_score(company_id: str) -> ESGScore:
    """Role B produces one of these per company."""
    e = round(rng.uniform(3, 9), 2)
    s = round(rng.uniform(3, 9), 2)
    g = round(rng.uniform(3, 9), 2)
    composite = round((e + s + g) / 3, 2)

    return ESGScore(
        company_id=company_id,
        e_score=DataPoint(
            value=e,
            unit="0-10 scale",
            confidence=ConfidenceLevel.ESTIMATED,
            source="esgEnvSocial CSV",
            extraction_method="sector_z_score_composite",
            vintage=datetime.now(timezone.utc),
        ),
        s_score=DataPoint(
            value=s,
            unit="0-10 scale",
            confidence=ConfidenceLevel.ESTIMATED,
            source="esgEnvSocial CSV",
            extraction_method="sector_z_score_composite",
            vintage=datetime.now(timezone.utc),
        ),
        g_score=DataPoint(
            value=g,
            unit="0-10 scale",
            confidence=ConfidenceLevel.ESTIMATED,
            source="esgGovernance CSV",
            extraction_method="sector_z_score_composite",
            vintage=datetime.now(timezone.utc),
        ),
        composite_esg_score=DataPoint(
            value=composite,
            unit="0-10 scale",
            confidence=ConfidenceLevel.ESTIMATED,
            source="derived",
            extraction_method="equal_weight_aggregation",
            vintage=datetime.now(timezone.utc),
        ),
        weighting_method="equal weight across pillars (prototype version)",
        normalisation_method="z-score within BICS Level 1 sector",
        sub_indicators_used={
            "E": ["ghg_intensity", "water_usage", "waste_management"],
            "S": ["board_diversity", "employee_safety", "supply_chain_audits"],
            "G": ["board_independence", "audit_quality", "executive_pay"],
        },
        exclusion_flag=False,
    )


def mock_climate_metrics(company_id: str) -> ClimateMetrics:
    """Role C's climate output per company."""
    scope1 = round(rng.uniform(1000, 500000), 0)
    scope2 = round(rng.uniform(500, 200000), 0)
    revenue = rng.uniform(500, 50000)
    intensity = round((scope1 + scope2) / revenue, 2)

    return ClimateMetrics(
        company_id=company_id,
        scope_1_emissions=DataPoint(
            value=scope1,
            unit="tCO2e",
            confidence=ConfidenceLevel.REPORTED,
            source="esgEnvSocial CSV",
            extraction_method="direct_field",
        ),
        scope_2_emissions=DataPoint(
            value=scope2,
            unit="tCO2e",
            confidence=ConfidenceLevel.REPORTED,
            source="esgEnvSocial CSV",
            extraction_method="direct_field",
        ),
        carbon_intensity_per_revenue=DataPoint(
            value=intensity,
            unit="tCO2e per EUR million",
            confidence=ConfidenceLevel.ESTIMATED,
            source="derived",
            extraction_method="scope_1_2_sum / revenue",
        ),
        sbti_validated=rng.random() > 0.7,
        scope_3_imputed=True,
    )


def mock_biodiversity_score(company_id: str) -> BiodiversityRiskScore:
    """Role C's biodiversity output per company."""
    encore_dep = round(rng.uniform(0.1, 0.9), 3)
    encore_imp = round(rng.uniform(0.1, 0.9), 3)
    composite = round(10 - (encore_dep + encore_imp) * 5, 2)  # higher score = better

    return BiodiversityRiskScore(
        company_id=company_id,
        encore_dependency_score=DataPoint(
            value=encore_dep,
            unit="0-1 scale",
            confidence=ConfidenceLevel.ESTIMATED,
            source="ENCORE sector mapping",
            extraction_method="BICS_to_ENCORE_lookup",
        ),
        encore_impact_score=DataPoint(
            value=encore_imp,
            unit="0-1 scale",
            confidence=ConfidenceLevel.ESTIMATED,
            source="ENCORE sector mapping",
            extraction_method="BICS_to_ENCORE_lookup",
        ),
        composite_biodiversity_score=DataPoint(
            value=composite,
            unit="0-10 scale (higher = better)",
            confidence=ConfidenceLevel.ESTIMATED,
            source="derived",
            extraction_method="layer_weighted_aggregation",
        ),
        aggregation_method="ENCORE layer dominant (prototype); location and disclosure layers added in v2",
        tnfd_adopter=rng.random() > 0.8,
        biodiversity_exclusion_flag=False,
    )


def mock_greenwashing_flag(company_id: str) -> GreenwashingFlag:
    """Role D's greenwashing output per company."""
    prob = round(rng.uniform(0.05, 0.6), 3)
    gap = round(rng.uniform(0, 8), 1)

    # Derive risk flag from probability (mirroring Agent 8 logic)
    if prob >= 0.65:
        risk_flag = "high"
    elif prob >= 0.35:
        risk_flag = "medium"
    else:
        risk_flag = "low"

    return GreenwashingFlag(
        company_id=company_id,
        risk_flag=risk_flag,
        greenwashing_probability=DataPoint(
            value=prob,
            unit="probability 0-1",
            confidence=ConfidenceLevel.AI_EXTRACTED,
            source="LogReg classifier on extracted claims",
            extraction_method="trained_logreg_v1",
        ),
        classifier_confidence="medium",
        claim_evidence_gap_score=DataPoint(
            value=gap,
            unit="failed_tests_count",
            confidence=ConfidenceLevel.AI_EXTRACTED,
            source="8-test framework",
            extraction_method="rule_based_scoring",
        ),
        vague_language_count=rng.randint(0, 15),
        quantitative_targets_count=rng.randint(0, 8),
        third_party_verifications_count=rng.randint(0, 3),
        supporting_evidence=[],  # populated in real implementation
        flag_for_review=prob > 0.5,
        recommended_action="include" if prob < 0.3 else ("watchlist" if prob < 0.6 else "exclude"),
    )
