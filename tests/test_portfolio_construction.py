"""Tests for the Portfolio Construction agent."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from agents.portfolio_construction import (
    PortfolioConstructionAgent,
    COMPOSITE_WEIGHTS,
)
from agents.mandate import Mandate, MandateAgent
from agents.decision_log import read_log, LOG_PATH
from schemas.esg import ESGScore
from schemas.biodiversity import BiodiversityRiskScore, ClimateMetrics
from schemas.confidence import DataPoint, ConfidenceLevel
from schemas.portfolio import FinalPortfolio, PortfolioHolding


DATA_DIR = Path("data/raw")
DATA_AVAILABLE = (DATA_DIR / "equityBicsV2.csv").exists()


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def _make_synthetic_inputs(n=20):
    """Build a small synthetic universe with all required inputs."""
    sectors = ["Technology", "Financials", "Health Care", "Consumer Staples",
               "Utilities", "Industrials", "Communications"]
    countries = ["DE", "FR", "NL", "ES", "IT", "BE"]

    master = pd.DataFrame({
        "company_id": [f"C{i:05d}" for i in range(n)],
        "idBbGlobalCompanyName": [f"Company_{i}" for i in range(n)],
        "classificationLevelName1": [sectors[i % len(sectors)] for i in range(n)],
        "cntryOfDomicile": [countries[i % len(countries)] for i in range(n)],
    })

    esg_scores = []
    bio_scores = []
    climate_metrics = []

    for i in range(n):
        cid = f"C{i:05d}"
        # ESG composite varies from 3 to 9
        esg_val = 3.0 + (i / n) * 6.0
        esg_scores.append(ESGScore(
            company_id=cid,
            e_score=DataPoint(
                value=esg_val, unit="0-10", confidence=ConfidenceLevel.REPORTED,
                source="test", extraction_method="test", vintage=datetime.now(timezone.utc)
            ),
            s_score=DataPoint(
                value=esg_val, unit="0-10", confidence=ConfidenceLevel.REPORTED,
                source="test", extraction_method="test", vintage=datetime.now(timezone.utc)
            ),
            g_score=DataPoint(
                value=esg_val, unit="0-10", confidence=ConfidenceLevel.REPORTED,
                source="test", extraction_method="test", vintage=datetime.now(timezone.utc)
            ),
            composite_esg_score=DataPoint(
                value=esg_val, unit="0-10", confidence=ConfidenceLevel.REPORTED,
                source="test", extraction_method="test", vintage=datetime.now(timezone.utc)
            ),
            weighting_method="test",
            normalisation_method="test",
            sub_indicators_used={"E": [], "S": [], "G": []},
            exclusion_flag=False,
        ))

        bio_val = 3.0 + ((i + 5) % n / n) * 6.0
        bio_scores.append(BiodiversityRiskScore(
            company_id=cid,
            encore_dependency_score=DataPoint(
                value=0.3, unit="0-1", confidence=ConfidenceLevel.ESTIMATED,
                source="test", extraction_method="test", vintage=datetime.now(timezone.utc)
            ),
            encore_impact_score=DataPoint(
                value=0.3, unit="0-1", confidence=ConfidenceLevel.ESTIMATED,
                source="test", extraction_method="test", vintage=datetime.now(timezone.utc)
            ),
            forest_risk_commodity_exposure=[],
            composite_biodiversity_score=DataPoint(
                value=bio_val, unit="0-10", confidence=ConfidenceLevel.REPORTED,
                source="test", extraction_method="test", vintage=datetime.now(timezone.utc)
            ),
            aggregation_method="test",
            biodiversity_exclusion_flag=False,
        ))

        intensity = 50 + (i * 20)  # 50 to 50 + 20*n
        climate_metrics.append(ClimateMetrics(
            company_id=cid,
            scope_1_emissions=DataPoint(
                value=10000.0, unit="tCO2e",
                confidence=ConfidenceLevel.REPORTED,
                source="test", extraction_method="test", vintage=datetime.now(timezone.utc)
            ),
            carbon_intensity_per_revenue=DataPoint(
                value=float(intensity), unit="tCO2e/€m",
                confidence=ConfidenceLevel.REPORTED,
                source="test", extraction_method="test", vintage=datetime.now(timezone.utc)
            ),
            sbti_validated=True,
            scope_3_imputed=False,
        ))

    return master, esg_scores, bio_scores, climate_metrics


def test_composite_weights_sum_to_one():
    """The portfolio composite weights must sum to 1.0."""
    total = sum(COMPOSITE_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001


def test_portfolio_construction_on_synthetic_data():
    """Agent should produce a FinalPortfolio with valid weights."""
    master, esg, bio, climate = _make_synthetic_inputs(n=20)
    mandate = MandateAgent().run()

    agent = PortfolioConstructionAgent()
    portfolio = agent.run(master, mandate, esg, bio, climate)

    assert isinstance(portfolio, FinalPortfolio)
    assert len(portfolio.holdings) >= mandate.n_holdings_min
    assert len(portfolio.holdings) <= mandate.n_holdings_max
    # Weights should sum to ~1.0
    total_weight = sum(h.weight for h in portfolio.holdings)
    assert abs(total_weight - 1.0) < 0.01


def test_portfolio_respects_single_name_cap():
    """No holding should exceed max_single_name_weight."""
    master, esg, bio, climate = _make_synthetic_inputs(n=20)
    mandate = MandateAgent().run()

    agent = PortfolioConstructionAgent()
    portfolio = agent.run(master, mandate, esg, bio, climate)

    for h in portfolio.holdings:
        assert h.weight <= mandate.max_single_name_weight + 0.001, (
            f"{h.company_name} weight {h.weight} > cap {mandate.max_single_name_weight}"
        )


def test_portfolio_factsheet_generates():
    """Factsheet should be generated as a Markdown string."""
    master, esg, bio, climate = _make_synthetic_inputs(n=20)
    mandate = MandateAgent().run()

    agent = PortfolioConstructionAgent()
    portfolio = agent.run(master, mandate, esg, bio, climate)
    factsheet = agent.build_factsheet(portfolio)

    # Required sections
    assert "Portfolio Factsheet" in factsheet
    assert "Mandate" in factsheet
    assert "Top Holdings" in factsheet
    assert "Sector Allocation" in factsheet
    assert "Exclusions" in factsheet
    assert "Limitations" in factsheet
    # Should have some content
    assert len(factsheet) > 500


def test_portfolio_factsheet_can_save_to_file(tmp_path):
    """Factsheet should write to disk when path provided."""
    master, esg, bio, climate = _make_synthetic_inputs(n=20)
    mandate = MandateAgent().run()

    agent = PortfolioConstructionAgent()
    portfolio = agent.run(master, mandate, esg, bio, climate)

    output_path = tmp_path / "factsheet.md"
    agent.build_factsheet(portfolio, output_path=output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Portfolio Factsheet" in content


def test_portfolio_logs_decisions():
    """Construction should write to the audit trail."""
    master, esg, bio, climate = _make_synthetic_inputs(n=20)
    mandate = MandateAgent().run()

    agent = PortfolioConstructionAgent()
    agent.run(master, mandate, esg, bio, climate)

    log = read_log()
    decision_types = {entry["decision_type"] for entry in log}
    assert "portfolio_construction_start" in decision_types
    assert "portfolio_constructed" in decision_types


def test_portfolio_respects_per_sector_cap():
    """No more than max_holdings_per_sector should be from any one sector."""
    # Build a universe where one sector has many candidates
    master = pd.DataFrame({
        "company_id": [f"C{i:05d}" for i in range(20)],
        "idBbGlobalCompanyName": [f"Company_{i}" for i in range(20)],
        # 10 Financials (the over-concentrated sector) + 10 other sectors
        "classificationLevelName1": (
            ["Financials"] * 10
            + ["Technology", "Health Care", "Industrials", "Consumer Discretionary",
               "Communications", "Utilities", "Real Estate", "Consumer Staples",
               "Materials", "Energy"]
        ),
        "cntryOfDomicile": ["DE"] * 20,
    })

    # All companies score the same so ranking is by index order — Financials
    # will appear first, then others
    _, esg, bio, climate = _make_synthetic_inputs(20)
    # Override the master with our test setup
    for i, score in enumerate(esg):
        # All ESG scores high so nothing is excluded
        score.exclusion_flag = False

    mandate = MandateAgent().run()
    agent = PortfolioConstructionAgent()
    portfolio = agent.run(master, mandate, esg, bio, climate)

    # Count holdings per sector
    sector_counts = {}
    for h in portfolio.holdings:
        sector_counts[h.sector_allocation] = sector_counts.get(h.sector_allocation, 0) + 1

    # No sector should have more than max_holdings_per_sector
    for sector, count in sector_counts.items():
        assert count <= mandate.max_holdings_per_sector, (
            f"{sector} has {count} holdings, exceeds cap of {mandate.max_holdings_per_sector}"
        )


def test_mandate_has_max_holdings_per_sector():
    """The mandate schema should expose max_holdings_per_sector as a constraint."""
    mandate = MandateAgent().run()
    assert hasattr(mandate, "max_holdings_per_sector")
    assert mandate.max_holdings_per_sector == 4
    # Should be consistent with sector weight cap at equal weighting
    # 4 holdings × ~5% each = 20% sector
    assert mandate.max_holdings_per_sector * 0.05 <= mandate.max_sector_weight + 0.01


def test_portfolio_excludes_data_poor_companies():
    """Companies flagged for exclusion by upstream agents should not appear in portfolio."""
    master, esg, bio, climate = _make_synthetic_inputs(n=20)

    # Mark a few companies for exclusion
    for i in range(3):
        esg[i].exclusion_flag = True
        esg[i].exclusion_reason = "Insufficient ESG data"
        bio[i + 3].biodiversity_exclusion_flag = True
        bio[i + 3].exclusion_reason = "High biodiversity risk"

    mandate = MandateAgent().run()
    agent = PortfolioConstructionAgent()
    portfolio = agent.run(master, mandate, esg, bio, climate)

    holding_ids = {h.company_id for h in portfolio.holdings}
    # Excluded companies should not be in the portfolio
    for i in range(3):
        assert f"C{i:05d}" not in holding_ids
    for i in range(3, 6):
        assert f"C{i:05d}" not in holding_ids


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Course data not in data/raw/")
def test_portfolio_on_real_data():
    """End-to-end on the real EURO STOXX 50 universe."""
    from agents.data_ingestion import DataIngestionAgent
    from agents.esg_scoring import ESGScoringAgent
    from agents.climate import ClimateAgent
    from agents.biodiversity import BiodiversityAgent
    from agents.bloomberg_integration import (
        load_eurostoxx50_constituents,
        filter_master_to_eurostoxx50,
    )

    ingest = DataIngestionAgent()
    master, _ = ingest.run(fetch_prices=False)

    constituents = load_eurostoxx50_constituents()
    filtered, _ = filter_master_to_eurostoxx50(master, constituents)

    esg = ESGScoringAgent().run(filtered)
    climate = ClimateAgent().run(filtered)
    bio = BiodiversityAgent().run(filtered)
    mandate = MandateAgent().run()

    agent = PortfolioConstructionAgent()
    portfolio = agent.run(filtered, mandate, esg, bio, climate)

    assert len(portfolio.holdings) >= 5
    assert len(portfolio.holdings) <= mandate.n_holdings_max
    # Weights sum to 1
    total = sum(h.weight for h in portfolio.holdings)
    assert abs(total - 1.0) < 0.05
