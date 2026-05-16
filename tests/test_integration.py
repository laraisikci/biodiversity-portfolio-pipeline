"""
End-to-end integration test for the AI-agent pipeline.

This test simulates the full pipeline using mock outputs that match each
role's contract. It verifies that:

1. Every role's output is schema-compliant (Pydantic validates it)
2. The Master Portfolio Agent can consume all upstream outputs together
3. A valid FinalPortfolio object can actually be constructed
4. The decision log captures every step

How to use:
- Run this test as-is to confirm the contracts work
- When a role is ready, replace their mock with the real agent
- Run the test again — if it still passes, integration is solid
- If it fails, the error tells you exactly which schema is being violated

Run with: pytest tests/test_integration.py -v
"""

import pytest
from datetime import datetime, timezone
from typing import List

from agents.decision_log import log_decision, read_log, LOG_PATH

from schemas.portfolio import FinalPortfolio, PortfolioHolding
from schemas.company import CompanyUniverse

from tests.mock_outputs import (
    mock_universe,
    mock_financial_metrics,
    mock_esg_score,
    mock_climate_metrics,
    mock_biodiversity_score,
    mock_greenwashing_flag,
)


@pytest.fixture(autouse=True)
def clean_log():
    """Clear the decision log before each test."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def test_phase1_data_ingestion():
    """Role A: producing a valid universe."""
    universe = mock_universe(n=10)
    assert isinstance(universe, CompanyUniverse)
    assert universe.universe_size == 10
    assert len(universe.companies) == 10
    assert all(c.country in {"ES", "FR", "DE", "NL", "GB"} for c in universe.companies)

    log_decision(
        agent="integration_test",
        decision_type="universe_built",
        details={"size": universe.universe_size},
    )


def test_phase2_per_company_scoring():
    """Roles A, B, C: produce per-company metrics that round-trip the schemas."""
    universe = mock_universe(n=5)

    for company in universe.companies:
        # Each role's output should validate against its schema
        financial = mock_financial_metrics(company.company_id)
        esg = mock_esg_score(company.company_id)
        climate = mock_climate_metrics(company.company_id)
        biodiversity = mock_biodiversity_score(company.company_id)

        # All should reference the same company
        assert financial.company_id == company.company_id
        assert esg.company_id == company.company_id
        assert climate.company_id == company.company_id
        assert biodiversity.company_id == company.company_id

        # Scores should be in expected ranges
        assert 0 <= esg.composite_esg_score.value <= 10
        assert 0 <= biodiversity.composite_biodiversity_score.value <= 10
        assert climate.carbon_intensity_per_revenue.value > 0


def test_phase3_greenwashing():
    """Role D: greenwashing flags for the universe."""
    universe = mock_universe(n=5)
    flags = [mock_greenwashing_flag(c.company_id) for c in universe.companies]

    assert len(flags) == 5
    for flag in flags:
        assert 0 <= flag.greenwashing_probability.value <= 1
        assert flag.recommended_action in {
            "include", "include_with_engagement", "watchlist", "exclude"
        }


def test_full_pipeline_end_to_end():
    """The critical test: all roles produce outputs, Master Agent synthesises."""

    # Phase 1 — Role A: universe
    universe = mock_universe(n=10)
    log_decision(agent="integration_test", decision_type="phase1_complete",
                 details={"size": universe.universe_size})

    # Phase 2 — Roles A/B/C: per-company analytics
    financial = {c.company_id: mock_financial_metrics(c.company_id) for c in universe.companies}
    esg = {c.company_id: mock_esg_score(c.company_id) for c in universe.companies}
    climate = {c.company_id: mock_climate_metrics(c.company_id) for c in universe.companies}
    biodiversity = {c.company_id: mock_biodiversity_score(c.company_id) for c in universe.companies}

    # Phase 3 — Role D: greenwashing
    greenwashing = {c.company_id: mock_greenwashing_flag(c.company_id) for c in universe.companies}

    # Phase 4 — Role E: Master Portfolio Agent synthesises
    # Pick top 5 by composite score (for testing, use simple ranking)
    ranked = sorted(
        universe.companies,
        key=lambda c: (
            esg[c.company_id].composite_esg_score.value * 0.4
            + biodiversity[c.company_id].composite_biodiversity_score.value * 0.6
        ),
        reverse=True,
    )
    selected = ranked[:5]

    # Equal-weight (prototype version)
    weight_per_holding = 1.0 / len(selected)

    holdings: List[PortfolioHolding] = []
    for c in selected:
        holding = PortfolioHolding(
            company_id=c.company_id,
            company_name=c.name,
            weight=weight_per_holding,
            inclusion_rationale=(
                f"High biodiversity score "
                f"({biodiversity[c.company_id].composite_biodiversity_score.value:.2f}) "
                f"and acceptable ESG ({esg[c.company_id].composite_esg_score.value:.2f})"
            ),
            composite_esg_score=esg[c.company_id].composite_esg_score.value,
            composite_biodiversity_score=biodiversity[c.company_id].composite_biodiversity_score.value,
            carbon_intensity=climate[c.company_id].carbon_intensity_per_revenue.value,
            greenwashing_probability=greenwashing[c.company_id].greenwashing_probability.value,
            sector_allocation=c.bics_level_1,
            country=c.country,
        )
        holdings.append(holding)
        log_decision(
            agent="portfolio_construction",
            decision_type="holding_included",
            company_id=c.company_id,
            details={"weight": weight_per_holding, "rationale_summary": "top 5 by composite"},
        )

    # Portfolio-level aggregates
    avg_carbon = sum(climate[c.company_id].carbon_intensity_per_revenue.value for c in selected) / len(selected)
    avg_esg = sum(esg[c.company_id].composite_esg_score.value for c in selected) / len(selected)
    avg_bio = sum(biodiversity[c.company_id].composite_biodiversity_score.value for c in selected) / len(selected)

    portfolio = FinalPortfolio(
        portfolio_name="Integration Test Portfolio",
        mandate_summary="Biodiversity-aware long-only equity (test)",
        benchmark="STOXX Europe 600 (equal-weighted)",
        construction_date=datetime.now(timezone.utc),
        holdings=holdings,
        excluded_companies=[c.company_id for c in universe.companies if c not in selected],
        portfolio_carbon_intensity=round(avg_carbon, 2),
        benchmark_carbon_intensity=avg_carbon * 1.5,  # placeholder benchmark
        portfolio_esg_score=round(avg_esg, 2),
        portfolio_biodiversity_score=round(avg_bio, 2),
        optimisation_method="equal_weight",
        constraints_applied=["top 5 by composite score", "equal weighted"],
    )

    log_decision(
        agent="integration_test",
        decision_type="pipeline_complete",
        details={
            "holdings_count": len(portfolio.holdings),
            "portfolio_carbon": portfolio.portfolio_carbon_intensity,
            "portfolio_esg": portfolio.portfolio_esg_score,
            "portfolio_biodiversity": portfolio.portfolio_biodiversity_score,
        },
    )

    # === Assertions ===
    assert isinstance(portfolio, FinalPortfolio)
    assert len(portfolio.holdings) == 5
    assert abs(sum(h.weight for h in portfolio.holdings) - 1.0) < 0.001  # weights sum to 1
    assert portfolio.portfolio_carbon_intensity > 0
    assert 0 <= portfolio.portfolio_esg_score <= 10
    assert 0 <= portfolio.portfolio_biodiversity_score <= 10
    assert len(portfolio.excluded_companies) == 5

    # Decision log should have accumulated entries
    log_entries = read_log()
    assert len(log_entries) >= 7  # phase markers + 5 holdings + pipeline_complete


def test_pipeline_handles_human_override():
    """Verify that human overrides flow through the schema correctly."""
    from schemas.portfolio import HumanOverride

    override = HumanOverride(
        decision_point="exclusion_of_company_C00003",
        ai_recommendation="include",
        human_decision="exclude",
        rationale="Documented historical biodiversity controversy not yet in our news feed",
        decided_by="Investment Committee (test)",
        timestamp=datetime.now(timezone.utc),
    )

    assert override.decision_point == "exclusion_of_company_C00003"
    log_decision(
        agent="human_review",
        decision_type="override_logged",
        company_id="C00003",
        confidence="judgement_based",
        details={
            "ai_said": override.ai_recommendation,
            "human_said": override.human_decision,
        },
        notes=override.rationale,
    )

    assert len(read_log()) == 1
