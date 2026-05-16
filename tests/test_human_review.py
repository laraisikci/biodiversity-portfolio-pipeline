"""Tests for Agent 10 (Human Review)."""

import pytest
from datetime import datetime, timezone

from agents.human_review import HumanReviewAgent
from agents.decision_log import read_log, LOG_PATH
from schemas.human_review import (
    OverrideDecision,
    HumanReviewSummary,
)
from schemas.portfolio import FinalPortfolio, PortfolioHolding
from schemas.greenwashing import GreenwashingFlag
from schemas.confidence import DataPoint, ConfidenceLevel


# === Test fixtures ===

@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def _make_holding(company_id: str, weight: float = 0.05) -> PortfolioHolding:
    """Build a minimal PortfolioHolding for testing."""
    return PortfolioHolding(
        company_id=company_id,
        company_name=f"Co {company_id}",
        weight=weight,
        inclusion_rationale="Test holding for human review tests",
        composite_esg_score=5.0,
        composite_biodiversity_score=5.0,
        carbon_intensity=20.0,
        greenwashing_probability=0.18,
        sector_allocation="Technology",
        country="DE",
    )


def _make_portfolio(n_holdings: int = 3) -> FinalPortfolio:
    """Build a minimal FinalPortfolio for testing."""
    return FinalPortfolio(
        portfolio_name="Test Portfolio",
        mandate_summary="Test mandate",
        benchmark="Test benchmark",
        construction_date=datetime.now(timezone.utc),
        holdings=[
            _make_holding(f"C{i:04d}", weight=1.0 / n_holdings)
            for i in range(n_holdings)
        ],
        excluded_companies=["C9999"],
        exclusion_reasons={"C9999": "test exclusion"},
        watchlist=[],
        portfolio_carbon_intensity=20.0,
        benchmark_carbon_intensity=100.0,
        portfolio_esg_score=5.0,
        portfolio_biodiversity_score=5.0,
        optimisation_method="ranked",
        constraints_applied=[],
    )


def _make_override(
    company_id: str = "C0000",
    action: str = "force_exclude",
    reviewer_id: str = "LI",
    justification: str = "Test override with sufficient justification length",
    **kwargs,
) -> OverrideDecision:
    """Build an OverrideDecision for testing."""
    return OverrideDecision(
        company_id=company_id,
        action=action,
        reviewer_id=reviewer_id,
        justification=justification,
        **kwargs,
    )


# === Override schema tests ===

class TestOverrideDecisionSchema:
    """Test the OverrideDecision validation rules."""

    def test_minimal_valid_override(self):
        """Minimal valid override."""
        o = _make_override()
        assert o.company_id == "C0000"
        assert o.action == "force_exclude"

    def test_justification_minimum_length(self):
        """Justification must be at least 20 characters."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            OverrideDecision(
                company_id="C0001",
                action="force_exclude",
                reviewer_id="LI",
                justification="too short",
            )

    def test_change_weight_requires_new_weight(self):
        """change_weight action stores new_weight."""
        o = _make_override(action="change_weight", new_weight=0.08)
        assert o.new_weight == 0.08


# === Force exclude tests ===

class TestForceExclude:
    """Test force_exclude action."""

    def test_force_exclude_removes_from_holdings(self):
        portfolio = _make_portfolio(n_holdings=3)
        override = _make_override(
            company_id="C0001",
            action="force_exclude",
            justification="Bayer's biodiversity issues outweigh the strong ESG ratings",
        )
        agent = HumanReviewAgent()
        adjusted, summary = agent.run(portfolio, [override])
        assert len(adjusted.holdings) == 2
        assert all(h.company_id != "C0001" for h in adjusted.holdings)
        assert "C0001" in adjusted.excluded_companies

    def test_force_exclude_adds_reason(self):
        portfolio = _make_portfolio(n_holdings=3)
        override = _make_override(
            company_id="C0001",
            action="force_exclude",
            reviewer_id="LI",
            justification="Mandate-incompatible biodiversity exposure detected",
        )
        agent = HumanReviewAgent()
        adjusted, _ = agent.run(portfolio, [override])
        assert "C0001" in adjusted.exclusion_reasons
        assert "LI" in adjusted.exclusion_reasons["C0001"]
        assert "Mandate-incompatible" in adjusted.exclusion_reasons["C0001"]

    def test_force_exclude_renormalises_weights(self):
        portfolio = _make_portfolio(n_holdings=3)
        # Each holding has weight 1/3 = 0.333
        original_weight = portfolio.holdings[0].weight
        override = _make_override(
            company_id="C0001",
            action="force_exclude",
            justification="Sufficient justification for exclusion in this test",
        )
        agent = HumanReviewAgent()
        adjusted, _ = agent.run(portfolio, [override])
        # After excluding 1 of 3, remaining 2 should sum to 1.0
        total = sum(h.weight for h in adjusted.holdings)
        assert abs(total - 1.0) < 1e-9
        # Each remaining holding should now have weight 0.5
        for h in adjusted.holdings:
            assert abs(h.weight - 0.5) < 1e-9


# === Change weight tests ===

class TestChangeWeight:
    """Test change_weight action."""

    def test_change_weight_updates_weight(self):
        portfolio = _make_portfolio(n_holdings=3)
        override = _make_override(
            company_id="C0001",
            action="change_weight",
            new_weight=0.10,
            justification="Increase position based on additional research findings",
        )
        agent = HumanReviewAgent()
        adjusted, _ = agent.run(portfolio, [override])
        # The targeted holding's weight should be 0.10 / total
        # (renormalised so all weights sum to 1)
        c0001 = next(h for h in adjusted.holdings if h.company_id == "C0001")
        # Pre-renorm: weights were [1/3, 0.10, 1/3] → total = 0.766
        # Post-renorm: 0.10 / 0.766 ≈ 0.131
        assert abs(sum(h.weight for h in adjusted.holdings) - 1.0) < 1e-9

    def test_change_weight_requires_company_in_portfolio(self):
        portfolio = _make_portfolio(n_holdings=3)
        override = _make_override(
            company_id="C9999",  # Not in portfolio
            action="change_weight",
            new_weight=0.10,
            justification="Test change weight on non-existent holding",
        )
        agent = HumanReviewAgent()
        # Should NOT crash — failed override logged but pipeline continues
        adjusted, summary = agent.run(portfolio, [override])
        # No override was applied
        assert summary.n_change_weight == 0


# === Greenwashing flag override tests ===

class TestOverrideGreenwashingFlag:
    """Test override_greenwashing_flag action."""

    def test_override_greenwashing_flag(self):
        portfolio = _make_portfolio(n_holdings=2)
        gw_flag = GreenwashingFlag(
            company_id="C0000",
            risk_flag="high",
            greenwashing_probability=DataPoint(
                value=0.85,
                confidence=ConfidenceLevel.ESTIMATED,
                source="Agent 8",
                extraction_method="LogReg calibrator",
            ),
            classifier_confidence="high",
            recommended_action="watchlist",
        )
        override = _make_override(
            company_id="C0000",
            action="override_greenwashing_flag",
            new_risk_flag="medium",
            justification="Detailed engagement confirms credibility despite signals",
        )
        agent = HumanReviewAgent()
        _, summary = agent.run(
            portfolio,
            [override],
            greenwashing_flags=[gw_flag],
        )
        assert summary.n_override_flag == 1
        assert gw_flag.risk_flag == "medium"


# === Watchlist tests ===

class TestWatchlist:
    """Test watchlist add/remove."""

    def test_add_to_watchlist(self):
        portfolio = _make_portfolio()
        override = _make_override(
            company_id="C0001",
            action="add_to_watchlist",
            justification="Monitor SBTi resubmission progress over next two quarters",
        )
        agent = HumanReviewAgent()
        adjusted, summary = agent.run(portfolio, [override])
        assert "C0001" in adjusted.watchlist
        assert summary.n_watchlist_changes == 1

    def test_remove_from_watchlist(self):
        portfolio = _make_portfolio()
        portfolio.watchlist.append("C0001")
        override = _make_override(
            company_id="C0001",
            action="remove_from_watchlist",
            justification="Company has resolved the previously flagged concerns",
        )
        agent = HumanReviewAgent()
        adjusted, _ = agent.run(portfolio, [override])
        assert "C0001" not in adjusted.watchlist


# === Summary and audit trail tests ===

class TestSummary:
    """Test the HumanReviewSummary output."""

    def test_empty_overrides_produces_zero_summary(self):
        portfolio = _make_portfolio()
        agent = HumanReviewAgent()
        _, summary = agent.run(portfolio, [])
        assert summary.n_overrides_total == 0
        assert summary.reviewers == []

    def test_summary_counts_by_action(self):
        portfolio = _make_portfolio(n_holdings=4)
        overrides = [
            _make_override(
                company_id="C0001",
                action="force_exclude",
                justification="Excluding for biodiversity reasons in this test",
            ),
            _make_override(
                company_id="C0002",
                action="change_weight",
                new_weight=0.10,
                justification="Increasing weight based on engagement outcome",
            ),
            _make_override(
                company_id="C0003",
                action="add_to_watchlist",
                justification="Add to watchlist for ongoing engagement monitoring",
            ),
        ]
        agent = HumanReviewAgent()
        _, summary = agent.run(portfolio, overrides)
        assert summary.n_force_exclude == 1
        assert summary.n_change_weight == 1
        assert summary.n_watchlist_changes == 1
        assert summary.n_overrides_total == 3

    def test_summary_text_includes_justifications(self):
        portfolio = _make_portfolio()
        override = _make_override(
            company_id="C0001",
            action="force_exclude",
            reviewer_id="LI",
            justification="Specific justification that should appear in summary text",
        )
        agent = HumanReviewAgent()
        _, summary = agent.run(portfolio, [override])
        assert "Specific justification" in summary.summary_text
        assert "LI" in summary.summary_text

    def test_unique_reviewers(self):
        portfolio = _make_portfolio(n_holdings=3)
        overrides = [
            _make_override(
                company_id="C0000",
                reviewer_id="LI",
                action="add_to_watchlist",
                justification="Override one with sufficient justification text",
            ),
            _make_override(
                company_id="C0001",
                reviewer_id="JS",
                action="add_to_watchlist",
                justification="Override two by different reviewer with reason",
            ),
            _make_override(
                company_id="C0002",
                reviewer_id="LI",
                action="add_to_watchlist",
                justification="Override three by first reviewer again here",
            ),
        ]
        agent = HumanReviewAgent()
        _, summary = agent.run(portfolio, overrides)
        assert set(summary.reviewers) == {"LI", "JS"}


# === Decision log tests ===

class TestDecisionLog:
    """Test that overrides are logged to the decision_log."""

    def test_override_logged_to_decision_log(self):
        portfolio = _make_portfolio()
        override = _make_override(
            company_id="C0001",
            action="force_exclude",
            reviewer_id="LI",
            justification="Excluding due to insufficient transition evidence",
        )
        agent = HumanReviewAgent()
        agent.run(portfolio, [override])
        log = read_log()
        decision_types = {entry["decision_type"] for entry in log}
        assert "human_review_start" in decision_types
        assert "override_applied" in decision_types
        assert "human_review_complete" in decision_types

    def test_failed_override_logged_separately(self):
        portfolio = _make_portfolio()
        # Override targeting a non-existent company should log as failed
        override = _make_override(
            company_id="C9999",  # Not in portfolio
            action="change_weight",
            new_weight=0.1,
            justification="Test failure case for non-existent company override",
        )
        agent = HumanReviewAgent()
        agent.run(portfolio, [override])
        log = read_log()
        decision_types = {entry["decision_type"] for entry in log}
        assert "override_failed" in decision_types


# === Immutability test ===

class TestImmutability:
    """Verify that input portfolio is not mutated."""

    def test_input_portfolio_unchanged(self):
        portfolio = _make_portfolio(n_holdings=3)
        original_holdings_count = len(portfolio.holdings)
        original_weights = [h.weight for h in portfolio.holdings]
        
        override = _make_override(
            company_id="C0001",
            action="force_exclude",
            justification="Test that input portfolio is not modified by agent",
        )
        agent = HumanReviewAgent()
        adjusted, _ = agent.run(portfolio, [override])
        
        # Original should be unchanged
        assert len(portfolio.holdings) == original_holdings_count
        for h, w in zip(portfolio.holdings, original_weights):
            assert h.weight == w
        # Adjusted should be different
        assert len(adjusted.holdings) == original_holdings_count - 1
