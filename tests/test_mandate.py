"""Tests for the Mandate agent."""

import pytest
from pathlib import Path

from agents.mandate import MandateAgent, Mandate
from agents.decision_log import read_log, LOG_PATH


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def test_mandate_default_runs():
    """Default mandate (proposal version) should be valid."""
    agent = MandateAgent()
    mandate = agent.run()
    assert isinstance(mandate, Mandate)
    assert mandate.client_name == "Prince Albert II of Monaco Foundation"
    assert mandate.n_holdings_min == 15
    assert mandate.n_holdings_max == 25
    assert mandate.benchmark_name == "STOXX Europe 600"
    assert mandate.max_single_name_weight == 0.08
    assert mandate.max_sector_weight == 0.20
    assert len(mandate.sustainability_objectives) >= 3
    assert len(mandate.geographic_scope) >= 10


def test_mandate_constraints_are_sensible():
    """Mandate constraints should be internally consistent."""
    agent = MandateAgent()
    mandate = agent.run()
    # Max single name × max holdings should be feasible
    assert mandate.max_single_name_weight * mandate.n_holdings_max >= 1.0, (
        "Cannot reach 100% weights with these constraints"
    )
    # Min sectors should be achievable
    assert mandate.min_sector_count >= 5
    # Carbon cap should be tighter than benchmark
    assert mandate.carbon_intensity_cap_vs_benchmark < 1.0


def test_mandate_logs_decision():
    """Running the agent should write to the audit trail."""
    agent = MandateAgent()
    agent.run()
    log = read_log()
    assert len(log) == 1
    entry = log[0]
    assert entry["agent"] == "mandate"
    assert entry["decision_type"] == "mandate_defined"
    assert entry["confidence"] == "judgement_based"
    assert "client" in entry["details"]


def test_mandate_override_via_config():
    """Should be able to override defaults for sensitivity analysis."""
    agent = MandateAgent()
    mandate = agent.run({"max_single_name_weight": 0.10, "n_holdings_max": 30})
    assert mandate.max_single_name_weight == 0.10
    assert mandate.n_holdings_max == 30
    # Other defaults remain
    assert mandate.client_name == "Prince Albert II of Monaco Foundation"


def test_mandate_summary_is_readable():
    """Summary should produce a string with the key facts."""
    agent = MandateAgent()
    mandate = agent.run()
    summary = agent.summarise(mandate)
    assert "Prince Albert II" in summary
    assert "STOXX Europe 600" in summary
    assert "biodiversity" in summary.lower()
    assert "15-25" in summary
