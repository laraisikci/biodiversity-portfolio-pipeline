"""Tests for the Data Ingestion agent.

These tests require the course data pack in data/raw/.
They skip gracefully if the data isn't present, so the test suite
still passes on fresh clones without the data.
"""

import pytest
from pathlib import Path

from agents.data_ingestion import DataIngestionAgent, EUROPEAN_COUNTRIES, EXCHANGE_SUFFIX_MAP
from agents.decision_log import read_log, LOG_PATH
from schemas.company import CompanyUniverse, CompanyBase


DATA_DIR = Path("data/raw")
DATA_AVAILABLE = (
    (DATA_DIR / "equityBicsV2.csv").exists()
    and (DATA_DIR / "esgEnvironmentalSocialConsolidatedV4.csv").exists()
    and (DATA_DIR / "esgGovernanceConsolidatedV4.csv").exists()
    and (DATA_DIR / "legalEntityEuTaxonomy.csv").exists()
)


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def test_european_country_list_is_sensible():
    """European country list should cover the main markets."""
    for c in ("GB", "FR", "DE", "ES", "IT", "NL"):
        assert c in EUROPEAN_COUNTRIES


def test_yahoo_suffix_map_covers_major_exchanges():
    """Should have suffixes for the main European exchanges."""
    assert EXCHANGE_SUFFIX_MAP["GB"] == ".L"
    assert EXCHANGE_SUFFIX_MAP["FR"] == ".PA"
    assert EXCHANGE_SUFFIX_MAP["DE"] == ".DE"
    assert EXCHANGE_SUFFIX_MAP["ES"] == ".MC"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Course data pack not in data/raw/")
def test_load_csv_works():
    """Loading the four CSVs should produce non-empty DataFrames."""
    agent = DataIngestionAgent()
    equity = agent._load_csv("equityBicsV2.csv")
    esg = agent._load_csv("esgEnvironmentalSocialConsolidatedV4.csv")
    assert len(equity) > 0
    assert len(esg) > 0
    # Equity file: identifiers + sectors, NOT country
    assert "idBbGlobalCompanyName" in equity.columns
    assert "classificationLevelName1" in equity.columns
    # ESG file: has country information
    assert "idBbGlobalCompanyName" in esg.columns
    assert "cntryOfDomicile" in esg.columns


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Course data pack not in data/raw/")
def test_full_ingestion_produces_universe():
    """End-to-end ingestion should produce a valid CompanyUniverse."""
    agent = DataIngestionAgent()
    master, universe = agent.run(fetch_prices=False)

    assert isinstance(universe, CompanyUniverse)
    assert universe.universe_size > 0
    assert universe.universe_size == len(universe.companies)
    assert universe.geographic_filter.startswith("Europe")

    # Every company should have an ID and a name
    for c in universe.companies[:10]:
        assert c.company_id.startswith("C")
        assert c.name != ""
        assert c.country in EUROPEAN_COUNTRIES


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Course data pack not in data/raw/")
def test_ingestion_logs_decisions():
    """Ingestion should write multiple decision log entries."""
    agent = DataIngestionAgent()
    agent.run(fetch_prices=False)
    log = read_log()
    # Should have: start + 4 CSV loads + filter + join + universe build = 8+ entries
    assert len(log) >= 7
    decision_types = {entry["decision_type"] for entry in log}
    assert "ingestion_start" in decision_types
    assert "equity_deduplicated" in decision_types
    assert "esg_deduplicated_to_latest" in decision_types
    assert "esg_inner_join" in decision_types
    assert "european_filter_applied" in decision_types
    assert "overlays_joined" in decision_types
    assert "universe_built" in decision_types


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Course data pack not in data/raw/")
def test_yahoo_ticker_is_constructed():
    """Yahoo tickers should have exchange suffixes for European companies."""
    agent = DataIngestionAgent()
    _, universe = agent.run(fetch_prices=False)

    # Check at least one Spanish company has .MC suffix
    spanish = [c for c in universe.companies if c.country == "ES"]
    if spanish:
        assert any(c.yahoo_ticker and c.yahoo_ticker.endswith(".MC") for c in spanish[:20])


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Course data pack not in data/raw/")
def test_universe_has_no_duplicate_companies():
    """Each company should appear exactly once in the universe.

    Regression test for the cartesian explosion bug where multiple share
    classes and ESG reporting periods produced duplicate rows.
    """
    agent = DataIngestionAgent()
    _, universe = agent.run(fetch_prices=False)
    names = [c.name for c in universe.companies]
    assert len(names) == len(set(names)), (
        f"Found duplicate company names. Universe size: {len(names)}, "
        f"unique names: {len(set(names))}"
    )
    # Sanity check on realistic size for a European ESG-disclosing universe
    assert 100 < universe.universe_size < 10000, (
        f"Universe size {universe.universe_size} seems implausible. "
        "Expected several hundred to a few thousand."
    )
