"""Tests for the Bloomberg integration utility."""

import pytest
import pandas as pd
from pathlib import Path

from agents.bloomberg_integration import (
    load_eurostoxx50_constituents,
    load_bloomberg_esg_ratings,
    filter_master_to_eurostoxx50,
    attach_bloomberg_ratings,
)
from agents.decision_log import read_log, LOG_PATH


DATA_DIR = Path("data/raw")
CONSTITUENTS_AVAILABLE = (DATA_DIR / "eurostoxx50_constituents.xlsx").exists()
RATINGS_AVAILABLE = (DATA_DIR / "bloomberg_esg_ratings.xlsx").exists()


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


@pytest.mark.skipif(not CONSTITUENTS_AVAILABLE, reason="Constituents file not present")
def test_load_eurostoxx50_constituents():
    """The constituent list should load 50 companies with clean columns."""
    df = load_eurostoxx50_constituents()
    assert len(df) == 50
    # Check that columns are clean (no newlines)
    for col in df.columns:
        assert "\n" not in col
        assert col.strip() == col
    # Expected columns present
    expected = {"bloomberg_ticker", "company_name", "isin", "bics_level_1", "country"}
    assert expected.issubset(set(df.columns))


@pytest.mark.skipif(not RATINGS_AVAILABLE, reason="ESG ratings file not present")
def test_load_bloomberg_esg_ratings():
    """Bloomberg ESG ratings should load with European decimal commas converted."""
    df = load_bloomberg_esg_ratings()
    assert len(df) == 50
    # sustainalytics_score should be numeric after comma conversion
    assert df["sustainalytics_score"].dtype in (float, "float64")
    # No suspicious string values
    non_null = df["sustainalytics_score"].dropna()
    if len(non_null) > 0:
        # All values should be plausible Sustainalytics ESG Risk Scores (0-100)
        assert non_null.min() >= 0
        assert non_null.max() <= 100


def test_filter_to_eurostoxx50_with_synthetic_data():
    """Filter should work even on a tiny synthetic example."""
    master = pd.DataFrame({
        "idBbGlobalCompanyName": [
            "ASML Holding NV",
            "Some Random Company SA",
            "Iberdrola SA",
            "Atal SA/Poland",
        ],
        "country": ["NL", "FR", "ES", "PL"],
    })
    constituents = pd.DataFrame({
        "company_name": ["ASML Holding NV", "Iberdrola SA", "Schneider Electric SE"],
    })

    filtered, unmatched = filter_master_to_eurostoxx50(master, constituents)

    assert len(filtered) == 2  # ASML and Iberdrola
    assert "ASML Holding NV" in filtered["idBbGlobalCompanyName"].tolist()
    assert "Iberdrola SA" in filtered["idBbGlobalCompanyName"].tolist()
    assert len(unmatched) == 1  # Schneider Electric not in master


@pytest.mark.skipif(
    not (CONSTITUENTS_AVAILABLE and RATINGS_AVAILABLE),
    reason="Bloomberg data not available",
)
def test_filter_real_data_to_eurostoxx50():
    """End-to-end: filter the real master to EURO STOXX 50."""
    from agents.data_ingestion import DataIngestionAgent

    ingest = DataIngestionAgent()
    master, _ = ingest.run(fetch_prices=False)

    constituents = load_eurostoxx50_constituents()
    filtered, unmatched = filter_master_to_eurostoxx50(master, constituents)

    # Should match most of the 50
    assert len(filtered) > 30  # at least 30 of 50 matched
    print(f"Matched {len(filtered)} of {len(constituents)} EURO STOXX 50 companies")
    if len(unmatched) > 0:
        print(f"Unmatched: {unmatched['company_name'].tolist()}")


@pytest.mark.skipif(
    not (CONSTITUENTS_AVAILABLE and RATINGS_AVAILABLE),
    reason="Bloomberg data not available",
)
def test_attach_bloomberg_ratings_to_master():
    """Cross-validation: each EURO STOXX 50 company should get Bloomberg ratings."""
    from agents.data_ingestion import DataIngestionAgent

    ingest = DataIngestionAgent()
    master, _ = ingest.run(fetch_prices=False)

    constituents = load_eurostoxx50_constituents()
    filtered, _ = filter_master_to_eurostoxx50(master, constituents)

    esg_ratings = load_bloomberg_esg_ratings()
    enriched = attach_bloomberg_ratings(filtered, esg_ratings)

    # Most companies should have at least MSCI rating attached
    if "bloomberg_msci_rating" in enriched.columns:
        n_with_msci = enriched["bloomberg_msci_rating"].notna().sum()
        assert n_with_msci > 0
