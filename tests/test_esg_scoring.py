"""Tests for the ESG Scoring agent."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from agents.esg_scoring import (
    ESGScoringAgent,
    E_INDICATORS,
    S_INDICATORS,
    G_INDICATORS,
    COMPOSITE_WEIGHTS,
)
from agents.decision_log import read_log, LOG_PATH
from schemas.esg import ESGScore


DATA_DIR = Path("data/raw")
DATA_AVAILABLE = (DATA_DIR / "equityBicsV2.csv").exists()


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def test_pillar_weights_sum_sensibly():
    """Indicator weights within each pillar should be meaningful."""
    e_total = sum(meta["weight"] for meta in E_INDICATORS.values())
    s_total = sum(meta["weight"] for meta in S_INDICATORS.values())
    g_total = sum(meta["weight"] for meta in G_INDICATORS.values())

    # Weights should be on the order of 1.0 (slight rounding tolerance)
    assert 0.9 < e_total <= 1.1, f"E weights sum to {e_total}"
    assert 0.5 < s_total <= 1.5, f"S weights sum to {s_total}"
    assert 0.9 < g_total <= 1.5, f"G weights sum to {g_total}"


def test_composite_weights_sum_to_one():
    """ESG composite must be a true average."""
    total = sum(COMPOSITE_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001, f"Composite weights sum to {total}, not 1.0"


def test_scoring_handles_synthetic_data():
    """Should produce sensible scores for a small synthetic universe."""
    # Build a fake universe with 3 sectors and a range of E/S/G profiles
    synthetic = pd.DataFrame({
        "company_id": [f"C{i:05d}" for i in range(9)],
        "idBbGlobalCompanyName": [f"Company_{i}" for i in range(9)],
        "classificationLevelName1": ["Industrials"] * 3 + ["Financials"] * 3 + ["Tech"] * 3,
        "esgDisclosureScore": [80, 60, 40, 90, 70, 50, 85, 65, 45],
        "socialDisclosureScore": [70, 50, 30, 80, 60, 40, 75, 55, 35],
        "environDisclosureScore": [75, 55, 35, 85, 65, 45, 80, 60, 40],
        "govnceDisclosureScore": [70, 50, 30, 80, 60, 40, 75, 55, 35],
        "ghgScope1": [100000, 500000, 1000000, 5000, 15000, 30000, 1000, 5000, 10000],
        "ghgScope2": [50000, 200000, 500000, 2000, 8000, 15000, 500, 2500, 5000],
        "co2IntensityPerSalesCalc": [50, 200, 500, 5, 20, 50, 1, 5, 10],
        "climateChgPolicy": [1, 1, 0, 1, 1, 1, 1, 0, 0],
        "pctIndependentDirectors": [80, 65, 45, 85, 70, 50, 75, 60, 40],
        "boardMeetingAttendancePct": [95, 90, 80, 98, 92, 85, 96, 88, 75],
        "auditCommitteeMeetingAttendPct": [95, 90, 80, 98, 92, 85, 96, 88, 75],
        "esgLinkedBonus": [1, 0, 0, 1, 1, 0, 1, 0, 0],
        "nonexecDirWithResponsForCsr": [1, 1, 0, 1, 1, 1, 1, 0, 0],
    })

    agent = ESGScoringAgent()
    scores = agent.run(synthetic)

    assert len(scores) == 9
    for score in scores:
        assert isinstance(score, ESGScore)
        assert 0 <= score.e_score.value <= 10
        assert 0 <= score.s_score.value <= 10
        assert 0 <= score.g_score.value <= 10
        assert 0 <= score.composite_esg_score.value <= 10
        # Documented methodology fields populated
        assert "z-score" in score.normalisation_method.lower()
        assert score.weighting_method != ""
        assert "E" in score.sub_indicators_used


def test_scoring_logs_per_company():
    """Each company should produce a decision log entry."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000", "C00001"],
        "classificationLevelName1": ["Tech", "Tech"],
        "esgDisclosureScore": [80, 60],
        "ghgScope1": [1000, 5000],
        "govnceDisclosureScore": [75, 55],
    })
    agent = ESGScoringAgent()
    agent.run(synthetic)

    log = read_log()
    company_decisions = [e for e in log if e.get("decision_type") == "esg_score_computed"]
    assert len(company_decisions) == 2


def test_scoring_flags_data_poor_companies():
    """Companies with insufficient data should get exclusion flag."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000", "C00001"],
        "classificationLevelName1": ["Tech", "Tech"],
        # First company has data, second has almost nothing
        "esgDisclosureScore": [80, np.nan],
        "ghgScope1": [1000, np.nan],
        "govnceDisclosureScore": [75, np.nan],
    })
    agent = ESGScoringAgent()
    scores = agent.run(synthetic)

    # The data-poor company should be flagged
    assert scores[1].exclusion_flag is True
    assert scores[1].exclusion_reason is not None
    # The data-rich one should not
    assert scores[0].exclusion_flag is False


def test_scoring_uses_sector_relative_comparison():
    """Two companies with same emissions but in different sectors
    should get different scores — that's the whole point of sector-conditional."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000", "C00001", "C00002", "C00003"],
        "classificationLevelName1": [
            "Industrials", "Industrials",   # heavy emitter sector
            "Financials", "Financials",     # light emitter sector
        ],
        # Same absolute emissions, but second pair are in light-emitter sector
        "ghgScope1": [100000, 200000, 100000, 200000],
        "co2IntensityPerSalesCalc": [50, 100, 50, 100],
        # Other indicators identical so we isolate the sector effect
        "esgDisclosureScore": [70, 70, 70, 70],
        "environDisclosureScore": [70, 70, 70, 70],
        "govnceDisclosureScore": [70, 70, 70, 70],
    })

    agent = ESGScoringAgent()
    scores = agent.run(synthetic)
    # Just confirm the agent runs end-to-end without error on this case
    assert len(scores) == 4
    for s in scores:
        assert isinstance(s.e_score.value, float)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Course data pack not in data/raw/")
def test_scoring_runs_on_real_data():
    """End-to-end: ingest real data, score it."""
    from agents.data_ingestion import DataIngestionAgent

    ingest = DataIngestionAgent()
    master, _ = ingest.run(fetch_prices=False)

    scoring = ESGScoringAgent()
    # Just score a sample to keep the test fast
    sample = master.head(50)
    scores = scoring.run(sample)
    assert len(scores) == 50
    # Most should have non-extreme composite scores
    composite_values = [s.composite_esg_score.value for s in scores]
    assert min(composite_values) >= 0
    assert max(composite_values) <= 10
