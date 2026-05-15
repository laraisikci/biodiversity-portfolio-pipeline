"""Tests for the Biodiversity agent."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from agents.biodiversity import (
    BiodiversityAgent,
    SECTOR_MATERIALITY_WEIGHTS,
    SECTOR_NATURE_DEPENDENCY,
)
from agents.decision_log import read_log, LOG_PATH
from schemas.biodiversity import BiodiversityRiskScore


DATA_DIR = Path("data/raw")
DATA_AVAILABLE = (DATA_DIR / "equityBicsV2.csv").exists()


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def test_sector_materiality_weights_are_valid():
    """Each sector's layer weights must sum to ~1.0."""
    for sector, weights in SECTOR_MATERIALITY_WEIGHTS.items():
        layer_weights = [weights[k] for k in ["L1", "L2", "L3", "L4"]]
        total = sum(layer_weights)
        assert abs(total - 1.0) < 0.001, f"Sector '{sector}' weights sum to {total}"


def test_sector_dependency_matrix_complete():
    """Every sector in materiality matrix should have a dependency entry."""
    for sector in SECTOR_MATERIALITY_WEIGHTS:
        assert sector in SECTOR_NATURE_DEPENDENCY, (
            f"Sector '{sector}' has materiality weights but no dependency scores"
        )


def test_sector_dependency_values_in_range():
    """Dependency and impact should be in 0-1."""
    for sector, data in SECTOR_NATURE_DEPENDENCY.items():
        assert 0 <= data["dependency"] <= 1, f"{sector} dependency out of range"
        assert 0 <= data["impact"] <= 1, f"{sector} impact out of range"


def test_biodiversity_agent_on_synthetic_data():
    """Agent should produce BiodiversityRiskScore for each company."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000", "C00001", "C00002"],
        "classificationLevelName1": ["Technology", "Materials", "Utilities"],
        "euTaxnmyEstmatdSubstantlContrbtnBiodiverstyPctRevenue": [0.0, 0.0, 15.0],
        "euTaxnmyEstmatdDnshBiodiverstyLevl1": [0.0, 1.5, 0.0],
        "euTaxnmyEstmatdDnshBiodiverstyLevl2": [0.0, 1.0, 0.0],
        "environDisclosureScore": [60.0, 70.0, 85.0],
        "esgDisclosureScore": [70.0, 75.0, 90.0],
    })

    agent = BiodiversityAgent()
    scores = agent.run(synthetic)

    assert len(scores) == 3
    for s in scores:
        assert isinstance(s, BiodiversityRiskScore)
        assert 0 <= s.composite_biodiversity_score.value <= 10
        assert 0 <= s.encore_dependency_score.value <= 1


def test_utilities_with_biodiversity_taxonomy_scores_high():
    """A utility company with positive Taxonomy biodiversity contribution
    should score higher than a Materials company with DNSH harm."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000", "C00001"],
        "classificationLevelName1": ["Utilities", "Materials"],
        # Utility: 20% biodiversity-positive revenue, no harm
        # Materials: 0% positive, significant DNSH harm
        "euTaxnmyEstmatdSubstantlContrbtnBiodiverstyPctRevenue": [20.0, 0.0],
        "euTaxnmyEstmatdDnshBiodiverstyLevl1": [0.0, 2.0],
        "euTaxnmyEstmatdDnshBiodiverstyLevl2": [0.0, 1.5],
        "environDisclosureScore": [80.0, 60.0],
    })

    agent = BiodiversityAgent()
    scores = agent.run(synthetic)

    utility_score = scores[0].composite_biodiversity_score.value
    materials_score = scores[1].composite_biodiversity_score.value

    assert utility_score > materials_score, (
        f"Utility ({utility_score}) should score higher than Materials ({materials_score})"
    )


def test_high_risk_sector_with_poor_disclosure_gets_excluded():
    """A high-dependency sector company with poor disclosure should be flagged."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000"],
        "classificationLevelName1": ["Materials"],  # 0.85 dependency
        "euTaxnmyEstmatdSubstantlContrbtnBiodiverstyPctRevenue": [None],
        "euTaxnmyEstmatdDnshBiodiverstyLevl1": [2.0],  # significant harm
        "euTaxnmyEstmatdDnshBiodiverstyLevl2": [2.0],
        "environDisclosureScore": [20.0],  # very poor disclosure
    })

    agent = BiodiversityAgent()
    scores = agent.run(synthetic)

    assert scores[0].biodiversity_exclusion_flag is True
    assert scores[0].exclusion_reason is not None


def test_sector_conditional_weighting_uses_different_weights():
    """Tech vs Materials should weight Layer 1 differently."""
    tech_weights = SECTOR_MATERIALITY_WEIGHTS["Technology"]
    materials_weights = SECTOR_MATERIALITY_WEIGHTS["Materials"]

    # Tech should weight Layer 4 (disclosure) more heavily than Materials
    assert tech_weights["L4"] > materials_weights["L4"]
    # Materials should weight Layer 2 (DNSH) more heavily than Tech
    assert materials_weights["L2"] > tech_weights["L2"]


def test_logs_decisions():
    """Agent should write to the audit trail."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000"],
        "classificationLevelName1": ["Technology"],
        "environDisclosureScore": [70.0],
    })
    agent = BiodiversityAgent()
    agent.run(synthetic)

    log = read_log()
    decision_types = {entry["decision_type"] for entry in log}
    assert "biodiversity_scoring_start" in decision_types
    assert "biodiversity_scoring_complete" in decision_types
    assert "biodiversity_score_computed" in decision_types


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Course data pack not in data/raw/")
def test_biodiversity_on_real_data():
    """End-to-end: real master data, real scoring."""
    from agents.data_ingestion import DataIngestionAgent

    ingest = DataIngestionAgent()
    master, _ = ingest.run(fetch_prices=False)

    bio = BiodiversityAgent()
    sample = master.head(100)
    scores = bio.run(sample)
    assert len(scores) == 100
    for s in scores:
        assert 0 <= s.composite_biodiversity_score.value <= 10
