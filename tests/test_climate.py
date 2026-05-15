"""Tests for the Climate agent."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from agents.climate import ClimateAgent
from agents.decision_log import read_log, LOG_PATH
from schemas.biodiversity import ClimateMetrics
from schemas.confidence import ConfidenceLevel


DATA_DIR = Path("data/raw")
DATA_AVAILABLE = (DATA_DIR / "equityBicsV2.csv").exists()


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def test_climate_agent_on_synthetic_data():
    """Climate agent should produce ClimateMetrics on a small synthetic universe."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000", "C00001", "C00002"],
        "classificationLevelName1": ["Industrials", "Industrials", "Tech"],
        "ghgScope1": [100_000, 200_000, 5_000],
        "ghgScope2": [50_000, 80_000, 2_000],
        "ghgScope3": [500_000, 800_000, 20_000],
        "co2IntensityPerSalesCalc": [120.0, 240.0, 12.0],
        "climateChgPolicy": ["Y", "Y", "Y"],
        "emissionReduction": ["Y", "Y", "Y"],
    })

    agent = ClimateAgent()
    metrics = agent.run(synthetic)

    assert len(metrics) == 3
    for m in metrics:
        assert isinstance(m, ClimateMetrics)
        assert m.scope_1_emissions is not None
        assert m.scope_1_emissions.value > 0
        assert m.carbon_intensity_per_revenue is not None
        assert m.carbon_intensity_per_revenue.value > 0


def test_climate_imputes_missing_intensity_to_sector_median():
    """Companies without disclosed intensity should get sector-median fallback."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000", "C00001", "C00002"],
        "classificationLevelName1": ["Industrials", "Industrials", "Industrials"],
        "ghgScope1": [100_000, 200_000, np.nan],
        "co2IntensityPerSalesCalc": [120.0, 240.0, np.nan],
    })

    agent = ClimateAgent()
    metrics = agent.run(synthetic)

    # The third company has no intensity disclosed
    assert metrics[2].carbon_intensity_per_revenue is not None
    # Imputed values should be flagged as estimated
    assert metrics[2].carbon_intensity_per_revenue.confidence == ConfidenceLevel.ESTIMATED.value


def test_climate_logs_decisions():
    """Climate scoring should write to the audit trail."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000"],
        "classificationLevelName1": ["Tech"],
        "ghgScope1": [1000],
        "co2IntensityPerSalesCalc": [5.0],
    })
    agent = ClimateAgent()
    agent.run(synthetic)

    log = read_log()
    decision_types = {entry["decision_type"] for entry in log}
    assert "climate_scoring_start" in decision_types
    assert "climate_scoring_complete" in decision_types


def test_portfolio_waci_calculation():
    """WACI should be the weighted average of holdings' intensities."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000", "C00001", "C00002"],
        "classificationLevelName1": ["Industrials", "Industrials", "Tech"],
        "co2IntensityPerSalesCalc": [100.0, 200.0, 50.0],
    })

    agent = ClimateAgent()
    metrics = agent.run(synthetic)

    # Equal-weight portfolio of 3 companies
    portfolio = {"C00000": 1/3, "C00001": 1/3, "C00002": 1/3}
    waci = agent.compute_portfolio_waci(portfolio, metrics)

    # Expected WACI: (100 + 200 + 50) / 3 = 116.67
    assert abs(waci - 116.67) < 0.01


def test_portfolio_waci_with_uneven_weights():
    """WACI should respect actual weights (heavier weights → larger contribution)."""
    synthetic = pd.DataFrame({
        "company_id": ["C00000", "C00001"],
        "classificationLevelName1": ["Industrials", "Tech"],
        "co2IntensityPerSalesCalc": [500.0, 10.0],
    })

    agent = ClimateAgent()
    metrics = agent.run(synthetic)

    # 90% weight in high-emitter, 10% in low-emitter
    portfolio = {"C00000": 0.9, "C00001": 0.1}
    waci = agent.compute_portfolio_waci(portfolio, metrics)

    # Expected WACI: 0.9 * 500 + 0.1 * 10 = 451
    assert abs(waci - 451.0) < 0.1


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Course data pack not in data/raw/")
def test_climate_on_real_data():
    """End-to-end: ingest real data, compute climate metrics."""
    from agents.data_ingestion import DataIngestionAgent

    ingest = DataIngestionAgent()
    master, _ = ingest.run(fetch_prices=False)

    climate = ClimateAgent()
    sample = master.head(100)
    metrics = climate.run(sample)

    assert len(metrics) == 100
    # At least some should have reported emissions
    n_with_scope_1 = sum(1 for m in metrics if m.scope_1_emissions is not None)
    assert n_with_scope_1 > 0
