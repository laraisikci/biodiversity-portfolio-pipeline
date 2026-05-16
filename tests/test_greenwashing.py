"""Tests for Agent 8: Greenwashing Detection."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from agents.greenwashing import (
    GreenwashingAgent,
    CALIBRATION_CASES,
    PROBABILITY_HIGH_THRESHOLD,
    PROBABILITY_MEDIUM_THRESHOLD,
)
from agents.decision_log import read_log, LOG_PATH
from schemas.document_extraction import DocumentExtraction, ExtractedClaim
from schemas.greenwashing import GreenwashingFlag


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def _make_extraction(
    company_id: str = "C99999",
    company_name: str = "Test Co",
    tnfd_adopter: bool = False,
    net_zero_year=None,
    sbti_status: str = "none",
    biodiversity_target_year=None,
    forest_commodities=None,
    transition_claim_text: str = "",
    has_quantitative_supply_chain: bool = False,
) -> DocumentExtraction:
    """Build a synthetic DocumentExtraction for testing."""
    top_claims = []
    if transition_claim_text:
        top_claims.append(ExtractedClaim(
            text=transition_claim_text,
            category="climate",
            is_quantitative=True,
        ))

    supply_chain = []
    if has_quantitative_supply_chain:
        supply_chain.append(ExtractedClaim(
            text="Reduce X by 50% by 2030",
            category="supply_chain",
            target_year=2030,
            is_quantitative=True,
        ))

    return DocumentExtraction(
        company_id=company_id,
        company_name=company_name,
        document_path="/fake/path.pdf",
        tnfd_adopter=tnfd_adopter,
        no_deforestation_pledge=False,
        net_zero_year=net_zero_year,
        sbti_status=sbti_status,
        biodiversity_target_year=biodiversity_target_year,
        water_stress_disclosed=False,
        forest_risk_commodities_mentioned=forest_commodities or [],
        top_sustainability_claims=top_claims,
        supply_chain_claims=supply_chain,
        biodiversity_commitments=[],
        climate_targets=[],
        water_disclosures=[],
        document_summary="",
        extraction_confidence="high",
    )


# === Signal tests ===

class TestSignalNetZeroWithoutSbti:
    """Signal 1: net-zero claim without SBTi validation."""

    def test_fires_when_net_zero_claimed_without_validation(self):
        ext = _make_extraction(net_zero_year=2040, sbti_status="committed")
        agent = GreenwashingAgent()
        assert agent._signal_net_zero_without_sbti(ext) == 1

    def test_fires_when_sbti_unknown(self):
        ext = _make_extraction(net_zero_year=2050, sbti_status="unknown")
        agent = GreenwashingAgent()
        assert agent._signal_net_zero_without_sbti(ext) == 1

    def test_does_not_fire_when_sbti_validated(self):
        ext = _make_extraction(net_zero_year=2040, sbti_status="validated")
        agent = GreenwashingAgent()
        assert agent._signal_net_zero_without_sbti(ext) == 0

    def test_does_not_fire_when_no_net_zero_claim(self):
        ext = _make_extraction(net_zero_year=None, sbti_status="none")
        agent = GreenwashingAgent()
        assert agent._signal_net_zero_without_sbti(ext) == 0


class TestSignalNatureClaimWithoutDisclosure:
    """Signal 2: TNFD claim without specific biodiversity target."""

    def test_fires_when_tnfd_claimed_without_bio_target(self):
        ext = _make_extraction(tnfd_adopter=True, biodiversity_target_year=None)
        agent = GreenwashingAgent()
        assert agent._signal_nature_claim_without_disclosure(ext) == 1

    def test_does_not_fire_when_both_present(self):
        ext = _make_extraction(tnfd_adopter=True, biodiversity_target_year=2030)
        agent = GreenwashingAgent()
        assert agent._signal_nature_claim_without_disclosure(ext) == 0

    def test_does_not_fire_when_neither_present(self):
        ext = _make_extraction(tnfd_adopter=False, biodiversity_target_year=None)
        agent = GreenwashingAgent()
        assert agent._signal_nature_claim_without_disclosure(ext) == 0


class TestSignalRatingDivergence:
    """Signal 4: ESG rating divergence > 1 SD."""

    def test_fires_with_diverging_ratings(self):
        # MSCI says A (0.7), Sustainalytics says risky (low normalised)
        master_row = pd.Series({
            "MSCI_ESG_Rating": "AAA",
            "Sustainalytics_ESG_Risk_Score": 45.0,
            "SP_Global_ESG_Score": 30,
            "RepRisk_RRI": 80,
        })
        agent = GreenwashingAgent()
        assert agent._signal_rating_divergence(master_row) == 1

    def test_does_not_fire_with_aligned_ratings(self):
        # All raters agree it's a strong ESG performer
        master_row = pd.Series({
            "MSCI_ESG_Rating": "AAA",
            "Sustainalytics_ESG_Risk_Score": 8.0,
            "SP_Global_ESG_Score": 85,
            "RepRisk_RRI": 15,
        })
        agent = GreenwashingAgent()
        assert agent._signal_rating_divergence(master_row) == 0

    def test_does_not_fire_with_missing_data(self):
        master_row = pd.Series({"MSCI_ESG_Rating": None})
        agent = GreenwashingAgent()
        assert agent._signal_rating_divergence(master_row) == 0

    def test_does_not_fire_with_no_master_row(self):
        agent = GreenwashingAgent()
        assert agent._signal_rating_divergence(None) == 0


class TestSignalForestCommodityGap:
    """Signal 5: Forest-risk commodities without quantitative targets."""

    def test_fires_with_commodities_no_quant_target(self):
        ext = _make_extraction(
            forest_commodities=["palm oil", "soy"],
            has_quantitative_supply_chain=False,
        )
        agent = GreenwashingAgent()
        assert agent._signal_forest_commodity_gap(ext) == 1

    def test_does_not_fire_with_commodities_and_quant_target(self):
        ext = _make_extraction(
            forest_commodities=["palm oil"],
            has_quantitative_supply_chain=True,
        )
        agent = GreenwashingAgent()
        assert agent._signal_forest_commodity_gap(ext) == 0

    def test_does_not_fire_without_commodities(self):
        ext = _make_extraction(forest_commodities=[])
        agent = GreenwashingAgent()
        assert agent._signal_forest_commodity_gap(ext) == 0


class TestSignalTransitionCapexGap:
    """Signal 6: Transition leadership claim contradicted by high emissions."""

    def test_fires_with_transition_claim_high_emissions(self):
        ext = _make_extraction(transition_claim_text="We are a transition leader")
        master_row = pd.Series({"co2IntensityPerSalesCalc": 250.0})  # High
        agent = GreenwashingAgent()
        assert agent._signal_transition_capex_gap(ext, master_row, None) == 1

    def test_does_not_fire_with_transition_claim_low_emissions(self):
        ext = _make_extraction(transition_claim_text="We are leading the transition")
        master_row = pd.Series({"co2IntensityPerSalesCalc": 30.0})  # Below threshold
        agent = GreenwashingAgent()
        assert agent._signal_transition_capex_gap(ext, master_row, None) == 0

    def test_does_not_fire_without_transition_claim(self):
        ext = _make_extraction(transition_claim_text="")
        master_row = pd.Series({"co2IntensityPerSalesCalc": 500.0})
        agent = GreenwashingAgent()
        assert agent._signal_transition_capex_gap(ext, master_row, None) == 0


# === Calibration classifier tests ===

class TestCalibrationClassifier:
    """Test the LogReg calibration logic."""

    def test_classifier_trains_successfully(self):
        agent = GreenwashingAgent()
        classifier = agent._train_calibration_classifier()
        assert classifier is not None
        assert agent._feature_scaler is not None

    def test_zero_signals_low_probability(self):
        agent = GreenwashingAgent()
        agent._classifier = agent._train_calibration_classifier()
        proba = agent._calibrate_probability(0)
        assert proba < 0.35  # Should be low

    def test_high_signals_high_probability(self):
        agent = GreenwashingAgent()
        agent._classifier = agent._train_calibration_classifier()
        proba = agent._calibrate_probability(6)
        assert proba > 0.65  # Should be high

    def test_monotonic_calibration(self):
        """More signals should yield higher (or equal) probability."""
        agent = GreenwashingAgent()
        agent._classifier = agent._train_calibration_classifier()
        probas = [agent._calibrate_probability(n) for n in range(7)]
        # Each should be >= the previous one
        for i in range(1, len(probas)):
            assert probas[i] >= probas[i-1] - 1e-6  # tiny float slack


# === Risk flag mapping ===

class TestRiskFlagMapping:
    """Test probability -> low/medium/high mapping."""

    def test_low_flag(self):
        agent = GreenwashingAgent()
        assert agent._probability_to_flag(0.10) == "low"
        assert agent._probability_to_flag(0.30) == "low"

    def test_medium_flag(self):
        agent = GreenwashingAgent()
        assert agent._probability_to_flag(0.40) == "medium"
        assert agent._probability_to_flag(0.60) == "medium"

    def test_high_flag(self):
        agent = GreenwashingAgent()
        assert agent._probability_to_flag(0.70) == "high"
        assert agent._probability_to_flag(0.95) == "high"


# === End-to-end tests ===

def test_run_with_empty_extractions():
    """Empty extractions should return empty list, not error."""
    agent = GreenwashingAgent()
    master = pd.DataFrame()
    result = agent.run(master=master, extractions=[])
    assert result == []


def test_run_produces_flags():
    """Running on an extraction produces a GreenwashingFlag."""
    agent = GreenwashingAgent()
    master = pd.DataFrame()

    # Strong greenwashing case: lots of claims, no evidence
    extraction = _make_extraction(
        company_id="C99999",
        tnfd_adopter=True,
        net_zero_year=2050,
        sbti_status="unknown",
        biodiversity_target_year=None,
        forest_commodities=["palm oil", "soy", "timber"],
        transition_claim_text="We are leading the transition to a sustainable future",
    )

    flags = agent.run(master=master, extractions=[extraction])
    assert len(flags) == 1

    flag = flags[0]
    assert isinstance(flag, GreenwashingFlag)
    assert flag.company_id == "C99999"
    assert flag.signals_fired > 0
    assert flag.risk_flag in ["low", "medium", "high"]
    assert 0 <= flag.greenwashing_probability.value <= 1


def test_clean_company_low_risk():
    """A company with all evidence and no claim gaps should be flagged low."""
    agent = GreenwashingAgent()
    master = pd.DataFrame()

    # Iberdrola-like profile: TNFD + SBTi validated + bio target
    extraction = _make_extraction(
        company_id="C00770",
        tnfd_adopter=True,
        net_zero_year=2040,
        sbti_status="validated",
        biodiversity_target_year=2030,
        forest_commodities=[],
        transition_claim_text="",
    )

    flags = agent.run(master=master, extractions=[extraction])
    assert len(flags) == 1
    assert flags[0].risk_flag == "low"


def test_decision_log_records_outcomes():
    """The agent should log start/complete/per-company decisions."""
    agent = GreenwashingAgent()
    master = pd.DataFrame()
    extraction = _make_extraction()

    agent.run(master=master, extractions=[extraction])

    log = read_log()
    decision_types = {entry["decision_type"] for entry in log}
    assert "greenwashing_start" in decision_types
    assert "greenwashing_complete" in decision_types
    assert "greenwashing_company_scored" in decision_types
    assert "calibration_classifier_trained" in decision_types


def test_calibration_dataset_well_formed():
    """The calibration set should have valid signal counts and labels."""
    for n_signals, label, name in CALIBRATION_CASES:
        assert 0 <= n_signals <= 6, f"{name}: signals out of range"
        assert label in (0, 1), f"{name}: label not binary"


def test_thresholds_make_sense():
    """High threshold should be above medium threshold."""
    assert PROBABILITY_HIGH_THRESHOLD > PROBABILITY_MEDIUM_THRESHOLD
    assert 0 < PROBABILITY_MEDIUM_THRESHOLD < PROBABILITY_HIGH_THRESHOLD < 1
