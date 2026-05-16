"""Tests for the Document Intelligence agent.

These use mocked Gemini calls so they pass without network access.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.document_intelligence import (
    DocumentIntelligenceAgent,
    EXTRACTION_PROMPT,
    CACHE_DIR,
)
from agents.decision_log import read_log, LOG_PATH
from schemas.document_extraction import DocumentExtraction, ExtractedClaim


# Sample Gemini response (what we expect Gemini to return)
SAMPLE_RESPONSE_JSON = {
    "company_name": "Iberdrola SA",
    "document_year": 2024,
    "tnfd_adopter": True,
    "no_deforestation_pledge": False,
    "net_zero_year": 2040,
    "sbti_status": "validated",
    "biodiversity_target_year": 2030,
    "water_stress_disclosed": True,
    "forest_risk_commodities": [],
    "biodiversity_commitments": [
        {
            "text": "Achieve net positive impact on biodiversity by 2030",
            "category": "biodiversity",
            "target_year": 2030,
            "is_quantitative": False,
            "source_page": 87,
        },
    ],
    "climate_targets": [
        {
            "text": "Net-zero emissions by 2040 across all scopes",
            "category": "climate",
            "target_year": 2040,
            "is_quantitative": True,
            "source_page": 45,
        },
    ],
    "water_disclosures": [],
    "supply_chain_claims": [],
    "top_sustainability_claims": [
        {
            "text": "Net-zero by 2040, biodiversity net positive by 2030",
            "category": "climate",
            "target_year": 2040,
            "is_quantitative": True,
            "source_page": 5,
        },
    ],
    "document_summary": "Iberdrola's report focuses on renewable energy transition and biodiversity net-positive commitments.",
    "extraction_confidence": "high",
    "extraction_notes": None,
}


@pytest.fixture(autouse=True)
def clean_log():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


@pytest.fixture(autouse=True)
def clean_cache():
    """Wipe the cache directory before each test."""
    test_cache = CACHE_DIR / "C99999.json"
    if test_cache.exists():
        test_cache.unlink()
    yield
    if test_cache.exists():
        test_cache.unlink()


@pytest.fixture
def fake_api_key():
    """Provide a fake API key so the agent can instantiate."""
    original = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = "fake-key-for-testing"
    yield
    if original is None:
        del os.environ["GOOGLE_API_KEY"]
    else:
        os.environ["GOOGLE_API_KEY"] = original


def test_extraction_prompt_contains_required_fields():
    """The extraction prompt should reference all the schema fields we want."""
    required = [
        "company_name", "tnfd_adopter", "no_deforestation_pledge",
        "net_zero_year", "sbti_status", "water_stress_disclosed",
        "forest_risk_commodities", "biodiversity_commitments",
        "climate_targets", "top_sustainability_claims",
        "document_summary", "extraction_confidence",
    ]
    for field in required:
        assert field in EXTRACTION_PROMPT, f"Prompt missing required field: {field}"


def test_agent_requires_api_key(fake_api_key):
    """Agent should error if no API key available."""
    # Remove the fake key temporarily
    del os.environ["GOOGLE_API_KEY"]
    with pytest.raises(ValueError, match="API key"):
        DocumentIntelligenceAgent()
    # Restore for cleanup
    os.environ["GOOGLE_API_KEY"] = "fake-key-for-testing"


def test_json_parser_handles_clean_json(fake_api_key):
    """Direct JSON should parse straight through."""
    agent = DocumentIntelligenceAgent()
    raw = json.dumps(SAMPLE_RESPONSE_JSON)
    parsed = agent._parse_json_response(raw)
    assert parsed["company_name"] == "Iberdrola SA"
    assert parsed["net_zero_year"] == 2040


def test_json_parser_handles_markdown_fences(fake_api_key):
    """Gemini sometimes wraps JSON in ```json fences despite instructions."""
    agent = DocumentIntelligenceAgent()
    raw = f"```json\n{json.dumps(SAMPLE_RESPONSE_JSON)}\n```"
    parsed = agent._parse_json_response(raw)
    assert parsed["company_name"] == "Iberdrola SA"


def test_json_parser_handles_fences_with_trailing_whitespace(fake_api_key):
    """Real failure case from Sanofi extraction — fences with trailing whitespace."""
    agent = DocumentIntelligenceAgent()
    # Reproduce what Gemini actually returned for Sanofi
    raw = f"```json\n{json.dumps(SAMPLE_RESPONSE_JSON)}\n```\n"
    parsed = agent._parse_json_response(raw)
    assert parsed["company_name"] == "Iberdrola SA"


def test_json_parser_handles_complex_nested_json(fake_api_key):
    """JSON with nested objects and strings containing braces."""
    agent = DocumentIntelligenceAgent()
    # A real-world case: claim text might contain braces, quotes, etc.
    complex_json = {
        "company_name": "Test Co",
        "biodiversity_commitments": [
            {
                "text": "We commit to {biodiversity} targets by 2030.",
                "category": "biodiversity",
                "target_year": 2030,
                "is_quantitative": True,
                "source_page": 42,
            }
        ],
    }
    raw = f"```json\n{json.dumps(complex_json)}\n```"
    parsed = agent._parse_json_response(raw)
    assert parsed["company_name"] == "Test Co"
    assert len(parsed["biodiversity_commitments"]) == 1


def test_json_parser_handles_preamble_text(fake_api_key):
    """Gemini sometimes adds explanatory text before/after JSON."""
    agent = DocumentIntelligenceAgent()
    raw = f"Here is the extracted JSON:\n{json.dumps(SAMPLE_RESPONSE_JSON)}\n\nThanks!"
    parsed = agent._parse_json_response(raw)
    assert parsed["company_name"] == "Iberdrola SA"


def test_json_parser_raises_on_garbage(fake_api_key):
    """Non-JSON should raise a clear error."""
    agent = DocumentIntelligenceAgent()
    with pytest.raises(ValueError, match="Failed to parse"):
        agent._parse_json_response("this is not JSON at all")


def test_build_extraction_creates_valid_object(fake_api_key):
    """Building the DocumentExtraction from parsed JSON should produce a valid object."""
    agent = DocumentIntelligenceAgent()
    extraction = agent._build_extraction(
        company_id="C99999",
        company_name="Test Co",
        pdf_path=Path("/fake/path.pdf"),
        parsed=SAMPLE_RESPONSE_JSON,
    )
    assert isinstance(extraction, DocumentExtraction)
    assert extraction.company_id == "C99999"
    assert extraction.tnfd_adopter is True
    assert extraction.net_zero_year == 2040
    assert extraction.sbti_status == "validated"
    assert len(extraction.biodiversity_commitments) == 1
    assert extraction.biodiversity_commitments[0].target_year == 2030


def test_run_with_mocked_gemini_call(fake_api_key, tmp_path):
    """End-to-end test with Gemini call mocked."""
    # Create a fake PDF file (just needs to exist)
    fake_pdf = tmp_path / "fake_report.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content")

    # Patch the actual Gemini call
    agent = DocumentIntelligenceAgent(use_cache=False)
    with patch.object(
        agent, "_call_gemini", return_value=json.dumps(SAMPLE_RESPONSE_JSON)
    ):
        documents = [
            {"company_id": "C99999", "company_name": "Test Co", "path": str(fake_pdf)},
        ]
        extractions = agent.run(documents)

    assert len(extractions) == 1
    assert extractions[0].company_id == "C99999"
    assert extractions[0].tnfd_adopter is True


def test_caching_works(fake_api_key, tmp_path):
    """Second call with same company_id should return cached result without calling Gemini."""
    fake_pdf = tmp_path / "fake_report.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content")

    documents = [
        {"company_id": "C99999", "company_name": "Test Co", "path": str(fake_pdf)},
    ]

    # First call: should hit Gemini (mocked)
    agent = DocumentIntelligenceAgent(use_cache=True)
    with patch.object(
        agent, "_call_gemini", return_value=json.dumps(SAMPLE_RESPONSE_JSON)
    ) as mock1:
        agent.run(documents)
        assert mock1.call_count == 1

    # Second call: should hit cache, NOT call Gemini
    agent2 = DocumentIntelligenceAgent(use_cache=True)
    with patch.object(agent2, "_call_gemini") as mock2:
        extractions = agent2.run(documents)
        assert mock2.call_count == 0  # Gemini never called
        assert len(extractions) == 1


def test_run_logs_decisions(fake_api_key, tmp_path):
    """Agent should write to the decision log."""
    fake_pdf = tmp_path / "fake_report.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content")

    agent = DocumentIntelligenceAgent(use_cache=False)
    with patch.object(
        agent, "_call_gemini", return_value=json.dumps(SAMPLE_RESPONSE_JSON)
    ):
        agent.run([
            {"company_id": "C99999", "company_name": "Test Co", "path": str(fake_pdf)},
        ])

    log = read_log()
    decision_types = {entry["decision_type"] for entry in log}
    assert "document_intelligence_start" in decision_types
    assert "document_intelligence_complete" in decision_types


def test_failed_extraction_does_not_break_pipeline(fake_api_key, tmp_path):
    """If one document fails, others should still be processed."""
    fake_pdf_1 = tmp_path / "good.pdf"
    fake_pdf_2 = tmp_path / "bad.pdf"
    fake_pdf_1.write_bytes(b"%PDF-1.4 good")
    fake_pdf_2.write_bytes(b"%PDF-1.4 bad")

    documents = [
        {"company_id": "C99998", "company_name": "Good Co", "path": str(fake_pdf_1)},
        {"company_id": "C99997", "company_name": "Bad Co", "path": str(fake_pdf_2)},
    ]

    agent = DocumentIntelligenceAgent(use_cache=False)

    # Mock Gemini: first call succeeds, second fails
    side_effects = [
        json.dumps(SAMPLE_RESPONSE_JSON),
        Exception("Gemini API error"),
    ]
    with patch.object(agent, "_call_gemini", side_effect=side_effects):
        extractions = agent.run(documents)

    # We should still get one successful extraction
    assert len(extractions) == 1
    assert extractions[0].company_id == "C99998"
