"""
Tests for the data quality agent and decision logger.

Run with: pytest tests/

The point of these tests is not coverage — it's to give the team a
working example of how to test their agent. Each role should add 1-2
tests for their own agent.
"""

import pandas as pd
import pytest
from pathlib import Path

from agents.data_quality import DataQualityAgent
from agents.decision_log import log_decision, read_log, LOG_PATH


@pytest.fixture(autouse=True)
def clean_log():
    """Wipe the log before each test so tests are isolated."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def test_decision_log_writes_and_reads():
    """The decision logger must round-trip cleanly."""
    log_decision(
        agent="test_agent",
        decision_type="test_decision",
        details={"foo": "bar"},
        company_id="C00001",
    )
    entries = read_log()
    assert len(entries) == 1
    assert entries[0]["agent"] == "test_agent"
    assert entries[0]["details"]["foo"] == "bar"
    assert entries[0]["company_id"] == "C00001"


def test_data_quality_flags_high_missingness():
    """Columns with >50% nulls should be flagged."""
    df = pd.DataFrame(
        {
            "good_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "missing_col": [1, None, None, None, None, None, None, 8, 9, 10],
        }
    )
    agent = DataQualityAgent()
    report = agent.run(df, dataset_name="test_data")

    assert report["row_count"] == 10
    assert report["columns"]["missing_col"]["null_pct"] == 60.0
    assert report["columns"]["missing_col"].get("flag") == "high_missingness"
    assert report["columns"]["good_col"].get("flag") is None


def test_data_quality_logs_decisions():
    """Running the agent should produce log entries."""
    df = pd.DataFrame({"col_a": [1, 2, None, 4], "col_b": [1, 2, 3, 4]})
    agent = DataQualityAgent()
    agent.run(df, dataset_name="small_test")

    entries = read_log()
    assert len(entries) >= 1
    assert any(e["agent"] == "data_quality" for e in entries)
