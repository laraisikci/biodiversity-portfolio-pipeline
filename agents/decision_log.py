"""
Shared decision logger.

Every agent imports this and logs its decisions. The output is a single
structured JSON file at outputs/logs/decision_log.jsonl that becomes our
audit trail. This is what we show the lecturer when they ask 'how was
this portfolio built?'

Usage:
    from agents.decision_log import log_decision

    log_decision(
        agent="esg_scoring",
        company_id="C00042",
        decision_type="score_calculated",
        details={"e_score": 7.4, "s_score": 6.1, "g_score": 8.2},
        confidence="estimated",
    )

That's it. Don't over-engineer it. Just call this whenever you make a
decision worth defending in Q&A.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_PATH = Path(__file__).parent.parent / "outputs" / "logs" / "decision_log.jsonl"


def log_decision(
    agent: str,
    decision_type: str,
    details: dict,
    company_id: Optional[str] = None,
    confidence: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Append a single decision to the shared log.

    Args:
        agent: Which agent is logging, e.g. 'data_quality', 'greenwashing'
        decision_type: What kind of decision, e.g. 'imputation', 'flag_raised',
            'score_calculated', 'exclusion', 'human_override'
        details: Free-form dict with the decision specifics
        company_id: If the decision is about a specific company
        confidence: One of: reported / observed / estimated / ai_extracted / judgement_based
        notes: Any human-readable context
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "decision_type": decision_type,
        "company_id": company_id,
        "confidence": confidence,
        "details": details,
        "notes": notes,
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_log() -> list:
    """Read the full decision log. Used by the dashboard and report assembly."""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_log(
    agent: Optional[str] = None,
    company_id: Optional[str] = None,
    decision_type: Optional[str] = None,
) -> list:
    """Filter log entries — useful for the dashboard's per-holding audit view."""
    entries = read_log()
    if agent:
        entries = [e for e in entries if e["agent"] == agent]
    if company_id:
        entries = [e for e in entries if e["company_id"] == company_id]
    if decision_type:
        entries = [e for e in entries if e["decision_type"] == decision_type]
    return entries
