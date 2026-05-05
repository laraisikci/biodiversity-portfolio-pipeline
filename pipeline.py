"""
Main pipeline orchestrator.

Runs the full pipeline end-to-end. In Phase 1 most agents are stubs;
in Phase 2 the stubs are replaced with real implementations.

Usage:
    python pipeline.py --config config.yaml

This is intentionally simple. We are NOT using LangGraph yet — that's
an upgrade decision after the prototype clinic on 15 May. For now,
modular Python with clear function calls and shared state is enough
and easier to debug.
"""

import argparse
from pathlib import Path
from datetime import datetime, timezone

from agents.decision_log import log_decision


def run_pipeline(config: dict) -> dict:
    """Run the full pipeline end-to-end.

    Args:
        config: Configuration dict with paths, mandate, hyperparameters

    Returns:
        Dict with final portfolio + audit trail location
    """
    log_decision(
        agent="orchestrator",
        decision_type="pipeline_start",
        details={"config_summary": list(config.keys())},
    )

    # Phase 1: Mandate + Data
    # mandate = MandateAgent().run(config["mandate"])
    # universe = DataIngestionAgent().run(config["data_paths"])
    # quality_report = DataQualityAgent().run(...)

    # Phase 2: Per-company scoring (parallel in spirit, sequential in code)
    # esg_scores = ESGScoringAgent().run(...)
    # biodiversity_scores = BiodiversityAgent().run(...)
    # climate_metrics = ClimateAgent().run(...)
    # financial_metrics = FinancialAnalysisAgent().run(...)

    # Phase 3: Document intelligence + greenwashing
    # evidence = DocumentIntelligenceAgent().run(...)
    # greenwashing_flags = GreenwashingAgent().run(...)

    # Phase 4: Portfolio construction
    # portfolio = PortfolioConstructionAgent().run(...)

    # Phase 5: Adversarial review
    # red_team_findings = RedTeamAgent().run(portfolio)

    # Phase 6: Reporting
    # report = ReportingAgent().run(portfolio, decision_log)

    log_decision(
        agent="orchestrator",
        decision_type="pipeline_end",
        details={"end_time": datetime.now(timezone.utc).isoformat()},
    )

    return {"status": "stub", "message": "Pipeline scaffolding ready"}


def main():
    parser = argparse.ArgumentParser(description="Sustainable portfolio pipeline")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Config file path"
    )
    args = parser.parse_args()

    config = {"mandate": {}, "data_paths": {}}  # TODO load from YAML

    result = run_pipeline(config)
    print(f"Pipeline result: {result}")


if __name__ == "__main__":
    main()
