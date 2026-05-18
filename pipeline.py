"""
Main pipeline orchestrator.

Runs the full 11-agent pipeline end-to-end on the EURO STOXX 50 candidate set.
Every agent logs its decisions to outputs/logs/decision_log.jsonl.

Usage:
    python pipeline.py

Output:
    outputs/portfolio_factsheet.md       (regenerated)
    outputs/logs/decision_log.jsonl      (regenerated, full audit trail)
    outputs/cache/document_extractions/  (reused from cache)
    outputs/pipeline_summary.json        (high-level run summary)

This orchestrator does NOT re-call Gemini — it uses the 10 cached
extractions. To re-extract, delete the cache files and the document
intelligence agent will re-call Gemini.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

# === Agent imports ===

from agents.decision_log import log_decision, LOG_PATH
from agents.mandate import MandateAgent
from agents.data_ingestion import DataIngestionAgent
from agents.data_quality import DataQualityAgent
from agents.bloomberg_integration import (
    load_eurostoxx50_constituents,
    load_bloomberg_esg_ratings,
    filter_master_to_eurostoxx50,
    attach_bloomberg_ratings,
)
from agents.esg_scoring import ESGScoringAgent
from agents.climate import ClimateAgent
from agents.biodiversity import BiodiversityAgent
from agents.greenwashing import GreenwashingAgent
from agents.portfolio_construction import PortfolioConstructionAgent
from agents.human_review import HumanReviewAgent
from agents import financial_analysis as fa

from schemas.document_extraction import DocumentExtraction
from schemas.human_review import OverrideDecision


# === Paths ===

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
CACHE = OUTPUTS / "cache" / "document_extractions"
LOGS = OUTPUTS / "logs"
FACTSHEET = OUTPUTS / "portfolio_factsheet.md"
SUMMARY = OUTPUTS / "pipeline_summary.json"


# === Helpers ===

def _phase(name: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {name}\n{bar}")


def _step(text: str) -> None:
    print(f"  → {text}")


# === The orchestrator ===

def run_pipeline(reset_log: bool = True) -> dict:
    """Run the full 11-agent pipeline end-to-end.

    Args:
        reset_log: If True, clear decision_log.jsonl before running.

    Returns:
        Dict with summary stats.
    """

    if reset_log and LOG_PATH.exists():
        LOG_PATH.unlink()
        print(f"Cleared decision log: {LOG_PATH}")

    log_decision(
        agent="orchestrator",
        decision_type="pipeline_start",
        details={
            "start_time": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        },
        notes="Full 11-agent pipeline run starting.",
    )

    # === Agent 1 — Mandate ===
    _phase("AGENT 1 — MANDATE")
    _step("Loading mandate configuration")
    mandate_agent = MandateAgent()
    mandate = mandate_agent.run()
    _step(f"Mandate loaded: universe={mandate.universe_description}, benchmark={mandate.benchmark_name}")

    # === Agent 2 — Data Ingestion ===
    _phase("AGENT 2 — DATA INGESTION")
    _step("Loading European universe from CSVs")
    ingest = DataIngestionAgent()
    master, _ = ingest.run(fetch_prices=False)
    _step(f"Universe loaded: {len(master)} European companies")

    # === Bloomberg Integration ===
    _phase("BLOOMBERG INTEGRATION UTILITY")
    _step("Loading EURO STOXX 50 constituents")
    constituents = load_eurostoxx50_constituents()
    _step(f"Constituents: {len(constituents)} companies")

    _step("Filtering master to EURO STOXX 50")
    filtered, _ = filter_master_to_eurostoxx50(master, constituents)
    _step(f"Filtered universe: {len(filtered)} companies")

    _step("Loading Bloomberg ESG ratings")
    ratings = load_bloomberg_esg_ratings()
    _step(f"Loaded ratings for {len(ratings)} companies")

    _step("Attaching Bloomberg ratings to master")
    enriched = attach_bloomberg_ratings(filtered, ratings)

    log_decision(
        agent="bloomberg_integration",
        decision_type="enrichment_complete",
        details={
            "n_universe": len(master),
            "n_eurostoxx50_constituents": len(constituents),
            "n_filtered": len(filtered),
            "n_with_ratings": len(enriched),
        },
        confidence="reported",
        notes=(
            f"Filtered {len(master)} European companies to "
            f"{len(filtered)} EURO STOXX 50 names; attached Bloomberg "
            f"ratings from 4 providers (MSCI, Sustainalytics, S&P Global, RepRisk)."
        ),
    )

    # === Agent 3 — Data Quality ===
    _phase("AGENT 3 — DATA QUALITY")
    _step("Profiling data quality across 6 dimensions")
    quality_agent = DataQualityAgent()
    quality_report = quality_agent.run(enriched, dataset_name="eurostoxx50_enriched")

    completeness_overall = None
    if isinstance(quality_report, dict):
        completeness_overall = (
            quality_report.get("completeness", {}).get("completeness_pct_overall")
        )

    log_decision(
        agent="data_quality",
        decision_type="quality_profile_complete",
        details={
            "n_rows": len(enriched),
            "completeness_pct_overall": completeness_overall,
        },
        confidence="reported",
        notes=(
            f"Profiled {len(enriched)} companies across completeness, validity, "
            f"accuracy, consistency, uniqueness, and timeliness."
        ),
    )
    _step("Quality profile complete")

    # === Agent 4 — Document Intelligence (cached) ===
    _phase("AGENT 4 — DOCUMENT INTELLIGENCE (CACHED)")
    _step("Loading cached Gemini extractions")
    extractions: List[DocumentExtraction] = []
    for json_file in sorted(CACHE.glob("*.json")):
        if json_file.name.startswith("C") and len(json_file.stem) <= 6:
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            extractions.append(DocumentExtraction(**data))
        except Exception as e:
            print(f"  [WARN] Could not load {json_file.name}: {e}")
    _step(f"Loaded {len(extractions)} document extractions")

    log_decision(
        agent="document_intelligence",
        decision_type="cached_extractions_loaded",
        details={
            "n_extractions": len(extractions),
            "companies": [e.company_id for e in extractions],
            "extraction_method": "cached from previous Gemini 2.5 Flash run",
        },
        confidence="reported",
        notes=(
            f"Loaded {len(extractions)} cached document extractions with "
            "TNFD/SBTi status, biodiversity commitments, climate targets, "
            "supply chain claims; each with source page references."
        ),
    )

    # === Agent 5 — ESG Scoring ===
    _phase("AGENT 5 — ESG SCORING")
    _step("Computing sector-conditional ESG composite z-scores")
    esg_agent = ESGScoringAgent()
    esg_scores = esg_agent.run(enriched)
    _step(f"ESG scores computed for {len(esg_scores)} companies")

    # === Agent 6 — Climate ===
    _phase("AGENT 6 — CLIMATE")
    _step("Computing carbon intensity and WACI vs EBA reference")
    climate_agent = ClimateAgent()
    climate_scores = climate_agent.run(enriched)
    climate_dict = {c.company_id: c for c in climate_scores}
    _step(f"Climate metrics computed for {len(climate_scores)} companies")

    # === Agent 7 — Biodiversity ===
    _phase("AGENT 7 — BIODIVERSITY")
    _step("Computing multi-layer biodiversity risk scores")
    bio_agent = BiodiversityAgent()
    bio_scores = bio_agent.run(enriched)
    _step(f"Biodiversity scores computed for {len(bio_scores)} companies")

    # === Agent 8 — Greenwashing ===
    _phase("AGENT 8 — GREENWASHING")
    _step("Running 7-signal greenwashing detection")
    gw_agent = GreenwashingAgent()
    greenwashing_flags = gw_agent.run(
        master=enriched,
        extractions=extractions,
        climate_metrics=climate_dict,
    )
    _step(f"Greenwashing flags computed: {len(greenwashing_flags)} companies")

    flag_dist = {"low": 0, "medium": 0, "high": 0}
    for f in greenwashing_flags:
        flag_dist[f.risk_flag] = flag_dist.get(f.risk_flag, 0) + 1
    _step(
        f"Distribution: LOW={flag_dist['low']}, "
        f"MEDIUM={flag_dist['medium']}, HIGH={flag_dist['high']}"
    )

    # === Agent 9 — Portfolio Construction ===
    _phase("AGENT 9 — PORTFOLIO CONSTRUCTION")
    _step("Applying mandate constraints and selecting holdings")
    construction_agent = PortfolioConstructionAgent()
    portfolio = construction_agent.run(
        master=enriched,
        mandate=mandate,
        esg_scores=esg_scores,
        biodiversity_scores=bio_scores,
        climate_metrics=climate_scores,
    )
    _step(
        f"Portfolio constructed: {len(portfolio.holdings)} holdings, "
        f"{len(portfolio.excluded_companies)} excluded"
    )

    # === Agent 10 — Human Review ===
    _phase("AGENT 10 — HUMAN REVIEW")
    _step("Applying analyst overrides")

    overrides: List[OverrideDecision] = [
        OverrideDecision(
            company_id="SAN_SANOFI",
            company_name="Sanofi",
            action="add_to_watchlist",
            reviewer_id="LI",
            reviewer_role="Sustainability Analyst",
            justification=(
                "Sanofi is currently resubmitting SBTi validation due to "
                "corporate restructuring (scope change). Re-evaluate in 6 months."
            ),
            overrides_agent="greenwashing",
            evidence_sources=["Sanofi 2025 Sustainability Statement"],
        ),
        OverrideDecision(
            company_id="ALV_ALLIANZ",
            company_name="Allianz",
            action="add_to_watchlist",
            reviewer_id="LI",
            reviewer_role="Sustainability Analyst",
            justification=(
                "Financial sector indirect biodiversity exposure: monitor loan "
                "book disclosures and PCAF alignment over next two quarters."
            ),
            overrides_agent="biodiversity",
        ),
    ]

    review_agent = HumanReviewAgent()
    adjusted_portfolio, review_summary = review_agent.run(
        portfolio=portfolio,
        overrides=overrides,
        greenwashing_flags=greenwashing_flags,
    )
    _step(
        f"Applied {review_summary.n_overrides_total} overrides "
        f"by {len(review_summary.reviewers)} reviewer(s)"
    )

    # === Agent 11 — Financial Analysis ===
    _phase("AGENT 11 — FINANCIAL ANALYSIS")
    _step("Mapping holdings to Yahoo Finance tickers")

    # Build Bloomberg tickers list from final holdings (where available in master)
    holding_ids = [h.company_id for h in adjusted_portfolio.holdings]
    # Try to find ticker columns in the enriched master
    ticker_col = None
    for col in ["ticker", "idBb", "id_bb_ticker", "bbgTicker"]:
        if col in enriched.columns:
            ticker_col = col
            break

    bloomberg_tickers = []
    if ticker_col:
        ticker_lookup = enriched.set_index(enriched.index)
        # Try to lookup tickers by company_id or name
        for h in adjusted_portfolio.holdings:
            ticker = None
            # match by company_id or company_name in master
            match = enriched[enriched["idBbGlobalCompanyName"] == h.company_name] \
                if "idBbGlobalCompanyName" in enriched.columns else pd.DataFrame()
            if len(match) > 0:
                ticker = match.iloc[0].get(ticker_col)
            if ticker:
                bloomberg_tickers.append(str(ticker))
    
    _step(f"Found {len(bloomberg_tickers)} Bloomberg tickers (of {len(holding_ids)} holdings)")

    # Map Bloomberg → Yahoo tickers
    yahoo_tickers = []
    for bbg in bloomberg_tickers:
        yahoo = fa.bloomberg_to_yahoo_ticker(bbg)
        if yahoo:
            yahoo_tickers.append(yahoo)
    _step(f"Mapped {len(yahoo_tickers)} to Yahoo Finance tickers")

    financial_metrics_dict = None
    if yahoo_tickers:
        try:
            _step("Fetching 3-year prices via yfinance")
            prices = fa.fetch_price_history(yahoo_tickers=yahoo_tickers, years=fa.BACKTEST_YEARS)
            if not prices.empty:
                daily_returns = fa.compute_daily_returns(prices)
                weights = pd.Series(
                    {t: 1.0 / len(daily_returns.columns) for t in daily_returns.columns}
                )
                portfolio_returns = fa.compute_portfolio_returns(daily_returns, weights)

                bench_prices = fa.fetch_price_history(
                    yahoo_tickers=[fa.EURO_STOXX_50_TICKER], years=fa.BACKTEST_YEARS
                )
                bench_returns = None
                if not bench_prices.empty:
                    bench_daily = fa.compute_daily_returns(bench_prices)
                    bench_returns = bench_daily.iloc[:, 0]

                metrics = fa.compute_full_metrics(
                    daily_returns=portfolio_returns,
                    benchmark_returns=bench_returns,
                )
                financial_metrics_dict = metrics
                _step(
                    f"Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                    f"AnnReturn={metrics.get('annualised_return', 0):.2%}"
                )

                log_decision(
                    agent="financial_analysis",
                    decision_type="backtest_complete",
                    details={
                        "n_tickers_mapped": len(daily_returns.columns),
                        "n_tickers_total": len(holding_ids),
                        **{k: float(v) if hasattr(v, '__float__') else str(v)
                           for k, v in metrics.items() if v is not None},
                    },
                    confidence="reported",
                    notes=(
                        f"3-year backtest via yfinance: mapped "
                        f"{len(daily_returns.columns)}/{len(holding_ids)} tickers."
                    ),
                )
            else:
                _step("No prices fetched")
        except Exception as e:
            _step(f"Financial analysis encountered an issue: {e}")
            log_decision(
                agent="financial_analysis",
                decision_type="backtest_failed",
                details={"error": str(e)[:200]},
                confidence="observed",
                notes="Financial analysis could not complete.",
            )
    else:
        _step("No tickers available; financial analysis skipped")
        log_decision(
            agent="financial_analysis",
            decision_type="backtest_skipped",
            details={"reason": "no ticker mapping available from master"},
            confidence="observed",
            notes="Financial analysis skipped — no ticker column found in master.",
        )

    # === Persist factsheet ===
    _phase("WRITING FACTSHEET")
    _step(f"Writing factsheet to {FACTSHEET}")
    try:
        if hasattr(construction_agent, "write_factsheet"):
            construction_agent.write_factsheet(adjusted_portfolio, FACTSHEET)
        else:
            _step("(No write_factsheet method available; existing factsheet preserved)")
    except Exception as e:
        _step(f"Factsheet write encountered an issue: {e}")

    # === Pipeline summary ===
    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_universe_companies": len(master),
        "n_eurostoxx50_filtered": len(filtered),
        "n_extractions": len(extractions),
        "n_esg_scores": len(esg_scores),
        "n_climate_scores": len(climate_scores),
        "n_biodiversity_scores": len(bio_scores),
        "n_greenwashing_flags": len(greenwashing_flags),
        "greenwashing_distribution": flag_dist,
        "n_holdings": len(adjusted_portfolio.holdings),
        "n_excluded": len(adjusted_portfolio.excluded_companies),
        "n_overrides_applied": review_summary.n_overrides_total,
        "reviewers": review_summary.reviewers,
        "financial_metrics": {
            k: float(v) if hasattr(v, '__float__') else str(v)
            for k, v in (financial_metrics_dict or {}).items() if v is not None
        } if financial_metrics_dict else None,
    }

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log_decision(
        agent="orchestrator",
        decision_type="pipeline_end",
        details=summary,
        confidence="reported",
        notes=f"Pipeline completed: {len(adjusted_portfolio.holdings)} holdings.",
    )

    _phase("PIPELINE COMPLETE")
    print(f"  Factsheet:    {FACTSHEET}")
    print(f"  Decision log: {LOG_PATH}")
    print(f"  Summary:      {SUMMARY}")
    print(f"  Holdings:     {len(adjusted_portfolio.holdings)}")
    print(f"  Excluded:     {len(adjusted_portfolio.excluded_companies)}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Sustainable portfolio research pipeline (11 agents)"
    )
    parser.add_argument(
        "--keep-log",
        action="store_true",
        help="Do NOT clear the decision log before running (default: clear it)",
    )
    args = parser.parse_args()

    result = run_pipeline(reset_log=not args.keep_log)
    print(f"\nDone. Summary written to {SUMMARY}\n")


if __name__ == "__main__":
    main()
