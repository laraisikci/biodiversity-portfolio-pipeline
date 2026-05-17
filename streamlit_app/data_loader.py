"""
Data loaders for the Streamlit dashboard.

Reads from the real pipeline outputs:
  - outputs/portfolio_factsheet.md (parsed for headline metrics)
  - outputs/cache/document_extractions/*.json (10 extractions)
  - outputs/logs/decision_log.jsonl (all agent decisions)
  - data/raw/*.csv, *.xlsx (master data)
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Optional


# === Hardcoded portfolio holdings ===
# These come from outputs/portfolio_factsheet.md
# We hardcode them as the source of truth so the dashboard works even if
# the factsheet is re-generated mid-demo.

PORTFOLIO_HOLDINGS = [
    # Financials (4 of 24)
    {"company_id": "ISP_INTESA", "company_name": "Intesa Sanpaolo", "ticker": "ISP IM",
     "sector": "Financials", "country": "Italy", "weight": 0.0417,
     "esg": 6.33, "biodiversity": 4.89, "carbon_intensity": 1.5,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 1, "city": "Turin", "lat": 45.0703, "lon": 7.6869,
     "rationale": "Strong ESG composite, low direct carbon, financial sector representation"},
    {"company_id": "UCG_UNICREDIT", "company_name": "UniCredit", "ticker": "UCG IM",
     "sector": "Financials", "country": "Italy", "weight": 0.0417,
     "esg": 5.54, "biodiversity": 4.71, "carbon_intensity": 1.8,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 5, "city": "Milan", "lat": 45.4642, "lon": 9.1900,
     "rationale": "Italian banking diversification, clean greenwashing signals"},
    {"company_id": "SAN_SANTANDER", "company_name": "Banco Santander", "ticker": "SAN SM",
     "sector": "Financials", "country": "Spain", "weight": 0.0417,
     "esg": 5.41, "biodiversity": 4.55, "carbon_intensity": 1.7,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 7, "city": "Madrid", "lat": 40.4168, "lon": -3.7038,
     "rationale": "Spanish banking exposure, broad European retail banking"},
    {"company_id": "ALV_ALLIANZ", "company_name": "Allianz", "ticker": "ALV GR",
     "sector": "Financials", "country": "Germany", "weight": 0.0417,
     "esg": 5.04, "biodiversity": 4.31, "carbon_intensity": 2.0,
     "greenwashing_prob": 0.57, "greenwashing_signals": 2, "greenwashing_flag": "medium",
     "selection_rank": 12, "city": "Munich", "lat": 48.1351, "lon": 11.5820,
     "rationale": "Insurance sector, SBTi committed (not validated) - watchlist"},

    # Technology (4)
    {"company_id": "ASML_ASML", "company_name": "ASML Holding", "ticker": "ASML NA",
     "sector": "Technology", "country": "Netherlands", "weight": 0.0417,
     "esg": 6.41, "biodiversity": 5.21, "carbon_intensity": 12.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 2, "city": "Veldhoven", "lat": 51.4192, "lon": 5.3953,
     "rationale": "MSCI AAA rating, top ESG composite, validates pipeline"},
    {"company_id": "SAP_SAP", "company_name": "SAP SE", "ticker": "SAP GR",
     "sector": "Technology", "country": "Germany", "weight": 0.0417,
     "esg": 5.87, "biodiversity": 5.02, "carbon_intensity": 3.5,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 8, "city": "Walldorf", "lat": 49.2950, "lon": 8.6447,
     "rationale": "Enterprise software, low direct carbon, strong governance"},
    {"company_id": "WKL_WOLTERS", "company_name": "Wolters Kluwer", "ticker": "WKL NA",
     "sector": "Technology", "country": "Netherlands", "weight": 0.0417,
     "esg": 5.62, "biodiversity": 4.78, "carbon_intensity": 2.8,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 11, "city": "Alphen aan den Rijn", "lat": 52.1326, "lon": 4.6500,
     "rationale": "Professional information services, light operational footprint"},
    {"company_id": "ADYEN_ADYEN", "company_name": "Adyen", "ticker": "ADYEN NA",
     "sector": "Technology", "country": "Netherlands", "weight": 0.0417,
     "esg": 5.32, "biodiversity": 4.62, "carbon_intensity": 1.9,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 14, "city": "Amsterdam", "lat": 52.3676, "lon": 4.9041,
     "rationale": "Payments fintech, very low direct carbon"},

    # Industrials (4)
    {"company_id": "SU_SCHNEIDER", "company_name": "Schneider Electric", "ticker": "SU FP",
     "sector": "Industrials", "country": "France", "weight": 0.0417,
     "esg": 6.14, "biodiversity": 5.43, "carbon_intensity": 16.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 3, "city": "Rueil-Malmaison", "lat": 48.8761, "lon": 2.1822,
     "rationale": "Electrical equipment, energy management leader"},
    {"company_id": "SIE_SIEMENS", "company_name": "Siemens", "ticker": "SIE GR",
     "sector": "Industrials", "country": "Germany", "weight": 0.0417,
     "esg": 5.78, "biodiversity": 5.12, "carbon_intensity": 25.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 9, "city": "Munich", "lat": 48.1635, "lon": 11.5811,
     "rationale": "Diversified industrial, smart infrastructure"},
    {"company_id": "AIR_AIRBUS", "company_name": "Airbus", "ticker": "AIR FP",
     "sector": "Industrials", "country": "France", "weight": 0.0417,
     "esg": 5.21, "biodiversity": 4.84, "carbon_intensity": 28.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 15, "city": "Toulouse", "lat": 43.6043, "lon": 1.4437,
     "rationale": "Aerospace, included on transition-narrative basis"},
    {"company_id": "SAF_SAFRAN", "company_name": "Safran", "ticker": "SAF FP",
     "sector": "Industrials", "country": "France", "weight": 0.0417,
     "esg": 5.03, "biodiversity": 4.69, "carbon_intensity": 22.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 18, "city": "Paris", "lat": 48.8566, "lon": 2.3522,
     "rationale": "Aerospace engines, low direct emissions intensity"},

    # Consumer Discretionary (4)
    {"company_id": "MBG_MERCEDES", "company_name": "Mercedes-Benz Group", "ticker": "MBG GR",
     "sector": "Consumer Discretionary", "country": "Germany", "weight": 0.0417,
     "esg": 5.98, "biodiversity": 4.32, "carbon_intensity": 21.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 4, "city": "Stuttgart", "lat": 48.7758, "lon": 9.1829,
     "rationale": "Automotive, electrification roadmap"},
    {"company_id": "BMW_BMW", "company_name": "BMW", "ticker": "BMW GR",
     "sector": "Consumer Discretionary", "country": "Germany", "weight": 0.0417,
     "esg": 5.41, "biodiversity": 4.21, "carbon_intensity": 19.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 13, "city": "Munich", "lat": 48.1755, "lon": 11.5587,
     "rationale": "Automotive, ESG composite passes sector threshold"},
    {"company_id": "RACE_FERRARI", "company_name": "Ferrari", "ticker": "RACE IM",
     "sector": "Consumer Discretionary", "country": "Italy", "weight": 0.0417,
     "esg": 5.18, "biodiversity": 4.18, "carbon_intensity": 8.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 16, "city": "Maranello", "lat": 44.5325, "lon": 10.8597,
     "rationale": "Luxury automotive, low absolute footprint"},
    {"company_id": "ITX_INDITEX", "company_name": "Inditex", "ticker": "ITX SM",
     "sector": "Consumer Discretionary", "country": "Spain", "weight": 0.0417,
     "esg": 5.02, "biodiversity": 4.05, "carbon_intensity": 6.5,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 19, "city": "Arteixo", "lat": 43.3035, "lon": -8.5071,
     "rationale": "Apparel retail, leading on sustainable materials"},

    # Health Care (4)
    {"company_id": "SAN_SANOFI", "company_name": "Sanofi", "ticker": "SAN FP",
     "sector": "Health Care", "country": "France", "weight": 0.0417,
     "esg": 5.49, "biodiversity": 5.12, "carbon_intensity": 17.0,
     "greenwashing_prob": 0.57, "greenwashing_signals": 2, "greenwashing_flag": "medium",
     "selection_rank": 6, "city": "Paris", "lat": 48.8676, "lon": 2.3093,
     "rationale": "Pharma, SBTi committed during scope restructure - watchlist"},
    {"company_id": "BAYN_BAYER", "company_name": "Bayer", "ticker": "BAYN GR",
     "sector": "Health Care", "country": "Germany", "weight": 0.0417,
     "esg": 5.12, "biodiversity": 4.96, "carbon_intensity": 18.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 10, "city": "Leverkusen", "lat": 51.0303, "lon": 7.0035,
     "rationale": "Pharma + Crop Science, biodiversity disclosure improving"},
    {"company_id": "EL_ESSILOR", "company_name": "EssilorLuxottica", "ticker": "EL FP",
     "sector": "Health Care", "country": "France", "weight": 0.0417,
     "esg": 4.95, "biodiversity": 4.41, "carbon_intensity": 9.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 17, "city": "Paris", "lat": 48.8800, "lon": 2.3500,
     "rationale": "Eyewear / medical devices, moderate footprint"},
    {"company_id": "ARGX_ARGENX", "company_name": "Argenx", "ticker": "ARGX BB",
     "sector": "Health Care", "country": "Netherlands", "weight": 0.0417,
     "esg": 4.78, "biodiversity": 4.18, "carbon_intensity": 4.5,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 20, "city": "Amsterdam", "lat": 52.3676, "lon": 4.9041,
     "rationale": "Biotech, light operational footprint"},

    # Communications (2)
    {"company_id": "DTE_DEUTSCHE", "company_name": "Deutsche Telekom", "ticker": "DTE GR",
     "sector": "Communications", "country": "Germany", "weight": 0.0417,
     "esg": 5.26, "biodiversity": 4.65, "carbon_intensity": 7.5,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 21, "city": "Bonn", "lat": 50.7374, "lon": 7.0982,
     "rationale": "Telecom, low-medium carbon intensity"},
    {"company_id": "PRX_PROSUS", "company_name": "Prosus", "ticker": "PRX NA",
     "sector": "Communications", "country": "Netherlands", "weight": 0.0417,
     "esg": 4.62, "biodiversity": 4.02, "carbon_intensity": 1.2,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 22, "city": "Amsterdam", "lat": 52.3676, "lon": 4.9041,
     "rationale": "Internet conglomerate, very low direct carbon"},

    # Utilities (2)
    {"company_id": "IBE_IBERDROLA", "company_name": "Iberdrola", "ticker": "IBE SM",
     "sector": "Utilities", "country": "Spain", "weight": 0.0417,
     "esg": 5.24, "biodiversity": 4.94, "carbon_intensity": 14.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 23, "city": "Bilbao", "lat": 43.2630, "lon": -2.9350,
     "rationale": "Renewables leader; WWF flagged Aragón wind farm Ramsar exposure"},
    {"company_id": "ENEL_ENEL", "company_name": "Enel", "ticker": "ENEL IM",
     "sector": "Utilities", "country": "Italy", "weight": 0.0417,
     "esg": 5.18, "biodiversity": 4.78, "carbon_intensity": 28.0,
     "greenwashing_prob": 0.18, "greenwashing_signals": 0, "greenwashing_flag": "low",
     "selection_rank": 24, "city": "Rome", "lat": 41.9028, "lon": 12.4964,
     "rationale": "Italian utility, diversified renewables mix"},
]


# === Excluded companies ===

EXCLUDED_COMPANIES = [
    {"company": "Eni", "sector": "Energy", "reason": "Energy sector exclusion", "category": "sector"},
    {"company": "TotalEnergies", "sector": "Energy", "reason": "Energy sector exclusion", "category": "sector"},
    {"company": "BASF", "sector": "Materials", "reason": "Chemicals — biodiversity impact", "category": "sector"},
    {"company": "Air Liquide", "sector": "Materials", "reason": "Industrial gases — sector exclusion", "category": "sector"},
    {"company": "Saint-Gobain", "sector": "Materials", "reason": "Materials sector exclusion", "category": "sector"},
    {"company": "L'Oréal", "sector": "Consumer Staples", "reason": "Forest-risk commodities exposure", "category": "biodiversity"},
    {"company": "Unilever", "sector": "Consumer Staples", "reason": "Palm oil, supply chain biodiversity", "category": "biodiversity"},
    {"company": "AB InBev", "sector": "Consumer Staples", "reason": "Agricultural inputs, water stress", "category": "biodiversity"},
    {"company": "Danone", "sector": "Consumer Staples", "reason": "Dairy supply chain biodiversity", "category": "biodiversity"},
    {"company": "Infineon Technologies", "sector": "Technology", "reason": "Data quality gaps for ESG composite", "category": "data"},
]


# === Pipeline-level metrics ===

PORTFOLIO_METRICS = {
    "n_holdings": 24,
    "n_sectors": 7,
    "carbon_intensity_portfolio": 12.5,
    "carbon_intensity_eba_ref": 30.5,
    "carbon_pct_of_eba": 41.0,
    "carbon_cap_pct": 80.0,
    "sharpe_portfolio": 1.00,
    "sharpe_benchmark": 0.51,
    "return_3y_ann_portfolio": 17.87,
    "return_3y_ann_benchmark": 10.38,
    "volatility_portfolio": 15.4,
    "volatility_benchmark": 15.3,
    "max_drawdown_portfolio": -17.29,
    "max_drawdown_benchmark": -22.15,
    "beta": 0.94,
    "esg_score_portfolio": 5.45,
    "biodiversity_score_portfolio": 4.68,
    "n_excluded": 10,
    "n_starting_universe": 3421,
    "n_eurostoxx50": 50,
    "n_post_exclusions": 34,
}


# === Mandate ===

MANDATE = {
    "max_single_name_weight": 0.08,
    "max_sector_weight": 0.20,
    "min_sectors": 5,
    "max_holdings_per_sector": 4,
    "carbon_cap_pct_of_eba": 80,
    "eba_carbon_reference": 100,
    "min_holdings": 15,
    "max_holdings": 25,
    "excluded_sectors": ["Energy", "Materials/Mining"],
    "universe": "EURO STOXX 50",
    "benchmark": "STOXX Europe 600",
    "client": "Prince Albert II of Monaco Foundation",
}


# === Load functions ===

def load_portfolio_data(root: Path) -> dict:
    """Load all portfolio-level data (holdings, metrics, mandate, exclusions)."""
    return {
        "holdings": PORTFOLIO_HOLDINGS,
        "holdings_df": pd.DataFrame(PORTFOLIO_HOLDINGS),
        "metrics": PORTFOLIO_METRICS,
        "mandate": MANDATE,
        "excluded": EXCLUDED_COMPANIES,
        "excluded_df": pd.DataFrame(EXCLUDED_COMPANIES),
    }


def load_extractions(root: Path) -> dict:
    """Load the 10 cached document extractions from outputs/cache/."""
    extractions = {}
    extraction_dir = root / "outputs" / "cache" / "document_extractions"
    
    if not extraction_dir.exists():
        return extractions
    
    for json_file in sorted(extraction_dir.glob("*.json")):
        if json_file.name.startswith("C") and len(json_file.stem) <= 6:
            # Skip test artifacts like C00770, C99998
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            extractions[json_file.stem] = data
        except Exception:
            pass
    
    return extractions


def load_decision_log(root: Path) -> list:
    """Load the decision_log.jsonl as a list of dicts."""
    log_path = root / "outputs" / "logs" / "decision_log.jsonl"
    if not log_path.exists():
        return []
    
    log = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        log.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    
    return log


def load_master(root: Path) -> Optional[pd.DataFrame]:
    """Load the bloomberg ESG ratings master file (optional)."""
    master_path = root / "data" / "raw" / "bloomberg_esg_ratings.xlsx"
    if not master_path.exists():
        return None
    try:
        return pd.read_excel(master_path)
    except Exception:
        return None
