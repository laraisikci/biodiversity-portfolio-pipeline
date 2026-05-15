"""
Bloomberg data integration utility.

Cleans the Bloomberg exports and provides functions to:
1. Load and clean the EURO STOXX 50 constituent list
2. Load and clean the ESG ratings cross-check data
3. Filter the master DataFrame to only EURO STOXX 50 companies

Bloomberg exports have quirks (trailing newlines in column names, European
decimal commas in some files) that we normalise here.

Owner: Role A (with Analytics Advisor)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from agents.decision_log import log_decision


def load_eurostoxx50_constituents(
    path: Path = Path("data/raw/eurostoxx50_constituents.xlsx"),
) -> pd.DataFrame:
    """Load and clean the EURO STOXX 50 constituent list from Bloomberg.

    Returns:
        DataFrame with standardised column names and 50 rows.
    """
    df = pd.read_excel(path)

    # Strip whitespace and trailing newlines from column names
    df.columns = [c.strip().replace("\n", "") for c in df.columns]

    # Rename to friendlier snake_case names for code use
    rename_map = {
        "Ticker": "bloomberg_ticker",
        "Name": "company_name",
        "Weight": "index_weight",
        "Shares": "shares",
        "Price": "price_eur",
        "Curncy": "currency",
        "Industry Sector - Realtime": "industry_sector",
        "BICS Level 1 Sector Name": "bics_level_1",
        "Cntry Terrtry Of Dom": "country",
        "Exchange Code": "exchange_code",
        "Curr Mkt Cap Sh Class": "market_cap",
        "ISIN": "isin",
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric columns are floats
    for col in ["price_eur", "market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log_decision(
        agent="bloomberg_integration",
        decision_type="constituents_loaded",
        details={
            "n_companies": len(df),
            "file": str(path),
            "vintage": "Bloomberg Terminal export, May 15 2026",
        },
        confidence="reported",
        notes=(
            "EURO STOXX 50 constituent list pulled from Bloomberg Terminal "
            "at ESADE university computer. Bloomberg Index ticker: SX5E Index."
        ),
    )

    return df


def load_bloomberg_esg_ratings(
    path: Path = Path("data/raw/bloomberg_esg_ratings.xlsx"),
) -> pd.DataFrame:
    """Load and clean the Bloomberg ESG ratings cross-check data.

    Handles European decimal commas ("21,07" → 21.07).

    Returns:
        DataFrame with clean column names and numeric values.
    """
    df = pd.read_excel(path)

    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # Standardise column names
    rename_map = {
        "RepRisk Rating": "reprisk_rating",
        "Controversies": "controversies",
    }
    df = df.rename(columns=rename_map)

    # Convert European decimal comma to dot for sustainalytics_score
    if "sustainalytics_score" in df.columns:
        df["sustainalytics_score"] = (
            df["sustainalytics_score"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace("nan", np.nan)
        )
        df["sustainalytics_score"] = pd.to_numeric(
            df["sustainalytics_score"], errors="coerce"
        )

    # sp_global should already be numeric
    if "sp_global" in df.columns:
        df["sp_global"] = pd.to_numeric(df["sp_global"], errors="coerce")

    log_decision(
        agent="bloomberg_integration",
        decision_type="esg_ratings_loaded",
        details={
            "n_companies": len(df),
            "providers": ["MSCI", "Sustainalytics", "S&P Global", "RepRisk"],
            "file": str(path),
            "vintage": "Bloomberg Terminal export, May 15 2026",
        },
        confidence="reported",
        notes=(
            "ESG ratings from 4 independent providers via Bloomberg Terminal. "
            "Used for external validation of internal scoring (lecture slide 86, "
            "Berg-Kölbel-Rigobon 2022 ratings divergence)."
        ),
    )

    return df


def filter_master_to_eurostoxx50(
    master: pd.DataFrame,
    constituents: pd.DataFrame,
    name_column_master: str = "idBbGlobalCompanyName",
    name_column_constituents: str = "company_name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter the master DataFrame to only EURO STOXX 50 companies.

    Joins by company name with two-stage matching:
    1. Exact normalised match (lowercase, suffix-stripped)
    2. Prefix match for truncated Bloomberg names (e.g. "Banco Bilbao Vizcaya
       Argentari" matches "Banco Bilbao Vizcaya Argentaria SA")

    Args:
        master: The full master DataFrame from data ingestion.
        constituents: The EURO STOXX 50 constituent list.
        name_column_master: Column in master with company names.
        name_column_constituents: Column in constituents with company names.

    Returns:
        (filtered_master, unmatched_constituents)
    """
    # Normalise names for fuzzy matching: lowercase, strip whitespace, simplify
    # Use word-boundary regex so we only strip complete corporate suffix words,
    # not random substrings (e.g. don't turn "Volkswagen" into "Volkswen")
    import re
    SUFFIX_PATTERN = re.compile(
        r"\b(ag|se|nv|sa|plc|spa|sca|kgaa|ab|oyj|gmbh|ltd|llc|inc|corp|nyrt|bv)\b",
        re.IGNORECASE,
    )

    def normalise(name):
        if pd.isna(name):
            return ""
        s = str(name).lower().strip()
        # Replace special characters first
        s = s.replace("&", "and").replace(".", "").replace(",", "")
        s = s.replace("-", " ").replace("/", " ")
        # Strip corporate suffixes (whole words only, not substrings)
        s = SUFFIX_PATTERN.sub("", s)
        # Collapse whitespace
        s = " ".join(s.split())
        return s.strip()

    master = master.copy()
    constituents = constituents.copy()
    master["_norm_name"] = master[name_column_master].apply(normalise)
    constituents["_norm_name"] = constituents[name_column_constituents].apply(normalise)

    # === Stage 1: exact normalised match ===
    master_norm_set = set(master["_norm_name"])
    constituents["_matched"] = constituents["_norm_name"].isin(master_norm_set)

    # === Stage 2: prefix match for truncated names ===
    # If a constituent's normalised name is a prefix of any master name,
    # consider it matched (handles Bloomberg's ~30-char truncation)
    for idx, row in constituents[~constituents["_matched"]].iterrows():
        c_norm = row["_norm_name"]
        if len(c_norm) < 10:  # Too short to safely prefix-match
            continue
        # Find any master company whose normalised name starts with this prefix
        matches = master[master["_norm_name"].str.startswith(c_norm, na=False)]
        if len(matches) == 1:
            # Unique prefix match — accept it
            # Update master's norm name to this constituent's name so the
            # filter step catches it
            master.loc[matches.index, "_norm_name"] = c_norm
            constituents.loc[idx, "_matched"] = True

    # Now filter master to companies matched via either stage
    matched_norms = set(constituents[constituents["_matched"]]["_norm_name"])
    matched = master[master["_norm_name"].isin(matched_norms)].copy()

    # Unmatched constituents are those not in the master after both stages
    unmatched = constituents[~constituents["_matched"]].copy()

    # Clean up helper columns
    matched = matched.drop(columns=["_norm_name"], errors="ignore")
    constituents = constituents.drop(columns=["_norm_name", "_matched"], errors="ignore")
    unmatched = unmatched.drop(columns=["_norm_name", "_matched"], errors="ignore")

    log_decision(
        agent="bloomberg_integration",
        decision_type="universe_filtered_to_eurostoxx50",
        details={
            "master_size_before": len(master),
            "constituents_target": len(constituents),
            "matched": len(matched),
            "unmatched_constituents": len(unmatched),
            "match_rate_pct": round(len(matched) / len(constituents) * 100, 1)
            if len(constituents) > 0 else 0,
        },
        confidence="judgement_based",
        notes=(
            f"Two-stage name matching (exact then prefix). "
            f"Matched: {len(matched)} of {len(constituents)} constituents."
        ),
    )

    return matched.reset_index(drop=True), unmatched


def attach_bloomberg_ratings(
    master: pd.DataFrame,
    esg_ratings: pd.DataFrame,
    name_column_master: str = "idBbGlobalCompanyName",
    name_column_ratings: str = "company_name",
) -> pd.DataFrame:
    """Attach Bloomberg ESG ratings to the master DataFrame.

    Uses the same two-stage name matching as filter_master_to_eurostoxx50:
    exact match first, then prefix match for truncated names.

    Args:
        master: Master DataFrame (already filtered to EURO STOXX 50).
        esg_ratings: Cleaned Bloomberg ESG ratings.

    Returns:
        Master DataFrame with bloomberg_* columns appended.
    """
    import re
    SUFFIX_PATTERN = re.compile(
        r"\b(ag|se|nv|sa|plc|spa|sca|kgaa|ab|oyj|gmbh|ltd|llc|inc|corp|nyrt|bv)\b",
        re.IGNORECASE,
    )

    def normalise(name):
        if pd.isna(name):
            return ""
        s = str(name).lower().strip()
        s = s.replace("&", "and").replace(".", "").replace(",", "")
        s = s.replace("-", " ").replace("/", " ")
        s = SUFFIX_PATTERN.sub("", s)
        s = " ".join(s.split())
        return s.strip()

    master = master.copy()
    esg_ratings = esg_ratings.copy()
    master["_norm_name"] = master[name_column_master].apply(normalise)
    esg_ratings["_norm_name"] = esg_ratings[name_column_ratings].apply(normalise)

    # Two-stage matching as in the filter function
    master_norm_set = set(master["_norm_name"])

    # Stage 2: prefix-match for ratings whose name isn't in master
    for idx, row in esg_ratings.iterrows():
        e_norm = row["_norm_name"]
        if e_norm in master_norm_set:
            continue
        if len(e_norm) < 10:
            continue
        # Find a unique master row whose name starts with this prefix
        matches = master[master["_norm_name"].str.startswith(e_norm, na=False)]
        if len(matches) == 1:
            # Update the ESG ratings row's name to match the master
            esg_ratings.loc[idx, "_norm_name"] = matches["_norm_name"].iloc[0]

    # Rename rating columns to bloomberg_ prefix
    rating_cols = ["msci_rating", "sustainalytics_score", "sp_global",
                   "reprisk_rating", "controversies"]
    available_cols = [c for c in rating_cols if c in esg_ratings.columns]
    esg_subset = esg_ratings[["_norm_name"] + available_cols]
    esg_subset = esg_subset.rename(columns={c: f"bloomberg_{c}" for c in available_cols})

    merged = master.merge(esg_subset, on="_norm_name", how="left")
    merged = merged.drop(columns=["_norm_name"])

    n_attached = (
        merged["bloomberg_msci_rating"].notna().sum()
        if "bloomberg_msci_rating" in merged.columns else 0
    )

    log_decision(
        agent="bloomberg_integration",
        decision_type="bloomberg_ratings_attached",
        details={
            "n_companies_in_master": len(master),
            "n_attached": int(n_attached),
            "match_rate_pct": round(n_attached / len(master) * 100, 1)
            if len(master) > 0 else 0,
        },
        confidence="reported",
        notes=(
            "Attached Bloomberg ESG ratings (MSCI, Sustainalytics, S&P Global, "
            "RepRisk) to master for external cross-validation. "
            "Two-stage name matching to handle Bloomberg's name truncation."
        ),
    )

    return merged
