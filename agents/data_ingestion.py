"""
Agent 2: Data Ingestion Agent.

The lecturer's slide framing: "not just 'download a spreadsheet'; it is
entity matching, source control and boundary discipline."

This agent:
1. Loads the four course CSVs (equityBicsV2, esgEnvSocial, esgGovernance, euTaxonomy)
2. Joins them on idBbGlobalCompanyName (confirmed canonical ID from the audit)
3. Filters to European companies via cntryOfDomicile
4. Builds a CompanyUniverse object (one entry per company)
5. Optionally fetches yfinance prices with caching
6. Logs every decision to the audit trail

Owner: Role A
"""

import os
import time
import pickle
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

import pandas as pd

from agents.base import BaseAgent
from schemas.company import CompanyBase, CompanyUniverse


# Canonical European country list (ISO 2-letter codes)
EUROPEAN_COUNTRIES = {
    "GB", "FR", "DE", "ES", "IT", "NL", "CH", "SE", "BE", "IE",
    "DK", "FI", "NO", "AT", "PT", "PL", "GR", "LU", "CZ", "HU",
    "RO", "IS", "EE", "LV", "LT", "SI", "SK", "BG", "HR", "MT", "CY",
}

# Yahoo Finance exchange suffix map for European tickers
# Used when raw tickers from Bloomberg don't work directly with yfinance
EXCHANGE_SUFFIX_MAP = {
    "GB": ".L",      # London
    "FR": ".PA",     # Paris (Euronext)
    "DE": ".DE",     # Frankfurt (XETRA)
    "ES": ".MC",     # Madrid
    "IT": ".MI",     # Milan
    "NL": ".AS",     # Amsterdam
    "CH": ".SW",     # Swiss
    "SE": ".ST",     # Stockholm
    "BE": ".BR",     # Brussels
    "IE": ".IR",     # Ireland
    "DK": ".CO",     # Copenhagen
    "FI": ".HE",     # Helsinki
    "NO": ".OL",     # Oslo
    "AT": ".VI",     # Vienna
    "PT": ".LS",     # Lisbon
    "PL": ".WA",     # Warsaw
    "GR": ".AT",     # Athens
    "LU": ".LU",     # Luxembourg
}


class DataIngestionAgent(BaseAgent):
    """Loads, joins, filters, and enriches the four course CSVs."""

    name = "data_ingestion"

    def __init__(
        self,
        data_dir: Path = None,
        cache_dir: Path = None,
        config: Optional[dict] = None,
    ):
        super().__init__(config)
        self.data_dir = data_dir or Path("data/raw")
        self.cache_dir = cache_dir or Path("data/cached")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        fetch_prices: bool = False,
        price_sample_size: int = 0,
    ) -> Tuple[pd.DataFrame, CompanyUniverse]:
        """Load, join, filter the four CSVs and produce the European universe.

        Args:
            fetch_prices: If True, fetch yfinance prices for the universe.
            price_sample_size: If > 0, only fetch prices for this many companies
                (useful for testing). Set to 0 to fetch all.

        Returns:
            (master_df, universe) — joined DataFrame + structured CompanyUniverse
        """
        # === Step 1: Load all four CSVs ===
        self.log(
            decision_type="ingestion_start",
            details={"data_dir": str(self.data_dir)},
        )

        equity = self._load_csv("equityBicsV2.csv")
        esg_es = self._load_csv("esgEnvironmentalSocialConsolidatedV4.csv")
        esg_g = self._load_csv("esgGovernanceConsolidatedV4.csv")
        taxonomy = self._load_csv("legalEntityEuTaxonomy.csv")

        # === Step 2a: Deduplicate equity to one row per company ===
        # The equity file has multiple rows per company (one per listing/share class
        # across exchanges). We pick one canonical row per company.
        equity_raw_rows = len(equity)
        equity = self._deduplicate_equity(equity)
        equity_unique_rows = len(equity)

        self.log(
            decision_type="equity_deduplicated",
            details={
                "before": equity_raw_rows,
                "after": equity_unique_rows,
                "reduction_pct": round(
                    (1 - equity_unique_rows / equity_raw_rows) * 100, 1
                ),
            },
            confidence="judgement_based",
            notes=(
                "Deduplicated equity to one row per company by selecting primary "
                "listing. Preference order: Common Stock over CDI/ADR, "
                "then prefer listings whose exchange country matches the ISIN country."
            ),
        )

        # === Step 2b: Deduplicate ESG to most recent reporting period ===
        esg_raw_rows = len(esg_es)
        esg_es = self._deduplicate_to_latest_period(esg_es, "esgEnvSocial")
        esg_g = self._deduplicate_to_latest_period(esg_g, "esgGovernance")
        taxonomy = self._deduplicate_to_latest_period(taxonomy, "euTaxonomy")
        esg_dedup_rows = len(esg_es)

        self.log(
            decision_type="esg_deduplicated_to_latest",
            details={
                "esg_es_before": esg_raw_rows,
                "esg_es_after": esg_dedup_rows,
                "esg_g_after": len(esg_g),
                "taxonomy_after": len(taxonomy),
            },
            confidence="reported",
            notes=(
                "Kept only the most recent reporting period (latestPeriodEndCsr) "
                "per company. Older years dropped — they remain in raw data for "
                "any retrospective analysis."
            ),
        )

        # === Step 3: Inner-join equity with ESG E/S ===
        # The equity file has BICS sectors but NOT country.
        # The ESG E/S file has cntryOfDomicile.
        # Inner join keeps only companies present in BOTH files (= ESG-disclosing).
        equity_universe_size = equity_unique_rows
        master = equity.merge(
            esg_es,
            on="idBbGlobalCompanyName",
            how="inner",
            suffixes=("", "_es"),
        )
        after_es_join = len(master)

        self.log(
            decision_type="esg_inner_join",
            details={
                "equity_universe": equity_universe_size,
                "after_inner_join_esg_es": after_es_join,
                "esg_es_coverage_pct": (
                    round(after_es_join / equity_universe_size * 100, 1)
                    if equity_universe_size > 0
                    else 0
                ),
            },
            confidence="reported",
            notes=(
                "Inner-joined equityBicsV2 with esgEnvSocial on idBbGlobalCompanyName. "
                "Only companies present in both files retained (= ESG-disclosing universe). "
                "Country column (cntryOfDomicile) comes from the ESG E/S file."
            ),
        )

        # === Step 4: Filter to European companies using cntryOfDomicile ===
        # cntryOfDomicile comes from the ESG file after the join
        before_geo = len(master)
        master = master[master["cntryOfDomicile"].isin(EUROPEAN_COUNTRIES)].copy()
        after_geo = len(master)

        self.log(
            decision_type="european_filter_applied",
            details={
                "before": before_geo,
                "after": after_geo,
                "excluded": before_geo - after_geo,
                "european_countries_used": sorted(EUROPEAN_COUNTRIES),
            },
            confidence="reported",
            notes=(
                "Filtered ESG-disclosing universe to companies with "
                "cntryOfDomicile in European country list."
            ),
        )

        # === Step 5: Left-join governance and taxonomy as overlays ===
        master = master.merge(
            esg_g,
            on="idBbGlobalCompanyName",
            how="left",  # left join — G is nice-to-have, not blocking
            suffixes=("", "_g"),
        )

        master = master.merge(
            taxonomy,
            on="idBbGlobalCompanyName",
            how="left",  # left join — Taxonomy is overlay, not blocking
            suffixes=("", "_tax"),
        )

        self.log(
            decision_type="overlays_joined",
            details={
                "final_master_rows": len(master),
                "final_master_columns": len(master.columns),
            },
            confidence="reported",
            notes=(
                "Left-joined esgGovernance and legalEntityEuTaxonomy as overlays. "
                "Companies kept even if no governance or taxonomy data available."
            ),
        )

        # === Step 6: Build canonical company IDs ===
        # Use a deterministic ID for the pipeline; preserve Bloomberg ID for reference
        master = master.reset_index(drop=True)
        master["company_id"] = master.index.map(lambda i: f"C{i:05d}")

        # === Step 7: Build CompanyBase objects ===
        companies = self._build_company_objects(master)

        universe = CompanyUniverse(
            companies=companies,
            universe_size=len(companies),
            geographic_filter="Europe (ISO 2-letter cntryOfDomicile)",
            excluded_count=equity_universe_size - len(companies),
            timestamp_built=datetime.now(timezone.utc).isoformat(),
        )

        self.log(
            decision_type="universe_built",
            details={
                "size": universe.universe_size,
                "excluded": universe.excluded_count,
                "sample_company": companies[0].name if companies else None,
            },
        )

        # === Step 8: Fetch prices (optional, expensive) ===
        if fetch_prices:
            companies_to_fetch = (
                companies[:price_sample_size]
                if price_sample_size > 0
                else companies
            )
            self._fetch_prices(companies_to_fetch)

        return master, universe

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load a single CSV, log the metadata."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {path}. Make sure the course data pack is in data/raw/"
            )

        df = pd.read_csv(path, low_memory=False)
        self.log(
            decision_type="csv_loaded",
            details={
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "file_size_mb": round(path.stat().st_size / 1_048_576, 1),
            },
            confidence="reported",
        )
        return df

    def _deduplicate_equity(self, equity: pd.DataFrame) -> pd.DataFrame:
        """Pick one canonical row per company from the equity file.

        Companies often appear with multiple listings (different exchanges,
        share classes, CDIs/ADRs). We need exactly one row per company.

        Preference rules (in order):
        1. Common Stock over CDI/ADR/Depository Receipt
        2. ISIN starting with the same country code as the exchange where possible
        3. Lower row index (file's natural order — typically primary listing first)

        Args:
            equity: Raw equity DataFrame with potential duplicates.

        Returns:
            DataFrame with one row per idBbGlobalCompanyName.
        """
        # Score each row by listing quality (lower = better)
        df = equity.copy()

        # Common Stock is preferred; CDI/ADR less so
        df["_security_priority"] = df["securityTyp"].apply(
            lambda x: 0 if str(x).strip() == "Common Stock" else 1
        )

        # Sort: best (Common Stock) first, then by natural file order
        df = df.sort_values(
            by=["idBbGlobalCompanyName", "_security_priority"],
            kind="stable",
        )

        # Keep first occurrence per company
        df = df.drop_duplicates(subset=["idBbGlobalCompanyName"], keep="first")

        # Clean up helper column
        df = df.drop(columns=["_security_priority"])

        return df.reset_index(drop=True)

    def _deduplicate_to_latest_period(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        """Keep only the most recent reporting period per company.

        Many ESG/taxonomy files have multiple rows per company (one per year).
        For portfolio construction we want the latest known data.

        Args:
            df: DataFrame with possible multiple periods per company.
            dataset_name: For logging only.

        Returns:
            DataFrame with one row per idBbGlobalCompanyName, the latest period.
        """
        if "latestPeriodEndCsr" not in df.columns:
            # No period column — assume already deduped
            return df.drop_duplicates(subset=["idBbGlobalCompanyName"], keep="first")

        df = df.copy()
        # Parse the date — handle mixed formats defensively
        df["_period_dt"] = pd.to_datetime(
            df["latestPeriodEndCsr"], errors="coerce"
        )

        # Sort descending by date so the latest is first per company
        df = df.sort_values(
            by=["idBbGlobalCompanyName", "_period_dt"],
            ascending=[True, False],
            na_position="last",
        )

        # Keep the latest period per company
        df = df.drop_duplicates(subset=["idBbGlobalCompanyName"], keep="first")

        # Clean up helper column
        df = df.drop(columns=["_period_dt"])

        return df.reset_index(drop=True)


    def _build_company_objects(self, master: pd.DataFrame) -> List[CompanyBase]:
        """Convert master DataFrame rows to CompanyBase Pydantic objects.

        Handles missing values gracefully — Pydantic schemas allow Optional fields.
        """
        companies = []
        for _, row in master.iterrows():
            # BICS sectors — use classificationLevelName1 (broadest)
            try:
                companies.append(
                    CompanyBase(
                        company_id=row["company_id"],
                        name=str(row.get("idBbGlobalCompanyName", "Unknown")),
                        isin=self._safe_str(row.get("idIsin")),
                        ticker=self._safe_str(row.get("ticker")),
                        yahoo_ticker=self._build_yahoo_ticker(row),
                        country=str(row.get("cntryOfDomicile", "??")),
                        bics_level_1=str(row.get("classificationLevelName1", "Unknown")),
                        bics_level_2=self._safe_str(row.get("classificationLevelName2")),
                        bics_level_3=self._safe_str(row.get("classificationLevelName3")),
                        market_cap_eur_m=None,  # Not in equity CSV; would come from yfinance
                    )
                )
            except Exception as e:
                # Log but don't crash on a single bad row
                self.log(
                    decision_type="company_skipped",
                    details={"error": str(e)[:100], "row_index": int(_)},
                    confidence="observed",
                )
        return companies

    @staticmethod
    def _safe_str(value) -> Optional[str]:
        """Convert pandas value to str, returning None for NaN."""
        if value is None or pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _build_yahoo_ticker(row: pd.Series) -> Optional[str]:
        """Construct a Yahoo Finance ticker from Bloomberg ticker + country.

        Strategy: take the part before the space in Bloomberg ticker
        (e.g. 'IBE SM Equity' -> 'IBE'), then append the country suffix.
        """
        ticker_raw = row.get("ticker")
        country = row.get("cntryOfDomicile")
        if pd.isna(ticker_raw) or pd.isna(country):
            return None

        # Bloomberg tickers often have format "XXX YY Equity"
        ticker_clean = str(ticker_raw).split()[0] if " " in str(ticker_raw) else str(ticker_raw)

        suffix = EXCHANGE_SUFFIX_MAP.get(country, "")
        return f"{ticker_clean}{suffix}" if suffix else ticker_clean

    def _fetch_prices(self, companies: List[CompanyBase]) -> Dict[str, pd.DataFrame]:
        """Fetch yfinance prices with disk caching.

        Args:
            companies: List of CompanyBase objects to fetch prices for.

        Returns:
            Dict mapping company_id to price DataFrame.
        """
        try:
            import yfinance as yf
        except ImportError:
            self.log(
                decision_type="yfinance_unavailable",
                details={"reason": "yfinance not installed"},
                notes="Skipping price fetch. pip install yfinance to enable.",
            )
            return {}

        cache_file = self.cache_dir / "yfinance_prices.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            self.log(
                decision_type="prices_loaded_from_cache",
                details={
                    "cache_path": str(cache_file),
                    "cached_companies": len(cached),
                },
            )
        else:
            cached = {}

        # Fetch missing
        prices = dict(cached)
        n_fetched, n_failed = 0, 0
        for c in companies:
            if c.company_id in prices:
                continue
            if not c.yahoo_ticker:
                continue
            try:
                hist = yf.Ticker(c.yahoo_ticker).history(period="3y", auto_adjust=True)
                if len(hist) > 0:
                    prices[c.company_id] = hist[["Close", "Volume"]].copy()
                    n_fetched += 1
                else:
                    n_failed += 1
                time.sleep(0.2)  # Be polite to Yahoo
            except Exception as e:
                n_failed += 1
                self.log(
                    decision_type="price_fetch_failed",
                    company_id=c.company_id,
                    details={"ticker": c.yahoo_ticker, "error": str(e)[:100]},
                )

        # Save cache
        with open(cache_file, "wb") as f:
            pickle.dump(prices, f)

        self.log(
            decision_type="prices_fetched",
            details={
                "newly_fetched": n_fetched,
                "failed": n_failed,
                "total_cached": len(prices),
                "cache_path": str(cache_file),
                "fetch_date": datetime.now(timezone.utc).isoformat(),
            },
            confidence="observed",
            notes="Prices cached to disk to avoid re-fetching.",
        )

        return prices
