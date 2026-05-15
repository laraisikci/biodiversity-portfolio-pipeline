"""
Agent 6: Climate Agent.

The lecturer's brief framing:
  "Calculates carbon intensity, WACI, emissions coverage and
   transition-readiness indicators."

This agent computes per-company climate metrics and the portfolio-level
WACI (Weighted Average Carbon Intensity), the industry-standard climate
risk metric used by every institutional asset manager.

Key methodological choices:
1. WACI is the headline metric — comparable across portfolios
2. Scope 1+2 always; Scope 3 when reported, flagged when imputed
3. Sector-relative intensity for ranking (heavy emitters vs their peers)
4. Confidence levels reflect data provenance per the brief's data dictionary

Owner: Role C
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional

from agents.base import BaseAgent
from schemas.confidence import DataPoint, ConfidenceLevel
from schemas.biodiversity import ClimateMetrics


# Column names in the course data (esgEnvSocial)
SCOPE_1_COL = "ghgScope1"
SCOPE_2_COL = "ghgScope2"
SCOPE_3_COL = "ghgScope3"

# Intensity per revenue (already calculated in the data)
INTENSITY_PER_SALES_COL = "co2IntensityPerSalesCalc"
TOT_INTENSITY_PER_SALES_COL = "totGhgCo2EmIntensPerSales"

# Climate policy / target columns (binary or text)
CLIMATE_POLICY_COL = "climateChgPolicy"
EMISSION_REDUCTION_COL = "emissionReduction"

# Default carbon intensity bounds (sanity checks)
INTENSITY_REASONABLE_MAX = 50_000  # tCO2e per EUR million revenue
INTENSITY_REASONABLE_MIN = 0.1


class ClimateAgent(BaseAgent):
    """Calculates per-company climate metrics + supports portfolio-level WACI.

    Inputs: master DataFrame (with ESG E/S data merged in)
    Outputs: List of ClimateMetrics objects, one per company
    """

    name = "climate"

    def run(
        self,
        master: pd.DataFrame,
        sector_column: str = "classificationLevelName1",
    ) -> List[ClimateMetrics]:
        """Compute climate metrics for every company in the master DataFrame.

        Args:
            master: Master DataFrame from Data Ingestion agent.
            sector_column: Column for sector grouping (default BICS L1).

        Returns:
            List of ClimateMetrics objects.
        """
        self.log(
            decision_type="climate_scoring_start",
            details={
                "n_companies": len(master),
                "sector_column": sector_column,
                "scopes_handled": [SCOPE_1_COL, SCOPE_2_COL, SCOPE_3_COL],
                "intensity_metric": "tCO2e per EUR million revenue",
                "framework": "TCFD-aligned, WACI methodology (PCAF)",
            },
            confidence="judgement_based",
            notes=(
                "Climate metrics follow TCFD/PCAF conventions. WACI is computed "
                "as the carbon intensity weighted by portfolio holdings — calculated "
                "at portfolio construction time, not here. This agent produces the "
                "per-company intensities that feed WACI."
            ),
        )

        # Compute sector-relative carbon intensity rankings for context
        sector_medians = self._compute_sector_intensity_medians(master, sector_column)

        metrics = []
        n_full_disclosure = 0
        n_partial_disclosure = 0
        n_no_disclosure = 0
        n_sbti_likely = 0

        for idx, row in master.iterrows():
            company_id = row.get("company_id", f"C{idx:05d}")

            metrics_obj = self._compute_per_company_climate(
                row, sector_medians, sector_column
            )
            metrics.append(metrics_obj)

            # Track disclosure quality for the summary
            scopes_reported = sum([
                metrics_obj.scope_1_emissions is not None
                and metrics_obj.scope_1_emissions.confidence == ConfidenceLevel.REPORTED.value,
                metrics_obj.scope_2_emissions is not None
                and metrics_obj.scope_2_emissions.confidence == ConfidenceLevel.REPORTED.value,
                metrics_obj.scope_3_emissions is not None
                and metrics_obj.scope_3_emissions.confidence == ConfidenceLevel.REPORTED.value,
            ])
            if scopes_reported >= 3:
                n_full_disclosure += 1
            elif scopes_reported >= 1:
                n_partial_disclosure += 1
            else:
                n_no_disclosure += 1

            if metrics_obj.sbti_validated:
                n_sbti_likely += 1

        # Summary log
        self.log(
            decision_type="climate_scoring_complete",
            details={
                "companies_scored": len(metrics),
                "n_full_disclosure_scope_1_2_3": n_full_disclosure,
                "n_partial_disclosure": n_partial_disclosure,
                "n_no_disclosure": n_no_disclosure,
                "n_with_climate_policy": n_sbti_likely,
                "disclosure_rate_pct": round(
                    (n_full_disclosure + n_partial_disclosure) / len(metrics) * 100, 1
                ) if metrics else 0,
            },
            confidence="observed",
        )

        return metrics

    def _compute_per_company_climate(
        self,
        row: pd.Series,
        sector_medians: pd.Series,
        sector_column: str,
    ) -> ClimateMetrics:
        """Build a ClimateMetrics object for one company."""
        company_id = row.get("company_id", "C00000")

        # === Scope 1 ===
        scope_1 = self._safe_float(row.get(SCOPE_1_COL))
        scope_1_dp = None
        if scope_1 is not None:
            scope_1_dp = DataPoint(
                value=round(scope_1, 0),
                unit="tCO2e",
                confidence=ConfidenceLevel.REPORTED,
                source=f"esgEnvSocial CSV ({SCOPE_1_COL})",
                extraction_method="direct field",
                vintage=datetime.now(timezone.utc),
            )

        # === Scope 2 ===
        scope_2 = self._safe_float(row.get(SCOPE_2_COL))
        scope_2_dp = None
        if scope_2 is not None:
            scope_2_dp = DataPoint(
                value=round(scope_2, 0),
                unit="tCO2e",
                confidence=ConfidenceLevel.REPORTED,
                source=f"esgEnvSocial CSV ({SCOPE_2_COL})",
                extraction_method="direct field",
                vintage=datetime.now(timezone.utc),
            )

        # === Scope 3 ===
        scope_3 = self._safe_float(row.get(SCOPE_3_COL))
        scope_3_imputed = False
        scope_3_dp = None
        if scope_3 is not None:
            scope_3_dp = DataPoint(
                value=round(scope_3, 0),
                unit="tCO2e",
                confidence=ConfidenceLevel.REPORTED,
                source=f"esgEnvSocial CSV ({SCOPE_3_COL})",
                extraction_method="direct field",
                vintage=datetime.now(timezone.utc),
            )

        # === Carbon intensity per revenue ===
        # Use the pre-calculated field where available, else derive
        intensity = self._safe_float(row.get(INTENSITY_PER_SALES_COL))
        if intensity is None:
            intensity = self._safe_float(row.get(TOT_INTENSITY_PER_SALES_COL))

        if intensity is not None and INTENSITY_REASONABLE_MIN <= intensity <= INTENSITY_REASONABLE_MAX:
            intensity_dp = DataPoint(
                value=round(intensity, 2),
                unit="tCO2e per EUR million revenue",
                confidence=ConfidenceLevel.REPORTED,
                source=f"esgEnvSocial CSV ({INTENSITY_PER_SALES_COL})",
                extraction_method="direct field (pre-calculated)",
                vintage=datetime.now(timezone.utc),
            )
        else:
            # Estimated/neutral value when no disclosure — use sector median
            sector = row.get(sector_column, "Unknown")
            sector_median_intensity = sector_medians.get(sector, 100.0)
            intensity_dp = DataPoint(
                value=round(float(sector_median_intensity), 2),
                unit="tCO2e per EUR million revenue",
                confidence=ConfidenceLevel.ESTIMATED,
                source="sector median (imputation)",
                extraction_method=f"BICS Level 1 sector median for '{sector}'",
                vintage=datetime.now(timezone.utc),
                notes="No company-level intensity disclosed; imputed to sector median.",
            )

        # === SBTi validation indicator (proxy from climate policy field) ===
        # The course data doesn't have direct SBTi validation, but we can
        # use the climate policy + emission reduction fields as a proxy
        climate_policy = row.get(CLIMATE_POLICY_COL)
        emission_reduction = row.get(EMISSION_REDUCTION_COL)
        # SBTi-likely if company has explicit climate policy AND emission reduction target
        sbti_proxy = bool(
            (climate_policy == "Y" or climate_policy == 1 or climate_policy is True)
            and emission_reduction not in (None, np.nan)
        )

        return ClimateMetrics(
            company_id=company_id,
            scope_1_emissions=scope_1_dp,
            scope_2_emissions=scope_2_dp,
            scope_3_emissions=scope_3_dp,
            carbon_intensity_per_revenue=intensity_dp,
            sbti_validated=sbti_proxy,
            sbti_target_year=None,  # Not in the course data; would come from SBTi public API
            transition_capex_share=None,  # Future: from EU Taxonomy aligned capex %
            scope_3_imputed=scope_3_imputed,
        )

    def _compute_sector_intensity_medians(
        self,
        master: pd.DataFrame,
        sector_column: str,
    ) -> pd.Series:
        """Compute the median carbon intensity per sector for imputation fallback."""
        intensities = pd.to_numeric(
            master.get(INTENSITY_PER_SALES_COL, pd.Series(dtype=float)),
            errors="coerce",
        )
        # Filter to reasonable range
        intensities = intensities[
            (intensities >= INTENSITY_REASONABLE_MIN)
            & (intensities <= INTENSITY_REASONABLE_MAX)
        ]
        sectors = master[sector_column]
        return intensities.groupby(sectors).median().fillna(100.0)

    def compute_portfolio_waci(
        self,
        portfolio_holdings: Dict[str, float],
        climate_metrics: List[ClimateMetrics],
    ) -> float:
        """Compute the portfolio-level WACI.

        WACI = sum( weight_i × carbon_intensity_i ) for all i in portfolio

        Args:
            portfolio_holdings: Dict of company_id -> weight (0-1, sum to 1)
            climate_metrics: List of ClimateMetrics for all candidates.

        Returns:
            WACI in tCO2e per EUR million revenue.
        """
        # Build lookup
        intensity_lookup = {
            m.company_id: m.carbon_intensity_per_revenue.value
            for m in climate_metrics
            if m.carbon_intensity_per_revenue is not None
        }

        waci = 0.0
        coverage = 0.0
        for company_id, weight in portfolio_holdings.items():
            if company_id in intensity_lookup:
                waci += weight * intensity_lookup[company_id]
                coverage += weight

        self.log(
            decision_type="portfolio_waci_computed",
            details={
                "waci": round(waci, 2),
                "coverage_pct": round(coverage * 100, 1),
                "n_holdings": len(portfolio_holdings),
            },
            confidence="observed",
            notes=(
                f"Portfolio WACI: {waci:.1f} tCO2e/€m revenue. "
                f"Coverage: {coverage*100:.0f}% of weights mapped to intensities."
            ),
        )

        return waci

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Convert pandas value to float, returning None for NaN."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
