"""
Agent 7: Biodiversity & Nature-Risk Agent.

THE HERO AGENT FOR OPTION C.

The brief's framing: "Assesses biodiversity exposure, nature dependency,
water risk or sector-based nature impact."

Design principles (the methodological differentiators):

1. MULTI-LAYER SCORING — biodiversity is inherently multi-source. We don't
   collapse it into a single black-box number.

2. SECTOR-CONDITIONAL WEIGHTING — biodiversity materiality varies enormously
   by sector. We use different layer weights per BICS Level 1 sector,
   grounded in TNFD/ENCORE published materiality assessments.

3. DOUBLE MATERIALITY EXPLICIT — we capture both:
   - Impact materiality (inside-out): how does this company affect biodiversity?
     → Layer 1 (Taxonomy biodiversity contribution) + Layer 2 (DNSH biodiversity)
   - Financial materiality (outside-in): how does biodiversity loss affect this company?
     → Layer 3 (sector dependency)

4. EVERY SCORE HAS A "PRIMARY DRIVER" — so we can explain in plain English
   why each company scored as it did.

5. REGULATORY EVIDENCE DOMINATES — the EU Taxonomy biodiversity fields are
   the most defensible data we have. Layers 1 and 2 typically get higher
   weight than ESG disclosure self-reporting.

Owner: Role C
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

from agents.base import BaseAgent
from schemas.confidence import DataPoint, ConfidenceLevel
from schemas.biodiversity import BiodiversityRiskScore


# === Column names from the course data ===
# EU Taxonomy biodiversity-specific fields
TAX_BIODIVERSITY_CONTRIB_PCT_REV = "euTaxnmyEstmatdSubstantlContrbtnBiodiverstyPctRevenue"
TAX_BIODIVERSITY_CONTRIB_AMT = "euTaxnmyEstmatdSubstantlContrbtnBiodiverstyAmountRev"
TAX_DNSH_BIODIVERSITY_L1 = "euTaxnmyEstmatdDnshBiodiverstyLevl1"
TAX_DNSH_BIODIVERSITY_L2 = "euTaxnmyEstmatdDnshBiodiverstyLevl2"

# Related EU Taxonomy fields (water, pollution — relevant to biodiversity)
TAX_WATER_CONTRIB_PCT_REV = "euTaxnmyEstmatdSubstantlContrbtnWaterPctRevenue"
TAX_POLLUTION_CONTRIB_PCT_REV = "euTaxnmyEstmatdSubstantlContrbtnPolltnPctRevenue"
TAX_DNSH_WATER_L1 = "euTaxnmyEstmatdDnshWaterLevl1"

# ESG disclosure proxies
ENVIRON_DISCLOSURE_SCORE = "environDisclosureScore"
ESG_DISCLOSURE_SCORE = "esgDisclosureScore"

# Sector classification
SECTOR_COL = "classificationLevelName1"


# === Sector materiality weights ===
# These weights are documented in the report's Section 7 (biodiversity methodology).
# Grounded in TNFD/ENCORE published sector materiality assessments.
#
# Layer 1: Taxonomy biodiversity contribution (positive — what the company does for biodiversity)
# Layer 2: Taxonomy DNSH biodiversity (negative — what the company does against biodiversity)
# Layer 3: Sector dependency on natural capital (financial materiality / outside-in)
# Layer 4: Disclosure quality (TNFD readiness proxy)
#
# Default is equal-weight as fallback for unrecognised sectors.

SECTOR_MATERIALITY_WEIGHTS = {
    "Materials": {
        # Mining, chemicals, forestry — DNSH dominates; these sectors most
        # likely to harm biodiversity through extraction or pollution
        "L1": 0.25, "L2": 0.35, "L3": 0.30, "L4": 0.10,
        "narrative": "DNSH and sector impact dominate — extractive activities",
    },
    "Energy": {
        # Oil & gas — DNSH critical; some Taxonomy alignment via transition
        "L1": 0.15, "L2": 0.40, "L3": 0.30, "L4": 0.15,
        "narrative": "DNSH dominates — extractive activities cause direct biodiversity harm",
    },
    "Utilities": {
        # Renewables, water mgmt — Taxonomy contribution highest
        "L1": 0.35, "L2": 0.25, "L3": 0.25, "L4": 0.15,
        "narrative": "Taxonomy contribution highest — renewables and water management",
    },
    "Consumer Staples": {
        # Food, beverages — agricultural supply chain dependency dominates
        "L1": 0.20, "L2": 0.25, "L3": 0.40, "L4": 0.15,
        "narrative": "Sector dependency dominates — agricultural supply chains",
    },
    "Consumer Discretionary": {
        # Auto, luxury, retail — supply chain dependency (leather, palm oil, cotton)
        "L1": 0.20, "L2": 0.25, "L3": 0.35, "L4": 0.20,
        "narrative": "Supply chain dependency dominates (commodity-driven sectors)",
    },
    "Health Care": {
        # Pharma, biotech — biological ingredient supply chains
        "L1": 0.30, "L2": 0.25, "L3": 0.30, "L4": 0.15,
        "narrative": "Balanced — biological supply chains and innovation potential",
    },
    "Industrials": {
        # Mixed — depends on subsector (aerospace vs cement vs paper)
        "L1": 0.25, "L2": 0.25, "L3": 0.25, "L4": 0.25,
        "narrative": "Mixed materiality across diverse industrial subsectors",
    },
    "Financials": {
        # Banks, insurance — mostly indirect via financed activities
        "L1": 0.25, "L2": 0.20, "L3": 0.30, "L4": 0.25,
        "narrative": "Indirect exposure via financed activities; disclosure quality matters",
    },
    "Technology": {
        # Semiconductors, software — low direct biodiversity exposure
        "L1": 0.15, "L2": 0.20, "L3": 0.25, "L4": 0.40,
        "narrative": "Low direct exposure; disclosure quality is the strongest signal",
    },
    "Communications": {
        # Telecom, media — low direct biodiversity exposure
        "L1": 0.15, "L2": 0.20, "L3": 0.25, "L4": 0.40,
        "narrative": "Low direct exposure; disclosure quality is the strongest signal",
    },
    "Real Estate": {
        # Property — land use directly affects biodiversity
        "L1": 0.25, "L2": 0.35, "L3": 0.25, "L4": 0.15,
        "narrative": "Land use change is the dominant biodiversity issue",
    },
}

# Default weights for any sector not in the matrix
DEFAULT_WEIGHTS = {
    "L1": 0.25, "L2": 0.25, "L3": 0.25, "L4": 0.25,
    "narrative": "Equal-weighted default (sector not in materiality matrix)",
}


# === Sector dependency scores ===
# These are Layer 3 inputs — how much does this sector depend on natural capital?
# Grounded in ENCORE's published dependency/impact assessments at sector level.
# Scale: 0 = very low dependency, 1 = very high dependency
# Score returned is inverted (10 - 10*dep) so HIGHER score = BETTER for biodiversity

SECTOR_NATURE_DEPENDENCY = {
    "Materials":              {"dependency": 0.85, "impact": 0.85},  # Mining, primary materials
    "Energy":                 {"dependency": 0.75, "impact": 0.95},  # Oil & gas extraction
    "Consumer Staples":       {"dependency": 0.85, "impact": 0.65},  # Food/agriculture
    "Utilities":              {"dependency": 0.55, "impact": 0.50},  # Power generation, water
    "Real Estate":            {"dependency": 0.50, "impact": 0.65},  # Land use
    "Industrials":            {"dependency": 0.45, "impact": 0.50},  # Manufacturing mixed
    "Health Care":            {"dependency": 0.45, "impact": 0.25},  # Pharma supply chains
    "Consumer Discretionary": {"dependency": 0.40, "impact": 0.40},  # Retail/luxury supply
    "Financials":             {"dependency": 0.20, "impact": 0.15},  # Indirect via lending
    "Communications":         {"dependency": 0.15, "impact": 0.15},  # Telecom, media
    "Technology":             {"dependency": 0.15, "impact": 0.15},  # Software, semiconductors
}


# Exclusion thresholds
HIGH_SECTOR_RISK_DEPENDENCY_THRESHOLD = 0.75  # Above this = high biodiversity risk
HIGH_DNSH_HARM_THRESHOLD = 1.0  # Above this = significant harm flagged


class BiodiversityAgent(BaseAgent):
    """Multi-layered biodiversity risk scoring.

    Inputs: master DataFrame (joined data including EU Taxonomy fields)
    Outputs: List of BiodiversityRiskScore objects, one per company
    """

    name = "biodiversity"

    def run(
        self,
        master: pd.DataFrame,
        sector_column: str = SECTOR_COL,
    ) -> List[BiodiversityRiskScore]:
        """Compute biodiversity risk scores for every company.

        Args:
            master: Master DataFrame from data ingestion.
            sector_column: Column for sector grouping (default BICS L1).

        Returns:
            List of BiodiversityRiskScore objects.
        """
        self.log(
            decision_type="biodiversity_scoring_start",
            details={
                "n_companies": len(master),
                "layers": {
                    "L1": "EU Taxonomy biodiversity contribution (impact materiality)",
                    "L2": "EU Taxonomy DNSH biodiversity (impact materiality)",
                    "L3": "Sector dependency on natural capital (financial materiality)",
                    "L4": "Disclosure quality (TNFD readiness proxy)",
                },
                "sector_conditional": True,
                "n_sectors_in_matrix": len(SECTOR_MATERIALITY_WEIGHTS),
                "frameworks_referenced": ["TNFD LEAP", "ENCORE", "EU Taxonomy"],
            },
            confidence="judgement_based",
            notes=(
                "Multi-layer biodiversity scoring with sector-conditional weighting. "
                "Captures both impact materiality (Taxonomy contribution + DNSH) and "
                "financial materiality (sector dependency). Weights documented per "
                "BICS Level 1 sector in SECTOR_MATERIALITY_WEIGHTS."
            ),
        )

        scores = []
        n_with_taxonomy_data = 0
        n_excluded = 0
        n_high_biodiversity_positive = 0

        for idx, row in master.iterrows():
            company_id = row.get("company_id", f"C{idx:05d}")
            sector = row.get(sector_column, "Unknown")

            score_obj = self._compute_per_company_biodiversity(
                row, sector, company_id
            )
            scores.append(score_obj)

            # Tracking stats
            if score_obj.encore_dependency_score.confidence == ConfidenceLevel.REPORTED.value:
                n_with_taxonomy_data += 1
            if score_obj.biodiversity_exclusion_flag:
                n_excluded += 1
            if score_obj.composite_biodiversity_score.value >= 7.0:
                n_high_biodiversity_positive += 1

            self.log(
                decision_type="biodiversity_score_computed",
                company_id=company_id,
                details={
                    "composite": score_obj.composite_biodiversity_score.value,
                    "sector": sector,
                    "exclusion_flag": score_obj.biodiversity_exclusion_flag,
                },
            )

        self.log(
            decision_type="biodiversity_scoring_complete",
            details={
                "companies_scored": len(scores),
                "n_with_taxonomy_data": n_with_taxonomy_data,
                "n_with_exclusion_flag": n_excluded,
                "n_high_biodiversity_positive": n_high_biodiversity_positive,
            },
            confidence="observed",
        )

        return scores

    def _compute_per_company_biodiversity(
        self,
        row: pd.Series,
        sector: str,
        company_id: str,
    ) -> BiodiversityRiskScore:
        """Compute a full BiodiversityRiskScore for one company."""
        # === Layer 1: EU Taxonomy biodiversity contribution ===
        l1_score, l1_confidence, l1_has_data = self._compute_layer_1_taxonomy_contribution(row)

        # === Layer 2: EU Taxonomy DNSH biodiversity ===
        l2_score, l2_confidence = self._compute_layer_2_taxonomy_dnsh(row)

        # === Layer 3: Sector dependency (ENCORE-style) ===
        l3_score, l3_dependency, l3_impact = self._compute_layer_3_sector_dependency(sector)

        # === Layer 4: Disclosure quality ===
        l4_score, l4_confidence = self._compute_layer_4_disclosure(row)

        # === Sector-conditional weighting ===
        weights = SECTOR_MATERIALITY_WEIGHTS.get(sector, DEFAULT_WEIGHTS)
        composite = (
            weights["L1"] * l1_score
            + weights["L2"] * l2_score
            + weights["L3"] * l3_score
            + weights["L4"] * l4_score
        )

        # Determine primary driver (which layer contributed most)
        layer_contributions = {
            "L1 (Taxonomy contribution)": weights["L1"] * l1_score,
            "L2 (Taxonomy DNSH)": weights["L2"] * l2_score,
            "L3 (Sector dependency)": weights["L3"] * l3_score,
            "L4 (Disclosure quality)": weights["L4"] * l4_score,
        }
        primary_driver = max(layer_contributions, key=layer_contributions.get)

        # === Exclusion logic ===
        # Flag if: high sector dependency AND DNSH harm signal AND low disclosure
        biodiversity_exclusion_flag = False
        exclusion_reason = None
        if l3_dependency >= HIGH_SECTOR_RISK_DEPENDENCY_THRESHOLD:
            if l2_score < 4.0:  # poor DNSH performance
                biodiversity_exclusion_flag = True
                exclusion_reason = (
                    f"Sector dependency on nature is very high "
                    f"({l3_dependency:.2f}) and DNSH biodiversity signal is "
                    f"weak (Layer 2 score: {l2_score:.1f}/10). High biodiversity risk."
                )
            elif l4_score < 3.0:  # poor disclosure compounds the risk
                biodiversity_exclusion_flag = True
                exclusion_reason = (
                    f"Sector dependency on nature is very high "
                    f"({l3_dependency:.2f}) and disclosure quality is poor "
                    f"(score: {l4_score:.1f}/10). Cannot assess risk credibly."
                )

        # === Build the schema object ===
        confidence_overall = (
            ConfidenceLevel.REPORTED if l1_has_data
            else ConfidenceLevel.ESTIMATED
        )

        return BiodiversityRiskScore(
            company_id=company_id,
            encore_dependency_score=DataPoint(
                value=round(float(l3_dependency), 3),
                unit="0-1 scale (1 = highest dependency on nature)",
                confidence=ConfidenceLevel.ESTIMATED,
                source="ENCORE-style sector mapping based on BICS Level 1",
                extraction_method=f"hardcoded matrix lookup for sector '{sector}'",
                vintage=datetime.now(timezone.utc),
                notes=(
                    "Sector-level proxy for nature dependency. Real ENCORE assessment "
                    "would be at NACE 4-digit level; we approximate at BICS L1."
                ),
            ),
            encore_impact_score=DataPoint(
                value=round(float(l3_impact), 3),
                unit="0-1 scale (1 = highest impact on nature)",
                confidence=ConfidenceLevel.ESTIMATED,
                source="ENCORE-style sector mapping based on BICS Level 1",
                extraction_method=f"hardcoded matrix lookup for sector '{sector}'",
                vintage=datetime.now(timezone.utc),
            ),
            water_stress_score=None,  # Future: WRI Aqueduct integration
            biodiversity_sensitive_areas_overlap=None,  # Future: WWF/IBAT integration
            tnfd_adopter=False,  # Future: TNFD adopter list cross-reference
            cdp_water_score=None,
            cdp_forests_score=None,
            forest_risk_commodity_exposure=[],  # Future: from document intelligence
            forest_500_score=None,
            composite_biodiversity_score=DataPoint(
                value=round(float(composite), 2),
                unit="0-10 scale (higher = better for biodiversity)",
                confidence=confidence_overall,
                source="derived from 4 layers, sector-conditional weighting",
                extraction_method=(
                    f"sector-weighted composite (L1={weights['L1']:.0%}, "
                    f"L2={weights['L2']:.0%}, L3={weights['L3']:.0%}, "
                    f"L4={weights['L4']:.0%}); primary driver: {primary_driver}"
                ),
                vintage=datetime.now(timezone.utc),
                notes=weights.get("narrative"),
            ),
            aggregation_method=(
                f"4-layer composite with sector-conditional weights. "
                f"For sector '{sector}': {weights.get('narrative', 'equal-weighted')}. "
                f"Primary driver of this company's score: {primary_driver}."
            ),
            biodiversity_exclusion_flag=biodiversity_exclusion_flag,
            exclusion_reason=exclusion_reason,
        )

    def _compute_layer_1_taxonomy_contribution(
        self, row: pd.Series
    ) -> Tuple[float, ConfidenceLevel, bool]:
        """Layer 1: EU Taxonomy substantial contribution to biodiversity.

        Returns (score_0_10, confidence, has_real_data).

        - 0% contribution → score 5.0 (neutral; absence of evidence)
        - >0-5% → score 6.0-7.0
        - >5-20% → score 7.5-9.0
        - >20% → score 9.0-10.0
        """
        pct = self._safe_float(row.get(TAX_BIODIVERSITY_CONTRIB_PCT_REV))

        if pct is None:
            # No data — neutral score, estimated
            return 5.0, ConfidenceLevel.ESTIMATED, False

        has_data = True
        if pct <= 0:
            score = 5.0  # No biodiversity-positive revenue but data was reported
        elif pct <= 5:
            score = 5.0 + (pct / 5) * 2.0  # 5.0 → 7.0
        elif pct <= 20:
            score = 7.0 + ((pct - 5) / 15) * 2.0  # 7.0 → 9.0
        else:
            score = 9.0 + min((pct - 20) / 80, 1.0) * 1.0  # 9.0 → 10.0

        return score, ConfidenceLevel.REPORTED, has_data

    def _compute_layer_2_taxonomy_dnsh(
        self, row: pd.Series
    ) -> Tuple[float, ConfidenceLevel]:
        """Layer 2: EU Taxonomy DNSH (Do No Significant Harm) for biodiversity.

        DNSH fields capture whether the company's activities are flagged as
        causing significant harm to biodiversity.

        Lower DNSH values = better (less harm); we invert for the score so
        HIGHER score = BETTER for biodiversity.
        """
        dnsh_l1 = self._safe_float(row.get(TAX_DNSH_BIODIVERSITY_L1))
        dnsh_l2 = self._safe_float(row.get(TAX_DNSH_BIODIVERSITY_L2))

        if dnsh_l1 is None and dnsh_l2 is None:
            # No data
            return 5.0, ConfidenceLevel.ESTIMATED

        # Average available levels (or use whichever is reported)
        values = [v for v in [dnsh_l1, dnsh_l2] if v is not None]
        avg_dnsh = sum(values) / len(values)

        # DNSH scale interpretation (rough):
        # 0 = no harm, higher values = more harm flagged
        # Map to 0-10 score where higher = better (less harm)
        if avg_dnsh <= 0:
            score = 9.0
        elif avg_dnsh <= 0.5:
            score = 7.0
        elif avg_dnsh <= 1.0:
            score = 5.0
        elif avg_dnsh <= 2.0:
            score = 3.0
        else:
            score = 1.0

        return score, ConfidenceLevel.REPORTED

    def _compute_layer_3_sector_dependency(
        self, sector: str
    ) -> Tuple[float, float, float]:
        """Layer 3: ENCORE-style sector dependency on natural capital.

        Returns (score_0_10, dependency_0_1, impact_0_1).

        High dependency = low score (more biodiversity risk).
        """
        sector_data = SECTOR_NATURE_DEPENDENCY.get(
            sector, {"dependency": 0.50, "impact": 0.50}
        )
        dependency = sector_data["dependency"]
        impact = sector_data["impact"]

        # Score is inverted: high dependency/impact → low score
        avg_pressure = (dependency + impact) / 2
        score = 10.0 - (avg_pressure * 10)

        return score, dependency, impact

    def _compute_layer_4_disclosure(
        self, row: pd.Series
    ) -> Tuple[float, ConfidenceLevel]:
        """Layer 4: Disclosure quality (TNFD readiness proxy).

        Uses environmental disclosure score as a proxy. Higher disclosure =
        higher score = more credible biodiversity assessment possible.
        """
        env_score = self._safe_float(row.get(ENVIRON_DISCLOSURE_SCORE))
        esg_score = self._safe_float(row.get(ESG_DISCLOSURE_SCORE))

        # Prefer env-specific, fall back to general ESG
        score_input = env_score if env_score is not None else esg_score

        if score_input is None:
            return 3.0, ConfidenceLevel.ESTIMATED  # No disclosure data → low score

        # Map 0-100 disclosure score to 0-10 layer score
        score = (score_input / 100) * 10

        return score, ConfidenceLevel.REPORTED

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Convert pandas value to float, returning None for NaN."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
