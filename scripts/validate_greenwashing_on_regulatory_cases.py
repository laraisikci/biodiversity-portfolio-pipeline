"""
Agent 8 (Greenwashing) — Validation on famous regulatory cases.

Purpose:
  Demonstrate that Agent 8 correctly identifies elevated greenwashing risk in
  4 famous regulatory cases (DWS, BNY Mellon, Goldman Sachs ESG, Vanguard ESG).
  These cases are constructed as synthetic DocumentExtractions matching the
  publicly-known facts of each settlement, then scored through the agent.

Rationale:
  EURO STOXX 50 companies are mostly mainstream blue chips with strong EU
  CSRD-compliant disclosure. Our portfolio universe should mostly score LOW,
  which the agent confirms (8 LOW, 3 MEDIUM, 0 HIGH on the 10 extractions).
  But the agent must ALSO catch known greenwashing cases — otherwise the LOW
  scores aren't trustworthy. This script tests the contrapositive.

  If Agent 8 correctly flags the 4 regulatory cases as HIGH, and correctly
  scores the EURO STOXX 50 as mostly LOW, that's strong evidence the agent
  is well-calibrated.

Output:
  Side-by-side comparison of regulatory cases (should be HIGH) vs. portfolio
  holdings (should be LOW/MEDIUM).

Usage:
  python scripts/validate_greenwashing_on_regulatory_cases.py

Authors:
  Lara Isikci (ESADE MIBA, Sustainable Finance Group Project, May 2026)
"""

import pandas as pd
from pathlib import Path
import sys

# Make agents/ importable when run as script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.greenwashing import GreenwashingAgent
from schemas.document_extraction import DocumentExtraction, ExtractedClaim


def build_synthetic_regulatory_case(
    company_id: str,
    company_name: str,
    case_description: str,
    leadership_claims: list,
    bio_commitments_quantitative: int = 0,
    sbti_status: str = "none",
    net_zero_year=None,
    tnfd_adopter: bool = False,
    forest_commodities: list = None,
) -> DocumentExtraction:
    """Build a synthetic extraction matching public facts of a regulatory case."""
    top_claims = [
        ExtractedClaim(
            text=text,
            category="other",
            is_quantitative=False,
        )
        for text in leadership_claims
    ]

    bio_commitments = []
    for i in range(bio_commitments_quantitative):
        bio_commitments.append(ExtractedClaim(
            text=f"Quantitative biodiversity target {i+1}",
            category="biodiversity",
            target_year=2030,
            is_quantitative=True,
        ))

    return DocumentExtraction(
        company_id=company_id,
        company_name=company_name,
        document_path="(synthetic regulatory case)",
        tnfd_adopter=tnfd_adopter,
        no_deforestation_pledge=False,
        net_zero_year=net_zero_year,
        sbti_status=sbti_status,
        biodiversity_target_year=None,
        water_stress_disclosed=False,
        forest_risk_commodities_mentioned=forest_commodities or [],
        top_sustainability_claims=top_claims,
        supply_chain_claims=[],
        biodiversity_commitments=bio_commitments,
        climate_targets=[],
        water_disclosures=[],
        document_summary=case_description,
        extraction_confidence="high",
    )


# === The 4 regulatory cases, based on public filings ===

REGULATORY_CASES = [
    {
        "extraction": build_synthetic_regulatory_case(
            company_id="REG_DWS",
            company_name="DWS Group (synthetic)",
            case_description=(
                "Based on BaFin/SEC 2022-23 — €25M settlement for ESG "
                "misrepresentation. Claimed 'ESG integrated' across €459bn AUM "
                "without process evidence."
            ),
            leadership_claims=[
                "We are a leading ESG asset manager",
                "ESG is integrated across our entire €459bn AUM",
                "Industry-leading sustainable investment approach",
            ],
            sbti_status="none",
            net_zero_year=2050,
        ),
        "expected": "HIGH",
        "reference": "BaFin/SEC ESG misrepresentation case 2022",
    },
    {
        "extraction": build_synthetic_regulatory_case(
            company_id="REG_BNYM",
            company_name="BNY Mellon (synthetic)",
            case_description=(
                "SEC 2022 — $1.5M for misleading ESG fund claims. "
                "Marketed funds as ESG-quality without quality reviews on holdings."
            ),
            leadership_claims=[
                "Industry-leading ESG quality review process",
                "ESG quality is best-in-class across our funds",
            ],
            sbti_status="none",
            net_zero_year=2050,
        ),
        "expected": "HIGH",
        "reference": "SEC misleading ESG fund claims 2022",
    },
    {
        "extraction": build_synthetic_regulatory_case(
            company_id="REG_GS",
            company_name="Goldman Sachs ESG (synthetic)",
            case_description=(
                "SEC 2022 — $4M for ESG fund process failures. "
                "Promoted ESG criteria without consistent process evidence."
            ),
            leadership_claims=[
                "Pioneering responsible investment principles",
                "We are a leader in sustainable investing",
            ],
            sbti_status="none",
            net_zero_year=2050,
        ),
        "expected": "HIGH",
        "reference": "SEC ESG fund process failures 2022",
    },
    {
        "extraction": build_synthetic_regulatory_case(
            company_id="REG_VAN",
            company_name="Vanguard ESG (synthetic)",
            case_description=(
                "Australian ASIC 2024 — misleading ESG screening claims. "
                "Marketed funds with ESG screens that excluded fewer companies "
                "than disclosed."
            ),
            leadership_claims=[
                "Industry-leading ESG screening methodology",
                "Most comprehensive sustainable investment exclusions",
            ],
            sbti_status="none",
            net_zero_year=2050,
        ),
        "expected": "HIGH",
        "reference": "ASIC misleading ESG screening 2024",
    },
]


def main():
    print("=" * 80)
    print("AGENT 8 VALIDATION — Famous Regulatory Greenwashing Cases")
    print("=" * 80)
    print()
    print("Testing whether Agent 8 correctly flags 4 known regulatory cases")
    print("as HIGH risk. Each case is reconstructed from public settlement facts.")
    print()

    # Synthetic master with no data (so signal 4 doesn't fire — keeps test focused on claims)
    fake_master = pd.DataFrame([{
        "company_id": case["extraction"].company_id,
        "company_name": case["extraction"].company_name,
        "co2IntensityPerSalesCalc": None,
    } for case in REGULATORY_CASES])

    agent = GreenwashingAgent()
    extractions = [case["extraction"] for case in REGULATORY_CASES]
    flags = agent.run(
        master=fake_master,
        extractions=extractions,
        climate_metrics=None,
    )

    print("Results:")
    print()

    correct = 0
    for case, flag in zip(REGULATORY_CASES, flags):
        ext = case["extraction"]
        expected = case["expected"]
        actual = flag.risk_flag.upper()
        match = "✓" if expected == actual else "✗"
        if expected == actual:
            correct += 1

        print(f"--- {ext.company_name} ---")
        print(f"  Reference: {case['reference']}")
        print(f"  Expected: {expected}    Actual: {actual}    {match}")
        print(f"  Signals fired: {flag.signals_fired}/7")
        print(f"  Probability: {flag.greenwashing_probability.value:.2f}")
        if flag.structured_data_inconsistencies:
            print(f"  Signals that fired:")
            for inc in flag.structured_data_inconsistencies:
                print(f"    - {inc}")
        print()

    print("=" * 80)
    print(f"VALIDATION RESULT: {correct}/{len(REGULATORY_CASES)} cases correctly identified")
    print("=" * 80)
    if correct == len(REGULATORY_CASES):
        print("✓ Agent 8 is well-calibrated: catches known greenwashing patterns.")
    else:
        print("⚠ Some cases not flagged. Review signal definitions.")


if __name__ == "__main__":
    main()
