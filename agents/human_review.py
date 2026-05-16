"""
Agent 10: Human Review Agent.

Implements Lecture 5 slide 44 ("Human Review Layer"):
  "The model proposes; the analyst disposes. Human review is not optional —
   it is where accountability sits."

And slide 45 ("AI governance: what must be documented"):
  "In finance, undocumented AI output is not analysis; it is operational risk.
   Students must show: tools used, prompts, sources, verification, OVERRIDES,
   limitations."

This agent does NOT make decisions. It records analyst decisions that
overrule the recommendations of other agents (especially Agent 9 — Portfolio
Construction, and Agent 8 — Greenwashing). The audit trail this produces is
what makes the pipeline accountable.

Key design choices:
  - Every override requires a written justification (min 20 chars) — prevents
    silent overrides
  - Override types are limited (Literal type) — prevents arbitrary changes
  - All overrides logged to decision_log + a separate audit summary
  - Applied to a copy of the portfolio — never mutates the agent outputs

Owner: Role E (with Analytics Advisor)
"""

from typing import List, Optional
from datetime import datetime, timezone
import copy

from agents.base import BaseAgent
from schemas.human_review import (
    OverrideDecision,
    OverrideAction,
    HumanReviewSummary,
)
from schemas.portfolio import FinalPortfolio, PortfolioHolding
from schemas.greenwashing import GreenwashingFlag


class HumanReviewAgent(BaseAgent):
    """Applies human overrides to AI recommendations and logs the audit trail.

    Inputs:
      - FinalPortfolio from Agent 9
      - GreenwashingFlags from Agent 8
      - List[OverrideDecision] from the human analyst

    Outputs:
      - Adjusted FinalPortfolio (with overrides applied)
      - HumanReviewSummary (audit trail for the report)
    """

    name = "human_review"

    def __init__(self):
        super().__init__()

    def run(
        self,
        portfolio: FinalPortfolio,
        overrides: List[OverrideDecision],
        greenwashing_flags: Optional[List[GreenwashingFlag]] = None,
    ) -> tuple[FinalPortfolio, HumanReviewSummary]:
        """Apply overrides to the portfolio and produce an audit summary.

        Args:
            portfolio: The final portfolio from Agent 9
            overrides: List of OverrideDecision objects from the analyst
            greenwashing_flags: Optional list of GreenwashingFlag objects
                (used for override_greenwashing_flag actions)

        Returns:
            Tuple of (adjusted_portfolio, review_summary)
        """
        self.log(
            decision_type="human_review_start",
            details={
                "n_overrides": len(overrides),
                "n_portfolio_holdings": len(portfolio.holdings),
                "n_greenwashing_flags": len(greenwashing_flags) if greenwashing_flags else 0,
            },
            confidence="judgement_based",
            notes=(
                f"Processing {len(overrides)} analyst overrides against "
                f"Agent 9's portfolio of {len(portfolio.holdings)} holdings."
            ),
        )

        # Make a deep copy of the portfolio — never mutate the input
        adjusted_portfolio = copy.deepcopy(portfolio)
        
        # Build the lookup of greenwashing flags by company_id
        gw_lookup = {}
        if greenwashing_flags:
            gw_lookup = {f.company_id: f for f in greenwashing_flags}

        # Apply each override in chronological order
        applied_overrides = []
        for override in overrides:
            try:
                self._apply_one_override(
                    adjusted_portfolio,
                    override,
                    gw_lookup,
                )
                applied_overrides.append(override)

                self.log(
                    decision_type="override_applied",
                    company_id=override.company_id,
                    details={
                        "action": override.action,
                        "reviewer": override.reviewer_id,
                        "justification": override.justification[:200],
                    },
                    confidence="judgement_based",
                    notes=(
                        f"Override applied by {override.reviewer_id}: "
                        f"{override.action} on {override.company_id}"
                    ),
                )
            except Exception as e:
                # Log but don't crash — other overrides may still apply
                self.log(
                    decision_type="override_failed",
                    company_id=override.company_id,
                    details={
                        "action": override.action,
                        "error": str(e),
                    },
                    confidence="observed",
                    notes=f"Override failed: {e}",
                )

        # Build the summary
        summary = self._build_summary(applied_overrides)

        self.log(
            decision_type="human_review_complete",
            details={
                "n_overrides_applied": summary.n_overrides_total,
                "n_force_include": summary.n_force_include,
                "n_force_exclude": summary.n_force_exclude,
                "n_change_weight": summary.n_change_weight,
                "n_override_flag": summary.n_override_flag,
                "n_watchlist_changes": summary.n_watchlist_changes,
                "reviewers": summary.reviewers,
            },
            confidence="reported",
        )

        return adjusted_portfolio, summary

    # === Override application ===

    def _apply_one_override(
        self,
        portfolio: FinalPortfolio,
        override: OverrideDecision,
        gw_lookup: dict,
    ) -> None:
        """Apply a single override to the portfolio (mutates the portfolio)."""
        action = override.action
        company_id = override.company_id

        if action == "force_include":
            self._force_include(portfolio, override)
        elif action == "force_exclude":
            self._force_exclude(portfolio, override)
        elif action == "change_weight":
            self._change_weight(portfolio, override)
        elif action == "override_greenwashing_flag":
            self._override_greenwashing_flag(override, gw_lookup)
        elif action == "add_to_watchlist":
            self._add_to_watchlist(portfolio, override)
        elif action == "remove_from_watchlist":
            self._remove_from_watchlist(portfolio, override)
        else:
            raise ValueError(f"Unknown override action: {action}")

    def _force_include(
        self,
        portfolio: FinalPortfolio,
        override: OverrideDecision,
    ) -> None:
        """Add a company that wasn't in the portfolio."""
        # Check if already in the portfolio
        existing_ids = {h.company_id for h in portfolio.holdings}
        if override.company_id in existing_ids:
            raise ValueError(
                f"Company {override.company_id} already in portfolio "
                f"(use change_weight instead)"
            )
        # Remove from excluded list if present
        if override.company_id in portfolio.excluded_companies:
            portfolio.excluded_companies.remove(override.company_id)
        # Note: actually adding a new holding requires the full PortfolioHolding
        # data (scores, sector, etc.). For Agent 10's purposes, we record the
        # override intent — the analyst is responsible for re-running Agent 9
        # with this company forced into the candidate set.
        # In practice, force_include is mostly used to UN-exclude a company.

    def _force_exclude(
        self,
        portfolio: FinalPortfolio,
        override: OverrideDecision,
    ) -> None:
        """Remove a company from the portfolio."""
        portfolio.holdings = [
            h for h in portfolio.holdings if h.company_id != override.company_id
        ]
        if override.company_id not in portfolio.excluded_companies:
            portfolio.excluded_companies.append(override.company_id)
        portfolio.exclusion_reasons[override.company_id] = (
            f"Human override by {override.reviewer_id}: {override.justification}"
        )
        # After removing a holding, weights must be renormalised
        self._renormalise_weights(portfolio)

    def _change_weight(
        self,
        portfolio: FinalPortfolio,
        override: OverrideDecision,
    ) -> None:
        """Change a holding's weight."""
        if override.new_weight is None:
            raise ValueError("change_weight requires new_weight to be set")
        found = False
        for holding in portfolio.holdings:
            if holding.company_id == override.company_id:
                holding.weight = override.new_weight
                found = True
                break
        if not found:
            raise ValueError(
                f"Company {override.company_id} not in portfolio — "
                f"use force_include first"
            )
        self._renormalise_weights(portfolio)

    def _override_greenwashing_flag(
        self,
        override: OverrideDecision,
        gw_lookup: dict,
    ) -> None:
        """Override the greenwashing risk flag for a company."""
        if override.new_risk_flag is None:
            raise ValueError(
                "override_greenwashing_flag requires new_risk_flag to be set"
            )
        if override.company_id not in gw_lookup:
            raise ValueError(
                f"No greenwashing flag exists for {override.company_id}"
            )
        flag = gw_lookup[override.company_id]
        flag.risk_flag = override.new_risk_flag
        # Append the override to the inconsistencies list so it's traceable
        flag.structured_data_inconsistencies.append(
            f"[HUMAN OVERRIDE by {override.reviewer_id}]: "
            f"Risk flag changed to '{override.new_risk_flag}'. "
            f"Justification: {override.justification}"
        )

    def _add_to_watchlist(
        self,
        portfolio: FinalPortfolio,
        override: OverrideDecision,
    ) -> None:
        """Add a company to the watchlist."""
        if override.company_id not in portfolio.watchlist:
            portfolio.watchlist.append(override.company_id)

    def _remove_from_watchlist(
        self,
        portfolio: FinalPortfolio,
        override: OverrideDecision,
    ) -> None:
        """Remove a company from the watchlist."""
        if override.company_id in portfolio.watchlist:
            portfolio.watchlist.remove(override.company_id)

    # === Helpers ===

    def _renormalise_weights(self, portfolio: FinalPortfolio) -> None:
        """After holdings change, renormalise weights to sum to 1.0."""
        total_weight = sum(h.weight for h in portfolio.holdings)
        if total_weight == 0:
            return
        for holding in portfolio.holdings:
            holding.weight = holding.weight / total_weight

    def _build_summary(
        self,
        applied_overrides: List[OverrideDecision],
    ) -> HumanReviewSummary:
        """Build the human-readable audit summary."""
        n_force_include = sum(
            1 for o in applied_overrides if o.action == "force_include"
        )
        n_force_exclude = sum(
            1 for o in applied_overrides if o.action == "force_exclude"
        )
        n_change_weight = sum(
            1 for o in applied_overrides if o.action == "change_weight"
        )
        n_override_flag = sum(
            1 for o in applied_overrides if o.action == "override_greenwashing_flag"
        )
        n_watchlist_changes = sum(
            1 for o in applied_overrides
            if o.action in ("add_to_watchlist", "remove_from_watchlist")
        )

        reviewers = sorted(set(o.reviewer_id for o in applied_overrides))

        # Build human-readable text
        lines = [
            "## Human Review Audit Trail",
            "",
            f"Total overrides applied: {len(applied_overrides)}",
            f"Reviewers: {', '.join(reviewers) if reviewers else 'none'}",
            "",
            "### Breakdown by override type",
            "",
            f"- Force include: {n_force_include}",
            f"- Force exclude: {n_force_exclude}",
            f"- Weight change: {n_change_weight}",
            f"- Greenwashing flag override: {n_override_flag}",
            f"- Watchlist changes: {n_watchlist_changes}",
            "",
        ]

        if applied_overrides:
            lines.append("### Individual overrides")
            lines.append("")
            for i, override in enumerate(applied_overrides, 1):
                lines.append(
                    f"**{i}. {override.action}** on `{override.company_id}` "
                    f"({override.company_name or 'unnamed'})"
                )
                lines.append(f"   - Reviewer: {override.reviewer_id}")
                lines.append(f"   - Time: {override.timestamp.isoformat()}")
                lines.append(f"   - Justification: {override.justification}")
                if override.new_weight is not None:
                    lines.append(f"   - New weight: {override.new_weight:.4f}")
                if override.new_risk_flag is not None:
                    lines.append(f"   - New risk flag: {override.new_risk_flag}")
                if override.evidence_sources:
                    lines.append(
                        f"   - Evidence: {', '.join(override.evidence_sources)}"
                    )
                lines.append("")

        summary_text = "\n".join(lines)

        return HumanReviewSummary(
            n_overrides_total=len(applied_overrides),
            n_force_include=n_force_include,
            n_force_exclude=n_force_exclude,
            n_change_weight=n_change_weight,
            n_override_flag=n_override_flag,
            n_watchlist_changes=n_watchlist_changes,
            reviewers=reviewers,
            overrides=applied_overrides,
            summary_text=summary_text,
        )
