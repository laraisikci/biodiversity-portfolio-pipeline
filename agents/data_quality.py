"""
EXAMPLE AGENT — Data Quality Agent (belongs to Role A).

This is a working stub demonstrating the pattern. Use it as the template
for your own agent.

What it does: profiles a DataFrame, flags missing values, logs every
decision, returns a coverage report.
"""

import pandas as pd
from typing import Dict
from agents.base import BaseAgent


class DataQualityAgent(BaseAgent):
    """Profiles input data and flags coverage issues.

    Inputs: pandas DataFrame
    Outputs: coverage report dict

    Logs to the shared decision log every time it flags something.
    """

    name = "data_quality"

    def run(self, df: pd.DataFrame, dataset_name: str = "unnamed") -> Dict:
        """Run the full data quality assessment.

        Args:
            df: The DataFrame to profile
            dataset_name: Human-readable name for logs, e.g. 'esgEnvSocial'

        Returns:
            Coverage report dict with per-column stats and flags.
        """
        report = {
            "dataset_name": dataset_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {},
        }

        for col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0

            col_report = {
                "null_count": int(null_count),
                "null_pct": round(float(null_pct), 2),
                "dtype": str(df[col].dtype),
                "unique_values": int(df[col].nunique()),
            }

            # Flag columns with high missingness
            if null_pct > 50:
                col_report["flag"] = "high_missingness"
                self.log(
                    decision_type="data_quality_flag",
                    details={
                        "dataset": dataset_name,
                        "column": col,
                        "null_pct": round(float(null_pct), 2),
                        "flag": "high_missingness",
                    },
                    confidence="observed",
                    notes=f"Column {col} is >50% null. May need imputation or exclusion.",
                )

            report["columns"][col] = col_report

        # Log summary
        self.log(
            decision_type="dataset_profiled",
            details={
                "dataset": dataset_name,
                "rows": report["row_count"],
                "columns": report["column_count"],
                "columns_with_high_missingness": sum(
                    1
                    for c in report["columns"].values()
                    if c.get("flag") == "high_missingness"
                ),
            },
            confidence="observed",
        )

        return report
