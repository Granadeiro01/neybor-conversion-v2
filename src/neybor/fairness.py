"""Fairness and sensitivity diagnostics for operational conversion scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def group_sensitivity_report(
    df: pd.DataFrame,
    y_true,
    y_score,
    *,
    sensitive_columns: list[str],
    threshold: float,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """Summarize outcomes and model treatment rates by sensitive/proxy groups."""
    work = df.copy()
    work["target"] = np.asarray(y_true).astype(int)
    work["score"] = np.asarray(y_score).astype(float)
    work["selected"] = (work["score"] >= threshold).astype(int)

    rows = []
    for column in sensitive_columns:
        if column not in work.columns:
            continue
        for group_value, group in work.groupby(column, dropna=False):
            if len(group) < min_group_size:
                continue
            selected = group[group["selected"] == 1]
            rows.append({
                "feature": column,
                "group": "__missing__" if pd.isna(group_value) else str(group_value),
                "n": len(group),
                "base_conversion_rate": float(group["target"].mean()),
                "mean_score": float(group["score"].mean()),
                "selection_rate": float(group["selected"].mean()),
                "precision_if_selected": (
                    float(selected["target"].mean()) if len(selected) > 0 else np.nan
                ),
                "recall_within_group": (
                    float(selected["target"].sum() / group["target"].sum())
                    if group["target"].sum() > 0
                    else np.nan
                ),
            })
    return pd.DataFrame(rows)


def save_group_sensitivity_report(
    df: pd.DataFrame,
    y_true,
    y_score,
    *,
    sensitive_columns: list[str],
    threshold: float,
    output_dir: Path,
    prefix: str = "fairness",
    min_group_size: int = 10,
) -> Path:
    """Write the group sensitivity report used in the thesis fairness discussion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = group_sensitivity_report(
        df,
        y_true,
        y_score,
        sensitive_columns=sensitive_columns,
        threshold=threshold,
        min_group_size=min_group_size,
    )
    out_path = output_dir / f"{prefix}_group_sensitivity.csv"
    report.to_csv(out_path, index=False)
    return out_path
