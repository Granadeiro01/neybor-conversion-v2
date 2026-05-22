"""Missingness handling (thesis Section 4.3.2).

Three strategies compared head-to-head:
  (a) median/mode imputation + explicit missingness indicator columns
  (b) iterative MICE imputation
  (c) native XGBoost sparsity-aware split-finding (no imputation)

This module focuses on (a). MICE is delegated to scikit-learn's IterativeImputer
inside the modeling pipeline; (c) is implicit in XGBoost when missing values are
left as NaN.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def fill_rate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column fill rate (1 - missingness rate). Useful for sanity-checking thesis
    Section 4.3.2's claim that several features have fill rates between 64-85%."""
    fill_rates = df.notna().mean().sort_values()
    return pd.DataFrame({
        "column": fill_rates.index,
        "fill_rate": fill_rates.values,
        "n_missing": df.isna().sum().reindex(fill_rates.index).values,
        "n_total": len(df),
    })


def add_missingness_indicators(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    suffix: str = "_was_missing",
) -> tuple[pd.DataFrame, list[str]]:
    """Add boolean `<col>_was_missing` columns for each specified column.

    Returns the augmented DataFrame and the list of indicator column names.
    Per the leakage registry, indicator names must end with `_was_missing`.
    """
    if columns is None:
        # Default: any column with at least one missing AND at least one non-missing value
        columns = [
            c for c in df.columns
            if df[c].isna().any() and df[c].notna().any() and df[c].dtype != bool
        ]

    indicators_added: list[str] = []
    for col in columns:
        if col not in df.columns:
            log.warning("Column %s not in DataFrame; skipping indicator", col)
            continue
        ind_name = f"{col}{suffix}"
        df[ind_name] = df[col].isna().astype(int)
        indicators_added.append(ind_name)

    log.info("Added %d missingness indicators", len(indicators_added))
    return df, indicators_added


def conditional_missingness_diagnostic(
    df: pd.DataFrame,
    target_col: str,
    test_col: str,
) -> dict:
    """Test whether missingness in `target_col` depends on the value of `test_col`.

    A non-trivial dependence is evidence against MCAR and toward MAR. Reports the
    chi-square test for categorical test_col and a t-test for numeric test_col.

    This is a quick diagnostic, not a full MAR/MNAR identification (which is
    fundamentally non-identifiable without additional assumptions).
    """
    if target_col not in df.columns or test_col not in df.columns:
        return {"error": f"Missing columns: {target_col} or {test_col}"}

    is_missing = df[target_col].isna()
    if is_missing.sum() == 0 or is_missing.sum() == len(df):
        return {"note": "No variation in missingness; cannot test"}

    if pd.api.types.is_numeric_dtype(df[test_col]):
        from scipy import stats
        present_vals = df.loc[~is_missing, test_col].dropna()
        missing_vals = df.loc[is_missing, test_col].dropna()
        if len(present_vals) < 5 or len(missing_vals) < 5:
            return {"note": "Sample too small for t-test"}
        t, p = stats.ttest_ind(present_vals, missing_vals, equal_var=False)
        return {
            "test": "welch_t",
            "t_statistic": float(t),
            "p_value": float(p),
            "interpretation": (
                "Evidence against MCAR (toward MAR)" if p < 0.05 else "Consistent with MCAR"
            ),
        }
    else:
        from scipy import stats
        ct = pd.crosstab(is_missing, df[test_col].fillna("__nan__"))
        if ct.shape[1] < 2:
            return {"note": "test_col has no variation"}
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        return {
            "test": "chi_square",
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "interpretation": (
                "Evidence against MCAR (toward MAR)" if p < 0.05 else "Consistent with MCAR"
            ),
        }
