"""Cleaning — apply the three filtering steps from thesis Section 4.3.

1. Retain only absorbing-state applications (Completed or Rejected) → 747 records
2. Remove records flagged as 'Testing' (49 dummy records in real export)
3. Treat 'Unreachable' as a sensitivity-analysis variant, not a hard exclude

End state: a DataFrame with a binary `target` column (1 = Completed, 0 = Rejected)
and a boolean `is_unreachable_sensitivity` flag for the optional inclusion variant.
"""
from __future__ import annotations

import logging

import pandas as pd

from neybor.config import (
    ABSORBING_STATES,
    CREATED_DATE_FIELD,
    EXCLUDE_REJECTION_REASON,
    MOVE_IN_FIELD,
    POSITIVE_STATE,
    STATUS_FIELD,
    UNREACHABLE_STATE,
)
from neybor.features.leakage import SENSITIVE_FIELDS, filter_to_allowed

log = logging.getLogger(__name__)


def clean_applications(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the three-step filter and add the target column."""
    n_initial = len(df)
    log.info("Cleaning applications: %d initial records", n_initial)

    if STATUS_FIELD not in df.columns:
        raise KeyError(
            f"Status column '{STATUS_FIELD}' missing. Available: {list(df.columns)[:10]}..."
        )

    # ------------------------------------------------------------------
    # Step 1: keep only absorbing-state records (+ Unreachable for sensitivity)
    # ------------------------------------------------------------------
    keep_states = set(ABSORBING_STATES) | {UNREACHABLE_STATE}
    df = df[df[STATUS_FIELD].isin(keep_states)].copy()
    n_after_state = len(df)
    log.info(
        "After absorbing-state filter: %d records (dropped %d transient)",
        n_after_state, n_initial - n_after_state,
    )

    # ------------------------------------------------------------------
    # Step 2: remove 'Testing' rejection-reason rows (dummy data)
    # ------------------------------------------------------------------
    if "Rejected_Lost_Reason__c" in df.columns:
        is_test = df["Rejected_Lost_Reason__c"] == EXCLUDE_REJECTION_REASON
        df = df[~is_test].copy()
        n_after_test = len(df)
        log.info(
            "After 'Testing' exclusion: %d records (dropped %d test rows)",
            n_after_test, n_after_state - n_after_test,
        )

    # ------------------------------------------------------------------
    # Step 3: flag Unreachable for sensitivity analysis
    # ------------------------------------------------------------------
    df["is_unreachable_sensitivity"] = df[STATUS_FIELD] == UNREACHABLE_STATE
    df["target"] = (df[STATUS_FIELD] == POSITIVE_STATE).astype(int)

    log.info(
        "Final cleaned sample: N=%d, positives=%d (%.1f%%), Unreachable flagged=%d",
        len(df),
        df["target"].sum(),
        100 * df["target"].mean(),
        df["is_unreachable_sensitivity"].sum(),
    )

    return df.reset_index(drop=True)


def primary_sample(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Primary modelling sample (Completed + Rejected only, no Unreachable)."""
    return df_cleaned[~df_cleaned["is_unreachable_sensitivity"]].reset_index(drop=True)


def select_model_feature_columns(
    df: pd.DataFrame,
    *,
    include_sensitive: bool = False,
) -> list[str]:
    """Return leakage-safe feature columns for conversion modelling.

    The pipeline keeps raw columns through cleaning and enrichment so audits remain
    transparent. This function is the final gate before modelling: only fields in
    the leakage registry survive, raw datetimes are removed after feature engineering,
    and structural identifiers are dropped.
    """
    drop_cols = {
        "target",
        "is_unreachable_sensitivity",
        STATUS_FIELD,
        CREATED_DATE_FIELD,
        MOVE_IN_FIELD,
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    feature_cols = filter_to_allowed(feature_cols)
    feature_cols = [
        c
        for c in feature_cols
        if c != "Id" and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if not include_sensitive:
        feature_cols = [
            c
            for c in feature_cols
            if c not in SENSITIVE_FIELDS
            and c.removesuffix("_was_missing") not in SENSITIVE_FIELDS
        ]
    return feature_cols


def model_ready_frame(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    include_sensitive: bool = False,
) -> pd.DataFrame:
    """Return a persisted model-ready frame: safe features plus `target`."""
    if "target" not in df.columns:
        raise KeyError("target column missing; run clean_applications() before modelling")

    selected = (
        select_model_feature_columns(df, include_sensitive=include_sensitive)
        if feature_cols is None
        else feature_cols
    )
    return df[[*selected, "target"]].copy()
