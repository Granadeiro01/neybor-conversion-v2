"""Tenant-solvency join — densify sparse Salesforce fields with the imputed CSV.

The raw applications export has near-zero fill on Professional_Situation__c (2.3%),
Nationality__c (3.4%), and most age fields. The headline modelling matrix
(`application_conversion_model_ready_with_solvency.csv`) is built by left-joining
`tenant_professional_solvency_feature_imputed_single.csv` onto applications and
overwriting / adding the dense versions. This module is that join.

Key choices (locked in by the headline CSV's schema):
  - Overwrite, not add, for fields that share the Salesforce name
    (`Professional_Situation__c`, `Nationality__c`).
  - Add, not overwrite, for the picklists that downstream feature engineering
    uses as a numeric source (`Monthly_Budget`, `length_of_stay`). The raw
    `Monthly_Budget__c` / `Length_of_Stay__c` columns stay intact; the parsed
    midpoints later prefer the imputed copies.
  - Add `tenant_age_at_application_created` as a separate column (the headline
    CSV uses this name; the raw DOB-derived `age_at_application_created` keeps
    its existing definition in `engineered.py`).
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

JOIN_KEY_LEFT = "Id"
JOIN_KEY_RIGHT = "salesforce_application_id"

# The exact 19-feature matrix produced by the headline solvency-feature run
# (mirrored from `reports/tables/solvency_csv_rerun_model_run_summary.json`).
# Kept here so the pipeline can reproduce that schema exactly when solvency is
# on. Order matters for the parity CSV.
HEADLINE_FEATURES: tuple[str, ...] = (
    "dshift__Source__c",
    "Gender__c",
    "Nationality__c",
    "Professional_Situation__c",
    "Working_for__c",
    "Monthly_Budget",
    "length_of_stay",
    "Type_Of_Living__c",
    "How_Many_Room_Do_You_Need__c",
    "dshift__MER_Property_Group__c",
    "dshift__MER_Unit_Type__c",
    "monthly_budget_midpoint",
    "length_of_stay_months",
    "lead_time_days",
    "submission_day_of_week",
    "submission_month",
    "tenant_age_at_application_created",
    "demand_pressure",
    "budget_unit_mismatch",
)

# target_col -> source_col in solvency CSV. Overwrite the target IN PLACE (raw is
# kept when the join doesn't have a value).
OVERRIDE_FIELDS: dict[str, str] = {
    "Professional_Situation__c": "tenant_professional_solvency_signal",
    "Nationality__c": "tenant_nationality",
}

# target_col -> source_col in solvency CSV. Add a NEW column under the target
# name. The raw counterpart (e.g. Monthly_Budget__c) is left untouched.
ADDED_FIELDS: dict[str, str] = {
    "Monthly_Budget": "monthly_budget_range",
    "length_of_stay": "length_of_stay",
    "tenant_age_at_application_created": "age_at_application_created",
}


def _project_solvency(solvency: pd.DataFrame) -> pd.DataFrame:
    needed_sources = list(OVERRIDE_FIELDS.values()) + list(ADDED_FIELDS.values())
    keep = [JOIN_KEY_RIGHT] + [c for c in dict.fromkeys(needed_sources) if c in solvency.columns]
    sol = solvency.loc[:, keep].drop_duplicates(subset=[JOIN_KEY_RIGHT])
    return sol


def join_solvency(
    applications: pd.DataFrame,
    solvency: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join solvency features onto applications.

    Parameters
    ----------
    applications
        Cleaned applications frame. Must contain ``Id``.
    solvency
        Tenant solvency CSV loaded by ``load_tenant_solvency``. Must contain
        ``salesforce_application_id`` plus the dense source columns.

    Returns
    -------
    pd.DataFrame
        Applications with overrides applied and the new columns appended.
    """
    if JOIN_KEY_LEFT not in applications.columns:
        raise KeyError(f"applications missing join key '{JOIN_KEY_LEFT}'")
    if JOIN_KEY_RIGHT not in solvency.columns:
        raise KeyError(f"solvency missing join key '{JOIN_KEY_RIGHT}'")

    sol = _project_solvency(solvency)

    # Solvency source columns may collide with target column names (e.g. solvency
    # has `length_of_stay`, and our target is also `length_of_stay`). To keep the
    # override / add semantics explicit we rename every source on the right side
    # before merging, then materialise targets ourselves.
    rename_map = {src: f"_sol__{src}" for src in sol.columns if src != JOIN_KEY_RIGHT}
    sol = sol.rename(columns=rename_map)

    n_left = len(applications)
    df = applications.merge(
        sol, how="left", left_on=JOIN_KEY_LEFT, right_on=JOIN_KEY_RIGHT,
    )
    if len(df) != n_left:
        raise RuntimeError(
            f"join_solvency changed row count: {n_left} -> {len(df)}. "
            "Check duplicate keys in the solvency export."
        )
    matched = df[JOIN_KEY_RIGHT].notna().sum()
    log.info(
        "join_solvency: %d/%d applications matched a solvency record (%.1f%%)",
        matched, n_left, 100 * matched / max(n_left, 1),
    )

    for target_col, source_col in OVERRIDE_FIELDS.items():
        prefixed = f"_sol__{source_col}"
        if prefixed not in df.columns:
            log.warning("solvency missing source column '%s'; skipping override", source_col)
            continue
        if target_col not in df.columns:
            df[target_col] = pd.NA
        before_fill = df[target_col].notna().sum()
        df[target_col] = df[prefixed].combine_first(df[target_col])
        after_fill = df[target_col].notna().sum()
        log.info(
            "Override %s ← %s: fill %d -> %d (+%d)",
            target_col, source_col, before_fill, after_fill, after_fill - before_fill,
        )

    for target_col, source_col in ADDED_FIELDS.items():
        prefixed = f"_sol__{source_col}"
        if prefixed not in df.columns:
            log.warning("solvency missing source column '%s'; skipping add", source_col)
            df[target_col] = pd.NA
            continue
        df[target_col] = df[prefixed]
        log.info(
            "Added %s ← %s: fill %d / %d",
            target_col, source_col, df[target_col].notna().sum(), len(df),
        )

    df = df.drop(columns=[c for c in df.columns if c.startswith("_sol__")])
    if JOIN_KEY_RIGHT in df.columns:
        df = df.drop(columns=[JOIN_KEY_RIGHT])

    return df


def drop_missing_created_date(
    df: pd.DataFrame,
    created_field: str = "CreatedDate",
) -> pd.DataFrame:
    """Drop rows with no ``CreatedDate`` — they cannot be temporally placed.

    The headline run records ``dropped_missing_dates: 19``; this matches that
    behaviour.
    """
    if created_field not in df.columns:
        return df
    n_before = len(df)
    df = df[df[created_field].notna()].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.info(
            "Dropped %d rows with missing %s (kept %d)",
            n_dropped, created_field, len(df),
        )
    return df
