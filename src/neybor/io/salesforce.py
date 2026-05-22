"""Load Salesforce CSV exports into typed pandas DataFrames.

This is the ONE place that touches the raw CSVs. Downstream code consumes DataFrames
with known dtypes; if a column type is wrong, fix it here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from neybor.config import CSV_FILES, SNAPSHOT_DIR

# Datetime columns (ISO-8601 with Salesforce's +0000 timezone offset)
APP_DATETIME_COLS: tuple[str, ...] = (
    "CreatedDate",
    "dshift__Closed_Date_Time__c",
    "Status_Last_Updated__c",
    "dshift__URA_Scheduled_Call_Date_Time__c",
    "Proposal_Sent_Datec__c",
    "First_Payment_Date__c",
    "Application.Date_of_Birth__c",
    "Date_of_Birth__c",
    "Account.Date_of_Birth__pc",
    "Account.PersonBirthdate",
)

# Date-only columns (no time component)
APP_DATE_COLS: tuple[str, ...] = (
    "dshift__Date_Submitted__c",
    "dshift__Start_Date__c",
    "dshift__End_Date__c",
)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a Salesforce CSV with permissive parsing."""
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        keep_default_na=True,
        na_values=["", "#N/A", "NULL", "null"],
    )


def _coerce_datetimes(df: pd.DataFrame, cols: tuple[str, ...], date_only: bool = False) -> pd.DataFrame:
    """Coerce datetime-like columns to pandas datetime64[ns, UTC]."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def load_applications(snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    """Load the primary Application__c table."""
    df = _read_csv(snapshot_dir / CSV_FILES["applications"])
    df = _coerce_datetimes(df, APP_DATETIME_COLS)
    df = _coerce_datetimes(df, APP_DATE_COLS, date_only=True)
    return df


def load_contracts(snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    df = _read_csv(snapshot_dir / CSV_FILES["contracts"])
    for col in ("CreatedDate", "dshift__Start_Date__c", "dshift__End_Date__c",
                "dshift__Signed_Date__c", "dshift__Move_in_Date__c"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def load_properties(snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    df = _read_csv(snapshot_dir / CSV_FILES["properties"])
    for col in ("CreatedDate", "Main_Lease_Date__c"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def load_units(snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    df = _read_csv(snapshot_dir / CSV_FILES["units"])
    if "CreatedDate" in df.columns:
        df["CreatedDate"] = pd.to_datetime(df["CreatedDate"], errors="coerce", utc=True)
    return df


def load_tenant_solvency(snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    """Load the tenant solvency / professional-situation imputed feature table.

    Provides dense values for fields that are sparse in the raw applications export
    (Professional_Situation__c, Nationality__c, age, monthly_budget_range,
    length_of_stay) — the actual source of the headline modelling matrix.
    """
    return _read_csv(snapshot_dir / CSV_FILES["tenant_solvency"])


def load_all(snapshot_dir: Path = SNAPSHOT_DIR) -> dict[str, pd.DataFrame]:
    """Load every Salesforce object into a single dict."""
    return {
        "applications": load_applications(snapshot_dir),
        "contracts": load_contracts(snapshot_dir),
        "properties": load_properties(snapshot_dir),
        "units": load_units(snapshot_dir),
        "tenant_solvency": load_tenant_solvency(snapshot_dir),
    }
