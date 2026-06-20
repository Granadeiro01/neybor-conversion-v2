"""Engineered features (thesis Section 4.3).

Five derived variables, all computed using only information available at the moment
of application creation. demand_pressure requires a temporal join with strict ordering
to avoid cross-record leakage (thesis 4.3.1).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from neybor.config import CREATED_DATE_FIELD, MOVE_IN_FIELD, PROPERTY_GROUP_FIELD

log = logging.getLogger(__name__)

AGE_AT_APPLICATION_COLUMN = "age_at_application_created"
TENANT_AGE_AT_APPLICATION_COLUMN = "tenant_age_at_application_created"
RAW_AGE_COLUMN = "Age__c"
MAX_PLAUSIBLE_AGE_YEARS = 120

DOB_SOURCE_COLUMNS: tuple[str, ...] = (
    "Application.Date_of_Birth__c",
    "Date_of_Birth__c",
    "Account.Date_of_Birth__pc",
    "Account.PersonBirthdate",
)


def parse_budget_bracket(s: pd.Series) -> pd.Series:
    """Parse Salesforce budget bracket strings into a numeric midpoint.

    Mirrors the conventions used by the headline solvency-feature pipeline so
    that the parsed midpoint matches the values in
    ``application_conversion_model_ready_with_solvency.csv``:

    Examples
    --------
    "€750 - €850"   → 800.0     (range midpoint)
    "<€750"         → 700.0     (below-cap; X - 50)
    "€2450+"        → 2450.0    (open-ended; lower bound only)
    NaN             → NaN
    """
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")

    cleaned = (
        s.fillna("")
        .astype(str)
        .str.replace("€", "", regex=False)
        .str.replace("\xa0", " ")
        .str.strip()
    )

    below_mask = cleaned.str.startswith("<")
    if below_mask.any():
        below_vals = pd.to_numeric(
            cleaned[below_mask].str.lstrip("<").str.strip(), errors="coerce"
        )
        out.loc[below_mask] = below_vals - 50

    open_mask = cleaned.str.endswith("+") & ~below_mask
    if open_mask.any():
        open_lower = pd.to_numeric(
            cleaned[open_mask].str.rstrip("+").str.strip(), errors="coerce"
        )
        out.loc[open_mask] = open_lower

    range_mask = cleaned.str.contains(" - ", regex=False) & ~open_mask & ~below_mask
    if range_mask.any():
        parts = cleaned[range_mask].str.split(" - ", n=1, expand=True)
        lo = pd.to_numeric(parts[0].str.strip(), errors="coerce")
        hi = pd.to_numeric(parts[1].str.strip(), errors="coerce")
        out.loc[range_mask] = (lo + hi) / 2

    plain_mask = ~range_mask & ~open_mask & ~below_mask & (cleaned != "")
    if plain_mask.any():
        plain_vals = pd.to_numeric(cleaned[plain_mask], errors="coerce")
        out.loc[plain_mask] = plain_vals

    return out


def parse_length_of_stay(s: pd.Series) -> pd.Series:
    """Parse Salesforce length-of-stay strings into a numeric month count.

    Examples
    --------
    "<3 months"     → 1.5
    "3-6 months"    → 4.5
    "6-12 months"   → 9.0
    "12 months +"   → 13.0
    NaN             → NaN
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    mapping = {
        "<3 months": 1.5,
        "3-6 months": 4.5,
        "6-12 months": 9.0,
        "12 months +": 13.0,
        "12+ months": 13.0,
    }
    return s.map(mapping).astype("float64")


def add_parsed_picklists(df: pd.DataFrame) -> pd.DataFrame:
    """Materialize numeric versions of the bracketed picklists.

    When the solvency join has run the imputed columns ``Monthly_Budget`` and
    ``length_of_stay`` will exist alongside the raw Salesforce picklists; the
    imputed copies have ~100% fill so we prefer them. Otherwise we fall back to
    ``Monthly_Budget__c`` / ``Length_of_Stay__c``.

    Adds:
      - monthly_budget_midpoint
      - length_of_stay_months
    """
    if "Monthly_Budget" in df.columns:
        df["monthly_budget_midpoint"] = parse_budget_bracket(df["Monthly_Budget"])
    elif "Monthly_Budget__c" in df.columns:
        df["monthly_budget_midpoint"] = parse_budget_bracket(df["Monthly_Budget__c"])
    if "length_of_stay" in df.columns:
        df["length_of_stay_months"] = parse_length_of_stay(df["length_of_stay"])
    elif "Length_of_Stay__c" in df.columns:
        df["length_of_stay_months"] = parse_length_of_stay(df["Length_of_Stay__c"])
    return df


def add_lead_time_days(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute number of days between application creation and move-in date.

    Matches the headline solvency pipeline's convention exactly: ``abs(MOVE_IN -
    CREATED)`` in fractional days. The few records where applicants apply just
    after their requested move-in date thus appear as small positive values
    rather than negatives.
    """
    if MOVE_IN_FIELD not in df.columns:
        log.warning("%s not in DataFrame; skipping lead_time_days", MOVE_IN_FIELD)
        return df
    delta = df[MOVE_IN_FIELD] - df[CREATED_DATE_FIELD]
    df["lead_time_days"] = (delta.dt.total_seconds() / 86_400.0).abs()
    return df


def add_submission_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week (0=Monday), month-of-year, and a continuous time index.

    ``submission_time_index`` is the number of calendar months elapsed since the
    first application month in the data (June 2025 = 0). It gives the model an
    explicit drift control: the data-capture process changed when Neybor Tech took
    over operations, so the relationship between features and conversion is not
    stationary across the observation window (thesis Sections 4.3, 5.1.3). Encoding
    time as a feature lets the model absorb part of that structural shift instead of
    silently confounding it with the substantive predictors.
    """
    created = df[CREATED_DATE_FIELD]
    df["submission_day_of_week"] = created.dt.dayofweek
    df["submission_month"] = created.dt.month
    # Continuous months since June 2025 (the first month present in the export).
    df["submission_time_index"] = (
        (created.dt.year - 2025) * 12 + (created.dt.month - 6)
    ).astype("float64")
    return df


# ---------------------------------------------------------------------------
# Demographic groupings (thesis Sections 4.3, 6.4 Ethical considerations)
# ---------------------------------------------------------------------------
# Coarse European / Non-European grouping of nationality. Using a two-level region
# rather than ~45 individual nationalities (a) avoids one-hot sparsity on a feature
# with many singleton levels and 62% missingness, and (b) limits the granularity at
# which a demographic attribute can drive a prescriptive decision (thesis ethics
# section). NaN nationality is kept as an explicit "Unknown" level rather than
# dropped, because missingness itself is informative and 62% of rows lack it.
EUROPEAN_NATIONALITIES: frozenset[str] = frozenset({
    "French", "German", "Italian", "Belgian", "Spanish", "Portuguese", "Greek",
    "Dutch", "Norwegian", "British", "Slovak", "Irish", "Polish", "Romanian",
    "Swedish", "Icelandic", "Danish", "Cypriot", "Ukrainian", "Estonian",
    "Hungarian", "Finnish", "Lithuanian", "Bulgarian", "Croatian", "Albanian",
    "Russian", "Czech", "Austrian", "Swiss", "Slovenian", "Latvian", "Luxembourgish",
    "Maltese", "Serbian", "Bosnian", "Macedonian", "Montenegrin", "Moldovan",
})

NATIONALITY_REGION_COLUMN = "nationality_region"
AGE_BAND_COLUMN = "age_band"
RAW_NATIONALITY_COLUMN = "Nationality__c"


def add_demographic_groupings(df: pd.DataFrame) -> pd.DataFrame:
    """Coarse, ethics-aware encodings of nationality and age.

    Adds:
      - ``nationality_region``: European / Non-European / Unknown
      - ``age_band``: <=24 / 25-29 / 30-34 / 35-44 / 45+ / Unknown
    """
    if RAW_NATIONALITY_COLUMN in df.columns:
        nat = df[RAW_NATIONALITY_COLUMN].astype("string").str.strip()
        region = pd.Series("Non-European", index=df.index, dtype="object")
        region[nat.isin(EUROPEAN_NATIONALITIES)] = "European"
        region[nat.isna() | (nat == "")] = "Unknown"
        df[NATIONALITY_REGION_COLUMN] = region
    else:
        log.warning("%s not in DataFrame; nationality_region set to Unknown",
                    RAW_NATIONALITY_COLUMN)
        df[NATIONALITY_REGION_COLUMN] = "Unknown"

    age_source = None
    if TENANT_AGE_AT_APPLICATION_COLUMN in df.columns:
        age_source = TENANT_AGE_AT_APPLICATION_COLUMN
    elif AGE_AT_APPLICATION_COLUMN in df.columns:
        age_source = AGE_AT_APPLICATION_COLUMN
    if age_source is not None:
        age = pd.to_numeric(df[age_source], errors="coerce")
        # Treat implausible ages (0 or >120) as missing before binning.
        age = age.where(age.between(15, MAX_PLAUSIBLE_AGE_YEARS), np.nan)
        band = pd.cut(
            age,
            bins=[14, 24, 29, 34, 44, MAX_PLAUSIBLE_AGE_YEARS],
            labels=["<=24", "25-29", "30-34", "35-44", "45+"],
        ).astype("object")
        band = band.where(age.notna(), "Unknown")
        df[AGE_BAND_COLUMN] = band
    else:
        log.warning("No age source column; age_band set to Unknown")
        df[AGE_BAND_COLUMN] = "Unknown"
    return df


def add_age_at_application_created(df: pd.DataFrame) -> pd.DataFrame:
    """Blanket age feature, preferring DOB-derived age and falling back to raw Age__c.

    When the solvency join has populated ``tenant_age_at_application_created``
    (the headline feature) we skip producing the DOB-derived
    ``age_at_application_created`` to keep the model_ready schema aligned with
    the headline CSV — the model uses the tenant_* column in that case.
    """
    if TENANT_AGE_AT_APPLICATION_COLUMN in df.columns:
        log.debug(
            "Skipping %s: %s is present (solvency join provides the headline age)",
            AGE_AT_APPLICATION_COLUMN, TENANT_AGE_AT_APPLICATION_COLUMN,
        )
        return df

    available_dob_cols = [col for col in DOB_SOURCE_COLUMNS if col in df.columns]
    has_existing_age = AGE_AT_APPLICATION_COLUMN in df.columns
    has_raw_age = RAW_AGE_COLUMN in df.columns
    if not available_dob_cols and not has_existing_age and not has_raw_age:
        log.warning("No age source columns in DataFrame; skipping %s", AGE_AT_APPLICATION_COLUMN)
        return df

    age = pd.Series(np.nan, index=df.index, dtype="float64")
    if has_existing_age:
        age = pd.to_numeric(df[AGE_AT_APPLICATION_COLUMN], errors="coerce")

    if available_dob_cols and CREATED_DATE_FIELD in df.columns:
        birth_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        for col in available_dob_cols:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            birth_date = birth_date.combine_first(parsed)

        created = pd.to_datetime(df[CREATED_DATE_FIELD], errors="coerce", utc=True)
        dob_age = created.dt.year - birth_date.dt.year
        birthday_has_passed = (
            (created.dt.month > birth_date.dt.month)
            | ((created.dt.month == birth_date.dt.month) & (created.dt.day >= birth_date.dt.day))
        )
        dob_age = dob_age - (~birthday_has_passed).astype("int64")
        dob_age = dob_age.where(
            birth_date.notna()
            & created.notna()
            & dob_age.between(0, MAX_PLAUSIBLE_AGE_YEARS),
            np.nan,
        )
        age = age.where(age.notna(), dob_age.astype("float64"))
    elif available_dob_cols:
        log.warning("%s not in DataFrame; DOB-derived age unavailable", CREATED_DATE_FIELD)

    if has_raw_age:
        raw_age = pd.to_numeric(df[RAW_AGE_COLUMN], errors="coerce")
        raw_age = raw_age.where(raw_age.between(0, MAX_PLAUSIBLE_AGE_YEARS), np.nan)
        age = age.where(age.notna(), raw_age.astype("float64"))

    df[AGE_AT_APPLICATION_COLUMN] = age.astype("float64")
    return df


def add_demand_pressure(df: pd.DataFrame, group_col: str = PROPERTY_GROUP_FIELD) -> pd.DataFrame:
    """Count prior applications for the same property group at creation time.

    CRITICAL: must use only applications that exist strictly before the focal record's
    CreatedDate. A naive groupby().count() over the full table would leak future
    applications, and same-timestamp rows must not count each other.
    """
    if group_col not in df.columns:
        log.warning("%s not in DataFrame; demand_pressure set to 0", group_col)
        df["demand_pressure"] = 0
        return df

    if CREATED_DATE_FIELD not in df.columns:
        log.warning("%s not in DataFrame; demand_pressure set to 0", CREATED_DATE_FIELD)
        df["demand_pressure"] = 0
        return df

    counts = (
        df.groupby([group_col, CREATED_DATE_FIELD], dropna=False)
        .size()
        .rename("same_timestamp_count")
        .reset_index()
        .sort_values([group_col, CREATED_DATE_FIELD], kind="mergesort")
    )
    counts["demand_pressure"] = (
        counts.groupby(group_col, dropna=False)["same_timestamp_count"].cumsum()
        - counts["same_timestamp_count"]
    )

    df_with_pressure = df.merge(
        counts[[group_col, CREATED_DATE_FIELD, "demand_pressure"]],
        how="left",
        on=[group_col, CREATED_DATE_FIELD],
        sort=False,
    )
    df["demand_pressure"] = df_with_pressure["demand_pressure"].fillna(0).astype(int)
    return df


def add_budget_unit_mismatch(
    df: pd.DataFrame,
    reference_price_col: str = "unit_type_median_price",
) -> pd.DataFrame:
    """Binary indicator: applicant budget below median price of preferred unit type.

    Uses monthly_budget_midpoint (parsed from the Salesforce picklist string) as the
    applicant's budget. Reference price must come from an as-of-safe external
    enrichment column. We intentionally do not derive fallback medians from the
    modelling sample because that would let test-period applicant budgets influence
    train-period features.
    """
    budget_col = "monthly_budget_midpoint"
    if budget_col not in df.columns:
        log.warning("%s missing; budget_unit_mismatch set to 0", budget_col)
        df["budget_unit_mismatch"] = 0
        return df

    if reference_price_col not in df.columns or not df[reference_price_col].notna().any():
        log.warning("%s unavailable; budget_unit_mismatch set to 0", reference_price_col)
        df["budget_unit_mismatch"] = 0
        return df

    reference_price = df[reference_price_col]
    mismatch = (df[budget_col] < reference_price).astype(int)
    mismatch = mismatch.where(df[budget_col].notna() & reference_price.notna(), 0)
    df["budget_unit_mismatch"] = mismatch.astype(int)
    return df


# ---------------------------------------------------------------------------
# Coliving enrichment (domain-rule data construction, thesis Section 4.3)
# ---------------------------------------------------------------------------
# Two operator-supplied business rules that exploit known structure of the
# product. (R2) When type-of-living is missing and the stated monthly budget
# falls in one of the coliving price brackets, the application is treated as a
# Coliving House. (R1) For any Coliving House, the number of rooms is "1-bedroom"
# by construction, because a coliving booking is a single private bedroom; the
# field was simply left blank rather than being genuinely unknown. Both rules
# FILL MISSING values only and never overwrite a value the applicant supplied.
# This is data construction in the advisor's sense, not statistical imputation,
# and is documented as such.
COLIVING_BUDGET_BRACKETS: frozenset[str] = frozenset({"<€750", "€850 - €950", "€1050 - €1250"})
TYPE_OF_LIVING_COLUMN = "Type_Of_Living__c"
ROOMS_COLUMN = "How_Many_Room_Do_You_Need__c"
COLIVING_VALUE = "Coliving House"


def add_coliving_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Type_Of_Living and rooms from operator domain rules (fill-only)."""
    budget_col = "Monthly_Budget" if "Monthly_Budget" in df.columns else "Monthly_Budget__c"

    def _norm(s: pd.Series) -> pd.Series:
        return s.astype("string").str.replace("\xa0", " ", regex=False).str.strip()

    # R2: missing type + coliving budget bracket -> Coliving House
    if TYPE_OF_LIVING_COLUMN in df.columns and budget_col in df.columns:
        type_norm = _norm(df[TYPE_OF_LIVING_COLUMN])
        type_missing = type_norm.isna() | (type_norm == "")
        in_bracket = _norm(df[budget_col]).isin(COLIVING_BUDGET_BRACKETS)
        n2 = int((type_missing & in_bracket).sum())
        df.loc[type_missing & in_bracket, TYPE_OF_LIVING_COLUMN] = COLIVING_VALUE
        log.info("Coliving enrichment R2: filled %d missing Type_Of_Living from budget bracket", n2)

    # R1: Coliving House + missing rooms -> "1-bedroom" (fill-only, no overwrite)
    if TYPE_OF_LIVING_COLUMN in df.columns and ROOMS_COLUMN in df.columns:
        is_coliving = _norm(df[TYPE_OF_LIVING_COLUMN]) == COLIVING_VALUE
        rooms_norm = _norm(df[ROOMS_COLUMN])
        rooms_missing = rooms_norm.isna() | (rooms_norm == "")
        n1 = int((is_coliving & rooms_missing).sum())
        df.loc[is_coliving & rooms_missing, ROOMS_COLUMN] = "1-bedroom"
        log.info("Coliving enrichment R1: filled %d missing rooms for Coliving House", n1)
    return df


def add_all_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all engineered features in the right order.

    Coliving enrichment runs first so the constructed Type_Of_Living and rooms
    values are visible to every downstream step. Picklist parsing then runs
    before budget_unit_mismatch, which depends on the parsed numeric midpoint.
    """
    df = add_coliving_enrichment(df)
    df = add_parsed_picklists(df)
    df = add_lead_time_days(df)
    df = add_submission_temporal(df)
    df = add_age_at_application_created(df)
    df = add_demographic_groupings(df)
    df = add_demand_pressure(df)
    df = add_budget_unit_mismatch(df)
    return df
