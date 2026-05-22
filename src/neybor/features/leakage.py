"""Leakage registry — single source of truth for which Salesforce fields are safe.

This module mirrors the leakage taxonomy in thesis Section 4.3.1 (temporal, outcome,
cross-record). Every model input MUST be in ALLOWED_AT_SUBMISSION. The pipeline calls
assert_safe() before fitting; tests/test_leakage.py asserts the same registry.

Field names below correspond to the real Salesforce schema — verified against the
production export on 2026-04-27.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# ALLOWED — features available at the moment of application creation
# ---------------------------------------------------------------------------
ALLOWED_NATIVE_FIELDS: frozenset[str] = frozenset({
    # Identity / key — kept as columns through the pipeline but dropped before fit
    "Id",
    "CreatedDate",

    # Applicant-supplied at submission time
    "dshift__Source__c",                          # marketing channel
    "Length_of_Stay__c",                          # months
    "Monthly_Budget__c",
    # Solvency-join densified versions of the picklists above (added when the
    # tenant_solvency CSV is joined). They live alongside the raw fields so the
    # legacy run_pipeline path is unaffected when solvency is off.
    "Monthly_Budget",
    "length_of_stay",
    "dshift__Start_Date__c",                      # desired move-in date
    "Type_Of_Living__c",                          # solo / couple / etc.
    "Location_Preference__c",                     # neighborhood preference
    "Preferred_Unit_type__c",
    "How_Many_Room_Do_You_Need__c",
    "Professional_Situation__c",                  # student / professional / etc.
    "Working_for__c",                             # company / institution
    "Country_of_Origin__c",
    "Nationality__c",

    # Property/unit selection at submission (foreign keys; values are at-submission)
    "dshift__MER_Property_Group__c",              # neighborhood / building cluster
    "dshift__MER_Unit_Type__c",
    "dshift__Property__c",                        # specific property if pre-selected

    # Demographic (gated by fairness analysis - thesis Section 5.3)
    "Gender__c",
})

SENSITIVE_FIELDS: frozenset[str] = frozenset({
    "age_at_application_created",
    "tenant_age_at_application_created",
    "Country_of_Origin__c",
    "Gender__c",
    "Nationality__c",
    "Working_for__c",
})

ALLOWED_ENGINEERED_FIELDS: frozenset[str] = frozenset({
    "lead_time_days",
    "demand_pressure",
    "submission_day_of_week",
    "submission_month",
    "budget_unit_mismatch",
    # Parsed numerics from picklist strings
    "monthly_budget_midpoint",      # parsed from "€750 - €850" → 800
    "length_of_stay_months",        # parsed from "3-6 months" → 4.5
    "age_at_application_created",   # computed from application/account DOB at CreatedDate
    # Solvency-join age column — used as the headline model's age feature when
    # the tenant_solvency CSV is joined.
    "tenant_age_at_application_created",
})

ALLOWED_ENRICHMENT_FIELDS: frozenset[str] = frozenset({
    "property_city",
    "property_country",
    "property_postal_code",
    "property_type",
    # Aggregate unit-level statistics per property group (computed in join.py)
    "group_median_unit_price",
    "group_median_surface",
    "group_n_units",
})


def is_missingness_indicator(name: str) -> bool:
    if not name.endswith("_was_missing"):
        return False
    base_name = name.removesuffix("_was_missing")
    return base_name in ALLOWED_AT_SUBMISSION


ALLOWED_AT_SUBMISSION: frozenset[str] = (
    ALLOWED_NATIVE_FIELDS | ALLOWED_ENGINEERED_FIELDS | ALLOWED_ENRICHMENT_FIELDS
)

# ---------------------------------------------------------------------------
# FORBIDDEN — outcome leakage (thesis Section 4.3.1)
# ---------------------------------------------------------------------------
FORBIDDEN_OUTCOME_LEAK: frozenset[str] = frozenset({
    "dshift__Status__c",                          # the target itself
    "Status",                                     # legacy / safety net
    # Contract-related: only populated when application converted
    "Has_Contract_Signed__c",
    "Has_First_Payment_Paid__c",
    "Has_Active_Contracts__c",
    "Has_Draft_Contracts__c",
    "Has_Cancelled_Expired_Contracts__c",
    "dshift__MER_Contract__c",
    "First_Payment_Amount__c",
    "First_Payment_Date__c",
    # Rejection metadata: only populated when Status == Rejected
    "Rejected_Lost_Reason__c",
    # Closure flags
    "dshift__Is_Closed__c",
    "dshift__Is_Duplicate__c",
    "Is_Room_Change__c",
})

# ---------------------------------------------------------------------------
# FORBIDDEN — temporal leakage (thesis Section 4.3.1)
# ---------------------------------------------------------------------------
FORBIDDEN_TEMPORAL_LEAK: frozenset[str] = frozenset({
    "dshift__Closed_Date_Time__c",
    "Status_Last_Updated__c",
    "Days_in_Current_Status__c",
    "Proposal_Sent_Datec__c",
    "Time_in_Days_Since_New_Lead__c",
    # URA = "Universal Resident Application" booking-engine flow; all post-submission
    "dshift__URA_Booking_Stage__c",
    "dshift__URA_Call_Status__c",
    "dshift__URA_Identity_Verification_Status__c",
    "dshift__URA_Scheduled_Call_Date_Time__c",
    "Identity_Verification_Employment_Status__c",
    "Identity_Verification_Residence_Country__c",
    # Salesforce system fields
    "LastModifiedDate",
    "SystemModstamp",
})

FORBIDDEN: frozenset[str] = FORBIDDEN_OUTCOME_LEAK | FORBIDDEN_TEMPORAL_LEAK

# ---------------------------------------------------------------------------
# DROPPED_NEUTRAL — fields that are neither leaky nor useful as model inputs
# ---------------------------------------------------------------------------
# These get filtered out by `filter_to_allowed()` but are not LEAKY — they're just
# not features (PII, free text, structural keys we don't want as predictors). The
# distinction matters because we don't raise LeakageError on these.
DROPPED_NEUTRAL_FIELDS: frozenset[str] = frozenset({
    "Name",
    "OwnerId",
    "dshift__Email__c",
    "dshift__First_Name__c",
    "dshift__Last_Name__c",
    "dshift__Account__c",
    "dshift__Unit__c",                            # 0% filled
    "Shadow_Lead__c",
    "GCLID__c",
    "Company_Sector__c",                          # 0% filled
    "dshift__MER_Min_Rent_Price__c",              # 0% filled
    "dshift__MER_Max_Rent_Price__c",
    "dshift__End_Date__c",                        # 0.1% filled
    "dshift__Date_Submitted__c",                  # duplicate of CreatedDate
    "dshift__Property_Name__c",
    # Raw age/DOB fields are source-only; the model sees the blanket engineered age.
    "Age__c",
    "Application.Date_of_Birth__c",
    "Date_of_Birth__c",
    "Account.Date_of_Birth__pc",
    "Account.PersonBirthdate",
    # Solvency-join source columns (the dense values are surfaced under the
    # canonical Salesforce names by join_solvency; the raw source columns are
    # not features themselves).
    "salesforce_application_id",
    "tenant_professional_solvency_signal",
    "tenant_age",
    "tenant_nationality",
    "monthly_budget_range",
    "monthly_budget_range_source",
    "length_of_stay_source",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class LeakageError(AssertionError):
    """Raised when a forbidden feature is found in the model input."""


def assert_safe(feature_names: set[str] | list[str] | tuple[str, ...]) -> None:
    """Assert that no forbidden field reaches the model.

    Call this immediately before fit() in every training routine.
    Failing closed is the point: better to crash loudly than to ship a leaky model.
    """
    feature_set = set(feature_names)

    forbidden_present = feature_set & FORBIDDEN
    if forbidden_present:
        raise LeakageError(
            f"Forbidden fields detected in model input: {sorted(forbidden_present)}. "
            f"See src/neybor/features/leakage.py for the registry and thesis "
            f"Section 4.3.1 for the leakage taxonomy."
        )

    unknown = {
        f for f in feature_set
        if f not in ALLOWED_AT_SUBMISSION and not is_missingness_indicator(f)
    }
    if unknown:
        raise LeakageError(
            f"Features not in the allowed registry: {sorted(unknown)}. "
            f"If these are legitimate, add them to ALLOWED_NATIVE_FIELDS or "
            f"ALLOWED_ENGINEERED_FIELDS in this file. The registry is a contract; "
            f"don't bypass it silently."
        )


def filter_to_allowed(columns: list[str]) -> list[str]:
    """Convenience: keep only the columns that are in the allowed registry."""
    return [
        c for c in columns
        if c in ALLOWED_AT_SUBMISSION or is_missingness_indicator(c)
    ]


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    """Bucket each column into allowed / outcome-leak / temporal-leak / dropped / unknown.

    Useful for the pre-flight audit script.
    """
    buckets: dict[str, list[str]] = {
        "allowed": [],
        "outcome_leak": [],
        "temporal_leak": [],
        "dropped_neutral": [],
        "unknown": [],
    }
    for c in columns:
        if c in ALLOWED_AT_SUBMISSION or is_missingness_indicator(c):
            buckets["allowed"].append(c)
        elif c in FORBIDDEN_OUTCOME_LEAK:
            buckets["outcome_leak"].append(c)
        elif c in FORBIDDEN_TEMPORAL_LEAK:
            buckets["temporal_leak"].append(c)
        elif c in DROPPED_NEUTRAL_FIELDS:
            buckets["dropped_neutral"].append(c)
        else:
            buckets["unknown"].append(c)
    return buckets
