"""Leakage tests — the most important tests in the project.

Field names match the real Neybor Salesforce schema (verified 2026-04-27).
"""
from __future__ import annotations

import pytest

from neybor.features.leakage import (
    ALLOWED_AT_SUBMISSION,
    DROPPED_NEUTRAL_FIELDS,
    FORBIDDEN,
    FORBIDDEN_OUTCOME_LEAK,
    FORBIDDEN_TEMPORAL_LEAK,
    SENSITIVE_FIELDS,
    LeakageError,
    assert_safe,
    classify_columns,
    filter_to_allowed,
)


class TestRegistryConsistency:
    def test_no_overlap_allowed_forbidden(self):
        overlap = ALLOWED_AT_SUBMISSION & FORBIDDEN
        assert not overlap, f"Fields in both ALLOWED and FORBIDDEN: {overlap}"

    def test_outcome_and_temporal_disjoint(self):
        overlap = FORBIDDEN_OUTCOME_LEAK & FORBIDDEN_TEMPORAL_LEAK
        assert len(overlap) == 0, f"Categories overlap: {overlap}"


class TestAssertSafe:
    def test_passes_for_known_safe_features(self):
        safe = {"lead_time_days", "demand_pressure", "submission_month",
                "Monthly_Budget__c", "dshift__Source__c"}
        assert_safe(safe)

    def test_passes_for_missingness_indicators(self):
        with_indicators = {"lead_time_days", "Monthly_Budget__c_was_missing"}
        assert_safe(with_indicators)

    def test_rejects_forbidden_field_missingness_indicator(self):
        with pytest.raises(LeakageError, match="not in the allowed registry"):
            assert_safe({"Has_Contract_Signed__c_was_missing"})

    def test_rejects_outcome_leak(self):
        leaky = {"lead_time_days", "Has_Contract_Signed__c"}
        with pytest.raises(LeakageError, match="Has_Contract_Signed__c"):
            assert_safe(leaky)

    def test_rejects_temporal_leak(self):
        leaky = {"lead_time_days", "dshift__Closed_Date_Time__c"}
        with pytest.raises(LeakageError, match="dshift__Closed_Date_Time__c"):
            assert_safe(leaky)

    def test_rejects_status_field(self):
        """The target itself must never appear as a feature."""
        with pytest.raises(LeakageError):
            assert_safe({"dshift__Status__c"})

    def test_rejects_rejected_lost_reason(self):
        with pytest.raises(LeakageError, match="Rejected_Lost_Reason__c"):
            assert_safe({"Rejected_Lost_Reason__c"})

    def test_rejects_ura_booking_stage(self):
        with pytest.raises(LeakageError, match="dshift__URA_Booking_Stage__c"):
            assert_safe({"dshift__URA_Booking_Stage__c"})

    def test_rejects_first_payment_amount(self):
        with pytest.raises(LeakageError, match="First_Payment_Amount__c"):
            assert_safe({"First_Payment_Amount__c"})

    def test_rejects_unknown_field(self):
        unknown = {"some_random_new_field_we_havent_documented"}
        with pytest.raises(LeakageError, match="not in the allowed registry"):
            assert_safe(unknown)


class TestFilterToAllowed:
    def test_keeps_only_allowed(self):
        cols = [
            "lead_time_days",
            "Monthly_Budget__c",
            "Has_Contract_Signed__c",
            "dshift__Closed_Date_Time__c",
            "lead_time_days_was_missing",
            "Has_Contract_Signed__c_was_missing",
            "completely_unknown_column",
        ]
        kept = filter_to_allowed(cols)
        assert "lead_time_days" in kept
        assert "Monthly_Budget__c" in kept
        assert "lead_time_days_was_missing" in kept
        assert "Has_Contract_Signed__c_was_missing" not in kept
        assert "Has_Contract_Signed__c" not in kept
        assert "dshift__Closed_Date_Time__c" not in kept
        assert "completely_unknown_column" not in kept


class TestClassifyColumns:
    def test_buckets_correctly(self):
        cols = [
            "lead_time_days",                # allowed (engineered)
            "Monthly_Budget__c",              # allowed (native)
            "Has_Contract_Signed__c",         # outcome leak
            "dshift__Closed_Date_Time__c",    # temporal leak
            "Name",                           # dropped neutral
            "some_unknown_column",            # unknown
        ]
        buckets = classify_columns(cols)
        assert "lead_time_days" in buckets["allowed"]
        assert "Monthly_Budget__c" in buckets["allowed"]
        assert "Has_Contract_Signed__c" in buckets["outcome_leak"]
        assert "dshift__Closed_Date_Time__c" in buckets["temporal_leak"]
        assert "Name" in buckets["dropped_neutral"]
        assert "some_unknown_column" in buckets["unknown"]


class TestRegistryContent:
    """Pin specific fields named in the thesis."""

    @pytest.mark.parametrize("field", [
        "Has_Contract_Signed__c",
        "Has_First_Payment_Paid__c",
        "Has_Active_Contracts__c",
        "Has_Draft_Contracts__c",
        "Has_Cancelled_Expired_Contracts__c",
        "Rejected_Lost_Reason__c",
        "dshift__Is_Closed__c",
        "dshift__Status__c",
        "First_Payment_Amount__c",
        "First_Payment_Date__c",
    ])
    def test_named_outcome_leaks_in_registry(self, field):
        assert field in FORBIDDEN_OUTCOME_LEAK, (
            f"{field} should be in FORBIDDEN_OUTCOME_LEAK."
        )

    @pytest.mark.parametrize("field", [
        "dshift__Closed_Date_Time__c",
        "Status_Last_Updated__c",
        "Days_in_Current_Status__c",
        "Proposal_Sent_Datec__c",
        "Time_in_Days_Since_New_Lead__c",
        "dshift__URA_Booking_Stage__c",
        "dshift__URA_Call_Status__c",
    ])
    def test_named_temporal_leaks_in_registry(self, field):
        assert field in FORBIDDEN_TEMPORAL_LEAK, (
            f"{field} should be in FORBIDDEN_TEMPORAL_LEAK."
        )


class TestSolvencyJoinRegistry:
    """The solvency join introduces extra columns into the modelling frame.

    These tests pin the registry decisions made when wiring that join in:
      - the imputed picklists and the tenant_age column are allowed features;
      - the solvency-side source columns are dropped_neutral, not features;
      - the new tenant_age column is treated as sensitive (gated by
        --include-sensitive at the pipeline level).
    """

    @pytest.mark.parametrize("field", [
        "Monthly_Budget",
        "length_of_stay",
        "tenant_age_at_application_created",
    ])
    def test_solvency_features_allowed(self, field):
        assert field in ALLOWED_AT_SUBMISSION
        assert_safe({field, "lead_time_days"})  # paired with a known-safe field

    def test_tenant_age_is_sensitive(self):
        assert "tenant_age_at_application_created" in SENSITIVE_FIELDS

    @pytest.mark.parametrize("field", [
        "salesforce_application_id",
        "tenant_professional_solvency_signal",
        "tenant_age",
        "tenant_nationality",
        "monthly_budget_range",
        "monthly_budget_range_source",
        "length_of_stay_source",
    ])
    def test_solvency_source_columns_dropped_neutral(self, field):
        assert field in DROPPED_NEUTRAL_FIELDS
        assert field not in ALLOWED_AT_SUBMISSION
        assert field not in FORBIDDEN
        # filter_to_allowed drops them so they never reach the model.
        assert filter_to_allowed([field, "lead_time_days"]) == ["lead_time_days"]
