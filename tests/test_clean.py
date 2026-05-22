"""Tests for the application cleaning and model-ready export contract."""
from __future__ import annotations

import pandas as pd

from neybor.data.clean import clean_applications, model_ready_frame, select_model_feature_columns


def _applications_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": ["a1", "a2", "a3", "a4"],
            "CreatedDate": pd.to_datetime(
                ["2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04"], utc=True
            ),
            "dshift__Status__c": ["Completed", "Rejected", "Submitted", "Unreachable"],
            "Rejected_Lost_Reason__c": [None, "Price", None, None],
            "Monthly_Budget__c": ["€750 - €850", "€850 - €950", "€950 - €1050", None],
            "Gender__c": ["Female", "Male", "Female", "Male"],
            "dshift__Start_Date__c": pd.to_datetime(
                ["2025-08-01", "2025-08-15", "2025-09-01", "2025-09-15"], utc=True
            ),
            "dshift__Closed_Date_Time__c": pd.to_datetime(
                ["2025-08-02", "2025-08-20", None, None], utc=True
            ),
            "Has_Contract_Signed__c": [True, False, False, False],
            "dshift__Email__c": [
                "one@example.com",
                "two@example.com",
                "three@example.com",
                "four@example.com",
            ],
        }
    )


def test_clean_applications_uses_completed_status_as_target():
    cleaned = clean_applications(_applications_df())

    assert cleaned["dshift__Status__c"].tolist() == ["Completed", "Rejected", "Unreachable"]
    assert cleaned["target"].tolist() == [1, 0, 0]
    assert cleaned["is_unreachable_sensitivity"].tolist() == [False, False, True]


def test_model_feature_selection_drops_leaks_identifiers_and_raw_datetimes():
    cleaned = clean_applications(_applications_df())
    feature_cols = select_model_feature_columns(cleaned)

    assert "Monthly_Budget__c" in feature_cols
    assert "target" not in feature_cols
    assert "dshift__Status__c" not in feature_cols
    assert "dshift__Closed_Date_Time__c" not in feature_cols
    assert "Has_Contract_Signed__c" not in feature_cols
    assert "dshift__Email__c" not in feature_cols
    assert "Gender__c" not in feature_cols
    assert "Gender__c_was_missing" not in feature_cols
    assert "Id" not in feature_cols
    assert "CreatedDate" not in feature_cols
    assert "dshift__Start_Date__c" not in feature_cols


def test_model_ready_frame_contains_only_safe_features_and_target():
    cleaned = clean_applications(_applications_df())
    model_ready = model_ready_frame(cleaned)

    assert list(model_ready.columns) == ["Monthly_Budget__c", "target"]
    assert model_ready["target"].tolist() == [1, 0, 0]


def test_sensitive_features_require_explicit_opt_in():
    cleaned = clean_applications(_applications_df())

    feature_cols = select_model_feature_columns(cleaned, include_sensitive=True)

    assert "Gender__c" in feature_cols
