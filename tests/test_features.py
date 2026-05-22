"""Tests for engineered features.

The most important test is that demand_pressure is computed without cross-record
leakage (thesis Section 4.3.1) — a naive groupby().count() would inflate it by
including future records.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neybor.features.engineered import (
    add_age_at_application_created,
    add_budget_unit_mismatch,
    add_demand_pressure,
    add_lead_time_days,
    add_submission_temporal,
    parse_budget_bracket,
    parse_length_of_stay,
)


def _build_app_df(rows):
    df = pd.DataFrame(rows)
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"], utc=True)
    if "dshift__Start_Date__c" in df.columns:
        df["dshift__Start_Date__c"] = pd.to_datetime(df["dshift__Start_Date__c"], utc=True)
    return df


class TestLeadTime:
    def test_basic(self):
        df = _build_app_df([
            {"CreatedDate": "2026-01-01", "dshift__Start_Date__c": "2026-01-15"},
            {"CreatedDate": "2026-01-10", "dshift__Start_Date__c": "2026-02-10"},
        ])
        df = add_lead_time_days(df)
        assert df["lead_time_days"].iloc[0] == pytest.approx(14.0)
        assert df["lead_time_days"].iloc[1] == pytest.approx(31.0)

    def test_skips_when_column_missing(self):
        df = _build_app_df([{"CreatedDate": "2026-01-01"}])
        df = add_lead_time_days(df)
        assert "lead_time_days" not in df.columns


class TestSubmissionTemporal:
    def test_dayofweek_and_month(self):
        df = _build_app_df([
            {"CreatedDate": "2026-01-05T10:00:00", "Id": "a"},  # Monday
            {"CreatedDate": "2026-04-15T10:00:00", "Id": "b"},  # Wednesday
        ])
        df = add_submission_temporal(df)
        assert df["submission_day_of_week"].iloc[0] == 0
        assert df["submission_day_of_week"].iloc[1] == 2
        assert df["submission_month"].iloc[0] == 1
        assert df["submission_month"].iloc[1] == 4


class TestAgeAtApplicationCreated:
    def test_uses_application_birth_date_before_account_fallbacks(self):
        df = _build_app_df([
            {
                "CreatedDate": "2026-04-01",
                "Application.Date_of_Birth__c": "2000-04-02",
                "Account.Date_of_Birth__pc": "1990-01-01",
                "Age__c": 99,
            },
            {
                "CreatedDate": "2026-04-01",
                "Application.Date_of_Birth__c": None,
                "Account.Date_of_Birth__pc": "2000-04-01",
            },
            {
                "CreatedDate": "2026-04-01",
                "Application.Date_of_Birth__c": None,
                "Account.Date_of_Birth__pc": None,
                "Account.PersonBirthdate": "2000-04-02",
            },
        ])
        df = add_age_at_application_created(df)
        assert df["age_at_application_created"].tolist() == [25.0, 26.0, 25.0]

    def test_supports_unprefixed_application_birth_date(self):
        df = _build_app_df([
            {"CreatedDate": "2026-04-01", "Date_of_Birth__c": "2000-04-01"},
        ])
        df = add_age_at_application_created(df)
        assert df["age_at_application_created"].iloc[0] == 26.0

    def test_raw_age_fills_only_when_birth_dates_are_missing(self):
        df = _build_app_df([
            {"CreatedDate": "2026-04-01", "Date_of_Birth__c": None, "Age__c": "42"},
            {"CreatedDate": "2026-04-01", "Date_of_Birth__c": None, "Age__c": 200},
        ])
        df = add_age_at_application_created(df)
        assert df["age_at_application_created"].iloc[0] == 42.0
        assert pd.isna(df["age_at_application_created"].iloc[1])

    def test_existing_age_at_application_created_is_decisive(self):
        df = _build_app_df([
            {
                "CreatedDate": "2026-04-01",
                "age_at_application_created": 31,
                "Date_of_Birth__c": "2000-04-01",
                "Age__c": 42,
            },
        ])
        df = add_age_at_application_created(df)
        assert df["age_at_application_created"].iloc[0] == 31.0

    def test_skips_when_no_age_source_columns(self):
        df = _build_app_df([{"CreatedDate": "2026-04-01"}])
        df = add_age_at_application_created(df)
        assert "age_at_application_created" not in df.columns


class TestDemandPressure:
    """Cross-record leakage prevention."""

    def test_no_future_leak(self):
        df = _build_app_df([
            {"Id": "a", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-01"},
            {"Id": "b", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-05"},
            {"Id": "c", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-10"},
            {"Id": "d", "dshift__MER_Property_Group__c": "Forest",  "CreatedDate": "2026-01-03"},
        ])
        df = add_demand_pressure(df)
        result = df.set_index("Id")["demand_pressure"].to_dict()
        assert result["a"] == 0
        assert result["b"] == 1
        assert result["c"] == 2
        assert result["d"] == 0

    def test_unsorted_input_yields_correct_result(self):
        df = _build_app_df([
            {"Id": "c", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-10"},
            {"Id": "a", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-01"},
            {"Id": "b", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-05"},
        ])
        df = add_demand_pressure(df)
        result = df.set_index("Id")["demand_pressure"].to_dict()
        assert result["a"] == 0
        assert result["b"] == 1
        assert result["c"] == 2

    def test_different_groups_independent(self):
        df = _build_app_df([
            {"Id": "a", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-01"},
            {"Id": "b", "dshift__MER_Property_Group__c": "Forest",  "CreatedDate": "2026-01-02"},
            {"Id": "c", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-03"},
            {"Id": "d", "dshift__MER_Property_Group__c": "Forest",  "CreatedDate": "2026-01-04"},
        ])
        df = add_demand_pressure(df)
        result = df.set_index("Id")["demand_pressure"].to_dict()
        assert result["a"] == 0
        assert result["b"] == 0
        assert result["c"] == 1
        assert result["d"] == 1

    def test_same_timestamp_records_do_not_count_each_other(self):
        df = _build_app_df([
            {"Id": "a", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-01"},
            {"Id": "b", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-01"},
            {"Id": "c", "dshift__MER_Property_Group__c": "Ixelles", "CreatedDate": "2026-01-02"},
        ])
        df = add_demand_pressure(df)
        result = df.set_index("Id")["demand_pressure"].to_dict()
        assert result["a"] == 0
        assert result["b"] == 0
        assert result["c"] == 2


class TestBudgetUnitMismatch:
    def test_with_group_median_unit_price(self):
        df = pd.DataFrame({
            "monthly_budget_midpoint": [800, 1500, 1200, np.nan],
            "group_median_unit_price": [1000, 1000, 1000, 1000],
        })
        df = add_budget_unit_mismatch(df)
        assert df["budget_unit_mismatch"].tolist() == [1, 0, 0, 0]

    def test_zero_when_no_asof_reference_price(self):
        df = pd.DataFrame({
            "monthly_budget_midpoint": [800, 1500, 1200],
            "Preferred_Unit_type__c": ["studio", "studio", "studio"],
        })
        df = add_budget_unit_mismatch(df)
        assert df["budget_unit_mismatch"].tolist() == [0, 0, 0]

    def test_zero_when_columns_missing(self):
        df = pd.DataFrame({"foo": [1, 2, 3]})
        df = add_budget_unit_mismatch(df)
        assert (df["budget_unit_mismatch"] == 0).all()


class TestParseBudgetBracket:
    def test_range_brackets(self):
        s = pd.Series(["€750 - €850", "€850 - €950", "€1250 - €1450"])
        result = parse_budget_bracket(s)
        assert result.tolist() == [800.0, 900.0, 1350.0]

    def test_open_ended_bracket(self):
        # The headline solvency pipeline maps "€2450+" → 2450 (lower bound), not
        # lower+250. parse_budget_bracket follows that convention.
        s = pd.Series(["€2450+", "€1850+"])
        result = parse_budget_bracket(s)
        assert result.tolist() == [2450.0, 1850.0]

    def test_below_cap_bracket(self):
        # "<€750" → 700 (X - 50), matching the headline pipeline's convention.
        s = pd.Series(["<€750"])
        result = parse_budget_bracket(s)
        assert result.iloc[0] == 700.0

    def test_nan_passthrough(self):
        s = pd.Series([None, "€750 - €850"])
        result = parse_budget_bracket(s)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 800.0

    def test_already_numeric(self):
        s = pd.Series([800.0, 900.0])
        result = parse_budget_bracket(s)
        assert result.tolist() == [800.0, 900.0]


class TestParseLengthOfStay:
    def test_known_values(self):
        s = pd.Series(["<3 months", "3-6 months", "6-12 months", "12 months +"])
        result = parse_length_of_stay(s)
        assert result.tolist() == [1.5, 4.5, 9.0, 13.0]

    def test_unknown_returns_nan(self):
        s = pd.Series(["forever"])
        result = parse_length_of_stay(s)
        assert pd.isna(result.iloc[0])

    def test_already_numeric(self):
        s = pd.Series([3.0, 6.0])
        result = parse_length_of_stay(s)
        assert result.tolist() == [3.0, 6.0]
