"""Tests for the tenant-solvency densification join."""
from __future__ import annotations

import pandas as pd
import pytest

from neybor.data.solvency import (
    HEADLINE_FEATURES,
    drop_missing_created_date,
    join_solvency,
)


def _solvency_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "salesforce_application_id": ["app1", "app2", "app3"],
        "tenant_professional_solvency_signal": ["Studies", "CDI", None],
        "tenant_age": [21.0, 33.0, None],
        "tenant_nationality": ["French", None, "Belgian"],
        "age_at_application_created": [21.0, 33.0, None],
        "monthly_budget_range": ["<€750", "€950 - €1050", None],
        "monthly_budget_range_source": ["x", "y", None],
        "length_of_stay": ["12 months +", "<3 months", None],
        "length_of_stay_source": ["x", "y", None],
    })


def _applications_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "Id": ["app1", "app2", "app3", "app4"],
        "CreatedDate": pd.to_datetime(
            ["2025-07-01", "2025-08-01", None, "2025-09-01"], utc=True,
        ),
        "Professional_Situation__c": [None, "Internship", None, "CDI"],
        "Nationality__c": [None, None, None, "Belgian"],
        "Monthly_Budget__c": [None, "€750 - €850", None, None],
    })


class TestJoinSolvency:
    def test_row_count_preserved(self):
        apps = _applications_fixture()
        joined = join_solvency(apps, _solvency_fixture())
        assert len(joined) == len(apps)

    def test_overrides_fill_target_when_solvency_has_value(self):
        joined = join_solvency(_applications_fixture(), _solvency_fixture())
        target = joined.set_index("Id")
        # app1: raw was NaN, solvency provides "Studies"
        assert target.loc["app1", "Professional_Situation__c"] == "Studies"
        # app2: raw was Internship, solvency provides CDI; solvency wins
        assert target.loc["app2", "Professional_Situation__c"] == "CDI"
        # app3: solvency value is null, raw was NaN, stays NaN
        assert pd.isna(target.loc["app3", "Professional_Situation__c"])
        # app4: not in solvency, raw value is preserved
        assert target.loc["app4", "Professional_Situation__c"] == "CDI"

    def test_nationality_override(self):
        joined = join_solvency(_applications_fixture(), _solvency_fixture())
        target = joined.set_index("Id")
        assert target.loc["app1", "Nationality__c"] == "French"
        assert target.loc["app3", "Nationality__c"] == "Belgian"
        assert target.loc["app4", "Nationality__c"] == "Belgian"

    def test_added_columns_present(self):
        joined = join_solvency(_applications_fixture(), _solvency_fixture())
        for col in ("Monthly_Budget", "length_of_stay", "tenant_age_at_application_created"):
            assert col in joined.columns

    def test_raw_picklist_columns_preserved(self):
        """Monthly_Budget__c should not be touched by the solvency join."""
        joined = join_solvency(_applications_fixture(), _solvency_fixture())
        target = joined.set_index("Id")
        assert target.loc["app2", "Monthly_Budget__c"] == "€750 - €850"

    def test_join_key_dropped(self):
        joined = join_solvency(_applications_fixture(), _solvency_fixture())
        assert "salesforce_application_id" not in joined.columns

    def test_duplicate_key_in_solvency_raises(self):
        sol = pd.concat([_solvency_fixture(), _solvency_fixture().head(1)], ignore_index=True)
        # join_solvency drops duplicates internally, so it should NOT raise
        # but should still preserve apps row count.
        joined = join_solvency(_applications_fixture(), sol)
        assert len(joined) == len(_applications_fixture())

    def test_missing_join_key_raises(self):
        with pytest.raises(KeyError, match="Id"):
            join_solvency(pd.DataFrame({"x": [1]}), _solvency_fixture())
        with pytest.raises(KeyError, match="salesforce_application_id"):
            join_solvency(_applications_fixture(), pd.DataFrame({"y": [1]}))


class TestDropMissingCreatedDate:
    def test_drops_only_null_rows(self):
        df = _applications_fixture()
        out = drop_missing_created_date(df)
        assert len(out) == 3
        assert out["CreatedDate"].notna().all()

    def test_no_op_when_column_missing(self):
        df = pd.DataFrame({"Id": ["a"]})
        out = drop_missing_created_date(df)
        assert len(out) == 1


class TestHeadlineFeatures:
    def test_count_matches_headline_summary(self):
        # The current run uses 20 model features: the solvency-era set with raw
        # Nationality and absolute age replaced by coarse nationality_region and
        # age_band, plus the submission_time_index drift control.
        assert len(HEADLINE_FEATURES) == 20

    def test_includes_solvency_added_features(self):
        for col in (
            "Monthly_Budget",
            "length_of_stay",
        ):
            assert col in HEADLINE_FEATURES

    def test_uses_coarse_demographic_and_drift_features(self):
        # Raw Nationality / absolute age are replaced by coarse encodings, and a
        # continuous time index controls for the capture-regime shift.
        for col in ("nationality_region", "age_band", "submission_time_index"):
            assert col in HEADLINE_FEATURES
        for col in ("Nationality__c", "tenant_age_at_application_created"):
            assert col not in HEADLINE_FEATURES

    def test_excludes_post_join_metadata(self):
        for col in (
            "salesforce_application_id",
            "delta_hours_created_to_scheduled_call",
            "scheduled_call_date_time",
            "created_date",
            "target",
        ):
            assert col not in HEADLINE_FEATURES


class TestFillRateUplift:
    """The solvency join's whole point: densify sparse Salesforce columns.

    These thresholds mirror the table the user shared:
        Professional_Situation__c   ~2%   →   ~98%
        Nationality__c              ~3%   →   ~38%
        tenant_age_at_application_created   absent in raw   →   ~82%
    """

    def test_professional_situation_uplift(self):
        from neybor.config import SNAPSHOT_DIR
        from neybor.io import load_applications, load_tenant_solvency
        from neybor.data import clean_applications

        apps = load_applications(SNAPSHOT_DIR)
        sol = load_tenant_solvency(SNAPSHOT_DIR)
        cleaned = clean_applications(apps)
        before = cleaned["Professional_Situation__c"].notna().mean()
        after = join_solvency(cleaned, sol)["Professional_Situation__c"].notna().mean()
        assert before < 0.10, f"raw fill rate unexpectedly high: {before:.3f}"
        assert after > 0.90, f"post-join fill rate too low: {after:.3f}"

    def test_tenant_age_present_after_join(self):
        from neybor.config import SNAPSHOT_DIR
        from neybor.io import load_applications, load_tenant_solvency
        from neybor.data import clean_applications

        apps = load_applications(SNAPSHOT_DIR)
        sol = load_tenant_solvency(SNAPSHOT_DIR)
        cleaned = clean_applications(apps)
        joined = join_solvency(cleaned, sol)
        fill = joined["tenant_age_at_application_created"].notna().mean()
        assert fill > 0.70, f"tenant_age fill rate too low: {fill:.3f}"
