"""Synthetic end-to-end smoke test for the thesis pipeline stages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from neybor.data import clean_applications, join_enrichment, primary_sample, select_model_feature_columns
from neybor.features import add_all_engineered_features, add_missingness_indicators, filter_to_allowed
from neybor.io import load_all, verify_manifest, write_manifest
from neybor.models import evaluate, select_model_and_threshold, temporal_split


class _ConstantModel:
    def __init__(self, positive_rate: float):
        self.positive_rate = positive_rate

    def predict_proba(self, X):
        scores = np.full(len(X), self.positive_rate)
        return np.column_stack([1 - scores, scores])


def _factory(_X_train, y_train):
    return _ConstantModel(float(y_train.mean()))


def _write_snapshot(snapshot_dir):
    applications = pd.DataFrame({
        "Id": [f"app{i}" for i in range(12)],
        "CreatedDate": pd.date_range("2025-10-01", periods=12, freq="20D", tz="UTC"),
        "dshift__Status__c": ["Completed", "Rejected"] * 6,
        "Rejected_Lost_Reason__c": [None, "Price"] * 6,
        "dshift__Start_Date__c": pd.date_range("2025-11-01", periods=12, freq="20D", tz="UTC"),
        "Monthly_Budget__c": ["€750 - €850", "€850 - €950"] * 6,
        "Length_of_Stay__c": ["12 months +", "6-12 months"] * 6,
        "dshift__MER_Property_Group__c": ["Ixelles", "Forest"] * 6,
        "dshift__Property__c": ["prop1", "prop2"] * 6,
    })
    properties = pd.DataFrame({
        "Id": ["prop1", "prop2"],
        "dshift__City__c": ["Brussels", "Brussels"],
        "dshift__Country_Code__c": ["BE", "BE"],
        "dshift__Postal_Code__c": ["1050", "1190"],
        "Type__c": ["Coliving", "Apartment"],
        "dshift__MER_Property_Group__c": ["Ixelles", "Forest"],
    })
    units = pd.DataFrame({
        "Id": ["unit1", "unit2"],
        "dshift__MER_Current_Unit_Price__c": [900.0, 950.0],
    })
    contracts = pd.DataFrame({"Id": ["contract1"]})
    tenant_solvency = pd.DataFrame({
        "salesforce_application_id": [f"app{i}" for i in range(12)],
        "tenant_professional_solvency_signal": ["Studies", "CDI"] * 6,
        "tenant_age": [22, 30] * 6,
        "tenant_nationality": ["French", "Belgian"] * 6,
        "age_at_application_created": [22, 30] * 6,
        "monthly_budget_range": ["€750 - €850", "€850 - €950"] * 6,
        "monthly_budget_range_source": ["x"] * 12,
        "length_of_stay": ["12 months +", "6-12 months"] * 6,
        "length_of_stay_source": ["x"] * 12,
    })

    applications.to_csv(snapshot_dir / "applications-primary-db.csv", index=False)
    contracts.to_csv(snapshot_dir / "contracts.csv", index=False)
    properties.to_csv(snapshot_dir / "properties.csv", index=False)
    units.to_csv(snapshot_dir / "units.csv", index=False)
    tenant_solvency.to_csv(
        snapshot_dir / "tenant_professional_solvency_feature_imputed_single.csv",
        index=False,
    )


def test_synthetic_pipeline_smoke(tmp_path):
    _write_snapshot(tmp_path)
    write_manifest(tmp_path, comment="test")
    ok, problems = verify_manifest(tmp_path)
    assert ok, problems

    raw = load_all(tmp_path)
    cleaned = clean_applications(raw["applications"])
    enriched = join_enrichment(cleaned, raw["properties"], raw["units"])
    sample = add_all_engineered_features(primary_sample(enriched))
    sample, _ = add_missingness_indicators(sample, columns=filter_to_allowed(list(sample.columns)))

    train_df, test_df = temporal_split(sample)
    feature_cols = select_model_feature_columns(train_df)
    selection = select_model_and_threshold(
        train_df,
        feature_cols,
        {"constant": _factory},
        n_splits=2,
    )
    y_score = selection.model.predict_proba(test_df[feature_cols])[:, 1]
    results = evaluate(test_df["target"], y_score, threshold=selection.threshold, n_resamples=10)

    assert selection.model_name == "constant"
    assert "pr_auc" in results
    assert any(key.startswith("f1_at_") for key in results)
