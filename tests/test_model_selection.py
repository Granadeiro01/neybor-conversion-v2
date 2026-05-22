"""Tests for training-period model and threshold selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from neybor.models.selection import select_model_and_threshold, temporal_cv_splits


class _MeanScoreModel:
    def __init__(self, positive_rate: float):
        self.positive_rate = positive_rate

    def predict_proba(self, X):
        scores = np.full(len(X), self.positive_rate)
        return np.column_stack([1 - scores, scores])


def _factory(X_train, y_train):
    return _MeanScoreModel(float(y_train.mean()))


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "CreatedDate": pd.date_range("2025-01-01", periods=12, freq="D", tz="UTC"),
        "feature": np.arange(12),
        "target": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    })


def test_temporal_cv_splits_are_chronological():
    df = _df()
    splits = temporal_cv_splits(df, n_splits=3)

    assert len(splits) == 3
    for train_idx, validation_idx in splits:
        assert df.loc[train_idx, "CreatedDate"].max() < df.loc[validation_idx, "CreatedDate"].min()


def test_select_model_and_threshold_uses_training_folds_only():
    df = _df()
    result = select_model_and_threshold(
        df,
        ["feature"],
        {"dummy": _factory},
        n_splits=3,
    )

    assert result.model_name == "dummy"
    assert 0.0 <= result.threshold <= 1.0
    assert set(result.cv_results["fold"]) == {1, 2, 3}
    assert len(result.cv_results) == 3
