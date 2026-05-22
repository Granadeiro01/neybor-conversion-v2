"""Tests for the temporal split — guarantees no future leak in evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neybor.models.splits import temporal_split


def _make_df(n: int, start: str = "2025-06-01", freq: str = "1D") -> pd.DataFrame:
    """Build a synthetic application table with a CreatedDate column.

    Default freq=1D and n=400 spans roughly 13 months, comfortably crossing the
    2026-01-31 / 2026-02-01 train/test cutoff used by the production config.
    """
    dates = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Id": [f"a{i:04d}" for i in range(n)],
        "CreatedDate": dates,
        "target": rng.integers(0, 2, size=n),
    })


class TestTemporalSplit:
    def test_basic_split_shape(self):
        df = _make_df(400)
        train, test = temporal_split(df)
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) <= len(df)

    def test_no_overlap(self):
        """The defining property: no test date may be ≤ any train date."""
        df = _make_df(400)
        train, test = temporal_split(df)
        assert train["CreatedDate"].max() < test["CreatedDate"].min()

    def test_train_dates_before_cutoff(self):
        df = _make_df(400)
        train, _ = temporal_split(df, train_end="2026-01-31")
        assert (train["CreatedDate"] < pd.Timestamp("2026-02-01", tz="UTC")).all()

    def test_test_dates_after_cutoff(self):
        df = _make_df(400)
        _, test = temporal_split(df, test_start="2026-02-01")
        assert (test["CreatedDate"] >= pd.Timestamp("2026-02-01", tz="UTC")).all()

    def test_raises_on_empty_train(self):
        df = _make_df(50, start="2026-03-01")  # all after cutoff
        with pytest.raises(ValueError, match="Training set is empty"):
            temporal_split(df)

    def test_raises_on_empty_test(self):
        df = _make_df(50, start="2025-06-01", freq="1h")  # all before cutoff
        with pytest.raises(ValueError, match="Test set is empty"):
            temporal_split(df)

    def test_raises_on_missing_date_col(self):
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(KeyError, match="CreatedDate"):
            temporal_split(df)

    def test_index_is_reset(self):
        df = _make_df(400)
        train, test = temporal_split(df)
        # Reset index means range from 0 to len-1
        assert (train.index == range(len(train))).all()
        assert (test.index == range(len(test))).all()

    def test_class_balance_logged(self, caplog):
        """Just exercise the logging branch — no assertion needed beyond the call."""
        df = _make_df(400)
        import logging
        with caplog.at_level(logging.INFO, logger="neybor.models.splits"):
            temporal_split(df)
        assert any("Class balance" in rec.message for rec in caplog.records)
