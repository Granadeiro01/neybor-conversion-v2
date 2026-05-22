"""Sklearn preprocessing pipeline.

Builds a ColumnTransformer that:
  - Median-imputes numeric columns
  - Mode-imputes categorical columns
  - One-hot encodes categoricals
  - Scales numerics (only used for logistic regression; tree models pass through)

This is a single shared pipeline used by all three model families. Imputation and
encoding choices are configurable so we can compare missingness strategies.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for the input frame.

    Boolean columns are treated as numeric since most models handle 0/1 directly.
    Datetime columns are excluded — they should have been turned into engineered
    features (submission_month, lead_time_days) by the time we reach this stage.
    """
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            continue  # silently drop raw datetime cols
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    scale_numeric: bool = True,
    onehot_handle_unknown: str = "ignore",
) -> ColumnTransformer:
    """Construct a ColumnTransformer.

    Parameters
    ----------
    scale_numeric : bool
        Set True for logistic regression, False for tree-based models.
    onehot_handle_unknown : str
        'ignore' tolerates unseen categories at test time (essential given the
        temporal split — new cities or unit types may appear in Feb-Apr 2026).
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown=onehot_handle_unknown,
            sparse_output=False,
            min_frequency=2,  # collapse very rare categories
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
