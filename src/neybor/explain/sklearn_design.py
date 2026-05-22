"""Helpers for explainability on fitted sklearn-style classification pipelines."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def classifier_step(pipeline: Any) -> Any:
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Expected an sklearn Pipeline with named_steps.")
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        raise ValueError("Pipeline missing 'clf' step.")
    return clf


def preprocessor_step(pipeline: Any) -> Any:
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Expected an sklearn Pipeline with named_steps.")
    pre = pipeline.named_steps.get("preprocess")
    if pre is None:
        raise ValueError("Pipeline missing 'preprocess' step.")
    return pre


def design_matrix_after_preprocess(pipeline: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted ColumnTransformer only (same matrix the classifier sees)."""
    pre = preprocessor_step(pipeline)
    Xt = pre.transform(X)
    try:
        import scipy.sparse as sp

        if sp.issparse(Xt):
            Xt = Xt.toarray()
    except ImportError:
        pass

    if isinstance(Xt, pd.DataFrame):
        df = Xt.copy()
    else:
        arr = np.asarray(Xt)
        try:
            names = list(pre.get_feature_names_out())
        except Exception:
            names = [f"f{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=names, index=X.index)
    return df.astype(np.float64, copy=False)
