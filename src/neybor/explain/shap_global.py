"""Global SHAP explanations (thesis Section 5.3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neybor.explain.sklearn_design import classifier_step, design_matrix_after_preprocess


def _sample_frame(X: pd.DataFrame, max_samples: int, random_state: int) -> pd.DataFrame:
    if len(X) <= max_samples:
        return X.copy()
    return X.sample(n=max_samples, random_state=random_state)


def compute_global_shap(
    model,
    X: pd.DataFrame,
    *,
    max_samples: int = 200,
    random_state: int = 42,
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """Compute model-agnostic SHAP values on the fitted preprocessor output (numeric)."""
    import shap

    X_sample_raw = _sample_frame(X, max_samples=max_samples, random_state=random_state)
    X_design = design_matrix_after_preprocess(model, X_sample_raw)
    clf = classifier_step(model)

    def predict_fn(data):
        X_in = pd.DataFrame(
            np.asarray(data, dtype=np.float64),
            columns=X_design.columns,
        )
        return clf.predict_proba(X_in)[:, 1]

    masker = shap.maskers.Independent(X_design)
    explainer = shap.Explainer(predict_fn, masker)
    shap_values = explainer(X_design)
    values = np.asarray(shap_values.values)
    importance = pd.DataFrame({
        "feature": X_design.columns,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    return shap_values, X_design, importance


def save_global_shap_artifacts(
    model,
    X: pd.DataFrame,
    *,
    output_dir: Path,
    prefix: str = "shap_global",
    max_samples: int = 200,
    random_state: int = 42,
) -> tuple[Path, Path, Path]:
    """Write global SHAP importance CSV and thesis-ready plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shap_values, _x_sample, importance = compute_global_shap(
        model,
        X,
        max_samples=max_samples,
        random_state=random_state,
    )
    importance_path = output_dir / f"{prefix}_importance.csv"
    bar_path = output_dir / f"{prefix}_bar.png"
    beeswarm_path = output_dir / f"{prefix}_beeswarm.png"
    importance.to_csv(importance_path, index=False)

    import matplotlib.pyplot as plt
    import shap

    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()
    return importance_path, bar_path, beeswarm_path
