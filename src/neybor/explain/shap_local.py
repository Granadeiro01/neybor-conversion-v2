"""Local SHAP explanations (thesis Section 5.3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neybor.explain.sklearn_design import classifier_step, design_matrix_after_preprocess


def compute_local_driver_table(shap_values, X: pd.DataFrame, *, top_n: int = 3) -> pd.DataFrame:
    """Return top positive/negative drivers for each explained record."""
    values = np.asarray(shap_values.values)
    rows = []
    for row_pos, (_, feature_row) in enumerate(X.iterrows()):
        order = np.argsort(np.abs(values[row_pos]))[::-1][:top_n]
        for rank, feature_pos in enumerate(order, start=1):
            feature = X.columns[feature_pos]
            rows.append({
                "row_index": row_pos,
                "rank": rank,
                "feature": feature,
                "feature_value": feature_row[feature],
                "shap_value": float(values[row_pos, feature_pos]),
            })
    return pd.DataFrame(rows)


def save_local_shap_artifacts(
    model,
    X: pd.DataFrame,
    *,
    output_dir: Path,
    prefix: str = "shap_local",
    n_records: int = 5,
    top_n: int = 3,
) -> tuple[Path, list[Path]]:
    """Write local driver table and waterfall plots for selected records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import shap

    X_explain_raw = X.head(n_records).copy()
    X_design = design_matrix_after_preprocess(model, X_explain_raw)
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

    drivers = compute_local_driver_table(shap_values, X_design, top_n=top_n)
    drivers_path = output_dir / f"{prefix}_drivers.csv"
    drivers.to_csv(drivers_path, index=False)

    figure_paths: list[Path] = []
    for row_pos in range(len(X_design)):
        figure_path = output_dir / f"{prefix}_waterfall_{row_pos}.png"
        shap.plots.waterfall(shap_values[row_pos], max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close()
        figure_paths.append(figure_path)
    return drivers_path, figure_paths
