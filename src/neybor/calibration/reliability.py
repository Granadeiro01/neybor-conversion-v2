"""Calibration diagnostics (thesis Section 5.4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    y_true,
    y_score,
    *,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Compute ECE and return the per-bin reliability table."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_score, bins, right=True) - 1, 0, n_bins - 1)

    rows = []
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            rows.append({
                "bin": bin_id,
                "n": 0,
                "mean_predicted": np.nan,
                "observed_rate": np.nan,
                "abs_gap": np.nan,
            })
            continue
        mean_predicted = float(y_score[mask].mean())
        observed_rate = float(y_true[mask].mean())
        abs_gap = abs(mean_predicted - observed_rate)
        weight = mask.mean()
        ece += weight * abs_gap
        rows.append({
            "bin": bin_id,
            "n": int(mask.sum()),
            "mean_predicted": mean_predicted,
            "observed_rate": observed_rate,
            "abs_gap": abs_gap,
        })
    return float(ece), pd.DataFrame(rows)


def calibration_report(
    y_true,
    y_score,
    *,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary metrics and reliability bins."""
    ece, bins = expected_calibration_error(y_true, y_score, n_bins=n_bins)
    summary = pd.DataFrame([
        {"metric": "brier_score", "value": float(brier_score_loss(y_true, y_score))},
        {"metric": "expected_calibration_error", "value": ece},
    ])
    return summary, bins


def save_calibration_artifacts(
    y_true,
    y_score,
    *,
    output_dir: Path,
    prefix: str = "calibration",
    n_bins: int = 10,
) -> tuple[Path, Path, Path]:
    """Write calibration tables and a reliability plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, bins = calibration_report(y_true, y_score, n_bins=n_bins)
    summary_path = output_dir / f"{prefix}_summary.csv"
    bins_path = output_dir / f"{prefix}_bins.csv"
    figure_path = output_dir / f"{prefix}_reliability.png"
    summary.to_csv(summary_path, index=False)
    bins.to_csv(bins_path, index=False)

    import matplotlib.pyplot as plt

    fraction_positive, mean_predicted = calibration_curve(y_true, y_score, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(mean_predicted, fraction_positive, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed conversion rate")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return summary_path, bins_path, figure_path
