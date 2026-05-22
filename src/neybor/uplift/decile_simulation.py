"""Operational uplift simulation (thesis Section 5.4 / RQ4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def decile_uplift_table(y_true, y_score, *, n_deciles: int = 10) -> pd.DataFrame:
    """Compare model-ranked conversion capture against a uniform baseline."""
    df = pd.DataFrame({
        "target": np.asarray(y_true).astype(int),
        "score": np.asarray(y_score).astype(float),
    }).sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["decile"] = pd.qcut(df["rank"], q=n_deciles, labels=False, duplicates="drop") + 1

    total_conversions = max(int(df["target"].sum()), 1)
    total_rows = len(df)
    rows = []
    cumulative_rows = 0
    cumulative_conversions = 0
    for decile, group in df.groupby("decile", sort=True):
        cumulative_rows += len(group)
        cumulative_conversions += int(group["target"].sum())
        contact_rate = cumulative_rows / total_rows
        captured_rate = cumulative_conversions / total_conversions
        baseline_capture = contact_rate
        rows.append({
            "decile": int(decile),
            "n": len(group),
            "mean_score": float(group["score"].mean()),
            "conversions": int(group["target"].sum()),
            "conversion_rate": float(group["target"].mean()),
            "cumulative_contact_rate": contact_rate,
            "cumulative_conversion_capture": captured_rate,
            "baseline_capture": baseline_capture,
            "uplift_vs_uniform": captured_rate - baseline_capture,
        })
    return pd.DataFrame(rows)


def save_decile_uplift_artifacts(
    y_true,
    y_score,
    *,
    output_dir: Path,
    prefix: str = "uplift",
) -> tuple[Path, Path]:
    """Write top-decile uplift table and chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table = decile_uplift_table(y_true, y_score)
    table_path = output_dir / f"{prefix}_deciles.csv"
    figure_path = output_dir / f"{prefix}_capture_curve.png"
    table.to_csv(table_path, index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        table["cumulative_contact_rate"],
        table["cumulative_conversion_capture"],
        marker="o",
        label="Model ranking",
    )
    ax.plot(
        table["cumulative_contact_rate"],
        table["baseline_capture"],
        linestyle="--",
        label="Uniform baseline",
    )
    ax.set_xlabel("Share of leads contacted")
    ax.set_ylabel("Share of conversions captured")
    ax.set_title("Operational Uplift Simulation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return table_path, figure_path
