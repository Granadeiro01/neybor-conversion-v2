"""Supplementary thesis artifacts produced after run_pipeline.py.

Generates the three deliverables the advisor's feedback asked for that the core
pipeline does not emit on its own:

  1. ``model_train_vs_test.csv`` — per-model training-period (temporal-CV) ROC-AUC
     and PR-AUC alongside the hold-out ROC-AUC and PR-AUC. This reconciles the
     "training AUC vs Section 6.1 test AUC" discrepancy by showing both, for every
     candidate model, in one table.
  2. ``decision_tree.png`` / ``decision_tree_rules.txt`` — a shallow, human-readable
     glass-box tree (advisor's requested complement to SHAP).
  3. ``feature_summary_statistics.csv`` — summary statistics for every model input
     (numeric: n / mean / sd / min / median / max / missing%; categorical: n levels
     / modal level / missing%), which the data section currently lacks.

Run after the pipeline:
    PYTHONPATH=src python3 scripts/thesis_artifacts.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neybor.config import (
    PROCESSED_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    TRAIN_END_DATE,
    TEST_START_DATE,
)
from neybor.data.solvency import HEADLINE_FEATURES
from neybor.models import (
    temporal_cv_splits,
    train_decision_tree,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("thesis_artifacts")

MODEL_FACTORIES = {
    "logreg": train_logistic_regression,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
}


def _load_split():
    """Load the model-ready parity CSV and re-derive the temporal split.

    The parity CSV retains ``created_date`` (the parquet drops it), which the
    temporal-CV splitter needs. We rename it to ``CreatedDate`` so the existing
    split helper works unchanged.
    """
    csv = PROCESSED_DIR / "application_conversion_model_ready_with_solvency.csv"
    df = pd.read_csv(csv)
    df["CreatedDate"] = pd.to_datetime(df["created_date"], errors="coerce", utc=True)
    df = df[df["CreatedDate"].notna()].reset_index(drop=True)
    train_end = pd.Timestamp(TRAIN_END_DATE, tz="UTC") + pd.Timedelta(days=1)
    test_start = pd.Timestamp(TEST_START_DATE, tz="UTC")
    train = df[df["CreatedDate"] < train_end].reset_index(drop=True)
    test = df[df["CreatedDate"] >= test_start].reset_index(drop=True)
    feats = [c for c in HEADLINE_FEATURES if c in df.columns]
    return train, test, feats


def build_train_vs_test_table(train, test, feats) -> pd.DataFrame:
    """Training-period CV ROC/PR-AUC vs hold-out ROC/PR-AUC, per model."""
    rows = []
    splits = temporal_cv_splits(train)
    for name, factory in MODEL_FACTORIES.items():
        cv_roc, cv_pr = [], []
        for fit_idx, val_idx in splits:
            fold_tr, fold_val = train.loc[fit_idx], train.loc[val_idx]
            model = factory(fold_tr[feats], fold_tr["target"])
            score = model.predict_proba(fold_val[feats])[:, 1]
            if fold_val["target"].nunique() > 1:
                cv_roc.append(roc_auc_score(fold_val["target"], score))
                cv_pr.append(average_precision_score(fold_val["target"], score))
        # Fit on full training set, evaluate once on the hold-out.
        final = factory(train[feats], train["target"])
        test_score = final.predict_proba(test[feats])[:, 1]
        rows.append({
            "model": name,
            "train_cv_roc_auc": float(np.mean(cv_roc)) if cv_roc else float("nan"),
            "train_cv_pr_auc": float(np.mean(cv_pr)) if cv_pr else float("nan"),
            "holdout_roc_auc": float(roc_auc_score(test["target"], test_score)),
            "holdout_pr_auc": float(average_precision_score(test["target"], test_score)),
        })
    out = pd.DataFrame(rows).round(3)
    out.to_csv(TABLES_DIR / "model_train_vs_test.csv", index=False)
    log.info("Wrote model_train_vs_test.csv:\n%s", out.to_string(index=False))
    return out


def build_decision_tree_viz(train, feats) -> None:
    """Fit a shallow tree on the full training set and render it on one page."""
    pipe = train_decision_tree(train[feats], train["target"])
    pre = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]
    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:  # pragma: no cover
        feat_names = [f"f{i}" for i in range(clf.n_features_in_)]
    # Strip the ColumnTransformer prefixes ("num__", "cat__") for readability.
    feat_names = [n.split("__", 1)[-1] for n in feat_names]

    fig, ax = plt.subplots(figsize=(20, 11))
    plot_tree(
        clf,
        feature_names=feat_names,
        class_names=["Not converted", "Converted"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "decision_tree.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    from sklearn.tree import export_text
    rules = export_text(clf, feature_names=feat_names, max_depth=4)
    (FIGURES_DIR / "decision_tree_rules.txt").write_text(rules)
    log.info("Wrote decision_tree.png and decision_tree_rules.txt")


def build_feature_summary(train, test, feats) -> pd.DataFrame:
    """Summary statistics for every model input, on the full modelling sample."""
    full = pd.concat([train, test], ignore_index=True)
    rows = []
    for col in feats:
        s = full[col]
        missing_pct = round(100 * s.isna().mean(), 1)
        if pd.api.types.is_numeric_dtype(s):
            rows.append({
                "feature": col, "type": "numeric", "n": int(s.notna().sum()),
                "missing_pct": missing_pct,
                "mean": round(float(s.mean()), 2),
                "sd": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "median": round(float(s.median()), 2),
                "max": round(float(s.max()), 2),
                "n_levels": "", "modal_level": "",
            })
        else:
            vc = s.value_counts(dropna=True)
            modal = f"{vc.index[0]} ({vc.iloc[0]})" if len(vc) else ""
            rows.append({
                "feature": col, "type": "categorical", "n": int(s.notna().sum()),
                "missing_pct": missing_pct,
                "mean": "", "sd": "", "min": "", "median": "", "max": "",
                "n_levels": int(s.nunique(dropna=True)), "modal_level": modal,
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "feature_summary_statistics.csv", index=False)
    log.info("Wrote feature_summary_statistics.csv (%d features)", len(out))
    return out


def main() -> int:
    train, test, feats = _load_split()
    log.info("Loaded train=%d test=%d feats=%d", len(train), len(test), len(feats))
    build_train_vs_test_table(train, test, feats)
    build_decision_tree_viz(train, feats)
    build_feature_summary(train, test, feats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
