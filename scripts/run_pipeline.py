"""End-to-end pipeline: raw CSVs → trained model → metrics on temporal hold-out.

Run with:
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neybor.config import (
    FIGURES_DIR,
    INTERIM_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    SNAPSHOT_DIR,
    TABLES_DIR,
)
from neybor.calibration.reliability import save_calibration_artifacts
from neybor.data import (
    HEADLINE_FEATURES,
    clean_applications,
    drop_missing_created_date,
    join_enrichment,
    join_solvency,
    model_ready_frame,
    primary_sample,
    select_model_feature_columns,
    validate,
)
from neybor.explain.shap_global import save_global_shap_artifacts
from neybor.explain.shap_local import save_local_shap_artifacts
from neybor.fairness import save_group_sensitivity_report
from neybor.features import (
    SENSITIVE_FIELDS,
    add_all_engineered_features,
    add_missingness_indicators,
    classify_columns,
    fill_rate_report,
    filter_to_allowed,
)
from neybor.io import load_all, verify_manifest, write_manifest
from neybor.models import (
    evaluate,
    metrics_to_dataframe,
    select_model_and_threshold,
    temporal_split,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from neybor.uplift.decile_simulation import save_decile_uplift_artifacts


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    log = logging.getLogger("pipeline")

    # ------------------------------------------------------------------
    # 1. Load + freeze snapshot
    # ------------------------------------------------------------------
    log.info("Loading snapshot from %s", SNAPSHOT_DIR)
    if not SNAPSHOT_DIR.exists():
        raise FileNotFoundError(
            f"Snapshot directory {SNAPSHOT_DIR} not found. Set SNAPSHOT_DIR in .env."
        )
    if args.write_manifest:
        manifest_path = write_manifest(SNAPSHOT_DIR, comment="Auto-frozen by run_pipeline")
        log.info("Wrote snapshot manifest: %s", manifest_path)
    if not args.skip_manifest_check:
        manifest_ok, manifest_problems = verify_manifest(SNAPSHOT_DIR)
        if not manifest_ok:
            for problem in manifest_problems:
                log.error("Snapshot manifest problem: %s", problem)
            log.error("Run with --write-manifest after freezing the raw CSV export.")
            return 1

    raw = load_all(SNAPSHOT_DIR)
    log.info("Loaded objects: %s", {k: len(v) for k, v in raw.items()})

    # ------------------------------------------------------------------
    # 2. Validate schemas
    # ------------------------------------------------------------------
    for name, df in raw.items():
        validate(name, df)
    log.info("Schema validation passed")

    # ------------------------------------------------------------------
    # 3. Pre-flight: classify every input column
    # ------------------------------------------------------------------
    buckets = classify_columns(list(raw["applications"].columns))
    if buckets["unknown"]:
        log.error(
            "Pre-flight audit FAILED. %d unknown columns: %s",
            len(buckets["unknown"]), buckets["unknown"],
        )
        log.error("Run scripts/audit_columns.py and update src/neybor/features/leakage.py")
        return 1
    log.info(
        "Pre-flight audit OK. allowed=%d, outcome_leak=%d, temporal_leak=%d, dropped=%d",
        len(buckets["allowed"]), len(buckets["outcome_leak"]),
        len(buckets["temporal_leak"]), len(buckets["dropped_neutral"]),
    )

    # ------------------------------------------------------------------
    # 4. Clean applications + (optional) solvency join + property enrichment
    # ------------------------------------------------------------------
    cleaned = clean_applications(raw["applications"])

    use_solvency = not args.no_solvency
    if use_solvency:
        if "tenant_solvency" not in raw:
            log.error(
                "tenant_solvency table not loaded. Re-freeze the snapshot or run "
                "with --no-solvency to use the legacy raw-only pipeline."
            )
            return 1
        cleaned = join_solvency(cleaned, raw["tenant_solvency"])
        log.info("Solvency join applied. Cleaned sample after join: N=%d", len(cleaned))
    else:
        log.info("--no-solvency: legacy raw-only pipeline")

    enriched = join_enrichment(cleaned, raw["properties"], raw["units"])
    enriched.to_parquet(INTERIM_DIR / "applications_enriched.parquet", index=False)

    sample = primary_sample(enriched) if not args.include_unreachable else enriched
    if use_solvency:
        # Match the headline run: drop missing CreatedDate from the primary
        # sample (the headline JSON records `dropped_missing_dates: 19`).
        sample = drop_missing_created_date(sample, created_field="CreatedDate")
    log.info("Modelling sample: N=%d, positive rate=%.3f",
             len(sample), sample["target"].mean())

    # ------------------------------------------------------------------
    # 5. Engineered features + missingness indicators
    # ------------------------------------------------------------------
    sample = add_all_engineered_features(sample)
    # Only add missingness indicators on ALLOWED feature columns to avoid creating
    # indicators for forbidden fields. When solvency is on we skip them entirely
    # because the upstream imputation already filled the relevant fields and the
    # headline modelling matrix has no `_was_missing` columns.
    if use_solvency:
        log.info("Skipping missingness indicators (solvency join provides imputed values)")
    else:
        indicator_candidates = filter_to_allowed(list(sample.columns))
        sample, indicator_cols = add_missingness_indicators(sample, columns=indicator_candidates)
        log.info("Added %d missingness indicator columns", len(indicator_cols))

    fill_rate_report(sample).to_csv(TABLES_DIR / "fill_rates.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Temporal split
    # ------------------------------------------------------------------
    train_df, test_df = temporal_split(sample)

    # Drop columns that aren't features (target, sensitivity flag, raw datetimes,
    # forbidden fields, dropped-neutral fields).
    if use_solvency:
        # Reproduce the exact headline feature matrix when the solvency join is
        # on. The legacy path keeps the leakage-registry-driven selection.
        feature_cols = [c for c in HEADLINE_FEATURES if c in train_df.columns]
        missing = [c for c in HEADLINE_FEATURES if c not in train_df.columns]
        if missing:
            log.warning("Headline features missing from train_df: %s", missing)
    else:
        feature_cols = select_model_feature_columns(
            train_df,
            include_sensitive=args.include_sensitive,
        )

    log.info("Final feature set (n=%d):", len(feature_cols))
    for c in feature_cols:
        log.info("  - %s", c)

    if not feature_cols:
        log.error("No features survived filtering. Check the leakage registry.")
        return 1

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    model_ready_frame(train_df, feature_cols).to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    model_ready_frame(test_df, feature_cols).to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    pd.Series(feature_cols, name="feature").to_csv(PROCESSED_DIR / "feature_columns.csv", index=False)

    if use_solvency:
        # Parity dump: the headline modelling matrix (id + metadata + features +
        # target) in the exact column order the headline CSV uses, so the new
        # output is directly diff-able against the stashed headline copy.
        parity_df = pd.concat([train_df, test_df], ignore_index=True).copy()
        scheduled_col = "dshift__URA_Scheduled_Call_Date_Time__c"
        if scheduled_col in parity_df.columns:
            sched = pd.to_datetime(parity_df[scheduled_col], errors="coerce", utc=True)
            created = pd.to_datetime(parity_df["CreatedDate"], errors="coerce", utc=True)
            parity_df["scheduled_call_date_time"] = sched
            parity_df["delta_hours_created_to_scheduled_call"] = (
                (sched - created).dt.total_seconds() / 3600.0
            )
        else:
            parity_df["scheduled_call_date_time"] = pd.NaT
            parity_df["delta_hours_created_to_scheduled_call"] = pd.NA

        parity_df = parity_df.rename(
            columns={"Id": "salesforce_application_id", "CreatedDate": "created_date"},
        )
        parity_cols = [
            "salesforce_application_id",
            "delta_hours_created_to_scheduled_call",
            "scheduled_call_date_time",
            "created_date",
            *feature_cols,
            "target",
        ]
        parity_cols = [c for c in parity_cols if c in parity_df.columns]
        parity_path = PROCESSED_DIR / "application_conversion_model_ready_with_solvency.csv"
        parity_df.loc[:, parity_cols].to_csv(parity_path, index=False)
        log.info("Wrote solvency parity CSV: %s", parity_path)

    log.info("Train: X=%s, y mean=%.3f", X_train.shape, y_train.mean())
    log.info("Test:  X=%s, y mean=%.3f", X_test.shape, y_test.mean())

    # ------------------------------------------------------------------
    # 7. Select model and threshold using training-period temporal CV
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Selecting model and threshold on training-period CV")
    log.info("=" * 60)

    model_factories = {
        "logreg": train_logistic_regression,
        "random_forest": train_random_forest,
        "xgboost": train_xgboost,
    }
    selection = select_model_and_threshold(train_df, feature_cols, model_factories)
    selection.cv_results.to_csv(TABLES_DIR / "temporal_cv_model_selection.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Evaluate selected model once on temporal hold-out
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info(
        "Evaluating %s on temporal hold-out test set (N=%d, threshold=%.4f)",
        selection.model_name,
        len(test_df),
        selection.threshold,
    )
    log.info("=" * 60)

    y_score = selection.model.predict_proba(X_test)[:, 1]
    results = evaluate(y_test, y_score, threshold=selection.threshold)
    metrics_df = metrics_to_dataframe(results)
    metrics_df.insert(0, "model", selection.model_name)
    metrics_df.insert(1, "threshold", selection.threshold)
    joblib.dump(selection.model, MODELS_DIR / f"{selection.model_name}.joblib")

    metrics_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    log.info("Saved metrics to %s", TABLES_DIR / "model_comparison.csv")

    predictions = pd.DataFrame({
        "target": y_test.to_numpy(),
        "score": y_score,
        "prediction": (y_score >= selection.threshold).astype(int),
    })
    predictions.to_csv(PROCESSED_DIR / "holdout_predictions.csv", index=False)

    save_calibration_artifacts(
        y_test,
        y_score,
        output_dir=TABLES_DIR,
        prefix=f"{selection.model_name}_holdout",
    )
    save_decile_uplift_artifacts(
        y_test,
        y_score,
        output_dir=TABLES_DIR,
        prefix=f"{selection.model_name}_holdout",
    )
    save_group_sensitivity_report(
        test_df,
        y_test,
        y_score,
        sensitive_columns=sorted(SENSITIVE_FIELDS),
        threshold=selection.threshold,
        output_dir=TABLES_DIR,
        prefix=f"{selection.model_name}_holdout",
    )
    save_global_shap_artifacts(
        selection.model,
        X_test,
        output_dir=FIGURES_DIR,
        prefix=f"{selection.model_name}_holdout_shap_global",
    )
    save_local_shap_artifacts(
        selection.model,
        X_test,
        output_dir=FIGURES_DIR,
        prefix=f"{selection.model_name}_holdout_shap_local",
    )

    print("\n" + "=" * 70)
    print("SELECTED MODEL PERFORMANCE (temporal hold-out)")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end Neybor conversion model pipeline")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--write-manifest", action="store_true",
                   help="Re-hash the snapshot CSVs and write SNAPSHOT.json")
    p.add_argument("--skip-manifest-check", action="store_true",
                   help="Allow exploratory runs without a frozen SNAPSHOT.json")
    p.add_argument("--include-unreachable", action="store_true",
                   help="Sensitivity variant: include Unreachable records")
    p.add_argument("--include-sensitive", action="store_true",
                   help="Opt in to sensitive/proxy features for sensitivity analysis")
    p.add_argument("--no-solvency", action="store_true",
                   help="Skip the tenant_solvency join (legacy raw-only pipeline). "
                        "Default behaviour applies the solvency join so the headline "
                        "modelling matrix is reproducible from the snapshot.")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
