# Neybor Conversion Prediction

Code for the master's thesis *Predicting Tenant Application Outcomes in Coliving: An Explainable AI Approach to Conversion Optimization.*

The pipeline takes a frozen Salesforce export of tenant applications, builds a leakage-guarded feature set, trains and compares four model families under a strict temporal split, and writes the figures and tables the thesis reports. One command runs the whole thing.

## What is not in this repository

The applicant data is confidential and is never committed, for GDPR reasons. A fresh clone therefore contains the code but no data, and it will not run until the export is in place. The jury receives the pseudonymised export separately, together with the Data Authenticity Declaration; the files go in `data/raw/` as described below. Generated outputs (figures, tables, fitted models) are not committed either, because the pipeline regenerates them.

## Requirements

Python 3.11.

## Run it end to end

```bash
# 1. Install, with dependencies pinned for reproducibility
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]" -c constraints.txt

# 2. Place the export in data/raw/
#      applications-primary-db.csv                              (applications)
#      contracts.csv                                            (context only)
#      properties.csv                                           (property enrichment)
#      units.csv                                                (unit prices)
#      tenant_professional_solvency_feature_imputed_single.csv  (densified fields)
#    No further configuration is needed when the files sit directly in data/raw/.
#    To read them from elsewhere, set SNAPSHOT_DIR in a .env file (see .env.example).

# 3. Freeze the export: writes a SHA-256 manifest so the data cannot drift unnoticed
python scripts/freeze_snapshot.py

# 4. Run the full pipeline
python scripts/run_pipeline.py
```

## Where the results land

`run_pipeline.py` writes everything under `reports/` and `models/`:

- `reports/tables/model_comparison.csv` — hold-out metrics for the selected model.
- `reports/tables/temporal_cv_model_selection.csv` — the cross-validation ranking that drives model selection.
- `reports/tables/logreg_holdout_*` — calibration (reliability diagram and bins), decile uplift, and the group-sensitivity report.
- `reports/figures/logreg_holdout_shap_*` — global and local SHAP attributions.
- `models/<model>.joblib` — the fitted model.

`python scripts/thesis_artifacts.py` adds the supplementary thesis exports: the training-versus-hold-out comparison table, the decision-tree figure, and the feature summary statistics.

## How it works, in one paragraph

The sample is the 698 applications that reached a final state, Completed or Rejected. Features are restricted to what is known at the post-call scoring moment; a leakage registry in `src/neybor/features/leakage.py` classifies every column and the run aborts if an unclassified one appears. The train/test split is strictly chronological, training through 2025-12-31 and testing from 2026-01-01, so the model is always evaluated on later data than it trained on. The model family and operating threshold are chosen on training-period cross-validation only, and the hold-out is scored exactly once. All randomness is seeded through `RANDOM_SEED` (default 42).

## Tests

```bash
pytest
```

The load-bearing tests are `tests/test_leakage.py`, which checks that no forbidden field reaches the model, and `tests/test_temporal_split.py`, which checks that the split never leaks the future. Both must pass.

## Layout

```
src/neybor/
  io/           CSV loading, snapshot freezing, hashing
  data/         schema validation, cleaning, joining, solvency densification
  features/     leakage registry, engineered features, coliving enrichment
  models/       temporal split, training (four families), selection, evaluation
  explain/      SHAP global and local
  calibration/  reliability, ECE, Brier
  uplift/       decile uplift simulation
  fairness.py   subgroup sensitivity report
scripts/        freeze_snapshot.py, run_pipeline.py, thesis_artifacts.py
tests/          unit tests (leakage and temporal split are the critical ones)
data/raw/       place the Salesforce export here (gitignored)
reports/        generated figures and tables (gitignored)
models/         fitted model artifacts (gitignored)
```
