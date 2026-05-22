# Neybor Conversion Prediction

Predicting tenant application conversion outcomes at Neybor Services SRL using explainable AI.

Companion codebase to the master's thesis *"Predicting Tenant Application Outcomes in
Coliving: An Explainable AI Approach to Conversion Optimization"*.

## Quickstart

```bash
# 1. Install (Python 3.11)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# For thesis reproduction, pin transitive dependency resolution:
python -m pip install -e ".[dev]" -c constraints.txt

# 2. Configure
cp .env.example .env
# Edit .env to set SNAPSHOT_DIR to where your Salesforce CSV exports live

# 3. Drop your CSV exports into data/raw/ or data/raw/<YYYY-MM-DD>/
#    Expected files:
#      - applications-primary-db.csv   (dshift__Application__c)
#      - contracts.csv      (dshift__Contract__c, for context only)
#      - properties.csv     (dshift__Property__c)
#      - units.csv          (dshift__Unit__c)
#    Then freeze the export with a SHA256 manifest:
python scripts/freeze_snapshot.py

# 4. Run the full pipeline
python scripts/run_pipeline.py

# 5. Run tests
pytest

# 6. Explore in notebooks
jupyter lab notebooks/
```

## Repository layout

```
src/neybor/
  io/         CSV loading, snapshot freezing, hashing
  data/       Schema validation, cleaning, joining
  features/   Leakage registry, engineered features, missingness handling
  models/     Splits, training, tuning, evaluation
  explain/    SHAP global/local, fairness analysis
  calibration/  Reliability, ECE, post-hoc recalibration
  uplift/     Operational uplift simulation

notebooks/    Exploratory analysis (numbered for sequence)
scripts/      One-shot runners (full pipeline, jury bundle export)
tests/        Unit tests — leakage and temporal split tests are CRITICAL
reports/      Generated figures and tables for the thesis
data/         Raw / interim / processed (all gitignored)
```

## Reproducibility

The full analysis is reproducible from a frozen Salesforce export with one command:

```bash
python scripts/run_pipeline.py
```

All randomness is seeded via `RANDOM_SEED` in `.env` (default 42). The training/test
split is strictly temporal — see `src/neybor/models/splits.py`.

## Defense criteria checklist

- [ ] Raw data files in `data/raw/` or `data/raw/<date>/` with `SNAPSHOT.json`
- [ ] GDPR consent template and Data Authenticity Declaration in `reports/`
- [ ] All notebooks executable end-to-end on a clean checkout
- [ ] `pytest` passes (especially `test_leakage.py` and `test_temporal_split.py`)
- [ ] At least 15 references in the thesis bibliography
- [ ] Trained model artifacts saved to `models/` with version tag
