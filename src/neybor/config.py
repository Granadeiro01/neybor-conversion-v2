"""Central configuration. Edit values here rather than scattering them across the codebase.

Every script and notebook should import from this module rather than hard-coding paths,
seeds, or grid values. This is the single source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - only used before dependencies are installed
    def load_dotenv() -> bool:
        return False

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"

for _d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Snapshot directory: where the frozen CSV export lives.
# Set via .env; otherwise accept a flat data/raw export before looking for dated subdirs.
CSV_FILES: dict[str, str] = {
    "applications": "applications-primary-db.csv",
    "contracts": "contracts.csv",
    "properties": "properties.csv",
    "units": "units.csv",
    "tenant_solvency": "tenant_professional_solvency_feature_imputed_single.csv",
}


def _default_snapshot_dir() -> Path:
    if all((RAW_DIR / filename).exists() for filename in CSV_FILES.values()):
        return RAW_DIR

    snapshot_dirs = [path for path in RAW_DIR.glob("*") if path.is_dir()]
    return max(snapshot_dirs, default=RAW_DIR / "_unset", key=os.path.getmtime)


SNAPSHOT_DIR: Path = (
    Path(os.environ["SNAPSHOT_DIR"])
    if os.environ.get("SNAPSHOT_DIR")
    else _default_snapshot_dir()
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# Canonical Salesforce field names
# ---------------------------------------------------------------------------
# The status field is `dshift__Status__c`, not `Status` — Salesforce convention is
# to namespace custom fields with the package prefix.
STATUS_FIELD: str = "dshift__Status__c"
CREATED_DATE_FIELD: str = "CreatedDate"
MOVE_IN_FIELD: str = "dshift__Start_Date__c"
PROPERTY_GROUP_FIELD: str = "dshift__MER_Property_Group__c"

# ---------------------------------------------------------------------------
# Target variable definition
# ---------------------------------------------------------------------------
# Absorbing states from thesis Section 2.1
ABSORBING_STATES: tuple[str, ...] = ("Completed", "Rejected")
POSITIVE_STATE: str = "Completed"
# Per thesis, Unreachable is retained as a sensitivity-analysis variant
UNREACHABLE_STATE: str = "Unreachable"

# Rejection reason that flags dummy data (50 records per thesis Section 2.2.1)
EXCLUDE_REJECTION_REASON: str = "Testing"

# ---------------------------------------------------------------------------
# Temporal split (thesis Section 5.1.3)
# ---------------------------------------------------------------------------
# Train: June 2025 - January 2026 (8 months)
# Test:  February - April 2026 (3 months)
TRAIN_END_DATE: str = "2026-01-31"
TEST_START_DATE: str = "2026-02-01"

# ---------------------------------------------------------------------------
# Cross-validation (thesis Section 5.1.1)
# ---------------------------------------------------------------------------
CV_K_FOLDS: int = 5
CV_N_REPEATS: int = 10

# ---------------------------------------------------------------------------
# Bootstrap CI (thesis Section 5.1.1)
# ---------------------------------------------------------------------------
BOOTSTRAP_N_RESAMPLES: int = 1000
BOOTSTRAP_CI_LEVEL: float = 0.95

# ---------------------------------------------------------------------------
# Class-imbalance handling (thesis Section 5.1.2)
# ---------------------------------------------------------------------------
# Two strategies compared head-to-head: "class_weight" vs "smote"
IMBALANCE_STRATEGIES: tuple[str, ...] = ("class_weight", "smote")

# ---------------------------------------------------------------------------
# Operating threshold target precision values (thesis Section 5.1.2)
# ---------------------------------------------------------------------------
PRECISION_TARGETS: tuple[float, ...] = (0.50, 0.70)

# ---------------------------------------------------------------------------
# Calibration (thesis Section 5.4)
# ---------------------------------------------------------------------------
# ECE threshold above which to apply post-hoc calibration
ECE_RECALIBRATION_THRESHOLD: float = 0.05
# Production drift threshold (thesis Section 5.5, Risk 5)
ECE_PRODUCTION_TRIGGER: float = 0.10

# ---------------------------------------------------------------------------
# Population Stability Index (thesis Section 5.5, Risk 2)
# ---------------------------------------------------------------------------
PSI_DRIFT_THRESHOLD: float = 0.25

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HyperGrids:
    """Coarse-then-fine grids. Refine after a first pass identifies promising regions."""

    logreg: dict[str, list] = field(default_factory=lambda: {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [2000],
    })

    random_forest: dict[str, list] = field(default_factory=lambda: {
        "n_estimators": [200, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", 0.5],
    })

    xgboost: dict[str, list] = field(default_factory=lambda: {
        "n_estimators": [200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "min_child_weight": [1, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1.0, 10.0],
    })


GRIDS = HyperGrids()
