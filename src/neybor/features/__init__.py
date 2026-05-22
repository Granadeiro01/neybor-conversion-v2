"""Feature engineering: leakage registry, engineered features, missingness, pipeline."""
from neybor.features.engineered import (
    add_age_at_application_created,
    add_all_engineered_features,
    add_budget_unit_mismatch,
    add_demand_pressure,
    add_lead_time_days,
    add_submission_temporal,
)
from neybor.features.leakage import (
    ALLOWED_AT_SUBMISSION,
    DROPPED_NEUTRAL_FIELDS,
    FORBIDDEN,
    LeakageError,
    SENSITIVE_FIELDS,
    assert_safe,
    classify_columns,
    filter_to_allowed,
)
from neybor.features.missingness import (
    add_missingness_indicators,
    conditional_missingness_diagnostic,
    fill_rate_report,
)

try:
    from neybor.features.pipeline import build_preprocessor, split_column_types
except ModuleNotFoundError:  # pragma: no cover - training dependencies are optional for audits
    def build_preprocessor(*args, **kwargs):
        raise ModuleNotFoundError("Install scikit-learn to build preprocessing pipelines.")

    def split_column_types(*args, **kwargs):
        raise ModuleNotFoundError("Install scikit-learn to split model column types.")

__all__ = [
    "ALLOWED_AT_SUBMISSION",
    "DROPPED_NEUTRAL_FIELDS",
    "FORBIDDEN",
    "LeakageError",
    "SENSITIVE_FIELDS",
    "assert_safe",
    "classify_columns",
    "filter_to_allowed",
    "add_age_at_application_created",
    "add_all_engineered_features",
    "add_lead_time_days",
    "add_demand_pressure",
    "add_submission_temporal",
    "add_budget_unit_mismatch",
    "fill_rate_report",
    "add_missingness_indicators",
    "conditional_missingness_diagnostic",
    "build_preprocessor",
    "split_column_types",
]
