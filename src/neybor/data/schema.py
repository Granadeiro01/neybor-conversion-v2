"""Pandera schemas — declarative dtypes and constraints per Salesforce object.

Permissive on optional fields, strict on join keys and the target. If a CSV export
changes shape, schema validation surfaces it immediately.
"""
from __future__ import annotations

import pandas as pd

# Pandera 0.20.x (pinned in constraints.txt): Column/DataFrameSchema live on the top
# level. Newer reorganized builds may require `from pandera.pandas import …`.
try:
    from pandera import Column, DataFrameSchema
except ImportError:
    from pandera.pandas import Column, DataFrameSchema  # type: ignore[no-redef]

try:
    from pandera.engines import pandas_engine
except ImportError:
    pandas_engine = None  # type: ignore[misc, assignment]


def _datetime_column(nullable: bool, *, required: bool = True):
    dtype = getattr(pandas_engine, "DateTime", None) if pandas_engine else None
    if dtype is not None:
        return Column(dtype, nullable=nullable, required=required)
    return Column("datetime64[ns, UTC]", nullable=nullable, required=required)


def _str_column(unique: bool = False, nullable: bool = True, *, required: bool = False):
    return Column(str, unique=unique, nullable=nullable, required=required)

# ---------------------------------------------------------------------------
# Applications — the primary modelling object
# ---------------------------------------------------------------------------
applications_schema = DataFrameSchema(
    {
        "Id": _str_column(unique=True, nullable=False, required=True),
        "CreatedDate": _datetime_column(nullable=False, required=True),
        "dshift__Status__c": _str_column(nullable=False, required=True),
        # Join keys
        "dshift__Property__c": Column(str, nullable=True, required=False),
        "dshift__MER_Property_Group__c": Column(str, nullable=True, required=False),
        # Common feature columns. Budget and length-of-stay are picklist STRINGS in
        # the real export ("€750 - €850", "3-6 months"); they get parsed into numerics
        # downstream by features/engineered.py.
        "dshift__Start_Date__c": _datetime_column(nullable=True, required=False),
        "Monthly_Budget__c": Column(str, nullable=True, required=False),
        "Length_of_Stay__c": Column(str, nullable=True, required=False),
        "Rejected_Lost_Reason__c": Column(str, nullable=True, required=False),
    },
    strict=False,
    coerce=True,
)

# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
properties_schema = DataFrameSchema(
    {
        "Id": Column(str, unique=True, nullable=False),
        "Name": Column(str, nullable=True, required=False),
        "dshift__MER_Property_Group__c": Column(str, nullable=True, required=False),
    },
    strict=False,
    coerce=True,
)

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
units_schema = DataFrameSchema(
    {
        "Id": Column(str, unique=True, nullable=False),
        "dshift__MER_Current_Unit_Price__c": Column(float, nullable=True, required=False, coerce=True),
    },
    strict=False,
    coerce=True,
)

# ---------------------------------------------------------------------------
# Contracts (used for context only)
# ---------------------------------------------------------------------------
contracts_schema = DataFrameSchema(
    {"Id": Column(str, unique=True, nullable=False)},
    strict=False,
    coerce=True,
)

# ---------------------------------------------------------------------------
# Tenant solvency / professional-situation imputed features
# ---------------------------------------------------------------------------
# Dense values for fields that are sparse in the raw applications export. The
# join key is `salesforce_application_id` and must be unique + non-null so the
# left-join semantics are well-defined.
tenant_solvency_schema = DataFrameSchema(
    {
        "salesforce_application_id": _str_column(unique=True, nullable=False, required=True),
        "tenant_professional_solvency_signal": Column(str, nullable=True, required=False),
        "tenant_age": Column(float, nullable=True, required=False, coerce=True),
        "tenant_nationality": Column(str, nullable=True, required=False),
        "age_at_application_created": Column(float, nullable=True, required=False, coerce=True),
        "monthly_budget_range": Column(str, nullable=True, required=False),
        "length_of_stay": Column(str, nullable=True, required=False),
    },
    strict=False,
    coerce=True,
)


def validate(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate one of the known DataFrames by name. Raises SchemaError on failure."""
    schemas = {
        "applications": applications_schema,
        "properties": properties_schema,
        "units": units_schema,
        "contracts": contracts_schema,
        "tenant_solvency": tenant_solvency_schema,
    }
    if name not in schemas:
        return df
    return schemas[name].validate(df, lazy=True)
