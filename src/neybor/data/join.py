"""Join cleaned applications with defensible at-submission enrichment.

The real Salesforce schema reveals that dshift__Unit__c is 0% filled at the
application level, so unit-level enrichment isn't possible. We keep only stable
property descriptors by default:

1. Property level (dshift__Property__c → properties.Id): city, country, postal code,
   and property type. Available for ~15% of applications (those that pre-selected
   a specific building).

Current-snapshot inventory aggregates are not joined by default because they may
contain future prices or availability relative to older applications.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def join_enrichment(
    applications: pd.DataFrame,
    properties: pd.DataFrame,
    units: pd.DataFrame,
    *,
    include_inventory_aggregates: bool = False,
) -> pd.DataFrame:
    """Left-join leakage-safe property enrichment onto cleaned applications.

    Parameters
    ----------
    include_inventory_aggregates:
        Off by default. Turning this on assumes the unit snapshot represents
        inventory available at every application creation time, which is usually
        too strong for a historical conversion model.
    """
    df = applications.copy()

    # ------------------------------------------------------------------
    # Property-level join (dshift__Property__c → properties.Id)
    # ------------------------------------------------------------------
    if "dshift__Property__c" in df.columns and len(properties) > 0:
        prop = properties.copy()

        rename_map = {
            "dshift__City__c": "property_city",
            "dshift__Country_Code__c": "property_country",
            "dshift__Postal_Code__c": "property_postal_code",
            "Type__c": "property_type",
        }
        keep = ["Id"] + [c for c in rename_map if c in prop.columns]
        prop = prop[keep].rename(columns=rename_map)

        df = df.merge(prop, how="left", left_on="dshift__Property__c",
                      right_on="Id", suffixes=("", "_prop"))
        # Clean up the right-hand Id column
        df = df.drop(columns=[c for c in df.columns if c == "Id_prop"], errors="ignore")
        log.info("Joined %d property-level fields", len(rename_map))

    # ------------------------------------------------------------------
    # Property-group aggregate enrichment
    # Group is a neighborhood string. Aggregate unit-level statistics per group
    # by joining units → properties → group, then summarizing.
    # ------------------------------------------------------------------
    group_col = "dshift__MER_Property_Group__c"
    aggregate_cols = ("group_median_unit_price", "group_median_surface", "group_n_units")
    if not include_inventory_aggregates:
        for c in aggregate_cols:
            df[c] = np.nan
        log.info("Skipped current-snapshot inventory aggregates for as-of-time safety")
        log.info("Joined sample shape: %s", df.shape)
        return df

    if (group_col in df.columns and group_col in properties.columns
            and "dshift__MER_Current_Unit_Price__c" in units.columns):

        # Map unit -> property -> group (units don't have group directly)
        # We use the property the unit's parent_unit belongs to via property listing
        # Simplest approach: unit lacks a direct group link. We aggregate at the
        # property level via the units' property column if present, otherwise punt.

        # Check if units have a property column
        unit_to_prop_col = None
        for candidate in ("dshift__Property__c", "dshift__MER_Property__c"):
            if candidate in units.columns:
                unit_to_prop_col = candidate
                break

        if unit_to_prop_col:
            unit_with_group = units.merge(
                properties[["Id", group_col]],
                how="left", left_on=unit_to_prop_col, right_on="Id",
                suffixes=("", "_prop"),
            )
            agg = (unit_with_group
                   .groupby(group_col, dropna=True)
                   .agg(
                       group_median_unit_price=("dshift__MER_Current_Unit_Price__c", "median"),
                       group_median_surface=("dshift__MER_Surface__c", "median"),
                       group_n_units=("Id", "count"),
                   )
                   .reset_index())
            df = df.merge(agg, how="left", on=group_col)
            log.info("Joined %d group-level aggregate fields", 3)
        else:
            # Fallback: aggregate unit prices globally per group via property mapping
            # by joining via property name. Since we can't link unit->group cleanly,
            # we set these to NaN but reserve the columns so downstream code is stable.
            log.warning(
                "Cannot link units to property groups (no Property FK on units). "
                "Group-level aggregates will be NaN."
            )
            for c in aggregate_cols:
                df[c] = np.nan

    log.info("Joined sample shape: %s", df.shape)
    return df
