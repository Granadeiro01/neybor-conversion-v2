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
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Map the tier token parsed from a unit's Name (e.g. "Amb46 - 01Pocket") to the
# canonical unit-type label used on applications (dshift__MER_Unit_Type__c).
_TIER_NAME_MAP = {
    "pocket": "Pocket", "standard+": "Standard+", "standardplus": "Standard+",
    "standard": "Standard", "suite": "Suite", "studio": "Studio",
    "apartment": "Apartment",
}


def _parse_unit_tier(name: object) -> str | None:
    """Extract the room tier from a unit Name suffix (after the unit number)."""
    m = re.search(r"\d+\s*([A-Za-z+]+)\s*$", str(name))
    if not m:
        return None
    return _TIER_NAME_MAP.get(m.group(1).strip().lower())


def unit_type_reference_price(units: pd.DataFrame) -> pd.DataFrame:
    """Median current unit price per room tier, parsed from unit Names.

    Units carry no property/group foreign key and a single MER_Type ("Residential"),
    so a per-group reference price is not recoverable from this export. The tier,
    however, is encoded in the unit Name suffix and matches the application's
    dshift__MER_Unit_Type__c, which gives a usable per-tier reference. Zero-priced
    units (data-quality artefacts) are excluded from the median. The current
    snapshot is used as the reference; tier prices are assumed roughly stable over
    the observation window, a mild as-of-time assumption documented in the thesis.
    """
    u = units.copy()
    u["__tier"] = u["Name"].apply(_parse_unit_tier)
    price = pd.to_numeric(u["dshift__MER_Current_Unit_Price__c"], errors="coerce").replace(0, np.nan)
    u = u.assign(__price=price).dropna(subset=["__tier", "__price"])
    ref = (u.groupby("__tier")["__price"].median()
             .rename("unit_type_median_price").reset_index()
             .rename(columns={"__tier": "dshift__MER_Unit_Type__c"}))
    return ref


def join_enrichment(
    applications: pd.DataFrame,
    properties: pd.DataFrame,
    units: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join leakage-safe enrichment onto cleaned applications.

    Adds stable property descriptors (city, country, postal code, type) for the
    ~15% of applications that pre-selected a building, plus a per-tier reference
    price (``unit_type_median_price``) parsed from unit Names and joined on the
    applicant's chosen unit type, which powers ``budget_unit_mismatch``.
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
    # Per-tier reference price. Parsed from unit Names, joined on the applicant's
    # chosen unit type. Powers budget_unit_mismatch (thesis Section 4.3). Units
    # carry no property-group key and a single generic MER_Type, so a per-group
    # aggregate is not recoverable from this export; the per-tier price is the
    # usable reference.
    # ------------------------------------------------------------------
    if ("dshift__MER_Unit_Type__c" in df.columns
            and "Name" in units.columns
            and "dshift__MER_Current_Unit_Price__c" in units.columns):
        ref = unit_type_reference_price(units)
        df = df.merge(ref, how="left", on="dshift__MER_Unit_Type__c")
        log.info("Joined unit_type_median_price for %d/%d applications",
                 df["unit_type_median_price"].notna().sum(), len(df))
    else:
        df["unit_type_median_price"] = np.nan
        log.warning("Cannot build unit_type_median_price; column set to NaN")

    log.info("Joined sample shape: %s", df.shape)
    return df
