"""Tests for leakage-safe enrichment joins."""
from __future__ import annotations

import pandas as pd

from neybor.data.join import join_enrichment


def test_join_enrichment_skips_current_inventory_aggregates_by_default():
    applications = pd.DataFrame({
        "Id": ["app1"],
        "dshift__Property__c": ["prop1"],
        "dshift__MER_Property_Group__c": ["Ixelles"],
    })
    properties = pd.DataFrame({
        "Id": ["prop1"],
        "dshift__City__c": ["Brussels"],
        "dshift__Country_Code__c": ["BE"],
        "dshift__Postal_Code__c": ["1050"],
        "Type__c": ["Coliving"],
        "Owned_by_Neybor_or_Not__c": [False],
        "Energy_Performance__c": ["A"],
        "Main_Lease_Date__c": pd.to_datetime(["2024-01-01"], utc=True),
        "dshift__MER_Property_Group__c": ["Ixelles"],
    })
    units = pd.DataFrame({
        "Id": ["unit1"],
        "dshift__Property__c": ["prop1"],
        "dshift__MER_Current_Unit_Price__c": [900.0],
        "dshift__MER_Surface__c": [20.0],
    })

    enriched = join_enrichment(applications, properties, units)

    assert enriched["property_city"].iloc[0] == "Brussels"
    assert "property_age_years" not in enriched.columns
    assert "property_owned_by_neybor" not in enriched.columns
    # The superseded per-group aggregates are gone; a per-tier reference price is
    # produced instead (NaN here because the units fixture has no parseable Name).
    assert "group_median_unit_price" not in enriched.columns
    assert "unit_type_median_price" in enriched.columns
    assert pd.isna(enriched["unit_type_median_price"].iloc[0])
