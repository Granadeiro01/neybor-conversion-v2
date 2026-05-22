"""Pre-flight schema audit.

Run this FIRST whenever you load a new Salesforce export. It classifies every
column into:
  - allowed: known feature, will be passed to the model
  - outcome_leak: forbidden, encodes the target
  - temporal_leak: forbidden, populated post-creation
  - dropped_neutral: PII / display strings / structural keys, dropped silently
  - unknown: NOT in the registry — must be reviewed before training

If any column lands in `unknown`, the pipeline will refuse to train on it. You
must either (a) move it to ALLOWED_NATIVE_FIELDS in leakage.py, or (b) move it
to DROPPED_NEUTRAL_FIELDS, or (c) move it into one of the FORBIDDEN_* sets.

Usage:
    python scripts/audit_columns.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from neybor.config import SNAPSHOT_DIR
from neybor.features.leakage import classify_columns
from neybor.io import load_applications


def main() -> int:
    print(f"Loading applications from: {SNAPSHOT_DIR}")
    df = load_applications(SNAPSHOT_DIR)
    print(f"Shape: {df.shape}\n")

    cols = list(df.columns)
    buckets = classify_columns(cols)

    for bucket_name in ("allowed", "outcome_leak", "temporal_leak", "dropped_neutral", "unknown"):
        items = sorted(buckets[bucket_name])
        print(f"=== {bucket_name.upper()} ({len(items)}) ===")
        for c in items:
            fill_pct = 100 * df[c].notna().mean() if c in df.columns else 0
            print(f"  {c:<55} fill={fill_pct:5.1f}%")
        print()

    if buckets["unknown"]:
        print("\n*** UNKNOWN COLUMNS DETECTED ***")
        print("Edit src/neybor/features/leakage.py and add each unknown column to the")
        print("appropriate set (ALLOWED_*, FORBIDDEN_*, or DROPPED_NEUTRAL_FIELDS).")
        return 1

    print("All columns classified. The registry covers every column in the export.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
