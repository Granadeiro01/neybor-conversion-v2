"""
Convert Salesforce SOQL JSON exports to CSV files for thesis analysis.
Produces:
  - applications.csv  (976 records from dshift__Application__c)
  - properties.csv    (68 records from dshift__Property__c)
  - units.csv         (315 records from dshift__Unit__c)
  - contracts.csv     (2505 records from dshift__Contract__c)
"""

import json
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_TOOLS = "/Users/andregranadeiro/.cursor/projects/Users-andregranadeiro-Documents-Neybor-Neybor/agent-tools"

SOURCES = {
    "applications": {
        "files": [
            "a1b9088e-4405-4d9b-a984-8124a1015aa9.txt",  # batch 1 (500)
            "ce4beace-7cea-4193-9e38-611c113e905e.txt",  # batch 2 (476)
        ],
        "output": "applications.csv",
    },
    "properties": {
        "files": ["f86e7e46-f38a-4a18-bc8f-2daa59b8a32c.txt"],
        "output": "properties.csv",
    },
    "units": {
        "files": ["edc3489c-bfc5-4460-a8f4-a7886ee13f33.txt"],
        "output": "units.csv",
    },
    "contracts": {
        "files": [
            "eaa67c3c-61e8-4c56-a000-f0a8de9645d6.txt",  # batch 1 (2000)
            "9abd4d5d-7c26-443c-a67b-779645d6efc6.txt",  # batch 2 (505)
        ],
        "output": "contracts.csv",
    },
}


def extract_records_from_file(filepath):
    """Parse the SOQL JSON output file and return the records list."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    json_start = content.index("{")
    json_str = content[json_start:]
    data = json.loads(json_str)
    return data.get("records", [])


def records_to_csv(records, output_path):
    """Write a list of Salesforce record dicts to a CSV file."""
    if not records:
        print(f"  WARNING: No records found for {output_path}")
        return 0

    fieldnames = [k for k in records[0].keys() if k != "attributes"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {k: v for k, v in rec.items() if k != "attributes"}
            writer.writerow(row)

    return len(records)


def main():
    os.makedirs(SCRIPT_DIR, exist_ok=True)

    for name, config in SOURCES.items():
        print(f"\nProcessing {name}...")
        all_records = []

        for fname in config["files"]:
            fpath = os.path.join(AGENT_TOOLS, fname)
            if not os.path.exists(fpath):
                print(f"  ERROR: File not found: {fpath}")
                sys.exit(1)

            records = extract_records_from_file(fpath)
            print(f"  Loaded {len(records)} records from {fname}")
            all_records.extend(records)

        seen_ids = set()
        deduped = []
        for rec in all_records:
            rid = rec.get("Id")
            if rid not in seen_ids:
                seen_ids.add(rid)
                deduped.append(rec)

        output_path = os.path.join(SCRIPT_DIR, config["output"])
        count = records_to_csv(deduped, output_path)
        print(f"  Wrote {count} records to {config['output']}")

    print("\nDone! All CSV files are in:", SCRIPT_DIR)


if __name__ == "__main__":
    main()
