# Solvency layer — known reproducibility gaps

The solvency-feature join in `src/neybor/data/solvency.py` reproduces the
densification of the headline modelling matrix
(`data/processed/application_conversion_model_ready_with_solvency.csv`) **as
far as the snapshot we have allows**. This file tracks what the snapshot does
NOT cover, so a follow-up workstream can close the loop.

The headline parity diff (rerun on 2026-05-04) shows the new pipeline producing
the same 19-feature schema, the same row count out of cleaning (698), and the
same model family selection (logistic regression). It also surfaces these
residual differences:

## 1. Headline-only imputation of raw Salesforce fields

For ~17 IDs that are *absent from* `tenant_professional_solvency_feature_imputed_single.csv`
but *present in* the headline CSV, the headline producer filled raw applications
fields with category-based defaults:

| Column | Repo (post-solvency) | Headline | Likely rule |
| --- | --- | --- | --- |
| `dshift__Source__c` | `NaN` | `Website` | default for missing source |
| `Type_Of_Living__c` | `NaN` | `Coliving House` | default for Studies / Coliving applicants |
| `How_Many_Room_Do_You_Need__c` | `NaN` (3.9% raw fill) | `1-bedroom` (densely filled) | default for Coliving |
| `dshift__MER_Unit_Type__c` | `NaN` | `Pocket` / `Standard` / `Standard+` / `Suite` | derived from imputed budget bracket |
| `Monthly_Budget__c` | `NaN` | `<€750` etc. | back-fill from the imputed `monthly_budget_range` |

The producer script for these defaults is not in the repo. Reproducing the
headline values exactly requires either:

- recovering that imputation script and porting it into a new module
  (e.g. `src/neybor/data/headline_defaults.py`); or
- replacing it with a documented imputation policy (e.g. a small rule table
  alongside `solvency.py`).

The solvency-only pipeline still works without this — the missing categorical
values are imputed at modelling time by `SimpleImputer(strategy="most_frequent")`
in the sklearn preprocessor.

## 2. `Working_for__c` external source

`Working_for__c` is filled in 43.7 % of headline rows with values such as
`Ford`, `Engie`, `Gucci`, `EUROCONTROL`. Raw `applications-primary-db.csv` has
this column populated for only 38/976 rows (3.9 %), and the value set is
disjoint (`Neybor`, `EU Commission`, `WPP`, `FGS Global`...). The
solvency CSV has no `working_for` column.

Conclusion: **there is at least one more upstream table** (probably a tenant
employer / professional directory) that the headline producer joined onto
applications. It is not in `data/raw/` and not in `data/raw/SNAPSHOT.json`.

Action: locate the upstream table, freeze it into `data/raw/`, extend
`SNAPSHOT.json`, and add `join_employment(applications, employers)` next to
`join_solvency`.

## 3. The "19 dropped CreatedDate rows" mismatch

The headline run summary records `dropped_missing_dates: 19` and a final train+
test of 679 rows. Inspecting the 19 IDs the headline left out: every single
one has a perfectly valid `CreatedDate` in `applications-primary-db.csv`. This
looks like a bug in the headline producer (perhaps it pulled `created_date`
from a join where the right-hand side was sometimes null, then filtered on
that column). The new pipeline keeps all 698 valid rows, yielding train=625
and test=73 instead of headline's 616/63.

This is a strict improvement, but it does mean the holdout precision/recall
numbers will not match the headline JSON byte-for-byte. The CV-selected model
family and threshold magnitude do match (logreg, threshold ≈ 0.73 vs 0.76).

## How to verify the parity check

```bash
# Stash the headline artefact for diffing
cp data/processed/application_conversion_model_ready_with_solvency.csv \
   data/processed/application_conversion_model_ready_with_solvency.csv.headline

python scripts/freeze_snapshot.py
python scripts/run_pipeline.py --include-sensitive

python -c "
import pandas as pd
new = pd.read_csv('data/processed/application_conversion_model_ready_with_solvency.csv')
ref = pd.read_csv('data/processed/application_conversion_model_ready_with_solvency.csv.headline')
both = new.merge(ref, on='salesforce_application_id', suffixes=('_new','_ref'))
gap_cols = ['Working_for__c', 'How_Many_Room_Do_You_Need__c', 'dshift__Source__c',
            'Type_Of_Living__c', 'dshift__MER_Unit_Type__c', 'Monthly_Budget__c']
# Excluding the documented gap columns above, all features should match on the
# overlapping 698 IDs.
"
```
