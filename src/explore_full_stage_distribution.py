"""One-off script (throwaway, not part of the permanent pipeline): explore AJCC stage
distribution across the full TCGA-PAAD cohort, to inform the collapse-vs-keep decision
for src/validate_mil_survival.py's STAGE_ORDINAL (currently full sub-stage granularity,
sparse at pilot n=18). Reads data/raw/clinical/full_cohort_clinical.json (produced by
src/fetch_full_clinical.py).
"""

import json
from pathlib import Path

import pandas as pd

CLINICAL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "clinical" / "full_cohort_clinical.json"
)


def extract_stage(case):
    """A case can have >1 diagnoses record (e.g. a follow-up entry with no stage);
    take the first diagnosis that actually carries ajcc_pathologic_stage."""
    for dx in case.get("diagnoses", []):
        if "ajcc_pathologic_stage" in dx:
            return dx["ajcc_pathologic_stage"]
    return None


def main():
    cases = json.loads(CLINICAL_PATH.read_text())
    rows = []
    for c in cases:
        rows.append({
            "submitter_id": c.get("submitter_id"),
            "stage": extract_stage(c),
            "vital_status": c.get("demographic", {}).get("vital_status"),
        })
    df = pd.DataFrame(rows)

    print(f"Total cases: {len(df)}")
    n_missing_stage = df["stage"].isna().sum()
    print(f"Missing stage: {n_missing_stage}\n")

    print("=== Stage value_counts (full sub-stage granularity) ===")
    stage_counts = df["stage"].value_counts(dropna=False)
    print(stage_counts.to_string())

    print("\n=== Stage x vital_status cross-tab ===")
    crosstab = pd.crosstab(df["stage"], df["vital_status"], dropna=False)
    print(crosstab.to_string())

    out_path = CLINICAL_PATH.parent / "full_cohort_stage_distribution.csv"
    stage_counts.rename("count").to_csv(out_path)
    crosstab.to_csv(CLINICAL_PATH.parent / "full_cohort_stage_x_vital_status.csv")
    print(f"\nSaved: {out_path}")
    print(f"Saved: {CLINICAL_PATH.parent / 'full_cohort_stage_x_vital_status.csv'}")


if __name__ == "__main__":
    main()
