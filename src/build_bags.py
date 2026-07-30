"""Step 4 (Bag construction): group each patient's Virchow2 tile embeddings
(Step 3 output) into one variable-length bag per patient, joined against the
patient-level survival outcome from Step 1's native clinical data.

Design doc: docs/plans/2026-07-27-pdac-riskscope-design.md (Step 4 row).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "processed" / "embeddings"
CLINICAL_PATH = DATA_DIR / "raw" / "clinical" / "pilot_clinical.json"
MANIFEST_PATH = DATA_DIR / "raw" / "pilot_manifest.csv"
BAGS_DIR = DATA_DIR / "processed" / "bags"
BAGS_MANIFEST_PATH = BAGS_DIR / "bags_manifest.csv"


def primary_diagnosis(diagnoses):
    """The one diagnosis record carrying stage + follow-up; others are
    secondary/duplicate GDC diagnosis entries with only age_at_diagnosis."""
    for dg in diagnoses:
        if "ajcc_pathologic_stage" in dg:
            return dg
    return {}


def build_outcome(record):
    dem = record.get("demographic", {})
    diag = primary_diagnosis(record.get("diagnoses", []))
    vital_status = dem.get("vital_status")
    issues = []

    if vital_status is None:
        issues.append("missing vital_status")
        return None, issues

    event = 1 if vital_status == "Dead" else 0
    if event:
        time = dem.get("days_to_death")
        if time is None:
            issues.append("Dead but missing days_to_death")
    else:
        time = diag.get("days_to_last_follow_up")
        if time is None:
            issues.append("Alive but missing days_to_last_follow_up")

    if time is None:
        return None, issues

    outcome = {
        "event": event,
        "time": float(time),
        "stage": diag.get("ajcc_pathologic_stage"),
    }
    return outcome, issues


def main():
    BAGS_DIR.mkdir(parents=True, exist_ok=True)

    clinical = {r["submitter_id"]: r for r in json.load(open(CLINICAL_PATH))}
    manifest = pd.read_csv(MANIFEST_PATH).set_index("case_submitter_id")

    embedding_files = sorted(EMBEDDINGS_DIR.glob("*_embeddings.npz"))
    patient_ids = [f.name.removesuffix("_embeddings.npz") for f in embedding_files]

    rows = []
    all_issues = {}

    for patient_id, emb_path in zip(patient_ids, embedding_files):
        issues = []

        data = np.load(emb_path)
        embeddings = data["embeddings"]
        n_tiles = embeddings.shape[0]
        if n_tiles == 0:
            issues.append("zero QC-passing tiles")

        record = clinical.get(patient_id)
        if record is None:
            issues.append("missing clinical record")
            all_issues[patient_id] = issues
            continue

        outcome, outcome_issues = build_outcome(record)
        issues.extend(outcome_issues)

        if n_tiles == 0 or outcome is None:
            all_issues[patient_id] = issues
            continue

        tss_code = manifest.loc[patient_id, "tss_code"] if patient_id in manifest.index else None

        np.savez(
            BAGS_DIR / f"{patient_id}_bag.npz",
            embeddings=embeddings,
            tile_x=data["tile_x"],
            tile_y=data["tile_y"],
        )

        rows.append({
            "patient_id": patient_id,
            "n_tiles": n_tiles,
            "event": outcome["event"],
            "time": outcome["time"],
            "stage": outcome["stage"],
            "tss_code": tss_code,
        })

        if issues:
            all_issues[patient_id] = issues

    bags_manifest = pd.DataFrame(rows).sort_values("patient_id")
    bags_manifest.to_csv(BAGS_MANIFEST_PATH, index=False)

    print(f"Bags built: {len(bags_manifest)}/{len(patient_ids)} patients")
    print(f"Events: {int(bags_manifest['event'].sum())} dead, "
          f"{int((bags_manifest['event'] == 0).sum())} alive")
    print(f"Tiles per bag: min={bags_manifest['n_tiles'].min()}, "
          f"median={bags_manifest['n_tiles'].median():.0f}, "
          f"max={bags_manifest['n_tiles'].max()}")
    if all_issues:
        print("Data-quality issues:")
        for pid, issues in all_issues.items():
            print(f"  {pid}: {', '.join(issues)}")
    else:
        print("No data-quality issues.")


if __name__ == "__main__":
    main()
