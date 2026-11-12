"""One-off script (throwaway, not part of the permanent pipeline): fetch clinical
metadata for the FULL TCGA-PAAD cohort (all cases, not just the 18-patient pilot).
Clinical-only -- does NOT touch slide download.

Reuses src/ingest.py's GDC query helpers. Run from PDAC-RiskScope/ with the venv:
    .venv/bin/python src/fetch_full_clinical.py
"""

import json
from pathlib import Path

import requests

from ingest import GDC_CASES_ENDPOINT, query_diagnostic_slides

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CLINICAL_DIR = DATA_DIR / "clinical"


def fetch_clinical_full(case_ids):
    """Same fields/endpoint as ingest.fetch_clinical, but POST (not GET) since the
    full-cohort case_id list is long enough to blow GDC's GET URI-length limit
    (confirmed: 414 at 183 case_ids via GET)."""
    filters = {"op": "in", "content": {"field": "case_id", "value": case_ids}}
    payload = {
        "filters": filters,
        "fields": ",".join([
            "case_id",
            "submitter_id",
            "diagnoses.ajcc_pathologic_stage",
            "diagnoses.age_at_diagnosis",
            "demographic.vital_status",
            "demographic.days_to_death",
            "diagnoses.days_to_last_follow_up",
            "demographic.gender",
            "demographic.race",
        ]),
        "format": "JSON",
        "size": "1000",
    }
    resp = requests.post(GDC_CASES_ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]["hits"]


def main():
    CLINICAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Querying GDC for full TCGA-PAAD diagnostic-slide cohort...")
    all_slides = query_diagnostic_slides()
    cases = all_slides.drop_duplicates(subset="case_submitter_id")
    print(f"  {len(cases)} cases across {cases['tss_code'].nunique()} sites")

    print("Fetching clinical metadata for full cohort...")
    clinical = fetch_clinical_full(cases["case_id"].tolist())

    out_path = CLINICAL_DIR / "full_cohort_clinical.json"
    with open(out_path, "w") as f:
        json.dump(clinical, f, indent=2)

    print(f"Done. {len(clinical)} cases -> {out_path}")


if __name__ == "__main__":
    main()
