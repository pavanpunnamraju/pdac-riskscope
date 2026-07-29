"""Step 1 (Ingest): download a pilot set of TCGA-PAAD H&E whole-slide images (SVS)
from GDC, plus clinical metadata, stratified across submitting sites where possible.

Design doc: docs/plans/2026-07-27-pdac-riskscope-design.md (Cancer Project root).
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SLIDES_DIR = DATA_DIR / "slides"
CLINICAL_DIR = DATA_DIR / "clinical"

PILOT_SIZE = 18


def query_diagnostic_slides():
    """Query GDC for TCGA-PAAD diagnostic-slide SVS files, one per case, with
    submitter/site info attached via the case's submitter_id (TCGA barcode
    encodes tissue source site in its second element, e.g. TCGA-XX-####)."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-PAAD"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join([
            "file_id",
            "file_name",
            "file_size",
            "cases.submitter_id",
            "cases.case_id",
            "cases.samples.tissue_source_site.name",
            "cases.samples.tissue_source_site.code",
        ]),
        "format": "JSON",
        "size": "1000",
    }
    resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]

    rows = []
    for h in hits:
        case = h["cases"][0]
        samples = case.get("samples", [{}])
        tss = samples[0].get("tissue_source_site", {}) if samples else {}
        rows.append({
            "file_id": h["file_id"],
            "file_name": h["file_name"],
            "file_size": h["file_size"],
            "case_submitter_id": case["submitter_id"],
            "case_id": case["case_id"],
            "tss_code": tss.get("code", case["submitter_id"].split("-")[1]),
            "tss_name": tss.get("name", "unknown"),
        })
    return pd.DataFrame(rows)


def select_pilot(df, n=PILOT_SIZE):
    """Select n slides, one per case, spreading across as many distinct
    tissue-source sites as available (site-stratification per Howard et al. 2021,
    design doc Milestone 1 requirement)."""
    df = df.drop_duplicates(subset="case_submitter_id").sort_values("tss_code")
    sites = df["tss_code"].unique().tolist()
    picked = []
    i = 0
    by_site = {s: df[df["tss_code"] == s].to_dict("records") for s in sites}
    cursors = {s: 0 for s in sites}
    while len(picked) < n and any(cursors[s] < len(by_site[s]) for s in sites):
        for s in sites:
            if len(picked) >= n:
                break
            if cursors[s] < len(by_site[s]):
                picked.append(by_site[s][cursors[s]])
                cursors[s] += 1
    return pd.DataFrame(picked)


def download_file(file_id, out_path, retries=3):
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    url = f"{GDC_DATA_ENDPOINT}/{file_id}"
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp_path = out_path.with_suffix(out_path.suffix + ".part")
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024 * 8):
                        f.write(chunk)
                tmp_path.rename(out_path)
            return
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def fetch_clinical(case_ids):
    filters = {"op": "in", "content": {"field": "case_id", "value": case_ids}}
    params = {
        "filters": json.dumps(filters),
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
    resp = requests.get(GDC_CASES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]["hits"]


def main():
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    CLINICAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Querying GDC for TCGA-PAAD diagnostic slides...")
    all_slides = query_diagnostic_slides()
    print(f"  {len(all_slides)} diagnostic slides across {all_slides['tss_code'].nunique()} sites")

    pilot = select_pilot(all_slides, PILOT_SIZE)
    pilot.to_csv(DATA_DIR / "pilot_manifest.csv", index=False)
    print(f"Selected pilot of {len(pilot)} slides across {pilot['tss_code'].nunique()} sites:")
    print(pilot["tss_code"].value_counts())

    print("Downloading slides...")
    for i, row in pilot.iterrows():
        out_path = SLIDES_DIR / row["file_name"]
        print(f"  [{i+1}/{len(pilot)}] {row['file_name']} ({row['file_size']/1e6:.0f} MB)")
        download_file(row["file_id"], out_path)

    print("Fetching clinical metadata...")
    clinical = fetch_clinical(pilot["case_id"].tolist())
    with open(CLINICAL_DIR / "pilot_clinical.json", "w") as f:
        json.dump(clinical, f, indent=2)

    print("Done.")
    print(f"Manifest: {DATA_DIR / 'pilot_manifest.csv'}")
    print(f"Slides:   {SLIDES_DIR}")
    print(f"Clinical: {CLINICAL_DIR / 'pilot_clinical.json'}")


if __name__ == "__main__":
    main()
