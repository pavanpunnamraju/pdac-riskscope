"""Task C (one-off, throwaway): select a Lightning AI batch of full-cohort patients
whose cumulative pass-QC tile count stays at/under a $19.00 spend cap, using the
pilot's real cost-per-tile ($7.23 / 270,416 tiles = $0.0000267/tile).

For each remaining (non-pilot) patient, in manifest order: download the slide
(free/CPU, GDC), run the same Step 2 tile+QC grid/background/dark checks as
src/tile_qc.py, record the per-tile laplacian_var, then DELETE the slide (disk is
scarce -- 97GB free vs. 191GB for the full remaining cohort). Stops collecting new
patients once the provisional (pre-blur) pass count comfortably clears the padded
target, then applies one global blur threshold (5th percentile, same method as
tile_qc.py) across the collected batch and trims from the end if that pushes the
final total over budget.

Does NOT run Lightning AI extraction -- that needs explicit human go-ahead per the
handoff (money is being spent there; this script only spends free GDC bandwidth +
local CPU).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import download_file
from tile_qc import BLUR_PERCENTILE, coarse_tissue_mask, tile_slide

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REMAINING_MANIFEST = DATA_DIR / "raw" / "remaining_manifest.csv"
SLIDES_BATCH_DIR = DATA_DIR / "raw" / "slides_batch"  # transient, deleted per-patient
TILES_BATCH_DIR = DATA_DIR / "processed" / "tiles_batch"

COST_PER_TILE = 7.23 / 270_416  # pilot: $7.23 for 270,416 pass-QC tiles (18 patients)
SPEND_CAP = 19.00
TARGET_TILES = SPEND_CAP / COST_PER_TILE  # ~710,000
# pad the collection target to absorb ~5% expected blur-cut loss (pilot's blur removed
# ~1.9% of tissue/dark-passing tiles: 270,416 pass / (270,416+~5200 blurry) -- use 8%
# padding to be safe rather than under-collecting and needing a second pass)
COLLECT_TARGET = TARGET_TILES / 0.92


def main():
    SLIDES_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    TILES_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(REMAINING_MANIFEST)
    print(f"Spend cap: ${SPEND_CAP:.2f} -> target {TARGET_TILES:,.0f} final pass-tiles "
          f"(collecting toward {COLLECT_TARGET:,.0f} pre-blur as padding)")
    print(f"{len(manifest)} remaining patients available, ~{manifest['file_size'].mean()/1e9:.2f} GB/slide avg\n")

    collected = []  # list of (case_submitter_id, file_name, tss_code, df)
    provisional_total = 0

    for i, row in manifest.iterrows():
        if provisional_total >= COLLECT_TARGET:
            break

        case_id = row["case_submitter_id"]
        slide_path = SLIDES_BATCH_DIR / row["file_name"]
        t0 = time.monotonic()
        print(f"[{i+1}/{len(manifest)}] {case_id} ({row['file_size']/1e9:.2f} GB) ... downloading", flush=True)
        try:
            download_file(row["file_id"], slide_path)
        except Exception as e:
            print(f"  download FAILED: {e}")
            continue

        try:
            slide = None
            import openslide
            slide = openslide.OpenSlide(str(slide_path))
            coarse_mask, coarse_scale = coarse_tissue_mask(slide)
            slide.close()
            df = tile_slide(slide_path, coarse_mask, coarse_scale)
        except Exception as e:
            print(f"  tile/QC FAILED: {e}")
            slide_path.unlink(missing_ok=True)
            continue

        slide_path.unlink(missing_ok=True)  # reclaim disk immediately

        df["case_submitter_id"] = case_id
        df["tss_code"] = row["tss_code"]
        n_provisional = int(df["qc_pass"].sum())  # pre-blur (background/dark filtered only)
        provisional_total += n_provisional
        elapsed = time.monotonic() - t0
        print(f"  {n_provisional:,} provisional-pass tiles ({elapsed:.0f}s) -- "
              f"running total {provisional_total:,.0f}/{COLLECT_TARGET:,.0f}")
        collected.append((case_id, row["file_name"], row["tss_code"], df))

    print(f"\nCollected {len(collected)} patients, {provisional_total:,} provisional-pass tiles.")

    all_tiles = pd.concat([c[3] for c in collected], ignore_index=True)
    passing = all_tiles[all_tiles["qc_pass"]]
    blur_threshold = float(np.percentile(passing["laplacian_var"], BLUR_PERCENTILE))
    is_blurry = all_tiles["qc_pass"] & (all_tiles["laplacian_var"] < blur_threshold)
    all_tiles.loc[is_blurry, "qc_pass"] = False
    all_tiles.loc[is_blurry, "qc_reason"] = "blurry"

    per_patient_final = all_tiles.groupby("case_submitter_id")["qc_pass"].sum().to_dict()

    # trim from the end (reverse collection order) until cumulative final n_pass <= TARGET_TILES
    selection = list(collected)
    def final_total(sel):
        return sum(per_patient_final[c[0]] for c in sel)

    while final_total(selection) > TARGET_TILES and len(selection) > 0:
        dropped = selection.pop()
        print(f"Trimming {dropped[0]} ({per_patient_final[dropped[0]]:,} tiles) to stay under cap.")

    # save per-patient tile parquets (final blur-adjusted) for the selected set only
    rows = []
    for case_id, file_name, tss_code, _ in selection:
        pdf = all_tiles[all_tiles["case_submitter_id"] == case_id].drop(columns=["case_submitter_id"])
        pdf.to_parquet(TILES_BATCH_DIR / f"{case_id}_tiles.parquet", index=False)
        rows.append({
            "case_submitter_id": case_id,
            "file_name": file_name,
            "tss_code": tss_code,
            "n_pass": int(per_patient_final[case_id]),
        })

    selection_df = pd.DataFrame(rows)
    selection_df["cumulative_n_pass"] = selection_df["n_pass"].cumsum()
    out_path = DATA_DIR / "raw" / "lightning_batch_selection.csv"
    selection_df.to_csv(out_path, index=False)

    total_pass = int(selection_df["n_pass"].sum())
    total_cost = total_pass * COST_PER_TILE
    print(f"\n=== Selection ===")
    print(selection_df.to_string(index=False))
    print(f"\n{len(selection_df)} patients, {total_pass:,} pass-tiles, "
          f"estimated cost ${total_cost:.2f} (cap ${SPEND_CAP:.2f})")
    print(f"Saved: {out_path}")
    print(f"Tile parquets: {TILES_BATCH_DIR}")


if __name__ == "__main__":
    main()
