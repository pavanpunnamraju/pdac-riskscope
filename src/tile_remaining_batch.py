"""Task H (one-off, throwaway): tile+QC the full ~165-patient remaining cohort (reusing
Task C's already-tiled 50 patients), then re-select the Lightning AI batch under the
$19.00 cap at the fp16-validated rate (Task F/handoff), replacing Task C's stale
fp32-rate selection.

Pipeline design (revised after a first attempt oversubscribed CPU):
- Downloading a slide is I/O-bound (waiting on the network); tiling it is CPU-bound
  (thumbnail + per-tile tissue-mask/laplacian-variance work). Running N downloads AND
  N tile jobs fully in parallel with N > core count thrashes the machine (measured:
  6 fully-parallel workers on an 8-core M3 produced a load average of 13+ and only 2
  patients finished in 42 minutes). Fix: downloads run in a thread pool sized well
  above core count (I/O-bound, cheap); tiling runs in a bounded process pool sized to
  the machine's performance-core count, so CPU work is never oversubscribed while
  downloads keep queuing up in the background.
- Each patient's raw (pre-global-blur-cut) tile QC result is written to a staging dir
  the moment it's ready, not held in memory until the very end -- so an interrupted run
  loses at most one in-flight patient, not all progress (the first attempt lost 2
  already-finished patients' work because parquets were only written after the full
  115-patient loop completed).

Task C already tiled+QC'd 50 of the 165 remaining patients (data/raw/lightning_batch_selection.csv,
data/processed/tiles_batch/*.parquet) using a blur threshold computed only over those 50
patients' tiles. This script re-tiles the other 115, then recomputes ONE global blur
threshold (5th percentile laplacian_var among background/dark-passing tiles) across the
FULL 165-patient pool -- the old 50's pre-blur status is reconstructed from their saved
parquets (qc_reason in {"", "blurry"} = passed background/dark checks, laplacian_var
already stored), so it does NOT require re-downloading those 50 slides. Finally
re-selects patients in manifest order under the fp16-rate $19.00 cap.

Does NOT run Lightning AI extraction -- money is spent there and needs explicit human
go-ahead (see docs/references/handoff.md).

Usage:
    python src/tile_remaining_batch.py tile      # download+tile un-tiled patients (resumable)
    python src/tile_remaining_batch.py finalize  # recompute global threshold + selection
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import download_file
from tile_qc import BLUR_PERCENTILE, coarse_tissue_mask, tile_slide

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REMAINING_MANIFEST = DATA_DIR / "raw" / "remaining_manifest.csv"
EXISTING_SELECTION = DATA_DIR / "raw" / "lightning_batch_selection.csv"
SLIDES_BATCH_DIR = DATA_DIR / "raw" / "slides_batch"
TILES_BATCH_DIR = DATA_DIR / "processed" / "tiles_batch"
STAGING_DIR = DATA_DIR / "processed" / "tiles_batch_raw"  # pre-global-blur, checkpointed

# fp32 pilot rate (Task C): $7.23 / 270,416 tiles
OLD_COST_PER_TILE = 7.23 / 270_416
# fp16 L4-measured speedup (Task F): 16.0ms/tile vs 47.8ms/tile fp32
FP16_SPEEDUP = 16.0 / 47.8
NEW_COST_PER_TILE = OLD_COST_PER_TILE * FP16_SPEEDUP
SPEND_CAP = 21.91
TARGET_TILES = SPEND_CAP / NEW_COST_PER_TILE

FULL_COUNTS_OUT = DATA_DIR / "raw" / "full_remaining_tile_counts.csv"

N_DOWNLOAD_WORKERS = int(os.environ.get("N_DOWNLOAD_WORKERS", 6))
N_TILE_WORKERS = int(os.environ.get("N_TILE_WORKERS", 3))  # CPU-bound, machine has 4 performance cores


def provisional_pass_mask(df):
    return df["qc_reason"].isin(["", "blurry"])


def tile_worker(slide_path_str):
    """Runs in the CPU-bound process pool. Pure function, no shared state."""
    slide_path = Path(slide_path_str)
    import openslide
    slide = openslide.OpenSlide(str(slide_path))
    coarse_mask, coarse_scale = coarse_tissue_mask(slide)
    slide.close()
    return tile_slide(slide_path, coarse_mask, coarse_scale)


def stage_patient(row, tile_pool, idx, total):
    """Runs in a download thread: download (I/O), hand off to the shared tile process
    pool (blocks this thread, not the other download threads), stage result to disk."""
    case_id = row["case_submitter_id"]
    slide_path = SLIDES_BATCH_DIR / f"{case_id}__{row['file_name']}"
    t0 = time.monotonic()
    print(f"[{idx}/{total}] {case_id} ... downloading", flush=True)
    try:
        download_file(row["file_id"], slide_path, retries=6)
    except Exception as e:
        print(f"  {case_id} download FAILED: {e}", flush=True)
        return case_id, False

    t_dl = time.monotonic() - t0
    print(f"[{idx}/{total}] {case_id} ... downloaded ({t_dl:.0f}s), queued for tiling", flush=True)
    try:
        df = tile_pool.submit(tile_worker, str(slide_path)).result()
    except Exception as e:
        print(f"  {case_id} tile/QC FAILED: {e}", flush=True)
        slide_path.unlink(missing_ok=True)
        return case_id, False

    slide_path.unlink(missing_ok=True)  # reclaim disk immediately
    df["tss_code"] = row["tss_code"]
    df.to_parquet(STAGING_DIR / f"{case_id}_tiles_raw.parquet", index=False)
    elapsed = time.monotonic() - t0
    n_provisional = int(provisional_pass_mask(df).sum())
    print(f"  {case_id} done: {n_provisional:,} provisional-pass tiles ({elapsed:.0f}s total)", flush=True)
    return case_id, True


def cmd_tile():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    SLIDES_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(REMAINING_MANIFEST)
    existing = pd.read_csv(EXISTING_SELECTION)
    already_tiled = set(existing["case_submitter_id"])
    already_staged = {p.stem.replace("_tiles_raw", "") for p in STAGING_DIR.glob("*_tiles_raw.parquet")}
    todo = manifest[
        ~manifest["case_submitter_id"].isin(already_tiled)
        & ~manifest["case_submitter_id"].isin(already_staged)
    ]
    print(f"{len(manifest)} total remaining patients; {len(already_tiled)} already tiled by "
          f"Task C; {len(already_staged)} already staged from a prior run of this script; "
          f"{len(todo)} left to tile now ({N_DOWNLOAD_WORKERS} download threads, "
          f"{N_TILE_WORKERS} tile processes).\n", flush=True)

    if len(todo) == 0:
        print("Nothing left to tile. Run with 'finalize' to recompute selection.")
        return

    with ProcessPoolExecutor(max_workers=N_TILE_WORKERS) as tile_pool:
        with ThreadPoolExecutor(max_workers=N_DOWNLOAD_WORKERS) as dl_pool:
            futures = [
                dl_pool.submit(stage_patient, row, tile_pool, i + 1, len(manifest))
                for i, row in todo.iterrows()
            ]
            done = 0
            for fut in as_completed(futures):
                case_id, ok = fut.result()
                done += 1
                if not ok:
                    print(f"  ({done}/{len(todo)} attempted, {case_id} failed)", flush=True)

    print(f"\nTiling pass complete. Run 'finalize' to recompute the global threshold and selection.")


def cmd_finalize():
    manifest = pd.read_csv(REMAINING_MANIFEST)
    existing = pd.read_csv(EXISTING_SELECTION)
    already_tiled = set(existing["case_submitter_id"])

    all_dfs = {}
    for case_id in already_tiled:
        all_dfs[case_id] = pd.read_parquet(TILES_BATCH_DIR / f"{case_id}_tiles.parquet")
    for path in STAGING_DIR.glob("*_tiles_raw.parquet"):
        case_id = path.stem.replace("_tiles_raw", "")
        all_dfs[case_id] = pd.read_parquet(path)

    missing = set(manifest["case_submitter_id"]) - set(all_dfs)
    print(f"{len(all_dfs)}/{len(manifest)} patients tiled ({len(missing)} missing/failed).")

    combined = pd.concat(
        [d.assign(case_submitter_id=cid) for cid, d in all_dfs.items()],
        ignore_index=True,
    )
    provisional_pass = provisional_pass_mask(combined)
    blur_threshold = float(np.percentile(combined.loc[provisional_pass, "laplacian_var"], BLUR_PERCENTILE))
    final_pass = provisional_pass & (combined["laplacian_var"] >= blur_threshold)
    combined["qc_pass"] = final_pass
    combined["qc_reason"] = np.where(
        final_pass, "",
        np.where(provisional_pass, "blurry", combined["qc_reason"]),
    )
    print(f"Global blur threshold (5th pct laplacian_var, n={int(provisional_pass.sum()):,} "
          f"provisional-pass tiles): {blur_threshold:.2f}")

    TILES_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for case_id, df in combined.groupby("case_submitter_id"):
        df.drop(columns=["case_submitter_id"]).to_parquet(TILES_BATCH_DIR / f"{case_id}_tiles.parquet", index=False)

    per_patient_final = combined.groupby("case_submitter_id")["qc_pass"].sum().astype(int).to_dict()

    full_rows = []
    for _, row in manifest.iterrows():
        cid = row["case_submitter_id"]
        if cid not in per_patient_final:
            continue
        full_rows.append({
            "case_submitter_id": cid,
            "file_name": row["file_name"],
            "tss_code": row["tss_code"],
            "n_pass": per_patient_final[cid],
        })
    full_df = pd.DataFrame(full_rows)
    full_df["cumulative_n_pass"] = full_df["n_pass"].cumsum()
    full_df.to_csv(FULL_COUNTS_OUT, index=False)
    print(f"\nFull-cohort tile counts saved: {FULL_COUNTS_OUT} ({len(full_df)} patients, "
          f"{int(full_df['n_pass'].sum()):,} total pass-tiles)")

    selection = full_df[full_df["cumulative_n_pass"] <= TARGET_TILES].copy()
    if len(selection) < len(full_df):
        next_idx = len(selection)
        print(f"Cap reached: {full_df.iloc[next_idx]['case_submitter_id']} would push cumulative "
              f"over budget ({full_df.iloc[next_idx]['cumulative_n_pass']:,} > {TARGET_TILES:,.0f}).")
    else:
        print("Entire remaining cohort fits under the cap -- no trimming needed.")

    selection.to_csv(EXISTING_SELECTION, index=False)

    total_pass = int(selection["n_pass"].sum())
    total_cost = total_pass * NEW_COST_PER_TILE
    print(f"\n=== New selection (fp16 rate) ===")
    print(f"Rate: ${NEW_COST_PER_TILE:.8f}/tile, cap ${SPEND_CAP:.2f}, target {TARGET_TILES:,.0f} tiles")
    print(f"{len(selection)} patients, {total_pass:,} pass-tiles, estimated cost ${total_cost:.2f}")
    print(f"Saved: {EXISTING_SELECTION}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tile"
    if cmd == "tile":
        cmd_tile()
    elif cmd == "finalize":
        cmd_finalize()
    else:
        print(f"Unknown command: {cmd} (use 'tile' or 'finalize')")
        sys.exit(1)
