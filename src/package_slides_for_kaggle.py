"""One-off local helper (throwaway, not part of the permanent pipeline): stage a
chunk of patients' full slides + tile-QC parquets into a folder shaped for upload as
a Kaggle Dataset, for src/extract_embeddings_kaggle.py to consume.

Why full slides (not just the tile parquet): the tile parquet only stores tile
coordinates + QC flags (a few hundred KB/patient) -- the actual pixel data that
Virchow2 embeds is read on demand from the .svs via openslide.read_region. So the
practical unit of upload is the slide file itself (~1-3GB/patient for TCGA-PAAD
diagnostic slides), not the tiny parquet. openslide-bin (already in requirements.txt)
bundles its own binary, so no system libopenslide install is needed on Kaggle.

Usage:
    .venv/bin/python src/package_slides_for_kaggle.py --patients TCGA-2J-AABU TCGA-2L-AAQA ...
    .venv/bin/python src/package_slides_for_kaggle.py --chunk-size 8 --chunk-index 0
(second form: auto-splits data/raw/pilot_manifest.csv -- or a full-cohort manifest if
present at that path -- into fixed-size chunks and stages chunk N)
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SLIDES_DIR = DATA_DIR / "raw" / "slides"
TILES_DIR = DATA_DIR / "processed" / "tiles"
MANIFEST_PATH = DATA_DIR / "raw" / "pilot_manifest.csv"
STAGE_ROOT = DATA_DIR / "kaggle_chunks"


def stage_chunk(patients_df, chunk_name):
    out_dir = STAGE_ROOT / chunk_name
    (out_dir / "slides").mkdir(parents=True, exist_ok=True)
    (out_dir / "tiles").mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    rows = []
    for _, row in patients_df.iterrows():
        case_id = row["case_submitter_id"]
        slide_src = SLIDES_DIR / row["file_name"]
        tiles_src = TILES_DIR / f"{case_id}_tiles.parquet"
        if not slide_src.exists():
            print(f"  WARNING: slide missing for {case_id}, skipping: {slide_src}")
            continue
        if not tiles_src.exists():
            print(f"  WARNING: tiles parquet missing for {case_id}, skipping: {tiles_src}")
            continue

        slide_dst = out_dir / "slides" / row["file_name"]
        if not slide_dst.exists():
            slide_dst.symlink_to(slide_src.resolve())
        tiles_dst = out_dir / "tiles" / tiles_src.name
        if not tiles_dst.exists():
            tiles_dst.symlink_to(tiles_src.resolve())

        total_bytes += slide_src.stat().st_size
        rows.append({"case_submitter_id": case_id, "file_name": row["file_name"]})

    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"Staged {len(rows)} patients -> {out_dir} ({total_bytes / 1e9:.1f} GB of slides, symlinked)")
    print("NOTE: symlinks won't survive a zip/upload as-is -- when uploading, either use")
    print("  `kaggle datasets create -p <dir> --dir-mode zip` (follows symlinks into the zip)")
    print("  or `rsync -L` / `cp -L` to materialize real copies first if your upload path needs that.")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", help="explicit case_submitter_id list")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)

    if args.patients:
        chunk_df = manifest[manifest["case_submitter_id"].isin(args.patients)]
        chunk_name = "manual_chunk"
    else:
        start = args.chunk_index * args.chunk_size
        end = start + args.chunk_size
        chunk_df = manifest.iloc[start:end]
        chunk_name = f"chunk_{args.chunk_index:02d}"

    if chunk_df.empty:
        print("No patients matched -- nothing to stage.")
        return

    stage_chunk(chunk_df, chunk_name)


if __name__ == "__main__":
    main()
