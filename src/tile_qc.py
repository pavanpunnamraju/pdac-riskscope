"""Step 2 (Tile/QC): tile the pilot TCGA-PAAD WSIs into fixed-physical-size patches
and filter background/artifact tiles. No feature extraction here (Step 3).

Design doc: docs/plans/2026-07-27-pdac-riskscope-design.md (Cancer Project root).

Output is a coordinate manifest per slide (tile location + QC verdict), not cropped
tile images — Step 3 re-reads tiles from the SVS on demand using these coordinates,
which avoids duplicating multi-GB pixel data on disk.
"""

from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageFilter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SLIDES_DIR = DATA_DIR / "raw" / "slides"
MANIFEST_PATH = DATA_DIR / "raw" / "pilot_manifest.csv"
OUT_DIR = DATA_DIR / "processed" / "tiles"

# Standardize all tiles to the same physical size regardless of native scan
# objective (this pilot mixes 40x and 20x scans, see tile_qc summary).
TARGET_MPP = 0.5  # microns/pixel, i.e. 20x-equivalent
TILE_PX = 224  # Virchow2's expected input size (224x224 @ 20x per its model card)

MIN_TISSUE_FRACTION = 0.5  # tile must be >=50% tissue to pass background filter
COARSE_MIN_TISSUE = 0.05  # coarse thumbnail pre-filter, cheap rejection of pure background
MAX_BLACK_FRACTION = 0.3  # ink marks / folds show up as large near-black regions
BLUR_PERCENTILE = 5  # bottom 5% of laplacian variance (within tissue-passing tiles) -> blurry

LAPLACIAN_KERNEL = [0, 1, 0, 1, -4, 1, 0, 1, 0]


def tissue_mask(rgb_arr):
    """Boolean tissue mask: has color (saturation) and isn't near-white or near-black."""
    hsv = np.array(Image.fromarray(rgb_arr).convert("HSV"))
    sat = hsv[..., 1].astype(np.float32)
    gray = rgb_arr.mean(axis=2)
    return (sat > 15) & (gray < 235) & (gray > 5)


def laplacian_variance(rgb_arr):
    gray = Image.fromarray(rgb_arr).convert("L")
    edges = gray.filter(ImageFilter.Kernel((3, 3), LAPLACIAN_KERNEL, scale=1))
    return float(np.array(edges, dtype=np.float32).var())


def coarse_tissue_mask(slide, thumb_max_dim=2048):
    """Cheap low-res tissue mask, used to skip full-resolution reads of pure background."""
    w, h = slide.dimensions
    scale = thumb_max_dim / max(w, h)
    thumb = slide.get_thumbnail((max(1, int(w * scale)), max(1, int(h * scale))))
    mask = tissue_mask(np.array(thumb.convert("RGB")))
    return mask, scale


def build_grid(slide):
    """Yield (x, y, read_level, read_size) for a non-overlapping grid of level-0-referenced
    tile origins, each covering TILE_PX x TILE_PX pixels at TARGET_MPP once resized."""
    mpp = float(slide.properties.get("openslide.mpp-x", TARGET_MPP))
    downsample = TARGET_MPP / mpp
    level = slide.get_best_level_for_downsample(downsample)
    level_downsample = slide.level_downsamples[level]
    read_size = max(1, round(TILE_PX * downsample / level_downsample))
    level0_tile_size = TILE_PX * downsample

    w, h = slide.dimensions
    y = 0.0
    while y < h:
        x = 0.0
        while x < w:
            yield int(x), int(y), level, read_size, level0_tile_size
            x += level0_tile_size
        y += level0_tile_size


def tile_slide(slide_path, coarse_mask, coarse_scale):
    slide = openslide.OpenSlide(str(slide_path))
    rows = []
    for x, y, level, read_size, level0_tile_size in build_grid(slide):
        # coarse rejection against the thumbnail mask, avoids a full-res read for
        # grid cells that are unambiguously background
        tx0, ty0 = int(x * coarse_scale), int(y * coarse_scale)
        tx1 = max(tx0 + 1, int((x + level0_tile_size) * coarse_scale))
        ty1 = max(ty0 + 1, int((y + level0_tile_size) * coarse_scale))
        coarse_region = coarse_mask[ty0:ty1, tx0:tx1]
        if coarse_region.size == 0 or coarse_region.mean() < COARSE_MIN_TISSUE:
            rows.append({"tile_x": x, "tile_y": y, "qc_pass": False, "qc_reason": "background",
                         "tissue_fraction": 0.0, "black_fraction": 0.0, "laplacian_var": 0.0})
            continue

        region = slide.read_region((x, y), level, (read_size, read_size)).convert("RGB")
        if read_size != TILE_PX:
            region = region.resize((TILE_PX, TILE_PX), Image.LANCZOS)
        arr = np.array(region)

        mask = tissue_mask(arr)
        tissue_fraction = float(mask.mean())
        black_fraction = float((arr.mean(axis=2) < 20).mean())
        lap_var = laplacian_variance(arr)

        if tissue_fraction < MIN_TISSUE_FRACTION:
            reason = "background"
            passed = False
        elif black_fraction > MAX_BLACK_FRACTION:
            reason = "artifact_dark"
            passed = False
        else:
            reason = ""
            passed = True

        rows.append({"tile_x": x, "tile_y": y, "qc_pass": passed, "qc_reason": reason,
                     "tissue_fraction": tissue_fraction, "black_fraction": black_fraction,
                     "laplacian_var": lap_var})
    slide.close()
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)

    per_slide = {}
    for _, row in manifest.iterrows():
        slide_path = SLIDES_DIR / row["file_name"]
        print(f"Tiling {row['case_submitter_id']} ({row['file_name']})...")
        slide = openslide.OpenSlide(str(slide_path))
        coarse_mask, coarse_scale = coarse_tissue_mask(slide)
        slide.close()
        df = tile_slide(slide_path, coarse_mask, coarse_scale)
        df["case_submitter_id"] = row["case_submitter_id"]
        df["tss_code"] = row["tss_code"]
        per_slide[row["case_submitter_id"]] = df
        out_path = OUT_DIR / f"{row['case_submitter_id']}_tiles.parquet"
        df.to_parquet(out_path, index=False)
        n_candidate = int((df["qc_reason"] != "background").sum()) if "qc_reason" in df else 0
        print(f"  grid cells: {len(df)}, candidates read at full-res: {n_candidate}, "
              f"provisional pass (pre-blur): {int(df['qc_pass'].sum())}")

    all_tiles = pd.concat(per_slide.values(), ignore_index=True)

    # Blur threshold: global 5th percentile of laplacian variance among tiles that
    # already passed the background/dark-artifact checks (percentile-based rather than
    # a fixed universal constant, since sharpness scale depends on this pilot's scans).
    passing = all_tiles[all_tiles["qc_pass"]]
    blur_threshold = float(np.percentile(passing["laplacian_var"], BLUR_PERCENTILE))
    is_blurry = all_tiles["qc_pass"] & (all_tiles["laplacian_var"] < blur_threshold)
    all_tiles.loc[is_blurry, "qc_pass"] = False
    all_tiles.loc[is_blurry, "qc_reason"] = "blurry"

    # re-save per-slide parquet with final blur-adjusted verdicts
    for case_id, df in all_tiles.groupby("case_submitter_id"):
        out_path = OUT_DIR / f"{case_id}_tiles.parquet"
        df.drop(columns=["case_submitter_id"]).to_parquet(out_path, index=False)

    summary = (
        all_tiles.groupby(["case_submitter_id", "tss_code"])
        .agg(
            n_grid_cells=("qc_pass", "size"),
            n_pass=("qc_pass", "sum"),
            n_background=("qc_reason", lambda s: (s == "background").sum()),
            n_artifact_dark=("qc_reason", lambda s: (s == "artifact_dark").sum()),
            n_blurry=("qc_reason", lambda s: (s == "blurry").sum()),
        )
        .reset_index()
    )
    summary["pass_rate"] = summary["n_pass"] / summary["n_grid_cells"]
    summary.to_csv(OUT_DIR / "tile_qc_summary.csv", index=False)

    print("\n=== Tile/QC summary ===")
    print(f"Tile size: {TILE_PX}px @ {TARGET_MPP} mpp (20x-equivalent)")
    print(f"Blur threshold (5th pct laplacian var among tissue-passing tiles): {blur_threshold:.2f}")
    print(summary.to_string(index=False))
    print(f"\nTotal grid cells: {len(all_tiles)}, total passing tiles: {int(all_tiles['qc_pass'].sum())}")
    print(f"Overall pass rate: {all_tiles['qc_pass'].mean():.1%}")
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
