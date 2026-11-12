"""Step 3 (Feature extraction), Kaggle-adapted: same Virchow2 embedding recipe as
src/extract_embeddings.py (class token concat mean-pooled patch tokens; register-token
offset unchanged), adapted to run resumable/chunked across multiple Kaggle sessions
for the full 185-cohort scale-up (Milestone 2). Pilot's src/extract_embeddings.py is
left untouched -- it's the locked pilot-scale record.

Full-cohort extraction is ~40+ GPU-hours; Kaggle caps a single session at 12h and a
free account at 30 GPU-hr/week, so this script must be safe to stop and restart:
  - per-patient idempotent resume: skips any patient whose output .npz already exists
    in OUT_DIR (checked BEFORE loading the slide, so a resumed run costs ~nothing for
    already-done patients).
  - a wall-clock budget (SESSION_BUDGET_HOURS) so the script stops itself with margin
    before Kaggle's 12h hard kill, rather than losing whatever patient was mid-flight.
  - a progress log (OUT_DIR/progress_log.csv) appended after every patient, so you can
    see exactly how far a session got even if it was killed uncleanly.

Kaggle usage:
  1. Package a chunk of patients locally with src/package_slides_for_kaggle.py, upload
     the resulting folder as a Kaggle Dataset, and attach it as this notebook's input.
  2. At the end of a session, save /kaggle/working/embeddings as a new Kaggle Dataset
     version (or download it) -- next session should start from a Kaggle input that
     includes that prior output, so INPUT patients whose .npz is already present are
     skipped instantly.
  3. Re-run this script (unchanged) each session until manifest patients are exhausted.

Paths default to Kaggle's standard layout but are all env-var overridable so the same
script also runs locally for testing.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

# --- paths -------------------------------------------------------------------
# INPUT_DIR: a Kaggle input dataset containing, per chunk: slides/<file_name>.svs,
# tiles/<case_submitter_id>_tiles.parquet, and manifest.csv (case_submitter_id,
# file_name) -- produced by src/package_slides_for_kaggle.py. Locally these default
# to this repo's existing data/ layout so the script is testable off-Kaggle too.
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/kaggle/input/pdac-riskscope-chunk"))
SLIDES_DIR = INPUT_DIR / "slides"
TILES_DIR = INPUT_DIR / "tiles"
MANIFEST_PATH = INPUT_DIR / "manifest.csv"

# OUT_DIR: also check for a PRIOR_OUT_DIR (a Kaggle input holding a previous session's
# already-completed .npz files) so resume works across dataset-version boundaries.
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/embeddings"))
PRIOR_OUT_DIR = Path(os.environ["PRIOR_OUT_DIR"]) if os.environ.get("PRIOR_OUT_DIR") else None

TILE_PX = 224
NUM_REGISTER_TOKENS = 4  # unchanged from src/extract_embeddings.py -- do not touch
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))

SESSION_BUDGET_HOURS = float(os.environ.get("SESSION_BUDGET_HOURS", "11.0"))  # margin under Kaggle's 12h cap


def load_model(device):
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


@torch.inference_mode()
def embed_tiles(model, transform, images, device):
    batch = torch.stack([transform(im) for im in images]).to(device)
    # Same mixed-precision fix as src/extract_embeddings.py -- see that file's comment.
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
    else:
        output = model(batch)
    class_token = output[:, 0]
    patch_tokens = output[:, NUM_REGISTER_TOKENS + 1:]
    embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
    return embedding.cpu().numpy()


def read_tile(slide, tile_x, tile_y):
    return slide.read_region((tile_x, tile_y), 0, (TILE_PX, TILE_PX)).convert("RGB")


def already_done(case_id):
    for d in (OUT_DIR, PRIOR_OUT_DIR):
        if d is not None and (d / f"{case_id}_embeddings.npz").exists():
            return d / f"{case_id}_embeddings.npz"
    return None


def extract_patient(model, transform, device, case_submitter_id, slide_path):
    tiles = pd.read_parquet(TILES_DIR / f"{case_submitter_id}_tiles.parquet")
    passing = tiles[tiles["qc_pass"]].reset_index(drop=True)
    if passing.empty:
        return None, passing

    slide = openslide.OpenSlide(str(slide_path))
    embeddings = []
    for start in range(0, len(passing), BATCH_SIZE):
        chunk = passing.iloc[start:start + BATCH_SIZE]
        images = [read_tile(slide, int(r.tile_x), int(r.tile_y)) for r in chunk.itertuples()]
        embeddings.append(embed_tiles(model, transform, images, device))
    slide.close()
    return np.concatenate(embeddings, axis=0), passing


def log_progress(case_id, status, n_tiles, elapsed_s):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "progress_log.csv"
    is_new = not log_path.exists()
    with open(log_path, "a") as f:
        if is_new:
            f.write("timestamp,case_id,status,n_tiles,elapsed_s\n")
        f.write(f"{datetime.utcnow().isoformat()},{case_id},{status},{n_tiles},{elapsed_s:.1f}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session_start = time.monotonic()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Virchow2 (device={device})...")
    model, transform = load_model(device)

    manifest = pd.read_csv(MANIFEST_PATH)
    n_total = len(manifest)
    n_skipped = n_done_this_session = 0

    for _, row in manifest.iterrows():
        elapsed_h = (time.monotonic() - session_start) / 3600
        if elapsed_h > SESSION_BUDGET_HOURS:
            print(f"\nSession budget ({SESSION_BUDGET_HOURS}h) reached -- stopping cleanly for resume next session.")
            break

        case_id = row["case_submitter_id"]
        prior = already_done(case_id)
        if prior is not None:
            n_skipped += 1
            continue

        slide_path = SLIDES_DIR / row["file_name"]
        t0 = time.monotonic()
        print(f"Embedding {case_id} ({row['file_name']})...")
        try:
            embeddings, passing = extract_patient(model, transform, device, case_id, slide_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            log_progress(case_id, "error", 0, time.monotonic() - t0)
            continue

        if embeddings is None:
            print(f"  no QC-passing tiles for {case_id}, skipping")
            log_progress(case_id, "no_passing_tiles", 0, time.monotonic() - t0)
            continue

        out_path = OUT_DIR / f"{case_id}_embeddings.npz"
        np.savez(
            out_path,
            embeddings=embeddings,
            tile_x=passing["tile_x"].to_numpy(),
            tile_y=passing["tile_y"].to_numpy(),
        )
        elapsed_s = time.monotonic() - t0
        print(f"  {len(passing)} tiles -> {embeddings.shape} -> {out_path} ({elapsed_s:.0f}s)")
        log_progress(case_id, "done", len(passing), elapsed_s)
        n_done_this_session += 1

    n_remaining = n_total - n_skipped - n_done_this_session
    print(f"\nSession summary: {n_done_this_session} done, {n_skipped} already-done (skipped), "
          f"{n_remaining} remaining out of {n_total} in this chunk's manifest.")
    print(f"Outputs: {OUT_DIR}")
    if n_remaining > 0:
        print("Save OUT_DIR as a new Kaggle Dataset version and re-run next session to continue.")


if __name__ == "__main__":
    main()
