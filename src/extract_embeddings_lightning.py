"""Step 3 (Feature extraction), Lightning AI-adapted: same Virchow2 fp16 embedding
recipe as src/extract_embeddings_kaggle.py, run on a Lightning AI Studio GPU for the
full-cohort batch selected by src/tile_remaining_batch.py finalize
(data/raw/lightning_batch_selection.csv).

Slides are downloaded fresh from GDC's public open-access endpoint (no token needed --
see src/ingest.py) rather than re-uploaded from local disk, because
src/tile_remaining_batch.py deletes each slide immediately after tiling to reclaim
disk. Tile coordinates + QC come from the pre-computed
data/processed/tiles_batch/<case>_tiles.parquet files, which already carry the final
qc_pass column (global blur threshold applied by tile_remaining_batch.py's finalize
step) -- this script does no QC of its own.

Resumable/idempotent per patient (skip if <case>_embeddings.npz already exists in
OUT_DIR AND was written by this exact recipe -- see RECIPE_VERSION/already_done).
A single background thread stays one patient ahead on the download so network I/O
overlaps with GPU compute; concurrency is kept to this one prefetch (not more)
because GDC has previously reset connections under higher parallel download load
(see tile_remaining_batch.py's docstring).

logic-agent review (docs/references/handoff.md, Task H): embedding recipe confirmed
sound; recommended (1) a recipe-version fingerprint in already_done() so a future
recipe change can't silently skip-and-mix incompatible embeddings, and (2)
torch.compile for the ~10-15% speedup Task G already validated as low-risk. Both
applied below.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import download_file

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
SELECTION_PATH = Path(os.environ.get("SELECTION_PATH", DATA_DIR / "raw" / "lightning_batch_selection.csv"))
MANIFEST_PATH = Path(os.environ.get("MANIFEST_PATH", DATA_DIR / "raw" / "remaining_manifest.csv"))
TILES_DIR = Path(os.environ.get("TILES_DIR", DATA_DIR / "processed" / "tiles_batch"))
SLIDES_DIR = Path(os.environ.get("SLIDES_DIR", DATA_DIR / "raw" / "slides_lightning"))
OUT_DIR = Path(os.environ.get("OUT_DIR", DATA_DIR / "processed" / "embeddings"))

TILE_PX = 224
NUM_REGISTER_TOKENS = 4  # Virchow2: class token, then 4 register tokens, then patch tokens
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))

# Bump this if embed_tiles/load_model's recipe changes -- already_done() uses it to
# refuse to skip a patient whose .npz was written by a different (possibly
# incompatible) recipe, rather than silently mixing embeddings from two recipes into
# one downstream bag.
RECIPE_VERSION = "virchow2-cls+meanpatch-fp16-v1"
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "1") == "1"


def load_env_file(path):
    """Minimal KEY=VALUE .env loader (no new dependency for a handful of vars).
    Needed for HF_TOKEN -- Virchow2 is a gated HuggingFace model."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_model(device):
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    if device == "cuda" and USE_TORCH_COMPILE:
        model = torch.compile(model)
    return model, transform


@torch.inference_mode()
def embed_tiles(model, transform, images, device):
    batch = torch.stack([transform(im) for im in images]).to(device)
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
    """True only if <case>_embeddings.npz exists AND was written by this exact
    recipe -- a file from a different/older recipe (or missing the field entirely,
    i.e. written before this check existed) is treated as not-done and re-extracted,
    rather than silently mixed with this run's output."""
    path = OUT_DIR / f"{case_id}_embeddings.npz"
    if not path.exists():
        return False
    try:
        with np.load(path) as d:
            if "recipe_version" not in d.files:
                return False
            return str(d["recipe_version"]) == RECIPE_VERSION
    except Exception:
        return False


def log_progress(case_id, status, n_tiles, elapsed_s):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "progress_log.csv"
    is_new = not log_path.exists()
    with open(log_path, "a") as f:
        if is_new:
            f.write("timestamp,case_id,status,n_tiles,elapsed_s\n")
        f.write(f"{datetime.utcnow().isoformat()},{case_id},{status},{n_tiles},{elapsed_s:.1f}\n")


def download_patient_slide(case_id, file_id, file_name):
    slide_path = SLIDES_DIR / f"{case_id}__{file_name}"
    download_file(file_id, slide_path, retries=6)
    return slide_path


def extract_patient(model, transform, device, case_id, slide_path):
    tiles = pd.read_parquet(TILES_DIR / f"{case_id}_tiles.parquet")
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


def submit_download(dl_pool, rows, idx):
    if idx >= len(rows):
        return None
    row = rows[idx]
    return dl_pool.submit(download_patient_slide, row["case_submitter_id"], row["file_id"], row["file_name"])


def main():
    load_env_file(Path(os.environ.get("ENV_PATH", Path(__file__).resolve().parent.parent / ".env")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Virchow2 (device={device})...", flush=True)
    model, transform = load_model(device)

    selection = pd.read_csv(SELECTION_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)[["case_submitter_id", "file_id"]]
    patients = selection.merge(manifest, on="case_submitter_id", how="left")
    if patients["file_id"].isna().any():
        missing = patients.loc[patients["file_id"].isna(), "case_submitter_id"].tolist()
        raise RuntimeError(f"No file_id found in manifest for: {missing}")

    todo = [row for _, row in patients.iterrows() if not already_done(row["case_submitter_id"])]
    n_skipped = len(patients) - len(todo)
    print(f"{len(patients)} patients in selection; {n_skipped} already done (skipped); "
          f"{len(todo)} left to extract.\n", flush=True)

    with ThreadPoolExecutor(max_workers=1) as dl_pool:
        next_future = submit_download(dl_pool, todo, 0)

        for i, row in enumerate(todo):
            case_id = row["case_submitter_id"]
            t0 = time.monotonic()
            try:
                slide_path = next_future.result()
            except Exception as e:
                print(f"  {case_id} download FAILED: {e}", flush=True)
                log_progress(case_id, "download_error", 0, time.monotonic() - t0)
                next_future = submit_download(dl_pool, todo, i + 1)
                continue

            next_future = submit_download(dl_pool, todo, i + 1)  # overlap next download with this GPU pass

            print(f"[{i + 1}/{len(todo)}] Embedding {case_id}...", flush=True)
            try:
                embeddings, passing = extract_patient(model, transform, device, case_id, slide_path)
            except Exception as e:
                print(f"  {case_id} FAILED: {e}", flush=True)
                log_progress(case_id, "error", 0, time.monotonic() - t0)
                continue
            finally:
                slide_path.unlink(missing_ok=True)  # reclaim disk immediately

            if embeddings is None:
                print(f"  no QC-passing tiles for {case_id}, skipping", flush=True)
                log_progress(case_id, "no_passing_tiles", 0, time.monotonic() - t0)
                continue

            out_path = OUT_DIR / f"{case_id}_embeddings.npz"
            np.savez(out_path, embeddings=embeddings, tile_x=passing["tile_x"].to_numpy(),
                     tile_y=passing["tile_y"].to_numpy(), recipe_version=RECIPE_VERSION)
            elapsed_s = time.monotonic() - t0
            print(f"  {len(passing)} tiles -> {embeddings.shape} -> {out_path} ({elapsed_s:.0f}s)", flush=True)
            log_progress(case_id, "done", len(passing), elapsed_s)

    print(f"\nDone: {len(todo)} patients processed this run, {n_skipped} already-done (skipped), "
          f"out of {len(patients)} total in selection.")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
