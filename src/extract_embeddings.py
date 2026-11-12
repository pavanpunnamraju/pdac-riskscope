"""Step 3 (Feature extraction): run Virchow2 (frozen) over each QC-passing tile and
produce one per-tile embedding, persisted per-patient for later bag construction (Step 4).

Design doc: docs/plans/2026-07-27-pdac-riskscope-design.md (Cancer Project root).

Per-tile embedding = class token concatenated with mean-pooled patch tokens, per
Virchow2's model card (paige-ai/Virchow2 on HuggingFace). Virchow2 prepends 4 register
tokens after the class token, so patch tokens are output[:, 5:] — NOT output[:, 1:],
which would wrongly fold register tokens into the mean-pool.

Pilot cohort only (18 patients) — full 185-cohort scaling is Milestone 2, not this step.
"""

import os
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SLIDES_DIR = DATA_DIR / "raw" / "slides"
MANIFEST_PATH = DATA_DIR / "raw" / "pilot_manifest.csv"
TILES_DIR = DATA_DIR / "processed" / "tiles"
OUT_DIR = DATA_DIR / "processed" / "embeddings"

TILE_PX = 224  # must match src/tile_qc.py's TILE_PX (Virchow2's expected input)
NUM_REGISTER_TOKENS = 4  # Virchow2-specific: class token, then 4 register tokens, then patch tokens

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))


def load_env_file(path):
    """Minimal KEY=VALUE .env loader (no new dependency for a handful of vars)."""
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
    return model, transform


@torch.inference_mode()
def embed_tiles(model, transform, images, device):
    """images: list of PIL RGB images, TILE_PX x TILE_PX. Returns (n, 2*embed_dim) array."""
    batch = torch.stack([transform(im) for im in images]).to(device)
    # Virchow2's model card recommends mixed-precision inference on GPU:
    # torch.autocast(device_type="cuda", dtype=torch.float16). Only apply on cuda --
    # autocast fp16 isn't the documented/beneficial path on cpu or mps.
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)  # (n, 1 + NUM_REGISTER_TOKENS + n_patches, embed_dim)
    else:
        output = model(batch)
    class_token = output[:, 0]
    patch_tokens = output[:, NUM_REGISTER_TOKENS + 1:]
    embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
    return embedding.cpu().numpy()


def read_tile(slide, tile_x, tile_y):
    region = slide.read_region((tile_x, tile_y), 0, (TILE_PX, TILE_PX)).convert("RGB")
    return region


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


def main():
    load_env_file(Path(__file__).resolve().parent.parent / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Virchow2 (device={device})...")
    model, transform = load_model(device)

    manifest = pd.read_csv(MANIFEST_PATH)
    for _, row in manifest.iterrows():
        case_id = row["case_submitter_id"]
        slide_path = SLIDES_DIR / row["file_name"]
        print(f"Embedding {case_id} ({row['file_name']})...")
        embeddings, passing = extract_patient(model, transform, device, case_id, slide_path)
        if embeddings is None:
            print(f"  no QC-passing tiles for {case_id}, skipping")
            continue

        out_path = OUT_DIR / f"{case_id}_embeddings.npz"
        np.savez(
            out_path,
            embeddings=embeddings,
            tile_x=passing["tile_x"].to_numpy(),
            tile_y=passing["tile_y"].to_numpy(),
        )
        print(f"  {len(passing)} tiles -> {embeddings.shape} -> {out_path}")

    print(f"\nDone. Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
