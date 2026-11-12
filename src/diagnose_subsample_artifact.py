"""Historical diagnostic, not a corrected validation procedure.

Uses historical epoch choices and globally fitted bins. Its correlation cannot
establish that subsampling is harmless. See docs/RESULTS.md.

Original question: does MAX_TILES=4000 subsampling explain Step 6's null result?

Requested by logic-agent via handoff (2026-08-10): correlate each patient's
pre-subsample original bag size (`n_tiles`, `bags_manifest.csv`) against their
held-out loss from the nested-CV run, to distinguish "real null" from
"subsampling artifact" without a full retrain.

The original Step 6 run (src/validate_mil_survival.py, on Lightning Studio)
never persisted per-fold checkpoints or per-patient held-out loss to disk --
only aggregate C-index printouts survived, and those aren't enough for this
diagnostic. Full nested CV also isn't needed here: the outer fold split is
deterministic (GroupKFold has no random_state) and the epoch count per outer
fold is already known (chosen_epochs=[3, 5, 3, 3, 3] from that run) so the
expensive inner sweep is skipped -- this only redoes the 5 outer-fold final
trainings, not the full nested CV.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mil_survival import (  # noqa: E402
    discretize_times,
    load_pilot_bags,
    nll_surv_loss,
    resolve_device,
    GatedAttentionMILSurvival,
    N_BINS,
)
from validate_mil_survival import STAGE_ORDINAL, train_one  # noqa: E402

OUTER_K = 5
SEED = 0
# From the completed Step 6 nested-CV run (docs/references/handoff.md,
# docs/live_task_progress.md "Milestone 2 result"): the inner sweep's chosen
# epoch per outer fold, in outer-fold order.
CHOSEN_EPOCHS = [3, 5, 3, 3, 3]

OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "subsample_diagnostic.csv"


def eval_per_patient_loss(model, bag_tensors, y_bin_t, censored_t, idxs):
    model.eval()
    losses = {}
    with torch.no_grad():
        for idx in idxs:
            hazards = model(bag_tensors[int(idx)])
            loss = nll_surv_loss(hazards, y_bin_t[idx].item(), censored_t[idx])
            losses[int(idx)] = loss.item()
    return losses


def main():
    manifest, bags = load_pilot_bags()

    stage_ordinal_full = manifest["stage"].map(STAGE_ORDINAL)
    missing_stage = stage_ordinal_full.isna()
    if missing_stage.any():
        keep_idx = np.flatnonzero(~missing_stage.to_numpy())
        manifest = manifest.iloc[keep_idx].reset_index(drop=True)
        bags = [bags[i] for i in keep_idx]

    n_patients = len(manifest)
    times = manifest["time"].to_numpy(dtype=np.float64)
    events = manifest["event"].to_numpy(dtype=np.int64)
    groups = manifest["tss_code"].to_numpy()
    n_tiles = manifest["n_tiles"].to_numpy(dtype=np.float64)

    print(f"n_patients={n_patients} (must match Step 6's 179; same drop logic reused)")

    y_bins, edges = discretize_times(times, events, N_BINS)
    device = resolve_device()
    print(f"device={device}")
    bag_tensors = [torch.from_numpy(b).to(device) for b in bags]
    y_bin_t = torch.from_numpy(y_bins).long().to(device)
    censored_t = torch.from_numpy(1 - events).float().to(device)
    all_idx = np.arange(n_patients)

    n_sites = len(np.unique(groups))
    n_outer_splits = min(OUTER_K, n_sites)
    outer_gkf = GroupKFold(n_splits=n_outer_splits)

    held_out_loss = np.full(n_patients, np.nan)

    for fold_i, (outer_train_pos, outer_test_pos) in enumerate(
        outer_gkf.split(all_idx, groups=groups), start=1
    ):
        outer_train_idx = all_idx[outer_train_pos]
        outer_test_idx = all_idx[outer_test_pos]
        outer_train_t = torch.as_tensor(outer_train_idx)
        best_epoch = CHOSEN_EPOCHS[fold_i - 1]

        print(f"outer fold {fold_i}/{n_outer_splits}: train={len(outer_train_idx)} "
              f"test={len(outer_test_idx)} epoch={best_epoch}")

        final_model = train_one(bag_tensors, y_bin_t, censored_t, outer_train_t, best_epoch, device)
        losses = eval_per_patient_loss(
            final_model, bag_tensors, y_bin_t, censored_t, torch.as_tensor(outer_test_idx)
        )
        for idx, loss in losses.items():
            held_out_loss[idx] = loss

    assert not np.isnan(held_out_loss).any(), "every patient must get exactly one held-out loss"

    out_df = pd.DataFrame({
        "patient_id": manifest["patient_id"],
        "n_tiles": n_tiles,
        "held_out_loss": held_out_loss,
        "subsampled": n_tiles > 4000,
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    pearson_r, pearson_p = stats.pearsonr(n_tiles, held_out_loss)
    spearman_r, spearman_p = stats.spearmanr(n_tiles, held_out_loss)
    print(f"\nPearson  r={pearson_r:+.3f}  p={pearson_p:.3f}")
    print(f"Spearman r={spearman_r:+.3f}  p={spearman_p:.3f}")

    n_subsampled = int((n_tiles > 4000).sum())
    print(f"\n{n_subsampled}/{n_patients} patients had bags subsampled (n_tiles > 4000)")
    print(f"held_out_loss by group: subsampled mean={out_df.loc[out_df.subsampled, 'held_out_loss'].mean():.4f}  "
          f"not-subsampled mean={out_df.loc[~out_df.subsampled, 'held_out_loss'].mean():.4f}")


if __name__ == "__main__":
    main()
