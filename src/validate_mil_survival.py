"""Corrected nested, site-grouped evaluation of MIL survival and Cox baselines.

Historical results predate score-direction and training-only binning fixes.
See docs/RESULTS.md for the audit and remaining methodological limitations.
Completed runs persist predictions and provenance under results/validation/.
"""

import sys
import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mil_survival import (  # noqa: E402
    BAGS_DIR,
    N_BINS,
    discretize_times,
    load_pilot_bags,
    nll_surv_loss,
    resolve_device,
    GatedAttentionMILSurvival,
)

OUTER_K = 5
INNER_K = 3
# A prior full-cohort run picked best_epoch=5 in every one of the 4 outer folds that
# completed -- the model overfits fast. Sweeping to 50 was 10x more inner-training
# epochs than the data ever uses; narrowed to bracket the observed optimum instead of
# blindly re-sweeping the full original range.
CHECKPOINT_EPOCHS = [3, 5, 8, 12, 20]
LR = 2e-4
WEIGHT_DECAY = 1e-5
SEED = 0
N_BOOTSTRAP = 2000

STAGE_ORDINAL = {
    # Full 6-level AJCC sub-stage granularity, kept (not collapsed to
    # I/II/III/IV) -- decision resolved against the real 165+18=183-patient
    # full-cohort distribution (data/raw/clinical/full_cohort_stage_distribution.csv):
    # every populated level has a mix of Alive/Dead outcomes (no perfect
    # separation), and after folding Stage 0/0a into the earliest-stage
    # ordinal level and dropping the 2 missing-stage patients (below), the
    # smallest level is Stage III at n=4 -- above the 1-2-patient
    # near-separation threshold that would have forced a collapse.
    "Stage 0": 1,
    "Stage 0a": 1,
    "Stage IA": 1,
    "Stage IB": 2,
    "Stage IIA": 3,
    "Stage IIB": 4,
    "Stage III": 5,
    "Stage IV": 6,
}


def risk_score_from_hazards(hazards):
    """1 - S(final bin): probability of event by end of observed horizon."""
    S = torch.cumprod(1 - hazards, dim=0)
    return (1.0 - S[-1]).item()


def train_one(bag_tensors, y_bin_t, censored_t, idxs, n_epochs, device, seed=SEED):
    torch.manual_seed(seed)
    model = GatedAttentionMILSurvival().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    for _ in range(n_epochs):
        perm = idxs[torch.randperm(len(idxs))]
        for idx in perm:
            optimizer.zero_grad()
            hazards = model(bag_tensors[idx])
            loss = nll_surv_loss(hazards, y_bin_t[idx].item(), censored_t[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
    return model


def eval_loss_and_scores(model, bag_tensors, y_bin_t, censored_t, idxs):
    model.eval()
    total_loss = 0.0
    scores = {}
    with torch.no_grad():
        for idx in idxs:
            hazards = model(bag_tensors[int(idx)])
            loss = nll_surv_loss(hazards, y_bin_t[idx].item(), censored_t[idx])
            total_loss += loss.item()
            scores[int(idx)] = risk_score_from_hazards(hazards)
    return total_loss / len(idxs), scores


def inner_sweep(bag_tensors, times, events, censored_t, outer_train_idx, groups, device):
    """Inner GroupKFold sweep over CHECKPOINT_EPOCHS.

    Returns: best_epoch, {epoch: {patient_idx: inner-out-of-fold risk score}}
    """
    inner_groups = groups[outer_train_idx]
    n_inner_splits = min(INNER_K, len(np.unique(inner_groups)))
    gkf = GroupKFold(n_splits=n_inner_splits)

    val_loss_by_epoch = {e: [] for e in CHECKPOINT_EPOCHS}
    scores_by_epoch = {e: {} for e in CHECKPOINT_EPOCHS}

    for inner_train_pos, inner_val_pos in gkf.split(outer_train_idx, groups=inner_groups):
        inner_train_idx = torch.as_tensor(outer_train_idx[inner_train_pos])
        inner_val_idx = torch.as_tensor(outer_train_idx[inner_val_pos])

        # Fit outcome-dependent bin boundaries using inner-training patients only.
        _, edges = discretize_times(times[inner_train_idx.numpy()], events[inner_train_idx.numpy()])
        bins, _ = discretize_times(times, events, edges=edges)
        y_bin_t = torch.as_tensor(bins, dtype=torch.long, device=device)
        torch.manual_seed(SEED)
        model = GatedAttentionMILSurvival().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        model.train()
        epoch = 0
        for target_epoch in CHECKPOINT_EPOCHS:
            while epoch < target_epoch:
                perm = inner_train_idx[torch.randperm(len(inner_train_idx))]
                for idx in perm:
                    optimizer.zero_grad()
                    hazards = model(bag_tensors[idx])
                    loss = nll_surv_loss(hazards, y_bin_t[idx].item(), censored_t[idx])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    loss.backward()
                    optimizer.step()
                epoch += 1
            val_loss, val_scores = eval_loss_and_scores(
                model, bag_tensors, y_bin_t, censored_t, inner_val_idx
            )
            val_loss_by_epoch[target_epoch].append(val_loss)
            scores_by_epoch[target_epoch].update(val_scores)
            model.train()

    mean_val_loss = {e: np.mean(v) for e, v in val_loss_by_epoch.items()}
    best_epoch = min(mean_val_loss, key=mean_val_loss.get)
    return best_epoch, mean_val_loss, scores_by_epoch[best_epoch]


def bootstrap_ci(times, scores, events, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(times)
    boot_c = []
    for _ in range(n_boot):
        sample = rng.integers(0, n, n)
        t, s, e = times[sample], scores[sample], events[sample]
        if e.sum() == 0:
            continue
        try:
            boot_c.append(concordance_index(t, s, e))
        except ZeroDivisionError:
            continue
    if not boot_c:
        raise ValueError("No comparable pairs in bootstrap samples")
    lo, hi = np.percentile(boot_c, [2.5, 97.5])
    return lo, hi, len(boot_c)


def main():
    manifest, bags = load_pilot_bags()

    stage_ordinal_full = manifest["stage"].map(STAGE_ORDINAL)
    missing_stage = stage_ordinal_full.isna()
    if missing_stage.any():
        print(f"Dropping {missing_stage.sum()} patient(s) with unmapped/missing stage "
              f"(required as a Cox covariate): "
              f"{manifest.loc[missing_stage, 'patient_id'].tolist()}")
        keep_idx = np.flatnonzero(~missing_stage.to_numpy())
        manifest = manifest.iloc[keep_idx].reset_index(drop=True)
        bags = [bags[i] for i in keep_idx]
        stage_ordinal_full = stage_ordinal_full.iloc[keep_idx].reset_index(drop=True)

    n_patients = len(manifest)
    times = manifest["time"].to_numpy(dtype=np.float64)
    events = manifest["event"].to_numpy(dtype=np.int64)
    groups = manifest["tss_code"].to_numpy()
    stage_ordinal = stage_ordinal_full.to_numpy(dtype=np.float64)

    if manifest["patient_id"].duplicated().any() or manifest["tss_code"].isna().any():
        raise ValueError("Unique patients and nonmissing site groups are required")
    if not np.isfinite(times).all() or (times < 0).any() or not np.isin(events, [0, 1]).all():
        raise ValueError("Expected finite nonnegative times and binary events")
    n_sites = len(np.unique(groups))
    if n_sites < 3:
        raise ValueError("Nested site-grouped validation requires at least three sites")
    site_counts = pd.Series(groups).value_counts()
    print(f"n_patients={n_patients}, n_sites={n_sites}, patients/site: "
          f"min={site_counts.min()} max={site_counts.max()}")
    if site_counts.max() == 1:
        print("WARNING: every site has exactly 1 patient at pilot scale -- "
              "site-stratification is mechanically present (GroupKFold on "
              "tss_code) but UNEXERCISED: it degenerates to plain K-fold. "
              "Not a bug; expected until sites repeat at full 185-cohort scale.")


    device = resolve_device()
    print(f"device={device}")
    bag_tensors = [torch.from_numpy(b).to(device) for b in bags]
    censored_t = torch.from_numpy(1 - events).float().to(device)
    all_idx = np.arange(n_patients)

    n_outer_splits = min(OUTER_K, n_sites)
    outer_gkf = GroupKFold(n_splits=n_outer_splits)

    image_oof_score = np.full(n_patients, np.nan)
    stage_oof_score = np.full(n_patients, np.nan)
    combined_oof_score = np.full(n_patients, np.nan)
    chosen_epochs = []
    fold_ids = np.zeros(n_patients, dtype=int)
    fold_details = []

    for fold_i, (outer_train_pos, outer_test_pos) in enumerate(
        outer_gkf.split(all_idx, groups=groups), start=1
    ):
        outer_train_idx = all_idx[outer_train_pos]
        outer_test_idx = all_idx[outer_test_pos]
        outer_train_t = torch.as_tensor(outer_train_idx)

        best_epoch, mean_val_loss, inner_oof_scores = inner_sweep(
            bag_tensors, times, events, censored_t, outer_train_idx, groups, device
        )
        chosen_epochs.append(best_epoch)
        print(f"outer fold {fold_i}/{n_outer_splits}: "
              f"train={len(outer_train_idx)} test={len(outer_test_idx)} "
              f"best_epoch={best_epoch} inner_val_loss={mean_val_loss}")

        _, edges = discretize_times(times[outer_train_idx], events[outer_train_idx])
        bins, _ = discretize_times(times, events, edges=edges)
        y_bin_t = torch.as_tensor(bins, dtype=torch.long, device=device)
        fold_ids[outer_test_idx] = fold_i
        fold_details.append({"fold": fold_i, "best_epoch": best_epoch, "time_bin_edges": edges.tolist()})
        final_model = train_one(bag_tensors, y_bin_t, censored_t, outer_train_t, best_epoch, device)
        _, test_scores = eval_loss_and_scores(
            final_model, bag_tensors, y_bin_t, censored_t, torch.as_tensor(outer_test_idx)
        )
        for idx, score in test_scores.items():
            image_oof_score[idx] = score

        # Stage-alone Cox baseline.
        train_df = pd.DataFrame({
            "time": times[outer_train_idx],
            "event": events[outer_train_idx],
            "stage": stage_ordinal[outer_train_idx],
        })
        cox_stage = CoxPHFitter()
        cox_stage.fit(train_df, duration_col="time", event_col="event")
        test_df = pd.DataFrame({"stage": stage_ordinal[outer_test_idx]})
        stage_hazard = cox_stage.predict_partial_hazard(test_df).to_numpy()
        for i, idx in enumerate(outer_test_idx):
            stage_oof_score[idx] = -stage_hazard[i]

        # Combined (image risk score + stage) Cox, image score from
        # inner-out-of-fold predictions on outer-train (non-leaky).
        train_image_scores = np.array([inner_oof_scores[int(i)] for i in outer_train_idx])
        combined_train_df = pd.DataFrame({
            "time": times[outer_train_idx],
            "event": events[outer_train_idx],
            "stage": stage_ordinal[outer_train_idx],
            "image_risk": train_image_scores,
        })
        cox_combined = CoxPHFitter()
        cox_combined.fit(combined_train_df, duration_col="time", event_col="event")
        combined_test_df = pd.DataFrame({
            "stage": stage_ordinal[outer_test_idx],
            "image_risk": [image_oof_score[idx] for idx in outer_test_idx],
        })
        combined_hazard = cox_combined.predict_partial_hazard(combined_test_df).to_numpy()
        for i, idx in enumerate(outer_test_idx):
            combined_oof_score[idx] = -combined_hazard[i]

    for scores in (image_oof_score, stage_oof_score, combined_oof_score):
        if not np.isfinite(scores).all():
            raise ValueError("Every patient must have finite out-of-fold scores for all models")

    print(f"\nchosen epochs per outer fold: {chosen_epochs}")

    results = {}
    for name, score in [
        ("image_model", -image_oof_score),
        ("stage_alone_cox", stage_oof_score),
        ("image_plus_stage_cox", combined_oof_score),
    ]:
        c = concordance_index(times, score, events)
        lo, hi, n_valid_boot = bootstrap_ci(times, score, events)
        results[name] = (c, lo, hi)
        print(f"{name}: C-index={c:.3f}  95% bootstrap CI=[{lo:.3f}, {hi:.3f}]  "
              f"(n_boot_valid={n_valid_boot}/{N_BOOTSTRAP})")

    incremental = results["image_plus_stage_cox"][0] - results["stage_alone_cox"][0]
    print(f"\nincremental C-index (image+stage vs. stage-alone) = {incremental:+.3f}")

    # Save scores and provenance so statistics can be independently audited.
    output = Path(__file__).resolve().parent.parent / "results" / "validation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    predictions = manifest.copy()
    predictions["outer_fold"] = fold_ids
    predictions["image_risk"] = image_oof_score
    predictions["image_survival_score"] = -image_oof_score
    predictions["stage_survival_score"] = stage_oof_score
    predictions["combined_survival_score"] = combined_oof_score
    predictions.to_csv(output / "predictions.csv", index=False)
    source_paths = [Path(__file__), Path(__file__).with_name("train_mil_survival.py"), BAGS_DIR / "bags_manifest.csv"]
    report = {
        "status": "corrected_evaluation", "n_patients": n_patients,
        "n_sites": n_sites, "seed": SEED, "max_tiles": 4000,
        "checkpoint_epochs": CHECKPOINT_EPOCHS, "folds": fold_details,
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "pooled_c_index": {k: {"estimate": float(v[0]), "ci_lower": float(v[1]), "ci_upper": float(v[2])} for k, v in results.items()},
        "incremental_c_index": float(incremental),
        "limitations": ["Patient bootstrap conditional on fitted models; not training uncertainty", "Pooled scores from different folds may not be on comparable scales", "No independent external cohort"],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved evaluation artifacts to {output}")



if __name__ == "__main__":
    main()
