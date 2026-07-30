"""Step 5 (Attention-MIL survival model): gated-attention MIL pooling
(CLAM-style) over per-patient Virchow2 tile-embedding bags, trained with a
discrete-time negative-log-likelihood survival loss (Zadeh & Schmid 2020),
following the PORPOISE (Chen et al. 2022) precedent validated on TCGA-PAAD.

Milestone 1 (this script): 18-patient pilot only, plumbing/feasibility smoke
test — trains on the full pilot with no held-out split, and no C-index
reporting. Nested-CV validation is Step 6, not built here.

Design doc: docs/plans/2026-07-27-pdac-riskscope-design.md (Step 5 row).

Architecture and loss follow the reference CLAM/PORPOISE implementation
(Mahmood Lab, `github.com/mahmoodlab/PORPOISE` / `CLAM`):
- Attn_Net_Gated: gated attention (Ilse et al. 2018) over tile embeddings,
  compressed to hidden dim D before attention.
- NLLSurvLoss: discrete-time survival NLL with censored/uncensored terms,
  alpha-weighted per Zadeh & Schmid (2020) / PORPOISE's default alpha=0.4.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BAGS_DIR = DATA_DIR / "processed" / "bags"
BAGS_MANIFEST_PATH = BAGS_DIR / "bags_manifest.csv"
CHECKPOINT_DIR = DATA_DIR / "processed" / "checkpoints"

N_BINS = 4
EMBED_DIM = 2560  # Virchow2 class token (1280) + mean-pooled patch tokens (1280)
HIDDEN_DIM = 256
ALPHA = 0.4  # PORPOISE/CLAM default censored/uncensored balance term
LR = 2e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
SEED = 0


class AttnNetGated(nn.Module):
    """Gated attention (Ilse et al. 2018), as used in CLAM's Attn_Net_Gated."""

    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(dim, 1)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        raw_scores = self.attention_c(a * b)  # (n_tiles, 1)
        return raw_scores


class GatedAttentionMILSurvival(nn.Module):
    """CLAM-style gated-attention MIL pooling -> discrete-time hazard head."""

    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, n_bins=N_BINS, dropout=0.25):
        super().__init__()
        self.compress = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention = AttnNetGated(hidden_dim, dropout=dropout)
        self.hazard_head = nn.Linear(hidden_dim, n_bins)

    def forward(self, bag):
        """bag: (n_tiles, embed_dim) -> hazards: (n_bins,)"""
        h = self.compress(bag)  # (n_tiles, hidden_dim)
        raw_scores = self.attention(h)  # (n_tiles, 1)
        attn = torch.softmax(raw_scores, dim=0)  # (n_tiles, 1)
        z = (attn * h).sum(dim=0)  # (hidden_dim,)
        logits = self.hazard_head(z)  # (n_bins,)
        hazards = torch.sigmoid(logits)
        return hazards


def nll_surv_loss(hazards, y_bin, censored, alpha=ALPHA, eps=1e-7):
    """Discrete-time survival NLL, PORPOISE/CLAM formulation.

    hazards: (n_bins,) sigmoid hazard per discrete time bin
    y_bin: int, discretized time-bin index of event/censoring
    censored: 1 if censored (no event observed), 0 if event observed
    """
    S = torch.cumprod(1 - hazards, dim=0)  # (n_bins,)
    S_padded = torch.cat([torch.ones(1, dtype=S.dtype), S])  # (n_bins + 1,), S_padded[0] = 1

    uncensored_loss = -(1 - censored) * (
        torch.log(S_padded[y_bin].clamp(min=eps)) + torch.log(hazards[y_bin].clamp(min=eps))
    )
    censored_loss = -censored * torch.log(S_padded[y_bin + 1].clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    return loss


def discretize_times(times, events, n_bins=N_BINS):
    """Bin edges from quantiles of uncensored (event=1) times, PORPOISE convention.

    Returns bin index per patient (0..n_bins-1) and the bin edges.
    """
    event_times = times[events == 1]
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(event_times, quantiles)
    edges[0] = 0.0
    edges[-1] = times.max() + 1.0
    edges = np.unique(edges)
    if len(edges) - 1 < n_bins:
        # too few distinct event times for the requested bin count at this n
        edges = np.linspace(0.0, times.max() + 1.0, n_bins + 1)
    bin_idx = np.digitize(times, edges[1:-1], right=False)
    return bin_idx, edges


def load_pilot_bags():
    manifest = pd.read_csv(BAGS_MANIFEST_PATH)
    bags = []
    for _, row in manifest.iterrows():
        npz_path = BAGS_DIR / f"{row['patient_id']}_bag.npz"
        data = np.load(npz_path)
        bags.append(data["embeddings"].astype(np.float32))
    return manifest, bags


def train():
    torch.manual_seed(SEED)
    manifest, bags = load_pilot_bags()
    n_patients = len(manifest)

    times = manifest["time"].to_numpy(dtype=np.float64)
    events = manifest["event"].to_numpy(dtype=np.int64)
    y_bins, edges = discretize_times(times, events, N_BINS)
    print(f"n_patients={n_patients}, n_events={events.sum()}, n_censored={(1 - events).sum()}")
    print(f"time-bin edges (days): {np.round(edges, 1).tolist()}")
    print(f"bin assignment counts: {np.bincount(y_bins, minlength=N_BINS).tolist()}")

    model = GatedAttentionMILSurvival()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    bag_tensors = [torch.from_numpy(b) for b in bags]
    y_bin_t = torch.from_numpy(y_bins).long()
    censored_t = torch.from_numpy(1 - events).float()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(n_patients)
        epoch_loss = 0.0
        nan_hit = False
        for idx in perm:
            optimizer.zero_grad()
            hazards = model(bag_tensors[idx])
            loss = nll_surv_loss(hazards, y_bin_t[idx].item(), censored_t[idx])
            if torch.isnan(loss):
                nan_hit = True
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_patients
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:3d}  mean_loss={epoch_loss:.4f}  nan_hit={nan_hit}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "mil_survival_pilot.ckpt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
