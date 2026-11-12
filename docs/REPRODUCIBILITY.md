# Reproduction and artifact provenance

## Lightweight audit

Run `python -m unittest discover -s tests -v` and `python src/summarize_evidence.py` after installing `requirements.txt`. The saved `results/evidence_audit.json` was generated locally from the manifest and historical diagnostic CSV. It contains a machine-dependent bag-availability count.

## Corrected full-cohort evaluation

Required files:

- `data/processed/bags/bags_manifest.csv`, with one unique patient per row and columns `patient_id,n_tiles,event,time,stage,tss_code`.
- `data/processed/bags/<patient_id>_bag.npz` for every manifest row, containing a finite, nonempty `embeddings` array with shape `(tiles, 2560)`.

Time is in days; event 1 denotes observed death, event 0 censoring. The current manifest has 179 rows and no missing stage. Validation drops unmapped/missing stage so all three models use the same cohort.

Run `python src/validate_mil_survival.py` from a configured environment. No external API calls or paid compute are launched by this command. The evaluator selects CUDA, then MPS, then CPU. It holds sampled bags on the chosen device, so memory requirements remain substantial. CPU execution of nested training is expensive; the audit session had no GPU available and did not complete this run.

Keep the emitted `predictions.csv` and `metrics.json` together. Source and manifest hashes identify the implementation and cohort; bag content hashes and a complete resolved environment lock are not currently emitted. Preserve original bags and record your environment separately (`python -m pip freeze`). The same seed does not guarantee bit-identical results across hardware.

## Rebuilding upstream data

Upstream scripts preserve the original staged workflow; they are not a single-command reproduction of the historical dataset:

1. `ingest.py` selects/downloads the pilot and fetches clinical records; `fetch_full_clinical.py` fetches full-cohort clinical data.
2. `tile_qc.py` processes pilot slides. `tile_remaining_batch.py tile` and `finalize` handle the remaining batch and budget-based selection. The batch tiler deletes downloaded working slides as part of its original disk-management workflow; inspect paths before running it.
3. `extract_embeddings.py` handles pilot embeddings; `extract_embeddings_lightning.py` handles selected batch slides. The Kaggle variant is an alternative historical execution path. Obtain authorized access to the Virchow2 model and configure credentials locally; never commit tokens.
4. `build_bags.py` joins embeddings to native clinical outcomes and writes bags plus the manifest. It needs the pilot and Lightning selection manifests and full clinical JSON.
5. `validate_mil_survival.py` performs the corrected nested evaluation. `train_mil_survival.py` alone fits all data and is not a validation run.

The historical notes describe 183 embedded cases and four outcome exclusions. The committed analysis manifest substantiates the final 179, but a complete frozen upstream data snapshot and exclusion ledger are not included. Live GDC queries and budget-based selection can change the cohort. Do not describe a newly downloaded cohort as an exact replication without checking IDs and preprocessing.

## Distribution

Raw slides, embeddings, bags, checkpoints, credentials, and intermediate batch parquets are excluded from Git. TCGA identifiers and analysis metadata are retained for reproducibility. Third-party datasets and model weights remain subject to their respective access and license terms. No new software license is assigned in this finalization; the owner should choose one before offering reuse under a specific license.
