# PDAC-RiskScope

**A retrospective feasibility study of histology-based survival prediction in TCGA-PAAD.**

This project evaluates frozen Virchow2 slide-tile embeddings with gated-attention multiple-instance learning (MIL), comparing image-only survival prediction with pathologic stage and an image-plus-stage model.

**Conclusion:** the historical experiment did not establish useful predictive performance. A subsequent code audit identified evaluation defects, so the original metrics are provisional. This repository does **not** establish that image ML cannot predict PDAC survival. It documents an unsuccessful initial experiment, its limitations, and a corrected evaluation implementation.

Research code only; no validated patient risk scores or treatment recommendations are provided.

## Study at a glance

| Item | Description |
|---|---|
| Analysis cohort | 179 patients; 100 observed deaths; 79 censored outcomes |
| Submitting sites | 27 in the saved analysis manifest |
| Input | H&E whole-slide image tiles; frozen 2,560-dimensional Virchow2 embeddings |
| Model | Gated-attention MIL with four discrete-time hazard outputs |
| Sampling | At most 4,000 tiles per patient, sampled once at load time |
| Validation | Five outer and three inner site-grouped folds |
| Comparators | Ordinal pathologic-stage Cox; image risk plus stage Cox |
| Current status | Evaluation corrections tested; corrected full-cohort run pending |

The project was originally named `TMEimage-PDAC`; the configured GitHub remote may retain that name.

## Historical results — not corrected performance estimates

These values were transcribed from the project's August 10, 2026 working notes. The original per-patient predictions and fold checkpoints were not retained.

| Model | Reported pooled C-index | Reported 95% bootstrap interval |
|---|---:|---:|
| Image MIL | 0.450 | 0.384–0.516 |
| Stage Cox | 0.445 | 0.386–0.507 |
| Image + stage Cox | 0.441 | 0.382–0.506 |

The reported combined-minus-stage difference was −0.004. **Do not cite these values as corrected validation results.** The image-only score had its direction reversed for concordance, and time-bin boundaries used held-out outcomes. See [the audit and interpretation](docs/RESULTS.md).

## Run the lightweight checks

Python 3.9 was used for local verification. Create an environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/summarize_evidence.py
```

Tests use small synthetic inputs: no slides, model download, GPU, or training dataset is required. The evidence summary reads the committed cohort manifest and historical diagnostic CSV. Its `bags_available` field depends on your local files; zero is expected in a fresh clone.

## Reproduce the evaluation

A full run requires the original embedding bags, which are excluded from Git (approximately 26 GB locally). Model access, slide downloads, and embedding generation are separate preparation steps; see [reproduction details](docs/REPRODUCIBILITY.md).

```bash
python src/validate_mil_survival.py
```

The corrected evaluator learns time bins separately within each training partition and negates image risk when computing concordance. Completed runs save patient predictions, fold assignments, training-bin edges, metrics, source hashes, and configuration under `results/validation/<UTC timestamp>/`. Those generated runs are ignored by Git so they can be reviewed before publication.

A full corrected run has **not** been completed as part of this audit. No corrected metrics are claimed here.

## Repository map

- `src/`: ingestion, tiling, embedding extraction, bag construction, training, evaluation, and historical utilities.
- `tests/`: regression checks for score orientation, training-only binning, and bootstrap handling.
- `data/processed/bags/bags_manifest.csv`: analysis cohort and outcomes.
- `data/processed/subsample_diagnostic.csv`: historical per-patient diagnostic loss.
- `results/`: historical metrics and descriptive audit, explicitly labeled by provenance.
- `docs/`: methods, limitations, reproduction instructions, and finalization scope.

This repository closes the original prototype as a documented feasibility study. External validation, treatment lookup, and patient-facing risk reports are outside this release. A future corrected run may change the scientific interpretation.
