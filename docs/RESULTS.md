# Results and evaluation audit

## What was recovered

The earlier working notes (`Cancer Project/docs/live_task_progress.md`, outside this repository) recorded a 179-patient nested-CV experiment and interpreted its results as negative. The saved manifest independently confirms 179 patients, 100 events, 79 censored observations, 27 submitting sites, and no missing stage values. Earlier code comments claiming 21 sites are stale.

The historical numbers are preserved in `results/historical_metrics.json`; they are transcribed summaries, not recomputed metrics. Original held-out predictions and fold checkpoints were not saved, preventing direct verification or complete statistical repair of that run.

## Confirmed evaluation defects

1. **Concordance direction.** `risk_score_from_hazards` returns event risk, where larger values imply earlier death. `lifelines.utils.concordance_index` expects a score increasing with survival. The old image-only calculation passed risk directly; both Cox comparisons already negated hazard. Negating the same historical image scores would algebraically turn 0.450 into approximately 0.550 (and its displayed interval into approximately 0.484–0.616). This is a sign-only illustration, not a corrected experiment: predictions are unavailable and leakage remains.
2. **Outcome-dependent preprocessing before CV.** The old evaluator estimated survival-bin quantiles from all patients, including held-out outcomes. The corrected code fits boundaries on each inner-training partition for model selection and each outer-training partition for final evaluation. Held-out times are assigned using those fixed boundaries.
3. **Bootstrap exclusion.** The old bootstrap skipped samples with all events observed. Such samples can contain valid comparable pairs. The corrected function retains them and rejects samples without comparable pairs.

The corrected evaluator also fails on non-finite training loss instead of silently skipping an update, and persists results for independent review.

## Subsampling diagnostic

Recomputing statistics from the saved diagnostic CSV gives Pearson r = 0.0208 (p = 0.7825) and Spearman rho = 0.0322 (p = 0.6684) between original tile count and historical held-out loss. Of 179 patients, 146 exceeded the 4,000-tile cap.

This is evidence of little observed association in this diagnostic, **not proof that subsampling has no effect**. Bag size is not a randomized sampling intervention; loss also depends on outcomes, censoring, and site. The diagnostic retrained outer models using historical fixed epoch choices and globally computed bins. It is not the corrected nested-CV experiment. `diagnose_subsample_artifact.py` is retained as a historical utility, not a recommended validation procedure.

## Remaining limitations

- The corrected full-cohort evaluation has not run. Tests establish specific code behavior, not model performance.
- The cohort query targets TCGA-PAAD. A pathology-level audit confirming exclusively PDAC histology is not supplied; the project name is not evidence of that restriction.
- A single cohort, model family, seed, sampling strategy, and endpoint cannot establish impossibility for image-based ML.
- Patient bootstrap intervals condition on fitted models; they do not capture model-training variability or site-cluster uncertainty.
- Pooled out-of-fold scores can have different scales between fitted models. Inspect within-fold discrimination before interpreting pooled metrics.
- The reported difference of −0.004 has no saved paired confidence interval. It is not an equivalence or non-inferiority test.
- Stage is represented as an ordinal covariate; this imposes a functional-form assumption. Stage 0/0a and IA share the earliest category.
- Epoch-grid selection was informed by earlier runs on this cohort, limiting its independence.
- Global image QC thresholds and historical pilot/batch processing are additional preprocessing limitations not resolved by correcting outcome binning.
- No independent external cohort, calibration analysis, or clinical-utility evaluation was completed.

## Defensible conclusion

The original experiment did not establish reliable survival prediction or incremental benefit over stage. Its reported performance requires reevaluation because of confirmed implementation defects. The project supports a transparent account of an inconclusive feasibility experiment, not a general claim that histology-based survival prediction cannot work.
