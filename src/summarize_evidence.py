"""Summarize saved cohort/diagnostic evidence without retraining or downloads."""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent


def summarize():
    manifest = pd.read_csv(ROOT / 'data/processed/bags/bags_manifest.csv')
    diagnostic = pd.read_csv(ROOT / 'data/processed/subsample_diagnostic.csv')
    if manifest.patient_id.duplicated().any() or diagnostic.patient_id.duplicated().any():
        raise ValueError('Duplicate patient IDs')
    if set(manifest.patient_id) != set(diagnostic.patient_id):
        raise ValueError('Diagnostic and manifest cohorts differ')
    x, y = diagnostic.n_tiles, diagnostic.held_out_loss
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        'status': 'descriptive_audit_of_saved_artifacts_not_corrected_validation',
        'patients': len(manifest), 'events': int(manifest.event.sum()),
        'sites': int(manifest.tss_code.nunique()),
        'missing_stage': int(manifest.stage.isna().sum()),
        'bags_available': sum((ROOT / 'data/processed/bags' / f'{p}_bag.npz').exists() for p in manifest.patient_id),
        'subsampled_patients': int((x > 4000).sum()),
        'tile_count_vs_historical_loss': {
            'pearson_r': float(pr), 'pearson_p': float(pp),
            'spearman_r': float(sr), 'spearman_p': float(sp),
        },
        'interpretation': 'Correlation does not establish whether subsampling harms predictive performance; a controlled sampling comparison is required.',
    }


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2))
