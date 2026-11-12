"""Regression tests for score orientation and held-out outcome isolation."""
import sys
import contextlib
import io
import json
import tempfile

import pandas as pd
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from lifelines.utils import concordance_index

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import train_mil_survival as training
import validate_mil_survival as validation


class EvaluationTests(unittest.TestCase):
    def test_risk_requires_negation_for_concordance(self):
        times = np.array([1., 2., 3.])
        risks = np.array([validation.risk_score_from_hazards(torch.tensor([h])) for h in [.9, .5, .1]])
        self.assertEqual(concordance_index(times, -risks, np.ones(3)), 1.)
        self.assertEqual(concordance_index(times, risks, np.ones(3)), 0.)

    def test_held_out_times_do_not_change_training_edges(self):
        train_times = np.array([2., 4., 8., 16., 32.])
        events = np.ones(5)
        _, edges = training.discretize_times(train_times, events)
        saved_edges = edges.copy()
        bins, applied_edges = training.discretize_times(np.array([0., 1000000.]), np.ones(2), edges=edges)
        np.testing.assert_array_equal(edges, saved_edges)
        np.testing.assert_array_equal(applied_edges, saved_edges)
        np.testing.assert_array_equal(bins, [0, 3])

    def test_inner_sweep_fits_only_inner_training_outcomes(self):
        # Patient 6 is an outer-test patient and must never fit any inner edges.
        times = np.arange(1., 8.)
        events = np.ones(7)
        groups = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'd'])
        bags = [torch.zeros((2, 4)) for _ in times]
        calls = []
        original = training.discretize_times
        def record(t, e, **kwargs):
            if kwargs.get('edges') is None:
                calls.append(set(t))
            return original(t, e, **kwargs)
        with patch.object(validation, 'CHECKPOINT_EPOCHS', [1]), patch.object(validation, 'discretize_times', side_effect=record), patch.object(validation, 'GatedAttentionMILSurvival', side_effect=lambda: training.GatedAttentionMILSurvival(embed_dim=4, hidden_dim=2)):
            _, _, scores = validation.inner_sweep(bags, times, events, torch.zeros(7), np.arange(6), groups, torch.device('cpu'))
        self.assertEqual(set(scores), set(range(6)))
        self.assertEqual(len(calls), 3)
        for fit_times in calls:
            self.assertEqual(len(fit_times), 4)
            self.assertNotIn(7., fit_times)

    def test_bootstrap_accepts_all_observed_events(self):
        lo, hi, valid = validation.bootstrap_ci(np.arange(5.), np.arange(5.), np.ones(5), n_boot=50)
        self.assertEqual((lo, hi), (1., 1.))
        self.assertGreater(valid, 0)

    def test_evaluator_saves_complete_synthetic_run(self):
        # Exercise orchestration and artifact persistence, not clinical performance.
        class FakeCox:
            def fit(self, frame, **kwargs):
                return self
            def predict_partial_hazard(self, frame):
                return frame["stage"] + frame.get("image_risk", 0)

        count = 18
        manifest = pd.DataFrame({
            "patient_id": [f"synthetic-{i}" for i in range(count)],
            "time": np.arange(1., count + 1), "event": np.ones(count, dtype=int),
            "stage": ["Stage IA", "Stage IIA", "Stage IIB"] * 6,
            "tss_code": np.repeat(np.arange(6), 3), "n_tiles": np.full(count, 2),
        })
        bags = [np.full((2, 4), i / count, dtype=np.float32) for i in range(count)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "bags").mkdir()
            for name in ["validate_mil_survival.py", "train_mil_survival.py"]:
                (root / "src" / name).write_text((Path(__file__).resolve().parents[1] / "src" / name).read_text())
            manifest.to_csv(root / "bags" / "bags_manifest.csv", index=False)
            bootstrap = validation.bootstrap_ci
            with patch.object(validation, "__file__", str(root / "src/validate_mil_survival.py")), patch.object(validation, "BAGS_DIR", root / "bags"), patch.object(validation, "load_pilot_bags", return_value=(manifest, bags)), patch.object(validation, "CHECKPOINT_EPOCHS", [1]), patch.object(validation, "CoxPHFitter", FakeCox), patch.object(validation, "resolve_device", return_value=torch.device("cpu")), patch.object(validation, "GatedAttentionMILSurvival", side_effect=lambda: training.GatedAttentionMILSurvival(embed_dim=4, hidden_dim=2)), patch.object(validation, "bootstrap_ci", side_effect=lambda t, s, e: bootstrap(t, s, e, n_boot=30)), contextlib.redirect_stdout(io.StringIO()):
                validation.main()
            output = next((root / "results/validation").iterdir())
            predictions = pd.read_csv(output / "predictions.csv")
            report = json.loads((output / "metrics.json").read_text())
            self.assertEqual(len(predictions), count)
            self.assertEqual(predictions.outer_fold.nunique(), 5)
            np.testing.assert_allclose(predictions.image_survival_score, -predictions.image_risk)
            actual = concordance_index(predictions.time, predictions.image_survival_score, predictions.event)
            self.assertAlmostEqual(report["pooled_c_index"]["image_model"]["estimate"], actual)
            self.assertEqual(len(report["folds"]), 5)

    def test_training_partition_without_events_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'observed events'):
            training.discretize_times(np.array([1., 2.]), np.zeros(2))


if __name__ == '__main__':
    unittest.main()
