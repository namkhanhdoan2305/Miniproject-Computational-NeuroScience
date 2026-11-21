"""Regression checks for per-model runs, inner curves, raw QC and optional GPU."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pickle
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src import backends, compare_ml, diagnostics, features, preprocess, train_ml, training
from src.common import load_config, read_json, write_json
from src.models import build_pipeline
from test_pipeline import write_brainvision, CHANNELS


class ModularWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(load_config())
        self.config['paths'] = {'artifacts': str(self.root / 'artifacts'), 'raw': str(self.root / 'raw')}
        self.config['preprocessing']['channels'] = CHANNELS
        self.config['preprocessing']['ica'].update(enabled=False, n_components=4, fit_max_seconds=25)
        self.config['training'].update(cv_folds=3, rf_estimators=8, n_jobs=1, device='cpu')
        self.config['qc']['enabled'] = False

    def tearDown(self):
        self.temp.cleanup()

    def make_cohort(self, count=6):
        """Use public checkpoint contracts to avoid repeating HF or preprocessing."""
        output = self.root / 'artifacts'
        subjects = [f'sub-{i+1:02d}' for i in range(count)]
        times = np.arange(-50, 251) / 250
        rng = np.random.default_rng(4)
        for subject in subjects:
            destination = output / 'subjects' / subject
            destination.mkdir(parents=True)
            y = np.tile([0, 0, 1], 6).astype(np.int8)
            x = rng.normal(0, 1e-6, (len(y), len(CHANNELS), len(times))).astype(np.float32)
            x[y == 1] += np.exp(-((times - .45) / .08) ** 2)[None, None, :] * 3e-6
            for name, value in [('X', x), ('y', y), ('ch_names', np.array(CHANNELS)), ('times', times)]:
                np.save(destination / (name + '.npy'), value, allow_pickle=False)
            write_json(destination / 'metadata.json', {'signature':'fixture', 'shape':list(x.shape), 'recordings':[]})
        preprocess.consolidate(subjects, output, 'fixture', 'local-synthetic', self.config)
        features.run(self.config)
        return output

    def test_independent_runs_compare_resume_and_reject_stale_results(self):
        output = self.make_cohort()
        with threadpool_limits(limits=1):
            train_ml.run(self.config, model='lda', device='cpu')
            self.assertFalse((output / 'ml_benchmark_results.csv').exists())
            with self.assertRaisesRegex(ValueError, 'Incomplete/missing'):
                compare_ml.run(self.config)
            for key in ['svm', 'random_forest', 'logistic_regression']:
                train_ml.run(self.config, model=key, device='cpu')
            summary = compare_ml.run(self.config)
        self.assertEqual(len(summary), 4)
        splits = read_json(output / 'cv_splits.json')
        outer_test = set(splits[0]['test_subjects'])
        for key in ['lda', 'svm', 'random_forest', 'logistic_regression']:
            folder = output / 'ml' / key
            metadata = training.validate_completed(folder, training.context(self.config)[-1])
            self.assertEqual(metadata['requested_device'], 'cpu')
            inner = read_json(folder / 'diagnostics.json')
            self.assertTrue(set(inner['train_subjects']).isdisjoint(inner['validation_subjects']))
            self.assertTrue(outer_test.isdisjoint(inner['train_subjects'] + inner['validation_subjects']))
            curve = pd.read_csv(folder / 'learning_curve.csv')
            self.assertEqual(curve.n_subjects.max(), len(inner['train_subjects']))
            self.assertFalse(inner['outer_test_used'])
            self.assertTrue((folder / 'learning_curve.png').exists())
            with patch.object(training, 'fit_checked', side_effect=AssertionError('resume must not fit')):
                train_ml.run(self.config, model=key, device='cpu', resume=True)
        forest = pd.read_csv(output / 'ml/random_forest/training_progress.csv')
        self.assertEqual(forest.n_trees.iloc[-1], 8)
        self.assertTrue(np.isfinite(forest).all().all())
        logistic = pd.read_csv(output / 'ml/logistic_regression/training_progress.csv')
        self.assertLess(logistic.train_objective.iloc[-1], logistic.train_objective.iloc[0])
        # Integrity guards reject edited results, not merely missing files.
        with (output / 'ml/svm/metrics.csv').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'Missing/changed'):
            compare_ml.run(self.config)
        changed = copy.deepcopy(self.config)
        changed['training']['svm_c'] = 2
        with self.assertRaisesRegex(ValueError, 'Stale/incompatible'):
            train_ml.run(changed, model='lda', resume=True)

    def test_raw_qc_only_preserves_cohort_and_matches_erp_trial_ids(self):
        header = write_brainvision(self.root / 'source', 'sub-01')
        fixture = self.root / 'source'
        files = [p.relative_to(fixture).as_posix() for p in fixture.rglob('*') if p.is_file()]
        self.config['dataset']['expected_subjects'] = 1
        output = self.root / 'artifacts'
        output.mkdir()
        # Existing cohort marker/data are opaque; QC-only must never read/replace them.
        (output / 'X.npy').write_bytes(b'keep-existing-cohort')
        (output / 'metadata.json').write_bytes(b'keep-existing-metadata')
        before = {p.name:p.read_bytes() for p in output.iterdir()}
        api = SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha='immutable-qc'),
                              list_repo_files=lambda *a, **kw: files)
        def download(**kwargs):
            for relative in kwargs['allow_patterns']:
                destination = Path(kwargs['local_dir']) / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fixture / relative, destination)
        with patch.object(preprocess, 'HfApi', return_value=api), patch.object(preprocess, 'snapshot_download', side_effect=download):
            preprocess.run(self.config, qc_only=True, subjects=['sub-01'])
        for name, data in before.items():
            self.assertEqual((output / name).read_bytes(), data)
        self.assertEqual(list((self.root / 'raw').iterdir()), [])
        folder = next((output / 'qc/sub-01').glob('*/metadata.json')).parent
        qc = read_json(folder / 'metadata.json')
        self.assertEqual(qc['stage_sfreq'], {'raw':500., 'filtered':250., 'post_ica':250.})
        accepted = set(qc['final_retained_event_indices'])
        for indices in qc['stage_event_indices'].values():
            self.assertTrue(accepted.issubset(indices))
        erps = pd.read_csv(folder / 'erp_stages_matched_trials.csv')
        for (_, _, label), table in erps.groupby(['stage', 'channel', 'class']):
            self.assertEqual(set(table.n_trials), {14 if label == 0 else 7})
        for file in ['report.html','psd_before_after.png','raw_filtered_ica_traces.png','p300_topography.png','event_retention.png']:
            self.assertTrue((folder / file).exists(), file)
        # QC metadata should contain only finite JSON-native values.
        json.dumps(qc, allow_nan=False)

    def test_ica_qc_records_real_exclusions_and_component_figures(self):
        header = write_brainvision(self.root / 'source', 'sub-01')
        self.config['preprocessing']['ica']['enabled'] = True
        folder = self.root / 'qc'
        with threadpool_limits(limits=1), patch('mne.preprocessing.ICA.find_bads_eog', return_value=([np.int64(0)], np.zeros(4))):
            preprocess.process_recording(header, self.config, folder)
        qc = read_json(folder / 'metadata.json')
        self.assertEqual(qc['report']['ica']['excluded'], [0])
        self.assertTrue((folder / 'ica_components_1.png').exists())
        self.assertTrue((folder / 'ica_sources.png').exists())

    def test_optional_gpu_fallback_and_explicit_gpu_failure(self):
        x = np.array([[0, 0], [1, 1], [1, 2], [2, 3], [3, 1], [4, 3]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        with patch.object(backends, 'gpu_status', return_value=(False, 'test: no GPU')):
            pipeline = build_pipeline('svm', self.config, 'auto').fit(x, y)
            self.assertEqual(pipeline[-1].backend_, 'cpu:sklearn')
            self.assertIn('no GPU', pipeline[-1].fallback_reason_)
            self.assertTrue(np.isfinite(pipeline.decision_function(x)).all())
            with self.assertRaisesRegex(RuntimeError, 'no GPU'):
                build_pipeline('svm', self.config, 'gpu').fit(x, y)
        # Exercise the GPU adapter contract using a stand-in, not hardware validation.
        class FakeGPU:
            fit_status_ = 0
            def fit(self, x, y):
                self.cpu = LogisticRegression(class_weight='balanced').fit(x, y)
                self.classes_, self.coef_, self.intercept_ = self.cpu.classes_, self.cpu.coef_, self.cpu.intercept_
                return self
            def predict(self, x): return self.cpu.predict(x)
            def predict_proba(self, x): return self.cpu.predict_proba(x)
        with patch.object(backends, 'gpu_status', return_value=(True, 'fake')), patch.object(backends, '_gpu_estimator', return_value=FakeGPU()):
            model = clone(build_pipeline('logistic_regression', self.config, 'auto')).fit(x, y)
            self.assertEqual(model[-1].backend_, 'gpu:cuml')
            np.testing.assert_allclose(expit(model.decision_function(x)), model.predict_proba(x)[:, 1])

    def test_logistic_callback_matches_weighted_l2_objective(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(70, 4))
        y = (x[:, 0] + rng.normal(size=70) > 1).astype(int)
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        c = .6
        trace, convergence = diagnostics.logistic_trace(x, y, x[:20], y[:20], c, 500)
        model = LogisticRegression(C=c, class_weight='balanced', tol=1e-8, max_iter=2000).fit(x, y)
        weights = np.where(y == 1, len(y)/(2*y.sum()), len(y)/(2*(y == 0).sum()))
        logits = model.decision_function(x)
        expected = np.average(np.logaddexp(0, logits) - y*logits, weights=weights) + np.sum(model.coef_**2)/(2*c*weights.sum())
        self.assertTrue(convergence['success'])
        self.assertAlmostEqual(trace.train_objective.iloc[-1], expected, places=7)

    def test_fit_checked_retries_then_refuses_nonconvergence(self):
        from sklearn.base import BaseEstimator, ClassifierMixin
        from sklearn.pipeline import make_pipeline
        import warnings
        class Slow(ClassifierMixin, BaseEstimator):
            def __init__(self, max_iter=1, never=False): self.max_iter, self.never = max_iter, never
            def fit(self, x, y):
                if self.max_iter < 2 or self.never:
                    warnings.warn('slow solver fixture', ConvergenceWarning)
                self.classes_ = np.unique(y)
                return self
        model = training.fit_checked(make_pipeline(StandardScaler(), Slow()), np.ones((4, 2)), np.array([0,1,0,1]))
        self.assertEqual(model[-1].max_iter, 2)
        self.assertEqual(model.fit_diagnostics_['attempts'], 2)
        with self.assertRaisesRegex(RuntimeError, 'did not converge'):
            training.fit_checked(make_pipeline(StandardScaler(), Slow(never=True)), np.ones((4, 2)), np.array([0,1,0,1]))

    def test_notebook_is_clean_and_cells_parse(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/colab_workflow.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertFalse(cell['outputs'])
                compile(''.join(cell['source']), '<notebook-cell>', 'exec')


if __name__ == '__main__':
    unittest.main()
