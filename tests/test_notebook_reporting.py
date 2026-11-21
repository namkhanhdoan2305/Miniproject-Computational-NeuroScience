"""Verify that the notebook reports saved measurements without changing estimands."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src import notebook
from src.common import write_json


def summary_fixture():
    rows = []
    for model, f1 in [('LDA', .61), ('SVM', .61), ('RandomForest', .3), ('LogisticRegression', .58)]:
        row = {'model': model, 'n_subjects': 42, 'n_trials': 23000, 'n_folds': 5,
               'backend': 'cpu:sklearn', 'fit_seconds_mean': 1.2, 'fit_seconds_std': .2, 'oof_f1': .99}
        for metric in notebook.METRIC_LABELS:
            row[metric + '_mean'], row[metric + '_std'] = .7, .04
        row['f1_mean'], row['f1_std'] = f1, .031
        rows.append(row)
    return pd.DataFrame(rows)


class NotebookReportingTests(unittest.TestCase):
    def test_table_preserves_fold_mean_sd_instead_of_pooled_oof(self):
        summary = summary_fixture().iloc[[3, 1, 0, 2]]
        before = summary.copy(deep=True)
        table = notebook.benchmark_table(summary)
        self.assertEqual(table.iloc[0]['Mô hình'], 'Logistic Regression L2')
        self.assertEqual(table.iloc[0]['F1 target'], '0.580 ± 0.031')
        self.assertEqual(table.iloc[1]['F1 target'], '0.610 ± 0.031')
        self.assertNotIn('0.990', table.to_string())
        pd.testing.assert_frame_equal(before, summary)

    def test_model_text_uses_saved_confusion_counts_and_diagnostic_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'ml' / 'lda'
            directory.mkdir(parents=True)
            write_json(directory / 'metadata.json', {'model': 'LDA'})
            summary_fixture().iloc[:1].to_csv(directory / 'metrics.csv', index=False)
            pd.DataFrame([{'fold': 1, 'n_test': 120, 'precision': .5, 'recall': .55, 'f1': .52,
                           'balanced_accuracy': .7, 'roc_auc': .8, 'average_precision': .6}]).to_csv(directory / 'metrics_per_fold.csv', index=False)
            pd.DataFrame([[80, 20], [9, 11]], index=['true_standard', 'true_target'],
                         columns=['pred_standard', 'pred_target']).to_csv(directory / 'confusion_matrix.csv')
            write_json(directory / 'diagnostics.json', {'status': 'complete', 'train_subjects': ['s1','s2'],
                                                         'validation_subjects': ['s3']})
            text = []
            with patch.object(notebook, '_text', side_effect=text.append), patch.object(notebook, '_table'), patch.object(notebook, 'show_images'):
                notebook.show_model('lda', temporary)
            content = '\n'.join(text)
            self.assertIn('11/20', content)
            self.assertIn('**9** target bị bỏ sót', content)
            self.assertIn('**20** standard', content)
            self.assertIn('2 người train, 1 người validation', content)
            self.assertNotIn('rationale', content)

    def test_comparison_handles_tied_f1_and_uses_observed_cohort_size(self):
        summary = summary_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            write_json(Path(temporary) / 'ml_benchmark_results.json', {'majority_accuracy_baseline': .86})
            text, tables = [], []
            with patch('src.visualize.validated_evaluation', return_value=(summary, None, None)), \
                 patch.object(notebook, '_text', side_effect=text.append), \
                 patch.object(notebook, '_table', side_effect=lambda caption, frame: tables.append(frame)), \
                 patch.object(notebook, 'show_images'):
                notebook.show_comparison(temporary)
            narrative = '\n'.join(text)
            self.assertIn('42 người, 23,000 trial', narrative)
            self.assertIn('Shrinkage LDA, SVM RBF', narrative)
            self.assertIn('đồng hạng', narrative)
            self.assertNotIn('Chênh lệch F1 trung bình với mô hình xếp sau', narrative)
            self.assertEqual(tables[0].iloc[-1]['Mô hình'], 'Random Forest')

if __name__ == '__main__':
    unittest.main()
