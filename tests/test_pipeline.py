"""Offline behavioral checks; real data are never downloaded by this test suite."""

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

import mne
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import features, preprocess, train_ml, visualize
from src.common import load_config, load_epochs, read_json


CHANNELS = ["Fp1", "Fp2", "Fz", "Pz", "Cz", "O1"]


def write_brainvision(root, subject, seed=1):
    """Write a small real-format, vectorized BrainVision recording with BIDS events."""
    folder = root / subject / "eeg"
    folder.mkdir(parents=True)
    stem = f"{subject}_task-oddball"
    sfreq, seconds = 500, 46
    rng = np.random.default_rng(seed)
    # BrainVision values are microvolts; MNE must convert them into volts.
    data = rng.normal(0, 0.7, (len(CHANNELS), sfreq * seconds))
    onsets = np.arange(2, 43, 1.5)
    codes = ["S  5", "S  5", "S  6", "S  7"] * 7
    rows, markers = [], []
    for onset, code in zip(onsets, codes):
        rows.append({"onset": onset, "duration": "n/a", "event_type": code, "trial_type": "stimulus"})
        markers.append(f"Mk{len(markers)+1}=Stimulus,{code},{round(onset * sfreq)+1},1,0")
        if code == "S  6":
            time = np.arange(data.shape[-1]) / sfreq
            data[3] += 7 * np.exp(-0.5 * ((time - onset - 0.45) / 0.06) ** 2)
        rows.append({"onset": onset + 0.7, "duration": "n/a", "event_type": "S  1", "trial_type": "response"})
    data.astype("<f4").tofile(folder / f"{stem}_eeg.eeg")
    header = "\n".join([
        "Brain Vision Data Exchange Header File Version 1.0", "", "[Common Infos]",
        f"DataFile={stem}_eeg.eeg", f"MarkerFile={stem}_eeg.vmrk", "DataFormat=BINARY",
        "DataOrientation=VECTORIZED", f"NumberOfChannels={len(CHANNELS)}",
        f"DataPoints={data.shape[-1]}", "SamplingInterval=2000", "", "[Binary Infos]",
        "BinaryFormat=IEEE_FLOAT_32", "", "[Channel Infos]",
    ] + [f"Ch{i}={ch},,1,µV" for i, ch in enumerate(CHANNELS, 1)])
    (folder / f"{stem}_eeg.vhdr").write_text(header, encoding="utf-8")
    (folder / f"{stem}_eeg.vmrk").write_text("\n".join([
        "Brain Vision Data Exchange Marker File, Version 1.0", "", "[Common Infos]",
        f"DataFile={stem}_eeg.eeg", "", "[Marker Infos]", *markers]), encoding="utf-8")
    pd.DataFrame(rows).to_csv(folder / f"{stem}_events.tsv", sep="\t", index=False)
    pd.DataFrame({"name": CHANNELS, "type": "EEG", "units": "uV"}).to_csv(
        folder / f"{stem}_channels.tsv", sep="\t", index=False)
    return folder / f"{stem}_eeg.vhdr"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(load_config())
        self.config["paths"] = {"artifacts": str(self.root / "artifacts"), "raw": str(self.root / "temp_raw")}
        self.config["preprocessing"]["channels"] = CHANNELS
        self.config["preprocessing"]["ica"].update({"n_components": 4, "fit_max_seconds": 25})
        self.config["training"].update({"cv_folds": 3, "rf_estimators": 8, "n_jobs": 1})

    def tearDown(self):
        self.temp.cleanup()

    def test_event_mapping_drops_distractors_and_responses(self):
        raw = mne.io.RawArray(np.zeros((1, 5000)), mne.create_info(["Pz"], 500, "eeg"), verbose=False)
        path = self.root / "events.tsv"
        pd.DataFrame({"onset": [1, 2, 3, 4], "event_type": [" S  5 ", "s6", "S 7", "S 1"],
                      "trial_type": ["stimulus", "stimulus", "stimulus", "response"]}).to_csv(path, sep="\t", index=False)
        events, counts = preprocess.read_events(path, raw, self.config)
        np.testing.assert_array_equal(events[:, [0, 2]], [[500, 1], [1000, 2]])
        self.assertEqual(counts, {"standard": 1, "target": 1, "distractor": 1})
        table = pd.read_csv(path, sep="\t")
        table.loc[2, "event_type"] = "S 99"
        table.to_csv(path, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "Unmapped stimulus"):
            preprocess.read_events(path, raw, self.config)

    def test_brainvision_resampling_baseline_and_p300_timing(self):
        header = write_brainvision(self.root, "sub-01")
        self.config["preprocessing"]["ica"]["enabled"] = False
        x, y, channels, times, report = preprocess.process_recording(header, self.config)
        self.assertEqual(x.shape, (21, 6, 301))
        self.assertEqual(channels, CHANNELS)
        self.assertEqual(report["event_counts"]["distractor"], 7)
        self.assertEqual(int(y.sum()), 7)
        baseline = (times >= -0.2) & (times <= 0)
        np.testing.assert_allclose(x[..., baseline].mean(axis=-1), 0, atol=1e-12)
        response = x[y == 1, 3].mean(axis=0)
        self.assertAlmostEqual(times[np.argmax(response)], 0.45, delta=0.025)
        self.assertGreater(response.max(), 3e-6)
        self.assertLess(response.max(), 10e-6)

    def test_ica_runs_and_records_exclusions(self):
        header = write_brainvision(self.root, "sub-01")
        with threadpool_limits(limits=1), patch.object(mne.preprocessing.ICA, "find_bads_eog",
                return_value=([np.int64(0)], np.zeros(4))):
            x, _, _, _, report = preprocess.process_recording(header, self.config)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(report["ica"]["enabled"])
        self.assertEqual(report["ica"]["n_components"], 4)
        self.assertTrue(report["ica"]["using_eeg_proxy"])
        self.assertEqual(set(report["ica"]["eog_scores"]), {"Fp1", "Fp2"})
        self.assertEqual(report["ica"]["excluded"], [0])
        json.dumps(report, allow_nan=False)  # Nonempty exclusions must serialize too.

    def test_recording_boundaries_are_not_accepted_as_continuous_epochs(self):
        header = write_brainvision(self.root, "sub-01")
        marker = header.with_suffix(".vmrk")
        with marker.open("a", encoding="utf-8") as handle:
            handle.write("\nMk29=New Segment,,2651,0,0\n")  # 5.3 s, within the first target epoch.
        self.config["preprocessing"]["ica"]["enabled"] = False
        x, y, _, _, report = preprocess.process_recording(header, self.config)
        self.assertEqual(len(x), 20)
        self.assertEqual(int(y.sum()), 6)
        self.assertEqual(report["drop_reasons"]["BAD boundary"], 1)

    def test_raw_cleanup_on_download_and_processing_failures(self):
        for fail_download in (True, False):
            def download(**kwargs):
                Path(kwargs["local_dir"]).mkdir()
                (Path(kwargs["local_dir"]) / "partial.eeg").write_bytes(b"raw")
                if fail_download:
                    raise RuntimeError("Download failure")
            with patch.object(preprocess, "snapshot_download", side_effect=download):
                with self.assertRaises(RuntimeError):
                    with preprocess.downloaded_subject("sub-01", ["sub-01/a.eeg"], "test", self.config):
                        raise RuntimeError("Processing failure")
            self.assertEqual(list((self.root / "temp_raw").iterdir()), [])

    def test_features_are_finite_and_identify_known_signal(self):
        times = np.arange(-50, 251) / 250
        signal = np.sin(2 * np.pi * 6 * times) * 5e-6
        x = np.stack([signal, np.zeros_like(signal)])[:, None, :]
        values, names = features.extract_batch(x, 250, times, ["Pz"], self.config["features"])
        self.assertEqual(values.shape, (2, 25))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(values[0, names.index("Pz__log10_power_theta")],
                           values[0, names.index("Pz__log10_power_alpha")] + 2)
        self.assertEqual(values[1, names.index("Pz__spectral_entropy")], 0)
        self.assertEqual(values[1, names.index("Pz__shannon_entropy")], 0)

    def test_full_offline_workflow_resume_and_subject_disjoint_exports(self):
        fixture = self.root / "source"
        subjects = ["sub-01", "sub-02", "sub-03"]
        for index, subject in enumerate(subjects):
            write_brainvision(fixture, subject, index + 1)
        files = [path.relative_to(fixture).as_posix() for path in fixture.rglob("*") if path.is_file()]
        self.config["dataset"]["expected_subjects"] = 3
        self.config["preprocessing"]["ica"]["enabled"] = False
        api = SimpleNamespace(dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="immutable-test-sha"),
                              list_repo_files=lambda *args, **kwargs: files)
        downloaded = []
        def download(**kwargs):
            selected = kwargs["allow_patterns"]
            selected_subjects = {Path(path).parts[0] for path in selected}
            self.assertEqual(len(selected_subjects), 1)
            self.assertEqual(len(list((self.root / "temp_raw").iterdir())), 1)
            downloaded.extend(selected_subjects)
            for relative in selected:
                target = Path(kwargs["local_dir"]) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fixture / relative, target)
        with patch.object(preprocess, "HfApi", return_value=api), patch.object(preprocess, "snapshot_download", side_effect=download):
            preprocess.run(self.config, dry_run=True)
            self.assertTrue((self.root / "artifacts" / "erp_sub01_check.png").exists())
            preprocess.run(self.config, resume=True)
            self.assertEqual(downloaded, subjects)
            with self.assertRaisesRegex(ValueError, "cannot replace a cohort"):
                preprocess.run(self.config, dry_run=True, resume=True)
        self.assertEqual(list((self.root / "temp_raw").iterdir()), [])
        output = self.root / "artifacts"
        x, y, groups, _, _, metadata = load_epochs(output)
        self.assertEqual(x.shape, (63, 6, 301))
        self.assertEqual(metadata["revision"], "immutable-test-sha")
        self.assertEqual(set(groups), set(subjects))
        features.run(self.config)
        with threadpool_limits(limits=1):
            metrics = train_ml.run(self.config)
        self.assertEqual(set(metrics.model), {"SVM", "RandomForest", "LDA"})
        self.assertTrue(np.isfinite(metrics.select_dtypes(include="number")).all().all())
        for split in read_json(output / "cv_splits.json"):
            self.assertFalse(set(split["train_subjects"]) & set(split["test_subjects"]))
        oof = pd.read_csv(output / "oof_predictions.csv")
        self.assertEqual(len(oof), len(y) * 3)
        self.assertFalse(oof.duplicated(["model", "trial_index"]).any())
        for name in ("svm", "randomforest", "lda"):
            matrix = pd.read_csv(output / f"confusion_matrix_{name}.csv", index_col=0)
            self.assertEqual(matrix.to_numpy().sum(), len(y))
            with (output / "models" / f"{name}_final.pkl").open("rb") as handle:
                bundle = pickle.load(handle)
            self.assertEqual(bundle["pipeline"].predict(np.load(output / "features.npy")[:2]).shape, (2,))
            self.assertTrue((output / f"confusion_matrix_{name}.png").exists())
        visualize.plot_erp(output, output / "grand_average_erp.png")
        visualize.plot_distributions(output, self.config["visualization"])
        self.assertTrue((output / "feature_distributions.png").exists())
        # Reprocessing changes the dataset identity; old features must be rejected.
        preprocess.consolidate(subjects, output, metadata["preprocessing_signature"], metadata["revision"], self.config)
        with self.assertRaisesRegex(ValueError, "stale"):
            features.load_features(output)

    def test_insufficient_subjects_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one-subject dry-run"):
            train_ml.group_splits(np.ones((10, 2)), np.tile([0, 1], 5), np.repeat("sub-01", 10), 5)


if __name__ == "__main__":
    unittest.main()
