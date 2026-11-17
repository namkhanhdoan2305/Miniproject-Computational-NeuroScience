# Single-Trial EEG Classification: Three-Stimulus Oddball

CPU-oriented Python pipeline for signal preprocessing, handcrafted single-trial features, and conventional machine learning. The source is [duylplsn/oddball](https://huggingface.co/datasets/duylplsn/oddball): 42 BrainVision recordings with BIDS event tables. No deep learning or `datasets.load_dataset()` is used.

## Google Colab workflow

Select a **CPU** runtime. Clone into Colab's local disk, mount Drive, and link the persistent output directory before processing. Run these cells in order.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/namkhanhdoan2305/Miniproject-Computational-NeuroScience.git /content/Miniproject-Computational-NeuroScience
%cd /content/Miniproject-Computational-NeuroScience
%pip install -q -r requirements.txt
```

```python
from pathlib import Path

drive_artifacts = Path('/content/drive/MyDrive/oddball_artifacts')
drive_artifacts.mkdir(parents=True, exist_ok=True)
link = Path('artifacts')
if not link.exists() and not link.is_symlink():
    link.symlink_to(drive_artifacts, target_is_directory=True)
assert link.is_symlink() and link.resolve() == drive_artifacts.resolve(), (
    'artifacts must point to the chosen Drive folder; preserve/move existing outputs first.'
)
```

First process the **complete sub-01 recording** as a preprocessing check. `--dry-run` downloads real EEG and runs filtering, ICA, epoching, and rejection; it is not a simulated or download-free mode.

```bash
!python -m src.preprocess --dry-run --resume
```

```python
from IPython.display import Image, display
display(Image('artifacts/erp_sub01_check.png'))
```

Inspect the Fz/Pz class averages and `artifacts/subjects/sub-01/metadata.json` (class counts, rejection reasons, ICA exclusions, EOG correlations, and convergence warnings). Then process the full cohort, reusing sub-01, and execute the remaining stages:

```bash
!python -m src.preprocess --resume
!python -m src.features
!python -m src.train_ml
!python -m src.visualize
```

```python
import pandas as pd
display(pd.read_csv('artifacts/metrics_summary.csv'))
display(Image('artifacts/grand_average_erp.png'))
display(Image('artifacts/roc_curves.png'))
```

Public downloads normally need no token. If Hugging Face requires authentication, use `huggingface_hub.login()` in the notebook before running the scripts; never commit a token. Internet access to Hugging Face is required for preprocessing.

## Pipeline behavior

**Events.** The repository's `trial_type` column contains `stimulus` or `response`; class identity comes from `event_type`. Whitespace and case are normalized. The mapping is **S5 → standard (0), S6 → target (1), S7 → distractor (discard)**, following [Dzianok et al., Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8900497/#tbl3). Response, break, and run markers are excluded. Unknown stimulus codes or a missing retained class raise an error. Labels are never inferred from frequency or behavior. TSV onset seconds are authoritative; BrainVision annotations are used for recording boundaries. The source's TSV timing can differ from one-based BrainVision marker positions by one original sample.

**Preprocessing.** Each subject is downloaded with `snapshot_download(repo_type='dataset', allow_patterns=...)` into a unique subdirectory of `/content/temp_raw`. The repository revision is resolved to a commit SHA. Raw data, partial downloads, and local Hub cache metadata are deleted with `shutil.rmtree` in a `finally` block before the next subject. The Xet cache is disabled to avoid leaving raw chunks in a global cache. Downloads are file transfers restricted to one subject, not the Hugging Face Datasets streaming API. A hard runtime kill can leave an orphaned temporary directory; after confirming no preprocessing process is running, remove its `sub-XX-*` directory before restarting.

The default ordered selection is 30 scalp channels, including Fz, Pz, Fp1, and Fp2, to reduce Colab CPU/RAM use. Set `preprocessing.channels: null` to retain all EEG channels (127 in sub-01), with higher RAM, storage, and training costs. All subjects/runs must have the same saved channel order and time grid. BIDS channel types and explicit bad-channel flags are read when available; marked bad channels are interpolated using standard positions. Missing requested channels fail explicitly.

Processing applies a 50 Hz notch, 0.5–30 Hz band-pass, and anti-aliased resampling to 250 Hz with jointly resampled events. ICA is fit to a separate 1 Hz high-passed copy, by default the first five minutes with 20 components and decimation by two, and applied to the full recording. Real EOG is preferred; this dataset lacks dedicated EOG channels, so Fp1/Fp2 are explicit default proxies. Frontal EEG proxies can also reflect neural activity; review the logged correlations/exclusions and ERP plots. The procedure automates ocular artifact removal, not identification of every possible artifact. Average referencing follows ICA. Epochs span −200 to 1000 ms with −200 to 0 ms baseline correction; peak-to-peak rejection defaults to 150 µV and flat rejection to 0.1 µV. Epochs crossing annotated boundaries are rejected. All retained standard/target trials are eligible, including behavioral errors.

**Features.** Each channel contributes 25 features by default (750 for 30 channels):

- Log10 absolute Welch band power in delta (0.5–4), theta (4–8), alpha (8–13), and beta (13–30 Hz). PSD is integrated with interpolated band edges; units before the log are µV². A one-second window provides coarse low-frequency resolution; interpret delta cautiously.
- Normalized spectral entropy within 0.5–30 Hz and amplitude-histogram Shannon entropy (32 within-trial bins), both using natural logarithms and returning 0 for constant/all-zero signals.
- `db4` discrete wavelet coefficient summaries at A4, D4, D3, D2, D1: mean absolute coefficient, standard deviation, and log10 mean-square energy. These fixed-size summaries represent the coefficients; the full coefficient vectors are not appended. Wavelet levels are not treated as exact physiological bands.
- Mean, positive peak, standard deviation, and peak latency in the 300–600 ms P300 window. Amplitudes are µV and latency is ms after onset.

Spectral, histogram, and wavelet features use 0–1000 ms; P300 uses its own window. Extraction runs in batches over a memory-mapped EEG array and uses the **saved** sampling frequency/time grid. Features use only their own trial; no cohort normalization or feature selection occurs here.

**Evaluation.** Five-fold `GroupKFold` assigns complete subjects to train/test sets. Standardization is inside each model pipeline and is fitted only on training folds. Models are a balanced linear SVM (`LinearSVC`), balanced Random Forest, and shrinkage LDA. Linear SVM keeps CPU costs practical; no test-fold tuning or random trial split is performed. LDA retains empirical training priors; SVM/forest use class weighting. No oversampling is applied.

Metrics report test-fold mean and sample standard deviation for accuracy, target-class F1, ROC-AUC, and balanced accuracy. Also included are pooled out-of-fold accuracy/F1, per-fold metrics, split subject lists, and trial-level held-out predictions. ROC figures average curves across folds; AUC is computed within each fold, avoiding cross-fold comparisons of uncalibrated SVM margins. Scores are decision values for SVM/LDA and target probabilities for the forest. Confusion matrices contain only out-of-fold predictions. Exported final models are refit on all subjects after evaluation and have no additional held-out score. A one-subject dry-run cannot support subject-wise CV.

ICA is an unsupervised, per-recording preprocessing step that uses the held-out subject's unlabeled recording. Evaluation therefore assumes a complete unlabeled recording is available for a new subject; it does not measure causal online decoding with a fixed preprocessing transform. Zero-phase filters also use future samples. Changing features/hyperparameters after viewing results requires a fresh independent evaluation or nested subject-wise CV.

**Visualization.** Grand averages first average trials per subject/class, then equally weight subjects. Shading is ±1 SEM across subjects and is omitted for the one-subject check. Feature distribution plots show subject/class means, so trial count imbalance does not determine subject weights. They are descriptive; they do not select features for training.

## Configuration and restarts

Edit [`configs/config.yaml`](configs/config.yaml) for filters, channel list, ICA, artifact thresholds, feature windows, CV folds, and CPU limits. Paths are relative to the current working directory; run from the repository root. Every script accepts `--config PATH` and `--artifacts PATH`; preprocessing also accepts `--raw-dir PATH`, `--resume`, and `--overwrite`. Both module and direct-script entry points work:

```bash
python src/preprocess.py --dry-run --raw-dir ./temp_raw --resume
python src/features.py
python src/train_ml.py
python src/visualize.py --erp-only
```

For a smaller cohort, set `dataset.subjects: [sub-01, sub-02, sub-03, sub-04, sub-05]` and use no more CV folds than subjects. Each train/test fold must include both classes. Use a separate output folder for experiments with different preprocessing settings.

Completed subject checkpoints are reusable only when their source commit, preprocessing/event configuration, and seed match. Interrupted subject processing restarts that subject; completed checkpoints remain on Drive. `--overwrite` reprocesses the selected subjects. Dry-run refuses to replace a full cohort; choose another `--artifacts` path for that check. Consolidation writes a completion marker last and never loads the entire cohort into RAM. Changes to consolidated arrays invalidate previously extracted features; rerun features, training, and visualization after preprocessing. Old figures/models may remain until their stage is rerun; `metadata.json`, `feature_metadata.json`, and `training_metadata.json` record provenance and completion for their respective stages.

## Outputs

All generated data and results are under `artifacts/`:

| Output | Meaning |
| --- | --- |
| `X.npy` | Float32 EEG in **volts**, shape `(trials, channels, samples)`; 301 samples by default |
| `y.npy`, `groups.npy` | Int8 labels 0/1; Unicode subject IDs, one per trial |
| `ch_names.npy`, `times.npy` | Ordered Unicode channel names; epoch time axis in seconds |
| `metadata.json`, `subjects/sub-XX/` | Cohort provenance and restartable per-subject processed checkpoints/QC |
| `features.npy`, `feature_names.npy`, `feature_metadata.json` | Float32 feature matrix, ordered names, extraction settings/source identity |
| `models/*.pkl` | Training-fold and final model bundles with pipelines and feature schema |
| `metrics_summary.csv`, `metrics_per_fold.csv`, `metrics_table.png` | Accuracy, F1, ROC-AUC and balanced accuracy results |
| `cv_splits.json`, `oof_predictions.csv`, `training_metadata.json` | Subject separation, held-out scores, training provenance |
| `roc_curves.png`, `roc_curves.csv` | Mean fold ROC curves and their plotted data |
| `confusion_matrix_*.png`, `confusion_matrix_*.csv` | Held-out counts and row-normalized plots |
| `erp_sub01_check.png`, `grand_average_erp.png` | Fz/Pz preprocessing check and subject-weighted cohort ERP |
| `feature_distributions.png`, `feature_distributions.csv` | Subject-mean feature comparisons |

Subject checkpoints duplicate the processed data in consolidated `X.npy` for resumability; plan Drive capacity for both. Only one subject's raw recording is stored locally at a time. Model bundles can be loaded with `pickle` in a compatible Python environment; load only your own trusted artifacts. Use `bundle['pipeline'].predict(features)` with exactly `bundle['feature_names']` order and matching preprocessing/extraction settings.

## Local validation

Python 3.10+ is supported. Install `requirements.txt`, then run the offline integration checks:

```bash
python -m unittest discover -s tests -v
```

Checks cover event-code mapping, actual synthetic BrainVision ingestion, ICA/epoch timing/baseline behavior, raw cleanup on failure, consolidation, finite feature extraction, subject-disjoint CV, and exported evaluation artifacts. Synthetic results validate software behavior and do not estimate performance on real EEG.

## References

- [Dataset mirror](https://huggingface.co/datasets/duylplsn/oddball), [OpenNeuro source](https://openneuro.org/datasets/ds004621/versions/1.0.4), and [Dzianok et al. (2022)](https://doi.org/10.1093/gigascience/giac015).
- [MNE ICA tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html), [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download), and [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
