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
!python -m src.visualize  # Optional: regenerate the figures already exported by train_ml.
```

```python
import pandas as pd
display(pd.read_csv('artifacts/ml_benchmark_results.csv'))
for filename in ['fig1_grand_average_erp.png', 'fig2_roc_and_pr_curves.png',
                 'fig3_confusion_matrices.png', 'fig4_feature_importance.png']:
    display(Image('artifacts/' + filename))
```

Public downloads normally need no token. If Hugging Face requires authentication, use `huggingface_hub.login()` in the notebook before running the scripts; never commit a token. Internet access to Hugging Face is required for preprocessing.

## Pipeline behavior

**Events.** The repository's `trial_type` column contains `stimulus` or `response`; class identity comes from `event_type`. Whitespace and case are normalized. The mapping is **S5 / standard / frequent → 0, S6 / target → 1, S7 / distractor / deviant → discard**, following the trigger definitions in [Dzianok et al., Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8900497/#tbl3). Response, break, and run markers are excluded. Unknown stimulus codes or a missing retained class raise an error. Labels are never inferred from frequency or behavior. TSV onset seconds are authoritative; BrainVision annotations are used for recording boundaries. The source's TSV timing can differ from one-based BrainVision marker positions by one original sample. After discarding distractors, class proportions change: use the exported actual counts, not assumed percentages from a three-class paradigm.

**Preprocessing.** Each subject is downloaded with `snapshot_download(repo_type='dataset', allow_patterns=...)` into a unique subdirectory of `/content/temp_raw`. The repository revision is resolved to a commit SHA. Raw data, partial downloads, and local Hub cache metadata are deleted with `shutil.rmtree` in a `finally` block before the next subject. The Xet cache is disabled to avoid leaving raw chunks in a global cache. Downloads are file transfers restricted to one subject, not the Hugging Face Datasets streaming API. A hard runtime kill can leave an orphaned temporary directory; after confirming no preprocessing process is running, remove its `sub-XX-*` directory before restarting.

The default ordered selection is 30 scalp channels, including Fz, Pz, Fp1, and Fp2, to reduce Colab CPU/RAM use. Set `preprocessing.channels: null` to retain all EEG channels (127 in sub-01), with higher RAM, storage, and training costs. All subjects/runs must have the same saved channel order and time grid. BIDS channel types and explicit bad-channel flags are read when available; marked bad channels are interpolated using standard positions. Missing requested channels fail explicitly.

Processing applies a 50 Hz notch, 0.5–30 Hz band-pass, and anti-aliased resampling to 250 Hz with jointly resampled events. ICA is fit to a separate 1 Hz high-passed copy, by default the first five minutes with 20 components and decimation by two, and applied to the full recording. Real EOG is preferred; this dataset lacks dedicated EOG channels, so Fp1/Fp2 are explicit default proxies. Frontal EEG proxies can also reflect neural activity; review the logged correlations/exclusions and ERP plots. The procedure automates ocular artifact removal, not identification of every possible artifact. Average referencing follows ICA. Epochs span −200 to 1000 ms with −200 to 0 ms baseline correction; peak-to-peak rejection defaults to 150 µV and flat rejection to 0.1 µV. Epochs crossing annotated boundaries are rejected. All retained standard/target trials are eligible, including behavioral errors.

**Features.** Each channel contributes 26 features. Five additional a priori midline ROI summaries give **785 features for 30 channels**:

- Log10 absolute Welch band power in delta (0.5–4), theta (4–8), alpha (8–13), and beta (13–30 Hz). PSD is integrated with interpolated band edges; units before the log are µV². A one-second window provides coarse low-frequency resolution; interpret delta cautiously.
- Normalized spectral entropy within 0.5–30 Hz and amplitude-histogram Shannon entropy (32 within-trial bins), both using natural logarithms and returning 0 for constant/all-zero signals.
- `db4` discrete wavelet coefficient summaries at A4, D4, D3, D2, D1: mean absolute coefficient, standard deviation, and log10 mean-square energy. These fixed-size summaries represent the coefficients; the full coefficient vectors are not appended. Wavelet levels are not treated as exact physiological bands.
- Mean, positive peak, standard deviation, peak latency, and **signed area under the waveform** in the 300–600 ms P300 window. Amplitudes are µV, latency is ms after onset, and waveform area is µV·s (not classification ROC-AUC). The area uses trapezoidal integration on the saved time grid; a negative waveform has a negative area.
- The same five P300 statistics from the mean Fz/Cz/Pz waveform (`features.roi_channels`). ROI peaks/latencies are computed after averaging waveforms. This explicitly emphasizes midline/parietal P300 without selecting channels from validation labels. Set this list to `[]` for a channel selection that omits those channels.

Spectral, histogram, and wavelet features use 0–1000 ms; P300 uses its own window. Extraction runs in batches over a memory-mapped EEG array and uses the **saved** sampling frequency/time grid. Features use only their own trial; no cohort normalization or feature selection occurs here.

**Evaluation.** Five-fold `GroupKFold` assigns complete subjects to train/test sets; every trial is tested exactly once, and every fold must contain both classes. Standardization is inside each model pipeline and is fitted only on training folds. Four classifiers share the same splits: **shrinkage LDA** (`solver='lsqr', shrinkage='auto'`, default priors `[0.5, 0.5]`), **RBF SVM**, **Random Forest**, and **L2 Logistic Regression**. SVM, forest, and logistic regression use `class_weight='balanced'`. No test-fold tuning, random trial split, oversampling, probability calibration, or evaluation-driven threshold selection is performed. RBF SVM can take substantially longer on the full cohort; its cache and forest parallelism are bounded in config.

`ml_benchmark_results.csv` and `.json` report test-fold **mean and sample standard deviation (ddof=1)** for Accuracy, Balanced Accuracy, target-class Precision, Recall, F1, ROC-AUC, and Average Precision (AP). Values are fractions, not percentages; precision/F1 are zero when there are no predicted targets. AP summarizes the original precision–recall steps and is distinct from trapezoidal PR AUC. The JSON includes actual class counts and a constant-majority baseline for context. Fold SD is descriptive; overlapping training folds are not independent replicates, and SD is not a confidence interval.

Also exported: per-fold and per-subject metrics, pooled out-of-fold accuracy/F1, split subject lists, and trial-level held-out predictions. Inside each fold, trial metrics weight trials equally; `metrics_per_subject.csv` supports inspecting variation between people. ROC and PR figures average within-fold curves, avoiding cross-fold comparisons of uncalibrated margins. Scores are decision values for SVM/LDA/logistic regression and target probabilities for the forest. Confusion matrices contain only out-of-fold predictions. Final models are refit on all subjects after each model's CV and have no additional held-out score. A one-subject dry-run cannot support subject-wise CV.

ICA is an unsupervised, per-recording preprocessing step that uses the held-out subject's unlabeled recording. Evaluation therefore assumes a complete unlabeled recording is available for a new subject; it does not measure causal online decoding with a fixed preprocessing transform. Zero-phase filters also use future samples. Changing features/hyperparameters after viewing results requires a fresh independent evaluation or nested subject-wise CV.

**Visualization.** All figures use 300 DPI. Grand averages first average trials per subject/class, then equally weight subjects. Fz/Cz/Pz shading is a **pointwise 95% Student-t confidence interval across subject means**, not a trial-level interval or a simultaneous significance test. No population CI is drawn for one subject. The P300 window is shaded. ROC/PR shading denotes ±1 fold SD and is labeled separately. Confusion matrices show raw counts and row-normalized percentages for all four models, ordered by target F1 for exploratory comparison.

The top-20 feature importance figure summarizes **training-fold Random Forest mean decrease in impurity**, with fold SD bars and a complete CSV. It uses no held-out labels for feature selection. Impurity importance can favor high-cardinality variables and distribute credit among correlated features; it describes this model and does not prove causal neurophysiological relevance or channel significance. The feature ranking is not fed back into the benchmark. Feature distribution plots show subject/class means with equal subject weights.

## Configuration and restarts

Edit [`configs/config.yaml`](configs/config.yaml) for filters, channel list, ICA, artifact thresholds, feature windows, CV folds, and CPU limits. Paths are relative to the current working directory; run from the repository root. Every script accepts `--config PATH` and `--artifacts PATH`; preprocessing also accepts `--raw-dir PATH`, `--resume`, and `--overwrite`. Both module and direct-script entry points work:

```bash
python src/preprocess.py --dry-run --raw-dir ./temp_raw --resume
python src/preprocess.py --raw-dir ./temp_raw --resume
python src/features.py
python src/train_ml.py
python src/visualize.py              # Regenerate all figures from completed ML outputs.
python src/visualize.py --erp-only   # Only ERP, before extracting features/training.
```

For a smaller cohort, set `dataset.subjects: [sub-01, sub-02, sub-03, sub-04, sub-05]` and use no more CV folds than subjects. Each train/test fold must include both classes. Use a separate output folder for experiments with different preprocessing settings.

Completed subject checkpoints are reusable only when their source commit, preprocessing implementation version, preprocessing/event configuration, and seed match. Interrupted subject processing restarts that subject; completed checkpoints remain on Drive. `--overwrite` reprocesses the selected subjects. Dry-run refuses to replace a full cohort; choose another `--artifacts` path for that check. Consolidation writes a completion marker last and never loads the entire cohort into RAM. Changes to consolidated arrays invalidate previously extracted features; rerun features, training, and visualization after preprocessing. Old figures/models may remain until their stage is rerun; `metadata.json`, `feature_metadata.json`, and `training_metadata.json` record provenance and completion for their respective stages.

After updating an older checkout, rerun `src.features` to obtain the new P300 area/ROI schema. If preprocessing/event configuration changed (including aliases), use `--overwrite` or a fresh artifacts folder; incompatible checkpoints are deliberately rejected. Plot-only commands reject stale/incomplete ML outputs instead of combining runs. `training_metadata.json` is written only when all required figures and tables succeed. ML convergence warnings stop training, while ICA warnings/exclusions remain available for QC.

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
| `ml_benchmark_results.csv`, `ml_benchmark_results.json` | Four-model benchmark: mean/std for all six required metrics, plus AP |
| `fig1_grand_average_erp.png`, `fig1_grand_average_erp.csv` | Fz/Cz/Pz ERP, pointwise 95% CI, shaded P300 window and plotted data |
| `fig2_roc_and_pr_curves.png` | Side-by-side ROC and PR curves, with mean fold AUC/AP ± SD |
| `fig3_confusion_matrices.png` | All models: raw held-out counts and row percentages |
| `fig4_feature_importance.png` | Top 20 training-fold RF features, with channel/window labels |
| `feature_importance.csv`, `feature_importance_per_fold.csv` | Complete RF impurity importance summaries and fold values |
| `metrics_summary.csv`, `metrics_per_fold.csv`, `metrics_per_subject.csv`, `metrics_table.png` | Benchmark alias, fold/subject metrics and formatted results table |
| `cv_splits.json`, `oof_predictions.csv`, `training_metadata.json` | Subject separation, held-out scores, training provenance |
| `roc_curves.png`, `roc_curves.csv`, `pr_curves.csv` | ROC/PR image alias and mean curve data |
| `confusion_matrix_*.png`, `confusion_matrix_*.csv` | Held-out counts and row-normalized plots |
| `erp_sub01_check.png`, `grand_average_erp.png` | Fz/Pz preprocessing check and three-channel cohort ERP alias |
| `feature_distributions.png`, `feature_distributions.csv` | Subject-mean feature comparisons |

Subject checkpoints duplicate the processed data in consolidated `X.npy` for resumability; plan Drive capacity for both. Only one subject's raw recording is stored locally at a time. Model bundles can be loaded with `pickle` in a compatible Python environment; load only your own trusted artifacts. Use `bundle['pipeline'].predict(features)` with exactly `bundle['feature_names']` order and matching preprocessing/extraction settings.

## Local validation

Python 3.10+ is supported. Install `requirements.txt`, then run the offline integration checks:

```bash
python -m unittest discover -s tests -v
```

Checks cover event aliases/distractor exclusion, synthetic BrainVision ingestion, ICA/epoch timing/baseline behavior, raw cleanup on failure, consolidation, signed P300 area/ROI, actual training-fold scaler means, balanced models, subject-disjoint CV, recomputed held-out metrics, subject-level 95% CI, and every required output including PNG resolution. Synthetic results validate software behavior and do not estimate performance on real EEG. See [AUDIT.md](AUDIT.md) for findings and remaining interpretation limits.

## References

- [Dataset mirror](https://huggingface.co/datasets/duylplsn/oddball), [OpenNeuro source](https://openneuro.org/datasets/ds004621/versions/1.0.4), and [Dzianok et al. (2022)](https://doi.org/10.1093/gigascience/giac015).
- [MNE ICA tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html), [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download), and [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [LDA priors/shrinkage](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html), [precision–recall curves](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html), and [Student-t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html).
