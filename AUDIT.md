# Biomedical signal processing and ML audit

Audited on 2026-11-17 against the requested binary oddball classification checklist. This is a software/methodology audit, not a claim of validated clinical performance.

| Area | Finding in the original implementation | Resolution |
| --- | --- | --- |
| Downloads and cleanup | Already used per-subject `snapshot_download`, `finally` cleanup and disposable Hub cache | Retained; failure/success cleanup and one-subject-at-a-time behavior tested |
| Event classes | S5/S6/S7 and named standard/target/distractor were correct; `frequent` alias absent | Added alias; tests ensure responses/distractors never enter the binary arrays |
| Signal parameters | 50 Hz notch, 0.5–30 Hz bandpass, 250 Hz resampling, ICA, −0.2–1 s epochs and baseline were present | Retained; timing, volts conversion, baseline, nonempty ICA exclusions and boundary rejection verified |
| EOG references | Explicit EOG channels could be dropped when not typed as EOG in a sidecar | Explicit overrides now set EOG type before selection, preserve the reference, and prohibit its use as a feature channel |
| ICA continuity and warnings | ICA-fit high-pass did not explicitly split on BAD boundaries; nonconvergence-independent warnings could be swallowed | BAD segments are honored and other warnings re-emitted; exclusions/correlations/convergence retained in QC |
| Leakage | GroupKFold and per-fold pipeline scaling were already correct | Same subject splits for all models; test coverage exactly once and compare saved scaler means with training-only data |
| Imbalance | SVM/forest weighted; LDA used empirical priors | Balanced SVM/RF/logistic weights and default equal LDA priors; actual class counts exported |
| Classifiers | Three models; SVM was linear | Four models: shrinkage LDA, RBF SVM, RF and L2 logistic regression |
| P300 | Per-channel mean/positive peak/std/latency present; area absent | Added signed trapezoidal area in µV·s and the five P300 summaries of the predefined Fz/Cz/Pz ROI waveform |
| Metrics | Accuracy/balanced accuracy/F1/ROC-AUC only; benchmark filenames differed | Required CSV/JSON, precision/recall and AP added; target=1 and zero-division convention explicit |
| ERP uncertainty | Fz/Pz only; ±SEM at 160 DPI | Fz/Cz/Pz, pointwise 95% Student-t CI across subject means, 300 DPI, plotted data CSV |
| ROC/PR | ROC only | Combined fold-averaged ROC/PR figure; AUC/AP independently calculated within folds |
| Confusion matrices | Individual figures only | Combined four-model figure; raw counts plus true-class row percentages |
| Importance | Absent | Top-20 training-fold RF impurity importance figure and complete fold/summary CSVs |
| Provenance | Dataset identity validated, but feature schema upgrades could be missed | Explicit feature schema version and config checks; plot-only rendering rejects stale evaluations |
| Failed consolidation | Disk allocation could fail before entering cleanup | Allocation is now inside cleanup; a simulated full-disk test verifies no partial temporary arrays remain |
| Deliverable completion | Training and ERP plotting were separate stages | Training exports all six required deliverables automatically and writes completion metadata last |

## Statistical interpretation

- Complete subjects remain separate between training and test folds. Scaling and all supervised fitting use training folds only. The fixed P300 window, channel list and ROI are specified before evaluation; importance ranking does not change the feature set.
- Fold metrics are averaged with equal fold weights; trial-level metrics inside a fold weight trials equally. Subject-level metrics are exported separately. Fold sample SD is descriptive because training sets overlap; it is not a confidence interval or a significance test between classifiers.
- Precision/recall/F1 refer to the target class. AP is average precision, not the area of an interpolated mean PR display. Uncalibrated decision scores from different folds are never pooled to estimate ROC-AUC/AP.
- ERP intervals use independent subject means and Student-t critical values, pointwise at each latency. These intervals are not simultaneous bands or paired target-minus-standard significance tests. One subject cannot estimate a population CI, so no such shading is shown for a dry-run.
- RF mean decrease in impurity is descriptive and can favor high-cardinality variables or split credit among correlated features. Its fold SD is not a confidence interval. A Pz feature ranking highly does not prove a P300 source or a causal channel effect.
- Model ordering by CV F1 is exploratory. Hyperparameter/feature tuning or selecting a winner based on this benchmark requires nested subject CV or a new independent confirmation set for an unbiased final claim.

## Neurophysiological and execution limits

- The default 30-channel selection and common-average reference are an explicit CPU/storage tradeoff; retaining all channels is configurable. No EOG-only channels enter feature extraction.
- Fp1/Fp2 are EEG proxies for ocular activity because dedicated EOG channels are absent in the inspected source recording. Proxy-based ICA can remove frontal neural activity or miss artifacts. Inspect per-subject QC and ERPs; this automated pipeline does not reproduce manual expert artifact review.
- ICA is estimated from up to the first five minutes of each subject's unlabeled recording, then applied to that recording. This is an offline, per-recording adaptation protocol. Zero-phase filters and complete-recording access are not causal online decoding.
- Signed area distinguishes positive and negative waveforms. Single-trial peak latency is noisy, especially on low-amplitude/flat windows; constant-waveform ties select the earliest eligible sample. Area and peak features do not independently establish that an observed waveform is P300.
- Target trials require a behavioral response, so response-related motor activity can contribute inside the feature windows. The benchmark discriminates stimulus classes and does not isolate a uniquely attentional P300 generator.
- One-second Welch estimates have limited delta resolution. Wavelet levels are not identical to physiological frequency bands. No synthetic performance metric should be interpreted as dataset accuracy.
- The previous real sub-01 validation retained 574 epochs (494 standard, 80 target) after rejecting five standard/target epochs; 81 distractors were excluded. Raw files were deleted. Full 42-subject benchmark performance has not been measured in this audit.
- Hard process/runtime termination cannot execute Python cleanup; restart instructions explain orphaned temporary folders. Generated raw/processed arrays, models, figures and metrics stay out of Git.

## Reproducible checks

**Validation performed:** 13 offline tests passed. A fresh real-data check processed sub-01/sub-02/sub-03 sequentially and removed every raw download, retaining **1,707 epochs** (1,467 standard, 240 target), **785 features**, and all six required CSV/JSON/figure deliverables under `artifacts/`. All four required PNGs were checked for 300 DPI. This check used **3 GroupKFold folds and 100 RF trees**; the default full-cohort configuration remains 5 folds and 300 trees. Its complete configuration is saved locally as `artifacts/audit_real_subset_config.yaml`, and the benchmark JSON records the actual subjects/parameters. Synthetic exports are kept separately under `artifacts/audit_synthetic_validation/` and explicitly labeled as software validation.

The three-subject check is not an estimate for all 42 subjects. Mean ROC-AUC ranged approximately 0.754–0.793, while target F1 ranged 0.168–0.353 and recall varied substantially between subjects. This illustrates why accuracy and balanced weights alone cannot establish useful rare-target detection; no thresholds or hyperparameters were tuned to these held-out results. Feature importance was reviewed only after fitting/evaluation.

```bash
python -m unittest discover -s tests -v
python -m src.preprocess --dry-run --resume
python -m src.preprocess --resume
python -m src.features
python -m src.train_ml
python -m src.visualize
```

For an old checkout whose preprocessing signature differs, use a fresh artifacts folder or rerun preprocessing with `--overwrite`. For local execution, add `--raw-dir ./temp_raw`. See README for Drive linking before running on Colab.

## 2026-11-21 — Modular Colab workflow and preprocessing QC

- Preserved preprocessing/feature schema and the four baseline hyperparameters; no claimed score improvement from this refactor.
- Moved each model definition into `src/models/`, added independent training entrypoints and common CV/evaluation engine. Per-model results and integrity manifests precede aggregation; comparison refuses missing, edited or incompatible runs.
- Added actual raw/filtered/post-ICA capture before raw cleanup, matched-retained-trial ERP, bounded trace/PSD, ICA correlations/components/sources, channel inspection flags, final trial heatmap and descriptive P300 scalp maps. QC-only redownloads selected subjects without replacing existing cohort/features. Missing montage positions are tolerated for clean recordings but still require correction when interpolation is necessary.
- Added subject-level diagnostic learning curves inside the first outer TRAIN fold. RF log loss uses growing forests on a fixed inner split (no trial-level OOB evaluation). Logistic optimizer trace is a separate CPU diagnostic with a verified balanced L2 objective; it is not represented as the history of GPU or outer-CV fits.
- Added optional direct cuML weighted RBF SVM / L2 Logistic Regression, lazy availability checks, recorded CPU fallback, explicit GPU failure mode. LDA shrinkage/priors and weighted RF remain CPU. GPU hardware performance has not been tested locally.
- Added clean Vietnamese Colab notebook with separate cells per model, checked subprocess execution, environment provenance, saved-image display and final comparison. Existing 42-subject Drive arrays can be reused.
- Validation: regression tests include independent-run/compare/resume, stale-file/protocol rejection, outer/inner subject isolation, GPU adapter contracts/fallback, convergence retry, raw QC-only preservation, and ICA QC. Full-cohort scientific performance must be measured separately on Colab; no promise of automatic F1 gains.
