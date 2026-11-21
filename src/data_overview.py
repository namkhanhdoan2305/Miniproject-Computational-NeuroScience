"""Descriptive cohort and preprocessing summaries, independent of ML training."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import load_epochs, read_json
from .visualize import save_figure


def export_data_overview(directory, p300_window=(.3, .6)):
    directory = Path(directory)
    x, y, groups, channels, times, metadata = load_epochs(directory)
    rows, preprocessing = [], []
    for subject in np.unique(groups):
        labels = y[groups == subject]
        rows.append({"subject": subject, "n_standard": int((labels == 0).sum()),
                     "n_target": int((labels == 1).sum()), "n_trials": len(labels),
                     "target_prevalence": float(labels.mean())})
        path = directory / "subjects" / subject / "metadata.json"
        if path.exists():
            for report in read_json(path)["recordings"]:
                ica = report["ica"]
                preprocessing.append({"subject": subject, "recording": report["recording"],
                    **{f"events_{k}": v for k, v in report["event_counts"].items()},
                    **{f"retained_{k}": v for k, v in report["retained"].items()},
                    "dropped_epochs": report["dropped_epochs"],
                    "ica_components_excluded": str(ica.get("excluded", [])),
                    "ica_eog_sources": str(ica.get("eog_sources", [])),
                    "ica_proxy": ica.get("using_eeg_proxy", False),
                    "ica_convergence_warnings": str(ica.get("convergence_warnings", [])),
                    "drop_reasons": str(report["drop_reasons"])})
    table = pd.DataFrame(rows)
    table.to_csv(directory / "dataset_overview.csv", index=False)
    if preprocessing:
        pd.DataFrame(preprocessing).to_csv(directory / "preprocessing_summary.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(table) * .28), 7), gridspec_kw={"height_ratios": [2, 1]})
    indices = np.arange(len(table))
    axes[0].bar(indices, table.n_standard, label="Standard (0)", color="#0072B2")
    axes[0].bar(indices, table.n_target, bottom=table.n_standard, label="Target (1)", color="#D55E00")
    axes[0].set(xticks=indices, xticklabels=table.subject, ylabel="Retained trials", title="Class distribution after distractor removal and epoch rejection")
    axes[0].tick_params(axis="x", labelrotation=90)
    axes[0].legend()
    axes[1].bar(indices, table.target_prevalence, color="#D55E00")
    axes[1].set(xticks=indices, xticklabels=table.subject, ylabel="Target fraction", ylim=(0, 1))
    axes[1].tick_params(axis="x", labelrotation=90)
    fig.suptitle(f"N={len(table)} subjects | X={tuple(x.shape)} | {metadata['sfreq']:g} Hz | {times[0]:g} to {times[-1]:g} s")
    fig.tight_layout()
    save_figure(fig, directory / "dataset_overview.png")
    # These final epochs visualize the current processed dataset, not raw recordings.
    subject = np.unique(groups)[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
    for ax, channel in zip(axes, ["Fz", "Cz", "Pz"]):
        if channel not in channels:
            ax.set_visible(False)
            continue
        i = int(np.flatnonzero(channels == channel)[0])
        for label, color in [(0, "#0072B2"), (1, "#D55E00")]:
            trials = np.flatnonzero((groups == subject) & (y == label))[:15]
            for j, trial in enumerate(trials):
                ax.plot(times * 1000, x[trial, i] * 1e6, color=color, alpha=.2, linewidth=.7,
                        label=("Standard" if label == 0 else "Target") if j == 0 else None)
        ax.axvspan(p300_window[0] * 1000, p300_window[1] * 1000, alpha=.12, color="gray")
        ax.set(title=channel, xlabel="Time (ms)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Amplitude (µV)")
    fig.suptitle(f"{subject}: first 15 retained trials per class (processed EEG, individual traces)")
    fig.tight_layout()
    save_figure(fig, directory / "processed_trial_examples.png")
    return table
