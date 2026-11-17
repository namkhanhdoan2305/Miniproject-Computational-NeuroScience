"""Subject-weighted Fz/Pz ERPs and descriptive handcrafted-feature distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .common import DEFAULT_CONFIG, LOGGER, artifact_dir, load_config, load_epochs, setup_logging
    from .features import load_features
except ImportError:
    from common import DEFAULT_CONFIG, LOGGER, artifact_dir, load_config, load_epochs, setup_logging
    from features import load_features


def plot_erp(directory, filename, channels=("Fz", "Pz")):
    """Average trials within each subject/class, then equally weight subjects."""
    x, y, groups, channel_names, times, _ = load_epochs(directory)
    missing = set(channels) - set(channel_names)
    if missing:
        raise ValueError(f"Cannot plot ERP; missing channels: {sorted(missing)}")
    indices = [int(np.flatnonzero(channel_names == ch)[0]) for ch in channels]
    fig, axes = plt.subplots(1, len(channels), figsize=(6 * len(channels), 4), squeeze=False, sharey=True)
    subjects = np.unique(groups)
    for label, name, color in [(0, "standard", "#2474a6"), (1, "target", "#d24a37")]:
        evokeds = []
        for subject in subjects:
            trials = np.flatnonzero((groups == subject) & (y == label))
            # Index channels first: never create a full-cohort copy of the EEG tensor.
            evokeds.append(np.stack([np.mean(x[trials, idx, :], axis=0, dtype=np.float64) for idx in indices]))
        evokeds = np.asarray(evokeds) * 1e6
        mean = evokeds.mean(axis=0)
        sem = (evokeds.std(axis=0, ddof=1) / np.sqrt(len(subjects))) if len(subjects) > 1 else np.zeros_like(mean)
        for i, (ax, channel) in enumerate(zip(axes[0], channels)):
            ax.plot(times * 1000, mean[i], color=color, label=name)
            if len(subjects) > 1:
                ax.fill_between(times * 1000, mean[i] - sem[i], mean[i] + sem[i], color=color, alpha=0.18)
            ax.set_title(channel)
    for ax in axes[0]:
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.axvspan(300, 600, color="gray", alpha=0.1, label="P300 window")
        ax.set_xlabel("Time after stimulus (ms)")
        ax.legend(fontsize=9)
    axes[0, 0].set_ylabel("Amplitude (µV, positive up)")
    description = f"Grand-average ERP — {len(subjects)} subjects; equal subject weights; shading ±SEM"
    if len(subjects) == 1:
        description = f"{subjects[0]} ERP check — trial averages by class (one subject)"
    fig.suptitle(description)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    LOGGER.info("Saved %s", filename)


def plot_distributions(directory, settings):
    x, y, groups, names, _ = load_features(directory)
    wanted = [f"{channel}__{suffix}" for channel in settings["channels"]
              for suffix in settings["distribution_features"]]
    missing = set(wanted) - set(names)
    if missing:
        raise ValueError(f"Configured distribution features not found: {sorted(missing)}")
    # Subject/class means give each subject equal influence and avoid imbalanced
    # target/standard trial counts dominating the descriptive comparisons.
    rows = []
    for name in wanted:
        column = int(np.flatnonzero(names == name)[0])
        channel, feature = name.split("__", 1)
        for subject in np.unique(groups):
            for label, condition in [(0, "standard"), (1, "target")]:
                values = x[(groups == subject) & (y == label), column]
                rows.append({"subject": subject, "channel": channel, "feature": feature,
                             "condition": condition, "value": float(np.mean(values))})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(directory) / "feature_distributions.csv", index=False)
    fig, axes = plt.subplots(len(settings["channels"]), len(settings["distribution_features"]),
                             figsize=(4 * len(settings["distribution_features"]), 3.7 * len(settings["channels"])), squeeze=False)
    for i, channel in enumerate(settings["channels"]):
        for j, feature in enumerate(settings["distribution_features"]):
            ax = axes[i, j]
            subset = frame[(frame.channel == channel) & (frame.feature == feature)]
            sns.boxplot(data=subset, x="condition", y="value", hue="condition", legend=False,
                        palette={"standard": "#2474a6", "target": "#d24a37"}, ax=ax, showfliers=False)
            sns.stripplot(data=subset, x="condition", y="value", color="black", alpha=0.5,
                          jitter=False, size=3, ax=ax)
            ax.set_title(f"{channel}: {feature.replace('_', ' ')}", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("Subject mean")
    fig.suptitle("Feature distributions — one point per subject and condition; descriptive only")
    fig.tight_layout()
    fig.savefig(Path(directory) / "feature_distributions.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--erp-only", action="store_true", help="Plot ERPs before extracting features.")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.artifacts:
        config["paths"]["artifacts"] = str(args.artifacts)
    output = artifact_dir(config)
    plot_erp(output, output / "grand_average_erp.png", config["visualization"]["channels"])
    if not args.erp_only:
        plot_distributions(output, config["visualization"])


if __name__ == "__main__":
    main()
