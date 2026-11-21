"""Publication figures from subject averages and strictly held-out ML predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

try:
    from .common import (DEFAULT_CONFIG, LOGGER, artifact_dir, load_config, load_epochs,
                         read_json, setup_logging)
    from .features import load_features
except ImportError:
    from common import (DEFAULT_CONFIG, LOGGER, artifact_dir, load_config, load_epochs,
                        read_json, setup_logging)
    from features import load_features

DPI = 300
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def save_figure(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def subject_confidence_interval(evokeds):
    """Pointwise 95% Student-t CI across independent subject means (not trials)."""
    mean = evokeds.mean(axis=0)
    if len(evokeds) < 2:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    half_width = student_t.ppf(0.975, len(evokeds) - 1) * evokeds.std(axis=0, ddof=1) / np.sqrt(len(evokeds))
    return mean, mean - half_width, mean + half_width


def plot_erp(directory, filename, channels=("Fz", "Cz", "Pz"), p300_window=(0.3, 0.6)):
    """Average trials within each subject/class, then equally weight subjects."""
    x, y, groups, channel_names, times, _ = load_epochs(directory)
    missing = set(channels) - set(channel_names)
    if missing:
        raise ValueError(f"Cannot plot ERP; missing channels: {sorted(missing)}")
    indices = [int(np.flatnonzero(channel_names == ch)[0]) for ch in channels]
    fig, axes = plt.subplots(1, len(channels), figsize=(4.5 * len(channels), 3.8), squeeze=False, sharey=True)
    subjects, summaries = np.unique(groups), []
    for label, name, color in [(0, "Standard", COLORS[0]), (1, "Target", COLORS[1])]:
        evokeds = []
        for subject in subjects:
            trials = np.flatnonzero((groups == subject) & (y == label))
            evokeds.append(np.stack([np.mean(x[trials, idx, :], axis=0, dtype=np.float64) for idx in indices]))
        evokeds = np.asarray(evokeds) * 1e6
        mean, lower, upper = subject_confidence_interval(evokeds)
        for i, (ax, channel) in enumerate(zip(axes[0], channels)):
            ax.plot(times * 1000, mean[i], color=color, label=name, linewidth=1.5)
            if len(subjects) > 1:
                ax.fill_between(times * 1000, lower[i], upper[i], color=color, alpha=0.17, linewidth=0)
            ax.set_title(channel)
            summaries.append(pd.DataFrame({"channel": channel, "condition": name, "time_s": times,
                                           "mean_uv": mean[i], "ci95_lower_uv": lower[i], "ci95_upper_uv": upper[i],
                                           "n_subjects": len(subjects)}))
    for ax in axes[0]:
        ax.axvline(0, color="black", linewidth=0.7)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.axvspan(p300_window[0] * 1000, p300_window[1] * 1000, color="gray", alpha=0.12, label="P300 window")
        ax.set(xlabel="Time after stimulus (ms)", xlim=(times[0] * 1000, times[-1] * 1000))
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Amplitude (µV; positive up)")
    title = f"Grand-average ERP | N = {len(subjects)} | equal subject weights; pointwise 95% CI"
    if len(subjects) == 1:
        title = f"{subjects[0]} ERP check | one subject; population CI unavailable"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_figure(fig, filename)
    pd.concat(summaries, ignore_index=True).to_csv(Path(filename).with_suffix(".csv"), index=False)


def plot_distributions(directory, settings):
    x, y, groups, names, _ = load_features(directory)
    wanted = [f"{channel}__{suffix}" for channel in settings["channels"] for suffix in settings["distribution_features"]]
    missing = set(wanted) - set(names)
    if missing:
        raise ValueError(f"Configured distribution features not found: {sorted(missing)}")
    rows = []
    for name in wanted:
        column = int(np.flatnonzero(names == name)[0])
        channel, feature = name.split("__", 1)
        for subject in np.unique(groups):
            for label, condition in [(0, "Standard"), (1, "Target")]:
                values = x[(groups == subject) & (y == label), column]
                rows.append({"subject": subject, "channel": channel, "feature": feature,
                             "condition": condition, "value": float(np.mean(values))})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(directory) / "feature_distributions.csv", index=False)
    fig, axes = plt.subplots(len(settings["channels"]), len(settings["distribution_features"]),
                             figsize=(3.6 * len(settings["distribution_features"]), 3 * len(settings["channels"])), squeeze=False)
    for i, channel in enumerate(settings["channels"]):
        for j, feature in enumerate(settings["distribution_features"]):
            ax = axes[i, j]
            subset = frame[(frame.channel == channel) & (frame.feature == feature)]
            # bxp avoids version-sensitive seaborn boxplot arguments. Each dot is
            # one subject mean; deterministic offsets keep plots reproducible.
            for position, (condition, color) in enumerate(zip(["Standard", "Target"], COLORS[:2]), 1):
                values = subset.loc[subset.condition.eq(condition), "value"].to_numpy()
                q1, median, q3 = np.percentile(values, [25, 50, 75])
                low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                within = values[(values >= low) & (values <= high)]
                stats = {"med": median, "q1": q1, "q3": q3, "whislo": within.min(), "whishi": within.max(), "fliers": []}
                ax.bxp([stats], positions=[position], patch_artist=True, showfliers=False,
                       boxprops={"facecolor": color, "alpha": 0.4}, medianprops={"color": "black"})
                ax.scatter(position + np.linspace(-0.07, 0.07, len(values)), values, s=9, color="black", alpha=0.5)
            ax.set_xticks([1, 2], ["Standard", "Target"])
            ax.set_title(f"{channel}: {feature.replace('_', ' ')}", fontsize=9)
            ax.set_ylabel("Subject mean")
    fig.suptitle("Feature distributions | one point per subject and condition; descriptive only", fontsize=11)
    fig.tight_layout()
    save_figure(fig, Path(directory) / "feature_distributions.png")


def validated_evaluation(directory, provenance=None):
    """Refuse incomplete/stale evaluation instead of plotting mixed old outputs."""
    directory = Path(directory)
    _, y, groups, _, info = load_features(directory)
    provenance = provenance if provenance is not None else read_json(directory / "training_metadata.json")
    if provenance["feature_signature"] != info["signature"]:
        raise ValueError("Evaluation is stale after feature extraction; rerun src.train_ml.")
    summary = pd.read_csv(directory / "ml_benchmark_results.csv")
    folds = pd.read_csv(directory / "metrics_per_fold.csv")
    oof = pd.read_csv(directory / "oof_predictions.csv", dtype={"subject": str})
    if set(oof.model) != set(summary.model) or set(folds.model) != set(summary.model):
        raise ValueError("Benchmark, fold metrics and held-out predictions contain different models.")
    for name, table in oof.groupby("model"):
        table = table.sort_values("trial_index")
        if not np.array_equal(table.trial_index, np.arange(len(y))):
            raise ValueError(f"Missing/duplicate trial predictions for {name}")
        if not np.array_equal(table.y_true, y) or not np.array_equal(table.subject, groups):
            raise ValueError("Held-out predictions do not match the current trial labels/subjects.")
        if not np.isfinite(table.score).all() or not set(table.y_pred).issubset({0, 1}):
            raise ValueError("Invalid held-out predictions/scores.")
        if (table.groupby("subject").fold.nunique() != 1).any():
            raise ValueError("A subject was split across held-out folds.")
    return summary, folds, oof


def plot_roc_pr(directory, summary, folds, oof):
    grid = np.linspace(0, 1, 301)
    roc_data, pr_data = {"fpr": grid}, {"recall": grid}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for color, row in zip(COLORS, summary.itertuples(index=False)):
        roc_curves, pr_curves = [], []
        for _, table in oof[oof.model.eq(row.model)].groupby("fold"):
            fpr, tpr, _ = roc_curve(table.y_true, table.score)
            precision, recall, _ = precision_recall_curve(table.y_true, table.score)
            roc_curves.append(np.interp(grid, fpr, tpr))
            # Interpolate fold curves only for display. AP is calculated on the
            # original precision/recall steps, independently within each fold.
            pr_curves.append(np.interp(grid, recall[::-1], precision[::-1]))
        for ax, curves, store in [(axes[0], roc_curves, roc_data), (axes[1], pr_curves, pr_data)]:
            mean = np.mean(curves, axis=0)
            std = np.std(curves, axis=0, ddof=1)
            store[f"{row.model}_mean"], store[f"{row.model}_std"] = mean, std
            label = (f"{row.model}: AUC {row.roc_auc_mean:.3f} ± {row.roc_auc_std:.3f}" if ax is axes[0]
                     else f"{row.model}: AP {row.average_precision_mean:.3f} ± {row.average_precision_std:.3f}")
            ax.plot(grid, mean, color=color, label=label, linewidth=1.6)
            ax.fill_between(grid, np.maximum(0, mean - std), np.minimum(1, mean + std), color=color, alpha=0.10)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    prevalence = float(folds.groupby("fold").target_prevalence.first().mean())
    axes[1].axhline(prevalence, color="black", linestyle="--", linewidth=0.8,
                   label=f"Mean fold target prevalence: {prevalence:.3f}")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC: mean held-out fold curves")
    axes[1].set(xlabel="Recall (target)", ylabel="Precision (target)", title="Precision–recall: mean held-out fold curves")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(fontsize=7.5, loc="lower right" if ax is axes[0] else "lower left")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Subject-wise cross-validation | shaded ±1 fold SD; scores are not pooled across fold scales", fontsize=10)
    fig.tight_layout()
    save_figure(fig, Path(directory) / "fig2_roc_and_pr_curves.png")
    pd.DataFrame(roc_data).to_csv(Path(directory) / "roc_curves.csv", index=False)
    pd.DataFrame(pr_data).to_csv(Path(directory) / "pr_curves.csv", index=False)
    shutil.copyfile(Path(directory) / "fig2_roc_and_pr_curves.png", Path(directory) / "roc_curves.png")


def draw_confusion(ax, matrix, name):
    percentages = matrix / matrix.sum(axis=1, keepdims=True)
    ax.imshow(percentages, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}\n({percentages[i, j]:.1%})", ha="center", va="center",
                    color="white" if percentages[i, j] > 0.55 else "black", fontsize=11)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Standard", "Target"],
           yticklabels=["Standard", "Target"], xlabel="Predicted class", ylabel="True class", title=name)


def plot_confusions(directory, summary, oof):
    ranked = summary.sort_values("f1_mean", ascending=False)
    ncols = 2
    fig, axes = plt.subplots(int(np.ceil(len(ranked) / ncols)), ncols, figsize=(9, 4 * int(np.ceil(len(ranked) / ncols))), squeeze=False)
    for ax, row in zip(axes.flat, ranked.itertuples(index=False)):
        table = oof[oof.model.eq(row.model)]
        matrix = confusion_matrix(table.y_true, table.y_pred, labels=[0, 1])
        draw_confusion(ax, matrix, f"{row.model} | target F1 = {row.f1_mean:.3f}")
        single, single_ax = plt.subplots(figsize=(4.5, 4))
        draw_confusion(single_ax, matrix, row.model)
        single.tight_layout()
        save_figure(single, Path(directory) / f"confusion_matrix_{row.model.lower()}.png")
    for ax in list(axes.flat)[len(ranked):]:
        ax.axis("off")
    fig.suptitle("Held-out confusion matrices | raw counts and row percentages\nOrdered by CV target F1; exploratory model comparison", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, Path(directory) / "fig3_confusion_matrices.png")


def plot_importance(directory):
    frame = pd.read_csv(Path(directory) / "feature_importance.csv")
    top = frame.sort_values("importance_mean", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    window = read_json(Path(directory) / "feature_metadata.json")["settings"]["p300_window"]
    p300_label = f"P300[{window[0] * 1000:g}–{window[1] * 1000:g} ms] "
    labels = [name.replace("__", " | ").replace("p300_", p300_label).replace("_", " ") for name in top.feature]
    ax.barh(labels, top.importance_mean, xerr=top.importance_std, color=COLORS[2], alpha=0.85, capsize=2)
    ax.set(xlabel="Random Forest mean decrease in impurity (mean ± fold SD)",
           title="Top 20 model-associated features | training-fold Random Forests")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.012, "Descriptive importance; correlated features share credit. Not causal evidence or a test for channel significance.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, Path(directory) / "fig4_feature_importance.png")


def plot_metrics_table(directory, summary):
    columns = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = [[row["model"], *[f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}" for metric in columns]]
            for row in summary.to_dict(orient="records")]
    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Model", "Accuracy", "Balanced acc.", "Precision", "Recall", "Target F1", "ROC-AUC"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("Held-out GroupKFold metrics | mean ± sample SD")
    fig.tight_layout()
    save_figure(fig, Path(directory) / "metrics_table.png")


def export_all_figures(directory, config, provenance=None):
    summary, folds, oof = validated_evaluation(directory, provenance)
    if config["features"] != read_json(Path(directory) / "feature_metadata.json")["settings"]:
        raise ValueError("Feature settings changed; use the configuration recorded at training time.")
    plot_erp(directory, Path(directory) / "fig1_grand_average_erp.png", config["visualization"]["channels"],
             config["features"]["p300_window"])
    shutil.copyfile(Path(directory) / "fig1_grand_average_erp.png", Path(directory) / "grand_average_erp.png")
    shutil.copyfile(Path(directory) / "fig1_grand_average_erp.csv", Path(directory) / "grand_average_erp.csv")
    plot_roc_pr(directory, summary, folds, oof)
    plot_confusions(directory, summary, oof)
    plot_importance(directory)
    plot_metrics_table(directory, summary)
    plot_distributions(directory, config["visualization"])
    from src.data_overview import export_data_overview
    export_data_overview(directory, config["features"]["p300_window"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--erp-only", action="store_true", help="Plot the three-channel ERP without requiring feature/training artifacts.")
    mode.add_argument("--data-only", action="store_true", help="Plot cohort/class distribution, processed trials, ERP and preprocessing summary before ML.")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.artifacts:
        config["paths"]["artifacts"] = str(args.artifacts)
    output = artifact_dir(config)
    if args.erp_only or args.data_only:
        if args.data_only:
            from src.data_overview import export_data_overview
            export_data_overview(output, config["features"]["p300_window"])
        plot_erp(output, output / "fig1_grand_average_erp.png", config["visualization"]["channels"], config["features"]["p300_window"])
        shutil.copyfile(output / "fig1_grand_average_erp.png", output / "grand_average_erp.png")
        shutil.copyfile(output / "fig1_grand_average_erp.csv", output / "grand_average_erp.csv")
    else:
        export_all_figures(output, config)


if __name__ == "__main__":
    main()
