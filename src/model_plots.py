"""Figures for a single independently completed model run."""
from html import escape
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from .common import read_json
from .visualize import draw_confusion, plot_roc_pr, save_figure


def export_model_figures(directory, metrics, folds, oof):
    directory = Path(directory)
    plot_roc_pr(directory, metrics, folds, oof)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    draw_confusion(ax, confusion_matrix(oof.y_true, oof.y_pred, labels=[0, 1]), metrics.iloc[0].model)
    fig.tight_layout()
    save_figure(fig, directory / "confusion_matrix.png")
    metric_names = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    fig, ax = plt.subplots(figsize=(9, 4))
    row = metrics.iloc[0]
    ax.bar(np.arange(len(metric_names)), [row[m + "_mean"] for m in metric_names],
           yerr=[row[m + "_std"] for m in metric_names], capsize=4, color="#0072B2")
    ax.set(xticks=np.arange(len(metric_names)), xticklabels=["Balanced Acc.", "Precision", "Recall", "F1", "ROC-AUC", "AP"],
           ylim=(0, 1.1), ylabel="Score (target = 1)", title=f"{row.model} | outer subject CV: mean ± fold SD")
    fig.tight_layout()
    save_figure(fig, directory / "metrics.png")
    if (directory / "learning_curve.csv").exists():
        data = pd.read_csv(directory / "learning_curve.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ["balanced_accuracy", "f1"]):
            for prefix, color in [("train", "#0072B2"), ("validation", "#D55E00")]:
                ax.plot(data.n_subjects, data[prefix + "_" + metric], "o-", label=prefix, color=color)
            ax.set(xlabel="Whole inner training subjects (nested subsets)", ylabel=metric, ylim=(0, 1.05))
            ax.set_xticks(data.n_subjects)
            ax.legend()
        fig.suptitle("Learning curve: fixed inner validation subjects; no outer test used")
        fig.tight_layout()
        save_figure(fig, directory / "learning_curve.png")
    if (directory / "training_progress.csv").exists():
        data = pd.read_csv(directory / "training_progress.csv")
        logistic = "iteration" in data
        fig, axes = plt.subplots(1, 2 if logistic else 1, figsize=(11 if logistic else 7, 4), squeeze=False)
        x = data["iteration" if logistic else "n_trees"]
        ax = axes[0, -1]
        for prefix in ["train", "validation"]:
            ax.plot(x, data[prefix + "_log_loss"], label=prefix)
        ax.set(xlabel="Optimizer iteration" if logistic else "Number of trees", ylabel="Unweighted log loss (natural log)")
        ax.legend()
        title = "Random Forest: diagnostic growth on inner subjects (not an optimization loss)"
        if logistic:
            axes[0, 0].plot(x, data.train_objective, color="#0072B2")
            axes[0, 0].set(xlabel="L-BFGS iteration", ylabel="Balanced L2 training objective")
            convergence = read_json(directory / "optimizer_diagnostics.json")
            title = "Separate CPU logistic diagnostic on inner subjects | " + ("converged" if convergence["success"] else "iteration/stopping limit reached")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        save_figure(fig, directory / "training_progress.png")
    if (directory / "feature_importance.csv").exists():
        importance = pd.read_csv(directory / "feature_importance.csv").head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(importance.feature, importance.importance_mean, xerr=importance.importance_std, color="#0072B2")
        ax.set(xlabel="Training-fold MDI importance (mean ± fold SD)", title="Random Forest | descriptive, correlated features share importance")
        fig.tight_layout()
        save_figure(fig, directory / "feature_importance.png")
    images = ["metrics.png", "fig2_roc_and_pr_curves.png", "confusion_matrix.png", "learning_curve.png", "training_progress.png", "feature_importance.png"]
    body = "".join(f'<h2>{escape(Path(p).stem)}</h2><img src="{p}" style="max-width:100%">'
                   for p in images if (directory / p).exists())
    (directory / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>EEG model results</title>'
        + f'<h1>{escape(str(row.model))}: subject-wise CV</h1>' + metrics.to_html(index=False)
        + '<p>Outer test subjects are used only for final evaluation. Curves use a separate inner subject split. '
        'Final model is an unevaluated refit on all subjects. Fold SD is descriptive.</p>' + body, encoding="utf-8")
