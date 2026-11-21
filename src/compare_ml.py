"""Compare four completed runs without retraining, refusing stale/mixed protocols."""
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
from .common import DEFAULT_CONFIG, LOGGER, load_config, package_versions, setup_logging, write_json
from .evaluation import METRICS
from .models import MODELS
from .training import context, validate_completed


def run(config):
    root, _, y, groups, _, info, _, manifest, signature = context(config)
    completed = [(key, validate_completed(root / "ml" / key, signature)) for key in MODELS]
    marker = root / "training_metadata.json"
    marker.unlink(missing_ok=True)
    directories = [root / "ml" / key for key, _ in completed]
    summary = pd.concat([pd.read_csv(d / "metrics.csv") for d in directories], ignore_index=True)
    summary.to_csv(root / "ml_benchmark_results.csv", index=False)
    for filename in ["metrics_per_fold.csv", "metrics_per_subject.csv", "oof_predictions.csv"]:
        pd.concat([pd.read_csv(d / filename, dtype={"subject": str}) for d in directories], ignore_index=True).to_csv(root / filename, index=False)
    write_json(root / "cv_splits.json", manifest)
    # Compatibility exports for existing Colab scripts and model consumers.
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    for key, metadata in completed:
        directory = root / "ml" / key
        for path in (directory / "models").glob("*.pkl"):
            shutil.copyfile(path, models_dir / path.name)
        shutil.copyfile(directory / "confusion_matrix.csv", root / f"confusion_matrix_{metadata['model'].lower()}.csv")
    for filename in ["feature_importance.csv", "feature_importance_per_fold.csv"]:
        shutil.copyfile(root / "ml" / "random_forest" / filename, root / filename)
    counts = {"standard": int((y == 0).sum()), "target": int((y == 1).sum())}
    report = {"results": summary.to_dict(orient="records"), "class_counts": counts,
              "majority_accuracy_baseline": float(max(counts.values()) / len(y)),
              "aggregation": "Mean ± sample SD (ddof=1) across outer subject-wise test folds; positive class target=1",
              "primary_comparison_metric": "target F1 (predeclared, exploratory comparison)",
              "best_by_metric": {m: str(summary.loc[summary[m + '_mean'].idxmax(), 'model']) for m in METRICS},
              "notes": ["Fold SD is not a confidence interval or a significance test.",
                        "Winner selection needs independent confirmation; no superiority claim from these folds alone.",
                        "Preprocessing uses the complete unlabeled recording of each new subject (offline setting).",
                        "Diagnostic curves use inner subjects, never outer held-out subjects.",
                        "GPU and CPU numerical solvers may differ. Actual backend and fit times are recorded."]}
    write_json(root / "ml_benchmark_results.json", report)
    provenance = {"feature_signature": info["signature"], "protocol_signature": signature,
                  "training": config["training"], "seed": config["seed"], "versions": package_versions(),
                  "evaluation": report["aggregation"], "class_counts": counts,
                  "source_runs": {key: {"total_seconds": meta["total_seconds"], "actual_backends": meta["actual_backends"]}
                                  for key, meta in completed},
                  "final_models": "All-subject refits, not independently evaluated"}
    from .visualize import export_all_figures, save_figure
    export_all_figures(root, config, provenance)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    positions = np.arange(len(summary))
    for i, metric in enumerate(["balanced_accuracy", "f1", "roc_auc", "average_precision"]):
        axes[0].errorbar(positions + (i - 1.5) * .13, summary[metric + "_mean"], yerr=summary[metric + "_std"],
                         fmt="o", capsize=3, label=metric)
    axes[0].set(xticks=positions, xticklabels=summary.model, ylim=(0, 1.05), ylabel="Mean ± fold SD", title="Same held-out subjects and features")
    axes[0].tick_params(axis="x", labelrotation=15)
    axes[0].legend(fontsize=8)
    axes[1].bar(positions, summary.fit_seconds_mean, yerr=summary.fit_seconds_std, capsize=3)
    axes[1].set(xticks=positions, xticklabels=summary.model, ylabel="Fit time per fold (seconds)", title="Observed time: backend/hardware dependent")
    axes[1].tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    save_figure(fig, root / "model_comparison.png")
    provenance["required_deliverables"] = ["ml_benchmark_results.csv", "ml_benchmark_results.json",
        "fig1_grand_average_erp.png", "fig2_roc_and_pr_curves.png", "fig3_confusion_matrices.png", "fig4_feature_importance.png", "model_comparison.png"]
    write_json(marker, provenance)
    LOGGER.info("Comparison complete:\n%s", summary[["model", "backend", *[m + "_mean" for m in METRICS]]].to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.artifacts:
        config["paths"]["artifacts"] = str(args.artifacts)
    run(config)


if __name__ == "__main__":
    main()
