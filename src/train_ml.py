"""Subject-wise GroupKFold evaluation of classical ML with training-only scaling."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

try:
    from .common import (DEFAULT_CONFIG, LOGGER, artifact_dir, load_config, package_versions,
                         setup_logging, write_json)
    from .features import load_features
except ImportError:
    from common import (DEFAULT_CONFIG, LOGGER, artifact_dir, load_config, package_versions,
                        setup_logging, write_json)
    from features import load_features


def build_models(config):
    t = config["training"]
    return {
        "SVM": make_pipeline(StandardScaler(), LinearSVC(
            C=t["svm_c"], class_weight="balanced", dual=False, max_iter=10000,
            random_state=config["seed"],
        )),
        "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=t["rf_estimators"], min_samples_leaf=t["rf_min_samples_leaf"],
            class_weight="balanced_subsample", n_jobs=t["n_jobs"], random_state=config["seed"],
        )),
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    }


def group_splits(x, y, groups, n_splits):
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"GroupKFold needs at least {n_splits} subjects; found {len(np.unique(groups))}. "
                         "A one-subject dry-run cannot train subject-wise CV.")
    splits = list(GroupKFold(n_splits=n_splits).split(x, y, groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Subject leakage detected in CV split.")
        if set(np.unique(y[train])) != {0, 1} or set(np.unique(y[test])) != {0, 1}:
            raise ValueError("Every train/test fold must contain standard and target trials.")
    return splits


def positive_scores(model, x):
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    index = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(x)[:, index]


def save_model(path, model, names, info, config, fold=None):
    with path.open("wb") as handle:
        pickle.dump({"pipeline": model, "feature_names": names.tolist(),
                     "feature_settings": info["settings"], "feature_signature": info["signature"],
                     "training": config["training"], "seed": config["seed"],
                     "labels": {"standard": 0, "target": 1}, "fold": fold,
                     "versions": package_versions()}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run(config):
    output = artifact_dir(config)
    x, y, groups, names, info = load_features(output)
    if not np.isfinite(x).all():
        raise ValueError("Training features must be finite.")
    splits = group_splits(x, y, groups, config["training"]["cv_folds"])
    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)
    # A missing manifest indicates an interrupted/incomplete training run.
    (output / "training_metadata.json").unlink(missing_ok=True)
    write_json(output / "cv_splits.json", [{
        "fold": fold, "train_subjects": sorted(set(groups[train].tolist())),
        "test_subjects": sorted(set(groups[test].tolist())),
        "n_train": len(train), "n_test": len(test),
    } for fold, (train, test) in enumerate(splits, 1)])
    fold_rows, oof_rows, summary_rows = [], [], []
    fpr_grid = np.linspace(0, 1, 201)
    roc_data = {"fpr": fpr_grid}
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    for name, prototype in build_models(config).items():
        predictions = np.full(len(y), -1, dtype=np.int8)
        scores = np.full(len(y), np.nan)
        fold_ids = np.zeros(len(y), dtype=int)
        interpolated, aucs, rows = [], [], []
        for fold, (train, test) in enumerate(splits, 1):
            LOGGER.info("%s | fold %d/%d | train subjects=%d, test subjects=%d", name, fold,
                        len(splits), len(np.unique(groups[train])), len(np.unique(groups[test])))
            model = clone(prototype)
            model.fit(x[train], y[train])  # Scaling is fit inside this fold's pipeline.
            predictions[test] = model.predict(x[test])
            scores[test] = positive_scores(model, x[test])
            fold_ids[test] = fold
            row = {"model": name, "fold": fold, "n_test": len(test),
                   "accuracy": accuracy_score(y[test], predictions[test]),
                   "balanced_accuracy": balanced_accuracy_score(y[test], predictions[test]),
                   "f1": f1_score(y[test], predictions[test], pos_label=1, zero_division=0),
                   "roc_auc": roc_auc_score(y[test], scores[test])}
            rows.append(row)
            fold_rows.append(row)
            fpr, tpr, _ = roc_curve(y[test], scores[test])
            curve = np.interp(fpr_grid, fpr, tpr)
            curve[0], curve[-1] = 0, 1
            interpolated.append(curve)
            aucs.append(row["roc_auc"])
            if config["training"]["save_fold_models"]:
                save_model(models_dir / f"{name.lower()}_fold{fold}.pkl", model, names, info, config, fold)
        if (predictions < 0).any() or not np.isfinite(scores).all():
            raise RuntimeError("Some trials have no held-out prediction.")
        summary = {"model": name, "n_subjects": len(np.unique(groups)), "n_trials": len(y)}
        for metric in ("accuracy", "balanced_accuracy", "f1", "roc_auc"):
            values = np.array([row[metric] for row in rows])
            summary[f"{metric}_mean"] = values.mean()
            summary[f"{metric}_std"] = values.std(ddof=1)
        summary["oof_accuracy"] = accuracy_score(y, predictions)
        summary["oof_f1"] = f1_score(y, predictions, pos_label=1, zero_division=0)
        summary_rows.append(summary)
        oof_rows.append(pd.DataFrame({"model": name, "trial_index": np.arange(len(y)), "subject": groups,
                                      "fold": fold_ids, "y_true": y, "y_pred": predictions, "score": scores}))
        mean_tpr = np.mean(interpolated, axis=0)
        std_tpr = np.std(interpolated, axis=0)
        line, = ax_roc.plot(fpr_grid, mean_tpr, label=f"{name}: fold AUC {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}")
        ax_roc.fill_between(fpr_grid, np.maximum(0, mean_tpr - std_tpr), np.minimum(1, mean_tpr + std_tpr),
                            alpha=0.12, color=line.get_color())
        roc_data[f"{name}_mean_tpr"], roc_data[f"{name}_std_tpr"] = mean_tpr, std_tpr
        matrix = confusion_matrix(y, predictions, labels=[0, 1])
        pd.DataFrame(matrix, index=["true_standard", "true_target"],
                     columns=["pred_standard", "pred_target"]).to_csv(output / f"confusion_matrix_{name.lower()}.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, norm, title in zip(axes, [None, "true"], ["Held-out counts", "Row normalized"]):
            display = matrix if norm is None else matrix / matrix.sum(axis=1, keepdims=True)
            ConfusionMatrixDisplay(display, display_labels=["standard", "target"]).plot(
                ax=ax, colorbar=False, cmap="Blues", values_format="d" if norm is None else ".2f")
            ax.set_title(title)
        fig.suptitle(f"{name} — subject-wise out-of-fold predictions")
        fig.tight_layout()
        fig.savefig(output / f"confusion_matrix_{name.lower()}.png", dpi=160)
        plt.close(fig)
        LOGGER.info("Refitting %s on all subjects for the exported final model", name)
        final = clone(prototype).fit(x, y)
        save_model(models_dir / f"{name.lower()}_final.pkl", final, names, info, config)
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01),
               title="ROC curves — mean across held-out subject folds")
    ax_roc.legend(loc="lower right", fontsize=9)
    fig_roc.tight_layout()
    fig_roc.savefig(output / "roc_curves.png", dpi=160)
    plt.close(fig_roc)
    metrics = pd.DataFrame(summary_rows)
    metrics.to_csv(output / "metrics_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "metrics_per_fold.csv", index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(roc_data).to_csv(output / "roc_curves.csv", index=False)
    columns = ["model", "accuracy_mean", "f1_mean", "roc_auc_mean"]
    fig, ax = plt.subplots(figsize=(8, 2.3))
    ax.axis("off")
    table = ax.table(cellText=[[row.model, f"{row.accuracy_mean:.3f}", f"{row.f1_mean:.3f}", f"{row.roc_auc_mean:.3f}"]
                               for row in metrics.itertuples()],
                     colLabels=["Model", "Accuracy", "Target F1", "ROC-AUC"], loc="center")
    table.scale(1, 1.7)
    ax.set_title("Mean GroupKFold test metrics")
    fig.tight_layout()
    fig.savefig(output / "metrics_table.png", dpi=160)
    plt.close(fig)
    write_json(output / "training_metadata.json", {"feature_signature": info["signature"],
               "training": config["training"], "seed": config["seed"], "versions": package_versions(),
               "evaluation": "GroupKFold; fold mean and sample SD; target=1; uncalibrated decision scores",
               "final_models": "Refit on all subjects after evaluation; no held-out score for these refits."})
    LOGGER.info("\n%s", metrics[columns].to_string(index=False))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.artifacts:
        config["paths"]["artifacts"] = str(args.artifacts)
    with threadpool_limits(limits=1):
        run(config)


if __name__ == "__main__":
    main()
