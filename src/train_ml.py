"""Leakage-safe subject CV, four imbalance-aware classifiers, and report exports."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

try:
    from .common import (DEFAULT_CONFIG, LOGGER, artifact_dir, load_config,
                         package_versions, setup_logging, write_json)
    from .features import load_features
except ImportError:
    from common import (DEFAULT_CONFIG, LOGGER, artifact_dir, load_config,
                        package_versions, setup_logging, write_json)
    from features import load_features

METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")


def build_models(config):
    """All preprocessing estimators are fit only inside each training pipeline."""
    t = config["training"]
    return {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=t["lda_priors"],
        )),
        "SVM": make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=t["svm_c"], gamma=t["svm_gamma"], class_weight="balanced",
            cache_size=t["svm_cache_mb"], random_state=config["seed"],
        )),
        "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=t["rf_estimators"], min_samples_leaf=t["rf_min_samples_leaf"],
            class_weight="balanced", n_jobs=t["n_jobs"], random_state=config["seed"],
        )),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(
            # The lbfgs default is L2; omitting the deprecated penalty argument
            # supports both older Colab releases and scikit-learn >= 1.8.
            C=t["logistic_c"], solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=config["seed"],
        )),
    }


def group_splits(x, y, groups, n_splits):
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"GroupKFold needs at least {n_splits} subjects; found {len(np.unique(groups))}. "
                         "A one-subject dry-run cannot train subject-wise CV.")
    splits = list(GroupKFold(n_splits=n_splits).split(x, y, groups))
    coverage = np.zeros(len(y), dtype=int)
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Subject leakage detected in CV split.")
        if set(np.unique(y[train])) != {0, 1} or set(np.unique(y[test])) != {0, 1}:
            raise ValueError("Every train/test fold must contain standard and target trials.")
        coverage[test] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Each trial must be tested exactly once.")
    return splits


def positive_scores(model, x):
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    index = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(x)[:, index]


def classification_metrics(y, predictions, scores):
    """Target=1 for precision/recall/F1; ranking metrics use continuous scores."""
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
    }


def fit_checked(model, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return model.fit(x, y)


def save_model(path, model, names, info, config, fold=None):
    with path.open("wb") as handle:
        pickle.dump({"pipeline": model, "feature_names": names.tolist(),
                     "feature_settings": info["settings"], "feature_signature": info["signature"],
                     "training": config["training"], "seed": config["seed"],
                     "labels": {"standard": 0, "target": 1}, "fold": fold,
                     "versions": package_versions()}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(config, x, y, groups, names, info, output, splits):
    """Produce held-out predictions before fitting any exported all-data models."""
    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)
    fold_rows, oof_rows, summary_rows, importance_rows, subject_rows = [], [], [], [], []
    for name, prototype in build_models(config).items():
        predictions = np.full(len(y), -1, dtype=np.int8)
        scores, fold_ids = np.full(len(y), np.nan), np.zeros(len(y), dtype=int)
        rows = []
        for fold, (train, test) in enumerate(splits, 1):
            LOGGER.info("%s | fold %d/%d | %d train / %d test subjects", name, fold, len(splits),
                        len(np.unique(groups[train])), len(np.unique(groups[test])))
            model = fit_checked(clone(prototype), x[train], y[train])
            predictions[test] = model.predict(x[test])
            scores[test] = positive_scores(model, x[test])
            fold_ids[test] = fold
            row = {"model": name, "fold": fold, "n_test": len(test),
                   "target_prevalence": float(np.mean(y[test])),
                   **classification_metrics(y[test], predictions[test], scores[test])}
            rows.append(row)
            fold_rows.append(row)
            for subject in np.unique(groups[test]):
                indices = test[groups[test] == subject]
                subject_rows.append({"model": name, "subject": str(subject), "fold": fold,
                                     **classification_metrics(y[indices], predictions[indices], scores[indices])})
            if name == "RandomForest":
                importance_rows.extend({"fold": fold, "feature": str(feature), "importance": float(value)}
                                       for feature, value in zip(names, model[-1].feature_importances_))
            if config["training"]["save_fold_models"]:
                save_model(models_dir / f"{name.lower()}_fold{fold}.pkl", model, names, info, config, fold)
        if (predictions < 0).any() or not np.isfinite(scores).all():
            raise RuntimeError("Some trials have no finite held-out predictions/scores.")
        summary = {"model": name, "n_subjects": len(np.unique(groups)), "n_trials": len(y), "n_folds": len(splits)}
        for metric in METRICS:
            values = np.array([row[metric] for row in rows])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        summary["oof_accuracy"] = float(accuracy_score(y, predictions))
        summary["oof_f1"] = float(f1_score(y, predictions, pos_label=1, zero_division=0))
        summary_rows.append(summary)
        oof_rows.append(pd.DataFrame({"model": name, "trial_index": np.arange(len(y)), "subject": groups,
                                      "fold": fold_ids, "y_true": y, "y_pred": predictions, "score": scores}))
        matrix = confusion_matrix(y, predictions, labels=[0, 1])
        pd.DataFrame(matrix, index=["true_standard", "true_target"], columns=["pred_standard", "pred_target"]).to_csv(
            output / f"confusion_matrix_{name.lower()}.csv")
        LOGGER.info("Refitting %s on all subjects for export", name)
        final = fit_checked(clone(prototype), x, y)
        save_model(models_dir / f"{name.lower()}_final.pkl", final, names, info, config)
    pd.DataFrame(fold_rows).to_csv(output / "metrics_per_fold.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(output / "metrics_per_subject.csv", index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(output / "oof_predictions.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(output / "feature_importance_per_fold.csv", index=False)
    importance.groupby("feature", sort=False).importance.agg(["mean", "std"]).reset_index().rename(
        columns={"mean": "importance_mean", "std": "importance_std"}).sort_values(
        "importance_mean", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    return pd.DataFrame(summary_rows)


def run(config):
    output = artifact_dir(config)
    x, y, groups, names, info = load_features(output)
    if config["features"] != info["settings"]:
        raise ValueError("Feature settings changed; rerun src.features before training.")
    if not np.isfinite(x).all():
        raise ValueError("Training features must be finite.")
    splits = group_splits(x, y, groups, config["training"]["cv_folds"])
    # A missing completion manifest indicates an interrupted run. Plot-only commands
    # cannot reuse results from a different source dataset/feature schema.
    marker = output / "training_metadata.json"
    marker.unlink(missing_ok=True)
    write_json(output / "cv_splits.json", [{
        "fold": fold, "train_subjects": sorted(set(groups[train].tolist())),
        "test_subjects": sorted(set(groups[test].tolist())), "n_train": len(train), "n_test": len(test),
    } for fold, (train, test) in enumerate(splits, 1)])
    metrics = evaluate(config, x, y, groups, names, info, output, splits)
    metrics.to_csv(output / "ml_benchmark_results.csv", index=False)
    metrics.to_csv(output / "metrics_summary.csv", index=False)  # Backward-compatible name.
    report = {
        "positive_class": "target (1)", "splitter": "GroupKFold", "n_splits": len(splits),
        "aggregation": "Unweighted fold mean; sample standard deviation (ddof=1). SD is not a confidence interval.",
        "score_units": "Fractions in [0,1]; target precision/recall/F1; AP is average precision, not trapezoidal PR AUC.",
        "zero_division": 0, "feature_signature": info["signature"],
        "dataset_id": info["source"]["dataset_id"], "subjects": sorted(np.unique(groups).tolist()),
        "training_config": config["training"], "seed": config["seed"], "versions": package_versions(),
        "class_counts": {"standard": int((y == 0).sum()), "target": int((y == 1).sum())},
        "majority_baseline": {"accuracy": float(max(np.mean(y == 0), np.mean(y == 1))), "balanced_accuracy": 0.5},
        "results": metrics.to_dict(orient="records"),
        "caveats": ["Fold training sets overlap; fold SD does not imply independent replicates.",
                    "Model comparison is exploratory; selecting the best model requires independent confirmation.",
                    "Per-recording unlabeled ICA and zero-phase filters assume offline access to the complete recording."],
    }
    write_json(output / "ml_benchmark_results.json", report)
    provenance = {"feature_signature": info["signature"], "training": config["training"],
                  "seed": config["seed"], "versions": package_versions(),
                  "evaluation": report["aggregation"], "class_counts": report["class_counts"],
                  "feature_importance": "Training-fold Random Forest mean decrease in impurity; descriptive, noncausal.",
                  "final_models": "Refit after each model's CV; these refits have no held-out performance estimate."}
    # Plot generation is part of the required training deliverables. The manifest
    # is passed in memory and committed to disk only when all figures succeed.
    try:
        from .visualize import export_all_figures
    except ImportError:
        from visualize import export_all_figures
    export_all_figures(output, config, provenance)
    provenance["required_deliverables"] = ["ml_benchmark_results.csv", "ml_benchmark_results.json",
                                         *[f"fig{i}_{suffix}.png" for i, suffix in enumerate(
                                             ["grand_average_erp", "roc_and_pr_curves", "confusion_matrices", "feature_importance"], 1)]]
    write_json(marker, provenance)
    LOGGER.info("\n%s", metrics[["model", *[f"{metric}_mean" for metric in METRICS]]].to_string(index=False))
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
