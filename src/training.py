"""Shared evaluation engine; independent model runs with comparable provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from .common import LOGGER, artifact_dir, fingerprint, package_versions, read_json, write_json
from .diagnostics import run_diagnostics
from .evaluation import METRICS, classification_metrics, group_splits, positive_scores
from .features import load_features
from .models import MODELS, build_pipeline

PROTOCOL_VERSION = 3


def fit_checked(model, x, y):
    """Retry a slow iterative CPU solver once; never silently export a failed fit."""
    messages = []
    for attempt in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        for w in caught:
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn(str(w.message), w.category, stacklevel=2)
        if not convergence:
            model.fit_diagnostics_ = {"attempts": attempt + 1, "converged": True, "warnings": messages}
            return model
        messages.extend(convergence)
        params = {k: v * 2 for k, v in model.get_params(deep=True).items()
                  if k.endswith("__max_iter") and isinstance(v, int) and v > 0}
        LOGGER.warning("Convergence warning: %s; retry parameters: %s", convergence, params)
        if attempt == 0 and params:
            model.set_params(**params)
        else:
            break
    raise RuntimeError("Solver did not converge after retry. No completed model was exported: " + "; ".join(messages))


def context(config):
    root = artifact_dir(config)
    x, y, groups, names, info = load_features(root)
    if config["features"] != info["settings"]:
        raise ValueError("Feature settings changed; rerun src.features before training.")
    if not np.isfinite(x).all():
        raise ValueError("Training features must be finite.")
    splits = group_splits(x, y, groups, config["training"]["cv_folds"])
    manifest = [{"fold": i, "train_subjects": sorted(np.unique(groups[a]).tolist()),
                 "test_subjects": sorted(np.unique(groups[b]).tolist())}
                for i, (a, b) in enumerate(splits, 1)]
    protocol = {"version": PROTOCOL_VERSION, "feature_signature": info["signature"],
                "training": config["training"], "seed": config["seed"], "splits": manifest}
    return root, x, y, groups, names, info, splits, manifest, fingerprint(protocol)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_completed(directory, protocol_signature):
    marker = directory / "metadata.json"
    if not marker.exists():
        raise ValueError(f"Incomplete/missing run: {directory}. Train it first (or use --overwrite).")
    metadata = read_json(marker)
    if metadata["protocol_signature"] != protocol_signature:
        raise ValueError(f"Stale/incompatible model run: {directory}. Rerun with --overwrite.")
    for name, expected in metadata["files"].items():
        path = directory / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Missing/changed result: {path}; rerun with --overwrite.")
    return metadata


def save_model(path, model, names, info, config, fold=None):
    with path.open("wb") as handle:
        pickle.dump({"pipeline": model, "feature_names": names.tolist(),
                     "feature_settings": info["settings"], "feature_signature": info["signature"],
                     "training": config["training"], "seed": config["seed"],
                     "labels": {"standard": 0, "target": 1}, "fold": fold,
                     "versions": package_versions()}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def fit_metadata(model):
    estimator = model[-1]
    actual = getattr(estimator, "estimator_", estimator)
    iterations = np.asarray(getattr(actual, "n_iter_", []))
    return {"backend": getattr(estimator, "backend_", "cpu:sklearn"),
            "fallback_reason": getattr(estimator, "fallback_reason_", None),
            "solver_iterations": iterations.tolist(), **model.fit_diagnostics_}


def train_one(config, key, device="auto", resume=False, overwrite=False):
    if device not in {"cpu", "gpu", "auto"}:
        raise ValueError("device must be cpu, gpu or auto")
    if resume and overwrite:
        raise ValueError("Choose --resume or --overwrite, not both")
    root, x, y, groups, names, info, splits, manifest, signature = context(config)
    directory = root / "ml" / key
    marker = directory / "metadata.json"
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        if not resume:
            raise FileExistsError(f"{directory} exists; use --resume or --overwrite.")
        metadata = validate_completed(directory, signature)
        if metadata["requested_device"] != device:
            raise ValueError("Requested backend changed; use --overwrite.")
        LOGGER.info("Reusing completed %s", key)
        return pd.read_csv(directory / "metrics.csv")
    # A new run invalidates the aggregate, but never removes processed EEG/features.
    (root / "training_metadata.json").unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    # Delete only files owned by the previous model-run manifest. Avoid stale curves
    # when diagnostics are disabled on an explicit overwrite.
    for old in ("learning_curve.csv", "learning_curve.png", "training_progress.csv", "training_progress.png",
                "optimizer_diagnostics.json"):
        (directory / old).unlink(missing_ok=True)
    models = directory / "models"
    models.mkdir(exist_ok=True)
    name = MODELS[key].NAME
    if overwrite:
        for old_model in models.glob(f"{name.lower()}_*.pkl"):
            old_model.unlink()
    fold_rows, subject_rows, importances, fit_records = [], [], [], []
    predictions = np.full(len(y), -1, dtype=np.int8)
    scores, fold_ids = np.full(len(y), np.nan), np.zeros(len(y), dtype=int)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for fold, (train, test) in enumerate(splits, 1):
            LOGGER.info("%s | fold %d/%d | train=%s test=%s", name, fold, len(splits),
                        manifest[fold - 1]["train_subjects"], manifest[fold - 1]["test_subjects"])
            tick = time.perf_counter()
            model = fit_checked(build_pipeline(key, config, device), x[train], y[train])
            fit_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            predictions[test], scores[test] = model.predict(x[test]), positive_scores(model, x[test])
            predict_seconds = time.perf_counter() - tick
            if not np.isfinite(scores[test]).all():
                raise RuntimeError("Nonfinite held-out scores")
            fold_ids[test] = fold
            fit_record = {"fold": fold, **fit_metadata(model)}
            fit_records.append(fit_record)
            fold_rows.append({"model": name, "fold": fold, "n_test": len(test),
                              "target_prevalence": float(np.mean(y[test])), "backend": fit_record["backend"],
                              "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                              **classification_metrics(y[test], predictions[test], scores[test])})
            for subject in np.unique(groups[test]):
                indices = test[groups[test] == subject]
                subject_rows.append({"model": name, "subject": str(subject), "fold": fold,
                                     **classification_metrics(y[indices], predictions[indices], scores[indices])})
            if key == "random_forest":
                importances.extend({"fold": fold, "feature": str(feature), "importance": float(value)}
                                   for feature, value in zip(names, model[-1].feature_importances_))
            if config["training"]["save_fold_models"]:
                save_model(models / f"{name.lower()}_fold{fold}.pkl", model, names, info, config, fold)
        if (predictions < 0).any() or not np.isfinite(scores).all():
            raise RuntimeError("Every trial must have one finite outer held-out prediction")
        folds = pd.DataFrame(fold_rows)
        summary = {"model": name, "n_subjects": len(np.unique(groups)), "n_trials": len(y), "n_folds": len(splits)}
        for metric in (*METRICS, "fit_seconds", "predict_seconds"):
            summary[metric + "_mean"] = float(folds[metric].mean())
            summary[metric + "_std"] = float(folds[metric].std(ddof=1))
        summary.update({"backend": ",".join(sorted(set(folds.backend))),
                        "oof_accuracy": float(np.mean(predictions == y)),
                        "oof_f1": classification_metrics(y, predictions, scores)["f1"]})
        metrics = pd.DataFrame([summary])
        oof = pd.DataFrame({"model": name, "trial_index": np.arange(len(y)), "subject": groups,
                            "fold": fold_ids, "y_true": y, "y_pred": predictions, "score": scores})
        metrics.to_csv(directory / "metrics.csv", index=False)
        folds.to_csv(directory / "metrics_per_fold.csv", index=False)
        pd.DataFrame(subject_rows).to_csv(directory / "metrics_per_subject.csv", index=False)
        oof.to_csv(directory / "oof_predictions.csv", index=False)
        matrix = confusion_matrix(y, predictions, labels=[0, 1])
        pd.DataFrame(matrix, index=["true_standard", "true_target"], columns=["pred_standard", "pred_target"]).to_csv(
            directory / "confusion_matrix.csv")
        write_json(directory / "cv_splits.json", manifest)
        write_json(directory / "fit_diagnostics.json", fit_records)
        if importances:
            imp = pd.DataFrame(importances)
            imp.to_csv(directory / "feature_importance_per_fold.csv", index=False)
            imp.groupby("feature", sort=False).importance.agg(["mean", "std"]).reset_index().rename(
                columns={"mean": "importance_mean", "std": "importance_std"}).sort_values(
                    "importance_mean", ascending=False).to_csv(directory / "feature_importance.csv", index=False)
        # Curves only consume subjects in the FIRST OUTER TRAIN split.
        run_diagnostics(key, config, x, y, groups, splits[0][0], directory, device, fit_checked)
        LOGGER.info("Refitting %s on all subjects for deployment export (not evaluated)", name)
        final = fit_checked(build_pipeline(key, config, device), x, y)
        save_model(models / f"{name.lower()}_final.pkl", final, names, info, config)
        write_json(directory / "final_fit.json", fit_metadata(final))
    from .model_plots import export_model_figures
    export_model_figures(directory, metrics, folds, oof)
    write_json(directory / "metrics.json", {"results": [summary], "positive_class": "target=1",
               "aggregation": "Mean ± sample SD of outer subject-wise folds; SD is not a confidence interval"})
    metadata = {"protocol_version": PROTOCOL_VERSION, "protocol_signature": signature,
                "feature_signature": info["signature"], "requested_device": device, "model": name, "key": key,
                "rationale": MODELS[key].RATIONALE, "training": config["training"], "seed": config["seed"],
                "actual_backends": sorted(set(folds.backend)), "versions": package_versions(),
                "total_seconds": time.perf_counter() - started,
                "files": {str(path.relative_to(directory)): file_hash(path) for path in directory.rglob("*")
                          if path.is_file() and path != marker and path.suffix not in {".pkl", ".tmp"}}}
    # Include models in integrity checks, too. This is an explicitly requested run.
    metadata["files"].update({str(p.relative_to(directory)): file_hash(p) for p in models.glob("*.pkl")})
    write_json(marker, metadata)  # Completion marker only after metrics, models and figures.
    LOGGER.info("%s complete:\n%s", name, metrics.to_string(index=False))
    return metrics
