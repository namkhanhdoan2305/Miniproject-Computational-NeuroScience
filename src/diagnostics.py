"""Honest training diagnostics on INNER subjects, never outer held-out subjects."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .common import write_json
from .evaluation import positive_scores
from .models import build_pipeline


def inner_partition(y, groups, outer_train, seed):
    """Use only the first outer fold's TRAIN subjects for all diagnostic curves."""
    unique = np.unique(groups[outer_train])
    if len(unique) < 2:
        return None
    a, b = next(GroupKFold(n_splits=min(3, len(unique))).split(
        np.zeros(len(outer_train)), y[outer_train], groups[outer_train]))
    train, valid = outer_train[a], outer_train[b]
    if set(np.unique(y[train])) != {0, 1} or set(np.unique(y[valid])) != {0, 1}:
        raise ValueError("Diagnostic inner subjects must contain both classes.")
    ordered = np.unique(groups[train])
    np.random.default_rng(seed).shuffle(ordered)
    return train, valid, ordered


def logistic_trace(x, y, xv, yv, c, max_iter):
    """A real L-BFGS callback trace, separate CPU diagnostic, same L2 objective.

    Objective = weighted mean logistic loss + ||w||² / (2*C*sum(weights)).
    Intercept is not penalized. Validation loss is unweighted and never optimized.
    """
    weights = compute_sample_weight("balanced", y)
    total = weights.sum()
    rows = []
    def objective(beta):
        logits = x @ beta[:-1] + beta[-1]
        residual = weights * (expit(logits) - y) / total
        loss = np.dot(weights, np.logaddexp(0, logits) - y * logits) / total
        loss += np.dot(beta[:-1], beta[:-1]) / (2 * c * total)
        gradient = np.r_[x.T @ residual + beta[:-1] / (c * total), residual.sum()]
        return float(loss), gradient
    def record(beta):
        rows.append({"iteration": len(rows), "train_objective": objective(beta)[0],
                     "train_log_loss": float(log_loss(y, expit(x @ beta[:-1] + beta[-1]), labels=[0, 1])),
                     "validation_log_loss": float(log_loss(yv, expit(xv @ beta[:-1] + beta[-1]), labels=[0, 1]))})
    beta = np.zeros(x.shape[1] + 1)
    record(beta)
    result = minimize(objective, beta, jac=True, method="L-BFGS-B", callback=record,
                      options={"maxiter": max_iter, "gtol": 1e-6, "ftol": 1e-9})
    return pd.DataFrame(rows), {"success": bool(result.success), "message": str(result.message),
                               "iterations": int(result.nit), "backend": "cpu:scipy diagnostic only"}


def run_diagnostics(key, config, x, y, groups, outer_train, output, device, fit_checked):
    settings = config["training"]["diagnostics"]
    if not settings["enabled"]:
        write_json(output / "diagnostics.json", {"status": "disabled"})
        return
    partition = inner_partition(y, groups, outer_train, config["seed"])
    if partition is None:
        write_json(output / "diagnostics.json", {"status": "insufficient_inner_subjects"})
        return
    train, valid, ordered = partition
    metadata = {
        "status": "complete", "scope": "First outer TRAIN fold, further split into disjoint inner subjects",
        "train_subjects": sorted(np.unique(groups[train]).tolist()),
        "validation_subjects": sorted(np.unique(groups[valid]).tolist()),
        "outer_test_used": False, "warning": "Curves are diagnostics, not the outer benchmark or a hyperparameter selection result.",
    }
    counts = sorted({max(1, min(len(ordered), int(np.ceil(f * len(ordered))))) for f in settings["fractions"]})
    rows = []
    for count in counts:
        chosen = train[np.isin(groups[train], ordered[:count])]
        model = build_pipeline(key, config, device)
        start = time.perf_counter()
        fit_checked(model, x[chosen], y[chosen])
        row = {"n_subjects": count, "n_trials": len(chosen), "fit_seconds": time.perf_counter() - start}
        for prefix, indices in (("train", chosen), ("validation", valid)):
            prediction = model.predict(x[indices])
            row[prefix + "_balanced_accuracy"] = balanced_accuracy_score(y[indices], prediction)
            row[prefix + "_f1"] = f1_score(y[indices], prediction, zero_division=0)
            row[prefix + "_roc_auc"] = roc_auc_score(y[indices], positive_scores(model, x[indices]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "learning_curve.csv", index=False)
    if key == "logistic_regression":
        scaler = StandardScaler().fit(x[train])
        trace, convergence = logistic_trace(scaler.transform(x[train]), y[train], scaler.transform(x[valid]), y[valid],
                                            config["training"]["logistic_c"], settings["logistic_max_iter"])
        trace.to_csv(output / "training_progress.csv", index=False)
        write_json(output / "optimizer_diagnostics.json", convergence)
    elif key == "random_forest":
        model = build_pipeline(key, config, "cpu")
        scaler = model[0].fit(x[train])
        xt, xv = scaler.transform(x[train]), scaler.transform(x[valid])
        forest = clone(model[-1])
        weights = compute_sample_weight("balanced", y[train])
        class_weights = {int(label): float(weights[y[train] == label][0]) for label in np.unique(y[train])}
        # Same fixed inner training set at every tree count; no OOB trial leakage.
        forest.set_params(warm_start=True, class_weight=class_weights)
        maximum = config["training"]["rf_estimators"]
        steps = sorted({max(1, int(np.ceil(f * maximum))) for f in settings["fractions"]} | {maximum})
        progress = []
        for count in steps:
            forest.set_params(n_estimators=count)
            forest.fit(xt, y[train])
            progress.append({"n_trees": count,
                             "train_log_loss": log_loss(y[train], forest.predict_proba(xt), labels=[0, 1]),
                             "validation_log_loss": log_loss(y[valid], forest.predict_proba(xv), labels=[0, 1])})
        pd.DataFrame(progress).to_csv(output / "training_progress.csv", index=False)
    write_json(output / "diagnostics.json", metadata)
