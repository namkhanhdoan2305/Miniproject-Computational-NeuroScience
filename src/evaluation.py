"""Shared subject splits and target-class metrics for every model."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")

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
