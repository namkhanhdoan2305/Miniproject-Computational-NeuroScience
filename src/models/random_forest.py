"""Balanced Random Forest: nonlinear interactions and descriptive importance."""
from sklearn.ensemble import RandomForestClassifier

NAME = "RandomForest"
RATIONALE = "Balanced ensemble tests nonlinear feature interactions; impurity importance is descriptive."


def build(config):
    t = config["training"]
    return RandomForestClassifier(n_estimators=t["rf_estimators"], min_samples_leaf=t["rf_min_samples_leaf"],
                                  max_depth=t.get("rf_max_depth"), class_weight="balanced",
                                  n_jobs=t["n_jobs"], random_state=config["seed"])
