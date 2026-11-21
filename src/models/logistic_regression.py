"""Balanced L2 logistic regression: regularized linear discriminative baseline."""
from sklearn.linear_model import LogisticRegression

NAME = "LogisticRegression"
RATIONALE = "Balanced L2 linear classifier; compare with generative LDA using identical folds."


def build(config):
    t = config["training"]
    return LogisticRegression(C=t["logistic_c"], solver="lbfgs", max_iter=t.get("max_iter", 2000),
                              class_weight="balanced", random_state=config["seed"])
