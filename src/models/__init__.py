"""One model per file; training, metrics, and split logic remain shared."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import lda, svm, random_forest, logistic_regression

MODELS = {"lda": lda, "svm": svm, "random_forest": random_forest, "logistic_regression": logistic_regression}


def build_pipeline(key, config, device="cpu"):
    estimator = MODELS[key].build(config)
    if device != "cpu" and key in {"svm", "logistic_regression"}:
        from ..backends import OptionalGPUClassifier
        estimator = OptionalGPUClassifier(kind=key, cpu_estimator=estimator, device=device)
    elif device == "gpu":
        raise ValueError(f"{key} uses CPU to preserve shrinkage/priors or balanced class weights. Use --device auto.")
    return make_pipeline(StandardScaler(), estimator)
