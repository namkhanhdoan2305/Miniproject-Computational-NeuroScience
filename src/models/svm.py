"""Balanced RBF SVM: test nonlinear separation of the same EEG features."""
from sklearn.svm import SVC

NAME = "SVM"
RATIONALE = "Balanced RBF classifier tests nonlinear separation; margins are not calibrated probabilities."


def build(config):
    t = config["training"]
    return SVC(kernel="rbf", C=t["svm_c"], gamma=t["svm_gamma"], class_weight="balanced",
               cache_size=t["svm_cache_mb"], random_state=config["seed"])
