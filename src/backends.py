"""Optional NVIDIA cuML for weighted SVM/logistic regression, with explicit fallback."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import logging
import sys

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

LOGGER = logging.getLogger("oddball")


def gpu_status():
    """Lazy probe: importing the project never requires CUDA or cuML."""
    if sys.platform != "linux":
        return False, "cuML GPU backend requires Linux and an NVIDIA CUDA GPU"
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False, "No NVIDIA CUDA device found"
        import cuml  # noqa: F401
        return True, "NVIDIA CUDA + cuML available"
    except (ImportError, RuntimeError, OSError) as exc:
        return False, f"CUDA/cuML unavailable: {type(exc).__name__}: {exc}"
    except Exception as exc:
        # CUDA runtime errors have backend-specific exception types.
        return False, f"GPU probe failed: {type(exc).__name__}: {exc}"


def _gpu_estimator(kind, cpu):
    """Keep balanced weights and C/gamma; never weaken the statistical protocol."""
    if kind == "svm":
        from cuml.svm import SVC
        return SVC(C=cpu.C, kernel="rbf", gamma=cpu.gamma, class_weight="balanced",
                   cache_size=cpu.cache_size, tol=cpu.tol, output_type="numpy")
    if kind == "logistic_regression":
        from cuml.linear_model import LogisticRegression
        return LogisticRegression(C=cpu.C, class_weight="balanced", max_iter=cpu.max_iter,
                                  tol=cpu.tol, penalty_normalized=True, output_type="numpy")
    raise ValueError(f"Unsupported GPU model: {kind}")


class OptionalGPUClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable sklearn adapter. auto falls back with a recorded reason; gpu fails."""
    def __init__(self, kind, cpu_estimator, device="auto"):
        self.kind = kind
        self.cpu_estimator = cpu_estimator
        self.device = device

    def fit(self, x, y):
        self.fallback_reason_ = None
        available, reason = gpu_status()
        if available:
            try:
                self.estimator_ = _gpu_estimator(self.kind, self.cpu_estimator)
                self.estimator_.fit(np.ascontiguousarray(x, dtype=np.float64), np.asarray(y))
                status = getattr(self.estimator_, "fit_status_", 0)
                if int(np.asarray(status).item()) != 0:
                    raise RuntimeError("GPU solver did not converge")
                iterations = np.asarray(getattr(self.estimator_, "n_iter_", 0))
                if self.kind == "logistic_regression" and iterations.max() >= self.cpu_estimator.max_iter:
                    raise RuntimeError("GPU logistic solver reached max_iter")
                self.backend_ = "gpu:cuml"
                try:
                    self.backend_version_ = version("cuml-cu12")
                except PackageNotFoundError:
                    self.backend_version_ = "unknown"
            except Exception as exc:
                if self.device == "gpu":
                    raise RuntimeError("Requested GPU fit failed; CPU fallback is disabled") from exc
                self.fallback_reason_ = f"GPU fit failed: {type(exc).__name__}: {exc}"
        else:
            if self.device == "gpu":
                raise RuntimeError(reason + "; install requirements-gpu.txt on a supported Colab GPU runtime")
            self.fallback_reason_ = reason
        if self.fallback_reason_:
            LOGGER.warning("%s: using CPU (%s)", self.kind, self.fallback_reason_)
            self.estimator_ = clone(self.cpu_estimator).fit(x, y)
            self.backend_ = "cpu:sklearn"
            self.backend_version_ = None
        self.classes_ = np.asarray(self.estimator_.classes_).reshape(-1)
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x):
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict(x)).reshape(-1)

    def decision_function(self, x):
        check_is_fitted(self, "estimator_")
        if hasattr(self.estimator_, "decision_function"):
            return np.asarray(self.estimator_.decision_function(x)).reshape(-1)
        if self.kind == "logistic_regression":
            # cuML releases expose probabilities but not always decision_function.
            coefficients = np.asarray(self.estimator_.coef_)
            intercept = np.asarray(self.estimator_.intercept_).reshape(-1)
            if coefficients.shape[0] == 2:
                return np.asarray(x) @ (coefficients[1] - coefficients[0]) + intercept[1] - intercept[0]
            return np.asarray(x) @ coefficients.reshape(-1) + intercept.item()
        raise AttributeError("Backend has no continuous decision score")

    def predict_proba(self, x):
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict_proba(x))
