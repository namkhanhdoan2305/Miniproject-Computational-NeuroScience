"""Shrinkage LDA: regularized covariance, equal priors, linear EEG baseline."""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

NAME = "LDA"
RATIONALE = "Linear ERP baseline; shrinkage stabilizes correlated features; equal class priors."


def build(config):
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=config["training"]["lda_priors"])
