"""Thin CLI: train one model, or train all four and compare their completed runs."""
from __future__ import annotations
import argparse
from pathlib import Path

# Also support the original 'python src/train_ml.py' invocation.
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .common import DEFAULT_CONFIG, load_config, setup_logging
from .evaluation import METRICS, classification_metrics, group_splits, positive_scores
from .models import MODELS, build_pipeline
from .training import fit_checked, train_one


def build_models(config):
    """Compatibility API: the four original CPU pipelines."""
    return {module.NAME: build_pipeline(key, config, "cpu") for key, module in MODELS.items()}


def run(config, model="all", device=None, resume=False, overwrite=False):
    device = device or config["training"].get("device", "auto")
    if model != "all":
        return train_one(config, model, device, resume, overwrite)
    for key in MODELS:
        train_one(config, key, device, resume, overwrite)
    from .compare_ml import run as compare
    return compare(config)


def main(default_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--model", choices=[*MODELS, "all"], default=default_model or "all")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], help="auto: GPU for supported weighted models, otherwise CPU")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse a fully completed compatible model run")
    mode.add_argument("--overwrite", action="store_true", help="Replace this model run; preserve processed EEG/features")
    args = parser.parse_args()
    if default_model and args.model != default_model:
        parser.error(f"This entrypoint trains {default_model}; use src.train_ml for other models")
    setup_logging()
    config = load_config(args.config)
    if args.artifacts:
        config["paths"]["artifacts"] = str(args.artifacts)
    run(config, args.model, args.device, args.resume, args.overwrite)


if __name__ == "__main__":
    main()
