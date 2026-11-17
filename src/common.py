"""Shared configuration, provenance, and memory-mapped artifact loading."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
LOGGER = logging.getLogger("oddball")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    p = config["preprocessing"]
    if not 0 < p["l_freq"] < p["h_freq"] < p["sfreq"] / 2:
        raise ValueError("Require 0 < l_freq < h_freq < output Nyquist frequency.")
    if not p["tmin"] <= p["baseline"][0] <= p["baseline"][1] <= 0 <= p["tmax"]:
        raise ValueError("Baseline must lie inside the epoch and end at/before stimulus onset.")
    for key in ("window", "p300_window"):
        lo, hi = config["features"][key]
        if not p["tmin"] <= lo < hi <= p["tmax"]:
            raise ValueError(f"features.{key} must lie inside the epoch.")
    if config["training"]["cv_folds"] < 2 or config["features"]["batch_size"] < 1:
        raise ValueError("Need at least two CV folds and a positive batch size.")
    return config


def artifact_dir(config):
    path = Path(config["paths"]["artifacts"]).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    """Replace a JSON file only after the entire new value has been written."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def package_versions():
    packages = ["numpy", "scipy", "mne", "scikit-learn", "PyWavelets", "huggingface_hub"]
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def load_epochs(directory):
    """Require a completed consolidation and validate its channel/time schema."""
    directory = Path(directory)
    meta = read_json(directory / "metadata.json")
    x = np.load(directory / "X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(directory / "y.npy", mmap_mode="r", allow_pickle=False)
    groups = np.load(directory / "groups.npy", mmap_mode="r", allow_pickle=False)
    channels = np.load(directory / "ch_names.npy", allow_pickle=False)
    times = np.load(directory / "times.npy", allow_pickle=False)
    if x.ndim != 3 or y.shape != (len(x),) or groups.shape != y.shape:
        raise ValueError("Epoch arrays have inconsistent shapes.")
    if x.shape[1:] != (len(channels), len(times)) or list(x.shape) != meta["shape"]:
        raise ValueError("Epoch schema does not match metadata.")
    if set(np.unique(y)) != {0, 1} or set(np.unique(groups)) != set(meta["subjects"]):
        raise ValueError("Require binary labels and the recorded subject identities.")
    if not np.allclose(np.diff(times), 1 / meta["sfreq"], atol=1e-9):
        raise ValueError("Time grid and saved sampling frequency disagree.")
    return x, y, groups, channels, times, meta
