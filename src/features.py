"""Per-trial spectral, entropy, wavelet, and P300 features without fitted transforms."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pywt
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.special import xlogy

FEATURE_SCHEMA_VERSION = 2

try:
    from .common import (DEFAULT_CONFIG, LOGGER, artifact_dir, fingerprint, load_config,
                         load_epochs, package_versions, read_json, setup_logging, write_json)
except ImportError:
    from common import (DEFAULT_CONFIG, LOGGER, artifact_dir, fingerprint, load_config,
                        load_epochs, package_versions, read_json, setup_logging, write_json)


def normalized_entropy(weights):
    """Shannon entropy in [0, 1]; all-zero and single-bin distributions return 0."""
    if weights.shape[-1] < 2:
        return np.zeros(weights.shape[:-1], dtype=float)
    total = weights.sum(axis=-1, keepdims=True)
    probabilities = np.divide(weights, total, out=np.zeros_like(weights, dtype=float), where=total > 0)
    return -np.sum(xlogy(probabilities, probabilities), axis=-1) / np.log(weights.shape[-1])


def histogram_entropy(x, bins):
    """Amplitude-histogram Shannon entropy; bins span each trial/channel's range."""
    if bins < 2:
        raise ValueError("entropy_bins must be at least 2.")
    rows = x.reshape(-1, x.shape[-1])
    low = rows.min(axis=-1, keepdims=True)
    span = np.ptp(rows, axis=-1, keepdims=True)
    scaled = np.divide(rows - low, span, out=np.zeros_like(rows), where=span > 0)
    indices = np.minimum((scaled * bins).astype(int), bins - 1)
    indices += np.arange(len(rows))[:, None] * bins
    counts = np.bincount(indices.ravel(), minlength=len(rows) * bins).reshape(len(rows), bins)
    return normalized_entropy(counts).reshape(x.shape[:-1])


def integrate_band(psd, frequencies, low, high):
    """Integrate PSD with interpolated band boundaries, including sub-bin edges."""
    if not frequencies[0] <= low < high <= frequencies[-1]:
        raise ValueError(f"Band [{low}, {high}] is outside the available PSD grid.")
    def interpolate(edge):
        right = int(np.clip(np.searchsorted(frequencies, edge, side="right"), 1, len(frequencies) - 1))
        left = right - 1
        weight = (edge - frequencies[left]) / (frequencies[right] - frequencies[left])
        return psd[..., left] * (1 - weight) + psd[..., right] * weight
    interior = (frequencies > low) & (frequencies < high)
    grid = np.r_[low, frequencies[interior], high]
    values = np.concatenate([interpolate(low)[..., None], psd[..., interior],
                             interpolate(high)[..., None]], axis=-1)
    return trapezoid(values, x=grid, axis=-1)


def extract_batch(x, sfreq, times, channels, settings):
    """Return (trials, features), names; X is in volts and amplitude features use µV."""
    x = np.asarray(x, dtype=np.float64) * 1e6
    times = np.asarray(times)
    if x.ndim != 3 or x.shape[1:] != (len(channels), len(times)) or not np.isfinite(x).all():
        raise ValueError("Expected finite (trials, channels, samples) EEG with matching schema.")
    epsilon = 1e-9 / sfreq
    window = (times >= settings["window"][0] - epsilon) & (times <= settings["window"][1] + epsilon)
    p300 = (times >= settings["p300_window"][0] - epsilon) & (times <= settings["p300_window"][1] + epsilon)
    if window.sum() < 8 or p300.sum() < 2:
        raise ValueError("Insufficient samples in feature or P300 windows.")
    signal = x[..., window]
    nperseg = min(signal.shape[-1], int(round(settings["welch_seconds"] * sfreq)))
    if nperseg < 8:
        raise ValueError("Welch window must contain at least eight samples.")
    frequencies, psd = welch(signal, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2,
                             detrend="constant", scaling="density", axis=-1)
    blocks, suffixes = [], []
    def add(values, name):
        blocks.append(values)
        suffixes.append(name)
    for band, (low, high) in settings["bands"].items():
        add(np.log10(np.maximum(integrate_band(psd, frequencies, low, high), 1e-20)), f"log10_power_{band}")
    limits = (min(b[0] for b in settings["bands"].values()), max(b[1] for b in settings["bands"].values()))
    spectral = psd[..., (frequencies >= limits[0]) & (frequencies <= limits[1])]
    if spectral.shape[-1] < 2:
        raise ValueError("Spectral entropy requires at least two PSD bins.")
    add(normalized_entropy(spectral), "spectral_entropy")
    add(histogram_entropy(signal, settings["entropy_bins"]), "shannon_entropy")
    wavelet = pywt.Wavelet(settings["wavelet"])
    level = int(settings["wavelet_level"])
    if not 1 <= level <= pywt.dwt_max_level(signal.shape[-1], wavelet.dec_len):
        raise ValueError("wavelet_level exceeds the supported level for the feature window.")
    coefficients = pywt.wavedec(signal, wavelet, mode="symmetric", level=level, axis=-1)
    labels = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    for label, coefficient in zip(labels, coefficients):
        add(np.mean(np.abs(coefficient), axis=-1), f"wavelet_{label}_mean_abs_uv")
        add(np.std(coefficient, axis=-1), f"wavelet_{label}_std_uv")
        add(np.log10(np.maximum(np.mean(coefficient ** 2, axis=-1), 1e-20)), f"wavelet_{label}_log10_energy")
    response = x[..., p300]
    add(response.mean(axis=-1), "p300_mean_uv")
    add(response.max(axis=-1), "p300_peak_uv")  # Positive peak, not absolute magnitude.
    add(response.std(axis=-1), "p300_std_uv")
    add(times[p300][response.argmax(axis=-1)] * 1000, "p300_peak_latency_ms")
    add(trapezoid(response, x=times[p300], axis=-1), "p300_auc_uv_s")
    # Channel-major order, then the same feature suffixes for every channel.
    values = np.stack(blocks, axis=-1).reshape(len(x), -1).astype(np.float32)
    names = [f"{channel}__{suffix}" for channel in channels for suffix in suffixes]
    roi = settings.get("roi_channels", [])
    if roi:
        if len(set(roi)) != len(roi) or not set(roi).issubset(channels):
            raise ValueError(f"P300 ROI channels must be unique and available: {roi}")
        # Average channel waveforms before taking the peak/latency, rather than
        # averaging unrelated individual-channel peaks.
        indices = [list(channels).index(channel) for channel in roi]
        roi_response = response[:, indices, :].mean(axis=1)
        roi_values = np.column_stack([
            roi_response.mean(axis=-1), roi_response.max(axis=-1),
            roi_response.std(axis=-1), times[p300][roi_response.argmax(axis=-1)] * 1000,
            trapezoid(roi_response, x=times[p300], axis=-1),
        ])
        roi_suffixes = ["p300_mean_uv", "p300_peak_uv", "p300_std_uv", "p300_peak_latency_ms", "p300_auc_uv_s"]
        names.extend(f"Midline_{'_'.join(roi)}__{suffix}" for suffix in roi_suffixes)
        values = np.column_stack([values, roi_values]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Feature extraction produced nonfinite values.")
    return values, names


def run(config):
    output = artifact_dir(config)
    x, _, _, channels, times, metadata = load_epochs(output)
    settings = config["features"]
    source = {"dataset_id": metadata["dataset_id"], "extractor_version": FEATURE_SCHEMA_VERSION, "features": settings,
              "sfreq": metadata["sfreq"], "channels": channels.tolist()}
    signature = fingerprint(source)
    marker = output / "feature_metadata.json"
    marker.unlink(missing_ok=True)
    temporary = output / ".features.tmp.npy"
    features = None
    try:
        for start in range(0, len(x), settings["batch_size"]):
            stop = min(start + settings["batch_size"], len(x))
            values, names = extract_batch(x[start:stop], metadata["sfreq"], times, channels, settings)
            if features is None:
                features = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(x), values.shape[1]))
            features[start:stop] = values
        shape = list(features.shape)
        features.flush()
        del features
        os.replace(temporary, output / "features.npy")
        np.save(output / "feature_names.npy", np.asarray(names, dtype=str), allow_pickle=False)
        write_json(marker, {"signature": signature, "source": source, "shape": shape,
                            "versions": package_versions(), "settings": settings})
    finally:
        temporary.unlink(missing_ok=True)
    LOGGER.info("Extracted feature matrix %s into %s", shape, output / "features.npy")


def load_features(directory):
    directory = Path(directory)
    _, y, groups, _, _, metadata = load_epochs(directory)
    info = read_json(directory / "feature_metadata.json")
    if info["source"].get("extractor_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema changed; rerun src.features before training.")
    if info["source"]["dataset_id"] != metadata["dataset_id"]:
        raise ValueError("Features are stale after preprocessing; rerun src.features.")
    values = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    names = np.load(directory / "feature_names.npy", allow_pickle=False)
    if values.shape != (len(y), len(names)) or list(values.shape) != info["shape"]:
        raise ValueError("Features and label/schema arrays disagree.")
    return values, y, groups, names, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.artifacts:
        config["paths"]["artifacts"] = str(args.artifacts)
    run(config)


if __name__ == "__main__":
    main()
