"""Download, clean, and checkpoint one subject at a time (no datasets library)."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import inspect
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import uuid
import warnings

# Avoid a persistent Xet chunk cache outside the disposable download directory.
# Must be set before importing huggingface_hub, including its constants.
os.environ["HF_HUB_DISABLE_XET"] = "1"

import mne
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

PREPROCESSING_VERSION = 2

try:
    from .common import (DEFAULT_CONFIG, LOGGER, artifact_dir, fingerprint, load_config,
                         package_versions, read_json, setup_logging, write_json)
except ImportError:  # python src/preprocess.py
    from common import (DEFAULT_CONFIG, LOGGER, artifact_dir, fingerprint, load_config,
                        package_versions, read_json, setup_logging, write_json)


def normalize_event(value):
    return re.sub(r"\s+", "", str(value)).casefold()


def read_events(path, raw, config):
    """Use BIDS onsets in seconds; never infer class meaning from event frequency."""
    table = pd.read_csv(path, sep="\t")
    column = config["events"]["column"]
    if column not in table or "onset" not in table:
        raise ValueError(f"{path}: require onset and {column!r}; found {list(table)}")
    mapping = {}
    for name, code in (("standard", 1), ("target", 2), ("distractor", 3)):
        for alias in config["events"][name]:
            key = normalize_event(alias)
            if key in mapping and mapping[key] != code:
                raise ValueError(f"Ambiguous event alias: {alias}")
            mapping[key] = code
    mapped = table[column].map(normalize_event).map(mapping)
    # Fail on unrecognized stimulus codes, while ignoring responses and run markers.
    if "trial_type" in table:
        stimulus = table["trial_type"].astype(str).str.casefold().eq("stimulus")
        unknown = table.loc[stimulus & mapped.isna(), column].unique().tolist()
        if unknown:
            raise ValueError(f"Unmapped stimulus codes in {path}: {unknown}")
    counts = {name: int((mapped == code).sum()) for name, code in
              (("standard", 1), ("target", 2), ("distractor", 3))}
    keep = mapped.isin([1, 2])
    if not counts["standard"] or not counts["target"]:
        raise ValueError(f"Both retained classes must be present in {path}; counts={counts}")
    onsets = pd.to_numeric(table.loc[keep, "onset"], errors="raise").to_numpy(float)
    if not np.isfinite(onsets).all() or (onsets < 0).any() or (onsets > raw.times[-1]).any():
        raise ValueError(f"Nonfinite or out-of-recording onsets in {path}")
    # BrainVision marker positions are one-based. The TSV is authoritative here;
    # do not mix its onsets with annotations or resampled sample indices.
    samples = raw.time_as_index(onsets, use_rounding=True) + raw.first_samp
    events = np.column_stack([samples, np.zeros(len(samples), int), mapped[keep].astype(int)])
    events = events[np.argsort(events[:, 0], kind="stable")].astype(int)
    if len(np.unique(events[:, 0])) != len(events):
        raise ValueError(f"Duplicate retained event samples in {path}")
    return events, counts


def prepare_channels(raw, header, config):
    sidecar = header.with_name(header.name.replace("_eeg.vhdr", "_channels.tsv"))
    if sidecar.exists():
        # The source channels.tsv uses a legacy encoding for the microvolt symbol.
        table = pd.read_csv(sidecar, sep="\t", encoding="latin-1")
        kinds = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "EMG": "emg", "MISC": "misc"}
        types = {row["name"]: kinds.get(str(row["type"]).upper(), "misc")
                 for _, row in table.iterrows() if row["name"] in raw.ch_names}
        raw.set_channel_types(types, verbose=False)
        if "status" in table:
            raw.info["bads"] = table.loc[table.status.eq("bad"), "name"].tolist()
    rename = {name: {"TP9 LEFT EAR": "TP9", "TP10 RIGHT EAR": "TP10"}.get(name, name.strip())
              for name in raw.ch_names}
    raw.rename_channels(rename)
    configured_eog = config["preprocessing"]["ica"]["eog_channels"]
    if not set(configured_eog).issubset(raw.ch_names):
        raise ValueError("Configured EOG reference is absent from the recording.")
    # Explicit EOG overrides can fix sidecars that incorrectly label EOG as EEG.
    requested = config["preprocessing"]["channels"]
    if requested is not None and set(configured_eog) & set(requested):
        raise ValueError("An EOG reference cannot also be a requested EEG feature channel.")
    if configured_eog:
        raw.set_channel_types({channel: "eog" for channel in configured_eog}, verbose=False)
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    if requested is None:
        requested = eeg
    if not set(requested).issubset(eeg) or len(set(requested)) != len(requested):
        raise ValueError(f"Requested channels are missing, duplicated, or not EEG: {requested}")
    auxiliaries = [name for name, kind in zip(raw.ch_names, raw.get_channel_types())
                   if kind in {"eog", "ecg"} and name not in requested]
    raw.pick(list(requested) + auxiliaries).load_data()
    # Positions are needed only when interpolating explicitly marked bad channels.
    if raw.info["bads"]:
        raw.set_montage("standard_1005", match_case=False, on_missing="raise", verbose=False)
        raw.interpolate_bads(reset_bads=True, verbose=False)
    return list(requested)


def clean_with_ica(raw, config):
    settings = config["preprocessing"]["ica"]
    if not settings["enabled"]:
        return {"enabled": False, "excluded": []}
    eeg = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg) < 3:
        raise ValueError("ICA requires at least three EEG channels.")
    fit_raw = raw.copy()
    if settings["fit_max_seconds"] is not None:
        fit_raw.crop(tmax=min(float(settings["fit_max_seconds"]), fit_raw.times[-1]))
    fit_raw.filter(settings["fit_highpass"], None, picks=eeg, n_jobs=1,
                   skip_by_annotation=("edge", "bad"), verbose=False)
    if raw.info["sfreq"] / settings["decim"] / 2 <= config["preprocessing"]["h_freq"]:
        raise ValueError("ICA decimation would alias the filtered signal; reduce ica.decim.")
    # MNE 1.13 introduced rng; retain compatibility with earlier Colab installs.
    rng_key = "rng" if "rng" in inspect.signature(mne.preprocessing.ICA).parameters else "random_state"
    ica = mne.preprocessing.ICA(
        n_components=min(int(settings["n_components"]), len(eeg) - 1),
        method="fastica", max_iter=settings["max_iter"], **{rng_key: config["seed"]},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        ica.fit(fit_raw, picks=eeg, decim=settings["decim"], verbose=False)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    for message in convergence:
        LOGGER.warning("ICA convergence: %s", message)
    for item in caught:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn(str(item.message), item.category, stacklevel=2)
    real_eog = [raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True, exclude="bads")]
    configured = settings["eog_channels"]
    if configured and not set(configured).issubset(raw.ch_names):
        raise ValueError("Configured ICA EOG channels are absent.")
    sources = configured or real_eog
    using_proxy = not sources
    if using_proxy:
        sources = [ch for ch in settings["eog_proxy_channels"] if ch in raw.ch_names]
        LOGGER.warning("No dedicated EOG: using frontal EEG proxies %s; inspect ERP and ICA QC.", sources)
    if not sources:
        raise ValueError("ICA has no EOG reference. Configure eog_proxy_channels or disable ICA explicitly.")
    excluded, scores = set(), {}
    for channel in sources:
        indices, values = ica.find_bads_eog(
            fit_raw, ch_name=channel, threshold=settings["eog_threshold"], verbose=False,
        )
        excluded.update(indices)
        scores[channel] = np.asarray(values).tolist()
    ica.exclude = sorted(int(index) for index in excluded)
    if not ica.exclude:
        LOGGER.warning("ICA found no EOG components; inspect the saved correlation scores.")
    if len(ica.exclude) >= ica.n_components_:
        raise ValueError("ICA marked every component; inspect data and EOG configuration.")
    ica.apply(raw, verbose=False)
    report = {"enabled": True, "n_components": int(ica.n_components_),
              "excluded": ica.exclude, "eog_sources": sources, "using_eeg_proxy": using_proxy,
              "eog_scores": scores, "convergence_warnings": convergence,
              "fit_seconds": float(fit_raw.times[-1])}
    del fit_raw, ica
    return report


def process_recording(header, config):
    p = config["preprocessing"]
    events_path = header.with_name(header.name.replace("_eeg.vhdr", "_events.tsv"))
    if not events_path.exists():
        raise FileNotFoundError(f"Missing BIDS event table: {events_path}")
    raw = mne.io.read_raw_brainvision(header, preload=False, verbose=False)
    try:
        channels = prepare_channels(raw, header, config)
        events, counts = read_events(events_path, raw, config)
        # Reject epochs that cross concatenation boundaries, including New Segment.
        descriptions = []
        for onset, description in zip(raw.annotations.onset, raw.annotations.description):
            if any(word in description.casefold() for word in ("boundary", "new segment")):
                # A boundary at time zero must not create an empty FIR segment.
                interior = raw.first_time < onset < raw.first_time + raw.times[-1]
                description = "BAD boundary" if interior else "BAD recording edge"
            descriptions.append(description)
        raw.set_annotations(mne.Annotations(raw.annotations.onset, raw.annotations.duration,
                                           descriptions, orig_time=raw.annotations.orig_time))
        skip = ("edge", "bad_acq_skip", "bad boundary")
        if p["notch_freq"] >= raw.info["sfreq"] / 2:
            raise ValueError("Notch frequency must be below the recording Nyquist frequency.")
        raw.notch_filter(p["notch_freq"], picks=["eeg", "eog"], n_jobs=p["n_jobs"],
                         skip_by_annotation=skip, verbose=False)
        raw.filter(p["l_freq"], p["h_freq"], picks=["eeg", "eog"], n_jobs=p["n_jobs"],
                   skip_by_annotation=skip, verbose=False)
        # Jointly resample events to avoid stimulus timing drift.
        raw, events = raw.resample(p["sfreq"], events=events, n_jobs=p["n_jobs"], verbose=False)
        if len(np.unique(events[:, 0])) != len(events):
            raise ValueError("Retained events collide after resampling.")
        ica_report = clean_with_ica(raw, config)
        if p["reference"] == "average":
            raw.set_eeg_reference("average", projection=False, verbose=False)
        elif p["reference"] is not None:
            raise ValueError("reference must be average or null (retain acquisition reference).")
        reject = None if p["reject_uv"] is None else {"eeg": p["reject_uv"] * 1e-6}
        flat = None if p["flat_uv"] is None else {"eeg": p["flat_uv"] * 1e-6}
        epochs = mne.Epochs(
            raw, events, event_id={"standard": 1, "target": 2}, tmin=p["tmin"], tmax=p["tmax"],
            baseline=tuple(p["baseline"]), picks=channels, preload=True, reject=reject, flat=flat,
            reject_by_annotation=True, event_repeated="error", verbose=False,
        )
        x = epochs.get_data(copy=False).astype(np.float32)
        y = (epochs.events[:, 2] == 2).astype(np.int8)
        if set(np.unique(y)) != {0, 1} or not np.isfinite(x).all():
            raise ValueError(f"{header.name}: no usable binary epochs or nonfinite signals.")
        report = {"recording": header.name, "event_counts": counts,
                  "retained": {"standard": int((y == 0).sum()), "target": int((y == 1).sum())},
                  "dropped_epochs": int(len(events) - len(y)),
                  "drop_reasons": {reason: sum(reason in reasons for reasons in epochs.drop_log)
                                   for reason in sorted({r for reasons in epochs.drop_log for r in reasons})},
                  "ica": ica_report}
        return x, y, epochs.ch_names, epochs.times.copy(), report
    finally:
        raw.close()


@contextmanager
def downloaded_subject(subject, files, revision, config):
    """All raw bytes and Hub metadata live under one disposable subject directory."""
    raw_root = Path(config["paths"]["raw"]).expanduser()
    artifacts = artifact_dir(config).resolve()
    if raw_root.is_symlink():
        raise ValueError("The disposable raw directory must not be a symlink.")
    raw_root = raw_root.resolve()
    if raw_root == artifacts or raw_root in artifacts.parents or artifacts in raw_root.parents:
        raise ValueError("Raw and persistent artifact directories must be separate.")
    raw_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{subject}-", dir=raw_root))
    try:
        local = work / "files"
        snapshot_download(
            repo_id=config["dataset"]["repo_id"], repo_type="dataset", revision=revision,
            allow_patterns=files, local_dir=local, cache_dir=work / "hub_cache", max_workers=2,
        )
        yield local
    finally:
        # Also remove partial downloads after an exception or Ctrl-C.
        shutil.rmtree(work)
        gc.collect()
        LOGGER.info("Deleted raw download for %s", subject)


def process_subject(subject, local, destination, signature, config):
    headers = sorted(local.glob(f"**/{subject}/**/*_task-oddball*_eeg.vhdr"))
    if not headers:
        raise FileNotFoundError(f"No oddball BrainVision header for {subject}")
    xs, ys, reports, channels, times = [], [], [], None, None
    for header in headers:
        x, y, run_channels, run_times, report = process_recording(header, config)
        if channels is not None and (run_channels != channels or not np.array_equal(times, run_times)):
            raise ValueError(f"Inconsistent run schema for {subject}")
        channels, times = run_channels, run_times
        xs.append(x)
        ys.append(y)
        reports.append(report)
    destination.mkdir(parents=True, exist_ok=True)
    # metadata.json is a completion marker; it is written last.
    (destination / "metadata.json").unlink(missing_ok=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.save(destination / "X.npy", x, allow_pickle=False)
    np.save(destination / "y.npy", y, allow_pickle=False)
    np.save(destination / "ch_names.npy", np.asarray(channels, dtype=str), allow_pickle=False)
    np.save(destination / "times.npy", times, allow_pickle=False)
    write_json(destination / "metadata.json", {"subject": subject, "signature": signature,
               "shape": list(x.shape), "recordings": reports})
    LOGGER.info("Checkpointed %s: %s", subject, x.shape)


def consolidate(subjects, output, signature, revision, config, dry_run=False):
    """Copy disk-backed subject shards without loading the entire cohort into RAM."""
    shards = output / "subjects"
    metadata = [read_json(shards / subject / "metadata.json") for subject in subjects]
    if any(m["signature"] != signature for m in metadata):
        raise ValueError("Subject checkpoints were produced by a different preprocessing configuration.")
    channels = np.load(shards / subjects[0] / "ch_names.npy", allow_pickle=False)
    times = np.load(shards / subjects[0] / "times.npy", allow_pickle=False)
    total = sum(m["shape"][0] for m in metadata)
    shape = (total, len(channels), len(times))
    # Invalidate the previous completion marker before replacing any array.
    (output / "metadata.json").unlink(missing_ok=True)
    names = ("X", "y", "groups")
    temporary = {name: output / f".{name}.tmp.npy" for name in names}
    arrays = {}
    offset, class_counts = 0, np.zeros(2, dtype=int)
    try:
        # Allocation itself can fail (e.g. Drive full). Keep it inside cleanup.
        for name, dtype, array_shape in (("X", "float32", shape), ("y", "int8", (total,)),
                                         ("groups", f"U{max(map(len, subjects))}", (total,))):
            arrays[name] = np.lib.format.open_memmap(temporary[name], mode="w+", dtype=dtype, shape=array_shape)
        for subject, meta in zip(subjects, metadata):
            directory = shards / subject
            if not np.array_equal(channels, np.load(directory / "ch_names.npy", allow_pickle=False)) or not np.array_equal(times, np.load(directory / "times.npy", allow_pickle=False)):
                raise ValueError(f"Channel/time order differs for {subject}; use a common channel list.")
            x = np.load(directory / "X.npy", mmap_mode="r", allow_pickle=False)
            y = np.load(directory / "y.npy", allow_pickle=False)
            if list(x.shape) != meta["shape"] or x.shape[1:] != shape[1:] or y.shape != (len(x),):
                raise ValueError(f"Corrupt checkpoint for {subject}")
            stop = offset + len(x)
            arrays["X"][offset:stop], arrays["y"][offset:stop] = x, y
            arrays["groups"][offset:stop] = subject
            class_counts += np.bincount(y, minlength=2)
            offset = stop
        for array in arrays.values():
            array.flush()
        arrays.clear()
        for name in names:
            os.replace(temporary[name], output / f"{name}.npy")
        np.save(output / "ch_names.npy", channels, allow_pickle=False)
        np.save(output / "times.npy", times, allow_pickle=False)
        write_json(output / "metadata.json", {
            "dataset_id": uuid.uuid4().hex, "preprocessing_signature": signature,
            "preprocessing_version": PREPROCESSING_VERSION,
            "repo_id": config["dataset"]["repo_id"], "revision": revision,
            "subjects": subjects, "dry_run": dry_run, "shape": list(shape),
            "units": "V", "sfreq": float(config["preprocessing"]["sfreq"]),
            "labels": {"standard": 0, "target": 1}, "class_counts": class_counts.tolist(),
            "config": config, "versions": package_versions(),
        })
    finally:
        arrays.clear()
        for path in temporary.values():
            path.unlink(missing_ok=True)
    LOGGER.info("Consolidated %d subjects, %d epochs into %s", len(subjects), total, output)


def run(config, dry_run=False, resume=False, overwrite=False):
    output = artifact_dir(config)
    api = HfApi()
    info = api.dataset_info(config["dataset"]["repo_id"], revision=config["dataset"]["revision"])
    revision = info.sha
    files = api.list_repo_files(config["dataset"]["repo_id"], repo_type="dataset", revision=revision)
    discovered = sorted({part for f in files for part in PurePosixPath(f).parts if re.fullmatch(r"sub-\d+", part)})
    if len(discovered) != config["dataset"]["expected_subjects"]:
        LOGGER.warning("Found %d subjects; configured expectation is %d", len(discovered), config["dataset"]["expected_subjects"])
    subjects = ["sub-01"] if dry_run else (config["dataset"]["subjects"] or discovered)
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects).issubset(discovered):
        raise ValueError(f"Subjects must be unique and present in the repo. Available: {discovered}")
    signature = fingerprint({"pipeline_version": PREPROCESSING_VERSION, "revision": revision, "repo": config["dataset"]["repo_id"],
                             "events": config["events"], "preprocessing": config["preprocessing"], "seed": config["seed"]})
    if (output / "metadata.json").exists():
        previous = read_json(output / "metadata.json")
        if dry_run and not previous["dry_run"]:
            raise ValueError("Dry-run cannot replace a cohort. Use a separate --artifacts directory.")
        if not (resume or overwrite):
            raise FileExistsError("Processed arrays exist. Use --resume or --overwrite.")
    for subject in subjects:
        destination = output / "subjects" / subject
        marker = destination / "metadata.json"
        if marker.exists() and not overwrite:
            cached = read_json(marker)
            if not resume:
                raise FileExistsError(f"{subject} checkpoint exists. Use --resume or --overwrite.")
            if cached["signature"] != signature:
                raise ValueError(f"{subject} checkpoint configuration/revision changed; use --overwrite.")
            for name in ("X", "y", "ch_names", "times"):
                if not (destination / f"{name}.npy").exists():
                    raise ValueError(f"Incomplete {subject} checkpoint; use --overwrite.")
            LOGGER.info("Reusing completed checkpoint for %s", subject)
            continue
        selected = [f for f in files if subject in PurePosixPath(f).parts and
                    "_task-oddball" in f and Path(f).suffix in {".eeg", ".vhdr", ".vmrk", ".tsv", ".json"}]
        if not any(f.endswith(".vhdr") for f in selected):
            raise FileNotFoundError(f"No raw files for {subject}")
        with downloaded_subject(subject, selected, revision, config) as local:
            process_subject(subject, local, destination, signature, config)
    consolidate(subjects, output, signature, revision, config, dry_run)
    if dry_run:
        try:
            from .visualize import plot_erp
        except ImportError:
            from visualize import plot_erp
        plot_erp(output, output / "erp_sub01_check.png", channels=["Fz", "Pz"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Process the complete sub-01 recording and plot its Fz/Pz ERPs.")
    parser.add_argument("--artifacts", type=Path, help="Override the persistent artifact directory.")
    parser.add_argument("--raw-dir", type=Path, help="Override /content/temp_raw for local execution.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse compatible completed subject checkpoints.")
    mode.add_argument("--overwrite", action="store_true", help="Reprocess the requested subjects.")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.artifacts:
        config["paths"]["artifacts"] = str(args.artifacts)
    if args.raw_dir:
        config["paths"]["raw"] = str(args.raw_dir)
    with threadpool_limits(limits=1):
        run(config, args.dry_run, args.resume, args.overwrite)


if __name__ == "__main__":
    main()
