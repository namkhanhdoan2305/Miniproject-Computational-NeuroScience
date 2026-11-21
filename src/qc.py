"""Capture actual raw/intermediate EEG while available, then export bounded QC."""
from __future__ import annotations
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from .common import package_versions, write_json
from .visualize import COLORS, save_figure


class RecordingQC:
    """Keep short traces, PSDs and midline epochs, never full raw copies."""
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.stages = {}
        self.ica_figures = []

    def capture(self, stage, raw, events):
        p, settings = self.config["preprocessing"], self.config["qc"]
        names = [c for c in ["Fp1", "Fp2", "Fz", "Cz", "Pz"] if c in raw.ch_names]
        if not names:
            names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)[:3]]
        n = min(raw.n_times, int(settings["trace_seconds"] * raw.info["sfreq"]))
        psd_n = min(raw.n_times, int(settings["psd_seconds"] * raw.info["sfreq"]))
        signal = raw.get_data(picks="eeg", start=0, stop=psd_n) * 1e6
        if stage == "raw":
            deviation = signal.std(axis=-1)
            median = max(float(np.median(deviation)), 1e-12)
            eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
            # Review flags only: not validated automatic bad-channel rejection.
            quality = pd.DataFrame({"channel": eeg_names, "std_uv": deviation,
                                    "std_over_median": deviation / median,
                                    "review_flag": (deviation < .1) | (deviation / median > 5)})
            quality.to_csv(self.directory / "channel_quality.csv", index=False)
            fig, ax = plt.subplots(figsize=(max(8, len(eeg_names) * .25), 4))
            ax.bar(eeg_names, deviation, color=np.where(quality.review_flag, "#D55E00", "#0072B2"))
            ax.tick_params(axis="x", labelrotation=90)
            ax.set(ylabel="Standard deviation (µV)", title="Raw selected EEG | orange: heuristic review flag, not automatic exclusion")
            fig.tight_layout()
            save_figure(fig, self.directory / "channel_quality.png")
        freq, power = welch(signal, fs=raw.info["sfreq"], nperseg=min(psd_n, int(4 * raw.info["sfreq"])), axis=-1)
        # PSD uses all selected EEG; a bounded segment is explicitly recorded.
        epochs = mne.Epochs(raw, events, {"standard": 1, "target": 2}, tmin=p["tmin"], tmax=p["tmax"],
                            baseline=tuple(p["baseline"]), picks=names, preload=True,
                            reject=None, flat=None, reject_by_annotation=True, verbose=False)
        self.stages[stage] = {"names": names, "trace": raw.get_data(picks=names, start=0, stop=n) * 1e6,
                              "trace_times": raw.times[:n].copy(), "frequency": freq,
                              "psd": power.mean(axis=0), "epoch_indices": epochs.selection.copy(),
                              "epochs": epochs.get_data(copy=False).astype(np.float32) * 1e6,
                              "times": epochs.times.copy(), "sfreq": float(raw.info["sfreq"]),
                              "psd_seconds": float(psd_n / raw.info["sfreq"])}

    def capture_ica(self, ica, fit_raw, report):
        scores = report["eog_scores"]
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for channel, values in scores.items():
            ax.plot(np.arange(len(values)), values, "o-", label=channel)
        for index in ica.exclude:
            ax.axvspan(index - .3, index + .3, color="#D55E00", alpha=.2)
        ax.set(xlabel="ICA component index (zero-based)", ylabel="EOG/proxy correlation",
               title=f"Excluded: {ica.exclude} | MNE iterative z-score threshold, not correlation cutoff")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, self.directory / "ica_eog_scores.png")
        self.ica_figures.append("ica_eog_scores.png")
        # All components: review the mixing pattern as well as EOG correlation.
        picks = mne.pick_types(fit_raw.info, eeg=True, exclude="bads")
        positions = np.asarray([fit_raw.info['chs'][i]['loc'][:3] for i in picks])
        valid = (len(picks) >= 4 and np.isfinite(positions).all()
                 and np.linalg.matrix_rank(positions[:, :2] - positions[:, :2].mean(axis=0)) == 2)
        if valid:
            figures = ica.plot_components(show=False)
            figures = figures if isinstance(figures, list) else [figures]
            for i, figure in enumerate(figures):
                filename = f"ica_components_{i + 1}.png"
                save_figure(figure, self.directory / filename)
                self.ica_figures.append(filename)
        source = ica.get_sources(fit_raw).get_data(start=0, stop=min(fit_raw.n_times, int(10 * fit_raw.info["sfreq"])))
        chosen = ica.exclude or list(range(min(3, len(source))))
        fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2 * len(chosen)), squeeze=False)
        for ax, index in zip(axes.flat, chosen):
            ax.plot(np.arange(source.shape[1]) / fit_raw.info["sfreq"], source[index], linewidth=.6)
            ax.set(ylabel=f"IC {index} (a.u.)")
        axes[-1, 0].set_xlabel("Time in ICA fitting segment (s)")
        fig.suptitle("ICA source inspection | excluded components, or first 3 when none excluded")
        fig.tight_layout()
        save_figure(fig, self.directory / "ica_sources.png")
        self.ica_figures.append("ica_sources.png")

    def finish(self, epochs, report):
        stages = list(self.stages.items())
        names = stages[0][1]["names"]
        fig, axes = plt.subplots(len(names), 1, figsize=(12, 2 * len(names)), squeeze=False)
        for stage, data in stages:
            for ax, channel in zip(axes.flat, names):
                i = data["names"].index(channel)
                ax.plot(data["trace_times"], data["trace"][i], linewidth=.65, alpha=.8, label=stage)
                ax.set(ylabel=f"{channel} (µV)")
        axes[0, 0].legend(ncol=3)
        axes[-1, 0].set_xlabel("Time from recording start (s); identical interval, acquisition reference")
        fig.tight_layout()
        save_figure(fig, self.directory / "raw_filtered_ica_traces.png")
        fig, ax = plt.subplots(figsize=(9, 4))
        psd_rows = []
        for stage, data in stages:
            keep = (data["frequency"] > 0) & (data["frequency"] <= 100)
            ax.plot(data["frequency"][keep], 10 * np.log10(np.maximum(data["psd"][keep], 1e-30)), label=stage)
            psd_rows.append(pd.DataFrame({"stage": stage, "frequency_hz": data["frequency"], "psd_uv2_hz": data["psd"]}))
        ax.axvline(50, color="black", linestyle="--", linewidth=.8)
        ax.set(xlabel="Frequency (Hz)", ylabel="Mean channel PSD (dB µV²/Hz)", title="Same recording segment | compare line noise and passband")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, self.directory / "psd_before_after.png")
        pd.concat(psd_rows, ignore_index=True).to_csv(self.directory / "psd_before_after.csv", index=False)
        channels = [c for c in ["Fz", "Cz", "Pz"] if c in epochs.ch_names]
        p300_start, p300_end = self.config["features"]["p300_window"]
        fig, axes = plt.subplots(1, max(1, len(channels)), figsize=(4.5 * max(1, len(channels)), 4), squeeze=False, sharey=True)
        erp_rows = []
        common_events = epochs.selection.copy()
        for _, data in stages:
            common_events = np.intersect1d(common_events, data["epoch_indices"])
        for ax, channel in zip(axes.flat, channels):
            for stage, data in stages:
                # Match the exact same FINAL accepted stimulus IDs at every stage.
                _, original, _ = np.intersect1d(data["epoch_indices"], common_events, return_indices=True)
                _, final, _ = np.intersect1d(epochs.selection, common_events, return_indices=True)
                for label, color in [(1, COLORS[0]), (2, COLORS[1])]:
                    mask = epochs.events[final, 2] == label
                    if not mask.any():
                        continue
                    curve = data["epochs"][original[mask], data["names"].index(channel)].mean(axis=0)
                    ax.plot(data["times"] * 1000, curve, color=color,
                            alpha=.45 if stage == "raw" else .85,
                            linestyle={"raw": ":", "filtered": "--", "post_ica": "-"}[stage],
                            label=f"{stage}: {'standard' if label == 1 else 'target'}")
                    erp_rows.append(pd.DataFrame({"stage": stage, "channel": channel, "class": label - 1,
                                                 "time_s": data["times"], "amplitude_uv": curve, "n_trials": int(mask.sum())}))
            ax.axvspan(p300_start * 1000, p300_end * 1000, color="gray", alpha=.15)
            ax.axvline(0, color="black", linewidth=.6)
            ax.set(title=channel, xlabel="Time after stimulus (ms)", xlim=(epochs.times[0] * 1000, epochs.times[-1] * 1000))
        if channels:
            axes[0, 0].set_ylabel("Amplitude (µV)")
            axes[0, -1].legend(fontsize=7)
        fig.suptitle("Same final retained trials & baseline; acquisition reference (isolates ICA from rereferencing)", fontsize=10)
        fig.tight_layout()
        save_figure(fig, self.directory / "erp_stages_matched_trials.png")
        if erp_rows:
            pd.concat(erp_rows, ignore_index=True).to_csv(self.directory / "erp_stages_matched_trials.csv", index=False)
        x = epochs.get_data(copy=False) * 1e6
        midline = channels[-1] if channels else epochs.ch_names[0]
        i = epochs.ch_names.index(midline)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        maximum = max(1, float(np.percentile(np.abs(x[:, i]), 98)))
        for ax, code, name in zip(axes, [1, 2], ["Standard", "Target"]):
            trials = x[epochs.events[:, 2] == code, i]
            image = ax.imshow(trials, aspect="auto", origin="lower", extent=[epochs.times[0] * 1000, epochs.times[-1] * 1000, 0, len(trials)],
                              cmap="RdBu_r", vmin=-maximum, vmax=maximum)
            ax.set(title=f"{name}: n={len(trials)}", xlabel="Time (ms)", ylabel="Retained trial index")
        fig.colorbar(image, ax=list(axes), label="µV (after final referencing/epoching)", shrink=.8)
        fig.suptitle(f"{midline} final single-trial EEG | no averaging between trials")
        save_figure(fig, self.directory / "single_trial_heatmap.png")
        # Sensor topography is descriptive scalp activity, not cortical source localization.
        positions = np.asarray([ch["loc"][:3] for ch in epochs.info["chs"]])
        good = np.isfinite(positions).all(axis=1) & (np.linalg.norm(positions, axis=1) > 0)
        topomap_status = "skipped: need at least 4 valid standard sensor locations"
        if good.sum() >= 4 and np.linalg.matrix_rank(positions[good, :2] - positions[good, :2].mean(axis=0)) == 2:
            window = (epochs.times >= p300_start) & (epochs.times <= p300_end)
            maps = [x[epochs.events[:, 2] == label][:, good][:, :, window].mean(axis=(0, 2)) for label in [1, 2]]
            maps.append(maps[1] - maps[0])
            limit = max(1e-6, max(np.max(np.abs(values)) for values in maps))
            info = mne.pick_info(epochs.info, np.flatnonzero(good))
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
            for ax, values, title in zip(axes, maps, ["Standard", "Target", "Target − standard"]):
                im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, vlim=(-limit, limit), cmap="RdBu_r", contours=4)
                ax.set_title(title)
            fig.colorbar(im, ax=list(axes), label=f"Mean {p300_start * 1000:g}–{p300_end * 1000:g} ms (µV)", shrink=.7)
            fig.suptitle("P300 scalp topography | standard montage; one recording, descriptive")
            save_figure(fig, self.directory / "p300_topography.png")
            topomap_status = "standard 10-05 approximate positions; configured final reference"
        labels = ["standard", "target", "distractor"]
        fig, ax = plt.subplots(figsize=(7, 4))
        before = [report["event_counts"][label] for label in labels]
        after = [report["retained"].get(label, 0) for label in labels]
        pos = np.arange(3)
        ax.bar(pos - .18, before, width=.36, label="BIDS events")
        ax.bar(pos + .18, after, width=.36, label="Retained epochs")
        ax.set(xticks=pos, xticklabels=labels, ylabel="Count", title="Class filtering + epoch rejection (distractor always excluded)")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, self.directory / "event_retention.png")
        final_selection = epochs.selection.tolist()
        manifest = {"recording": report["recording"], "report": report,
                    "stage_sfreq": {stage: data["sfreq"] for stage, data in stages},
                    "psd_seconds": {stage: data["psd_seconds"] for stage, data in stages},
                    "final_retained_event_indices": final_selection,
                    "matched_erp_event_indices": common_events.tolist(),
                    "stage_event_indices": {stage: data["epoch_indices"].tolist() for stage, data in stages},
                    "topomap": topomap_status,
                    "configuration": {"preprocessing": self.config["preprocessing"], "events": self.config["events"], "seed": self.config["seed"]},
                    "versions": package_versions(),
                    "notes": ["raw stage is after channel selection/BIDS bad-channel interpolation, before temporal filtering",
                              "Stage ERP and traces use acquisition reference; final heatmap/topography use configured final reference",
                              "Channel quality flags are heuristic inspection aids; unmarked bad channels are not automatically interpolated",
                              "QC is descriptive and does not change rejection thresholds or model settings"]}
        write_json(self.directory / "metadata.json", manifest)
        images = [p.name for p in self.directory.glob("*.png")]
        (self.directory / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>EEG preprocessing QC</title>'
            + f'<h1>{escape(report["recording"])}</h1><p>Raw → notch/bandpass/resample → ICA → reference/epoch/baseline/rejection.</p>'
            + '<p>These are recording-level QC figures. Sensor maps use approximate standard positions.</p>'
            + ''.join(f'<h2>{escape(Path(p).stem)}</h2><img src="{escape(p)}" style="max-width:100%">' for p in sorted(images)), encoding="utf-8")
