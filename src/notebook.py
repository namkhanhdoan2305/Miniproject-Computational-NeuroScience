"""Thin notebook helpers: checked subprocesses and inline display of saved outputs."""
from pathlib import Path
import subprocess
import sys
from .common import read_json


def run_step(module, *arguments):
    """A failed command raises, stopping Run all before later ML cells."""
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def show_images(directory, filenames):
    from IPython.display import Image, display
    for name in filenames:
        path = Path(directory) / name
        if path.exists():
            display(Image(filename=str(path)))
        else:
            print("Not generated:", path)


def show_model(key, artifacts="artifacts"):
    from IPython.display import Markdown, display
    import pandas as pd
    directory = Path(artifacts) / "ml" / key
    metadata = read_json(directory / "metadata.json")
    display(Markdown(f"**{metadata['model']}** — {metadata['rationale']}\n\n"
                     f"Backend: {', '.join(metadata['actual_backends'])}. "
                     "Bảng dưới là outer CV; đường học dùng inner subjects riêng."))
    display(pd.read_csv(directory / "metrics.csv"))
    display(pd.read_csv(directory / "metrics_per_fold.csv"))
    show_images(directory, ["metrics.png", "fig2_roc_and_pr_curves.png", "confusion_matrix.png"])
    diagnostics = read_json(directory / "diagnostics.json")
    if diagnostics["status"] == "complete":
        show_images(directory, ["learning_curve.png"])
        if (directory / "training_progress.png").exists():
            show_images(directory, ["training_progress.png"])
    else:
        print("Diagnostics:", diagnostics["status"])
    if (directory / "feature_importance.png").exists():
        show_images(directory, ["feature_importance.png"])


def show_qc(artifacts="artifacts", subject="sub-01"):
    directory = Path(artifacts) / "qc" / subject
    reports = sorted(directory.glob("*/metadata.json"))
    if not reports:
        raise FileNotFoundError(f"No actual raw QC for {subject}. Run src.preprocess --qc-only --subjects {subject}.")
    source_path = directory / "source.json"
    cohort_path = Path(artifacts) / "metadata.json"
    if source_path.exists() and cohort_path.exists():
        if read_json(source_path)["preprocessing_signature"] != read_json(cohort_path)["preprocessing_signature"]:
            print("QC uses different preprocessing/source settings from the saved cohort; inspect metadata before interpreting.")
    for report in reports:
        print(report.parent)
        show_images(report.parent, ["event_retention.png", "channel_quality.png", "raw_filtered_ica_traces.png", "psd_before_after.png",
                                  "ica_eog_scores.png", "ica_sources.png", "erp_stages_matched_trials.png",
                                  "single_trial_heatmap.png", "p300_topography.png"])
        for figure in sorted(report.parent.glob("ica_components_*.png")):
            show_images(report.parent, [figure.name])
