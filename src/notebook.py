"""Report tables and figure captions for the executable Vietnamese notebook."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .common import load_epochs, read_json

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision", "recall": "Recall", "f1": "F1 target",
    "roc_auc": "ROC-AUC", "average_precision": "AP",
}
MODEL_LABELS = {"LDA": "Shrinkage LDA", "SVM": "SVM RBF", "RandomForest": "Random Forest",
                "LogisticRegression": "Logistic Regression L2"}
MODEL_SECTIONS = {"lda": "5.1", "svm": "5.2", "random_forest": "5.3", "logistic_regression": "5.4"}


def run_step(module, *arguments):
    """Run repo modules; command failure propagates to the notebook cell."""
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _text(text):
    from IPython.display import Markdown, display
    display(Markdown(text))


def _table(caption, frame):
    from IPython.display import HTML, display
    _text(caption)
    display(HTML(frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")))


def show_images(directory, filenames, captions=None):
    from IPython.display import Image, display
    for name in filenames:
        path = Path(directory) / name
        if path.exists():
            display(Image(filename=str(path)))
            if captions and name in captions:
                _text(captions[name])
        else:
            print("Chưa có tệp hình:", path)


def benchmark_table(summary):
    """Use saved outer-fold means/SDs, never replace them with pooled OOF metrics."""
    frame = pd.DataFrame({"Mô hình": summary.model.map(MODEL_LABELS).fillna(summary.model)})
    for key, label in METRIC_LABELS.items():
        frame[label] = [f"{mean:.3f} ± {std:.3f}" for mean, std in zip(summary[key + "_mean"], summary[key + "_std"])]
    return frame


def show_environment(artifacts="artifacts"):
    environment = read_json(Path(artifacts) / "environment.json")
    rows = [("Git commit", environment["git_commit"]), ("Python", environment["python"].split()[0])]
    rows.extend((name, value) for name, value in environment["versions"].items() if value != "unavailable")
    rows.append(("CUDA/cuML khả dụng", "Có" if environment["gpu"][0] else "Không"))
    _table("**Bảng 2.1. Môi trường tính toán của lần thực nghiệm.**", pd.DataFrame(rows, columns=["Thành phần", "Phiên bản / trạng thái"]))


def show_preprocessing_config(config):
    p, ica = config["preprocessing"], config["preprocessing"]["ica"]
    rows = [("Lọc thông dải", f"{p['l_freq']:g}–{p['h_freq']:g} Hz"),
            ("Lọc notch", f"{p['notch_freq']:g} Hz"), ("Tần số lấy mẫu đầu ra", f"{p['sfreq']:g} Hz"),
            ("Cửa sổ epoch", f"{p['tmin'] * 1000:g} đến {p['tmax'] * 1000:g} ms"),
            ("Baseline", f"{p['baseline'][0] * 1000:g} đến {p['baseline'][1] * 1000:g} ms"),
            ("Tham chiếu", str(p["reference"])), ("Ngưỡng peak-to-peak", f"{p['reject_uv']} µV"),
            ("Ngưỡng gần phẳng", f"{p['flat_uv']} µV"), ("ICA", "Bật" if ica["enabled"] else "Tắt"),
            ("Phương pháp ICA", "FastICA"),
            ("Số thành phần ICA tối đa", ica["n_components"]),
            ("High-pass riêng cho ICA fit", f"{ica['fit_highpass']:g} Hz"),
            ("Thời lượng ICA fit tối đa", "Toàn bộ bản ghi" if ica["fit_max_seconds"] is None else f"{ica['fit_max_seconds']} s"),
            ("Hệ số decimation ICA", ica["decim"]), ("Ngưỡng z-score phát hiện EOG component", ica["eog_threshold"])]
    _table("**Bảng 3.1. Cấu hình tiền xử lý.** Giá trị lấy từ cấu hình đang chạy.", pd.DataFrame(rows, columns=["Tham số", "Giá trị"]))


def show_dataset(artifacts="artifacts"):
    directory = Path(artifacts)
    x, y, groups, channels, times, meta = load_epochs(directory)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    rows = [("Số người", len(np.unique(groups))), ("Số epoch", len(y)), ("Standard", n0), ("Target", n1),
            ("Tỷ lệ target", f"{n1 / len(y):.2%}"), ("Số kênh", len(channels)),
            ("Số mẫu / epoch", x.shape[-1]), ("Tần số lấy mẫu", f"{meta['sfreq']:g} Hz"),
            ("Khoảng thời gian", f"{times[0] * 1000:g} đến {times[-1] * 1000:g} ms"),
            ("Revision dữ liệu", meta["revision"])]
    _table("**Bảng 3.3. Dữ liệu sau tiền xử lý.**", pd.DataFrame(rows, columns=["Đại lượng", "Giá trị"]))
    show_images(directory, ["dataset_overview.png", "processed_trial_examples.png", "fig1_grand_average_erp.png"], {
        "dataset_overview.png": "**Hình 3.2.** Số epoch standard/target và tỷ lệ target của từng người sau loại nhiễu.",
        "processed_trial_examples.png": "**Hình 3.3.** Các epoch riêng lẻ tại Fz, Cz và Pz; mỗi đường là một trial, đơn vị µV.",
        "fig1_grand_average_erp.png": "**Hình 3.4.** ERP standard và target tại Fz/Cz/Pz. Mỗi người có trọng số bằng nhau; vùng tô là CI Student-t 95% tại từng thời điểm. Không ước lượng CI quần thể nếu chỉ có một người.",
    })


def show_features(artifacts="artifacts"):
    directory = Path(artifacts)
    info = read_json(directory / "feature_metadata.json")
    names = np.load(directory / "feature_names.npy", allow_pickle=False)
    families = {"Công suất dải tần": 0, "Entropy": 0, "Thống kê wavelet": 0, "P300 theo kênh": 0, "P300 ROI đường giữa": 0}
    for name in names:
        channel, suffix = str(name).split("__", 1)
        if suffix.startswith("log10_power_"):
            family = "Công suất dải tần"
        elif suffix in {"spectral_entropy", "shannon_entropy"}:
            family = "Entropy"
        elif suffix.startswith("wavelet_"):
            family = "Thống kê wavelet"
        elif suffix.startswith("p300_"):
            family = "P300 ROI đường giữa" if channel.startswith("Midline_") else "P300 theo kênh"
        else:
            family = "Khác"
        families[family] = families.get(family, 0) + 1
    _table("**Bảng 4.1. Số chiều đặc trưng thực tế.**", pd.DataFrame(families.items(), columns=["Nhóm đặc trưng", "Số chiều"]))
    _text(f"Ma trận đặc trưng có kích thước **{info['shape'][0]:,} × {info['shape'][1]:,}** (trial × đặc trưng).")
    show_images(directory, ["feature_distributions.png"], {
        "feature_distributions.png": "**Hình 4.1.** Phân bố đặc trưng theo lớp tại Fz/Cz/Pz. Mỗi điểm biểu diễn giá trị trung bình của một người trong một lớp.",
    })


def show_protocol(config, artifacts="artifacts"):
    from .evaluation import group_splits
    from .features import load_features
    x, y, groups, _, _ = load_features(artifacts)
    rows = []
    for fold, (train, test) in enumerate(group_splits(x, y, groups, config["training"]["cv_folds"]), 1):
        rows.append({"Fold": fold, "Người train": len(np.unique(groups[train])), "Người test": len(np.unique(groups[test])),
                     "Trial train": len(train), "Trial test": len(test), "Target trong test (%)": f"{y[test].mean() * 100:.2f}"})
    t = config["training"]
    models = [
        ("Shrinkage LDA", f"solver=lsqr; shrinkage=auto; priors={t['lda_priors']}"),
        ("SVM RBF", f"C={t['svm_c']}; gamma={t['svm_gamma']}; class_weight=balanced"),
        ("Random Forest", f"n_estimators={t['rf_estimators']}; min_samples_leaf={t['rf_min_samples_leaf']}; max_depth={t.get('rf_max_depth')}; class_weight=balanced"),
        ("Logistic Regression L2", f"C={t['logistic_c']}; max_iter={t['max_iter']}; class_weight=balanced"),
    ]
    _table("**Bảng 4.2. Tham số bốn bộ phân loại.**", pd.DataFrame(models, columns=["Mô hình", "Tham số"]))
    _table("**Bảng 4.3. Phân bố dữ liệu trong các fold kiểm tra theo người.**", pd.DataFrame(rows))
    _text(f"Seed trong cấu hình thực nghiệm: **{config['seed']}**.")


def show_model(key, artifacts="artifacts"):
    directory = Path(artifacts) / "ml" / key
    metadata = read_json(directory / "metadata.json")
    summary = pd.read_csv(directory / "metrics.csv")
    folds = pd.read_csv(directory / "metrics_per_fold.csv")
    section = MODEL_SECTIONS[key]
    _table(f"**Bảng {section}a. Kết quả kiểm tra của {MODEL_LABELS[metadata['model']]}.** Trung bình ± SD giữa các fold; thang 0–1.", benchmark_table(summary))
    selected = ["fold", "n_test", "precision", "recall", "f1", "balanced_accuracy", "roc_auc", "average_precision"]
    _table(f"**Bảng {section}b. Kết quả từng fold.**", folds[selected].rename(columns={
        "fold": "Fold", "n_test": "Trial test", **METRIC_LABELS}))
    row = summary.iloc[0]
    _text(f"Thời gian fit mỗi fold: **{row.fit_seconds_mean:.3f} ± {row.fit_seconds_std:.3f} s**. "
          f"Backend ghi nhận: `{row.backend}`.")
    matrix = pd.read_csv(directory / "confusion_matrix.csv", index_col=0).to_numpy()
    _text(f"Trên các dự đoán out-of-fold, **{int(matrix[1, 1]):,}/{int(matrix[1].sum()):,}** epoch target "
          f"được nhận diện; **{int(matrix[1, 0]):,}** target bị bỏ sót và **{int(matrix[0, 1]):,}** standard bị dự đoán nhầm thành target.")
    show_images(directory, ["fig2_roc_and_pr_curves.png", "confusion_matrix.png"], {
        "fig2_roc_and_pr_curves.png": f"**Hình {section}a.** ROC và precision–recall trên test folds. Đường cong được lấy trung bình theo fold; vùng tô biểu diễn ±1 SD giữa các fold.",
        "confusion_matrix.png": f"**Hình {section}b.** Ma trận nhầm lẫn từ dự đoán out-of-fold. Phần trăm được chuẩn hóa theo lớp thật (theo hàng).",
    })
    diagnostic = read_json(directory / "diagnostics.json")
    if diagnostic["status"] == "complete":
        train = len(diagnostic["train_subjects"])
        valid = len(diagnostic["validation_subjects"])
        _text(f"Đường học được ước lượng trên inner split gồm **{train} người train, {valid} người validation**, "
              "lấy từ train của outer fold thứ nhất. Các đường này không dùng người thuộc outer test.")
        show_images(directory, ["learning_curve.png"], {
            "learning_curve.png": f"**Hình {section}c.** Balanced accuracy và F1 khi tăng số người train; tập inner validation được giữ cố định.",
        })
        if (directory / "training_progress.png").exists():
            caption = ("Log loss theo số cây trên cùng inner split; đây là thước đo dự báo khi tăng kích thước rừng."
                       if key == "random_forest" else "Lượt chẩn đoán L-BFGS trên CPU: objective có trọng số và L2 (trái), log loss không trọng số (phải).")
            show_images(directory, ["training_progress.png"], {"training_progress.png": f"**Hình {section}d.** {caption}"})
        if (directory / "optimizer_diagnostics.json").exists():
            optimizer = read_json(directory / "optimizer_diagnostics.json")
            _text(f"Lượt chẩn đoán thực hiện **{optimizer['iterations']} vòng tối ưu**; trạng thái: "
                  + ("hội tụ." if optimizer["success"] else f"chưa xác nhận hội tụ (`{optimizer['message']}`)."))
    else:
        status = {"disabled": "không được bật trong cấu hình", "insufficient_inner_subjects": "không đủ người để tách inner train/validation"}
        _text("Đường học: " + status.get(diagnostic["status"], diagnostic["status"]) + ".")
    if (directory / "feature_importance.png").exists():
        show_images(directory, ["feature_importance.png"], {
            "feature_importance.png": f"**Hình {section}e.** Hai mươi đặc trưng có MDI importance cao nhất trong Random Forest, tổng hợp từ các fold train.",
        })


def show_qc(artifacts="artifacts", subject="sub-01"):
    directory = Path(artifacts) / "qc" / subject
    reports = sorted(directory.glob("*/metadata.json"))
    if not reports:
        raise FileNotFoundError(f"Thiếu QC của {subject}; chạy src.preprocess --qc-only --subjects {subject}.")
    source_path, cohort_path = directory / "source.json", Path(artifacts) / "metadata.json"
    if source_path.exists() and cohort_path.exists():
        if read_json(source_path)["preprocessing_signature"] != read_json(cohort_path)["preprocessing_signature"]:
            _text("**Phạm vi đối chiếu:** QC này dùng nguồn/cấu hình khác với cohort đã lưu; xem metadata trước khi so sánh trực tiếp.")
    captions = {
        "event_retention.png": "Số sự kiện BIDS và epoch giữ lại theo lớp. Distractor bị loại theo thiết kế bài toán.",
        "channel_quality.png": "Độ lệch chuẩn của từng kênh trước lọc. Màu cam biểu thị cờ kiểm tra heuristic, chưa phải kết luận kênh hỏng.",
        "raw_filtered_ica_traces.png": "Tín hiệu trên cùng đoạn ghi trước lọc, sau lọc và sau ICA; đơn vị µV.",
        "psd_before_after.png": "PSD trung bình giữa các kênh trên cùng đoạn ghi; so sánh dải thông và thành phần 50 Hz.",
        "ica_eog_scores.png": "Tương quan giữa ICA components và EOG hoặc tín hiệu thay thế vùng trán; vùng tô đánh dấu các thành phần bị loại.",
        "ica_sources.png": "Diễn biến theo thời gian của các thành phần ICA được kiểm tra; biên độ theo đơn vị tùy ý.",
        "erp_stages_matched_trials.png": "ERP trước/sau xử lý trên cùng tập trial được giữ lại. Các giai đoạn dùng acquisition reference, tách ảnh hưởng của ICA khỏi bước đổi tham chiếu.",
        "single_trial_heatmap.png": "Biên độ từng epoch sau tiền xử lý; mỗi hàng là một trial, không lấy trung bình giữa trial.",
        "p300_topography.png": "Điện thế trung bình trong cửa sổ P300 trên da đầu; vị trí điện cực lấy từ montage chuẩn.",
    }
    for index, report_path in enumerate(reports, 1):
        report = read_json(report_path)["report"]
        _text(f"**Bản ghi kiểm tra:** `{report['recording']}`.")
        counts, retained = report["event_counts"], report["retained"]
        rows = [{"Lớp": label, "Sự kiện BIDS": counts[label], "Epoch giữ lại": retained.get(label, 0)}
                for label in ["standard", "target", "distractor"]]
        suffix = "" if len(reports) == 1 else chr(96 + index)
        _table(f"**Bảng 3.2{suffix}. Số lượng sự kiện trước/sau tiền xử lý.**", pd.DataFrame(rows))
        ica = report["ica"]
        _text(f"Epoch standard/target bị loại: **{report['dropped_epochs']}**. "
              + (f"ICA loại các thành phần `{ica['excluded']}`; tín hiệu tham chiếu: `{ica.get('eog_sources', [])}`."
                 if ica["enabled"] else "ICA không được áp dụng trong cấu hình của bản ghi này."))
        figures = [name for name in captions if (report_path.parent / name).exists()]
        group = "3.1" if len(reports) == 1 else f"3.1.{index}"
        for i, filename in enumerate(figures):
            show_images(report_path.parent, [filename], {filename: f"**Hình {group}{chr(97 + i)}.** {captions[filename]}"})
        for i, figure in enumerate(sorted(report_path.parent.glob("ica_components_*.png")), len(figures)):
            show_images(report_path.parent, [figure.name], {figure.name: f"**Hình {group}{chr(97 + i)}.** Phân bố không gian của các thành phần ICA; chỉ số component bắt đầu từ 0."})


def show_comparison(artifacts="artifacts"):
    from .visualize import validated_evaluation
    directory = Path(artifacts)
    summary, _, _ = validated_evaluation(directory)
    ranked = summary.sort_values("f1_mean", ascending=False)
    _table("**Bảng 6.1. So sánh bốn mô hình trên cùng các test folds.** Trung bình ± SD; sắp theo F1 target.", benchmark_table(ranked))
    cost = pd.DataFrame({"Mô hình": ranked.model.map(MODEL_LABELS), "Backend": ranked.backend,
                         "Fit / fold (s)": [f"{m:.3f} ± {s:.3f}" for m, s in zip(ranked.fit_seconds_mean, ranked.fit_seconds_std)]})
    _table("**Bảng 6.2. Chi phí tính toán quan sát được.**", cost)
    show_images(directory, ["model_comparison.png", "fig2_roc_and_pr_curves.png", "fig3_confusion_matrices.png", "fig4_feature_importance.png"], {
        "model_comparison.png": "**Hình 6.1.** Chỉ số kiểm tra và thời gian fit; thanh sai số biểu diễn SD giữa các fold.",
        "fig2_roc_and_pr_curves.png": "**Hình 6.2.** ROC và precision–recall trung bình của bốn mô hình. Đường nền PR phụ thuộc tỷ lệ target trong test folds.",
        "fig3_confusion_matrices.png": "**Hình 6.3.** Ma trận nhầm lẫn out-of-fold; số đếm và phần trăm theo hàng cho standard/target.",
        "fig4_feature_importance.png": "**Hình 6.4.** MDI importance của Random Forest trên fold train. Đặc trưng tương quan có thể chia sẻ mức quan trọng; thứ hạng này không phải kiểm định sinh lý.",
    })
    _text("### 6.1. Nhận xét từ kết quả")
    top = ranked.iloc[0]
    tied = ranked[np.isclose(ranked.f1_mean, top.f1_mean, atol=1e-12, rtol=0)]
    names = ", ".join(tied.model.map(MODEL_LABELS))
    _text(f"Trên cohort gồm **{int(top.n_subjects)} người, {int(top.n_trials):,} trial**, F1 target trung bình cao nhất "
          f"thuộc về **{names}**, đạt **{top.f1_mean:.3f} ± {top.f1_std:.3f}** qua {int(top.n_folds)} fold. "
          + (f"Chênh lệch F1 trung bình với mô hình xếp sau là **{top.f1_mean - ranked.iloc[1].f1_mean:.3f}** trên thang 0–1."
             if len(tied) == 1 and len(ranked) > 1 else "Các mô hình cùng mức F1 này được ghi nhận đồng hạng."))
    for metric, label in [("recall", "Recall target"), ("roc_auc", "ROC-AUC"), ("average_precision", "AP")]:
        winner = ranked.loc[ranked[metric + "_mean"].idxmax()]
        _text(f"{label} trung bình cao nhất: **{MODEL_LABELS[winner.model]}**, "
              f"**{winner[metric + '_mean']:.3f} ± {winner[metric + '_std']:.3f}**.")
    report = read_json(directory / "ml_benchmark_results.json")
    _text(f"Accuracy của bộ dự đoán luôn chọn lớp đa số là **{report['majority_accuracy_baseline']:.3f}**. "
          "Vì vậy, accuracy được đọc cùng F1, recall và AP của target. Các chênh lệch trên là mô tả thực nghiệm; "
          "chưa có kiểm định để kết luận một thuật toán vượt trội trong quần thể.")
