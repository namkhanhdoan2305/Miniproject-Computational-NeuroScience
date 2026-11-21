# Phân loại EEG từng trial: Oddball Standard / Target

Codebase xử lý tín hiệu EEG, trích xuất đặc trưng thủ công và so sánh **Shrinkage LDA, SVM RBF, Random Forest, Logistic Regression L2**. Dữ liệu [duylplsn/oddball](https://huggingface.co/datasets/duylplsn/oddball) gồm 42 người, BrainVision + BIDS. Bài toán nhị phân: **standard/frequent → 0, target → 1; loại distractor**. Không có Deep Learning.

[Mở notebook thực nghiệm trên Colab](https://colab.research.google.com/github/namkhanhdoan2305/Miniproject-Computational-NeuroScience/blob/main/notebooks/colab_workflow.ipynb). Notebook trình bày bằng tiếng Việt theo cấu trúc kỹ thuật: mục tiêu và dữ liệu, phương pháp tiền xử lý, đặc trưng, thiết kế đánh giá, kết quả từng mô hình và thảo luận. Có công thức, tham chiếu tài liệu, bảng cấu hình thực tế và chú thích hình. Mỗi thuật toán chạy ở **một cell riêng**; bảng và nhận xét định lượng được tạo từ kết quả của lần chạy. Logic xử lý và huấn luyện nằm trong repo; hướng dẫn thao tác được giữ trong README này.

## Chạy trên Google Colab

Dùng notebook [`notebooks/colab_workflow.ipynb`](notebooks/colab_workflow.ipynb) và chạy tuần tự:

1. Mount Drive, clone/cập nhật repo sạch, cài thư viện, liên kết `artifacts/` đến `MyDrive/oddball_artifacts`, ghi SHA/phiên bản/backend vào `environment.json`.
2. Kiểm tra sub-01: tín hiệu trước/sau lọc, PSD, ICA, ERP trước/sau trên cùng trial, heatmap và bản đồ P300.
3. Xử lý đủ người; xem tỷ lệ lớp, số epoch bị loại, ERP toàn nhóm và từng trial sau xử lý.
4. Trích xuất hoặc dùng lại đặc trưng tương thích, xem phân bố.
5. Train LDA → xem bảng chỉ số, ROC/PR, confusion matrix, learning curve.
6. Train SVM → xem kết quả và learning curve.
7. Train Random Forest → xem kết quả, log loss theo số cây, feature importance.
8. Train Logistic Regression → xem kết quả và lượt chẩn đoán loss theo vòng tối ưu.
9. Đọc bốn kết quả hoàn chỉnh, xuất bảng và hình so sánh.

Notebook dùng `subprocess.run(..., check=True)`: lệnh lỗi tạo exception để bạn xử lý trước khi chạy tiếp. Không dùng một cell chứa nhiều lệnh `!python` khiến bước sau vẫn chạy khi bước trước thất bại. Kết quả chạy được lưu trên Drive; các hình chỉ được hiển thị từ file đã xuất, không chứa dữ liệu mẫu giả trong notebook.

### Các lệnh tương đương

Chạy tại thư mục gốc repo, Python 3.10+:

```bash
python -m pip install -r requirements.txt
```

Dry-run **thật**: tải toàn bộ bản ghi sub-01, xử lý và kiểm tra. Nếu chưa có cohort:

```bash
python -m src.preprocess --dry-run --resume
```

Xuất `artifacts/erp_sub01_check.png`, checkpoint và QC của sub-01. Sau đó xử lý đầy đủ, dùng lại checkpoint:

```bash
python -m src.preprocess --resume
python -m src.visualize --data-only
python -m src.features
```

**Đã có X.npy của 42 người trên Drive:** giữ lại. Không cần tiền xử lý lại chỉ vì tách module train. Notebook sẽ dùng lại cohort có cùng cấu hình. Để bổ sung hình tín hiệu thô mà lần trước chưa lưu, chỉ tải lại người cần kiểm tra:

```bash
python -m src.preprocess --qc-only --subjects sub-01
```

`--qc-only` xử lý để vẽ QC, xóa raw ngay sau đó và **không ghi đè cohort, checkpoint hay đặc trưng**. Có thể chỉ định `sub-01 sub-02 sub-03`. Chế độ này dùng cấu hình hiện tại; metadata QC ghi rõ cấu hình/phiên bản nguồn để đối chiếu với dữ liệu cũ. Muốn dry-run riêng khi cohort đã tồn tại:

Notebook tự truyền `--revision <SHA đã lưu trong metadata.json>` khi bổ sung QC cho cohort có sẵn. Khi chạy CLI thủ công, có thể dùng cờ này để đối chiếu đúng phiên bản dữ liệu nguồn.

```bash
python -m src.preprocess --dry-run --artifacts artifacts/check_sub01 --resume
```

Train riêng từng thuật toán — mỗi lệnh dưới đây đặt trong một cell:

```bash
python -m src.train_lda --device auto --resume
python -m src.train_svm --device auto --resume
python -m src.train_random_forest --device auto --resume
python -m src.train_logistic_regression --device auto --resume
```

Trong notebook, hiển thị ngay sau lệnh tương ứng bằng:

```python
from src.notebook import run_step, show_model
run_step('src.train_svm', '--device', 'auto', '--resume')
show_model('svm')
```

Sau khi đủ bốn kết quả, so sánh mà **không train lại**:

```bash
python -m src.compare_ml
```

Vẫn hỗ trợ lệnh tổng hợp cũ:

```bash
python -m src.train_ml --device auto --resume
```

Hoặc `python -m src.train_ml --model svm --device cpu`. Các entrypoint `python src/train_svm.py`, `python src/train_ml.py`, `python src/preprocess.py` cũng được hỗ trợ. Mọi CLI nhận `--config PATH`, `--artifacts PATH`. Tiền xử lý có `--raw-dir ./temp_raw` cho máy ngoài Colab. `--overwrite` thay kết quả của đúng mô hình được yêu cầu; không xóa dữ liệu EEG/đặc trưng. `--resume` chỉ dùng lại run hoàn chỉnh cùng cấu hình và yêu cầu backend; muốn chạy lại sau khi chuyển runtime CPU/GPU, dùng `--overwrite`.

### GPU hoạt động thế nào?

CPU luôn đủ để chạy toàn bộ pipeline. Khi có NVIDIA, Linux x86_64 và CUDA 12 tương thích, cài thêm:

```bash
python -m pip install -r requirements-gpu.txt
```

Notebook tự thử cài phần này khi thấy `nvidia-smi`; có thể tắt bằng `ENABLE_GPU = False`. Với `--device auto`:

| Thuật toán/bước | Backend | Lý do |
| --- | --- | --- |
| SVM RBF | cuML GPU nếu khả dụng; nếu lỗi fit/chưa cài thì sklearn CPU | Giữ RBF và `class_weight='balanced'` |
| Logistic Regression | cuML GPU nếu khả dụng; nếu lỗi fit/chưa cài thì sklearn CPU | Giữ L2, C, balanced weights; `penalty_normalized=True` |
| Shrinkage LDA | sklearn CPU | Giữ `lsqr`, shrinkage tự động và priors cân bằng |
| Random Forest | sklearn CPU | Giữ đúng trọng số lớp cân bằng và baseline hiện tại |
| Lọc, ICA, đặc trưng, vẽ hình | CPU | MNE/SciPy/PyWavelets trong pipeline này |

`--device cpu` ép CPU. `--device gpu` ép GPU cho SVM/LR, báo lỗi nếu không dùng được; không âm thầm đổi sang CPU. Dùng `auto` khi chạy cả bốn mô hình. Backend thật và lý do fallback được ghi trong `fit_diagnostics.json`, backend/thời gian theo fold trong CSV. Thời gian bao gồm scaler và fit; GPU không đảm bảo nhanh hơn ở tập nhỏ. Model cuML đã lưu cần môi trường cuML/CUDA tương thích khi nạp lại.

Theo tài liệu [cuML LogisticRegression](https://docs.nvidia.com/cuml/26.10/api/generated/cuml.linear_model.LogisticRegression/) và [QN penalty normalization](https://docs.rapids.ai/api/libcuml/26.08/structqn__params), solver/stopping criterion có thể khác sklearn. Không giả định CPU/GPU cho kết quả giống từng bit. Nhánh GPU chưa được xác nhận hiệu năng trên phần cứng NVIDIA trong lần kiểm thử local này.

## Vì sao chọn bốn mô hình và vẽ đường học nào?

| Mô hình | Câu hỏi so sánh | Chẩn đoán huấn luyện |
| --- | --- | --- |
| Shrinkage LDA | Baseline ERP tuyến tính với ước lượng hiệp phương sai ổn định có đủ tốt? | Balanced Accuracy/F1 khi tăng **số người** train |
| SVM RBF | Ranh giới phi tuyến có cải thiện so với mô hình tuyến tính? | Learning curve theo số người, không gán thành loss theo epoch |
| Random Forest | Tương tác phi tuyến giữa các thống kê tín hiệu có ích? | Log loss train/validation theo số cây + learning curve |
| Logistic Regression L2 | Baseline tuyến tính phân biệt khác LDA thế nào? | Learning curve + callback L-BFGS của **lượt chẩn đoán CPU riêng** |

Tất cả đường chẩn đoán chỉ dùng **subjects trong train của outer fold đầu**, sau đó chia inner train/validation theo người. Learning curve dùng các tập người train lồng nhau, cố định inner validation. Không dùng outer test để vẽ đường train-validation rồi chọn siêu tham số. Với ít người, có thể chỉ có một điểm; nếu không đủ inner subjects thì lưu lý do và bỏ đường chẩn đoán.

LDA không tối ưu theo epoch. sklearn SVC không cung cấp lịch sử objective theo từng vòng. Random Forest thêm cây không phải tối ưu loss bằng gradient. Với Logistic Regression, `training_progress.csv` ghi objective thật qua callback SciPy: trung bình logistic loss có trọng số + `||w||²/(2*C*sum(weights))`; intercept không bị phạt. Objective và log loss không trọng số được vẽ riêng. Đây không phải lịch sử của fit GPU hoặc các fit outer CV. Validation loss không bắt buộc giảm. Trạng thái dừng của lượt chẩn đoán được ghi rõ.

## Tiền xử lý và đặc trưng

- Mã kích thích lấy từ cột `event_type`: **S5 / standard / frequent → 0; S6 / target → 1; S7 / distractor / deviant → bỏ**, theo [Bảng 3, Dzianok và cộng sự](https://pmc.ncbi.nlm.nih.gov/articles/PMC8900497/#tbl3). Không suy nhãn từ tỷ lệ xuất hiện. Mã stimulus lạ gây lỗi; response/run marker bị bỏ. Onset BIDS (giây) là nguồn thời điểm chia epoch.
- `snapshot_download` tải đúng một người vào `/content/temp_raw/sub-XX-...`, cố định revision HF bằng SHA. `finally: shutil.rmtree(...)` xóa cả raw/tải dở/cache riêng trước người kế tiếp. Tắt Xet cache dùng chung; không dùng `datasets.load_dataset()`. Đây là tải file theo người, không phải API streaming của Hugging Face Datasets.
- Mặc định chọn 30 kênh da đầu, có Fp1/Fp2/Fz/Cz/Pz. Đọc loại kênh, EOG override và cờ kênh xấu BIDS; nội suy kênh đã được đánh dấu. Montage 10-05 chuẩn phục vụ nội suy và QC; thiếu vị trí chỉ cảnh báo khi không cần nội suy. Nếu cần nội suy mà thiếu geometry, báo lỗi cụ thể để tránh dựng vị trí sai. Chưa có thuật toán tự động phát hiện/nội suy mọi kênh hỏng; `channel_quality.csv` cung cấp cờ heuristic để kiểm tra, không tự đổi tín hiệu.
- Notch **50 Hz** trước bandpass **0.5–30 Hz**, resample chống alias về **250 Hz**, cập nhật event đồng thời. Epoch vượt ranh giới bản ghi bị loại.
- ICA fit trên bản sao high-pass **1 Hz**, tối đa 300 giây đầu, 20 thành phần, decimation 2, seed cố định; áp dụng lên toàn bản ghi đã lọc 0.5–30 Hz. Ưu tiên EOG thật; nếu thiếu dùng Fp1/Fp2. Lưu excluded components, EOG scores và cảnh báo hội tụ. Proxy vùng trán có thể chứa tín hiệu thần kinh; cần xem QC, không xem ICA là đã loại hết nhiễu.
- Sau ICA: average reference, epoch **−0.2 đến 1.0 s**, baseline **−0.2 đến 0 s**, loại peak-to-peak >150 µV hoặc gần phẳng <0.1 µV. Lưu EEG theo volt, trục thời gian và thứ tự kênh nhất quán.
- Schema baseline giữ nguyên: **26 đặc trưng/kênh + 5 ROI = 785 cho 30 kênh**. Gồm 4 log10 band power Welch (delta/theta/alpha/beta), spectral entropy, Shannon entropy, 15 thống kê `db4` ở 5 mức, 5 thống kê P300 mean/peak/std/latency/signed AUC trong **300–600 ms**, thêm 5 thống kê trên waveform trung bình Fz/Cz/Pz. Peak/latency single-trial nhạy với nhiễu; delta trên cửa sổ 1 giây có độ phân giải thấp. Signed AUC (µV·s) khác ROC-AUC. Không coi mức wavelet trùng chính xác dải tần sinh lý.

Bản cập nhật này giữ đặc trưng và tham số baseline để so sánh với kết quả cũ. Chưa tự bật xDAWN, chuẩn hóa theo toàn subject mới, feature selection hay chọn ngưỡng F1. Các hướng đó cần thí nghiệm riêng, fit/chọn trên inner training folds và nêu rõ giả định triển khai; không suy rằng chúng chắc chắn tăng điểm. Nhận xét “3.460 đặc trưng/128 kênh” không đúng cấu hình mặc định đang chạy 30 kênh.

## Đánh giá và cách đọc kết quả

**GroupKFold theo người:** cùng một người không xuất hiện trong cả train/test. Tất cả mô hình dùng chung split manifest. `StandardScaler` nằm trong từng pipeline và chỉ fit trên train fold. SVM/RF/LR dùng balanced weights; LDA dùng shrinkage tự động, priors `[0.5, 0.5]`. Không oversample hay chọn ngưỡng bằng outer labels. Nếu solver benchmark chưa hội tụ, tăng giới hạn một lần; nếu vẫn thất bại thì dừng và không đánh dấu run hoàn tất.

CSV/JSON báo mean ± **sample SD (`ddof=1`)** giữa các fold cho Accuracy, Balanced Accuracy, Precision, Recall, F1 target, ROC-AUC và Average Precision. Có baseline accuracy luôn chọn lớp đa số. **F1 target** là tiêu chí so sánh định trước; vẫn đọc Recall/Precision và AP vì target hiếm. AP tính từ đường PR bậc thang gốc, không phải diện tích hình thang. ROC/PR tính riêng mỗi fold rồi mới lấy trung bình hình, không gộp decision score chưa hiệu chuẩn giữa các fold.

Confusion matrix dùng duy nhất dự đoán outer out-of-fold, có số đếm và phần trăm theo hàng. SD giữa fold chỉ mô tả biến thiên; train folds chồng lặp nên không dùng như thí nghiệm độc lập hay CI. Chọn mô hình đứng đầu trên bảng là so sánh thăm dò, chưa chứng minh vượt trội thống kê. `metrics_per_subject.csv` giúp xem khác biệt từng người.

ERP toàn nhóm lấy trung bình trial trong từng người/lớp trước, rồi trung bình người với trọng số bằng nhau. Shading là Student-t CI 95% **pointwise theo người**, không phải theo trial hay kiểm định đồng thời cả waveform. Một người thì không vẽ population CI. Bản đồ P300 là điện thế **da đầu** với vị trí chuẩn gần đúng, không phải định vị nguồn vỏ não. RF importance là MDI từ fold train, có bias với biến nhiều giá trị và chia sẻ giữa các biến tương quan; chỉ mô tả mô hình, không chứng minh nhân quả sinh lý.

ICA fit trên toàn bản ghi không nhãn của từng người mới và bộ lọc zero-phase sử dụng mẫu tương lai: đây là **offline decoding**, giả định đã có bản ghi người mới, không phải mô hình online causal. Model `*_final.pkl` được refit trên toàn cohort sau CV và không có điểm đánh giá độc lập riêng. Khi chỉnh đặc trưng/tham số sau khi xem benchmark, cần nested CV hoặc dữ liệu xác nhận mới.

## File kết quả

Mọi đầu ra đều dưới `artifacts/` (được Git ignore):

```text
artifacts/
├── X.npy, y.npy, groups.npy, ch_names.npy, times.npy
├── metadata.json, feature_metadata.json, environment.json
├── features.npy, feature_names.npy
├── subjects/sub-XX/                  # checkpoint để resume
├── qc/sub-XX/<recording>/            # raw/stage/ICA/ERP/topomap + metadata + report.html
├── dataset_overview.csv/.png
├── preprocessing_summary.csv, processed_trial_examples.png
├── ml/
│   ├── lda/
│   ├── svm/
│   ├── random_forest/
│   └── logistic_regression/
│       ├── metrics.csv/.json, metrics_per_fold.csv, metrics_per_subject.csv
│       ├── oof_predictions.csv, cv_splits.json, fit_diagnostics.json
│       ├── metrics.png, fig2_roc_and_pr_curves.png, confusion_matrix.csv/.png
│       ├── diagnostics.json, learning_curve.csv/.png
│       ├── training_progress.csv/.png  # RF và LR; nếu đủ inner subjects
│       ├── optimizer_diagnostics.json # LR diagnostic
│       ├── models/*_foldN.pkl, *_final.pkl
│       └── metadata.json, report.html
├── ml_benchmark_results.csv/.json
├── metrics_per_fold.csv, metrics_per_subject.csv, oof_predictions.csv, cv_splits.json
├── fig1_grand_average_erp.png
├── fig2_roc_and_pr_curves.png
├── fig3_confusion_matrices.png
├── fig4_feature_importance.png
├── model_comparison.png, metrics_table.png, feature_distributions.png
└── training_metadata.json
```

Tất cả hình 300 DPI. `metadata.json` của mỗi model chỉ được ghi sau khi đủ metrics, model và hình; chứa checksum kết quả. `compare_ml` từ chối run thiếu, file bị sửa hoặc protocol không khớp. `training_metadata.json` tổng hợp chỉ xuất khi đủ deliverables. Các file tổng hợp/model alias của luồng cũ được giữ tương thích; model export cũ ở `artifacts/models/` là bản sao của model trong từng run.

Mặc định QC chi tiết cho sub-01/02/03 trong lúc preprocessing mới; mọi người đều có số liệu reject/ICA trong checkpoint. `--resume` không tự tải lại raw chỉ vì thiếu hình QC. Dùng `--qc-only` khi cần. QC raw có trace/PSD giới hạn thời lượng để tiết kiệm RAM; cấu hình và chỉ số event giữ lại nằm trong metadata. Không thể dựng lại raw thật từ `X.npy` đã lọc.

## Cấu trúc code

```text
configs/config.yaml
notebooks/colab_workflow.ipynb
src/
  common.py, preprocess.py, qc.py, features.py
  models/lda.py, svm.py, random_forest.py, logistic_regression.py
  train_lda.py, train_svm.py, train_random_forest.py, train_logistic_regression.py
  train_ml.py          # CLI tổng hợp hoặc --model
  training.py          # fit, CV, xuất từng model, kiểm tra tính tương thích
  evaluation.py        # GroupKFold và metrics dùng chung
  diagnostics.py       # inner learning curves / loss thật khi có ý nghĩa
  backends.py          # cuML tùy chọn và CPU fallback
  compare_ml.py        # tổng hợp bốn run
  visualize.py, model_plots.py, data_overview.py, notebook.py
```

Cấu hình tại [`configs/config.yaml`](configs/config.yaml). Không đổi bộ lọc/ICA chỉ để bật QC hoặc GPU. Khi đổi preprocessing dùng checkpoint/output mới hoặc `--overwrite` có chủ đích; khi đổi feature settings chạy lại `src.features`, rồi train lại models. Kết quả cũ được phát hiện qua dataset/feature signature. Nếu runtime bị tắt cưỡng bức, thư mục raw tạm có thể còn lại; chỉ xóa khi chắc chắn không có tiến trình tải/xử lý đang chạy.

## Kiểm thử local

```bash
python -m unittest discover -s tests -v
```

Tests dùng BrainVision/BIDS tổng hợp và mock downloader: kiểm tra ánh xạ sự kiện, baseline/timing, ICA, ranh giới bản ghi, cleanup khi lỗi, P300 AUC/latency, CI theo người, scaler/train-test isolation, từng model/resume/so sánh, chặn kết quả cũ, inner diagnostic isolation, objective Logistic, QC-only giữ nguyên cohort, GPU fallback và cú pháp notebook. Không coi fixture tổng hợp hay validation vài người là kết quả khoa học của 42 người. Mở các hình thực tế trên Colab trước khi diễn giải sinh lý.
