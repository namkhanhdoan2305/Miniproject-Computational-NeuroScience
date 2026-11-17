# Phân loại EEG từng lần kích thích: mô hình Oddball ba loại kích thích

Pipeline Python dành cho CPU, tập trung vào tiền xử lý tín hiệu, trích xuất đặc trưng thủ công trên từng lần kích thích và học máy truyền thống. Dữ liệu lấy từ [duylplsn/oddball](https://huggingface.co/datasets/duylplsn/oddball), gồm 42 bản ghi BrainVision kèm bảng sự kiện theo chuẩn BIDS. Dự án không sử dụng học sâu hoặc `datasets.load_dataset()`.

## Quy trình chạy trên Google Colab

Chọn môi trường chạy **CPU**. Clone repo vào ổ đĩa cục bộ của Colab, kết nối Google Drive và liên kết thư mục lưu kết quả trước khi xử lý. Chạy lần lượt các ô lệnh dưới đây.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/namkhanhdoan2305/Miniproject-Computational-NeuroScience.git /content/Miniproject-Computational-NeuroScience
%cd /content/Miniproject-Computational-NeuroScience
%pip install -q -r requirements.txt
```

```python
from pathlib import Path

drive_artifacts = Path('/content/drive/MyDrive/oddball_artifacts')
drive_artifacts.mkdir(parents=True, exist_ok=True)
link = Path('artifacts')
if not link.exists() and not link.is_symlink():
    link.symlink_to(drive_artifacts, target_is_directory=True)
assert link.is_symlink() and link.resolve() == drive_artifacts.resolve(), (
    'artifacts phải trỏ tới thư mục Drive đã chọn; hãy sao lưu hoặc di chuyển kết quả cũ trước.'
)
```

Trước tiên, xử lý **toàn bộ bản ghi của sub-01** để kiểm tra tiền xử lý. Chế độ `--dry-run` vẫn tải dữ liệu EEG thật và thực hiện lọc, ICA, chia đoạn tín hiệu (epoch) và loại đoạn nhiễu; đây không phải chế độ mô phỏng hoặc chạy không cần tải dữ liệu.

```bash
!python -m src.preprocess --dry-run --resume
```

```python
from IPython.display import Image, display
display(Image('artifacts/erp_sub01_check.png'))
```

Kiểm tra dạng sóng trung bình của từng lớp tại Fz/Pz và file `artifacts/subjects/sub-01/metadata.json`, trong đó có số lượng từng lớp, lý do loại epoch, các thành phần ICA bị loại, tương quan EOG và cảnh báo hội tụ. Sau đó xử lý toàn bộ nhóm người tham gia, dùng lại kết quả sub-01 và chạy các bước còn lại:

```bash
!python -m src.preprocess --resume
!python -m src.features
!python -m src.train_ml
!python -m src.visualize  # Tùy chọn: vẽ lại các hình mà train_ml đã xuất.
```

```python
import pandas as pd
display(pd.read_csv('artifacts/ml_benchmark_results.csv'))
for filename in ['fig1_grand_average_erp.png', 'fig2_roc_and_pr_curves.png',
                 'fig3_confusion_matrices.png', 'fig4_feature_importance.png']:
    display(Image('artifacts/' + filename))
```

Thông thường có thể tải dữ liệu công khai mà không cần token. Nếu Hugging Face yêu cầu xác thực, gọi `huggingface_hub.login()` trong notebook trước khi chạy các script; không commit token lên Git. Bước tiền xử lý cần kết nối Internet để tải dữ liệu từ Hugging Face.

## Cách pipeline hoạt động

**Đọc và gán nhãn sự kiện.** Cột `trial_type` trong dữ liệu chỉ chứa `stimulus` hoặc `response`; thông tin phân lớp nằm trong cột `event_type`. Code chuẩn hóa khoảng trắng và chữ hoa/chữ thường. Quy tắc gán nhãn là **S5 / standard / frequent → 0, S6 / target → 1, S7 / distractor / deviant → loại bỏ**, dựa trên định nghĩa mã kích thích trong [Bảng 3 của Dzianok và cộng sự](https://pmc.ncbi.nlm.nih.gov/articles/PMC8900497/#tbl3). Các mốc phản hồi, nghỉ và bắt đầu lượt ghi không được đưa vào phân loại. Code báo lỗi nếu gặp mã kích thích chưa được ánh xạ hoặc thiếu một trong hai lớp cần giữ. Nhãn không được suy ra từ tần suất xuất hiện hay hành vi phản hồi. Thời điểm bắt đầu sự kiện trong TSV, tính bằng giây, được dùng để chia epoch; annotation của BrainVision được dùng để xác định ranh giới bản ghi. Thời điểm trong TSV có thể lệch một mẫu gốc so với vị trí marker BrainVision đánh số từ 1. Sau khi loại distractor, tỷ lệ các lớp thay đổi: cần dùng số lượng thực tế được xuất ra, thay vì giả định tỷ lệ của mô hình ba lớp ban đầu.

**Tiền xử lý.** Mỗi người được tải bằng `snapshot_download(repo_type='dataset', allow_patterns=...)` vào một thư mục con riêng trong `/content/temp_raw`. Phiên bản repo được xác định bằng mã commit SHA cụ thể. Dữ liệu thô, file tải dở và metadata bộ nhớ đệm Hub cục bộ được xóa bằng `shutil.rmtree` trong khối `finally` trước khi chuyển sang người tiếp theo. Bộ nhớ đệm Xet được tắt để tránh lưu các phần dữ liệu thô vào cache dùng chung. Đây là cách tải file giới hạn theo từng người, không phải API streaming của thư viện Hugging Face Datasets. Nếu môi trường chạy bị tắt cưỡng bức, thư mục tạm có thể còn sót lại; sau khi chắc chắn không có tiến trình tiền xử lý đang chạy, hãy xóa thư mục `sub-XX-*` còn sót trước khi khởi động lại.

Cấu hình mặc định chọn 30 kênh da đầu theo thứ tự cố định, bao gồm Fz, Pz, Fp1 và Fp2, nhằm giảm nhu cầu CPU/RAM trên Colab. Đặt `preprocessing.channels: null` nếu muốn giữ toàn bộ kênh EEG, chẳng hạn 127 kênh ở sub-01; cách này sẽ tốn nhiều RAM, dung lượng và thời gian huấn luyện hơn. Tất cả người tham gia và lượt ghi phải có cùng thứ tự kênh và trục thời gian khi lưu. Code đọc loại kênh và cờ đánh dấu kênh xấu trong BIDS nếu có; các kênh xấu được nội suy dựa trên vị trí điện cực chuẩn. Nếu thiếu kênh được yêu cầu, code sẽ báo lỗi.

Các bước gồm lọc notch 50 Hz, lọc thông dải 0.5–30 Hz và đổi tần số lấy mẫu về 250 Hz có chống chồng phổ, đồng thời cập nhật vị trí sự kiện. ICA được ước lượng trên một bản sao đã lọc thông cao 1 Hz, mặc định dùng năm phút đầu, 20 thành phần và giảm số mẫu đầu vào theo hệ số 2; sau đó phép loại thành phần được áp dụng cho toàn bộ bản ghi. Code ưu tiên kênh EOG thật; dữ liệu này không có kênh EOG chuyên dụng nên mặc định dùng Fp1/Fp2 làm tín hiệu thay thế. EEG vùng trán cũng có thể chứa hoạt động thần kinh, vì vậy cần kiểm tra tương quan, thành phần bị loại và các hình ERP. Quy trình tự động xử lý nhiễu liên quan đến mắt, không nhận diện được mọi loại nhiễu có thể có. Sau ICA, tín hiệu được chuyển sang tham chiếu trung bình. Epoch kéo dài từ −200 đến 1000 ms, với hiệu chỉnh đường nền (baseline) từ −200 đến 0 ms; ngưỡng loại theo biên độ đỉnh–đỉnh mặc định là 150 µV và ngưỡng loại tín hiệu gần phẳng là 0.1 µV. Các epoch đi qua ranh giới bản ghi được đánh dấu sẽ bị loại. Tất cả lần kích thích standard/target còn lại đều có thể được sử dụng, kể cả những lần người tham gia phản hồi sai.

**Trích xuất đặc trưng.** Mỗi kênh tạo ra 26 đặc trưng. Năm đặc trưng bổ sung từ vùng quan tâm (ROI) đường giữa được xác định trước tạo thành **785 đặc trưng cho 30 kênh**:

- Log10 công suất tuyệt đối ước lượng bằng Welch trong các dải delta (0.5–4), theta (4–8), alpha (8–13) và beta (13–30 Hz). Mật độ phổ công suất (PSD) được tích phân với nội suy tại biên dải tần; đơn vị trước khi lấy log là µV². Cửa sổ một giây có độ phân giải thấp ở tần số thấp, nên cần thận trọng khi diễn giải dải delta.
- Entropy phổ đã chuẩn hóa trong dải 0.5–30 Hz và entropy Shannon của histogram biên độ, gồm 32 khoảng chia trong từng lần kích thích. Cả hai dùng logarit tự nhiên và trả về 0 đối với tín hiệu không đổi hoặc toàn số 0.
- Các thống kê hệ số biến đổi wavelet rời rạc `db4` tại A4, D4, D3, D2, D1: trung bình trị tuyệt đối, độ lệch chuẩn và log10 trung bình bình phương hệ số. Những thống kê có kích thước cố định này đại diện cho các hệ số; code không nối toàn bộ vector hệ số vào đặc trưng. Các mức wavelet không được xem là trùng chính xác với các dải tần sinh lý.
- Biên độ trung bình, đỉnh dương, độ lệch chuẩn, độ trễ đỉnh và **diện tích có dấu dưới dạng sóng** trong cửa sổ P300 từ 300 đến 600 ms. Biên độ tính bằng µV, độ trễ tính bằng ms kể từ lúc kích thích, diện tích tính bằng µV·s và khác với ROC-AUC trong phân loại. Diện tích được tính bằng quy tắc hình thang trên trục thời gian đã lưu; dạng sóng âm có diện tích âm.
- Năm thống kê P300 tương tự trên dạng sóng trung bình của Fz/Cz/Pz (`features.roi_channels`). Đỉnh và độ trễ của ROI được tính sau khi lấy trung bình các dạng sóng. Cách này nhấn mạnh P300 ở vùng đường giữa/đỉnh đầu mà không chọn kênh dựa trên nhãn của tập đánh giá. Đặt danh sách này thành `[]` nếu lựa chọn kênh không bao gồm các kênh nói trên.

Đặc trưng phổ, histogram và wavelet sử dụng cửa sổ 0–1000 ms; P300 dùng cửa sổ riêng. Việc trích xuất được thực hiện theo từng lô trên mảng EEG ánh xạ bộ nhớ (memory-mapped), sử dụng tần số lấy mẫu và trục thời gian **đã lưu**. Đặc trưng chỉ sử dụng dữ liệu của chính lần kích thích đó; bước này không chuẩn hóa trên toàn bộ nhóm người tham gia và không lựa chọn đặc trưng.

**Huấn luyện và đánh giá.** `GroupKFold` với năm fold chia tập huấn luyện/kiểm tra theo toàn bộ người tham gia; mỗi lần kích thích được kiểm tra đúng một lần và mỗi fold phải chứa cả hai lớp. Bộ chuẩn hóa nằm trong pipeline của từng mô hình và chỉ được ước lượng từ phần huấn luyện của mỗi fold. Bốn mô hình dùng chung cách chia: **Shrinkage LDA** (`solver='lsqr', shrinkage='auto'`, xác suất tiên nghiệm mặc định `[0.5, 0.5]`), **SVM nhân RBF**, **Random Forest** và **Logistic Regression điều chuẩn L2**. SVM, Random Forest và Logistic Regression sử dụng `class_weight='balanced'`. Code không tinh chỉnh theo fold kiểm tra, chia ngẫu nhiên từng lần kích thích, tăng mẫu lớp hiếm (oversampling), hiệu chuẩn xác suất hoặc chọn ngưỡng dựa trên kết quả đánh giá. SVM RBF có thể mất nhiều thời gian hơn khi chạy toàn bộ nhóm người tham gia; dung lượng cache của SVM và số tác vụ song song của Random Forest được giới hạn trong cấu hình.

`ml_benchmark_results.csv` và `.json` lưu **trung bình và độ lệch chuẩn mẫu (ddof=1)** trên các fold kiểm tra cho Accuracy, Balanced Accuracy, Precision, Recall, F1 của lớp target, ROC-AUC và Average Precision (AP). Giá trị nằm trong thang 0–1, không phải phần trăm; Precision/F1 bằng 0 khi không có mẫu nào được dự đoán là target. AP tóm tắt các bậc của đường precision–recall gốc, khác với diện tích PR tính bằng quy tắc hình thang. File JSON còn chứa số lượng thực tế của từng lớp và mốc so sánh là mô hình luôn dự đoán lớp đa số. Độ lệch chuẩn giữa các fold chỉ mang tính mô tả; các tập huấn luyện chồng lặp không phải những lần lặp độc lập, và SD không phải khoảng tin cậy.

Các kết quả khác gồm chỉ số theo từng fold và từng người, Accuracy/F1 trên các dự đoán kiểm tra gộp từ các fold (out-of-fold), danh sách người trong từng tập và dự đoán kiểm tra cho từng lần kích thích. Trong mỗi fold, các lần kích thích có trọng số bằng nhau khi tính chỉ số; `metrics_per_subject.csv` giúp kiểm tra khác biệt giữa người tham gia. Các hình ROC và PR lấy trung bình những đường cong được tính riêng trong từng fold, tránh so sánh trực tiếp điểm quyết định chưa hiệu chuẩn giữa các fold. Điểm số là giá trị hàm quyết định đối với SVM/LDA/Logistic Regression và xác suất target đối với Random Forest. Ma trận nhầm lẫn chỉ chứa dự đoán out-of-fold. Sau CV của từng mô hình, mô hình cuối được huấn luyện lại trên toàn bộ người tham gia và không có thêm điểm đánh giá trên tập độc lập. Chế độ dry-run một người không đủ để chạy CV theo người.

ICA là bước tiền xử lý không giám sát, thực hiện riêng trên bản ghi không nhãn của từng người, kể cả người thuộc fold kiểm tra. Do đó, cách đánh giá này giả định có sẵn toàn bộ bản ghi không nhãn của người mới; nó không đo khả năng giải mã trực tuyến theo thời gian thực với phép tiền xử lý cố định. Bộ lọc không lệch pha cũng sử dụng các mẫu ở thời điểm tương lai. Nếu thay đổi đặc trưng hoặc siêu tham số sau khi xem kết quả, cần đánh giá trên dữ liệu độc lập mới hoặc dùng CV lồng nhau theo người.

**Trực quan hóa.** Tất cả hình được xuất ở 300 DPI. ERP trung bình toàn nhóm được tính bằng cách lấy trung bình từng lớp trong từng người trước, rồi lấy trung bình giữa những người với trọng số bằng nhau. Vùng tô tại Fz/Cz/Pz là **khoảng tin cậy Student-t 95% tại từng thời điểm, tính trên các giá trị trung bình theo người**, không phải khoảng tin cậy trên các lần kích thích hay kiểm định ý nghĩa đồng thời trên toàn bộ dạng sóng. Không vẽ khoảng tin cậy quần thể khi chỉ có một người. Cửa sổ P300 được tô nền. Vùng tô trên hình ROC/PR biểu thị ±1 độ lệch chuẩn giữa các fold và được ghi chú riêng. Ma trận nhầm lẫn hiển thị cả số lượng thực tế và tỷ lệ phần trăm chuẩn hóa theo hàng cho cả bốn mô hình, sắp theo F1 của target để so sánh thăm dò.

Hình top 20 đặc trưng quan trọng tổng hợp **mức giảm độ hỗn tạp trung bình (mean decrease in impurity) của Random Forest trong các fold huấn luyện**, kèm thanh sai số SD giữa các fold và file CSV đầy đủ. Nhãn của tập kiểm tra không được dùng để lựa chọn đặc trưng. Chỉ số này có thể ưu tiên biến có nhiều giá trị khác nhau và chia mức quan trọng giữa các đặc trưng tương quan; nó mô tả mô hình, không chứng minh quan hệ nhân quả về sinh lý thần kinh hay ý nghĩa thống kê của từng kênh. Thứ hạng đặc trưng không được dùng ngược lại để thay đổi benchmark. Biểu đồ phân bố đặc trưng sử dụng giá trị trung bình theo từng người/lớp, với trọng số bằng nhau giữa những người tham gia.

## Cấu hình và tiếp tục xử lý

Chỉnh [`configs/config.yaml`](configs/config.yaml) để thay đổi bộ lọc, danh sách kênh, ICA, ngưỡng loại nhiễu, cửa sổ đặc trưng, số fold CV và giới hạn CPU. Các đường dẫn tương đối được tính từ thư mục làm việc hiện tại; hãy chạy lệnh tại thư mục gốc của repo. Mọi script đều nhận `--config PATH` và `--artifacts PATH`; riêng tiền xử lý còn nhận `--raw-dir PATH`, `--resume` và `--overwrite`. Có thể chạy dưới dạng module hoặc gọi trực tiếp script:

```bash
python src/preprocess.py --dry-run --raw-dir ./temp_raw --resume
python src/preprocess.py --raw-dir ./temp_raw --resume
python src/features.py
python src/train_ml.py
python src/visualize.py              # Vẽ lại toàn bộ hình từ kết quả ML đã hoàn tất.
python src/visualize.py --erp-only   # Chỉ vẽ ERP, trước khi trích đặc trưng/huấn luyện.
```

Nếu muốn chạy nhóm nhỏ hơn, đặt `dataset.subjects: [sub-01, sub-02, sub-03, sub-04, sub-05]` và chọn số fold CV không lớn hơn số người. Mỗi tập huấn luyện/kiểm tra phải có cả hai lớp. Nên dùng thư mục kết quả riêng cho những thử nghiệm có cấu hình tiền xử lý khác nhau.

Điểm lưu tiến độ (checkpoint) đã hoàn thành của từng người chỉ được dùng lại khi mã commit nguồn, phiên bản triển khai tiền xử lý, cấu hình tiền xử lý/sự kiện và seed khớp nhau. Nếu xử lý một người bị gián đoạn, người đó sẽ được xử lý lại; checkpoint đã hoàn tất vẫn được giữ trên Drive. `--overwrite` xử lý lại những người được chọn. Dry-run không ghi đè bộ dữ liệu nhiều người đã xử lý; hãy chọn đường dẫn `--artifacts` khác nếu cần kiểm tra riêng. Bước gộp dữ liệu ghi dấu hoàn tất sau cùng và không nạp toàn bộ dữ liệu của nhóm vào RAM. Thay đổi mảng dữ liệu đã gộp sẽ làm các đặc trưng cũ không còn hợp lệ; cần chạy lại trích đặc trưng, huấn luyện và vẽ hình sau tiền xử lý. Hình/mô hình cũ có thể vẫn còn cho đến khi chạy lại bước tương ứng; `metadata.json`, `feature_metadata.json` và `training_metadata.json` ghi nguồn gốc và trạng thái hoàn tất của từng bước.

Sau khi cập nhật một bản repo cũ, chạy lại `src.features` để tạo cấu trúc đặc trưng mới có diện tích P300 và ROI. Nếu cấu hình tiền xử lý/sự kiện thay đổi, kể cả danh sách tên tương đương của nhãn, hãy dùng `--overwrite` hoặc thư mục artifacts mới; code sẽ từ chối dùng checkpoint không tương thích. Lệnh chỉ vẽ hình sẽ từ chối kết quả ML cũ hoặc chưa hoàn tất, thay vì trộn kết quả từ nhiều lần chạy. `training_metadata.json` chỉ được ghi khi tất cả hình và bảng bắt buộc đã được tạo thành công. Cảnh báo không hội tụ của mô hình ML sẽ dừng huấn luyện; cảnh báo và thành phần ICA bị loại vẫn được lưu để kiểm tra chất lượng (QC).

## File kết quả

Toàn bộ dữ liệu và kết quả được tạo trong `artifacts/`:

| File/thư mục | Nội dung |
| --- | --- |
| `X.npy` | EEG Float32, đơn vị **volt**, kích thước `(trials, channels, samples)` tương ứng số lần kích thích, số kênh và số mẫu; mặc định 301 mẫu mỗi epoch |
| `y.npy`, `groups.npy` | Nhãn Int8 0/1; mã người tham gia dạng Unicode, một mã cho mỗi lần kích thích |
| `ch_names.npy`, `times.npy` | Tên kênh Unicode theo thứ tự; trục thời gian của epoch tính bằng giây |
| `metadata.json`, `subjects/sub-XX/` | Nguồn gốc dữ liệu toàn nhóm và checkpoint/QC đã xử lý theo từng người để tiếp tục khi gián đoạn |
| `features.npy`, `feature_names.npy`, `feature_metadata.json` | Ma trận đặc trưng Float32, tên theo thứ tự, cấu hình trích xuất và thông tin nguồn dữ liệu |
| `models/*.pkl` | Gói mô hình của từng fold và mô hình cuối, kèm pipeline và cấu trúc đặc trưng |
| `ml_benchmark_results.csv`, `ml_benchmark_results.json` | So sánh bốn mô hình: trung bình/SD của sáu chỉ số bắt buộc và AP |
| `fig1_grand_average_erp.png`, `fig1_grand_average_erp.csv` | ERP tại Fz/Cz/Pz, CI 95% theo thời điểm, cửa sổ P300 được tô nền và dữ liệu dùng để vẽ |
| `fig2_roc_and_pr_curves.png` | ROC và PR đặt cạnh nhau, kèm trung bình AUC/AP giữa các fold ± SD |
| `fig3_confusion_matrices.png` | Số lượng dự đoán kiểm tra thực tế và tỷ lệ phần trăm theo hàng của mọi mô hình |
| `fig4_feature_importance.png` | Top 20 đặc trưng RF từ các fold huấn luyện, có nhãn kênh/cửa sổ thời gian |
| `feature_importance.csv`, `feature_importance_per_fold.csv` | Toàn bộ thống kê mức quan trọng theo độ hỗn tạp của RF và giá trị từng fold |
| `metrics_summary.csv`, `metrics_per_fold.csv`, `metrics_per_subject.csv`, `metrics_table.png` | Bản sao bảng benchmark, chỉ số theo fold/người và bảng kết quả dạng ảnh |
| `cv_splits.json`, `oof_predictions.csv`, `training_metadata.json` | Danh sách chia theo người, điểm số kiểm tra và thông tin huấn luyện |
| `roc_curves.png`, `roc_curves.csv`, `pr_curves.csv` | Bản sao ảnh ROC/PR và dữ liệu đường cong trung bình |
| `confusion_matrix_*.png`, `confusion_matrix_*.csv` | Số lượng dự đoán kiểm tra và hình ma trận chuẩn hóa theo hàng |
| `erp_sub01_check.png`, `grand_average_erp.png` | Hình kiểm tra tiền xử lý Fz/Pz và bản sao ERP toàn nhóm tại ba kênh |
| `feature_distributions.png`, `feature_distributions.csv` | So sánh đặc trưng dựa trên trung bình theo người |

Checkpoint từng người và `X.npy` đã gộp chứa hai bản dữ liệu đã xử lý để hỗ trợ tiếp tục chạy; cần dự trù dung lượng Drive cho cả hai. Tại mỗi thời điểm, ổ đĩa cục bộ chỉ lưu bản ghi thô của một người. Có thể nạp gói mô hình bằng `pickle` trong môi trường Python tương thích; chỉ nạp các file đáng tin cậy do chính bạn tạo. Dùng `bundle['pipeline'].predict(features)` với đúng thứ tự `bundle['feature_names']` và cấu hình tiền xử lý/trích đặc trưng tương ứng.

## Kiểm tra trên máy cục bộ

Dự án hỗ trợ Python 3.10 trở lên. Cài `requirements.txt`, sau đó chạy bộ kiểm tra tích hợp không cần mạng:

```bash
python -m unittest discover -s tests -v
```

Các bài kiểm tra bao gồm tên nhãn tương đương và loại distractor, đọc dữ liệu BrainVision mô phỏng, ICA/thời điểm epoch/baseline, xóa raw khi lỗi, gộp dữ liệu, diện tích P300 có dấu và ROI, trung bình thực tế của scaler trên fold huấn luyện, cấu hình cân bằng lớp, CV tách theo người, tính lại chỉ số kiểm tra, CI 95% theo người và tất cả file kết quả bắt buộc, bao gồm độ phân giải PNG. Kết quả mô phỏng dùng để xác minh hành vi phần mềm, không ước lượng hiệu năng trên EEG thật. Xem [AUDIT.md](AUDIT.md) để biết các phát hiện và giới hạn diễn giải còn lại.

## Tài liệu tham khảo

- [Bản dữ liệu trên Hugging Face](https://huggingface.co/datasets/duylplsn/oddball), [nguồn OpenNeuro](https://openneuro.org/datasets/ds004621/versions/1.0.4) và [Dzianok và cộng sự (2022)](https://doi.org/10.1093/gigascience/giac015).
- [Hướng dẫn ICA của MNE](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html), [hướng dẫn tải dữ liệu Hugging Face](https://huggingface.co/docs/huggingface_hub/guides/download) và [GroupKFold của scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [xác suất tiên nghiệm/shrinkage của LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html), [đường precision–recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) và [phân phối Student-t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html).
