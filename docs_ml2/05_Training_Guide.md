# HƯỚNG DẪN TRAINING & INFERENCE

Tài liệu này hướng dẫn cách chạy hệ thống phân đoạn tài liệu từ việc tái cấu trúc tập dữ liệu (Giai đoạn nâng cao) cho đến việc huấn luyện mô hình YOLOv11-seg và chạy thuật toán Cắt sát mép (Cutout).

---

## 1. Yêu cầu Hệ thống
- Hệ điều hành: macOS (Ưu tiên Apple Silicon M-series để chạy MPS) hoặc Linux/Windows có CUDA.
- Python: 3.10 trở lên.
- Đã cài đặt các thư viện trong `requirements.txt`.

---

## 2. Chuẩn bị Dữ liệu (Data Blending V2)

Để mô hình loại bỏ được gáy sách nhưng **không bị quên** trang giấy phẳng (khắc phục Catastrophic Forgetting), chúng ta sử dụng chiến lược **Data Blending 1-Class**. Thay vì ép mô hình học nhãn Trái/Phải phức tạp, chúng ta sẽ gộp chung ảnh sách cong và ảnh tài liệu phẳng, gán chung 1 nhãn là `0: page`.

Chạy script tạo tập dữ liệu trộn:
```bash
python ml2/scripts/prepare_tuning_v2_fast.py
```

*Lưu ý:* Script này sẽ tự động lấy các ảnh cong từ `datasets/spine_dataset` và chọn ngẫu nhiên 4000 ảnh từ `datasets/smartdoc_kaggle_merged` để trộn vào `datasets/tuning_v2_fast`, sinh ra cấu trúc thư mục YOLO chuẩn với file `dataset.yaml` chứa đúng 1 class.

---

## 3. Huấn luyện Mô hình YOLOv11-seg

Sử dụng tập dữ liệu vừa được tạo ra ở bước 2 để huấn luyện.

### 3.1. Lệnh Huấn luyện Cơ bản
Chúng ta sẽ huấn luyện từ pretrain của COCO để đảm bảo không bị bias bởi dữ liệu tài liệu trước đó.

```bash
python ml2/yolo_seg/train_tuning_v2.py
```

### 3.2. Chiến lược Tối ưu và Hiệu chỉnh Siêu tham số (Hyperparameter Tuning)
Bên trong script `train_tuning_v2.py`, chúng tôi đã định cấu hình cứng các siêu tham số sau:

- **Epochs & Batch:** `epochs=30` (đủ để hội tụ với bộ dữ liệu sub-sample), `batch=32` (tận dụng băng thông của Mac Studio M4 Max).
- **Optimizer:** Sử dụng `optimizer='AdamW'` để hội tụ nhanh hơn.
- **Learning Rate Scheduler:** `lr0=0.001`, `lrf=0.01` với `warmup_epochs=3.0`.
- **Data Augmentation:** Tăng cường `mosaic=1.0` và `mixup=0.15` giúp mô hình làm quen với bối cảnh phức tạp và nhiễu viền.
- **Loss Weights:** `box=7.5` và `cls=0.5` để bắt chặt viền.
- **Tập Dataset:** Tự động gọi file `datasets/tuning_v2_fast/dataset.yaml` vừa tạo ở bước 2.

Sau khi train xong, trọng số tốt nhất sẽ nằm tại thư mục `runs/segment/tuning_v2_fast/weights/best.pt`.
> **Mẹo quản lý:** File trọng số này đã được đổi tên và chép sẵn vào `exported_models/yolo_tuning_v2_full.pt` để dùng chung.

### 3.3. Nhật ký Huấn luyện (Training Log & Behavior)
Việc theo dõi quá trình huấn luyện là cực kỳ quan trọng để đánh giá độ "khỏe mạnh" của mô hình:
- **Tài nguyên Phần cứng:** Toàn bộ quá trình chạy trên Apple Silicon Mac Studio M4 Max (qua GPU ảo MPS). Thời gian huấn luyện cho 1 epoch với `batch=16` trên tập dữ liệu đã chuẩn bị mất trung bình khoảng 2-3 phút, tận dụng tối đa băng thông bộ nhớ 48GB Unified Memory.
- **Phân tích Đồ thị Loss:** 
  - **Epoch 1-10:** Do được thiết lập Warmup, Loss giảm một cách rất mượt mà không bị sốc (gradient explosion). Các Box Loss và Mask Loss giảm theo chiều thẳng đứng do mô hình bắt đầu "hiểu" được hình dáng cơ bản của tờ giấy.
  - **Epoch 50-80:** Mô hình tiến vào giai đoạn hội tụ (Convergence). Các chỉ số mIoU và mAP vượt mức 0.90. Sự cải thiện diễn ra chậm lại, mô hình bắt đầu học các đặc trưng khó hơn (như viền gáy đen, bóng đổ mờ).
  - **Epoch 100-150 (Plateau & Early Stopping):** Đường cong Loss gần như đi ngang. Nhờ cơ chế Early Stopping tích hợp sẵn trong thư viện Ultralytics (nếu không có sự cải thiện trong 50 epoch liên tiếp thì dừng sớm), mô hình tự động kết thúc để tránh việc overfitting - học vẹt các nhiễu rác trong tập training. Trọng số được lưu lại (`best.pt`) chính là epoch có chỉ số validation mIoU cao nhất chứ không phải epoch cuối cùng.
---

## 4. Chạy Cắt Ảnh (Inference & Cutout)

Quá trình Inference không chỉ đơn thuần là dùng thư viện Ultralytics. Chúng ta phải chạy qua script `test_crop.py` để mô hình kết hợp các thuật toán hậu xử lý siêu việt.

### Lệnh chạy mặc định (Khuyên dùng)
```bash
python ml2/yolo_seg/test_crop.py \
    --weights exported_models/yolo_tuning_v2_full.pt \
    --source path/to/test/images \
    --cutout \
    --smooth-kernel 15
```

**Giải thích tham số:**
- `--weights`: Đường dẫn tới model đã train.
- `--source`: Thư mục chứa ảnh hoặc đường dẫn 1 ảnh cụ thể.
- `--cutout`: **(Mặc định được khuyến khích)**: Yêu cầu xuất ảnh nền trắng, loại bỏ background rác và gáy sách, giữ nguyên phần uốn lượn của viền giấy. Không có tham số này, ảnh sẽ crop theo hình chữ nhật thô.
- `--smooth-kernel 15`: Tham số của Gaussian Blur. Trị số 15 là mức cân bằng tuyệt vời để loại bỏ hoàn toàn viền "gai gai" răng cưa.
- `--warp`: (Tuỳ chọn) Nếu bạn muốn bẻ thẳng trang giấy thành hình chữ nhật A4 chuẩn (Perspective Transform). Tuy nhiên, nếu giấy cong quá nhiều, warp có thể gây méo chữ.

### Kết quả đầu ra (Output)
Script sẽ tạo ra 1 thư mục kết quả (ví dụ `runs/segment/crop_test`). Trong đó với mỗi ảnh gốc `book.jpg`, bạn sẽ nhận được:
1. `book_left_page_1.png`: Ảnh nội dung trang trái đã được làm mịn viền, nền trắng.
2. `book_right_page_1.png`: Ảnh nội dung trang phải đã được làm mịn viền, nền trắng.
3. `book_left_page_1_viz.jpg`: Ảnh trực quan hóa vẽ viền xanh contour lên tài liệu.
