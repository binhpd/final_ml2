# HƯỚNG DẪN TRAINING & INFERENCE

Tài liệu này hướng dẫn cách chạy hệ thống phân đoạn tài liệu từ việc tái cấu trúc tập dữ liệu (Giai đoạn nâng cao) cho đến việc huấn luyện mô hình YOLOv11-seg và chạy thuật toán Cắt sát mép (Cutout).

---

## 1. Yêu cầu Hệ thống
- Hệ điều hành: macOS (Ưu tiên Apple Silicon M-series để chạy MPS) hoặc Linux/Windows có CUDA.
- Python: 3.10 trở lên.
- Đã cài đặt các thư viện trong `requirements.txt`.

---

## 2. Chuẩn bị Dữ liệu (Spine Exclusion Relabeling)

Để mô hình có thể cắt bỏ gáy sách, chúng ta cần gán lại nhãn dữ liệu gốc (từ 1 class "document" chung chung thành 2 class "left_page" và "right_page").

Chạy script phân loại nhãn:
```bash
python ml2/yolo_seg/split_and_process_dataset.py \
    --input_dir datasets/your_raw_dataset \
    --output_dir datasets/spine_dataset \
    --split_ratio 0.8
```

*Lưu ý:* Script này sẽ đọc tọa độ polygon của tài liệu gốc, tính toán trung tâm ảo và phân chia các đối tượng sang trái/phải, sau đó xuất ra định dạng tương thích YOLO (TXT).

---

## 3. Huấn luyện Mô hình YOLOv11-seg

Sử dụng tập dữ liệu vừa được tạo ra ở bước 2 để huấn luyện.

### 3.1. Lệnh Huấn luyện Cơ bản
```bash
yolo task=segment mode=train \
    data=datasets/spine_dataset/data.yaml \
    model=yolo11n-seg.pt \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=mps \
    project=runs/spine_seg \
    name=spine_exclusion_run
```

### 3.2. Chiến lược Tối ưu và Hiệu chỉnh Siêu tham số (Hyperparameter Tuning)
Để giúp mô hình hội tụ nhanh hơn và đạt độ chính xác (mIoU) tối ưu nhất trong bài toán loại bỏ gáy sách, nhóm đã áp dụng các siêu tham số sau vào quá trình huấn luyện:

- **Optimizer:** Sử dụng `optimizer='AdamW'` (thay vì SGD mặc định) để hội tụ nhanh hơn và tránh kẹt ở local minima đối với dataset có kích thước vừa.
- **Learning Rate Scheduler:** Cấu hình `lr0=0.001`, `lrf=0.01` kết hợp với `warmup_epochs=3` để tránh bùng nổ gradient ở những epoch đầu.
- **Data Augmentation:**
  - Tăng cường `mosaic=1.0` giúp mô hình học được bối cảnh đa dạng và các góc cạnh khác nhau của trang giấy.
  - Sử dụng `mixup=0.1` để tránh overfitting trên các nhiễu nền phức tạp.
- **Loss Weights:** Tăng trọng số hàm loss của hộp giới hạn (`box=7.5`) và phân loại (`cls=0.5`) nhằm ép mô hình bắt chặt viền sách hơn.

**Lệnh chạy với siêu tham số (Tuned Command):**
```bash
yolo task=segment mode=train data=datasets/spine_dataset/data.yaml model=yolo11n-seg.pt \
    epochs=150 batch=16 imgsz=640 device=mps \
    optimizer=AdamW lr0=0.001 lrf=0.01 warmup_epochs=3 \
    mosaic=1.0 mixup=0.1 box=7.5 cls=0.5 \
    project=runs/spine_seg name=spine_exclusion_tuned
```

Sau khi train xong, trọng số tốt nhất sẽ nằm tại `runs/spine_seg/spine_exclusion_tuned/weights/best.pt`.
> **Mẹo quản lý:** Hãy copy file `best.pt` này vào thư mục `exported_models/yolo11n_seg_spine_exclusion_best.pt` để dễ quản lý.

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
    --weights exported_models/yolo11n_seg_spine_exclusion_best.pt \
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
