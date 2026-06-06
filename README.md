# Dự án Quét và Phân tích Tài liệu (Document Scanner)

Dự án này bao gồm hai cách tiếp cận chính để xử lý trích xuất và tối ưu hoá giao diện tài liệu (Document Scanner/Dewarping):

1. **Pipeline Without ML (Truyền thống)**: Dựa hoàn toàn vào các kỹ thuật xử lý ảnh (Computer Vision) thuần tuý thông qua OpenCV như Canny Edge Detection, Hough Lines, Thresholding, Morphology.
2. **Pipeline With ML (Hiện đại)**: Kết hợp các kỹ thuật Computer Vision với sức mạnh của Trí tuệ Nhân tạo (Machine Learning/Deep Learning) sử dụng các mô hình học sâu như **DocAligner**, **YOLOv8** để định vị góc cực kỳ chính xác.

---

## 1. Pipeline Wihtout ML (Chỉ dùng Xử lý ảnh OpenCV)

Thư mục: `Pipeline Without ML/`

Đây là cách tiếp cận ban đầu, sử dụng các phép biến đổi toán học phân đoạn. Gồm các bước rời rạc để bạn nghiên cứu từng phần:

* **Step 1 canny edge detection**:
    * Phát hiện cạnh bằng thuật toán Canny.
    * Tìm Contour lớn nhất có 4 đỉnh (`approxPolyDP`) hoặc nội suy bằng Hough Lines nếu bị khuyết.
    * *Lệnh chạy (Ví dụ)*: `python3 "Pipeline Without ML/Step 1 canny edge detection/main.py"`

* **Step 2 perspective transform**:
    * Cắt ảnh và biến đổi phối cảnh 3D thành bản phẳng 2D dựa trên 4 góc đã tìm được ở Step 1.
    * *Lệnh chạy (Ví dụ)*: `python3 "Pipeline Without ML/Step 2 perspective transform/main.py"`

* **Step 3 enhancement**:
    * Khử đổ bóng (Shadow Removal) bằng phép chia ảnh với nền (Dilate & Gaussian).
    * Binarization (nhị phân hoá chữ đen nền trắng) để thành chất lượng chuẩn máy scan.
    * *Lệnh chạy (Ví dụ)*: `python3 "Pipeline Without ML/Step 3 enhancement/main.py"`

> **Lưu ý**: Nhược điểm của phương pháp này là dễ bị thất bại với các ảnh tối, nền phức tạp hoặc tài liệu bị ngón tay che khuất viền.

---

## 2. Pipeline With ML (Sử dụng AI Phân vùng vùng ảnh)

Thư mục: `Pipeline With ML/`

Đây là Pipeline rút gọn, nối tự động (End-to-End) cả 3 quá trình trên vào trong một file chạy duy nhất. Đặc biệt, nó tích hợp **Mạng nơ-ron Tích chập (CNN)** chuyên dụng giải quyết mọi nhược điểm của OpenCV truyền thống.

### Hướng dẫn Cài đặt & Khởi chạy (Môi trường ML):
Pipeline này sử dụng Machine Learning nên yêu cầu cài đặt môi trường ảo riêng biệt chứa các bộ thư viện hạng nặng (`torch`, `onnxruntime`, `docaligner`...).

**Bước 1: Tạo môi trường ảo (Tuỳ chọn nhưng Khuyến nghị)**
```bash
# Tạo môi trường ảo tên là venv2
python3 -m venv venv2
```

**Bước 2: Kích hoạt môi trường ảo**
```bash
# Trên macOS/Linux:
source venv2/bin/activate
# Trên Windows:
# venv2\Scripts\activate
```

**Bước 3: Cài đặt các thư viện cơ bản**
```bash
pip install -r requirements.txt
```

**Bước 4: Cài đặt thư viện ML DocAligner (Bắt buộc cho tuỳ chọn --docaligner)**
DocAligner cần được cài từ file wheel của tác giả thay vì qua pip thông thường:
```bash
pip install https://github.com/DocsaidLab/DocAligner/releases/download/v0.1.0/docaligner-0.1.0-py3-none-any.whl
```

### Các Lệnh Chạy Toàn Diện:

**1. Chạy với ảnh tự chọn bất kỳ:**
```bash
python3 "Pipeline With ML/main.py" "/đường/dẫn/đến/file/ảnh.jpg"
```

**2. Sử dụng siêu mô hình AI DocAligner (Độ chính xác cao nhất):**
AI tự động vẽ bản đồ nhiệt và tìm chính xác 4 đỉnh tài liệu dù nền rất hỗn loạn hay móp méo.
```bash
python3 "Pipeline With ML/main.py" <thư_mục_test> <số_thứ_tự> --docaligner

# Ví dụ test trên ảnh chụp nghiêng số 9:
python3 "Pipeline With ML/main.py" perspective 9 --docaligner
```

**3. Liệt kê các bộ ảnh có sẵn trong hệ thống:**
```bash
python3 "Pipeline With ML/main.py" list
```

**4. Kích hoạt ML Dewarping lật phẳng giấy cong (Chưa bao gồm model):**
```bash
python3 "Pipeline With ML/main.py" <thư_mục_test> <số_thứ_tự> --dewarp-ml
```

---

## 3. Dự án Cắt Gáy Sách & Phân đoạn Tài liệu (ML2 Phase 2)

Thư mục: `ml2/`

Đây là giải pháp hiện đại nhất của dự án, sử dụng mạng **U²-Net** và **YOLOv11-seg** được tuning (tái cấu trúc nhãn) để cắt chính xác trang tài liệu và loại bỏ gáy sách, đi kèm thuật toán hậu xử lý (Cutout & Gaussian Blur).

### 🚀 Lệnh Khởi chạy Nhanh (Build & Test)

**Bước 1: Setup Môi trường**
```bash
python3 -m venv venv_ml2
source venv_ml2/bin/activate
pip install -r ml2/requirements.txt
```

**Bước 2: Chạy Thử (Test) Tính năng Cắt Gáy Sách**
Mặc định sử dụng Cutout và Gaussian Smoothing để làm mờ viền cắt, đảm bảo độ mịn tuyệt đối.
```bash
python ml2/yolo_seg/test_crop.py \
    --weights exported_models/yolo11n_seg_spine_exclusion_best.pt \
    --source path/to/your/test_images \
    --cutout \
    --smooth-kernel 15
```

**Bước 3: Tải và xử lý dataset thật (Nếu muốn Train lại)**
```bash
python ml2/yolo_seg/split_and_process_dataset.py \
    --input_dir datasets/your_raw_dataset \
    --output_dir datasets/spine_dataset \
    --split_ratio 0.8
```

**Bước 4: Huấn luyện (Train/Tuning) YOLOv11-seg**
```bash
yolo task=segment mode=train data=datasets/spine_dataset/data.yaml model=yolo11n-seg.pt \
    epochs=150 batch=16 imgsz=640 device=mps \
    optimizer=AdamW lr0=0.001 lrf=0.01 warmup_epochs=3 \
    mosaic=1.0 mixup=0.1 box=7.5 cls=0.5 \
    project=runs/spine_seg name=spine_exclusion_tuned
```

---

## 4. Tài liệu Báo cáo môn học (Học sâu & Tích hợp)

Các tài liệu nghiên cứu chuyên sâu (phục vụ bảo vệ đồ án), spec kỹ thuật và báo cáo huấn luyện được lưu trữ trong thư mục [docs_ml2/](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/):

- [README.md - Master Index](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/README.md)
- [01_Problem_Statement.md - Phát biểu bài toán 2 giai đoạn & Chiến lược Data](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/01_Problem_Statement.md)
- [02_Research_Review.md - Phân tích kiến trúc U2Net vs YOLOv11](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/02_Research_Review.md)
- [03_Project_Plan.md - Kế hoạch triển khai & Quản lý rủi ro](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/03_Project_Plan.md)
- [04_Technical_Spec_and_KPI.md - Thông số kỹ thuật & Cấu hình Hàm Loss](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/04_Technical_Spec_and_KPI.md)
- [05_Training_Guide.md - Nhật ký Huấn luyện & Hyperparameter Tuning](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/05_Training_Guide.md)
- [06_Evaluation_Results.md - Đánh giá mIoU, viền răng cưa & Edge Cases](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/06_Evaluation_Results.md)
