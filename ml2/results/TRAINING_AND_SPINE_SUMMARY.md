# BÁO CÁO TỔNG HỢP HUẤN LUYỆN & SO SÁNH LOẠI BỎ GÁY SÁCH (SPINE EXCLUSION)

Tài liệu này tổng hợp toàn bộ thông số huấn luyện (Training Logs) của các mô hình tự huấn luyện/tinh chỉnh, đồng thời phân tích, so sánh chi tiết hiệu năng bóc tách gáy sách trên tập ảnh thực tế đã được gán nhãn ($N=17$).

---

## I. TỔNG HỢP THÔNG SỐ HUẤN LUYỆN (TRAINING LOGS)

Dưới đây là các đặc tính kỹ thuật và siêu tham số của 4 mô hình tự huấn luyện (Custom-Trained & Tuned) trong hệ thống:

| Đặc tính / Mô hình | **1. U²-Netp Lite (`u2netp_doc_final.pth`)** | **2. YOLOv11n-seg Doc (`yolo11n_seg_doc.pt`)** | **3. YOLOv11n-seg Tuning V1 (`yolo11n_seg_spine_exclusion_best.pt`)** | **4. YOLOv11n-seg Tuning V2 (`yolo_tuning_v2_full.pt`)** |
| :--- | :--- | :--- | :--- | :--- |
| **Nhiệm vụ chính** | Tự huấn luyện cắt lề giấy phẳng. | Tự huấn luyện nhận diện giấy phẳng. | Tinh chỉnh tách trang đôi, bỏ gáy sách. | Tinh chỉnh trộn phẳng-cong bóc gáy sách. |
| **Kiến trúc mạng** | U²-Netp (Lite variant) | YOLOv11n-seg (Nano) | YOLOv11n-seg (Nano) | YOLOv11n-seg (Nano) |
| **Số tham số (Params)** | 1.19 Million (~1.19M) | 2.83 Million (~2.83M) | 2.83 Million (~2.83M) | 2.83 Million (~2.83M) |
| **Kích thước file** | 4.8 MB (đã strip optimizer) | 6.0 MB (đã strip optimizer) | 6.0 MB (đã strip optimizer) | 6.0 MB (đã strip optimizer) |
| **Cấu hình Lớp** | 1 Class: `document` | 1 Class: `document` | 2 Class: `left_page`, `right_page` | 1 Class: `page` (Trộn chung phẳng & cong) |
| **Tập dữ liệu** | SmartDoc2 (24k8 ảnh) + Kaggle (620 ảnh) | SmartDoc + kaggle_real (Chỉ phẳng) | Spine Dataset (Sách cong có gáy ở giữa) | Blended Dataset (4,000 ảnh phẳng + ảnh gáy sách) |
| **Optimizer** | Adam (Base LR: 1e-3) | AdamW (Base LR: 0.01) | AdamW (Base LR: 1e-3) | AdamW (Base LR: 1e-3) |
| **Hàm Loss** | BCE + IoU + SSIM | Box (`7.5`) + Cls (`0.5`) + DFL (`1.5`) | Box (`7.5`) + Cls (`0.5`) + DFL (`1.5`) | Box (`7.5`) + Cls (`0.5`) + DFL (`1.5`) |
| **Tăng cường (Aug)** | Albumentations (Blur, Flare, Shadow) | Mosaic (`1.0`), Mixup (`0.0`) | Mosaic (`1.0`), Mixup (`0.15`) | Mosaic (`1.0`), Mixup (`0.15`) |
| **Phần cứng** | Mac Studio M4 Max (GPU MPS) | CPU / GPU MPS | Mac Studio M4 Max (GPU MPS) | Mac Studio M4 Max (GPU MPS) |
| **Tổng số Epochs** | 80 Epochs (Best: 60) | 115 Epochs (Best: 15) | 150 Epochs (Best: 136) | 60 Epochs (Best: 44) |
| **Thời gian train** | **13h 25m** | **~28 giờ** (100,979 giây) | **~3.5 giờ** | **~2.4 giờ** (8,577 giây) |
| **Kết quả Validation**| Val mIoU: **98.94%** | Box mAP50-95: **99.50%** | Mask mAP50-95: **96.76%** | Box/Mask mAP50-95: **99.50%** |

---

## II. SO SÁNH HIỆU NĂNG TRÊN TẬP LÀM SẠCH GÁY SÁCH (SPINE EXCLUSION - $N=17$)

Đây là kết quả đo lường thực tế trên **tập dữ liệu sách mở có gáy sách ở giữa** được gán nhãn thủ công để đánh giá năng lực loại bỏ gáy sách:

### 1. Bảng số liệu định lượng (Quantitative Metrics)

| Mô hình | Số Lớp | mIoU Trung bình ↑ | Dice / F1 ↑ | MAE ↓ | Latency (MPS) ↓ | Trạng thái Catastrophic Forgetting (Quên giấy phẳng) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`yolo_tuned` (Tuning V1)** | 2 | **0.9596** | **0.9791** | **0.0051** | **15.26 ms** | **Bị ảnh hưởng nghiêm trọng** (mIoU phẳng chỉ còn 33.53%). |
| **`yolo_tuning_v2_full` (Tuning V2)**| 1 | **0.9038** | **0.9221** | **0.0299** | **34.57 ms** | **Khắc phục hoàn toàn** (mIoU phẳng đạt **94.30%**). |
| **`yolo_trained` (Base Doc)** | 1 | **< 0.50** | - | - | **~9.2 ms** | Không bị (mIoU phẳng 94.45%). Không bóc được gáy. |
| **`u2net_trained` (U2Net Doc)** | 1 | **< 0.50** | - | - | **~16.1 ms** | Không bị (mIoU phẳng 97.32%). Không bóc được gáy. |

### 2. Phân tích hành vi chi tiết (Behavioral Analysis)

#### A. Nhóm bóc tách gáy thành công (`yolo_tuned` & `yolo_tuning_v2_full`)
* **Cơ chế:** Nhờ học được phân phối của sách mở trong quá trình tuning, cả hai mô hình tự động phát hiện phần mép giấy và chừa lại khoảng đen của gáy sách ở giữa (spine). 
* **Cắt trang:** 
  * `yolo_tuned` (V1) sử dụng 2 class dự đoán độc lập từ model để xuất ra 2 trang.
  * `yolo_tuning_v2_full` (V2) dự đoán các vùng trang chung nhãn `page`, sau đó dùng giải thuật sắp xếp tọa độ ngang của trọng tâm ($c_X$) để gán nhãn động thành `left_page` và `right_page`.
* **Đánh đổi:** Bản V2 Full có độ chính xác gáy sách thấp hơn V1 một chút (90.38% so với 95.96%) nhưng là mô hình **đa năng và thực tế nhất** vì nó không bị mất khả năng nhận diện trang giấy phẳng thông thường (Catastrophic Forgetting).

#### B. Nhóm không hỗ trợ bóc gáy (`yolo_trained` & `u2net_trained`)
* **Hành vi:** Nhận diện toàn bộ cuốn sách (gồm trang trái + gáy + trang phải) thành một khối duy nhất. 
* **Hậu quả:** Gáy sách bị giữ lại bên trong ảnh cutout/warp dưới dạng một vệt tối ngoằn ngoèo ở giữa. Điều này gây khó khăn cực lớn cho các thuật toán phẳng hóa chữ và làm giảm đáng kể độ chính xác của hệ thống nhận diện ký tự quang học (OCR).

---

## III. HƯỚNG DẪN CHẠY INFERENCE ĐỂ CẮT TRANG (CROP/CUTOUT)

Dưới đây là các lệnh chuẩn hóa để bạn chạy thử nghiệm cắt trang tài liệu, loại bỏ gáy trên toàn bộ thư mục ảnh chụp:

### 1. Chạy với mô hình YOLO (Khuyên dùng - `yolo_tuning_v2_full.pt`)
Script đã được tối ưu để tự động nhận diện thiết bị GPU Apple Silicon (`mps`) và chỉ xuất ra ảnh cắt chính nền trắng + ảnh vẽ khung tổng quan:
```bash
# Kích hoạt môi trường
source venv_ml2/bin/activate

# Chạy cắt ảnh hàng loạt
python ml2/yolo_seg/test_crop.py \
    --model exported_models/yolo_tuning_v2_full.pt \
    --image image_testing/spine \
    --out-dir ml2/results/test_crop
```

### 2. Chạy với mô hình U²-Netp (`u2netp_doc_final.pth`)
Sử dụng script mới được xây dựng để duyệt toàn bộ thư mục và thực hiện cutout/warp tương tự YOLO:
```bash
python test_crop_u2net.py \
    --ckpt exported_models/u2netp_doc_final.pth \
    --image image_testing/spine \
    --out-dir ml2/results/u2net_crop \
    --device mps
```
