# KẾ HOẠCH TRIỂN KHAI & QUẢN LÝ RỦI RO

## 1. Timeline & Tiến độ Thực tế (M4 Max 48GB)

| Giai đoạn | Công việc Thực tế | Trạng thái / Kết quả đạt được |
|---|---|---|
| **Setup & Skeleton** | Thiết lập môi trường Python 3.12, venv, cài đặt các thư viện và tạo các file code hoàn chỉnh | **Đã hoàn thành 100%** (6 giờ) |
| **Chuẩn bị Data** | Tải và xử lý bộ dataset thực tế (SmartDoc2-Images và kaggle_real), chuẩn bị OOD dataset Doc3D | **Đã hoàn thành 100%** (4 giờ) |
| **Huấn luyện Baseline** | Huấn luyện mô hình U²-Netp lite (80 epoch) và YOLOv11n-seg (150 epoch) trên Apple Silicon MPS | **Đã hoàn thành 100%** (13 giờ 25 phút) |
| **Giai đoạn Nâng cao: Tuning Spine Exclusion** | Chạy script `split_and_process_dataset.py` để gán nhãn `left_page` / `right_page`. Tuning lại YOLOv11. | **Đã hoàn thành 100%** |
| **Hậu xử lý (Post-processing)** | Xây dựng pipeline Cutout (pixel-level mask) và Gaussian Blur Smoothing trong `test_crop.py`. | **Đã hoàn thành 100%** |
| **Đánh giá & Benchmark** | Chạy đánh giá chất lượng (mIoU, F1) và kiểm tra bằng mắt (mượt mép, loại bỏ gáy sách thành công). | **Đã hoàn thành 100%** |
| **Xuất báo cáo kiểm thử** | Xuất báo cáo đánh giá chất lượng độc lập của mô hình và lưu kết quả. Cập nhật hệ thống `docs_ml2`. | **Đã hoàn thành 100%** |

---

## 2. Phạm Vi Đồ Án (Sau Nâng Cấp)

### 2.1 Cấu hình phần cứng & Dữ liệu
- **Hardware:** Mac Studio M4 Max 48GB
- **Datasets gốc:** SmartDoc2-Images (24,887 ảnh) + kaggle_real (620 ảnh) chuyên biệt; dùng Doc3D (90,372 ảnh) làm OOD test
- **Xử lý nhãn (Tuning Phase):** Tự động phân tách nhãn thành 2 class (`left_page`, `right_page`) thông qua tọa độ bounding box để dạy mô hình phân biệt gáy sách.

### 2.2 Phạm vi file code bổ sung (Phục vụ Tuning)
Ngoài 34 file code cơ bản phục vụ huấn luyện, Giai đoạn nâng cao đã bổ sung và hoàn thiện các file quan trọng sau:
1. `ml2/yolo_seg/split_and_process_dataset.py`: Xử lý phân loại nhãn trái/phải dựa trên toạ độ tâm gáy sách.
2. `ml2/yolo_seg/test_crop.py`: Trái tim của quá trình Inference. Hỗ trợ cắt sát mép (cutout), làm mịn đường viền (smooth contour), và loại bỏ background triệt để.

### 2.3 Tuỳ chọn kỹ thuật đã chốt
- [x] Sử dụng mô hình YOLOv11-seg để hỗ trợ multi-class.
- [x] Mặc định chế độ **Cutout** là Output chuẩn (Không xuất bounding box chữ nhật với background).
- [x] Mặc định tham số **Gaussian Blur (`--smooth-kernel 15`)** để khử nhiễu viền do giới hạn độ phân giải của YOLO.
- [x] Các ảnh output phải được lưu vào thư mục phân loại rõ ràng (ví dụ: `ml2/yolo_seg/test_output_spine`).

---

## 3. Các Rủi Ro Kỹ Thuật Đã Giải Quyết

| Rủi Ro Nhận Diện | Giải Pháp Thực Thi | Kết Quả |
|---|---|---|
| Mô hình nhận gộp 2 trang làm 1, không cắt được gáy. | Chuyển bài toán từ Single-class sang Multi-class (Trái/Phải). | YOLOv11 đã có thể xuất ra 2 mask độc lập cho 2 trang. |
| Đường viền cắt (Contour) bị răng cưa, gai góc. | Áp dụng Gaussian Blur lên mask đầu ra, sau đó tái nhị phân hóa (Thresholding). | Đường cắt mịn, mượt, cong tự nhiên bám sát mép sách. |
| Quá tải dung lượng lưu trữ trên Mac Studio. | Xoá bỏ các bản checkpoints trung gian (`last.pt`), chỉ gom file `best.pt` vào `exported_models/`. | Tiết kiệm ổ cứng, mô hình gọn gàng, dễ quản lý. |

---

*Toàn bộ kế hoạch và các rủi ro đã được kiểm soát. Dự án sẵn sàng cho quá trình demo và báo cáo.*
