# Dự án Phân đoạn Tài liệu & Cắt Gáy Sách (Document Segmentation & Spine Exclusion)

> Đồ án ML2 cuối kỳ — Nhóm 6 | Mac Studio M4 Max 48GB

Dự án này là giải pháp hiện đại sử dụng Mạng Nơ-ron Tích chập (CNN) kết hợp các thuật toán hậu xử lý để tách nền tài liệu, phân biệt trang trái/phải, và loại bỏ phần gáy sách.

## 🚀 Hướng dẫn Chạy Thử nghiệm (Build & Test)

Dưới đây là các lệnh cần thiết để thiết lập môi trường và chạy test mô hình cắt trang giấy ngay lập tức.

### Bước 1: Khởi tạo Môi trường (Environment Setup)
Yêu cầu Python 3.10+
```bash
# 1. Tạo môi trường ảo
python3 -m venv venv_ml2

# 2. Kích hoạt môi trường
source venv_ml2/bin/activate  # macOS/Linux
# venv_ml2\Scripts\activate   # Windows

# 3. Cài đặt thư viện
pip install -r ml2/requirements.txt
```

### Bước 2: Chạy Test Phân đoạn & Cắt Trang (Inference)
Sử dụng script `test_crop.py` kết hợp với model đã train tối ưu nhất (nằm trong thư mục `exported_models`).

Lệnh cắt ảnh tiêu chuẩn (sử dụng Cutout & Gaussian Smoothing để viền siêu mượt):
```bash
python ml2/yolo_seg/test_crop.py \
    --weights exported_models/yolo11n_seg_spine_exclusion_best.pt \
    --source path/to/your/test_images \
    --cutout \
    --smooth-kernel 15
```
*Ghi chú:*
- Thay `path/to/your/test_images` bằng đường dẫn tới file ảnh cụ thể hoặc thư mục chứa ảnh cần test.
- Hệ thống sẽ tự động tách rời 2 trang giấy, loại bỏ gáy đen ở giữa, và xuất ra ảnh với nền trong suốt/trắng tinh.

### Bước 3: Huấn luyện lại (Tuỳ chọn)
Nếu bạn muốn tự train lại model từ đầu trên dữ liệu của riêng mình:

```bash
# 1. Chuẩn bị dữ liệu (Tách nhãn trái/phải)
python ml2/yolo_seg/split_and_process_dataset.py \
    --input_dir datasets/your_raw_dataset \
    --output_dir datasets/spine_dataset \
    --split_ratio 0.8

# 2. Train model bằng YOLOv11-seg
yolo task=segment mode=train data=datasets/spine_dataset/data.yaml model=yolo11n-seg.pt \
    epochs=150 batch=16 imgsz=640 device=mps \
    optimizer=AdamW lr0=0.001 lrf=0.01 warmup_epochs=3 \
    mosaic=1.0 mixup=0.1 box=7.5 cls=0.5 \
    project=runs/spine_seg name=spine_exclusion_tuned
```

---

## 📚 Tài liệu Báo cáo & Nghiên cứu (docs_ml2)

Toàn bộ các tài liệu khoa học phục vụ cho việc bảo vệ đồ án (bao gồm cấu trúc mạng, thông số hàm Loss, biểu đồ huấn luyện) được lưu trong thư mục `docs_ml2/`:

- [README.md - Master Index](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/README.md)
- [01_Problem_Statement.md - Phát biểu bài toán 2 giai đoạn & Chiến lược Data](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/01_Problem_Statement.md)
- [02_Research_Review.md - Phân tích kiến trúc U2Net vs YOLOv11](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/02_Research_Review.md)
- [03_Project_Plan.md - Kế hoạch triển khai & Quản lý rủi ro](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/03_Project_Plan.md)
- [04_Technical_Spec_and_KPI.md - Thông số kỹ thuật & Cấu hình Hàm Loss](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/04_Technical_Spec_and_KPI.md)
- [05_Training_Guide.md - Nhật ký Huấn luyện & Hyperparameter Tuning](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/05_Training_Guide.md)
- [06_Evaluation_Results.md - Đánh giá mIoU, viền răng cưa & Edge Cases](file:///Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/docs_ml2/06_Evaluation_Results.md)
