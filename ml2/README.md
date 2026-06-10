# ml2/ — Phân đoạn Tài liệu & Loại bỏ Gáy Sách (U²-Net + YOLO-Seg)

> Đồ án ML2 cuối kỳ — Nhóm 6 | Mac Studio M4 Max 48GB

## 🎯 Chủ đề Đồ Án
Xây dựng mô hình AI để phân đoạn và cắt trang tài liệu từ ảnh chụp điện thoại. Dự án được chia làm 2 giai đoạn:
- **Giai đoạn 1 (Cơ bản):** Xây dựng và so sánh hiệu năng giữa hai họ mạng U²-Net (tách nền pixel-level) và YOLOv11-seg (instance segmentation) trong bài toán cắt trang giấy đơn cơ bản. Đánh giá về cách train, chất lượng đầu ra và các KPI.
- **Giai đoạn 2 (Nâng cao):** Giải quyết vấn đề thực tế khi chụp tài liệu thường dính gáy sách. Mô hình YOLO được huấn luyện (tuning) nâng cao với dữ liệu gán nhãn chi tiết để phân biệt trang trái/phải, từ đó tự động loại bỏ phần gáy sách dư thừa và cắt chính xác nội dung trang.

## 🗂️ Cấu trúc

```
ml2/
├── u2net/                  # U²-Netp lite (Giai đoạn 1)
├── yolo_seg/               # YOLOv11n-seg (Giai đoạn 1 & 2 - Spine Exclusion)
├── pipeline_integration/   # Drop-in wrappers
├── benchmark/              # Đánh giá KPI
├── scripts/                # Hỗ trợ (Tải & Chuẩn bị data)
├── notebooks/              # Demos
├── checkpoints/            # Chứa các model đang train
├── exported_models/        # Các model đã train & tuning (đã export)
└── requirements.txt
```

## 🚀 Quick Start (Lệnh Build & Test)

```bash
# 1. Setup & Build Environment
python -m venv venv_ml2
source venv_ml2/bin/activate
pip install -r ml2/requirements.txt

# 2. Verify Environment
python ml2/scripts/check_environment.py

# 3. Chạy Thử (Test) Tính năng Cắt Gáy Sách (Spine Exclusion)
# Mặc định sử dụng Cutout và Gaussian Smoothing để làm mờ viền cắt
python ml2/yolo_seg/test_crop.py \
    --weights exported_models/yolo_tuning_v2_full.pt \
    --source path/to/your/test_images \
    --cutout \
    --smooth-kernel 15

# 4. Chuẩn bị dữ liệu (Data Blending 1-Class cho Tuning V2)
python ml2/scripts/prepare_tuning_v2_fast.py

# 5. Huấn luyện (Train/Tuning) YOLOv11-seg V2
python ml2/yolo_seg/train_tuning_v2.py
```

## 📦 Exported Models
Các mô hình đã được train và tuning hoàn chỉnh được lưu tại thư mục `exported_models/` ở thư mục gốc:
- `u2netp_doc_final.pth`: Mô hình U²-Net lite (Giai đoạn 1).
- `yolo11n_seg_doc.pt`: Mô hình YOLO-Seg cơ bản (Giai đoạn 1).
- `yolo_tuning_v2_full.pt`: Mô hình YOLO-Seg V2 đã tuning (Data Blending 1-Class) trên toàn bộ dữ liệu (Full Dataset) để loại bỏ gáy sách và chống quên dữ liệu phẳng (Giai đoạn 2).

## 📊 Tham khảo Tài liệu (docs_ml2)
Chi tiết về toàn bộ quá trình nghiên cứu, chuẩn bị dữ liệu, đào tạo và đánh giá KPI, vui lòng xem tại thư mục `docs_ml2/`:
- `docs_ml2/01_Problem_Statement.md`: Phát biểu bài toán 2 giai đoạn.
- `docs_ml2/02_Research_Review.md`: Phân tích U2Net vs YOLO.
- `docs_ml2/03_Project_Plan.md`: Kế hoạch và xử lý rủi ro.
- `docs_ml2/04_Technical_Spec_and_KPI.md`: Thông số kỹ thuật YOLO tuning.
- `docs_ml2/05_Training_Guide.md`: Hướng dẫn chi tiết train.
- `docs_ml2/06_Evaluation_Results.md`: Đánh giá mIoU, viền răng cưa.
