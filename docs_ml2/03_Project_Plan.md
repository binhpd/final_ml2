# KẾ HOẠCH TRIỂN KHAI & QUẢN LÝ RỦI RO

## Timeline & Tiến độ Thực tế & Tiến độ Thực tế (M4 Max 48GB)

| Giai đoạn | Công việc Thực tế | Trạng thái / Kết quả đạt được |
|---|---|---|
| **Setup & Skeleton** | Thiết lập môi trường Python 3.12, venv, cài đặt các thư viện và tạo các file code hoàn chỉnh | **Đã hoàn thành 100%** (6 giờ) |
| **Chuẩn bị Data** | Tải và xử lý bộ dataset thực tế (SmartDoc2-Images và kaggle_real), chuẩn bị OOD dataset Doc3D | **Đã hoàn thành 100%** (4 giờ) |
| **Huấn luyện U²-Netp** | Huấn luyện mô hình U²-Netp lite (tối ưu xuống 80 epoch do hội tụ sớm) trên Apple Silicon MPS | **Đã hoàn thành 100%** (13 giờ 25 phút)<br>• Best Epoch: 60 (val IoU 0.9894)<br>• Output: `u2netp_doc_final.pth` |
| **Huấn luyện YOLOv11n-seg** | Huấn luyện mô hình YOLOv11n-seg (150 epoch, `--batch 32 --imgsz 640` trên MPS) | **Đã hoàn thành 100%**<br>• Hội tụ tốt với mAP50-95 xuất sắc<br>• Output: `yolo11n-seg_doc.pt` |
| **Đánh giá & Benchmark** | Chạy đánh giá (mIoU, Dice, MAE, Boundary F1, Speed) trên Test Set (N=2,550) và OOD Test Set (N=4,520) | **Đã hoàn thành 100%**<br>• Cả U²-Netp và YOLO đều vượt mọi KPI mục tiêu (xem Bảng vàng) |
| **Xuất báo cáo kiểm thử** | Xuất báo cáo đánh giá chất lượng độc lập của mô hình và lưu kết quả | **Đã hoàn thành 100%**<br>• Đã tổng hợp đầy đủ số liệu tại Bảng vàng và `yolo_eval.csv` |

---


## Phạm Vi Đồ Án


### 5.1 Tổng thể

- [x] **Plan đã chọn:** Plan B
- [x] **Hardware:** Mac Studio M4 Max 48GB
- [x] **Datasets:** SmartDoc2-Images (24,887 ảnh) + kaggle_real (620 ảnh) chuyên biệt; dùng Doc3D (90,372 ảnh) làm OOD test
- [x] **Loại bỏ tiền huấn luyện trên DUTS-TR** (data đích đã đủ 25K ảnh chất lượng)
- [x] **Loại bỏ sinh dữ liệu nhân tạo** (Sử dụng dữ liệu thực tế lớn từ SmartDoc và kaggle_real)
- [x] **Nhóm nghiên cứu quyết định huấn luyện 5-7 ngày trên M4 Max** (Đã tối ưu chạy MPS cực nhanh, chỉ mất 13.5h cho U²-Netp)
- [x] **Báo cáo bảo vệ được thực hiện độc lập** (Sử dụng công cụ hỗ trợ phân tích mã nguồn)

### 5.2 Phạm vi file code 

> **Trạng thái:** **Đã hoàn thành 100%** — Toàn bộ file code core đã được tạo lập, liên kết và chạy thử thành công trên Mac Studio M4 Max (đã lược bỏ phần tích hợp pipeline để tập trung hoàn toàn vào huấn luyện và đánh giá model).

| Module | File | Vai trò |
|--------|------|---------|
| Foundation (3) | requirements + .gitignore + README | Cài đặt + cấu trúc |
| U²-Net (11) | model + loss + dataset + aug + train + eval + infer + viz + 3 configs | Train U²-Netp lite |
| YOLO (7) | prepare + train + eval + viz + demo + tta + export | Fine-tune YOLOv11n |
| Benchmark (5) | 4 KPI scripts + aggregate | Đo KPI 4 chiều của các model |
| Scripts (6) | download + 2 prepare + dummy + check_env + caffeinate | Hỗ trợ |
| Notebooks (2) | u2net + yolo demo | Demo độc lập mô hình |
| **TỔNG** | **34 file** | Tập trung phát triển mô hình |

### 5.3 Tuỳ chọn kỹ thuật

- [x] Loại bỏ thư viện `pydensecrf` — khó build trên macOS ARM
- [x] Loại bỏ thư viện `coremltools` — chỉ tập trung xuất `.onnx` để sử dụng đa nền tảng
- [x] Bỏ hoàn toàn Tesseract OCR — lược bỏ pipeline, chỉ tập trung build & evaluate mô hình
- [x] Default batch_size = 16 (M4 Max 48GB đủ rộng)
- [x] Tắt AMP mặc định (MPS AMP còn buggy PyTorch 2.x)
- [x] U²-Netp lite 300 epoch, input 320 (Tối ưu xuống 80 epoch do hội tụ sớm)
- [x] YOLOv11n 150 epoch, imgsz 640 (Đã hoàn thành huấn luyện trên MPS, vượt mọi KPI)

### 5.4 Ablation tối thiểu cho báo cáo

- [x] U²-Net: BCE-only vs +IoU vs +SSIM (Đã kiểm chứng combo loss BCE + IoU + SSIM cho kết quả vượt trội)
- [x] YOLO: Fine-tune COCO vs from-scratch (Đã hoàn thành và so sánh)
- [x] Per-dataset eval: SmartDoc / Doc3D (Đã kiểm thử và phân tích chi tiết)

### 5.5 Lệnh chạy mẫu

```bash
# Setup
source venv_ml2/bin/activate
pip install -r ml2/requirements.txt
python ml2/scripts/check_environment.py

# Test code chạy được (dummy data)
python ml2/scripts/build_dummy_data.py --n 100
python ml2/u2net/train.py --config ml2/u2net/configs/mps_mini.yaml --dummy --epochs 1

# Tải datasets thật
python ml2/scripts/download_datasets.py --smartdoc --doc3d --subset

# Prepare labels
python ml2/scripts/prepare_smartdoc.py
python ml2/scripts/prepare_doc3d.py

# Train (chạy đêm với caffeinate)
caffeinate -i python ml2/u2net/train.py --config ml2/u2net/configs/doc_lite_planB.yaml
caffeinate -i python ml2/yolo_seg/train.py --epochs 150 --device mps

# Benchmark
python ml2/benchmark/aggregate_results.py
```

---

