# Kế hoạch Triển khai Chi tiết — Train U2NET + YOLO-Seg cho Document Segmentation

> **Nguồn gốc:** Tổng hợp từ 9 file trong [claude_plan_docs/](claude_plan_docs/) (~4145 dòng).
> **Mục đích file này:** Bản kế hoạch đã được hợp nhất, có timeline thực tế, spec từng file code cần viết, lệnh chạy mẫu, và rủi ro cụ thể.
> **Trạng thái:** ✅ **Plan B đã chốt** (xem [PlanB_Datasets_Final.md](PlanB_Datasets_Final.md))

---

## 0. Quyết định đã chốt (CẬP NHẬT)

| Mục | Quyết định | Ghi chú |
|------|------------|---------|
| Phạm vi | **Plan B**: U2NETp lite + YOLOv11n | ~42 file Python + notebook (~4500 dòng) |
| Hardware train | **Mac Studio M4 Max 48GB** | MPS với 40-core GPU, ~28 TFLOPS |
| Datasets | **SmartDoc + MIDV-500 + Doc3D** (online) | 12,000 ảnh, ~20GB download |
| Stage training | **1 stage** trên doc data | Bỏ DUTS-TR pretrain vì data đích đủ lớn |
| Code style | `.py` sạch + `.ipynb` demo | Type hints, docstring tiếng Việt |
| Train thật | **User tự chạy** trên M4 Max sau khi build xong | 5-7 ngày xuyên đêm |

---

## 1. Phân tích thực tế (Reality check)

### 1.1 Cái kế hoạch gốc đề xuất

| Phase | Tuần | Hoạt động |
|-------|------|-----------|
| 0 (Setup) | 1 | Tải dataset, auto-label 1020 ảnh |
| 1 (U2NET) | 2-6 | Build + train 2 variant U2NET, 2 stage, ablation |
| 2 (YOLO) | 7-10 | Train 3 size YOLO, viz, export |
| 3 (Integration) | 11 | Wrappers + pipelines |
| 4 (Benchmark) | 12 | KPI 4 chiều |

→ **Tổng: 12 tuần với 1× CUDA GPU.**

### 1.2 Thực tế với M4 Max 48GB (đã cập nhật)

| Việc | RTX 3090 | **M4 Max 48GB** | M1/M2 Pro |
|------|----------|-----------------|-----------|
| U2NET full Stage 1 (DUTS-TR, 600 ep) | ~3 ngày | ~5-7 ngày | Không khả thi |
| U2NETp lite **1 stage doc data (10.5K ảnh, 300 ep)** | ~24h | **~3-4 ngày** ⭐ | ~7+ ngày |
| YOLOv11n-seg (150 ep, imgsz 640) | ~4h | **~12-16h** ⭐ | ~1-2 ngày |
| YOLOv11s-seg (200 ep, imgsz 1024) | ~12h | ~36h | Không khuyến nghị |

**M4 Max ưu thế:**
- 48GB unified memory → batch_size 16 không vấn đề (3090 chỉ 24GB)
- Train xuyên đêm im lặng, ít điện (~140W)
- CoreML export native cho mobile demo

### 1.3 Plan B đã chốt cho M4 Max

| Quyết định | Lý do |
|------------|-------|
| ✅ Train **U2NETp lite** (1.1M params), 300 epoch, 1 stage trên 10,500 ảnh | Bỏ DUTS-TR vì có 12K ảnh doc đã đủ |
| ✅ Train **YOLOv11n-seg** fine-tune COCO, 150 epoch, imgsz 640 | Khả thi 12-16h MPS |
| ❌ U2NET full, YOLO s/m → bỏ qua | Plan A cần CUDA |
| ❌ DUTS-TR pretrain → bỏ qua | Data đích đã có 12K ảnh doc chuyên biệt |
| ❌ Synthetic data generator → bỏ qua | Doc3D đã có 100K synthetic sẵn |

→ Code skeleton vẫn hỗ trợ U2NET full + YOLO s/m qua YAML configs (cho ai có CUDA).

---

## 2. Danh sách File sẽ tạo

### 2.1 Phase 1 — U2NET (12 files)

| # | File | Dòng dự kiến | Vai trò |
|---|------|--------------|---------|
| 1 | `ml2/u2net/__init__.py` | 5 | Module init |
| 2 | `ml2/u2net/model.py` | 350 | RSU-7, RSU-6, RSU-5, RSU-4, RSU-4F + U2NET + U2NETp |
| 3 | `ml2/u2net/loss.py` | 120 | Combo BCE + IoU + SSIM + EdgeLoss + multi-supervision |
| 4 | `ml2/u2net/dataset.py` | 180 | DocSegDataset đa nguồn (DUTS, SmartDoc, Nhom6, Synthetic) |
| 5 | `ml2/u2net/augmentation.py` | 100 | Albumentations pipeline (basic + strong) |
| 6 | `ml2/u2net/train.py` | 280 | Training loop + AMP + TensorBoard + checkpoint |
| 7 | `ml2/u2net/eval.py` | 200 | mIoU, F1, MAE, Boundary-F1 + per-category |
| 8 | `ml2/u2net/infer.py` | 150 | Inference + TTA + CRF (optional) |
| 9 | `ml2/u2net/visualize.py` | 150 | Training curves, sample predictions |
| 10 | `ml2/u2net/configs/stage1_duts_full.yaml` | 50 | Config stage 1 full |
| 11 | `ml2/u2net/configs/stage1_duts_lite.yaml` | 50 | Config stage 1 lite |
| 12 | `ml2/u2net/configs/stage2_doc_full.yaml` | 50 | Config stage 2 full |
| 13 | `ml2/u2net/configs/stage2_doc_lite.yaml` | 50 | Config stage 2 lite |
| 14 | `ml2/u2net/configs/mps_mini.yaml` | 40 | **Plan B: cấu hình MPS khả thi** |

**Tổng Phase 1: ~1770 dòng**

### 2.2 Phase 2 — YOLO-Seg (8 files)

| # | File | Dòng dự kiến | Vai trò |
|---|------|--------------|---------|
| 15 | `ml2/yolo_seg/__init__.py` | 5 | Module init |
| 16 | `ml2/yolo_seg/prepare_dataset.py` | 200 | Convert mask → YOLO polygon + split train/val/test |
| 17 | `ml2/yolo_seg/train.py` | 180 | Wrapper Ultralytics + 3-phase schedule + MPS |
| 18 | `ml2/yolo_seg/sweep.py` | 100 | Hyperparameter sweep |
| 19 | `ml2/yolo_seg/eval.py` | 180 | mAP + custom mIoU + speed benchmark |
| 20 | `ml2/yolo_seg/visualize.py` | 250 | YOLODocVisualizer (bbox + mask + corners + info) |
| 21 | `ml2/yolo_seg/demo_viz.py` | 80 | Batch demo + grid montage |
| 22 | `ml2/yolo_seg/infer_tta.py` | 60 | TTA inference |
| 23 | `ml2/yolo_seg/export_all.py` | 80 | Multi-format export (ONNX/CoreML/OpenVINO/TFLite) |

**Tổng Phase 2: ~1135 dòng**

### 2.3 Phase 3 — Integration (5 files)

| # | File | Dòng dự kiến | Vai trò |
|---|------|--------------|---------|
| 24 | `ml2/pipeline_integration/__init__.py` | 5 | Module init |
| 25 | `ml2/pipeline_integration/u2net_wrapper.py` | 120 | Drop-in replacement cho rembg |
| 26 | `ml2/pipeline_integration/yolo_wrapper.py` | 150 | Wrapper YOLO + viz + corner extraction |
| 27 | `ml2/pipeline_integration/pipeline_u2net.py` | 200 | Pipeline copy + patch dùng U2NET |
| 28 | `ml2/pipeline_integration/pipeline_yolo.py` | 250 | Pipeline mới với YOLO + viz đầy đủ |
| 29 | `ml2/pipeline_integration/test_integration.py` | 80 | Test trên 1020 ảnh, đo tỷ lệ thành công |

**Tổng Phase 3: ~805 dòng**

### 2.4 Phase 4 — Benchmark (5 files)

| # | File | Dòng dự kiến | Vai trò |
|---|------|--------------|---------|
| 30 | `ml2/benchmark/__init__.py` | 5 | Module init |
| 31 | `ml2/benchmark/kpi_speed.py` | 180 | CPU + MPS + CoreML benchmark |
| 32 | `ml2/benchmark/kpi_accuracy.py` | 200 | 4 metrics + so sánh song song |
| 33 | `ml2/benchmark/kpi_robustness.py` | 150 | Per-category 7 nhóm |
| 34 | `ml2/benchmark/kpi_e2e.py` | 180 | PSNR + SSIM + OCR-CER + total time |
| 35 | `ml2/benchmark/aggregate_results.py` | 100 | Gộp tất cả + xuất CSV + bảng final |

**Tổng Phase 4: ~815 dòng**

### 2.5 Scripts hỗ trợ (6 files)

| # | File | Dòng dự kiến | Vai trò |
|---|------|--------------|---------|
| 36 | `ml2/scripts/download_datasets.py` | 200 | Tải DUTS-TR/TE, SmartDoc, MIDV-500, MIT Indoor BG |
| 37 | `ml2/scripts/autolabel_nhom6.py` | 150 | Ensemble rembg (3 model) auto-label 1020 ảnh |
| 38 | `ml2/scripts/verify_masks.py` | 200 | UI helper để manual verify mask |
| 39 | `ml2/scripts/synthesize_documents.py` | 250 | Sinh synthetic data |
| 40 | `ml2/scripts/build_dummy_data.py` | 150 | **Sinh dummy dataset để test code chạy ngay** |
| 41 | `ml2/scripts/check_environment.py` | 100 | Verify torch/MPS/CUDA + dependency |

**Tổng Scripts: ~1050 dòng**

### 2.6 Notebooks demo (4 files)

| # | File | Cells | Vai trò |
|---|------|-------|---------|
| 42 | `ml2/notebooks/01_u2net_demo.ipynb` | ~20 | Load model, forward, train 1 epoch dummy, viz |
| 43 | `ml2/notebooks/02_yolo_demo.ipynb` | ~15 | Load YOLO, predict, visualize, export |
| 44 | `ml2/notebooks/03_integration_demo.ipynb` | ~15 | Run cả 2 pipeline trên 1 ảnh thật |
| 45 | `ml2/notebooks/04_benchmark_demo.ipynb` | ~20 | Chạy mini-benchmark + biểu đồ |

### 2.7 File chung

| # | File | Vai trò |
|---|------|---------|
| 46 | `ml2/requirements.txt` | ✅ Đã tạo |
| 47 | `ml2/README.md` | Hướng dẫn cài đặt + chạy + cấu trúc |
| 48 | `ml2/.gitignore` | Bỏ qua weights/, runs/, datasets/raw/ |

---

## 3. Tổng kết Khối lượng

| Loại | Số file | Dòng code |
|------|---------|-----------|
| Python module | 35 | ~5575 |
| YAML config | 5 | ~240 |
| Jupyter notebook | 4 | ~70 cells |
| Markdown / config | 3 | ~150 |
| **TỔNG** | **47** | **~6035** |

**Ước tính thời gian build (session này):** ~6-8 giờ liên tục viết code có chất lượng. Có thể chia 2-3 session.

---

## 4. Lệnh chạy mẫu (sau khi build xong)

### 4.1 Setup môi trường

```bash
cd "/Users/binhpham/Documents/Study/MSE/ML 2/Final_deeplearning"
python3 -m venv venv_ml2
source venv_ml2/bin/activate
pip install -r ml2/requirements.txt

# Verify
python ml2/scripts/check_environment.py
```

### 4.2 Tạo dummy data + chạy thử

```bash
# Sinh 100 ảnh dummy (paper trên random bg) để test code
python ml2/scripts/build_dummy_data.py --n 100 --out ml2/datasets/dummy

# Test U2NET forward + 1 epoch trên dummy
python ml2/u2net/train.py --config ml2/u2net/configs/mps_mini.yaml --dummy

# Test YOLO trên dummy
python ml2/yolo_seg/prepare_dataset.py --src ml2/datasets/dummy --dst ml2/datasets/yolo_format
python ml2/yolo_seg/train.py --size n --epochs 3 --dummy
```

### 4.3 Tải dataset thật (khi đã sẵn sàng)

```bash
python ml2/scripts/download_datasets.py --duts --smartdoc
# (Yêu cầu mạng tốt, ~2.5GB)
```

### 4.4 Auto-label 1020 ảnh nhóm

```bash
# Trỏ ảnh nhóm vào trước (placeholder)
# cp -r /path/to/1020/photos ml2/datasets/nhom6_1020/images

python ml2/scripts/autolabel_nhom6.py --ensemble 3
```

### 4.5 Train chính (Plan B — khả thi trên MPS)

```bash
# Stage 1 lite trên DUTS-TR subset
python ml2/u2net/train.py --config ml2/u2net/configs/stage1_duts_lite.yaml --device mps

# Stage 2 doc data
python ml2/u2net/train.py --config ml2/u2net/configs/stage2_doc_lite.yaml --device mps

# YOLO fine-tune nano
python ml2/yolo_seg/train.py --size n --imgsz 640 --epochs 100 --device mps
```

### 4.6 Benchmark

```bash
python ml2/benchmark/kpi_speed.py
python ml2/benchmark/kpi_accuracy.py
python ml2/benchmark/kpi_robustness.py
python ml2/benchmark/kpi_e2e.py
python ml2/benchmark/aggregate_results.py  # gộp + xuất CSV
```

---

## 5. Adaptation cho MPS (Apple Silicon)

Trong code skeleton, tôi sẽ thêm các điều chỉnh cụ thể cho MPS:

| Điều chỉnh | Lý do |
|------------|-------|
| `device = "mps" if torch.backends.mps.is_available() else "cpu"` | Auto-detect |
| `torch.mps.synchronize()` trước khi đo thời gian | MPS async, đo sai nếu thiếu |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` trong env | Một số op chưa hỗ trợ MPS → fallback CPU |
| Batch size mặc định = 8 (thay vì 32) | MPS memory không lớn |
| `pin_memory=False` | MPS không hỗ trợ pinned memory |
| **AMP tắt mặc định** | MPS AMP còn buggy trong PyTorch 2.x |
| Config `mps_mini.yaml` với subset DUTS-TR 3000 ảnh, 200 epoch | Khả thi 1-2 ngày |
| YOLO `device='mps'` thay vì `device=0` | Ultralytics đã hỗ trợ MPS từ v8.x |

---

## 6. Rủi ro thực tế + Giảm thiểu

| Rủi ro | Xác suất | Tác động | Giảm thiểu |
|--------|----------|----------|------------|
| MPS chậm hơn dự kiến (5-7 ngày/run) | Cao | Cao | Plan B: chỉ U2NETp lite trên subset 3K ảnh; train từng đêm; checkpoint mỗi epoch |
| Một số op không hỗ trợ MPS (Sobel, dilated conv) | Trung bình | Trung bình | `PYTORCH_ENABLE_MPS_FALLBACK=1`; nếu fail vẫn dùng `device=cpu` mạnh hơn rembg |
| `pydensecrf` không build được trên macOS ARM | Cao | Thấp | Comment trong requirements; CRF là optional |
| `coremltools` xung đột phiên bản torch | Trung bình | Thấp | Tách env riêng `venv_export` cho export |
| Tải DUTS-TR 2GB chậm/fail | Trung bình | Cao | Script có resume + mirror Google Drive |
| 1020 ảnh nhóm chưa có ở vị trí nào trong dự án | Cao | Cao | Tôi tạo `scripts/build_dummy_data.py` để code chạy được không cần ảnh thật |
| Auto-label noise cao → train kém | Trung bình | Trung bình | Ensemble 3 rembg + verify subset 200 ảnh |
| Không đủ ảnh verified ground truth → metric không có ý nghĩa | Cao | Cao | UI verify helper + cho phép user mark "skip" |
| Báo cáo bảo vệ cần hiểu code → user không đủ kiến thức | Trung bình | Cao | Code có docstring tiếng Việt giải thích; notebook demo từng bước; tôi có thể giải thích từng module qua chat |

---

## 7. Định nghĩa "Hoàn thành" (Definition of Done)

**Sau session build (do tôi làm):**
- ✅ Tất cả 47 file tồn tại với code chạy được trên dummy data
- ✅ `python ml2/scripts/check_environment.py` pass
- ✅ `python ml2/scripts/build_dummy_data.py` sinh được 100 ảnh dummy
- ✅ `python ml2/u2net/train.py --config mps_mini.yaml --dummy --epochs 1` chạy 1 epoch không lỗi
- ✅ `python ml2/yolo_seg/train.py --size n --epochs 1 --dummy` chạy 1 epoch không lỗi
- ✅ 4 notebook chạy được không lỗi với dummy data
- ✅ README đầy đủ hướng dẫn

**Sau khi bạn tự chạy training (không phải tôi làm):**
- 📌 1 checkpoint U2NETp lite trên dataset thật
- 📌 1 checkpoint YOLOv11n-seg fine-tuned
- 📌 Bảng KPI thật trên ≥ 100 ảnh test
- 📌 Báo cáo `report_final.md`

---

## 8. Thứ tự build (đã sắp xếp dependency)

```
Bước 1: Foundation
   ├─ requirements.txt (✅ đã có)
   ├─ README.md
   └─ scripts/check_environment.py + build_dummy_data.py

Bước 2: U2NET core (parallel-able)
   ├─ u2net/model.py
   ├─ u2net/loss.py
   ├─ u2net/augmentation.py
   └─ u2net/dataset.py

Bước 3: U2NET training
   ├─ u2net/configs/*.yaml
   ├─ u2net/train.py (cần dataset + model + loss)
   ├─ u2net/eval.py
   ├─ u2net/infer.py
   └─ u2net/visualize.py

Bước 4: YOLO module
   ├─ yolo_seg/prepare_dataset.py
   ├─ yolo_seg/visualize.py (độc lập)
   ├─ yolo_seg/train.py
   ├─ yolo_seg/eval.py
   ├─ yolo_seg/demo_viz.py
   ├─ yolo_seg/infer_tta.py
   └─ yolo_seg/export_all.py

Bước 5: Integration
   ├─ pipeline_integration/u2net_wrapper.py (cần u2net/model)
   ├─ pipeline_integration/yolo_wrapper.py (cần yolo_seg/visualize)
   ├─ pipeline_integration/pipeline_u2net.py
   ├─ pipeline_integration/pipeline_yolo.py
   └─ pipeline_integration/test_integration.py

Bước 6: Benchmark
   ├─ benchmark/kpi_speed.py
   ├─ benchmark/kpi_accuracy.py
   ├─ benchmark/kpi_robustness.py
   ├─ benchmark/kpi_e2e.py
   └─ benchmark/aggregate_results.py

Bước 7: Scripts còn lại
   ├─ scripts/download_datasets.py
   ├─ scripts/autolabel_nhom6.py
   ├─ scripts/verify_masks.py
   └─ scripts/synthesize_documents.py

Bước 8: Notebooks
   ├─ notebooks/01_u2net_demo.ipynb
   ├─ notebooks/02_yolo_demo.ipynb
   ├─ notebooks/03_integration_demo.ipynb
   └─ notebooks/04_benchmark_demo.ipynb
```

---

## 9. Tài liệu liên quan (đã có trong dự án)

- [Master Index](claude_plan_docs/00_TongHop_Claude.md)
- [Kế hoạch tổng quan](claude_plan_docs/KeHoach_TongQuan_Claude.md)
- [Chi tiết U2NET](claude_plan_docs/01_U2NET_ChiTiet_Claude.md)
- [Chi tiết YOLO-Seg](claude_plan_docs/02_YOLO_ChiTiet_Claude.md)
- [Integration + KPI](claude_plan_docs/03_Integration_KPI_Claude.md)
- [Checklist Review](Checklist_Review.md) — file song hành với file này
- [Tóm tắt 1 trang](TomTat_1Trang.md) — bản ngắn để nhìn nhanh

---

## 10. Điểm cần bạn duyệt trước khi tôi code

**✅ Đã chốt:**
1. ~~Phạm vi: Plan A/B/C~~ → **Plan B**
2. ~~1020 ảnh nhóm 6?~~ → **Không có, dùng online datasets**
3. ~~Synthetic data generator?~~ → **Bỏ qua** (Doc3D đã có 100K synthetic)
4. **CRF**: bỏ qua (khó build trên macOS ARM)
5. **CoreML export**: giữ lại (M4 Max có Neural Engine, demo mobile được)

**📋 Còn chờ trả lời (6 câu trong Checklist mục G):**
- G3: Deadline đồ án?
- G4: Báo cáo tiếng Việt hay Anh?
- G5: Mục tiêu điểm? (cân chiều sâu vs scope)
- G6: M4 Max chạy được xuyên đêm 5-7 ngày không?
- G7: Có muốn auto pause/resume train script?
- G8: Có muốn build CoreML export demo iOS/macOS?

→ Tick checklist + trả lời 6 câu là tôi bắt đầu build 42 file Plan B.
