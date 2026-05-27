# Tóm tắt 1 Trang — Đồ án U2NET + YOLO-Seg (Plan B đã chốt)

## ✅ Quyết định đã chốt

| Mục | Quyết định |
|------|------------|
| **Plan** | **Plan B** (U2NETp lite + YOLOv11n) |
| **Hardware** | Mac Studio M4 Max **48GB** unified memory |
| **Datasets** | **SmartDoc ICDAR2015 + MIDV-500 + Doc3D** (online) |
| **Bỏ DUTS-TR** | Vì đã có 12K ảnh document chuyên biệt |
| **Stage** | **1 stage** trên doc data (thay vì 2 stage) |

## 🎯 Mục tiêu

| Task | Model | Vai trò |
|------|-------|---------|
| **A** | **U2NETp lite** (1.1M params) train from scratch | Thay `rembg` Step 1 |
| **B** | **YOLOv11n-seg** fine-tune COCO | Thay U2NET + visualization |

## 📦 Datasets sử dụng

| Dataset | Số dùng | Mục đích |
|---------|---------|----------|
| **SmartDoc ICDAR 2015** | 4,000 frames | Train chính — A4 trên 5 background |
| **MIDV-500** | 3,000 frames | ID cards, glare, occlusion |
| **Doc3D** | 5,000 ảnh subset | Giấy nhăn/cong/gập |
| **TỔNG** | **12,000 ảnh** | Train 10,500 / Val 900 / Test 600 |

## 📐 4 Phase

```
Phase 1: U2NETp lite — 1 stage train trên 10,500 ảnh doc
   ↓
Phase 2: YOLOv11n-seg — fine-tune COCO trên 7,000 ảnh (SmartDoc + MIDV)
   ↓
Phase 3: Integration vào Pipeline With ML/
   ↓
Phase 4: KPI Speed + Accuracy + Robustness (per-dataset) + E2E
```

## 📊 Khối lượng code Plan B

| Phase | File | Dòng |
|-------|------|------|
| U2NET | 11 | 1500 |
| YOLO | 7 | 1000 |
| Integration | 5 | 800 |
| Benchmark | 5 | 815 |
| Scripts | 7 | 1200 |
| Notebooks | 4 | 60 cells |
| Foundation | 3 | 100 |
| **TỔNG** | **42 file** | **~4500 dòng** |

> Giảm so với Plan A (47 → 42 file) vì bỏ stage1_duts configs, synthesize, verify_masks, sweep.

## ⏱️ Timeline (M4 Max 48GB)

| Ngày | Việc |
|------|------|
| **Hôm nay** | Claude build skeleton (~6h) — chờ user duyệt checklist |
| **Ngày 1** | Tải datasets ~20GB qua đêm |
| **Ngày 2** | Extract frames + prepare labels |
| **Ngày 3-4** | Train U2NETp lite (300 epoch) |
| **Ngày 5** | Train YOLOv11n-seg (150 epoch) |
| **Ngày 6** | Integration test |
| **Ngày 7** | Benchmark + báo cáo |

## 🎯 KPI mục tiêu (Plan B)

| Metric | rembg (baseline) | U2NETp lite | YOLOv11n-seg |
|--------|------------------|-------------|--------------|
| mIoU | 0.78 | **≥ 0.83** | **≥ 0.81** |
| F1 | 0.82 | ≥ 0.87 | ≥ 0.85 |
| Boundary F1 | 0.65 | ≥ 0.76 | ≥ 0.72 |
| FPS (MPS) | 8 | ≥ 20 | ≥ 35 |
| FPS (CoreML) | N/A | ≥ 28 | ≥ 55 |
| OCR-CER VN | 8.5% | < 7.5% | < 8.0% |

## ⚠️ Rủi ro lớn nhất

1. **Train chạy 5-7 ngày** → cần caffeinate chống sleep, backup checkpoint thường xuyên
2. **Tải dataset ~20GB** → mạng tốt + thời gian ban đêm
3. **Doc3D quá lớn (85GB full)** → script chỉ tải subset 8.5GB
4. **Báo cáo cần hiểu code** → Claude giải thích từng module khi user hỏi

## ✅ Cần làm trước khi Claude code (USER)

1. Duyệt [Checklist_Review.md](Checklist_Review.md) — tick các mục đồng ý
2. Trả lời 6 câu hỏi còn lại (G3-G8) trong checklist
3. Xác nhận M4 Max có thể chạy train xuyên đêm

## 📚 Tài liệu hệ thống

```
docs_ml2/
├── TomTat_1Trang.md             ← File này (overview Plan B)
├── PlanB_Datasets_Final.md      ← Chi tiết 3 dataset + timeline
├── KeHoach_TrienKhai_Detail.md  ← Chi tiết kỹ thuật từng file
├── Checklist_Review.md          ← Tick để duyệt (Plan B)
└── claude_plan_docs/            ← 9 file research gốc
    ├── 00_TongHop_Claude.md
    ├── KeHoach_TongQuan_Claude.md
    ├── 01_U2NET_ChiTiet_Claude.md
    ├── 02_YOLO_ChiTiet_Claude.md
    └── 03_Integration_KPI_Claude.md
```

## 🚀 Lệnh chạy mẫu (sau khi build xong)

```bash
# Setup
cd "/Users/binhpham/Documents/Study/MSE/ML 2/Final_deeplearning"
python3 -m venv venv_ml2
source venv_ml2/bin/activate
pip install -r ml2/requirements.txt
python ml2/scripts/check_environment.py

# Test code chạy được (dummy data, không cần tải dataset)
python ml2/scripts/build_dummy_data.py --n 100
python ml2/u2net/train.py --config ml2/u2net/configs/mps_mini.yaml --dummy --epochs 1
python ml2/yolo_seg/train.py --epochs 1 --dummy

# Tải datasets thật (chạy qua đêm)
python ml2/scripts/download_datasets.py --smartdoc --midv500 --doc3d --subset

# Prepare labels
python ml2/scripts/prepare_smartdoc.py
python ml2/scripts/prepare_midv.py
python ml2/scripts/prepare_doc3d.py

# Train chính (Plan B)
caffeinate -i python ml2/u2net/train.py --config ml2/u2net/configs/doc_lite_planB.yaml --device mps
caffeinate -i python ml2/yolo_seg/train.py --epochs 150 --device mps

# Benchmark
python ml2/benchmark/aggregate_results.py
```

## 📋 Trạng thái build

```
[x] Foundation: requirements.txt + .gitignore (2/3)
[ ] U2NET module (0/11)
[ ] YOLO module (0/7)
[ ] Integration (0/5)
[ ] Benchmark (0/5)
[ ] Scripts (0/7)
[ ] Notebooks (0/4)
[ ] README.md (0/1)

TỔNG: 2 / 42 file (5%)
```

---

*Đã chốt Plan B + 3 datasets. Đọc xong → [Checklist_Review.md](Checklist_Review.md) tick duyệt → tôi build.*
