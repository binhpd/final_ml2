# Plan B — Datasets & Training Strategy (Đã chốt)

> **Quyết định:** Plan B, M4 Max 48GB, 3 dataset online (không có dataset local)

---

## 1. Datasets sử dụng

### 1.1 SmartDoc ICDAR 2015 (Document Capture)
- **Mô tả:** Video chụp tờ giấy A4 từ camera điện thoại trên 5 background khác nhau (5 background × 6 documents × 5 angles = 150 video)
- **Label:** Tọa độ 4 góc tờ giấy mỗi frame
- **Số ảnh sau extract:** ~24,000 frames (lấy mỗi 5 frame ~4,800 ảnh là đủ)
- **Use case:** Train chính cho Document Segmentation
- **Download:** [smartdoc.univ-lr.fr/datasets/icdar2015](http://smartdoc.univ-lr.fr/) hoặc Kaggle mirror
- **Size:** ~2GB video, ~600MB extracted frames

### 1.2 MIDV-500 / MIDV-2020 (Identity Documents)
- **Mô tả:** 500 video clips của 50 thẻ ID (CMND, passport, driver license) trên đủ background
- **Label:** Polygon 4 góc cực kỳ chính xác (Quadrilateral)
- **Số ảnh sau extract:** ~15,000 frames
- **Use case:** Mở rộng domain (ID cards) + bổ sung challenging scenarios (glare, occlusion)
- **Download:** [github.com/SmartEngines/midv-500](https://github.com/SmartEngines/midv-500) | [midv-2020 paper](https://arxiv.org/abs/2107.00396)
- **Size:** MIDV-500 ~9GB, MIDV-2020 ~14GB

### 1.3 Doc3D (Wrinkled/Curved Papers)
- **Mô tả:** 100,000 ảnh synthetic của giấy bị nhàu nát, gập, cong vênh
- **Label:** Có sẵn UV map + foreground mask
- **Use case:** Augmentation cho ablation "khả năng cắt giấy nhăn"
- **Download:** [github.com/cvlab-stonybrook/doc3D-dataset](https://github.com/cvlab-stonybrook/doc3D-dataset)
- **Size:** ~85GB full | subset 10K ~8.5GB

### 1.4 Quy mô dùng (Plan B trên M4 Max)

| Dataset | Tổng có sẵn | **Dùng cho Plan B** | Lý do |
|---------|-------------|---------------------|-------|
| SmartDoc ICDAR 2015 | 24,000 frames | **4,000 frames** (extract mỗi 6 frame) | Đủ + giảm correlation giữa frames |
| MIDV-500 | 15,000 frames | **3,000 frames** (sample uniform) | Domain bổ sung |
| MIDV-2020 | ~10,000 frames | Bỏ qua / optional 2,000 | Trùng domain với MIDV-500 |
| Doc3D | 100,000 ảnh | **5,000 ảnh subset** | Quá lớn, 5K đủ cho ablation |
| **TỔNG TRAIN** | 149,000 | **~12,000 ảnh** | Khả thi trong 5-7 ngày MPS |
| **TỔNG TEST** | - | **600 ảnh** | 200 SmartDoc + 200 MIDV + 200 Doc3D |

---

## 2. Phân chia Train/Val/Test

```
                     SmartDoc      MIDV-500    Doc3D       TỔNG
─────────────────────────────────────────────────────────────────
Train               3,500          2,500        4,500       10,500
Val                   300            300          300         900
Test                  200            200          200          600
─────────────────────────────────────────────────────────────────
TỔNG                4,000          3,000        5,000       12,000
```

**Chiến lược split:**
- SmartDoc: split theo `background_id` (5 background) → train 4, val + test 1 → tránh leakage
- MIDV-500: split theo `document_id` (50 cards) → train 40, val 5, test 5
- Doc3D: random split 90/5/5 (vì synthetic, không có leakage rủi ro)

---

## 3. Pipeline Training cập nhật

### Phase 1 — U2NET (BỎ DUTS-TR pretrain!)

**Lý do thay đổi:** Vì giờ có **12,000 ảnh document chuyên biệt**, có thể train trực tiếp 1 stage thay vì 2 stage. DUTS-TR pretrain cho saliency tổng quát ít giá trị khi data đích đã đủ lớn.

| Stage cũ | Stage mới |
|----------|-----------|
| Stage 1: DUTS-TR (saliency) → Stage 2: Doc | **Stage 1 duy nhất: Doc data (12K ảnh)** |
| 2 stage × 400+200 epoch = 600 epoch | **1 stage 300 epoch** |
| ~4 ngày MPS | **~2-3 ngày MPS** |

### Phase 2 — YOLO-Seg

**Datasets đầu vào cho YOLO:** SmartDoc + MIDV-500 (cả 2 đều có polygon 4 góc sẵn → convert trực tiếp). Bỏ Doc3D cho YOLO vì foreground mask không phải 4-corner polygon.

| Dataset | YOLO labels | Số ảnh dùng |
|---------|-------------|-------------|
| SmartDoc | ✅ Có (4 góc) | 4,000 |
| MIDV-500 | ✅ Có (polygon) | 3,000 |
| Doc3D | ❌ Không (UV map) | Bỏ qua |
| **TỔNG** | | **7,000 ảnh** |

---

## 4. Lịch trình dự kiến (M4 Max 48GB)

| Ngày | Việc | Output |
|------|------|--------|
| **Hôm nay** | Claude build skeleton (~6h) | 35 file Python + 4 notebook |
| **Ngày 1** | User tải datasets (mạng đêm) | ~25GB downloaded |
| **Ngày 2** | Extract frames + prepare labels | 12,000 ảnh + labels |
| **Ngày 3-4** | Train U2NETp lite (300 epoch) | `u2netp_doc.pth` |
| **Ngày 5** | Train YOLOv11n-seg (200 epoch) | `yolo11n_seg_doc.pt` |
| **Ngày 6** | Integration test trên pipeline | `pipeline_u2net.py`, `pipeline_yolo.py` chạy được |
| **Ngày 7** | Benchmark 4 KPI + báo cáo | `benchmark.csv`, report.md |

---

## 5. Mục tiêu KPI (Plan B)

| Model | mIoU | F1 | Boundary F1 | FPS (MPS) |
|-------|------|----|-----|---------|
| **U2NETp lite** | ≥ 0.83 | ≥ 0.87 | ≥ 0.76 | ≥ 20 |
| **YOLOv11n-seg** | ≥ 0.81 | ≥ 0.85 | ≥ 0.72 | ≥ 35 |
| **rembg (baseline)** | 0.78 | 0.82 | 0.65 | 8 |

→ **Cải thiện kỳ vọng:** mIoU +5-7%, FPS 3-5× so với baseline rembg.

---

## 6. Hardware checklist trước khi train

- [ ] M4 Max 48GB — verify với `python ml2/scripts/check_environment.py`
- [ ] Đủ ổ cứng: **~50GB free** (25GB datasets raw + 25GB extracted + checkpoints)
- [ ] `PYTORCH_ENABLE_MPS_FALLBACK=1` đã set
- [ ] Caffeinate hoặc Amphetamine để máy không sleep trong khi train
- [ ] TensorBoard mở port 6006 để theo dõi
- [ ] Backup checkpoint sang external drive mỗi 24h (an toàn)

---

*Plan B đã chốt. Bắt đầu build code skeleton.*
