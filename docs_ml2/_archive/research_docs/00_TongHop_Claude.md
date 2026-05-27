# Tổng Hợp Master — Kế hoạch Training 2 Model U2NET + YOLO

> **Mục tiêu trọng tâm:** Train 2 model Deep Learning **U2NET** và **YOLO-Seg** cho bài toán Document Segmentation
> **Bối cảnh:** Mở rộng pipeline hiện có (`Pipeline With ML/`)

---

## 📖 Bản đồ thư mục `docs_ml2/`

```
docs_ml2/
│
├── 00_TongHop_Claude.md ← BẠN ĐANG ĐỌC FILE NÀY (Master Index)
│
├── ─── PHẦN A: Phân tích & Đề xuất (đã hoàn thành research) ───
│   ├── DeXuat_ChuDe_DeepLearning_Claude.md          (7 chủ đề DL)
│   ├── SoSanh_YOLO_vs_SOTA_DocumentLayout_Claude.md (YOLO vs SOTA DLA)
│   ├── XuHuong_Gap_HuongNghienCuu_YOLO_DLA_Claude.md (Gap + 7 hướng N1-N7)
│   └── DeCuong_DoAn_3CapDo_DocLayoutYOLOv2_Claude.md (Đề cương 3 cấp)
│
└── ─── PHẦN B: Kế hoạch Build & Training (TRỌNG TÂM) ───
    ├── KeHoach_TongQuan_Claude.md       (Master plan + setup môi trường)
    ├── 01_U2NET_ChiTiet_Claude.md       (Build & Train U2NET from scratch)
    ├── 02_YOLO_ChiTiet_Claude.md        (Build & Train YOLO-Seg)
    └── 03_Integration_KPI_Claude.md     (Integration + KPI Benchmark)
```

---

## 🎯 MỤC TIÊU CỐT LÕI: Train 2 Model

| Task | Model | Variant | Training Strategy | Vai trò |
|---|---|---|---|---|
| **A** | **U2NET** | Full (44M) + Lite (1.1M) | **Train from scratch** 2-stage | Thay `rembg` trong Step 1 |
| **B** | **YOLO-Seg** | v11n (2.9M) + v11s (10M) + v11m (22M) | Fine-tune COCO + ablation from-scratch | Thay U2NET + visualization |

### Sự khác biệt cốt lõi

| Tiêu chí | U2NET | YOLO-Seg |
|---|---|---|
| Loại task | Salient Object Detection | Instance Segmentation |
| Output | 1 mask duy nhất | Multi-instance: mask + bbox + class + conf |
| Training mode | **From scratch** (2 stage) | Fine-tune COCO + ablation |
| Architecture | Nested U-Net (RSU) | CSPNet + PAN-FPN + ProtoNet |
| Khả năng detect multi-doc | Không | Có |
| Visualization | Không | Có (built-in + custom) |

---

## 🔨 4 Phase chính

```
        ┌─────────────────────────────────────────────┐
        │ PHASE 4: KPI Benchmark & So sánh            │
        │ Speed/Accuracy/Robustness/E2E Pipeline      │
        └────────────────────┬────────────────────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │ PHASE 3: Integration                        │
        │ U2NET → pipeline cũ                         │
        │ YOLO → thay U2NET + Visualization           │
        └────────────────────┬────────────────────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │ PHASE 2: Build & Train YOLO-Seg             │
        │ 3 variant (n/s/m) + ablation from-scratch   │
        └────────────────────┬────────────────────────┘
                             │ song song được
        ┌────────────────────▼────────────────────────┐
        │ PHASE 1: Build & Train U2NET (TRỌNG TÂM)   │
        │ 2 variant (full + lite) train from scratch  │
        │ Stage 1 (DUTS-TR) → Stage 2 (Doc data)      │
        └─────────────────────────────────────────────┘
```

---

## 📊 PHẦN 1 — Tóm tắt Bài toán + Pipeline hiện tại

### 1.1 Bài toán

Biến ảnh chụp điện thoại (nghiêng, méo, có bóng, mờ, nhiễu) thành **bản scan phẳng, rõ nét** — tương đương CamScanner/Adobe Scan.

### 1.2 Pipeline hiện tại (3 bước)

| Bước | Tên | Trạng thái |
|---|---|---|
| **Step 1** | Document Detection | DL: U²-Net (rembg), DocAligner, YOLOv8-seg |
| **Step 2** | Geometric & Dewarping | DL: UVDoc, Perspective Transform (cổ điển) |
| **Step 3** | Image Enhancement | 100% OpenCV thuần |

### 1.3 Quan sát

- Pipeline gắn nhãn "ML" nhưng thực chất 100% là Deep Learning
- Hiện dùng `yolov8n-seg.pt` (COCO 80-class) — sai use-case, đang hack class 'book'
- → Cần **train chuyên biệt** cho document task

---

## 🎯 PHẦN 2 — Plan Training Chi Tiết

### 2.1 Task A: U2NET (Phase 1)

→ Chi tiết: **[01_U2NET_ChiTiet_Claude.md](./01_U2NET_ChiTiet_Claude.md)**

**Strategy:** Train **from scratch** trên DUTS-TR + SmartDoc + 1020 ảnh nhóm. Train cả 2 variant (full + lite) để so sánh.

**2 Stage training:**

```
Stage 1: Pretrain trên DUTS-TR (10,553 ảnh)
   ↓ Target: mIoU DUTS-TE ≥ 0.87 (full), ≥ 0.84 (lite)
   ↓
Stage 2: Fine-tune trên Doc data (SmartDoc + auto-label 1020 + synthetic)
   ↓ Target: mIoU verified test ≥ 0.88 (full), ≥ 0.85 (lite)
```

**Loss function:** Combo BCE + IoU + SSIM (multi-supervision 6 side outputs + fused)

**Ablation chính:**
- BCE-only vs +IoU vs +SSIM vs +EdgeLoss
- Input size 256/288/320/384/512
- U2NET full vs U2NETp lite
- Synthetic data on/off
- TTA on/off

### 2.2 Task B: YOLO-Seg (Phase 2)

→ Chi tiết: **[02_YOLO_ChiTiet_Claude.md](./02_YOLO_ChiTiet_Claude.md)**

**Strategy:** Fine-tune từ COCO pretrained cho 3 sizes (n, s, m) + ablation train from-scratch để so sánh.

**3 Phase training:**

```
Phase 1: Warmup + Head only (epoch 1-20)
   ↓ Freeze backbone, lr=1e-3, mosaic=1.0, mixup=0.15
   ↓
Phase 2: Full unfreeze (epoch 21-150)
   ↓ All layers, cosine lr decay, mosaic off ở 20 epoch cuối
   ↓
Phase 3: High-res fine-tune (epoch 151-200)
   ↓ imgsz 1024, lr=1e-5, no mosaic/mixup
```

**Output:** mask + bbox + corners (trích từ mask)

**Ablation chính:**
- Size n vs s vs m
- imgsz 640 vs 1024
- Fine-tune COCO vs from-scratch
- Aug strength: basic vs strong
- TTA on/off
- Multi-class (1 vs 3) option

### 2.3 Visualization Module (Phase 2 bổ sung)

Vẽ:
1. Bounding box + label + confidence
2. Segmentation mask overlay (bán trong suốt)
3. 4 corners trích từ mask
4. FPS + inference time + detection count
5. Class name (nếu multi-class)

---

## 🔗 PHẦN 3 — Integration + Benchmark

→ Chi tiết: **[03_Integration_KPI_Claude.md](./03_Integration_KPI_Claude.md)**

### 3.1 Integration U2NET (Phase 3.1)

- Wrapper class `U2NETSegmentor` thay thế `rembg.remove()`
- Drop-in replacement: giữ nguyên logic Step 1 còn lại
- Copy `main.py` → `pipeline_u2net.py`

### 3.2 Integration YOLO + Visualization (Phase 3.2)

- Wrapper class `YOLODocSegmentor` với visualization
- Thay U2NET hoàn toàn
- Pipeline mới `pipeline_yolo.py` với output viz đầy đủ

### 3.3 KPI Benchmark 4 chiều (Phase 4)

```
        ┌─────────────────────────┐
        │   KPI BENCHMARK         │
        │                         │
   ┌────┼─────────────┬───────────┤
   │    │             │           │
 SPEED  ACCURACY  ROBUSTNESS    E2E
 FPS    mIoU/F1   Per-cat       PSNR/SSIM
 ms     mAP/MAE   (7 nhóm)       OCR-CER
```

**Comprehensive Comparison Table mong đợi:**

| | rembg (cũ) | U2NET full | U2NETp lite | YOLOv11n | YOLOv11s | YOLOv11m |
|---|---|---|---|---|---|---|
| Speed GPU (ms) | 120 | 15 | 5 | 8 | 12 | 22 |
| FPS | 8 | 65 | 200 | 130 | 80 | 45 |
| mIoU | 0.78 | 0.88 | 0.85 | 0.83 | 0.86 | 0.88 |
| F1 | 0.82 | 0.91 | 0.88 | 0.85 | 0.88 | 0.90 |
| Boundary F1 | 0.65 | 0.82 | 0.78 | 0.72 | 0.76 | 0.79 |
| mAP@0.5 (box) | N/A | N/A | N/A | 0.88 | 0.92 | 0.94 |
| Model size | 176MB | 176MB | 4.7MB | 6MB | 22MB | 50MB |

**Insight cho báo cáo:** Trade-off rõ — U2NET thắng boundary precision, YOLO thắng speed + multi-instance + viz.

---

## 🗺️ Hành trình research đến thời điểm này

```
1. Phân tích codebase hiện tại                      → Bài toán + Pipeline 3 bước
            ▼
2. Phân loại ML vs DL trong pipeline                → Toàn bộ là DL, không có ML cổ điển
            ▼
3. Nghiên cứu SOTA Scanner App 2024-2025           → Hybrid Edge+Cloud, VLM trend
            ▼
4. Đề xuất 7 chủ đề DL                              → Top 3: DocEnhance-Lite, AdaPipeline, VN-DocScan
            ▼
5. Đào sâu YOLO trong Document Layout Analysis     → DocLayout-YOLO 79.7 mAP DocLayNet
            ▼
6. So sánh YOLO vs SOTA Transformer                → DocLayout-YOLO đánh bại DiT, LayoutLMv3
            ▼
7. Tìm Gap + Đề xuất 7 hướng nghiên cứu mới        → N1-N7, kết hợp = DocLayout-YOLOv2
            ▼
8. Cấu trúc 3 cấp độ theo tiêu chí                  → Cấp 1 (reproduce) + Cấp 2 (customize) + Cấp 3 (own model)
            ▼
★ 9. Kế hoạch Training U2NET + YOLO + KPI         ← VỊ TRÍ HIỆN TẠI
        Tập trung train 2 model từ đầu, không chỉ tái sử dụng pretrained
```

---

## 📌 PHẦN 4 — Quick Reference

### 4.1 Stack công nghệ

| Mục đích | Tool |
|---|---|
| Training framework | PyTorch 2.x (CUDA/MPS/CPU đều OK) |
| YOLO framework | Ultralytics |
| Image processing | OpenCV, PIL, Albumentations |
| SSIM loss | pytorch-msssim |
| CRF refinement | pydensecrf |
| Metrics | scikit-learn, scipy |
| Logging | TensorBoard, W&B |
| Export | ONNX, OpenVINO, CoreML, TFLite |
| OCR (eval) | PaddleOCR Vietnamese, Tesseract |

### 4.2 Datasets cần thiết

| Dataset | Số ảnh | Mục đích |
|---|---|---|
| DUTS-TR | 10,553 | U2NET Stage 1 train |
| DUTS-TE | 5,019 | U2NET Stage 1 eval |
| SmartDoc-QA | ~150 | Document segmentation |
| MIDV-500 (optional) | ~15K frames | ID documents |
| Nhóm 6 1020 ảnh | 1020 | Domain-specific (auto-label + verify 200-400) |
| Synthetic (optional) | 5K-10K | Sinh từ SmartDoc + backgrounds |

### 4.3 Mục tiêu cuối

| Model | mIoU | F1 | Speed (GPU FP32) | Model size |
|---|---|---|---|---|
| U2NET full | ≥ 0.88 | ≥ 0.91 | ~15ms | 176MB |
| U2NETp lite | ≥ 0.85 | ≥ 0.88 | ~5ms | 4.7MB |
| YOLOv11n-seg | ≥ 0.83 (mAP_box ≥ 0.88) | ≥ 0.85 | ~8ms | 6MB |
| YOLOv11s-seg | ≥ 0.86 (mAP_box ≥ 0.92) | ≥ 0.88 | ~12ms | 22MB |
| YOLOv11m-seg | ≥ 0.88 (mAP_box ≥ 0.94) | ≥ 0.90 | ~22ms | 50MB |

### 4.4 Quick command reference

```bash
# Setup
python3 -m venv venv_ml2 && source venv_ml2/bin/activate
pip install torch ultralytics opencv-python albumentations pytorch-msssim rembg

# Auto-label 1020 ảnh
python ml2/scripts/autolabel_nhom6.py

# Train U2NET full Stage 1
python ml2/u2net/train.py --config configs/stage1_duts_full.yaml

# Train U2NET full Stage 2
python ml2/u2net/train.py --config configs/stage2_doc_full.yaml

# Train U2NETp lite Stage 1
python ml2/u2net/train.py --config configs/stage1_duts_lite.yaml

# Train YOLO 3 sizes
python ml2/yolo_seg/train.py --size n --imgsz 640 --epochs 200
python ml2/yolo_seg/train.py --size s --imgsz 640 --epochs 200
python ml2/yolo_seg/train.py --size m --imgsz 1024 --epochs 150

# Hyperparameter sweep
python ml2/yolo_seg/sweep.py

# Run benchmark sau khi train xong
python ml2/benchmark/kpi_speed.py
python ml2/benchmark/kpi_accuracy.py
python ml2/benchmark/kpi_robustness.py
python ml2/benchmark/kpi_e2e.py
```

---

## ✅ PHẦN 5 — Checklist Master

### Phase 0: Setup & Data
- [ ] Setup môi trường PyTorch + Ultralytics
- [ ] Tải DUTS-TR + DUTS-TE
- [ ] Tải SmartDoc-QA
- [ ] Auto-label 1020 ảnh với rembg ensemble
- [ ] Manual verify 200-400 ảnh subset
- [ ] (Optional) Sinh synthetic data

### Phase 1: U2NET
- [ ] Implement U2NET + U2NETp architecture
- [ ] Implement combo loss + dataset + training loop
- [ ] Train U2NET full Stage 1 (DUTS-TR)
- [ ] Train U2NET full Stage 2 (Doc data)
- [ ] Train U2NETp Stage 1
- [ ] Train U2NETp Stage 2
- [ ] Ablation studies ≥ 8 experiments
- [ ] Per-category evaluation 7 nhóm

### Phase 2: YOLO-Seg
- [ ] Convert mask → YOLO polygon format
- [ ] Train YOLOv11n-seg
- [ ] Train YOLOv11s-seg
- [ ] Train YOLOv11m-seg
- [ ] (Optional) Train v11n từ scratch (ablation)
- [ ] Hyperparameter sweep top 5
- [ ] Implement visualization module
- [ ] Demo viz 50+ ảnh
- [ ] Export ONNX/CoreML/OpenVINO/TFLite

### Phase 3: Integration
- [ ] U2NET wrapper class
- [ ] Pipeline U2NET (thay rembg)
- [ ] YOLO wrapper class
- [ ] Pipeline YOLO với viz
- [ ] Test trên 1020 ảnh, đo tỷ lệ thành công

### Phase 4: Benchmark
- [ ] Speed benchmark: CPU + GPU + Mobile (3 backend)
- [ ] Accuracy benchmark: mIoU/F1/MAE/BF + mAP
- [ ] Robustness per-category 7 nhóm
- [ ] E2E pipeline: PSNR/SSIM/OCR-CER/total time
- [ ] Comprehensive comparison table
- [ ] Báo cáo final markdown

---

## 🔗 PHẦN 6 — Tài liệu nhóm gốc

Các file gốc trong `docs/`:

| File | Nội dung |
|---|---|
| `BaiToan.md` | Định nghĩa bài toán + pipeline |
| `PIPELINE_WORKFLOW.md` | Workflow chi tiết |
| `DanhGiaGiaiPhap.md` | So sánh truyền thống vs ML |
| `GiaiThich_Step1/2/3_*.md` | Giải thích từng Step |
| `ThachThuc_AnhChup_TaiLieu.md` | 20 thách thức |
| `Pipeline_Diagram_HighRes.png` | Sơ đồ pipeline |

---

## 📎 PHẦN 7 — References

### Papers chính
| Paper | arXiv | Link |
|---|---|---|
| U-2-Net (2020) | 2005.09007 | https://arxiv.org/abs/2005.09007 |
| YOLOv11 | 2410.17725 | https://arxiv.org/abs/2410.17725 |
| YOLOv8 | - | Ultralytics docs |
| DocLayout-YOLO | 2410.12628 | https://arxiv.org/abs/2410.12628 |
| RoDLA (CVPR 2024) | 2403.14442 | https://arxiv.org/abs/2403.14442 |
| DUTS dataset | - | saliencydetection.net/duts |

### Code repos
| Repo | URL |
|---|---|
| U-2-Net | github.com/xuebinqin/U-2-Net |
| Ultralytics YOLO | github.com/ultralytics/ultralytics |
| rembg | github.com/danielgatis/rembg |
| Albumentations | github.com/albumentations-team/albumentations |
| pytorch-msssim | github.com/VainF/pytorch-msssim |

---

*Master index — gom toàn bộ research + kế hoạch training 2 model U2NET + YOLO-Seg của Nhóm 6.*
