# 03 — Research Note: Tóm tắt nền lý thuyết

> **Mục đích:** Tóm tắt 9 file research gốc (~4145 dòng) thành reference ngắn để dùng khi viết báo cáo. File gốc đầy đủ vẫn còn ở [_archive/research_docs/](_archive/research_docs/).
> **Khi cần đào sâu:** Click link đến file gốc tương ứng.

---

## 1. Bối cảnh: Tại sao chọn U²-Net + YOLO?

### 1.1 Lịch sử hành trình research

```
1. Phân tích codebase hiện tại                      → Pipeline 3 bước
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
★ 8. Kế hoạch Training U²-Net + YOLO + KPI         ← Plan B đã chốt
```

### 1.2 Kết luận research

- **Pipeline hiện tại** dùng `rembg` (U²-Net pre-trained generic) → **sai use-case**, hack class 'book' của COCO trong YOLO
- **Cần train chuyên biệt** cho document task với data document-specific
- **DocLayout-YOLO** đánh bại Transformer trên DocLayNet, chứng minh YOLO mới (v8-v11) **vượt SOTA Transformer** cho doc tasks

---

## 2. Tóm tắt 9 file research

### 2.1 `00_TongHop_Claude.md` (14KB) — Master Index

→ [Xem chi tiết](_archive/research_docs/00_TongHop_Claude.md)

**Nội dung chính:**
- Bản đồ thư mục `docs_ml2/` ban đầu
- Bảng so sánh U²-Net vs YOLO-Seg (Salient Object Det vs Instance Seg)
- 4 Phase chính: U²-Net → YOLO → Integration → KPI
- Stack công nghệ + datasets gợi ý

**Trích quan trọng:**

| Tiêu chí | U²-Net | YOLO-Seg |
|---|---|---|
| Loại task | Salient Object Detection | Instance Segmentation |
| Output | 1 mask duy nhất | Multi-instance: mask + bbox + class + conf |
| Multi-doc | Không | Có |
| Visualization | Không | Có (built-in) |

---

### 2.2 `KeHoach_TongQuan_Claude.md` (14KB) — Plan tổng quan ban đầu

→ [Xem chi tiết](_archive/research_docs/KeHoach_TongQuan_Claude.md)

**Nội dung chính:**
- Timeline 12 tuần (Plan A đầy đủ) — **Plan B đã rút xuống 7 ngày**
- 4 option hardware (CUDA single/multi, Cloud, MPS)
- Setup môi trường PyTorch
- Cấu trúc thư mục `ml2/`

→ Phần lớn đã đưa vào [02_Spec_KyThuat.md](02_Spec_KyThuat.md). File gốc giữ làm reference.

---

### 2.3 `01_U2NET_ChiTiet_Claude.md` (28KB) — Chi tiết U²-Net

→ [Xem chi tiết](_archive/research_docs/01_U2NET_ChiTiet_Claude.md)

**Phần đã chuyển sang [02_Spec_KyThuat.md](02_Spec_KyThuat.md):**
- §1 — Kiến trúc U²-Net + RSU blocks
- §3 — Combo loss BCE + IoU + SSIM
- §5 — Evaluation metrics chi tiết

**Phần chưa lấy nhưng còn giá trị (xem gốc):**
- §2.3 — Synthetic data generation code (hiện không dùng, Doc3D đã có sẵn)
- §6 — Hyperparameter sweep + ablation table E1-E8 chi tiết
- §7 — CRF post-processing với pydensecrf (skip cho macOS)
- §8 — Risk matrix chi tiết

---

### 2.4 `02_YOLO_ChiTiet_Claude.md` (30KB) — Chi tiết YOLO-Seg

→ [Xem chi tiết](_archive/research_docs/02_YOLO_ChiTiet_Claude.md)

**Phần đã chuyển sang [02_Spec_KyThuat.md](02_Spec_KyThuat.md):**
- §1 — So sánh YOLOv8/v11 sizes
- §2 — Convert mask → YOLO polygon format
- §3 — Training schedule 3 phase
- §6 — Visualization module YOLODocVisualizer
- §7 — Multi-format export

**Phần chưa lấy nhưng còn giá trị:**
- §3.3 — Mixed Resolution Training (random imgsz mỗi batch)
- §5.2 — Hyperparameter sweep grid
- §5.3 — Fine-tune vs From-scratch ablation
- §9 — Risk matrix YOLO

---

### 2.5 `03_Integration_KPI_Claude.md` (23KB) — Integration + KPI Benchmark

→ [Xem chi tiết](_archive/research_docs/03_Integration_KPI_Claude.md)

**Phần đã chuyển sang [02_Spec_KyThuat.md](02_Spec_KyThuat.md):**
- §5 — KPI formulas (mIoU, F1, MAE, BF, Corner RMSE)
- §6 — File spec wrappers

**Phần chưa lấy:**
- §A.2.6 — Acceptance criteria Integration chi tiết
- §B.4 — Robustness per-category 7 nhóm (Curved, Fold, Incomplete, Perspective, Rotate, Random, Normal) — Plan B đã đổi sang per-dataset (SmartDoc/MIDV/Doc3D)
- §B.5 — OCR-CER E2E benchmark code mẫu
- §B.6 — Comprehensive comparison table template
- §B.7 — Báo cáo cấu trúc gợi ý

---

### 2.6 `DeXuat_ChuDe_DeepLearning_Claude.md` (14KB) — 7 chủ đề DL

→ [Xem chi tiết](_archive/research_docs/DeXuat_ChuDe_DeepLearning_Claude.md)

**Tóm tắt 7 chủ đề:**

1. **DocEnhance-Lite** — End-to-end mobile model thay 3 step truyền thống
2. **AdaPipeline** — Adaptive pipeline chọn module theo input
3. **VN-DocScan** — Dataset tiếng Việt + benchmark
4. **DocSeg-Cascade** — Cascade YOLO + U-Net cho fine boundary
5. **DocAligner-Mobile** — Lite version chạy mobile
6. **MultiTask-DocScan** — Multi-task: seg + dewarp + enhance shared
7. **VLM-DocScan** — Dùng VLM (Vision-Language Model) cho scan

→ **Đề tài đã chọn:** Train U²-Net + YOLO + Integration (gần với chủ đề #4 DocSeg-Cascade)

---

### 2.7 `SoSanh_YOLO_vs_SOTA_DocumentLayout_Claude.md` (9KB) — YOLO vs SOTA

→ [Xem chi tiết](_archive/research_docs/SoSanh_YOLO_vs_SOTA_DocumentLayout_Claude.md)

**Kết luận chính:**

| Model | DocLayNet mAP | FPS | Phù hợp |
|-------|---------------|-----|---------|
| **DocLayout-YOLO** | **79.7** | High | Mobile + Server |
| DiT (Document Image Transformer) | 71.3 | Low | Server only |
| LayoutLMv3 | 73.5 | Low | Server only |
| Donut | 67.2 | Very low | Server, end-to-end |

→ **YOLO mới đánh bại Transformer SOTA** cho document layout. Đây là lý do chọn YOLOv11n-seg cho Plan B.

---

### 2.8 `XuHuong_Gap_HuongNghienCuu_YOLO_DLA_Claude.md` (10KB) — Gap + Hướng N1-N7

→ [Xem chi tiết](_archive/research_docs/XuHuong_Gap_HuongNghienCuu_YOLO_DLA_Claude.md)

**7 Gap đã xác định:**

| # | Gap | Hướng nghiên cứu N |
|---|-----|---------------------|
| 1 | Thiếu domain knowledge văn bản | N1: Pre-train text-aware backbone |
| 2 | Loss không đặc thù doc | N2: Doc-specific loss (line awareness) |
| 3 | Boundary ráp | N3: Cascade refinement với U-Net |
| 4 | Yếu với occlusion | N4: Synthetic occlusion training |
| 5 | Không tận dụng text content | N5: VLM hybrid |
| 6 | Single-scale | N6: Multi-scale attention C2PSA-Doc |
| 7 | Không adapt theo device | N7: Mobile-aware quantization |

→ **Kết hợp N1+N3+N6 = "DocLayout-YOLOv2"** (concept future work cho báo cáo)

---

### 2.9 `DeCuong_DoAn_3CapDo_DocLayoutYOLOv2_Claude.md` (16KB) — Đề cương 3 cấp

→ [Xem chi tiết](_archive/research_docs/DeCuong_DoAn_3CapDo_DocLayoutYOLOv2_Claude.md)

**3 cấp độ đề tài:**

| Cấp | Phạm vi | Workload | Phù hợp |
|-----|---------|----------|---------|
| **1: Reproduce** | Tái sản xuất DocLayout-YOLO baseline | Thấp | Đồ án trung bình |
| **2: Customize** | Fine-tune trên doc Vietnamese + ablation | Trung | Đồ án khá |
| **3: Own model** | Build DocLayout-YOLOv2 mới với N1-N7 | Cao | Đồ án xuất sắc + paper |

→ **Plan B của bạn nằm giữa Cấp 1-2**: tự train U²-Net + YOLO, không build kiến trúc mới.

---

## 3. Trích dẫn (References) chính

### 3.1 Papers nền tảng

| Paper | Năm | arXiv | Vai trò trong Plan B |
|-------|-----|-------|----------------------|
| U-2-Net | 2020 | [2005.09007](https://arxiv.org/abs/2005.09007) | Kiến trúc U²-Net + RSU |
| YOLOv11 | 2024 | [2410.17725](https://arxiv.org/abs/2410.17725) | Backbone + C2PSA |
| YOLOv8 | 2023 | Ultralytics docs | Framework + segmentation head |
| DocLayout-YOLO | 2024 | [2410.12628](https://arxiv.org/abs/2410.12628) | Inspiration |
| MIDV-500 | 2018 | [1807.05786](https://arxiv.org/abs/1807.05786) | Dataset chính |
| MIDV-2020 | 2021 | [2107.00396](https://arxiv.org/abs/2107.00396) | Dataset bổ sung |
| RoDLA | 2024 | [2403.14442](https://arxiv.org/abs/2403.14442) | Robustness benchmark |
| Doc3D | 2019 | [1909.02314](https://arxiv.org/abs/1909.02314) | Wrinkled dataset |

### 3.2 Code repos

| Repo | URL | Vai trò |
|------|-----|---------|
| U-2-Net | [github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) | Reference architecture |
| Ultralytics YOLO | [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | YOLO training framework |
| rembg | [github.com/danielgatis/rembg](https://github.com/danielgatis/rembg) | Baseline cũ |
| MIDV-500 | [github.com/SmartEngines/midv-500](https://github.com/SmartEngines/midv-500) | Dataset |
| SmartDoc | [smartdoc.univ-lr.fr](http://smartdoc.univ-lr.fr/) | Dataset |
| Doc3D | [github.com/cvlab-stonybrook/doc3D-dataset](https://github.com/cvlab-stonybrook/doc3D-dataset) | Dataset |
| Albumentations | [github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations) | Augmentation |
| pytorch-msssim | [github.com/VainF/pytorch-msssim](https://github.com/VainF/pytorch-msssim) | SSIM loss |

---

## 4. Datasets references (mở rộng)

### 4.1 So sánh dataset đã loại bỏ và đã chọn

| Dataset | Plan A cũ | Plan B mới | Lý do |
|---------|-----------|------------|-------|
| **DUTS-TR** (saliency tổng quát) | ✅ Stage 1 pretrain | ❌ Bỏ | Data đích đủ lớn (12K doc-specific) |
| **DUTS-TE** | ✅ Eval Stage 1 | ❌ Bỏ | Không train Stage 1 nữa |
| **SmartDoc-QA** (~150 ảnh) | ✅ Stage 2 train | ❌ Bỏ | Quá ít, dùng SmartDoc ICDAR 2015 |
| **SmartDoc ICDAR 2015** | ❌ Không có | ✅ **Chính** | 4,000 frames có XML 4-góc |
| **MIDV-500** | Optional | ✅ **Chính** | Polygon JSON chuẩn |
| **MIDV-2020** | Không nhắc | Optional | Trùng domain MIDV-500 |
| **Doc3D** | Không nhắc | ✅ **Chính** | Chuyên giấy nhăn/cong |
| **Nhóm 6 1020 ảnh** | ✅ Train + verify | ❌ Bỏ | User không có local data |
| **MIT Indoor Scenes** (bg) | ✅ Synthetic gen | ❌ Bỏ | Bỏ synthetic generation |

### 4.2 Dataset chính (Plan B) — chi tiết link

**SmartDoc ICDAR 2015:**
- Trang chính: http://smartdoc.univ-lr.fr/
- 150 video × 5 background × 6 docs × 5 angles
- Label: XML với 4 corner coords mỗi frame
- Kaggle mirror nếu cần: search "smartdoc 2015"

**MIDV-500:**
- GitHub: https://github.com/SmartEngines/midv-500
- 500 video clips của 50 ID cards
- Label: JSON với polygon quadrilateral
- Paper: arXiv 1807.05786

**Doc3D:**
- GitHub: https://github.com/cvlab-stonybrook/doc3D-dataset
- 100,000 ảnh rendered 3D
- Label: foreground mask + UV map + normals
- Paper: ICCV 2019

---

## 5. Hành trình tài liệu (lịch sử thay đổi)

| Mốc | Thay đổi |
|-----|----------|
| **Khởi đầu** | User viết `KeHoach_Train_TichHop_U2Net_YOLO.md` (18KB) |
| **Research phase** | Claude tạo 9 file phân tích trong `claude_plan_docs/` |
| **Plan v1** | Claude viết 4 file plan top-level (Detail, PlanB_Datasets, TomTat, Checklist) |
| **Chốt scope** | User chọn Plan B + 3 datasets online (SmartDoc/MIDV/Doc3D) |
| **Hệ thống lại** | 14 file → 4 file (README + 01_KeHoach + 02_Spec + 03_Research) + `_archive/` |

---

## 6. Khi nào dùng file research nào

| Tình huống | Đọc file |
|------------|----------|
| Viết phần "Phương pháp" báo cáo | `01_U2NET_ChiTiet_Claude.md` + `02_YOLO_ChiTiet_Claude.md` |
| Viết phần "Đánh giá" báo cáo | `03_Integration_KPI_Claude.md` |
| Viết phần "Liên quan" báo cáo | `SoSanh_YOLO_vs_SOTA_DocumentLayout_Claude.md` |
| Viết phần "Future work" | `XuHuong_Gap_HuongNghienCuu_YOLO_DLA_Claude.md` |
| Defense Q&A về kiến trúc | `01_U2NET_ChiTiet_Claude.md` §1 |
| Defense Q&A về dataset choice | `KeHoach_TongQuan_Claude.md` |
| Khi gặp bug hyperparameter | `01_U2NET_ChiTiet_Claude.md` §6.2 |

---

*Note này tóm tắt nền lý thuyết. Plan thực hiện → [01_KeHoach.md](01_KeHoach.md) | Spec kỹ thuật → [02_Spec_KyThuat.md](02_Spec_KyThuat.md).*
