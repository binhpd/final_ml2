# Checklist Review — Đồ án U2NET + YOLO-Seg (Plan B đã chốt)

> **Trạng thái:** Plan B đã chọn (M4 Max 48GB, 3 dataset online: SmartDoc + MIDV-500 + Doc3D).
> **Cách dùng:** Tick `[x]` vào việc đã duyệt. Còn câu hỏi mở thì ghi đáp án.
> **File song hành:** [KeHoach_TrienKhai_Detail.md](KeHoach_TrienKhai_Detail.md) | [PlanB_Datasets_Final.md](PlanB_Datasets_Final.md) | [TomTat_1Trang.md](TomTat_1Trang.md)

---

## ✅ Quyết định đã chốt

| Mục | Quyết định |
|------|------------|
| Plan | **Plan B** (U2NETp lite + YOLOv11n, 5-7 ngày MPS) |
| Hardware | Mac Studio M4 Max **48GB** unified memory |
| Datasets | **SmartDoc ICDAR2015 + MIDV-500 + Doc3D** (online, không có local) |
| Pretrain DUTS-TR | **BỎ** (data đích đã đủ 12K ảnh) |
| Stage training | **1 stage duy nhất** trên doc data (thay vì 2 stage) |

---

## A. Phê duyệt phạm vi tổng

- [ ] **A1.** Tôi đồng ý phạm vi Plan B (~35 file Python + 4 notebook, ~4500 dòng code)
- [ ] **A2.** Tôi hiểu Claude **không train được model** trong session — chỉ build code, tôi tự train sau
- [ ] **A3.** Tôi hiểu sẽ phải tự chạy training 5-7 ngày trên M4 Max
- [ ] **A4.** Tôi sẽ tự tải datasets từ mạng (SmartDoc, MIDV-500, Doc3D) — Claude viết script tải
- [ ] **A5.** Tôi sẽ tự viết báo cáo bảo vệ — Claude chỉ giải thích code khi tôi hỏi

---

## B. Phê duyệt cấu trúc file Plan B

### B.1 Phase 1 — U2NET (11 file, đã trim cho Plan B)

- [ ] **B1.** `ml2/u2net/model.py` — U2NET + U2NETp + RSU blocks (giữ cả 2 class cho flexibility)
- [ ] **B2.** `ml2/u2net/loss.py` — BCE + IoU + SSIM + EdgeLoss combo
- [ ] **B3.** `ml2/u2net/dataset.py` — Dataset class đọc SmartDoc + MIDV + Doc3D
- [ ] **B4.** `ml2/u2net/augmentation.py` — Albumentations basic + strong
- [ ] **B5.** `ml2/u2net/train.py` — Training loop + MPS support + TensorBoard
- [ ] **B6.** `ml2/u2net/eval.py` — 4 metrics + per-dataset
- [ ] **B7.** `ml2/u2net/infer.py` — Inference + TTA
- [ ] **B8.** `ml2/u2net/visualize.py` — Plot curves + predictions
- [ ] **B9.** `ml2/u2net/configs/doc_lite_planB.yaml` ⭐ **Config chính Plan B**
- [ ] **B10.** `ml2/u2net/configs/doc_full_optional.yaml` (cho ai có CUDA)
- [ ] **B11.** `ml2/u2net/configs/mps_mini.yaml` ⭐ **Config test nhanh trên MPS**

### B.2 Phase 2 — YOLO-Seg (7 file, đã trim cho Plan B)

- [ ] **B12.** `ml2/yolo_seg/prepare_dataset.py` — Convert SmartDoc + MIDV polygon → YOLO format
- [ ] **B13.** `ml2/yolo_seg/train.py` — Wrapper Ultralytics + MPS device
- [ ] **B14.** `ml2/yolo_seg/eval.py` — mAP + custom mIoU + speed
- [ ] **B15.** `ml2/yolo_seg/visualize.py` ⭐ **YOLODocVisualizer (bbox + mask + corners + info)**
- [ ] **B16.** `ml2/yolo_seg/demo_viz.py` — Batch demo + grid montage
- [ ] **B17.** `ml2/yolo_seg/infer_tta.py` — TTA inference
- [ ] **B18.** `ml2/yolo_seg/export_all.py` — ONNX + CoreML (cho mobile M4 Max)

### B.3 Phase 3 — Integration (5 file)

- [ ] **B19.** `ml2/pipeline_integration/u2net_wrapper.py` — Drop-in replacement cho rembg
- [ ] **B20.** `ml2/pipeline_integration/yolo_wrapper.py` — Wrapper YOLO + viz + corner extraction
- [ ] **B21.** `ml2/pipeline_integration/pipeline_u2net.py` — Pipeline với U2NET
- [ ] **B22.** `ml2/pipeline_integration/pipeline_yolo.py` — Pipeline với YOLO + viz đầy đủ
- [ ] **B23.** `ml2/pipeline_integration/test_integration.py` — Test trên test set

### B.4 Phase 4 — Benchmark (5 file)

- [ ] **B24.** `ml2/benchmark/kpi_speed.py` — CPU + MPS + CoreML benchmark
- [ ] **B25.** `ml2/benchmark/kpi_accuracy.py` — 4 metrics so sánh song song
- [ ] **B26.** `ml2/benchmark/kpi_robustness.py` — Per-dataset (SmartDoc / MIDV / Doc3D)
- [ ] **B27.** `ml2/benchmark/kpi_e2e.py` — PSNR + SSIM + OCR-CER + total time
- [ ] **B28.** `ml2/benchmark/aggregate_results.py` — Gộp + xuất CSV + biểu đồ

### B.5 Scripts hỗ trợ (7 file — đã cập nhật theo dataset mới)

- [ ] **B29.** `ml2/scripts/download_datasets.py` — Tải **SmartDoc + MIDV-500 + Doc3D** (script tự động)
- [ ] **B30.** `ml2/scripts/prepare_smartdoc.py` — Extract frames từ video + parse polygon labels
- [ ] **B31.** `ml2/scripts/prepare_midv.py` — Parse MIDV polygon → mask + sample frames
- [ ] **B32.** `ml2/scripts/prepare_doc3d.py` — Extract mask từ Doc3D foreground + subset
- [ ] **B33.** `ml2/scripts/build_dummy_data.py` ⭐ **Để code chạy ngay không cần dataset thật**
- [ ] **B34.** `ml2/scripts/check_environment.py` — Verify M4 Max MPS + dependencies
- [ ] **B35.** `ml2/scripts/caffeinate_train.sh` — Chạy train kèm caffeinate (chống sleep)

### B.6 Notebooks demo (4 file)

- [ ] **B36.** `ml2/notebooks/01_u2net_demo.ipynb` — Load model, forward, train 1 epoch dummy
- [ ] **B37.** `ml2/notebooks/02_yolo_demo.ipynb` — Predict + visualize trên dummy data
- [ ] **B38.** `ml2/notebooks/03_integration_demo.ipynb` — Run cả 2 pipeline
- [ ] **B39.** `ml2/notebooks/04_benchmark_demo.ipynb` — Mini-benchmark + biểu đồ

### B.7 File chung (3 file)

- [x] **B40.** `ml2/requirements.txt` ← **đã tạo**
- [x] **B41.** `ml2/.gitignore` ← **đã tạo**
- [ ] **B42.** `ml2/README.md` — Hướng dẫn cài + chạy

---

## C. Phê duyệt option kỹ thuật

### C.1 Tuỳ chọn module

- [ ] **C1.** Bỏ `pydensecrf` (CRF refinement) — khó build trên macOS ARM
- [ ] **C2.** Giữ `coremltools` cho export mobile (M4 Max có Neural Engine)
- [ ] **C3.** Bỏ Tesseract OCR mặc định — chỉ cài khi chạy `kpi_e2e.py`
- [ ] **C4.** Bỏ synthetic data generator — đã có Doc3D 100K ảnh synthetic sẵn rồi
- [ ] **C5.** Multi-class (3 class) → **bỏ qua**, chỉ 1 class "document"

### C.2 Tuỳ chọn training MPS

- [ ] **C6.** Default batch_size = **16** (M4 Max 48GB đủ rộng, không cần 8)
- [ ] **C7.** Tắt AMP mặc định (MPS AMP còn buggy PyTorch 2.x)
- [ ] **C8.** Input size U2NET = 320 (không cần 384 cho Plan B)
- [ ] **C9.** U2NETp lite **300 epoch** trên 12K ảnh doc (1 stage)
- [ ] **C10.** YOLOv11n **150 epoch** trên 7K ảnh (SmartDoc + MIDV)

### C.3 Tuỳ chọn ablation (Plan B chỉ giữ ablation chính)

- [ ] **C11.** ✅ U2NET: BCE-only vs +IoU vs +SSIM (3 short runs)
- [ ] **C12.** ✅ YOLO: Fine-tune vs from-scratch (2 runs)
- [ ] **C13.** ✅ YOLO: imgsz 640 vs 1024 (2 runs)
- [ ] **C14.** ❌ U2NET full vs Lite — bỏ (chỉ train lite)
- [ ] **C15.** ❌ YOLO size n vs s vs m — bỏ (chỉ train n)
- [ ] **C16.** ✅ Per-dataset eval: SmartDoc vs MIDV vs Doc3D (insight quan trọng)

**Tối thiểu cho báo cáo:** C11 + C12 + C16 (3 ablation chính)

---

## D. Phê duyệt dataset (đã cập nhật)

### D.1 Datasets sẽ tải về

| Dataset | Số ảnh dùng | Size tải | Use case |
|---------|-------------|----------|----------|
- [ ] **D1.** SmartDoc ICDAR 2015 | **4,000 frames** (extract từ 24K) | ~2GB video | Train chính |
- [ ] **D2.** MIDV-500 | **3,000 frames** | ~9GB | Domain bổ sung (ID cards) |
- [ ] **D3.** Doc3D | **5,000 ảnh** (subset của 100K) | ~8.5GB | Giấy nhăn/cong |
- [ ] **D4.** (Optional) MIDV-2020 | 2,000 frames | ~14GB | Mở rộng |

**Tổng download:** ~20GB | **Tổng dùng:** 12,000 ảnh train + 600 test

### D.2 Strategy auto-label

- [ ] **D5.** SmartDoc: parse XML labels có sẵn → mask + 4 corners polygon
- [ ] **D6.** MIDV-500: parse JSON polygon → mask
- [ ] **D7.** Doc3D: dùng foreground mask có sẵn (không cần auto-label)
- [ ] **D8.** **KHÔNG cần manual verify** vì labels từ 3 dataset đều đã ground truth

### D.3 Phân chia split

| Dataset | Train | Val | Test | Strategy |
|---------|-------|-----|------|----------|
| SmartDoc | 3,500 | 300 | 200 | Split theo background_id |
| MIDV-500 | 2,500 | 300 | 200 | Split theo document_id |
| Doc3D | 4,500 | 300 | 200 | Random 90/5/5 |
| **TỔNG** | **10,500** | **900** | **600** | |

- [ ] **D9.** Đồng ý strategy split tránh leakage ở trên

---

## E. Phê duyệt KPI Benchmark

### E.1 4 chiều KPI (giữ nguyên)

- [ ] **E1.** Speed: median latency CPU + MPS + CoreML
- [ ] **E2.** Accuracy: mIoU + F1 + MAE + Boundary-F1 + mAP@0.5 (cho YOLO)
- [ ] **E3.** Robustness: per-dataset (SmartDoc / MIDV / Doc3D)
- [ ] **E4.** E2E: PSNR + SSIM + OCR-CER + total pipeline time

### E.2 Test set (cập nhật)

- [ ] **E5.** 200 ảnh SmartDoc hold-out
- [ ] **E6.** 200 ảnh MIDV-500 hold-out
- [ ] **E7.** 200 ảnh Doc3D hold-out
- [ ] **E8.** Tổng **600 ảnh test set**

### E.3 So sánh

- [ ] **E9.** rembg (baseline) vs U2NETp lite (tự train) vs YOLOv11n (tự train)
- [ ] **E10.** Per-dataset comparison table
- [ ] **E11.** Speed vs Accuracy scatter plot

---

## F. Phê duyệt báo cáo cuối

- [ ] **F1.** Cấu trúc 10 mục (theo plan cũ)
- [ ] **F2.** Figures: training curves + sample predictions + per-dataset bar chart
- [ ] **F3.** Comprehensive comparison table
- [ ] **F4.** Ngôn ngữ báo cáo: ___ (Tiếng Việt / Anh)

---

## G. Câu hỏi vẫn cần trả lời

| # | Câu hỏi | Đáp án |
|---|---------|--------|
| ~~G1~~ | Chọn Plan? | ✅ **Plan B** |
| ~~G2~~ | 1020 ảnh nhóm 6? | ✅ **Không có — dùng online datasets** |
| G3 | Deadline đồ án? | ___ |
| G4 | Báo cáo nộp bằng tiếng Việt hay Anh? | ___ |
| G5 | Mục tiêu điểm? (cân chiều sâu vs scope) | ___ |
| G6 | M4 Max có thể chạy train xuyên đêm 5-7 ngày không? | ___ |
| G7 | Có muốn tôi viết script auto-pause/resume train theo phiên không? | ___ |
| G8 | Có muốn build CoreML export để demo trên iOS/macOS app không? | ___ |

---

## H. Trạng thái tổng (Plan B)

| Phase | File / 42 | Tiến độ |
|-------|-----------|---------|
| Foundation | 3 | 2 / 3 ✅ requirements.txt, .gitignore |
| Phase 1 — U2NET | 11 | 0 / 11 |
| Phase 2 — YOLO | 7 | 0 / 7 |
| Phase 3 — Integration | 5 | 0 / 5 |
| Phase 4 — Benchmark | 5 | 0 / 5 |
| Scripts | 7 | 0 / 7 |
| Notebooks | 4 | 0 / 4 |
| **TỔNG Plan B** | **42** | **2 / 42 (5%)** |

> **Giảm so với Plan A:** 47 → 42 file (bỏ stage1_duts configs, bỏ synthesize_documents.py, bỏ verify_masks.py vì labels có sẵn, bỏ sweep cho YOLO)

---

## I. Action items sau khi review

1. ✅ Tick các mục đồng ý ở trên
2. 📝 Trả lời 6 câu hỏi còn lại mục G
3. 💬 Comment thay đổi (nếu có) bằng cách edit trực tiếp file này

**Sau đó tôi sẽ:**
- Build code theo thứ tự đã chốt
- Báo cáo tiến độ mỗi 5-10 file
- Test code chạy được trên dummy data trước khi báo "xong"

---

## J. Theo dõi tiến độ build (sẽ cập nhật khi build)

```
[x] Bước 0: Foundation - requirements + .gitignore (đã có)
[ ] Bước 1: README + check_environment + build_dummy_data
[ ] Bước 2: U2NET core (model + loss + dataset + augmentation)
[ ] Bước 3: U2NET training (train + eval + infer + visualize + 3 configs)
[ ] Bước 4: Dataset prep scripts (download + 3 prepare scripts)
[ ] Bước 5: YOLO module (7 file)
[ ] Bước 6: Integration (5 file)
[ ] Bước 7: Benchmark (5 file)
[ ] Bước 8: Notebooks (4 file)
[ ] Bước 9: Test end-to-end trên dummy data
```

---

*Checklist Plan B đã cập nhật theo 3 dataset online (SmartDoc + MIDV-500 + Doc3D). Tick và gửi lại để bắt đầu build.*
