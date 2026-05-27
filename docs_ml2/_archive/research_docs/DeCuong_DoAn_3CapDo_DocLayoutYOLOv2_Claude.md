# Đề cương Đồ án "DocLayout-YOLOv2" — Cấu trúc 3 Cấp Độ

> **Nhóm:** 6
> **Đề tài:** Cải tiến mô hình YOLO cho bài toán Document Layout Analysis trên ảnh chụp điện thoại
> **Tiêu chí:** Mức 3 (Tự dựng model dựa trên thuật toán đã chọn) — phủ cả Mức 1 và Mức 2
> **Ràng buộc:** Mac Apple Silicon (MPS) ONLY, 14 tuần (1 học kỳ)

---

## A. Triết lý thiết kế 3 cấp chồng

Cấu trúc đề tài là kim tự tháp: cấp 3 là đỉnh nhưng được xây dựng vững chắc trên cấp 1 và 2. Mỗi cấp chứng minh độc lập giá trị của mình, đồng thời cấp sau dùng output của cấp trước.

```
                ┌─────────────────────────────────┐
                │  CẤP 3: Model mới (Own design)  │
                │  DocLayout-YOLOv2 unified       │
                │  + DiT-Distillation pretraining │
                │  + Calibration head             │
                └──────────────┬──────────────────┘
                               │ build on
                ┌──────────────▼──────────────────┐
                │  CẤP 2: Customize architecture  │
                │  4 ablation modifications:      │
                │  M1 GD-neck | M2 P2 head        │
                │  M3 AR priors | M4 DropBlock    │
                └──────────────┬──────────────────┘
                               │ build on
                ┌──────────────▼──────────────────┐
                │  CẤP 1: Reproduce Baseline      │
                │  DocLayout-YOLO 79.7 mAP        │
                │  Setup pipeline & evaluation    │
                └─────────────────────────────────┘
```

---

## B. CẤP ĐỘ 1 — Reproduce Baseline (Tuần 1–3)

### B.1 Mục tiêu
Build lại DocLayout-YOLO baseline trên Mac MPS:
- Code chạy được, số liệu reproducible khớp paper (±1 mAP cho phép)
- Pipeline evaluation hoàn chỉnh
- Hiểu sâu kiến trúc gốc

### B.2 Tasks chi tiết

| Task | Mô tả | Tuần |
|---|---|---|
| 1.1 | Setup môi trường PyTorch MPS + Ultralytics + DocLayout-YOLO repo | T1 |
| 1.2 | Tải pretrained DocStructBench checkpoint | T1 |
| 1.3 | Tải dataset: DocLayNet, D4LA, DocStructBench eval | T1 |
| 1.4 | Chạy inference, validate mAP gốc | T2 |
| 1.5 | Reproduce training: fine-tune 5 epoch trên DocLayNet train | T2 |
| 1.6 | Setup evaluation pipeline: COCO mAP, per-class AP, FPS | T3 |
| 1.7 | Profile latency: CPU, MPS, CoreML export thử | T3 |
| 1.8 | Viết Chương 1 báo cáo | T3 |

### B.3 Deliverables cấp 1

| Output | Mô tả |
|---|---|
| `code/baseline/` | Fork DocLayout-YOLO với patch MPS compatibility |
| `results/baseline_metrics.json` | mAP DocLayNet/D4LA, FPS Mac M-series |
| `report/chap1_baseline.md` | Chương 1: Phân tích kiến trúc + reproduce results |
| `pretrained/docstructbench.pt` | Local copy checkpoint |

### B.4 Success criteria

- Reproduce mAP DocLayNet 78.5–80.5 (paper 79.7, ±1 cho phép)
- Đo được FPS trên Mac M1/M2 CPU và MPS (đóng góp đầu tiên — Gap #7)
- Có baseline đầy đủ để cấp 2 so sánh

### B.5 Rủi ro & mitigation

| Rủi ro | Mức | Mitigation |
|---|---|---|
| Dataset DocLayNet quá nặng (~80GB) | Cao | Dùng split 10% (~8GB) cho dev, full chỉ ở phase eval cuối |
| MPS một số op không support | Trung bình | `PYTORCH_ENABLE_MPS_FALLBACK=1`, dùng CPU cho phần lỗi |
| Pretrained weights load fail | Thấp | Có nhiều checkpoint mirror trên HuggingFace |

---

## C. CẤP ĐỘ 2 — Customize Architecture (Tuần 4–9)

### C.1 Mục tiêu

Thực hiện 4 modification độc lập lên baseline, mỗi cái có ablation study riêng chứng minh tính hiệu quả. Đây là phần chiếm phần lớn báo cáo vì có 4 thí nghiệm.

### C.2 4 Modifications

#### M1 — Gold-YOLO GD-neck thay C2f neck

- **Thay đổi:** Thay PAN neck mặc định bằng Low-GD + High-GD module của Gold-YOLO
- **Lý do:** FPN/PAN truyền feature tuần tự → mất thông tin giữa scale rất xa nhau (footnote vs figure)
- **Cách làm:** Copy module Gold-YOLO, thay block trong YAML config Ultralytics
- **Ablation:** Baseline neck vs GD neck (giữ nguyên backbone + head)
- **Target gain:** +1 đến +2 mAP DocLayNet, đặc biệt class "footnote", "caption"

#### M2 — Thêm P2 detection head (stride 4) cho tiny text

- **Thay đổi:** Bổ sung head thứ 4 ở P2 level (stride 4) bên cạnh P3/P4/P5 mặc định
- **Lý do:** Footnote/superscript chỉ cao 8-12 px → collapse ở stride 8
- **Cách làm:** Thêm 1 detection head + skip connection từ stage 2 backbone
- **Ablation:** 3-head vs 4-head, đo cả mAP overall lẫn mAP@small
- **Target gain:** +0.5 đến +1.5 mAP overall, **+3 đến +6 AP@small classes**

#### M3 — Aspect-Ratio-Aware Regression Priors

- **Thay đổi:** Fit GMM (k=5) lên phân phối WH bbox trong DocLayNet train; dùng các mean làm scale prior
- **Lý do:** YOLO regression assume aspect ratio ~ COCO; text line có AR 1:20-1:30+
- **Cách làm:** Script Python fit GMM bằng sklearn; sửa head config gán prior
- **Ablation:** COCO default priors vs GMM-fitted priors
- **Target gain:** +0.5 đến +1 mAP, giảm lỗi localization "text-line", "list-item"

#### M4 — MC-DropBlock cho Calibration

- **Thay đổi:** Thêm DropBlock layer (drop_prob=0.1, block_size=7) vào neck; inference T=5 forward → trung bình + variance làm uncertainty
- **Lý do:** Pipeline cần biết khi nào dự đoán sai trên ảnh chụp điện thoại OOD
- **Cách làm:** DropBlock từ `dropblock` package; wrapper MC inference
- **Ablation:** Single forward vs MC-T5, đo Expected Calibration Error (ECE)
- **Target gain:** ECE giảm từ ~0.18 xuống <0.07; abstention recall RoDLA-severe ≥0.7

### C.3 Lịch trình cấp 2

| Tuần | Modification | Hoạt động |
|---|---|---|
| 4 | M1 (GD-neck) | Implement + train 5 epoch + eval |
| 5 | M2 (P2 head) | Implement + train + eval |
| 6 | M3 (AR priors) | Fit GMM, integrate, train + eval |
| 7 | M4 (DropBlock) | Implement, eval calibration |
| 8 | All M1-M4 | Ablation đầy đủ: baseline + M1 + M2 + M3 + M4 + combos |
| 9 | Báo cáo | Viết Chương 2 với 4 ablation tables |

### C.4 Deliverables cấp 2

| Output | Mô tả |
|---|---|
| `code/customize/M1_gd_neck/` | Code + config + checkpoint |
| `code/customize/M2_p2_head/` | (như trên) |
| `code/customize/M3_ar_priors/` | (như trên) + GMM file |
| `code/customize/M4_dropblock/` | (như trên) + MC inference script |
| `results/ablation_table.csv` | So sánh đầy đủ: baseline vs M1-M4 vs combos |
| `report/chap2_customize.md` | Chương 2: Ablation studies + phân tích why |

### C.5 Success criteria

- Mỗi M1-M4 đều có gain dương so với baseline
- Tổng gain combo > 2 mAP (estimated)
- Ablation table có 8+ rows (single + pairwise + triplet + full)
- Phân tích nguyên nhân được — không chỉ báo số

### C.6 Rủi ro & mitigation

| Rủi ro | Mức | Mitigation |
|---|---|---|
| M1 GD-neck quá nặng cho MPS | Trung bình | Dùng GD light, hoặc chỉ Low-GD |
| M2 P2 head làm OOM | Cao | Giảm input size 1024 → 800 |
| M3 GMM fit không đại diện | Thấp | Visualize, kiểm tra silhouette score |
| M4 MC inference chậm 5× | Trung bình | Chỉ chạy MC khi confidence < threshold |

---

## D. CẤP ĐỘ 3 — Build Own Model: DocLayout-YOLOv2 (Tuần 10–13)

### D.1 Mục tiêu

Tự dựng model riêng = N4 (DiT distillation) + tích hợp toàn bộ M1-M4 thành 1 architecture thống nhất tên DocLayout-YOLOv2.

Đây là phần novelty thực sự — combo này chưa ai làm:
1. Self-supervised pretraining cho YOLO docs (Gap #10)
2. Architectural design hợp nhất 4 modifications + 1 pretraining strategy

### D.2 Kiến trúc DocLayout-YOLOv2

```
INPUT IMAGE (1024×1024 phone-captured doc)
        ▼
┌─────────────────────────────────────────────────────┐
│ BACKBONE: YOLOv11-S/v12-S                            │
│ ● Khởi tạo từ DiT-distilled weights (N4 - NOVEL)    │
│   (DiT-base teacher → YOLO backbone student         │
│    via cosine + masked-feature loss trên 50K        │
│    trang IIT-CDIP synthetic)                        │
│ ● 4 stage: P2/P3/P4/P5 feature                      │
└────────────────┬────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────┐
│ NECK: Gold-YOLO GD module (M1)                       │
│ ● Low-GD: gather P2-P5, distribute back              │
│ ● High-GD: global self-attention fusion              │
│ ● + DropBlock layers (M4)                            │
└────────────────┬────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────┐
│ DETECTION HEADS (4 levels)                           │
│ ● P2 head (stride 4) — NEW for tiny text (M2)       │
│ ● P3 head (stride 8)                                 │
│ ● P4 head (stride 16)                                │
│ ● P5 head (stride 32)                                │
│ ● Mỗi head dùng GMM-fitted AR priors (M3)           │
└────────────────┬────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────┐
│ INFERENCE                                            │
│ ● Standard: 1 forward → boxes + scores               │
│ ● Calibrated: T=5 MC sampling → uncertainty (M4)    │
│ ● Abstention: drop box if uncertainty > threshold    │
└─────────────────────────────────────────────────────┘
        ▼
OUTPUT: Layout boxes + calibrated confidence + abstention
```

### D.3 4 phase cấp 3

#### Phase D.3.1 — DiT Distillation Pretraining (Tuần 10–11)

Phase chiếm thời gian nhất nhưng novelty cao nhất.

- **Setup teacher:** Load DiT-base pretrained (`microsoft/dit-base`, ~110M params), frozen
- **Setup student:** YOLOv11-S backbone (~10M params)
- **Dataset:** Subset 50K trang từ IIT-CDIP hoặc DocSynth-300K
- **Loss:**
  ```
  L_distill = α · cosine(student_features, teacher_patch_tokens)
            + β · MSE(student_features, masked_teacher_features)
  ```
- **Training:** 50K iter, batch 8, AdamW, MPS với fp16
- **Time estimate:** 1.5-2 ngày trên M2 Max/Pro

#### Phase D.3.2 — Integration (Tuần 11)

- Merge backbone đã pretrained + neck GD + 4 heads + AR priors + DropBlock
- Fine-tune 10 epoch trên DocLayNet train với LR giảm dần

#### Phase D.3.3 — Comprehensive Evaluation (Tuần 12)

| Chiều | Baselines so sánh | Dataset |
|---|---|---|
| Accuracy chuẩn | DocLayout-YOLO, LayoutLMv3+Cascade, DiT-L | DocLayNet, D4LA |
| Robustness (phone) | RoDLA, DocLayout-YOLO | RoDLA-perturbed |
| Cross-domain | DocLayout-YOLO | Subset M6Doc (photographed) |
| Calibration | DocLayout-YOLO | ECE, Brier score |
| Latency mobile | YOLOv8m-doclaynet | M1/M2 CPU, MPS, CoreML iPhone |

#### Phase D.3.4 — Mobile Deployment (Tuần 13)

- Export ONNX → CoreML
- Test trên iPhone qua app `DocScannerMobile` hiện có
- Đo end-to-end latency (camera → boxes)

### D.4 Deliverables cấp 3

| Output | Mô tả |
|---|---|
| `code/yolov2/` | Code DocLayout-YOLOv2 unified |
| `pretrained/yolov2_dit_distilled.pt` | Backbone đã distilled |
| `pretrained/yolov2_full.pt` | Model trained đầy đủ |
| `models/yolov2.mlpackage` | CoreML export |
| `results/comparison_5dims.csv` | Bảng so sánh 5 chiều |
| `demo/iphone_app/` | App demo trên iPhone |
| `report/chap3_yolov2.md` | Chương 3: Kiến trúc + novelty |
| `report/chap4_evaluation.md` | Chương 4: Phân tích kết quả 5 chiều |

### D.5 Success criteria

| Tiêu chí | Target |
|---|---|
| mAP DocLayNet | ≥ 81.5 (vs baseline 79.7) |
| mAP D4LA | ≥ 72.5 (vs baseline 70.3) |
| Robustness mAP RoDLA-severe | +3-5 vs baseline |
| Cross-domain mAP M6Doc photographed | +3+ vs baseline |
| ECE | <0.07 (vs baseline ~0.18) |
| FPS Mac M2 MPS | ≥ 30 |
| Latency iPhone CoreML | < 200 ms/ảnh |

### D.6 Rủi ro & mitigation

| Rủi ro | Mức | Mitigation |
|---|---|---|
| DiT distillation không converge | **Cao** | Backup: dùng MAE pretraining đơn giản hơn, hoặc skip pretraining chỉ kết hợp M1-M4 |
| Total time vượt 14 tuần | **Cao** | Cấp 3 có thể downscale: bỏ N4, chỉ kết hợp M1-M4 + đặt tên + đo |
| Kết quả không vượt baseline | Trung bình | Cấp 1 + 2 đã có đóng góp; cấp 3 vẫn ghi "đề xuất kiến trúc + đánh giá comprehensive" |
| iPhone deployment fail | Thấp | Fallback dừng ở ONNX export trên Mac |

---

## E. Tổng kết cấu trúc báo cáo (5 chương chuẩn)

| Chương | Nội dung | Cấp |
|---|---|---|
| Chương 1 | Tổng quan + Bài toán + SOTA DLA | Foundation |
| Chương 2 | Reproduce DocLayout-YOLO + Phân tích baseline | **Cấp 1** |
| Chương 3 | 4 Modifications: ablation studies | **Cấp 2** |
| Chương 4 | DocLayout-YOLOv2: Kiến trúc thống nhất + Distillation | **Cấp 3** |
| Chương 5 | Comprehensive Evaluation 5 chiều + Demo iPhone | Kết quả |

---

## F. Đóng góp khoa học có thể publish

1. Lần đầu reproduce + công bố FPS DocLayout-YOLO trên Apple Silicon (Gap #7)
2. Ablation study 4 modifications độc lập (chưa ai làm trên DocLayout-YOLO)
3. Lần đầu distill DiT vào YOLO backbone cho DLA — workshop paper potential (Gap #10)
4. Robustness evaluation trên RoDLA-perturbed cho DocLayout-YOLO (chưa có trong paper gốc) — Gap #1
5. Calibration framework cho DLA (Gap #8)

---

## G. Tổng kết deliverable cuối cùng

- **Code:** 4 thư mục cho 4 phase (baseline, customize, yolov2, demo), có README chạy Mac MPS
- **Báo cáo:** ~50-60 trang Markdown/PDF, 4-5 ablation tables, 5 comparison tables, 8+ figures
- **Model artifacts:** Pretrained baseline + 4 customize + yolov2 + CoreML
- **Demo:** App DocScannerMobile chạy model mới + video record

---

## H. Tài liệu tham khảo chính

| Paper | arXiv | Repo |
|---|---|---|
| DocLayout-YOLO (baseline) | 2410.12628 | opendatalab/DocLayout-YOLO |
| YOLOv11 | 2410.17725 | ultralytics/ultralytics |
| YOLOv12 (Area Attention) | 2502.12524 | — |
| Gold-YOLO (M1) | 2309.11331 | huawei-noah/Efficient-Computing |
| DiT (N4 teacher) | 2203.02378 | microsoft/unilm/dit |
| MC DropBlock (M4) | 2108.03614 | — |
| RoDLA (eval) | 2403.14442 | yufanchen96/RoDLA |
| LayoutLMv3 (eval baseline) | 2204.08387 | microsoft/unilm/layoutlmv3 |
| DocLayNet dataset | 2206.01062 | DS4SD/DocLayNet |
| M6Doc dataset | — | HCIILAB/M6Doc |

---

*Đề cương được soạn 5/2026 dựa trên research thị trường tháng 5/2026.*
