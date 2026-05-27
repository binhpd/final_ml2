# Xu hướng Tối ưu YOLO Document Layout + Gap Analysis + Hướng Nghiên cứu

> **Nhóm:** 6 — Hệ thống Tự động Căn chỉnh và Làm rõ nét Ảnh chụp Tài liệu
> **Mục tiêu:** Xác định xu hướng tối ưu YOLO cho Document Layout Analysis (DLA), tìm gap nghiên cứu chưa được giải quyết, và đề xuất hướng xây model YOLO mới có cải tiến.
> **Ràng buộc nhóm:** Mac Apple Silicon (MPS) ONLY, 1 học kỳ.

---

## A. 5 Xu hướng tối ưu YOLO DLA chính (2024–2025)

### A.1 Attention-centric backbones (xu hướng nóng nhất)

YOLO truyền thống là CNN thuần. Từ 2024 trend lớn nhất là tích hợp attention nhưng giữ real-time:

| Model | Năm | Innovation | Liên quan DLA |
|---|---|---|---|
| YOLOv11 | 9/2024 | C3k2 block + C2PSA spatial attention | Spatial attention tập trung vùng text dày đặc |
| YOLOv12 | 2/2025 | Area Attention (A²) — chia spatial thành area, attention tuyến tính + R-ELAN | Giữ receptive field rộng cho ngữ cảnh toàn trang ở chi phí tuyến tính |
| YOLOv13 | 6/2025 | HyperACE (hypergraph correlations) + FullPAD | Hyperedge mô hình quan hệ parent-child layout tự nhiên |

### A.2 Neck redesign (Multi-scale fusion mạnh hơn FPN/PAN)

- **Gold-YOLO** (NeurIPS 2023, arXiv 2309.11331): **Gather-and-Distribute (GD)** neck — fuse global multi-level features thay vì truyền tuần tự như FPN
- Khắc phục vấn đề kinh điển: text nhỏ (footnote) vs figure to (full-page) chênh lệch scale quá lớn

### A.3 Open-vocabulary detection (Zero-shot class mới)

- **YOLO-World** (CVPR 2024, arXiv 2401.17270): YOLO + CLIP text encoder → detect class chưa từng train chỉ qua text prompt
- **YOLO-UniOW** (12/2024): mở rộng open-world thật sự
- Tiềm năng cho DLA: thêm class "QR code", "con dấu", "chữ ký tay" mà không cần retrain

### A.4 Anchor-free + DETR hybrid

- **RT-DETR / RT-DETRv2** (CVPR 2024): DETR đạt real-time → PP-DocLayout-L đạt 90.4% mAP@0.5 dùng nền này
- Xu hướng: YOLO đang dần "ăn cắp" ý tưởng từ DETR (anchor-free, query-based)

### A.5 Linear-complexity backbones (Mamba/SSM cho ảnh)

- **Mamba-YOLO** (6/2024, arXiv 2406.05835), **MambaNeXt-YOLO** (6/2025)
- State Space Model → complexity O(N) thay vì O(N²) → xử lý ảnh tài liệu phân giải cao (1280px+) hiệu quả

---

## B. 10 Gap nghiên cứu chính trong YOLO DLA

| # | Gap | Bằng chứng |
|---|---|---|
| 1 | Camera-captured vs scanned | DocLayout-YOLO train trên DocSynth-300K (render từ PDF), không có paper nào đo performance drop trên ảnh chụp điện thoại |
| 2 | Long-tail classes | Table/formula/caption hiếm vs paragraph; DocLayout-YOLO không class-balanced loss |
| 3 | Tiny text (no P2 head) | DocLayout-YOLO chỉ có 3 head P3/P4/P5; footnote h~8-12px collapse ở stride 8 |
| 4 | Reading order joint | DLAFormer làm joint trong transformer; **không YOLO nào làm joint detect + reading order** |
| 5 | Vietnamese underserved | Survey 6/2025 (arXiv 2506.05061) xác nhận; chỉ có VietCER (~3K) |
| 6 | Cross-domain drop | DocLayout-YOLO: 79.7 DocLayNet → 70.3 D4LA → 0 paper đo trên hóa đơn/form |
| 7 | Mobile latency thật | 0 paper báo cáo FPS trên Apple Neural Engine hay Snapdragon NPU |
| 8 | Confidence calibration | Không paper DLA-specific về calibration; YOLO overconfident trên OOD |
| 9 | Anchor cho text dài | Text-line AR 1:20-1:30+; YOLO regression scale prior từ COCO không phù hợp |
| 10 | Self-supervised cho YOLO docs | DiT làm MIM cho ViT trên 42M ảnh; chưa ai làm tương tự cho backbone YOLO |

---

## C. 7 Hướng nghiên cứu cụ thể

### N1 ⭐ Gold-YOLO GD-neck + Aspect-Ratio-Aware Priors

- **Gap:** #3, #9
- **Đề xuất:** Thay C2f neck của DocLayout-YOLO bằng Low-GD + High-GD của Gold-YOLO; fit regression scale priors qua GMM (k=5) trên phân phối WH của DocLayNet bbox
- **Tại sao work:** Global fused features fix scale gap; AR priors khớp text line dài
- **Mac MPS:** ✅ Pure conv + light attention, fine-tune từ ImageNet weights
- **Expected gain:** +1.5 đến +2.5 mAP DocLayNet; +3-5 AP small classes
- **Risk:** GD self-attention tốn memory hơn C2f

### N2 ⭐ YOLOv12 Area-Attention + P2 head cho tiny text

- **Gap:** #3
- **Đề xuất:** YOLOv12-S backbone + thêm P2 detection head (stride 4) + fine-tune trên DocLayNet+DocSynth-300K
- **Tại sao:** A² cho linear-complexity attention ở P2 high-res (quadratic sẽ chết trên M-series)
- **Mac MPS:** ✅ Ultralytics đã support MPS
- **Expected gain:** +4-6 AP tail small classes
- **Risk:** P2 head làm chậm ~30%

### N3 YOLOv13 HyperACE + Reading-Order auxiliary head

- **Gap:** #4
- **Đề xuất:** Thêm small head trên HyperACE hyperedge features dự đoán pairwise ordering relation (DLAFormer-style); loss = relation BCE + box loss
- **Mac MPS:** ✅ Thêm <5% params
- **Expected gain:** 80%+ pairwise ordering accuracy trong 1 model
- **Risk:** Cần label reading order → data engineering nặng

### N4 ⭐ DiT/DINOv2 Feature Distillation vào YOLO backbone

- **Gap:** #10
- **Đề xuất:** Freeze DiT-base làm teacher; distill patch token features vào YOLOv11-S backbone qua cosine + masked-feature loss trên ~50K trang IIT-CDIP. Chỉ backbone YOLO update.
- **Tại sao:** Hấp thụ document priors của DiT nhưng giữ inference cost của YOLO
- **Mac MPS:** ✅ Teacher inference batch=4 khả thi trong 1-2 ngày
- **Expected gain:** +2-3 mAP DocLayNet; **+4-6 mAP cross-domain**
- **Risk:** Cần `PYTORCH_ENABLE_MPS_FALLBACK=1`

### N5 YOLO-World cho Vietnamese open-vocabulary DLA

- **Gap:** #5
- **Đề xuất:** YOLO-World-S + swap CLIP bằng multilingual-CLIP / BGE-M3; fine-tune trên M6Doc + VietCER + custom Vietnamese
- **Mac MPS:** ✅ YOLO-World-S ~13M params
- **Expected gain:** +10-15 AP trên class Vietnamese novel
- **Risk:** Multilingual CLIP yếu hơn CLIP

### N6 MC-DropBlock Calibration cho OOD phone photos

- **Gap:** #8
- **Đề xuất:** Thêm DropBlock vào neck DocLayout-YOLO; T-step Monte Carlo sampling khi inference; threshold abstention học từ RoDLA-perturbed
- **Mac MPS:** ✅ Chỉ thêm dropout
- **Expected gain:** ECE giảm ~0.15 → <0.05; abstention recall RoDLA-severe >0.7
- **Risk:** Inference chậm 5× (T=5)

### N7 Mamba-Neck (linear complexity)

- **Gap:** #1 + memory constraint
- **Đề xuất:** Thay PAN neck bằng Mamba-YOLO ODSS blocks
- **Mac MPS:** ⚠️ Mamba kernels chưa có MPS native → cần pure-PyTorch SSM scan
- **Expected gain:** +1 mAP, -30% memory @ 1280px
- **Risk:** Hiệu năng MPS bị penalty

---

## D. ⭐ Khuyến nghị: Đề tài "DocLayout-YOLOv2" (Combo N1+N2+N4+N6)

### Công thức

```
DocLayout-YOLOv2 = 
    YOLOv11/v12 backbone (N2: Area Attention + P2 head)
  + Gold-YOLO GD neck (N1: better multi-scale fusion)
  + Aspect-Ratio-Aware Priors (N1: GMM-fit từ DocLayNet)
  + DiT feature distillation pretraining (N4: SSL document priors)
  + MC-DropBlock calibration head (N6: confidence on phone photos)
```

### Tại sao combo này mạnh

| Tiêu chí | Đánh giá |
|---|---|
| Mỗi component ablatable | ✅ 4 ablation studies riêng cho 4 chương báo cáo |
| Tấn công 4 gap khác nhau | ✅ Multi-scale + neck + SSL + calibration |
| Mac MPS feasible | ✅ Tất cả 4 component khả thi |
| Có baseline rõ | ✅ So sánh với DocLayout-YOLO 79.7 mAP |
| Target số liệu | 🎯 81-83% mAP DocLayNet (estimated), 73-75% D4LA |

### Lộ trình 14 tuần

| Tuần | Việc |
|---|---|
| 1–2 | Reproduce DocLayout-YOLO baseline trên Mac MPS, xác nhận 79.7 mAP DocLayNet |
| 3–4 | Implement N1: Gold-YOLO GD neck + GMM aspect-ratio priors |
| 5–6 | Implement N2: thêm P2 head + Area Attention |
| 7–9 | Implement N4: DiT distillation pretraining (phase nặng nhất) |
| 10 | Implement N6: MC-DropBlock + calibration |
| 11 | Ablation study: bật/tắt từng component |
| 12 | Benchmark DocLayNet, D4LA, RoDLA-perturbed (robustness phone-photo) |
| 13 | Export CoreML, đo latency Mac/iPhone (lấp Gap #7) |
| 14 | Viết report + demo |

### Đóng góp học thuật

1. **Architecture novelty**: Lần đầu kết hợp GD-neck + P2 head + AR priors cho DLA
2. **Pretraining novelty**: Lần đầu distill DiT vào YOLO backbone
3. **Evaluation novelty**: Lần đầu công bố FPS trên Apple Neural Engine
4. **Robustness**: Lần đầu đo DocLayout-YOLO variants trên RoDLA-perturbed

---

## E. Hướng "Wild card" (rủi ro cao – reward cao)

### N3 (HyperACE Reading-Order) — nếu nhóm muốn paper hội nghị

- Gap chưa ai giải quyết: Joint YOLO + reading order là vùng trắng
- Có thể publish ICDAR 2026 / DAS 2026 (paper ngắn 4-6 trang)
- Rủi ro: Cần annotate reading order, nặng về data engineering

### N5 (YOLO-World Vietnamese) — nếu nhóm có sẵn dataset Việt

- Đóng góp dataset Vietnamese DLA = giá trị lâu dài
- Submit paper local hoặc track Vietnamese NLP
- Rủi ro: Cần thời gian collect/label

---

## F. Tóm tắt 1 dòng

**Xu hướng đang dịch chuyển từ "scale up YOLO" → "design YOLO chuyên cho document priors"**: attention-centric backbones (v12/v13), GD-neck thay FPN, hypergraph cho relations, open-vocabulary cho class mới, distillation từ DiT/DINOv2.

**Gap đáng khai thác nhất cho 1 học kỳ với Mac:** Combo N1+N2+N4+N6 = **DocLayout-YOLOv2**, target 81-83% mAP DocLayNet, lấp 4 gap, có sẵn baseline.

---

## G. Tài liệu tham khảo

| Paper | arXiv | Repo |
|---|---|---|
| DocLayout-YOLO | 2410.12628 | opendatalab/DocLayout-YOLO |
| YOLOv11 | 2410.17725 | ultralytics/ultralytics |
| YOLOv12 (Area Attention) | 2502.12524 | — |
| YOLOv13 (HyperACE) | 2506.17733 | iMoonLab/yolov13 |
| Gold-YOLO | 2309.11331 | — |
| YOLO-World | 2401.17270 | AILab-CVC/YOLO-World |
| YOLO-UniOW | 2412.20645 | — |
| RT-DETR / v2 | 2407.17140 | lyuwenyu/RT-DETR |
| RTMDet | 2212.07784 | — |
| Mamba-YOLO | 2406.05835 | — |
| MambaNeXt-YOLO | 2506.03654 | — |
| PP-DocLayout | 2503.17213 | PaddlePaddle |
| DLAFormer | 2405.11757 | — |
| RoDLA | 2403.14442 | yufanchen96/RoDLA |
| DiT | 2203.02378 | microsoft/unilm/dit |
| LayoutReader | 2108.11591 | — |
| KD-DETR | 2211.08071 | — |
| DiffusionDet | 2211.09788 | ShoufaChen/DiffusionDet |
| MC DropBlock | 2108.03614 | — |
| Vietnamese DAR Survey | 2506.05061 | — |
| Apple Silicon ML Profiling | 2501.14925 | — |
| SCAN | 2505.14381 | — |
| Revisiting OOD | 2503.07330 | — |

---

*Báo cáo này tổng hợp từ research tháng 5/2026.*
