# Đề Xuất Chủ Đề Deep Learning — Mở Rộng Document Scanner App

> **Nhóm:** 6 — Hệ thống Tự động Căn chỉnh và Làm rõ nét Ảnh chụp Tài liệu
> **Bối cảnh:** Tiếp nối pipeline hiện có (`Pipeline With ML/`) — đề xuất hướng nghiên cứu/đồ án cho môn Deep Learning.
> **Ràng buộc nhóm:** Phạm vi 1 học kỳ, GPU = Mac Apple Silicon (MPS) ONLY (không có NVIDIA GPU), triển khai Hybrid (Mobile + Cloud).

---

## A. Tổng quan kiến trúc Scanner App SOTA dùng Deep Learning

### A.1 Mô hình triển khai Hybrid (Edge + Cloud)

Các app thương mại (CamScanner, Adobe Scan, WPS, Microsoft Lens trước khi bị Copilot thay thế 9/2025) thiết kế theo nguyên tắc:

| Lớp | Vị trí thực thi | Lý do |
|---|---|---|
| **Edge (On-device)** — Detection + Dewarping | CoreML / TFLite / ONNX Runtime Mobile | Realtime viewfinder 60 FPS, không upload, privacy |
| **Cloud (Server GPU)** — Enhancement nặng + OCR + Layout + VLM | PyTorch Server / Triton | Mô hình lớn, phục vụ batch |

> **Insight quan trọng:** Microsoft Lens bị khai tử 9/2025 và được hấp thụ vào Copilot — dấu hiệu rõ rằng VLM (Vision-Language Model) đang nuốt dần stack scanner truyền thống. Hướng tương lai là 1 VLM duy nhất làm cả detect + dewarp + OCR + layout + Q&A.

### A.2 SOTA Model 2024–2025 theo từng bước

**Step 1 — Detection/Segmentation:**
- DocAligner (Pattern Recognition 2025) — đã dùng trong pipeline
- Polar-Doc (arXiv 2312.07925) — one-stage seg + dewarp
- MobileNetV4-UNet (ECCV 2024) — Pareto-optimal cho Apple Neural Engine

**Step 2 — Dewarping:**
- UVDoc (~8M params) — đang dùng
- DocTr++, DocGeoNet, Marior (ACMMM 2022)
- DocHFormer (CVPR 2024), D2Dewarp (2025), TADoc (2025) — các paper mới nhất

**Step 3 — Enhancement (đang là OpenCV thuần — cơ hội lớn):**
- DocBinFormer (2023) — SOTA DIBCO binarization
- DocDiff (ACMMM 2023) — Diffusion residual
- **Uni-DocDiff (ACMMM 2025)** — Một diffusion model unified cho deblur + deshadow + binarize + dewarp + handwriting removal với task prompts
- ShaDocFormer (2023), FSENet/SD7K (2024) — shadow removal
- DocNLC (ICPR 2024) — enhancement nhẹ

**Bổ sung:** LayoutLMv3 / DiT cho Layout | TextSR (2025) cho Super-Resolution text | ESDNet cho Demoiré

### A.3 Pattern thiết kế chung của app scanner hiệu năng cao

1. **Backbone share weights**: 1 encoder phục vụ nhiều head → tiết kiệm 40% latency
2. **Cascading early-exit**: Skip step nếu không cần → giảm 60% thời gian trung bình
3. **Distillation pipeline**: Teacher lớn (UVDoc/DocDiff) → Student nhỏ + INT8 → CoreML
4. **Synthetic data**: Doc3D (100K) cho dewarp, SD7K cho shadow

---

## B. 7 Chủ Đề Deep Learning Đề Xuất

Mỗi chủ đề được đánh giá:
- **Khả thi với Mac MPS (no NVIDIA):** ✅✅✅ cao | ✅✅ trung bình | ✅ thấp
- **Tính mới (Novelty):** ✅✅✅ cao | ✅✅ trung bình | ✅ thấp
- **Mức độ rủi ro deliverable:** ✅✅✅ thấp | ⚠️ trung bình | ⚠️⚠️ cao
- **Mức ảnh hưởng đến pipeline hiện tại:** ✅✅✅ cao | ✅✅ trung bình | ✅ thấp

### ⭐ Chủ đề 1 (KHUYẾN NGHỊ MẠNH NHẤT) — DocEnhance-Lite: Mạng nhẹ Multi-task thay toàn bộ Step 3

**Vấn đề:**
Step 3 hiện tại 100% OpenCV (CLAHE + RGB Shadow Division + Unsharp + Inpaint) → fail trên nền phức tạp, bóng đậm, mực phai, dấu mộc đỏ. Đây là khu vực trống của pipeline.

**Tính mới:**
Hợp nhất 4 nhiệm vụ (deshadow + deglare + denoise + binarization) thành **một U-Net nhẹ (~3–5M params) đa nhánh head**, học multi-task với uncertainty loss weighting (Kendall 2018).

**Kỹ thuật cốt lõi:**
- Backbone: MobileNetV4-Small encoder (pretrained ImageNet)
- Decoder: U-Net 3 cấp với 3 head song song (shadow-free image, binarized mask, color-enhanced)
- Loss: L1 + SSIM + Perceptual (VGG) + Cross-entropy cho head binarize
- Knowledge Distillation từ DocBinFormer (teacher) → student
- Quantize INT8 + export CoreML

**Dataset:**
SD7K (shadow) + DIBCO 2009–2019 (binarize) + augmentation glare/blur lên 500 ảnh từ dataset 1.020 ảnh của nhóm

**Deliverable cụ thể:**
1. Model `.pth` ~5MB
2. Model `.mlpackage` CoreML chạy được trên app DocScannerMobile hiện có
3. So sánh PSNR/SSIM + OCR Word Error Rate vs Step 3 OpenCV cũ
4. Báo cáo: ms/ảnh trên M1/M2 CPU vs MPS

**Đánh giá:** Khả thi ✅✅✅ | Mới ✅✅ | Rủi ro thấp ✅✅✅ | Tác động ✅✅✅

---

### ⭐ Chủ đề 2 — Uni-DocStudent: Knowledge Distillation từ Uni-DocDiff vào model mobile

**Vấn đề:**
Uni-DocDiff (ACMMM 2025) là SOTA giải quyết 6 task tài liệu trong 1 mạng nhưng quá nặng (>500M params, ~10s/ảnh CPU). Không thể chạy mobile.

**Tính mới:**
**Decoder-Free Distillation** từ Uni-DocDiff (teacher, frozen pretrained) → student là Feed-Forward U-Net ~10M params. Học mimicking output cuối thay vì học từng denoising step. Một trong những hướng "hot" của 2024-2025 (Consistency Models, Score Distillation).

**Kỹ thuật cốt lõi:**
- Teacher: Uni-DocDiff pretrained (chạy trên Colab GPU để generate target, KHÔNG train lại)
- Student: U-Net + Task Prompt embedding (5 task)
- Loss: MSE + Perceptual + Style (Gram matrix)
- Quantize FP16 → CoreML

**Dataset:**
Dùng Uni-DocDiff tạo synthetic pairs (input → ideal output) trên 5.000 ảnh DocAligner/MIDV-2020

**Đánh giá:** Khả thi ✅✅ (cần Colab cho phase teacher inference) | Mới ✅✅✅ | Rủi ro trung bình ⚠️ | Tác động ✅✅✅

---

### ⭐ Chủ đề 3 — MobileDocOne: End-to-End Unified Detection + Dewarping

**Vấn đề:**
Step 1 + Step 2 hiện chạy nối tiếp (DocAligner → cắt → UVDoc) → 2 lần forward, không share feature.

**Tính mới:**
Hợp nhất Step 1 + Step 2 thành **single-pass network** với shared backbone MobileNetV4 + 2 head:
- Head A: Heatmap 4 góc (DocAligner-style)
- Head B: UV grid 45×31 (UVDoc-style)
- Cross-attention giữa 2 head để correction lẫn nhau

Tương đồng với Polar-Doc (2023) nhưng dùng grid-based dewarp + tối ưu cho mobile.

**Kỹ thuật cốt lõi:**
- Warm-start: copy weights từ pretrained UVDoc + MobileNetV4-S
- Loss đa task: L1 corner + L2 grid + Edge-aware regularization
- Fine-tune trên Doc3D (100K, đã sinh sẵn)

**Deliverable:**
1. Tốc độ tổng thể Step 1+2: giảm từ ~2s xuống <500ms trên M1
2. Demo iOS hoặc qua DocScannerMobile

**Đánh giá:** Khả thi ✅✅ | Mới ✅✅✅ | Rủi ro trung bình ⚠️ | Tác động ✅✅✅

---

### Chủ đề 4 — VN-DocScan-1K: Vietnamese Document Scanner Benchmark + Báo cáo so sánh

**Vấn đề:**
Tất cả benchmark hiện tại (DocUNet, WarpDoc, DIBCO) dùng tài liệu tiếng Anh/Trung. Chưa có benchmark cho tài liệu tiếng Việt với đặc thù: dấu tiếng Việt, mực xanh chữ ký, dấu mộc đỏ, giấy A4 phổ thông, CCCD.

**Tính mới:**
Xây dataset 1.000+ ảnh tiếng Việt (CCCD, hợp đồng, hóa đơn VAT, sách giáo khoa, văn bản hành chính) với annotation đầy đủ: 4 góc + dewarp grid + OCR ground truth. Benchmark toàn bộ pipeline hiện tại + 5–6 SOTA model.

**Đóng góp:**
1. Public dataset có giá trị nghiên cứu lâu dài
2. Báo cáo so sánh khoa học: OpenCV-only vs U²-Net vs DocAligner vs UVDoc trên tiếng Việt
3. Đề xuất metric mới: **DSR (Document Scan Readiness Score)** = α·PSNR + β·SSIM + γ·OCR-CER + δ·Layout-IoU

**Đánh giá:** Khả thi ✅✅✅ (ít train, chủ yếu evaluate + label) | Mới ✅✅ | Rủi ro rất thấp ✅✅✅ | Tác động ✅✅

> **Lưu ý:** Đây là chủ đề **an toàn nhất** nếu nhóm e ngại train DL nặng. Submit dạng "Empirical Study + Dataset Contribution" — nhiều paper top-tier theo dạng này.

---

### Chủ đề 5 — AdaPipeline: Adaptive Routing Network cho Cascading Hybrid

**Vấn đề:**
Pipeline hiện tại bắt người dùng chọn cờ `--u2net`, `--uvdoc`, `--dewarp-ml` thủ công. Một số ảnh đáng lẽ skip Step 2 thì lại chạy → chậm 3-5 lần.

**Tính mới:**
Train một **Quality Assessment Network siêu nhẹ (1-2M params)** chạy trước pipeline, dự đoán 5 chỉ số:
- `needs_segmentation` (boolean): nền phức tạp?
- `needs_dewarp` (boolean): giấy cong hay phẳng?
- `needs_shadow_removal` (continuous 0-1)
- `needs_deglare` (boolean)
- `needs_binarize` vs `keep_color`

Pipeline tự bật/tắt → giảm 50-70% latency trung bình.

**Kỹ thuật:**
- MobileNetV4-S backbone + 5 head classification/regression
- Train tự giám sát: input ảnh → output từ "oracle pipeline" (chạy mọi tùy chọn rồi chọn best) → label các flag
- Distillation từ oracle

**Đánh giá:** Khả thi ✅✅✅ | Mới ✅✅ | Rủi ro thấp ✅✅✅ | Tác động ✅✅✅

---

### Chủ đề 6 — DocLayoutNet: Layout Analysis cho tài liệu hỗn hợp (Text + Image)

**Vấn đề:**
`ThachThuc_AnhChup_TaiLieu.md` đã nêu rõ — pipeline hiện áp 1 filter Step 3 cho toàn bộ ảnh nên phá hủy hình ảnh màu nếu trang giấy có hình minh họa.

**Tính mới:**
Fine-tune **DiT (Document Image Transformer)** hoặc **LayoutLMv3-base** trên DocLayNet → vẽ bounding box phân vùng (text/figure/table/background). Sau đó áp filter Step 3 chọn lọc theo từng vùng:
- Vùng text: binarize + deshadow
- Vùng figure: chỉ denoise + giữ màu RGB
- Composite lại

**Deliverable:**
Pipeline mới `Pipeline With ML + Layout` có chất lượng tương đương Adobe Scan trên tài liệu có hình.

**Đánh giá:** Khả thi ✅✅ (DiT-base 87M params, fit MPS với fp16) | Mới ✅✅ | Rủi ro trung bình ⚠️ | Tác động ✅✅✅

---

### Chủ đề 7 — VLM-Verifier: Vision-Language Model làm Quality Critic + Retake Advisor

**Vấn đề:**
App scanner hiện tại đưa ảnh ra là xong — không có cơ chế tự đánh giá "ảnh này có đáng tin không, có cần chụp lại không".

**Tính mới (đón đầu xu hướng 2025):**
Plug-in **Phi-3-Vision (4.2B)** hoặc **Qwen2.5-VL-3B** (chạy được trên M1/M2 với MLX hoặc llama.cpp) làm post-processor:
1. Đánh giá điểm chất lượng 0–100
2. Phát hiện lỗi: "Tài liệu bị che 1 góc bởi ngón tay", "Có vết mờ ở giữa trang", "Tài liệu xoay 180°"
3. Đưa hướng dẫn chụp lại bằng tiếng Việt

**Tại sao novel:**
Đây là hướng VLM-as-Critic mới hoàn toàn cho document scanning, hầu như chưa có paper Việt Nam nào làm. MLX framework của Apple → tối ưu Apple Silicon.

**Đánh giá:** Khả thi ✅✅ (chạy inference, không train) | Mới ✅✅✅ | Rủi ro trung bình ⚠️ (chất lượng output VLM khó kiểm soát) | Tác động ✅✅

---

## C. Khuyến nghị cuối cùng + Lộ trình 1 học kỳ

### Top 3 chủ đề khuyến nghị mạnh

1. **Chủ đề 1 (DocEnhance-Lite)** — Lựa chọn an toàn nhất, deliverable rõ ràng. Step 3 đang là điểm yếu nhất pipeline. Khả thi 100% trên Mac.
2. **Chủ đề 5 (AdaPipeline)** — Practical impact cao, dễ demo "trước/sau" cho thầy thấy ngay tốc độ giảm.
3. **Chủ đề 4 (VN-DocScan Benchmark)** — Đóng góp khoa học bền vững, ít rủi ro train DL.

### Kết hợp 2 chủ đề (Mức paper hội nghị)

**Đề tài kết hợp khuyến nghị:** Chủ đề 4 (dataset tiếng Việt) + Chủ đề 1 hoặc 5 (model) → vừa có dataset vừa có model trained-on-it. Đây là cấu trúc paper top-tier rất phổ biến.

### Lộ trình gợi ý (14 tuần)

| Tuần | Việc |
|---|---|
| 1–2 | Đọc paper, dựng môi trường MPS PyTorch trên Mac, xác nhận topic |
| 3–4 | Chuẩn bị dataset (label/synthetic data generation) |
| 5–8 | Train model (fine-tune pretrained, KHÔNG from scratch) |
| 9–10 | Knowledge Distillation + Quantization + Export CoreML/ONNX |
| 11–12 | Tích hợp vào `Backend/DocScannerMobile` hiện có |
| 13 | Benchmark đo PSNR/SSIM/OCR/latency vs pipeline cũ |
| 14 | Viết báo cáo + chuẩn bị demo |

### Stack công cụ khuyến nghị cho Mac Apple Silicon

| Mục đích | Công cụ |
|---|---|
| Training framework | PyTorch 2.x với MPS backend |
| Mô hình lớn | MLX (Apple), llama.cpp |
| Pretrained models | HuggingFace Hub, torch.hub, GitHub releases |
| Knowledge Distillation | `torch.nn.KLDivLoss`, Intel Neural Compressor |
| Quantization | `torch.quantization` (PTQ), CoreMLTools |
| Mobile Export | CoreMLTools (CoreML), ONNX → CoreML, TFLite |
| Synthetic data | Doc3D-renderer, custom OpenCV augment |
| Eval OCR | PaddleOCR Vietnamese, Tesseract |

---

## D. Tài liệu tham khảo chính

| Paper / Repo | URL |
|---|---|
| DocAligner | https://arxiv.org/abs/2306.05749 |
| UVDoc | https://arxiv.org/abs/2302.02887 |
| Marior | https://arxiv.org/abs/2207.11515 |
| DocBinFormer | https://arxiv.org/abs/2312.03568 |
| Uni-DocDiff (ACMMM 2025) | https://arxiv.org/abs/2508.04055 |
| ShaDocFormer | https://arxiv.org/abs/2309.06670 |
| FSENet (SD7K dataset) | https://arxiv.org/abs/2308.14221 |
| MobileNetV4 (ECCV 2024) | https://arxiv.org/abs/2404.10518 |
| Polar-Doc | https://arxiv.org/abs/2312.07925 |
| Awesome Document Image Rectification | https://github.com/fh2019ustc/Awesome-Document-Image-Rectification |
| DocAligner Code (DocsaidLab) | https://github.com/DocsaidLab/DocAligner |
| Apple MLX Framework | https://github.com/ml-explore/mlx |

---

*Tài liệu này được sinh ra từ phân tích codebase hiện tại + research SOTA 2024-2025.*
