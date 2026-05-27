# YOLO trong Document Layout Analysis vs SOTA — Báo cáo So sánh

> **Nhóm:** 6 — Hệ thống Tự động Căn chỉnh và Làm rõ nét Ảnh chụp Tài liệu
> **Mục tiêu:** Khảo sát các paper YOLO cho Document Layout Analysis (DLA) và so sánh với các giải pháp SOTA mạnh nhất hiện nay, đặc biệt cho ảnh chụp từ điện thoại.

---

## 1. Bối cảnh

DLA (Document Layout Analysis) là bài toán **phát hiện và phân loại các vùng cấu trúc trên ảnh tài liệu**: text block, title, figure, table, caption, list, formula, footnote… Đây chính là module còn thiếu trong pipeline hiện tại của nhóm — và là tiền đề để giải quyết "Binarization phá hủy hình ảnh" mà tài liệu `ThachThuc_AnhChup_TaiLieu.md` đã nêu.

DLA về bản chất là **Object Detection chuyên biệt cho tài liệu**, vì vậy YOLO được áp dụng rất mạnh ở đây.

---

## 2. Các paper YOLO cho Document Layout Analysis

### 2.1 ⭐ Paper quan trọng nhất: DocLayout-YOLO (OpenDataLab/Alibaba, 10/2024)

**Link:** https://arxiv.org/abs/2410.12628 | https://github.com/opendatalab/DocLayout-YOLO

Kiến trúc trên nền YOLOv10-M (~19.6M params) với 2 đóng góp cốt lõi:

1. **Mesh-Candidate BestFit** — Framework sinh dữ liệu synthetic theo dạng bài toán 2D Bin-Packing, tạo ra dataset DocSynth-300K (300.000 ảnh tài liệu đa dạng)
2. **GL-CRM (Global-to-Local Controllable Receptive Module)** — Module nhân chập đa tỉ lệ với dilated conv, tối ưu cho receptive field đặc thù tài liệu

**Kết quả:**
- DocLayNet: **79.7 mAP** (state-of-the-art!)
- D4LA: 70.3 mAP
- DocStructBench: 78.8 mAP
- **85.5 FPS trên T4 (TensorRT FP16)** — nhanh hơn DINO-4scale **3.2×** và DiT-Cascade-L **14.3×**

DocLayout-YOLO (~19.6M params) đánh bại cả LayoutLMv3+Cascade (138M params) và DiT-L+Cascade (340M params) trên DocLayNet.

### 2.2 Các YOLO biến thể khác

| Method | Năm | mAP DocLayNet | Params | Repo |
|---|---|---|---|---|
| DocLayout-YOLO | 2024 | **79.7** | 19.6M | opendatalab/DocLayout-YOLO |
| PP-DocLayout-L (RT-DETR) | 2025 | n/r (90.4 mAP@0.5 nội bộ) | ~32M | PaddlePaddle HF |
| YOLOv8x-doclaynet | 2024 | 78.7 | 68.2M | hantian/yolo-doclaynet |
| YOLOv8m-doclaynet | 2024 | 77.5 | 25.9M | hantian/yolo-doclaynet |
| YOLOv10-DocLayNet (moured) | 2024 | ~77 | ~15M | moured/YOLOv10 |

**Lưu ý quan trọng:** DocLayout-YOLO (19.6M) đánh bại YOLOv8x stock (68.2M) → chứng minh **chỉ scale-up size không hiệu quả bằng thiết kế chuyên biệt cho tài liệu**.

---

## 3. SOTA Non-YOLO mạnh nhất cho DLA

### 3.1 LayoutLMv3 + Cascade R-CNN (Microsoft, 2022)
- Multimodal pre-training trên 11M trang tài liệu
- **PubLayNet: 95.1 mAP** (top performer trên PubLayNet)
- Params: ~138.4M | FPS: ~10 trên T4
- arXiv: 2204.08387 | https://github.com/microsoft/unilm/tree/master/layoutlmv3

### 3.2 DiT (Document Image Transformer) (Microsoft, 2022)
- ViT pre-trained self-supervised trên 42M ảnh tài liệu (DiT-L: 340M params)
- PubLayNet: 94.5 mAP | FPS: 6 trên T4 (chậm nhất)
- arXiv: 2203.02378 | https://github.com/microsoft/unilm/tree/master/dit

### 3.3 VGT (Vision Grid Transformer) (ICCV 2023)
- Hai stream: image + text grid
- **PubLayNet: 96.2 mAP** — kỷ lục cao nhất hiện tại
- ~250M params, FPS ~5
- arXiv: 2308.14978

### 3.4 RoDLA (Robust Document Layout Analyzer) (CVPR 2024)
- **Phương pháp non-YOLO duy nhất thiết kế đặc biệt cho ảnh có nhiễu thực tế**
- Benchmark trên 36 loại perturbation: blur, noise, perspective, lighting, content interference
- +7.1 mAP so với baseline trên DocLayNet-P (perturbed)
- → Mạnh nhất cho ảnh chụp điện thoại
- arXiv: 2403.14442 | https://github.com/yufanchen96/RoDLA

### 3.5 PP-StructureV3 / PP-DocLayoutV3 (PaddlePaddle 2025)
- SOTA trên OmniDocBench (CVPR 2025)
- V3 dự đoán multi-point bbox cho tài liệu cong/nghiêng — thiết kế đặc biệt cho phone camera
- PP-DocLayout-S: 14.5 ms/ảnh trên CPU — hiếm hoi có chỉ số CPU
- arXiv: 2503.17213

---

## 4. Bảng so sánh trực tiếp: YOLO vs SOTA

| Tiêu chí | DocLayout-YOLO | LayoutLMv3+Cascade | DiT-L | VGT | RoDLA |
|---|---|---|---|---|---|
| Năm | 2024 | 2022 | 2022 | 2023 | 2024 |
| Params | **19.6M** ⭐ | 138M | 340M | 250M | 250M |
| mAP DocLayNet | **79.7** ⭐ | ~76 | ~76 | n/r | +7.1 (-P) |
| mAP PubLayNet | n/r | 95.1 | 94.5 | **96.2** ⭐ | n/r |
| FPS GPU (T4) | **85.5** ⭐ | 10 | 6 | 5 | thấp |
| Mobile-ready | ✅ ONNX/CoreML/TFLite 1 lệnh | ❌ quá nặng | ❌ quá nặng | ❌ quá nặng | ❌ quá nặng |
| Robust ảnh chụp | ⚠️ trung bình | ⚠️ trung bình | ⚠️ thấp | ⚠️ thấp | ✅✅✅ **best** |
| Học thuật novelty | ✅✅ Mesh-Candidate + GL-CRM | ✅ Multimodal pretrain | ✅ Self-supervised DiT | ✅✅ Dual-stream | ✅✅✅ Robustness focus |

### Kết luận so sánh

| Tiêu chí | Người thắng |
|---|---|
| Độ chính xác trên ảnh PDF sạch | **VGT (96.2)** hoặc LayoutLMv3 (95.1) |
| Độ chính xác trên tài liệu thực tế đa dạng (DocLayNet) | **DocLayout-YOLO (79.7)** — đánh bại cả Transformer |
| Tốc độ inference | **DocLayout-YOLO (85.5 FPS)** — nhanh hơn DiT-L 14×, LayoutLMv3 8.5× |
| Tỷ lệ Accuracy/Size | **DocLayout-YOLO (19.6M)** — gấp 17× hiệu quả so với DiT-L (340M) |
| Robust với ảnh điện thoại | **RoDLA** (chuyên dụng) hoặc **PP-DocLayoutV3** (multi-point bbox) |
| Mobile-ready cho app | **DocLayout-YOLO** (Ultralytics) hoặc **PP-DocLayout-S** (14.5 ms CPU) |

---

## 5. Khuyến nghị cụ thể cho ảnh chụp điện thoại

Pipeline của nhóm **đã có Step 1 (Dewarp) + Step 2 (Perspective)** — sau khi 2 bước này chạy ảnh đã sạch/phẳng, không cần lo perspective/curve cho DLA nữa. Do đó:

### Lựa chọn tối ưu: DocLayout-YOLO + DocStructBench checkpoint

**Lý do:**
1. SOTA trên DocLayNet (79.7 mAP) — dataset gần nhất với tài liệu thực tế đa dạng
2. Nhanh nhất: 85.5 FPS GPU, export ONNX/CoreML/TFLite trong 1 lệnh Ultralytics
3. Nhẹ nhất: 19.6M params, fit Mac MPS + xuất CoreML cho iPhone
4. Có pretrained weights — không cần train from scratch
5. Tương thích pipeline hiện tại: team đã quen workflow Ultralytics (đã có `yolov8n-seg.pt`)

### Backup: PP-DocLayout-S

Nếu cần triển khai chính trên CPU mobile (14.5 ms/ảnh CPU đã verified).

### Cho phần so sánh học thuật (báo cáo)

So DocLayout-YOLO với **LayoutLMv3+Cascade** hoặc **RoDLA** để chứng minh "small + smart > big". Đây là so sánh có sức nặng học thuật cao.

---

## 6. Liên hệ với pipeline hiện tại của Nhóm 6

Quan sát: Pipeline hiện đã có `yolov8n-seg.pt` (7MB) được train trên COCO 80-class. **Đây là YOLO sai use-case** — COCO không có class "tài liệu", pipeline đang phải hack qua class 'book' (#73) hoặc fallback "mask lớn nhất". DocLayout-YOLO là **YOLO đúng use-case**, train trên tài liệu thực tế.

### Đề xuất nâng cấp pipeline ngay (low-cost, high-impact):

1. **Thay `yolov8n-seg.pt` (COCO) → DocLayout-YOLO DocStructBench checkpoint** ở Step 1
2. Bổ sung **DLA module mới giữa Step 2 và Step 3**: chạy DocLayout-YOLO phân vùng text/figure/table → áp filter Step 3 chọn lọc
3. Giải quyết **Issue #20 trong `ThachThuc_AnhChup_TaiLieu.md`** (dấu mộc đỏ bị grayscale làm hỏng): detect vùng "stamp/signature" riêng → keep RGB, không binarize

Đây cũng chính là **Chủ đề 6 (DocLayoutNet)** trong file `DeXuat_ChuDe_DeepLearning.md`. Với DocLayout-YOLO ra mắt 10/2024, chủ đề này **vừa novel vừa cực kỳ khả thi với Mac MPS**.

---

## 7. Tài liệu tham khảo

| Paper / Repo | Link |
|---|---|
| DocLayout-YOLO (arXiv 2024) | https://arxiv.org/abs/2410.12628 |
| DocLayout-YOLO Code | https://github.com/opendatalab/DocLayout-YOLO |
| PP-DocLayout (arXiv 2025) | https://arxiv.org/abs/2503.17213 |
| PP-DocLayout-L HF | https://huggingface.co/PaddlePaddle/PP-DocLayout-L |
| LayoutLMv3 | https://arxiv.org/abs/2204.08387 |
| LayoutLMv3 Code | https://github.com/microsoft/unilm/tree/master/layoutlmv3 |
| DiT | https://arxiv.org/abs/2203.02378 |
| DiT Code | https://github.com/microsoft/unilm/tree/master/dit |
| VGT (ICCV 2023) | https://arxiv.org/abs/2308.14978 |
| RoDLA (CVPR 2024) | https://arxiv.org/abs/2403.14442 |
| RoDLA Code | https://github.com/yufanchen96/RoDLA |
| DocLayNet Dataset | https://github.com/DS4SD/DocLayNet |
| M6Doc Dataset | https://github.com/HCIILAB/M6Doc |
| OmniDocBench | https://arxiv.org/abs/2412.07626 |
| YOLOv8-DocLayNet baselines | https://huggingface.co/hantian/yolo-doclaynet |
| PDF-Extract-Kit (production pipeline) | https://github.com/opendatalab/PDF-Extract-Kit |

---

*Tài liệu này được sinh từ research tháng 5/2026.*
