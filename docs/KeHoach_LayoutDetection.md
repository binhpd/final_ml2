# Kế hoạch: Nâng cấp phát hiện bố cục văn bản (Block + Line)

> Mục tiêu: Sau khi tài liệu đã được dewarp và enhance (Step 1-3 hiện tại), thêm **Step 4: Layout Analysis** để phát hiện cấu trúc tài liệu ở 2 cấp độ — **Block** (tiêu đề, đoạn văn, bảng, ảnh) và **Line** (dòng văn bản). Đây là bước nền tảng cho OCR có cấu trúc và trích xuất thông tin.

---

## 1. Bối cảnh & lý do

Pipeline hiện tại dừng ở [step3_enhancer.py](Pipeline%20With%20ML/step3_enhancer.py) → cho ra ảnh tài liệu sạch, phẳng. Nhưng:

- Người dùng vẫn cần OCR thủ công từng vùng
- Không có thông tin cấu trúc (đâu là tiêu đề, đâu là cột)
- Không hỗ trợ tài liệu nhiều cột, bảng, biên lai phức tạp

**Step 4 mới** sẽ trả về JSON cấu trúc kiểu:
```json
{
  "blocks": [
    {"type": "title", "bbox": [x,y,w,h], "lines": [{"bbox":..., "text_ready": true}]},
    {"type": "paragraph", "bbox": [...], "lines": [...]},
    {"type": "table", "bbox": [...], "cells": [...]}
  ]
}
```

---

## 2. Lựa chọn công nghệ

### 2.1. So sánh các phương án Block-level

| Framework | Pretrained | Tiếng Việt | Tốc độ CPU | Ưu | Nhược |
|---|---|---|---|---|---|
| **LayoutParser + PubLayNet** | Detectron2/Faster-RCNN | OK (layout không phụ thuộc ngôn ngữ) | ~2s | Dễ tích hợp, có visualizer | Mô hình lớn (~200MB) |
| **PaddleOCR PP-StructureV2** | Picodet-LCNet | Có support TV | ~0.5s | Nhẹ, có sẵn line + table | Cần PaddlePaddle (~500MB) |
| **DocLayout-YOLO** | YOLOv10-doc | OK | ~0.3s | Nhanh nhất, YOLOv10 base | Mới (2024), ít doc |
| **Donut/LayoutLMv3** | Transformer | Có | ~5s+ | SOTA, end-to-end | Quá nặng cho CPU |

**Khuyến nghị**: **DocLayout-YOLO** cho block + **DBNet** (qua PaddleOCR) cho line.
Lý do: YOLOv8 đã có trong dự án → DocLayout-YOLO tái sử dụng tốt; PaddleOCR có DBNet built-in và hỗ trợ tiếng Việt.

### 2.2. So sánh các phương án Line-level

| Model | Đặc trưng | Tốc độ | Tiếng Việt |
|---|---|---|---|
| **DBNet (PaddleOCR)** | Khuyến nghị — Differentiable Binarization | Nhanh | Có |
| CRAFT | Character region awareness | Trung bình | Cần fine-tune |
| EAST | Cũ, đơn giản | Nhanh | Có nhưng kém |

---

## 3. Kiến trúc đề xuất

```
Sau Step 3 (ảnh dewarped + enhanced)
        │
        ▼
┌───────────────────────────────────────────────┐
│ Step 4a: Block Detection (DocLayout-YOLO)     │
│  → Output: list of (block_type, bbox)         │
│  Classes: title, text, list, table, figure    │
└──────────────────┬────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────┐
│ Step 4b: Line Detection (DBNet)               │
│  - Chạy DBNet trong từng text block           │
│  - Skip table & figure blocks                 │
│  → Output: list of (line_bbox, polygon)       │
└──────────────────┬────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────┐
│ Step 4c: Reading Order (XY-cut hoặc rule)    │
│  - Sắp xếp blocks theo thứ tự đọc tự nhiên   │
│  - Trong block: line theo top-to-bottom      │
└──────────────────┬────────────────────────────┘
                   │
                   ▼
        JSON cấu trúc + ảnh visualize
```

---

## 4. Module mới

### 4.1. Cấu trúc thư mục

```
Pipeline With ML/
├── step4_layout/                  # MỚI
│   ├── __init__.py
│   ├── block_detector.py          # DocLayout-YOLO wrapper
│   ├── line_detector.py           # DBNet wrapper
│   ├── reading_order.py           # XY-cut sorting
│   └── layout_visualizer.py       # Vẽ kết quả lên ảnh
├── models/
│   ├── doclayout_yolov10n.pt      # MỚI - tải về (~20MB)
│   └── (PaddleOCR tự cache DBNet)
```

### 4.2. Khung API

```python
# Pipeline With ML/step4_layout/__init__.py
from .block_detector import BlockDetector
from .line_detector import LineDetector
from .reading_order import sort_reading_order

class LayoutAnalyzer:
    def __init__(
        self,
        block_model_path: str = "models/doclayout_yolov10n.pt",
        use_paddle_dbnet: bool = True,
        lang: str = "vi",
    ): ...

    def analyze(self, image_bgr: np.ndarray) -> dict:
        """
        Returns:
            {
                "image_size": [H, W],
                "blocks": [
                    {
                        "id": 0,
                        "type": "title" | "text" | "list" | "table" | "figure",
                        "bbox": [x, y, w, h],
                        "confidence": 0.95,
                        "lines": [
                            {"bbox": [...], "polygon": [...]}
                        ]
                    },
                    ...
                ],
                "reading_order": [0, 2, 1, 3]  # block IDs theo thứ tự đọc
            }
        """
```

### 4.3. Tích hợp vào pipeline chính

Trong [Pipeline With ML/main.py](Pipeline%20With%20ML/main.py), sau Step 3:

```python
# Sau enhancer.enhance(...)
if self.use_layout:
    from step4_layout import LayoutAnalyzer
    if self.layout_analyzer is None:
        self.layout_analyzer = LayoutAnalyzer()
    layout = self.layout_analyzer.analyze(enhanced)
    result['layout'] = layout
```

Thêm CLI flag `--layout` và endpoint `POST /api/analyze-layout` trong Backend/api.py.

---

## 5. Các bước triển khai (10 bước)

| # | Việc | File | Ước tính |
|---|------|------|---|
| 1 | Cài deps: `paddlepaddle`, `paddleocr`, `doclayout_yolo` | requirements.txt | 15' |
| 2 | Tạo skeleton `step4_layout/` + `__init__.py` | mới | 10' |
| 3 | Implement `BlockDetector` (DocLayout-YOLO) | mới | 1h |
| 4 | Implement `LineDetector` (DBNet via PaddleOCR) | mới | 1h |
| 5 | Implement `reading_order.py` (XY-cut algorithm) | mới | 1.5h |
| 6 | Implement `layout_visualizer.py` | mới | 30' |
| 7 | Tích hợp `LayoutAnalyzer` vào main.py + CLI flag | sửa main.py | 30' |
| 8 | Thêm endpoint `/api/analyze-layout` | sửa Backend/api.py | 30' |
| 9 | Tải `doclayout_yolov10n.pt` về `models/` | script tải | 10' |
| 10 | Test trên 10 ảnh thuộc 3 dạng: 1 cột, nhiều cột, có bảng | data/input | 1h |

**Tổng**: ~7 giờ.

---

## 6. Dependencies bổ sung

Thêm vào `requirements.txt`:

```
# Layout analysis
paddlepaddle==2.6.1       # CPU version (~500MB)
paddleocr==2.7.3          # bao gồm DBNet
doclayout-yolo>=0.0.1     # block detector
```

**Lưu ý**: PaddlePaddle khá nặng. Nếu muốn nhẹ hơn → thay bằng `onnxruntime` + DBNet ONNX export thủ công (giảm ~300MB).

---

## 7. Reading order: XY-cut algorithm

Đây là phần đặc thù — viết riêng vì không có thư viện nhỏ gọn nào làm sẵn:

```python
def sort_reading_order(blocks: list, image_size: tuple) -> list:
    """
    XY-cut đệ quy: chia ảnh theo gap ngang/dọc lớn nhất.
    1. Tìm gap ngang lớn nhất → chia thành nhóm hàng
    2. Trong mỗi hàng, tìm gap dọc lớn nhất → chia thành cột
    3. Đệ quy
    4. Trả về thứ tự ID block
    """
```

Phù hợp với layout phổ biến (1-3 cột). Với layout phức tạp (báo, magazine) cần dùng GNN — ngoài phạm vi.

---

## 8. Đánh giá

| Metric | Cách đo | Mục tiêu |
|---|---|---|
| Block detection mAP | So với annotation thủ công 30 ảnh | ≥ 0.85 @ IoU 0.5 |
| Line detection precision/recall | Tỷ lệ dòng đúng | P ≥ 0.92, R ≥ 0.88 |
| Reading order accuracy | Tỷ lệ thứ tự đúng (Kendall tau) | ≥ 0.9 |
| Thời gian | Trung bình ảnh A4 (1500×2100) trên CPU M1 | ≤ 2s |

Dataset đánh giá đề xuất:
- **PubLayNet val** (tự nhiên, học thuật)
- **30 ảnh tự annotate** (hoá đơn, văn bản hành chính tiếng Việt)

---

## 9. Use case sau khi hoàn thành

1. **OCR có cấu trúc**: feed từng line vào VietOCR/Tesseract → giữ được layout
2. **Trích xuất hoá đơn**: phân biệt header / item list / total
3. **Export sang Markdown**: tự sinh tài liệu structured
4. **PDF có thể tìm kiếm**: gắn text vào đúng vị trí

---

## 10. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| PaddlePaddle khó cài trên M1 Mac (ARM) | Dùng `paddlepaddle` arm64 wheel, hoặc Docker |
| Block detector miss bảng phức tạp | Thêm rule fallback: gom các line cùng baseline |
| Reading order sai với layout đa cột không cân | Cho phép user override qua UI |
| Model size lớn ảnh hưởng mobile (DocScannerMobile) | Tách layout API server-side, mobile chỉ gọi |
| Tiếng Việt có dấu khó cho DBNet pretrained EN | Test trước; nếu kém → fine-tune trên Vietnamese Receipt Dataset |

---

## 11. Câu hỏi cần chốt trước khi code

1. **Layout có cần chạy trên mobile (DocScannerMobile) hay chỉ server-side?**
   - Nếu mobile → bắt buộc dùng model nhỏ (DocLayout-YOLO nano + ONNX)
2. **Có chạy luôn OCR ở Step 4 hay chỉ phát hiện vùng (để OCR ở Step 5 riêng)?**
3. **Output format**: JSON, hOCR (HTML), hay ALTO XML?
4. **Phạm vi tài liệu chính**: học thuật (PubLayNet phù hợp) hay hoá đơn tiếng Việt (cần data riêng)?

---

## Liên quan

- [Kế hoạch Ensemble YOLO + U2Net](KeHoach_Ensemble_YOLO_U2Net.md) — chạy trước Step 4 này
- [Pipeline Workflow hiện tại](PIPELINE_WORKFLOW.md)
- [Thách thức ảnh chụp tài liệu](ThachThuc_AnhChup_TaiLieu.md)
