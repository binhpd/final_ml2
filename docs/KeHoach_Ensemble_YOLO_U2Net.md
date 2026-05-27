# Kế hoạch: Ensemble YOLOv8-seg + U2Net cho Background Removal

> Mục tiêu: Xây dựng pipeline 2 tầng kết hợp YOLOv8-seg (detection thô) và U2Net (refine mask), thay thế cách dùng độc lập hiện tại, để tách nền tài liệu chính xác hơn trong các điều kiện khó (nền lộn xộn, tay che, thiếu sáng).

---

## 1. Tình trạng hiện tại

| Thành phần | Vị trí | Vai trò | Vấn đề |
|---|---|---|---|
| `YOLOSegmentor` | [Pipeline With ML/step1_ml_segmentor.py:64](Pipeline%20With%20ML/step1_ml_segmentor.py#L64) | Pre-trained COCO, tìm class `book` (id=73) | False negative nhiều khi tài liệu không phải sách (hoá đơn, A4) |
| U2-Net (qua `rembg`) | [Pipeline With ML/main.py:133-186](Pipeline%20With%20ML/main.py#L133-L186) | SOTA background removal | Chậm (~1-2s/ảnh CPU), không có guidance vùng quan tâm |
| Cascading fallback | [Pipeline With ML/main.py](Pipeline%20With%20ML/main.py) | approxPolyDP → Hough → YOLO HOẶC U2Net | Hai model ML chạy **độc lập**, không bổ trợ cho nhau |

**Hạn chế quan trọng**: U2Net chạy trên *toàn ảnh* → bị nhiễu khi nền có vật thể có cấu trúc tương tự giấy (bàn trắng, tường). YOLO ngược lại, khoanh được vùng nhưng mask rìa thô.

---

## 2. Kiến trúc ensemble đề xuất

```
                  ┌──────────────────────────────────────┐
   Ảnh gốc ───►   │  Tầng 1: YOLOv8-seg (Detection thô) │
                  │  - Tìm bounding box + mask thô        │
                  │  - Class: book/paper/document         │
                  └──────────────┬───────────────────────┘
                                 │ ROI + mask thô
                                 ▼
                  ┌──────────────────────────────────────┐
                  │  Tầng 2: U2Net (Refinement)         │
                  │  - Crop ROI mở rộng (padding 15%)    │
                  │  - Inference trên crop (nhanh hơn)   │
                  │  - Output: alpha mask tinh           │
                  └──────────────┬───────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────────────┐
                  │  Tầng 3: Fusion + Post-process       │
                  │  - Weighted fusion 2 mask            │
                  │  - Morphology cleanup                │
                  │  - Trích 4 góc qua approxPolyDP      │
                  └──────────────┬───────────────────────┘
                                 │
                                 ▼
                       Mask cuối + 4 góc
```

### Logic fusion

```
mask_final = α · mask_yolo + (1-α) · mask_u2net   với α = 0.3
mask_final = morphology_open(mask_final, kernel=5)
mask_final = morphology_close(mask_final, kernel=11)
```

Lý do: U2Net cho biên rìa tốt hơn (trọng số cao), YOLO định vị vùng (trọng số thấp nhưng giúp loại trừ noise xa).

---

## 3. Các module mới

### 3.1. File chính

| File mới | Mục đích |
|---|---|
| `Pipeline With ML/step1_ensemble_segmentor.py` | Class `EnsembleSegmentor` thực hiện 3 tầng trên |
| `Pipeline With ML/utils/mask_fusion.py` | Hàm tiện ích: `fuse_masks()`, `expand_roi()`, `mask_to_corners()` |

### 3.2. Khung class

```python
# Pipeline With ML/step1_ensemble_segmentor.py
class EnsembleSegmentor:
    def __init__(
        self,
        yolo_path: str = "models/yolov8n-seg.pt",
        u2net_session_name: str = "u2net",   # rembg session
        roi_padding: float = 0.15,
        fusion_alpha: float = 0.3,
        device: str = "cpu",
    ): ...

    def detect(self, image_bgr: np.ndarray) -> dict:
        """
        Returns:
            {
                'mask': np.ndarray (H,W) uint8 0/255,
                'corners': np.ndarray (4,2) float32 | None,
                'method': 'ensemble' | 'yolo_only' | 'u2net_only',
                'confidence': float,
                'debug': {...}
            }
        """
        # 1. YOLO inference
        # 2. Nếu YOLO có mask → crop ROI mở rộng → chạy U2Net trên crop
        # 3. Nếu YOLO miss → fallback U2Net trên toàn ảnh
        # 4. Fusion + extract corners
```

### 3.3. Tích hợp vào `DocumentDetector`

Trong [Pipeline With ML/main.py:40](Pipeline%20With%20ML/main.py#L40), thêm flag mới `use_ensemble=True` (mặc định bật khi cả YOLO + U2Net khả dụng):

```python
def __init__(..., use_ensemble=False, ...):
    ...
    self.ensemble_segmentor = EnsembleSegmentor(...) if use_ensemble else None
```

Chèn vào pipeline **trước** nhánh `use_u2net` hiện tại (ưu tiên ensemble nếu bật).

---

## 4. Các bước triển khai (8 bước)

| # | Việc | File | Ước tính |
|---|------|------|---|
| 1 | Tạo `utils/mask_fusion.py` với 3 hàm tiện ích | mới | 30' |
| 2 | Implement `EnsembleSegmentor.__init__` + lazy load | mới | 20' |
| 3 | Implement `_yolo_detect()`: trả mask thô + bbox | mới | 30' |
| 4 | Implement `_u2net_refine()`: chạy rembg trên ROI crop | mới | 30' |
| 5 | Implement `detect()` orchestration + fusion | mới | 45' |
| 6 | Thêm flag `--ensemble` vào CLI `main.py` | sửa main.py | 20' |
| 7 | Thêm endpoint param `ensemble: bool` vào FastAPI | sửa Backend/api.py | 15' |
| 8 | Test trên 5-10 ảnh mẫu, so sánh với baseline | data/input | 1h |

**Tổng**: ~4 giờ (chưa kể fine-tune YOLO nếu cần — xem mục 6).

---

## 5. Dependencies bổ sung

Hiện `requirements.txt` đã có `ultralytics` và `rembg`. Cần thêm:

```
# Pipeline With ML/requirements.txt (nếu chưa có)
onnxruntime>=1.16  # rembg backend, nhanh hơn torch CPU
pillow>=10.0       # rembg dependency
```

Không cần thêm thư viện mới ngoài 2 dòng trên.

---

## 6. Tuỳ chọn nâng cao (giai đoạn 2)

### 6.1. Fine-tune YOLOv8-seg trên dataset tài liệu
- Dataset gợi ý: [MIDV-500](https://arxiv.org/abs/1807.05786), [SmartDoc-QA](http://smartdoc.univ-lr.fr/), [DocUNet](https://www3.cs.stonybrook.edu/~cvl/projects/dewarpnet/)
- Lý do: pre-trained COCO chỉ có class `book` → bỏ sót A4 trắng, hoá đơn
- Output: `models/yolov8n-seg-doc.pt` (custom)
- Lệnh: `yolo segment train data=docs.yaml model=yolov8n-seg.pt epochs=50`

### 6.2. Thay U2Net bằng U2Net-portrait (nhẹ hơn ~5x)
- `rembg(session_name="u2netp")` thay vì `"u2net"`
- Đánh đổi: nhanh hơn nhưng rìa kém tinh

### 6.3. Confidence-based gating
- Nếu YOLO conf > 0.85 và mask area > 60% ROI → skip U2Net (tăng tốc)

---

## 7. Cách đánh giá

| Metric | Cách đo | Baseline (hiện tại) | Mục tiêu |
|---|---|---|---|
| IoU mask vs ground truth | So mask cuối với `data/ground_truth/` | ~0.85 (U2Net riêng) | ≥ 0.92 |
| Tỷ lệ tìm đủ 4 góc | Đếm trên 50 ảnh test | ~75% | ≥ 90% |
| Thời gian inference | Trung bình 10 lần (CPU M1) | ~1.8s | ≤ 1.5s (nhờ crop ROI) |
| Robust với tay che | Test trên 10 ảnh có tay | Fail ~40% | Fail ≤ 15% |

Script benchmark đề xuất: `Pipeline With ML/benchmark_ensemble.py`.

---

## 8. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| YOLO miss tài liệu → ROI sai → U2Net cũng sai | Fallback U2Net toàn ảnh khi YOLO conf < 0.4 |
| ROI quá hẹp khi tài liệu sát mép | Padding 15% + clip vào ảnh |
| Tốc độ chậm hơn vì 2 lần inference | Crop ROI giảm input U2Net từ ảnh gốc xuống ~30% diện tích |
| Mask fusion sai hướng (α không phù hợp) | A/B test α ∈ {0.2, 0.3, 0.4, 0.5} trên tập val |

---

## 9. Câu hỏi cần chốt trước khi code

1. **Có cần fine-tune YOLO không, hay dùng pre-trained?** (mục 6.1)
2. **Có dataset ground truth mask sẵn trong `data/ground_truth/` không?** (đang rỗng) → nếu không, cần tự annotate ~30 ảnh.
3. **Mục tiêu deploy**: CPU only (mobile-friendly) hay có GPU?
4. **Tích hợp luôn vào API hay tạo endpoint riêng `/api/scan-ensemble`?**
