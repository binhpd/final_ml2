# Phase 3 + 4 — Integration vào Pipeline + KPI Benchmark

> **Thời gian:** Phase 3 (Tuần 9-10) + Phase 4 (Tuần 11-12)
> **Mục tiêu:** Tích hợp 2 model vào pipeline hiện có, đo KPI 4 chiều, viết báo cáo so sánh

---

# PHẦN A — PHASE 3: Integration vào Pipeline (Tuần 9-10)

## A.1 Phân tích pipeline hiện có

Trong `Pipeline With ML/main.py`, luồng U2NET hiện gọi qua `rembg`:

```python
# Code hiện tại (main.py line ~135-180)
if self.use_u2net:
    from rembg import remove
    subject_orig = remove(orig)  # ← Thay bằng U2NET tự build / YOLO
    alpha_orig = subject_orig[:, :, 3]
    
    contours, _ = cv2.findContours(alpha_orig, ...)
    # ... extract 4 corners ...
    result['u2net_doc'] = pure_doc
    result['u2net_mask'] = alpha_orig
    result['corners'] = corners
```

→ **Điểm thay thế:** Function `remove()` từ `rembg` được thay bằng wrapper riêng.

---

## A.2 Integration U2NET (Tuần 9)

### A.2.1 Strategy: Giữ nguyên pipeline cũ

Chỉ thay function `remove()` trong logic `--u2net`. Mọi thứ khác (corner extraction, perspective transform, Step 3 enhancement) giữ nguyên.

### A.2.2 Module wrapper

```python
# ml2/pipeline_integration/u2net_wrapper.py
import torch
import cv2
import numpy as np
import sys
sys.path.append("ml2")
from u2net.model import U2NETp

class U2NETSegmentor:
    """
    Wrapper U2NET tự build, drop-in replacement cho rembg.remove().
    """
    def __init__(self, weights_path="ml2/u2net/weights/u2netp_final.pth"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = U2NETp(in_ch=3, out_ch=1)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.input_size = 320
        print(f"[U2NET] Loaded weights from {weights_path}")
    
    def segment(self, image_bgr):
        """
        Input: ảnh BGR (numpy)
        Output: ảnh BGRA (numpy) — A channel là mask
        """
        h, w = image_bgr.shape[:2]
        
        # Preprocess
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        img_norm = (img_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        
        # Inference
        with torch.no_grad():
            d0, *_ = self.model(img_tensor)
        
        # Postprocess
        pred = d0.squeeze().cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-7)
        mask = (pred > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask, (w, h))
        
        # Build BGRA (compatible with rembg output)
        bgra = np.concatenate([image_bgr, mask_resized[:, :, np.newaxis]], axis=2)
        return bgra
```

### A.2.3 Modify main.py

```python
# Patch trong main.py
# Thay:
#   from rembg import remove
#   subject_orig = remove(orig)
# Bằng:

from ml2.pipeline_integration.u2net_wrapper import U2NETSegmentor

# Trong __init__ của DocumentDetector
self.u2net_segmentor = U2NETSegmentor() if use_u2net else None

# Trong detect()
if self.use_u2net:
    print(f"[1e] 👑 Bắt đầu bóc nền bằng U²-Net tự build...")
    subject_orig = self.u2net_segmentor.segment(orig)  # ← Thay đổi
    alpha_orig = subject_orig[:, :, 3]
    # ... phần còn lại y nguyên ...
```

### A.2.4 Tạo file pipeline mới

Để giữ pipeline cũ làm baseline, tạo bản copy:

```bash
cp "Pipeline With ML/main.py" "ml2/pipeline_integration/pipeline_u2net.py"
# Sau đó modify pipeline_u2net.py theo A.2.3
```

### A.2.5 Test trên 1020 ảnh

```python
# ml2/pipeline_integration/test_u2net_pipeline.py
from pipeline_u2net import DocumentDetector
import glob, time, cv2

detector = DocumentDetector(use_u2net=True)
results = []

for img_path in glob.glob("ml2/datasets/nhom6_1020/images/*.jpg"):
    t0 = time.time()
    image = cv2.imread(img_path)
    result = detector.detect(image)
    t1 = time.time()
    
    results.append({
        "image": img_path,
        "method": result['method'],
        "has_corners": result['corners'] is not None,
        "time_ms": (t1 - t0) * 1000,
    })

# Báo cáo:
import pandas as pd
df = pd.DataFrame(results)
print(f"Tỷ lệ tìm thấy corners: {df['has_corners'].mean()*100:.1f}%")
print(f"Latency trung bình: {df['time_ms'].mean():.1f}ms")
```

### A.2.6 Acceptance criteria Integration U2NET

- ✅ Pipeline chạy không lỗi trên 1020 ảnh
- ✅ Tỷ lệ phát hiện corners ≥ 95% (so với rembg ~93%)
- ✅ Latency Step 1 (U2NET): ≤ 100ms trên M2 MPS
- ✅ Visual output identical/better than rembg version trên 20 ảnh sample

---

## A.3 Integration YOLO + Visualization (Tuần 10)

### A.3.1 Strategy: Thay U2NET hoàn toàn + thêm Visualization

```python
# ml2/pipeline_integration/yolo_wrapper.py
import sys
sys.path.append("ml2")
from yolo_seg.visualize import YOLODocVisualizer
from ultralytics import YOLO
import cv2
import numpy as np

class YOLODocSegmentor:
    """
    Wrapper YOLO-Seg, replacement cho U2NET trong pipeline.
    Thêm visualization và corner extraction.
    """
    def __init__(self, weights_path="ml2/yolo_seg/weights/yolo11n_seg_doc.pt"):
        self.model = YOLO(weights_path)
        self.visualizer = YOLODocVisualizer()
        print(f"[YOLO] Loaded weights from {weights_path}")
    
    def segment_and_visualize(self, image_bgr, save_viz_path=None):
        """
        Returns:
            mask: binary mask H×W
            corners: 4 corners array (4,2) or None
            bbox: (x1, y1, x2, y2) or None
            confidence: float
            annotated_image: image with viz drawings (if save_viz_path provided)
        """
        results = self.model(image_bgr, verbose=False)
        
        if len(results[0].boxes) == 0:
            return None, None, None, 0, image_bgr
        
        # Extract mask
        if results[0].masks is not None:
            mask_data = results[0].masks.data[0].cpu().numpy()
            h, w = image_bgr.shape[:2]
            mask = cv2.resize(mask_data, (w, h))
            mask_bin = (mask > 0.5).astype(np.uint8) * 255
        else:
            mask_bin = None
        
        # Extract bbox
        bbox = results[0].boxes[0].xyxy[0].cpu().numpy()
        confidence = float(results[0].boxes[0].conf[0])
        
        # Extract corners
        corners = self.visualizer._extract_corners(mask_bin) if mask_bin is not None else None
        
        # Visualization
        annotated = self.visualizer.draw_results(image_bgr, results)
        
        if save_viz_path:
            cv2.imwrite(save_viz_path, annotated)
        
        return mask_bin, corners, bbox, confidence, annotated
```

### A.3.2 Pipeline mới hoàn chỉnh

```python
# ml2/pipeline_integration/pipeline_yolo.py
"""
Pipeline mới: YOLO thay U2NET + Visualization
Mọi bước Step 2, Step 3 giữ nguyên từ Pipeline With ML/
"""

import os, sys
sys.path.append("Pipeline With ML")  # Import các module gốc

from step1_preprocessor import Preprocessor
from step2_perspective_transform import PerspectiveTransformer
from step2_uvdoc_dewarper import UVDocDewarper
from step3_enhancer import DocumentEnhancer
from corner_sorter import sort_corners

from ml2.pipeline_integration.yolo_wrapper import YOLODocSegmentor


class YOLOPipeline:
    def __init__(self, use_uvdoc=False):
        self.yolo = YOLODocSegmentor()
        self.preprocessor = Preprocessor()
        
        baseline_transformer = PerspectiveTransformer()
        if use_uvdoc:
            self.transformer = UVDocDewarper(fallback_transformer=baseline_transformer)
        else:
            self.transformer = baseline_transformer
            
        self.enhancer = DocumentEnhancer()
    
    def process(self, image_bgr, save_dir=None):
        """End-to-end process với YOLO + visualization."""
        os.makedirs(save_dir, exist_ok=True) if save_dir else None
        
        # ─── Step 1: YOLO Segmentation + Visualization ───
        viz_path = f"{save_dir}/01_yolo_detection.jpg" if save_dir else None
        mask, corners, bbox, conf, annotated = self.yolo.segment_and_visualize(
            image_bgr, save_viz_path=viz_path
        )
        
        if corners is None:
            print("[Pipeline] YOLO không phát hiện tài liệu")
            return None
        
        # Sort corners chuẩn TL-TR-BR-BL
        corners = sort_corners(corners)
        
        # Apply mask (xoá nền thành trắng)
        white_bg = np.ones_like(image_bgr) * 255
        mask_f = mask[:, :, np.newaxis] / 255.0
        masked_doc = (image_bgr * mask_f + white_bg * (1 - mask_f)).astype(np.uint8)
        
        if save_dir:
            cv2.imwrite(f"{save_dir}/02_masked.jpg", masked_doc)
        
        # ─── Step 2: Perspective / Dewarping ───
        warped = self.transformer.transform(masked_doc, corners,
                                             save_prefix=f"{save_dir}/03" if save_dir else None)
        if save_dir:
            cv2.imwrite(f"{save_dir}/04_warped.jpg", warped)
        
        # ─── Step 3: Enhancement ───
        final = self.enhancer.enhance(warped,
                                       save_prefix=f"{save_dir}/05" if save_dir else None,
                                       mode="color")
        
        if save_dir:
            cv2.imwrite(f"{save_dir}/06_final.jpg", final)
        
        return {
            "annotated": annotated,
            "mask": mask,
            "corners": corners,
            "bbox": bbox,
            "confidence": conf,
            "masked": masked_doc,
            "warped": warped,
            "final": final,
        }
```

### A.3.3 Acceptance criteria Integration YOLO

- ✅ Pipeline chạy end-to-end không lỗi
- ✅ Visualization hiển thị đủ: bbox + mask overlay + 4 corners + FPS
- ✅ Mỗi ảnh output 6 file (detection, masked, perspective marked, warped, enhance steps, final)
- ✅ FPS overall ≥ 5 (chậm nhất ở Step 3) — Step 1 YOLO ≥ 20

---

# PHẦN B — PHASE 4: KPI Benchmark Protocol (Tuần 11-12)

## B.1 4 Chiều KPI

```
                    ┌─────────────────────────┐
                    │   KPI BENCHMARK         │
                    │                         │
            ┌───────┼─────────────────┐       │
            │       │                 │       │
    ┌───────▼──┐ ┌─▼────────┐ ┌─────▼────┐ ┌▼──────────┐
    │ B.2      │ │ B.3      │ │ B.4      │ │ B.5       │
    │ SPEED    │ │ ACCURACY │ │ ROBUST   │ │ E2E       │
    │ FPS, ms  │ │ mAP, IoU │ │ Per-cat  │ │ Pipeline  │
    └──────────┘ └──────────┘ └──────────┘ └───────────┘
```

---

## B.2 Speed KPIs

### B.2.1 Setup chuẩn đo

- **Hardware:** Mac M2 (hoặc M1/M3, ghi rõ)
- **Backends:** CPU, MPS, CoreML (xuất từ ONNX)
- **Warmup:** 10 ảnh trước khi đo
- **Sample:** 100 ảnh test set
- **Metric chính:** Median latency (robust hơn mean)

### B.2.2 Code

```python
# ml2/benchmark/kpi_speed.py
import time, torch, cv2
import numpy as np
from tqdm import tqdm

def benchmark_speed(model, images, device, n_warmup=10):
    """Trả về median, mean, p95 latency."""
    latencies = []
    
    for i, img in enumerate(images):
        if i < n_warmup:
            _ = model(img)
            continue
        
        t0 = time.perf_counter()
        _ = model(img)
        if device == "mps":
            torch.mps.synchronize()  # Quan trọng để đo chính xác
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms
    
    return {
        "median_ms": np.median(latencies),
        "mean_ms": np.mean(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "fps": 1000 / np.median(latencies),
        "n_samples": len(latencies),
    }
```

### B.2.3 Bảng kết quả mong đợi

| Model | Backend | Median (ms) | FPS | p95 (ms) |
|---|---|---|---|---|
| U2NET (rembg) | CPU | ~250 | 4.0 | ~280 |
| U2NET (rembg) | MPS | ~120 | 8.3 | ~140 |
| **U2NET (tự build)** | CPU | ~180 | 5.5 | ~200 |
| **U2NET (tự build)** | MPS | ~50 | 20 | ~65 |
| **U2NET (tự build)** | CoreML | ~35 | 28 | ~45 |
| **YOLOv11n-seg** | CPU | ~80 | 12.5 | ~95 |
| **YOLOv11n-seg** | MPS | ~25 | 40 | ~30 |
| **YOLOv11n-seg** | CoreML | ~18 | 55 | ~22 |

---

## B.3 Accuracy KPIs

### B.3.1 Test set chung

100 ảnh **verified mask** từ subset của 1020 ảnh + 100 SmartDoc-QA = **200 ảnh**.

### B.3.2 4 Metrics chung cho cả 2 model

```python
# ml2/benchmark/kpi_accuracy.py
def compare_models(u2net_model, yolo_model, test_set):
    """Đánh giá song song trên cùng test set."""
    results = {"U2NET": {}, "YOLO": {}}
    
    for name, model in [("U2NET", u2net_model), ("YOLO", yolo_model)]:
        ious, f1s, maes, bfs = [], [], [], []
        
        for img_path, gt_mask in test_set:
            img = cv2.imread(img_path)
            pred_mask = model.predict(img)
            
            ious.append(compute_iou(pred_mask, gt_mask))
            f1s.append(compute_f1(pred_mask, gt_mask))
            maes.append(compute_mae(pred_mask, gt_mask))
            bfs.append(compute_boundary_f1(pred_mask, gt_mask))
        
        results[name] = {
            "mIoU": np.mean(ious),
            "F1": np.mean(f1s),
            "MAE": np.mean(maes),
            "BF": np.mean(bfs),
        }
    
    return results
```

### B.3.3 YOLO bổ sung Detection metrics

```python
# Chỉ YOLO có
yolo_metrics = {
    "mAP@0.5_box": ...,
    "mAP@0.5:0.95_box": ...,
    "mAP@0.5_mask": ...,
    "precision": ...,
    "recall": ...,
}
```

### B.3.4 Bảng kết quả mong đợi

| Metric | U2NET (tự build) | YOLOv11n-seg |
|---|---|---|
| mIoU | 0.85 | 0.83 |
| F1 (Dice) | 0.88 | 0.87 |
| MAE | 0.04 | 0.05 |
| Boundary F1 | 0.78 | 0.74 |
| mAP@0.5 (box) | N/A | 0.91 |
| mAP@0.5 (mask) | N/A | 0.85 |

→ **Nhận xét điển hình:** U2NET thắng về boundary precision, YOLO thắng về tổng thể detection + tốc độ.

---

## B.4 Robustness KPIs (theo 7 nhóm dataset)

### B.4.1 Mục tiêu

Đo accuracy của 2 model trên từng category của 1020 ảnh:

```python
CATEGORIES = ["Curved", "Fold", "Incomplete", "Perspective", "Rotate", "Random", "Normal"]
```

### B.4.2 Code

```python
# ml2/benchmark/kpi_robustness.py
def robustness_per_category(model, test_root="ml2/datasets/nhom6_1020"):
    results = {}
    for cat in CATEGORIES:
        imgs = glob(f"{test_root}/images/{cat}/*.jpg")
        gts = [cv2.imread(p.replace("images", "verified_masks"), 0) for p in imgs]
        
        ious = []
        for img_path, gt in zip(imgs, gts):
            if gt is None:
                continue
            img = cv2.imread(img_path)
            pred = model.predict(img)
            ious.append(compute_iou(pred, gt))
        
        results[cat] = {
            "mIoU": np.mean(ious),
            "n_samples": len(ious),
            "min_iou": np.min(ious),
            "max_iou": np.max(ious),
        }
    return results
```

### B.4.3 Bảng kết quả mong đợi

| Category | U2NET mIoU | YOLO mIoU | Winner |
|---|---|---|---|
| Normal | 0.92 | 0.90 | U2NET |
| Perspective | 0.88 | 0.89 | YOLO |
| Rotate | 0.85 | 0.91 | YOLO |
| Curved | 0.79 | 0.78 | U2NET |
| Fold | 0.82 | 0.81 | ~tie |
| Incomplete | 0.71 | 0.83 | YOLO |
| Random (nhàu) | 0.74 | 0.72 | U2NET |

→ **Insight cho báo cáo:** U2NET tốt hơn cho ảnh nhàu/cong (semantic), YOLO tốt hơn cho thiếu góc/xoay (vì có bbox khôi phục context).

---

## B.5 End-to-End Pipeline Impact KPI

### B.5.1 Mục tiêu

Đo chất lượng ảnh **cuối cùng** của pipeline (sau Step 3) khi dùng U2NET vs YOLO:

```
Input → [U2NET] → Step 2 → Step 3 → Output_A
Input → [YOLO]  → Step 2 → Step 3 → Output_B

So sánh Output_A vs Output_B
```

### B.5.2 Metrics

| Metric | Cách đo |
|---|---|
| **PSNR** | So với ground truth scan (nếu có) |
| **SSIM** | Structural similarity với GT |
| **OCR-CER** | Tesseract OCR → Character Error Rate vs ground truth text |
| **Pipeline total time** | End-to-end latency |
| **Visual quality (subjective)** | Đánh giá 5-điểm bởi 3 người |

### B.5.3 OCR-CER là metric quan trọng nhất

```python
# ml2/benchmark/kpi_e2e.py
import pytesseract
from difflib import SequenceMatcher

def char_error_rate(pred_text, gt_text):
    matcher = SequenceMatcher(None, pred_text, gt_text)
    ratio = matcher.ratio()
    return 1 - ratio

def evaluate_e2e(pipeline_u2net, pipeline_yolo, test_imgs, gt_texts):
    cers_u, cers_y = [], []
    
    for img_path, gt_text in zip(test_imgs, gt_texts):
        img = cv2.imread(img_path)
        
        out_u = pipeline_u2net.process(img)
        out_y = pipeline_yolo.process(img)
        
        text_u = pytesseract.image_to_string(out_u['final'], lang='vie')
        text_y = pytesseract.image_to_string(out_y['final'], lang='vie')
        
        cers_u.append(char_error_rate(text_u, gt_text))
        cers_y.append(char_error_rate(text_y, gt_text))
    
    return {
        "U2NET": {"CER": np.mean(cers_u)},
        "YOLO": {"CER": np.mean(cers_y)},
    }
```

### B.5.4 Bảng kết quả mong đợi

| Metric | Pipeline gốc (rembg) | Pipeline U2NET tự build | Pipeline YOLO |
|---|---|---|---|
| PSNR (vs GT scan) | 22.5 | 23.1 | 22.8 |
| SSIM | 0.81 | 0.84 | 0.82 |
| **OCR-CER (Vietnamese)** | 8.5% | **7.2%** | 7.8% |
| End-to-end time | 4.2s | 3.5s | 2.9s |

---

## B.6 Comprehensive Comparison Table

Bảng tổng hợp cuối cùng cho báo cáo:

```
═══════════════════════════════════════════════════════════════════
                          U2NET tự build   |   YOLOv11n-seg
═══════════════════════════════════════════════════════════════════
SPEED
  CPU (M2)                180 ms          |    80 ms
  MPS                      50 ms          |    25 ms
  CoreML                   35 ms          |    18 ms
  FPS (MPS)               20.0            |    40.0
═══════════════════════════════════════════════════════════════════
ACCURACY (200 verified test)
  mIoU                    0.85            |    0.83
  F1                      0.88            |    0.87
  MAE                     0.04            |    0.05
  Boundary F1             0.78            |    0.74
  mAP@0.5 (box)           N/A             |    0.91
  mAP@0.5 (mask)          N/A             |    0.85
═══════════════════════════════════════════════════════════════════
ROBUSTNESS (mIoU per category)
  Normal                  0.92            |    0.90
  Perspective             0.88            |    0.89
  Rotate                  0.85            |    0.91
  Curved                  0.79            |    0.78
  Fold                    0.82            |    0.81
  Incomplete              0.71            |    0.83
  Random                  0.74            |    0.72
═══════════════════════════════════════════════════════════════════
E2E PIPELINE IMPACT
  PSNR                    23.1            |    22.8
  SSIM                    0.84            |    0.82
  OCR-CER (VN)            7.2%            |    7.8%
  Total pipeline (s)      3.5             |    2.9
═══════════════════════════════════════════════════════════════════
RESOURCES
  Model size              4.7 MB          |    6.2 MB
  Params                  1.1 M           |    2.9 M
  Memory peak (MPS)       1.2 GB          |    0.8 GB
═══════════════════════════════════════════════════════════════════
```

---

## B.7 Báo cáo so sánh — Cấu trúc đề xuất

```markdown
# Báo cáo So sánh U2NET (tự build) vs YOLO (fine-tune) cho Document Segmentation

## 1. Tóm tắt
- U2NET thắng về độ chính xác boundary và mIoU trên ảnh thường (+2-3%)
- YOLO thắng về tốc độ (2-3x), khả năng detect multi-doc, robustness với ảnh thiếu góc
- E2E pipeline với U2NET cho OCR-CER thấp hơn 0.6% (tài liệu sạch)
- E2E pipeline với YOLO nhanh hơn 17%

## 2. Quyết định triển khai
- **Production mobile/realtime**: Dùng YOLO (tốc độ, có viz, multi-doc)
- **Maximum quality (server, batch)**: Dùng U2NET (boundary cao)
- **Best of both**: Cascade — YOLO làm coarse detect → U2NET fine-tune mask

## 3. Hạn chế
- U2NET train from scratch nên thiếu domain knowledge so với DUTS pretrained
- YOLO hiện chỉ có 1 class "document", chưa phân biệt text/figure/table
- Dataset verify chỉ 200 ảnh, statistical power chưa cao
```

---

## B.8 Deliverables Phase 3 + Phase 4

| Output | Mô tả |
|---|---|
| `ml2/pipeline_integration/u2net_wrapper.py` | Module wrapper U2NET |
| `ml2/pipeline_integration/yolo_wrapper.py` | Module wrapper YOLO |
| `ml2/pipeline_integration/pipeline_u2net.py` | Pipeline với U2NET |
| `ml2/pipeline_integration/pipeline_yolo.py` | Pipeline với YOLO + viz |
| `ml2/benchmark/kpi_*.py` | 4 file KPI scripts |
| `ml2/benchmark/results/benchmark.csv` | Bảng số liệu |
| `ml2/benchmark/results/figures/` | Charts + sample visualizations |
| `docs_ml2/report_final.md` | Báo cáo tổng kết so sánh |
| `demo/` | Video screenshot demo viz YOLO |

---

## B.9 Checklist hoàn thành Phase 3 + 4

**Phase 3 — Integration**
- [ ] U2NET wrapper hoàn chỉnh, test pass trên 20 ảnh
- [ ] Pipeline U2NET tích hợp xong, chạy được trên 1020 ảnh
- [ ] YOLO wrapper với visualization hoàn chỉnh
- [ ] Pipeline YOLO + viz, chạy end-to-end 1020 ảnh
- [ ] Visualization output đạt yêu cầu thẩm mỹ (bbox + mask + corners + info)

**Phase 4 — Benchmark**
- [ ] Speed benchmark: CPU + MPS + CoreML cho cả 2 model
- [ ] Accuracy benchmark: mIoU, F1, MAE, BF trên 200 verified test
- [ ] Detection metrics riêng cho YOLO (mAP)
- [ ] Robustness per-category: 7 categories
- [ ] E2E pipeline impact: PSNR, SSIM, OCR-CER, total time
- [ ] Comprehensive comparison table
- [ ] Báo cáo final markdown
- [ ] Visualization figures (8-10 ảnh) cho báo cáo

---

*Phase 3+4 hoàn thành = Cả 2 model đã tích hợp pipeline, có KPI 4 chiều đầy đủ, báo cáo so sánh khoa học.*
