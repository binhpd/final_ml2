# Phase 2 — Build & Train YOLO-Seg cho Document Segmentation + BBox (Chi tiết)

> **Mục tiêu:** Train YOLO-Seg cho task "document detection + segmentation", đạt mAP@0.5 ≥ 0.90, mIoU ≥ 0.87, có visualization module hoàn chỉnh
> **Đầu ra:** Model weights + training logs + visualization + ablation studies

---

## 1. Chiến lược Lựa chọn YOLO Variant

### 1.1 Bảng so sánh các option

| Model | Params | FLOPs (G) | mAP COCO | Phù hợp |
|---|---|---|---|---|
| YOLOv8n-seg | 3.4M | 12.6 | 30.5 | Mobile demo |
| YOLOv8s-seg | 11.8M | 42.6 | 36.8 | Balance |
| YOLOv8m-seg | 27.3M | 110.2 | 40.8 | Server quality |
| YOLOv8l-seg | 46.0M | 220.5 | 42.6 | High accuracy |
| YOLOv11n-seg | 2.9M | 10.4 | 30.5 | **Mobile ⭐** |
| YOLOv11s-seg | 10.1M | 35.5 | 37.8 | **Balance ⭐** |
| YOLOv11m-seg | 22.4M | 123.3 | 41.5 | **Server quality ⭐** |
| YOLOv11l-seg | 27.6M | 142.2 | 42.9 | High accuracy |
| YOLOv11x-seg | 62.1M | 319.0 | 43.8 | **Maximum quality ⭐** |

### 1.2 Strategy đa-variant: Train 3 sizes

Khuyến nghị train **3 sizes** để có comprehensive comparison:

| Variant | Mục đích | Train epochs |
|---|---|---|
| **YOLOv11n-seg** | Mobile / Real-time deployment | 200 |
| **YOLOv11s-seg** | Balance accuracy-speed (chính) | 200 |
| **YOLOv11m-seg** | Server / Maximum quality baseline | 150 |

Bonus: train **YOLOv11n từ scratch** (không dùng COCO pretrained) để so sánh với fine-tune → có thể đóng góp khoa học.

### 1.3 Kiến trúc YOLOv11-seg

```
INPUT (640×640×3 hoặc 1024×1024)
       ▼
┌──────────────────────────────────────────┐
│ Backbone (CSPNet + C3k2 blocks)          │
│   Stem → P1 → P2 → P3 → P4 → P5          │
│   ● C3k2: lightweight cross-stage block  │
│   ● SPPF: spatial pyramid pooling fast   │
└────────────────────┬─────────────────────┘
                     ▼
┌──────────────────────────────────────────┐
│ Neck (PAN-FPN + C2PSA attention)         │
│   ● Top-down + Bottom-up feature fusion  │
│   ● C2PSA: Partial Self-Attention        │
└────────────────────┬─────────────────────┘
                     ▼
┌──────────────────────────────────────────┐
│ Detection Head (cls + bbox + mask coef)  │
│   ● Decoupled head                       │
│   ● Anchor-free (TaskAlignedAssigner)    │
│                                          │
│ Segmentation Head (ProtoNet)             │
│   ● 32 prototype masks                   │
│   ● Linear combination: coef × proto      │
└────────────────────┬─────────────────────┘
                     ▼
OUTPUT (per detection):
   ● bbox (x, y, w, h)
   ● class id + confidence
   ● binary mask
```

---

## 2. Dataset Preparation

### 2.1 YOLO Segmentation format

YOLO-Seg yêu cầu format:
```
[class_id] [x_center] [y_center] [width] [height] [x1 y1 x2 y2 ... xN yN]
```

Trong đó (x_i, y_i) là **polygon vertices** của mask, **normalized 0-1**.

### 2.2 Convert mask → YOLO polygon (chi tiết)

```python
# ml2/yolo_seg/prepare_dataset.py
import cv2, os, glob, json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def mask_to_yolo_polygons(mask, img_w, img_h, class_id=0, max_points=32):
    """
    Convert binary mask → YOLO segmentation labels (có thể có nhiều instance).
    
    Returns: list of strings, mỗi string là 1 detection line.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Lọc nhiễu
            continue
        
        # Simplify polygon
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Đảm bảo đủ points
        if len(approx) < 4:
            continue
        
        # Limit points
        if len(approx) > max_points:
            indices = np.linspace(0, len(approx)-1, max_points, dtype=int)
            approx = approx[indices]
        
        # Bbox
        x, y, w, h = cv2.boundingRect(cnt)
        x_c = (x + w/2) / img_w
        y_c = (y + h/2) / img_h
        w_n = w / img_w
        h_n = h / img_h
        
        # Polygon normalized
        poly_norm = []
        for pt in approx.reshape(-1, 2):
            poly_norm.extend([
                np.clip(pt[0]/img_w, 0, 1),
                np.clip(pt[1]/img_h, 0, 1)
            ])
        
        line = f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
        line += " " + " ".join(f"{v:.6f}" for v in poly_norm)
        lines.append(line)
    
    return lines


def build_yolo_dataset(image_dir, mask_dir, output_root, test_size=0.1, val_size=0.1):
    """Build YOLO format dataset từ images + masks."""
    
    # Collect pairs
    img_files = sorted(glob.glob(f"{image_dir}/**/*.jpg", recursive=True))
    pairs = []
    for img_path in img_files:
        name = os.path.basename(img_path).replace('.jpg', '')
        mask_path = f"{mask_dir}/{name}_mask.png"
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    
    # Split
    train_pairs, test_pairs = train_test_split(pairs, test_size=test_size, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_root}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_root}/labels/{split}", exist_ok=True)
    
    # Convert each split
    for split, pair_list in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        for img_path, mask_path in tqdm(pair_list, desc=split):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            
            lines = mask_to_yolo_polygons(mask, w, h)
            if not lines:
                continue
            
            name = os.path.basename(img_path).replace('.jpg', '')
            cv2.imwrite(f"{output_root}/images/{split}/{name}.jpg", img)
            with open(f"{output_root}/labels/{split}/{name}.txt", "w") as f:
                f.write("\n".join(lines))
    
    # Create data.yaml
    yaml_content = f"""path: {os.path.abspath(output_root)}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: document
"""
    with open(f"{output_root}/data.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"Dataset ready at {output_root}")
```

### 2.3 Phân chia dataset tổng

| Split | Số ảnh | Nguồn |
|---|---|---|
| **Train** | 6,000+ | SmartDoc train (120) + Nhóm6 auto (820) + Synthetic (5,000) |
| **Val** | 300 | SmartDoc val (30) + Nhóm6 verified val (200) + Synthetic val (70) |
| **Test** | 200 | Nhóm6 verified hold-out (100) + SmartDoc test (100) |

### 2.4 Multi-class option (mở rộng)

Thay vì 1 class "document", có thể train với 3 classes:

```yaml
# data_3class.yaml
nc: 3
names:
  0: document       # Main document boundary
  1: text_region    # Vùng có chữ chính
  2: figure         # Hình ảnh/đồ thị trên document
```

→ Đóng góp scientifically richer, gần với DLA (Document Layout Analysis).

---

## 3. Training Strategy

### 3.1 Cấu hình training chính

```python
# ml2/yolo_seg/train.py
from ultralytics import YOLO
import argparse, os

def train(args):
    # Load pretrained COCO hoặc from scratch
    if args.from_scratch:
        model = YOLO(f"yolo11{args.size}-seg.yaml")  # Random init
    else:
        model = YOLO(f"yolo11{args.size}-seg.pt")    # COCO pretrained
    
    results = model.train(
        # ─── Dataset ───
        data=args.data_yaml,
        
        # ─── Schedule ───
        epochs=args.epochs,
        patience=50,                  # Early stopping patience
        
        # ─── Image ───
        imgsz=args.imgsz,
        rect=False,                   # Rectangular training off
        cache=False,                  # Cache images RAM
        
        # ─── Batch ───
        batch=args.batch_size,
        workers=args.workers,
        
        # ─── Optimizer ───
        optimizer="AdamW",            # Best general purpose
        lr0=0.001,                    # Initial LR
        lrf=0.01,                     # Final LR fraction
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # ─── Loss weights ───
        box=7.5,                      # Box loss gain
        cls=0.5,                      # Cls loss gain
        dfl=1.5,                      # DFL loss gain
        
        # ─── Mask specific ───
        overlap_mask=True,
        mask_ratio=4,
        
        # ─── Augmentation ───
        hsv_h=0.015,                  # Hue augment
        hsv_s=0.7,                    # Saturation augment
        hsv_v=0.4,                    # Value augment
        degrees=15.0,                 # Rotation deg
        translate=0.1,                # Translation
        scale=0.5,                    # Scale
        shear=2.0,                    # Shear
        perspective=0.0005,           # Perspective transform
        flipud=0.0,                   # Vertical flip
        fliplr=0.5,                   # Horizontal flip
        mosaic=1.0,                   # Mosaic prob
        mixup=0.15,                   # Mixup prob
        copy_paste=0.3,               # Copy-paste prob (good for seg)
        
        # ─── Mixed precision ───
        amp=True,
        
        # ─── Tracking ───
        project=args.project,
        name=args.run_name,
        save=True,
        save_period=10,
        verbose=True,
        plots=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["n","s","m","l","x"], default="s")
    parser.add_argument("--data_yaml", default="ml2/datasets/yolo_format/data.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--project", default="ml2/yolo_seg/runs")
    parser.add_argument("--run_name", default="exp")
    args = parser.parse_args()
    
    train(args)
```

### 3.2 Training schedule 3 phase

#### Phase 1: Warmup + Head Training (Epoch 1-20)

```yaml
freeze_layers: 10           # Freeze backbone đầu
lr0: 1e-3
mosaic: 1.0
mixup: 0.15
imgsz: 640
batch: 32
```

**Mục tiêu:** mAP@0.5 (box) ≥ 0.65

#### Phase 2: Full Network Unfreeze (Epoch 21-150)

```yaml
freeze_layers: 0
lr0: 1e-3 → 1e-4 (cosine decay)
mosaic: 1.0 (off at epoch 130 — 20 epoch cuối)
imgsz: 640
batch: 32
```

**Mục tiêu:** mAP@0.5 (box) ≥ 0.90, mAP@0.5 (mask) ≥ 0.85

#### Phase 3: High-resolution Fine-tune (Epoch 151-200)

```yaml
freeze_layers: 0
lr0: 1e-4 → 1e-5
mosaic: 0  # Off hoàn toàn
mixup: 0
imgsz: 1024  # Tăng resolution
batch: 16
```

**Mục tiêu:** mAP@0.5 ≥ 0.92, mIoU ≥ 0.87

### 3.3 Mixed Resolution Training (option mở rộng)

Train với random imgsz trong range [640, 1024] mỗi batch để model robust nhiều resolution:

```python
# Custom collate_fn
def random_resize_collate(batch):
    imgsz = random.choice([640, 768, 896, 1024])
    # Resize batch về imgsz...
    return batch
```

### 3.4 Loss curves mong đợi

```
                ┌── train_loss
Loss            │
 ▲              │
 │ ╱╲           │
 │╱  ╲╲         │
 │    ╲╲╲╲      │     ╱── val_loss
 │       ╲╲╲╲╲╲╲│╲╲╲╲╱
 │            ╲╲│  ╲╲╲╲╲╲___
 │              │
 └──────────────┴──────────────► Epoch
                ↑
                Phase 1 → 2 transition
```

---

## 4. Evaluation Metrics

### 4.1 4 nhóm metric

#### a. Detection metrics (BBox)

| Metric | Mục tiêu | Mô tả |
|---|---|---|
| Precision (box) | ≥ 0.92 | TP/(TP+FP) |
| Recall (box) | ≥ 0.92 | TP/(TP+FN) |
| mAP@0.5 (box) | **≥ 0.92** | mAP với IoU threshold 0.5 |
| mAP@0.5:0.95 (box) | ≥ 0.78 | mAP trung bình các IoU thresholds |

#### b. Segmentation metrics (Mask)

| Metric | Mục tiêu | Mô tả |
|---|---|---|
| mAP@0.5 (mask) | **≥ 0.87** | mAP cho mask |
| mAP@0.5:0.95 (mask) | ≥ 0.73 | Multi-threshold |
| mIoU (vs U2NET) | ≥ 0.85 | So sánh với U2NET cùng test set |
| F1 (mask) | ≥ 0.89 | Dice score |

#### c. Speed metrics

| Backend | Latency mục tiêu |
|---|---|
| GPU FP32 | < 15 ms |
| GPU FP16 | < 8 ms |
| CPU | < 100 ms |
| ONNX | < 20 ms |
| CoreML | < 25 ms |
| TensorRT | < 5 ms |

#### d. Robustness metrics (per-category)

Eval trên 7 categories của dataset Nhóm 6.

### 4.2 Code custom evaluation

```python
# ml2/yolo_seg/eval.py
from ultralytics import YOLO
import numpy as np, cv2, glob, time

def evaluate_full(weights_path, data_yaml, test_imgs, test_masks):
    """Đánh giá đầy đủ YOLO model."""
    model = YOLO(weights_path)
    
    # ─── Built-in metrics ───
    metrics = model.val(data=data_yaml, split="test", plots=True)
    detection_results = {
        "box_mAP_50": float(metrics.box.map50),
        "box_mAP_50_95": float(metrics.box.map),
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "mask_mAP_50": float(metrics.seg.map50),
        "mask_mAP_50_95": float(metrics.seg.map),
    }
    
    # ─── Custom mIoU (so sánh trực tiếp với U2NET) ───
    ious = []
    for img_path, mask_gt_path in zip(test_imgs, test_masks):
        results = model(img_path, verbose=False)
        if len(results[0].masks) == 0:
            ious.append(0)
            continue
        
        mask_pred = results[0].masks.data[0].cpu().numpy()
        mask_gt = cv2.imread(mask_gt_path, 0)
        mask_pred_resized = cv2.resize(mask_pred, mask_gt.shape[::-1])
        
        inter = ((mask_pred_resized > 0.5) & (mask_gt > 127)).sum()
        union = ((mask_pred_resized > 0.5) | (mask_gt > 127)).sum()
        ious.append(inter / (union + 1e-7))
    
    custom_metrics = {"mIoU_vs_GT": float(np.mean(ious))}
    
    # ─── Speed benchmark ───
    speed_results = benchmark_speed(model, test_imgs[:50])
    
    return {
        "detection": detection_results,
        "segmentation": custom_metrics,
        "speed": speed_results,
    }


def benchmark_speed(model, imgs, n_warmup=10):
    """Đo latency median + p95."""
    latencies = []
    for i, img_path in enumerate(imgs):
        t0 = time.perf_counter()
        _ = model(img_path, verbose=False)
        t1 = time.perf_counter()
        if i >= n_warmup:
            latencies.append((t1 - t0) * 1000)
    return {
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": float(1000 / np.median(latencies)),
    }
```

---

## 5. Ablation Studies

### 5.1 Bảng ablation cần báo cáo

| Exp | Size | Init | imgsz | Epochs | Aug | mAP@0.5 box | mAP@0.5 mask | mIoU | FPS |
|---|---|---|---|---|---|---|---|---|---|
| Y1 (baseline) | n | COCO | 640 | 200 | basic | 0.86 | 0.81 | 0.81 | 130 |
| Y2 | n | scratch | 640 | 300 | basic | 0.78 | 0.73 | 0.74 | 130 |
| Y3 | n | COCO | 640 | 200 | strong | 0.89 | 0.85 | 0.84 | 130 |
| Y4 | n | COCO | 1024 | 200 | strong | 0.91 | 0.87 | 0.86 | 60 |
| Y5 | s | COCO | 640 | 200 | strong | 0.92 | 0.88 | 0.86 | 90 |
| Y6 | s | COCO | 1024 | 200 | strong | **0.94** | **0.90** | **0.88** | 45 |
| Y7 | m | COCO | 1024 | 150 | strong | 0.95 | 0.91 | 0.89 | 25 |
| Y8 | Y6 + TTA | s | COCO | 1024 | strong | 0.95 | 0.91 | 0.89 | 15 |
| Y9 | Y6 + Multi-class (3) | s | COCO | 1024 | strong | 0.93 | 0.89 | 0.87 | 45 |

### 5.2 Hyperparameter Sweep

```python
# ml2/yolo_seg/sweep.py
from itertools import product

SWEEP_GRID = {
    "imgsz": [640, 768, 1024],
    "batch": [16, 32],
    "lr0": [1e-4, 5e-4, 1e-3, 5e-3],
    "weight_decay": [0, 1e-5, 5e-4],
    "mosaic": [0.5, 1.0],
    "mixup": [0.0, 0.1, 0.15],
    "copy_paste": [0.0, 0.3, 0.5],
}

def run_sweep():
    results = []
    for combo in product(*SWEEP_GRID.values()):
        config = dict(zip(SWEEP_GRID.keys(), combo))
        # Train shortened (50 epochs) để screen
        model = YOLO("yolo11s-seg.pt")
        r = model.train(
            data="data.yaml",
            epochs=50,
            **config,
            verbose=False
        )
        results.append({
            "config": config,
            "mAP_box": r.box.map50,
            "mAP_mask": r.seg.map50,
        })
    
    # Pick top 5 → full training
    results.sort(key=lambda x: x["mAP_mask"], reverse=True)
    return results[:5]
```

### 5.3 So sánh từ scratch vs Fine-tune

Đây là 1 ablation **có giá trị học thuật cao**:

```
                    COCO Pretrained        From Scratch
Y1 (n, 200 ep)      mAP_mask = 0.81        mAP_mask = 0.73
Y3 (n, 200 ep)      mAP_mask = 0.85        mAP_mask = ?
Y5 (s, 200 ep)      mAP_mask = 0.88        mAP_mask = ?

Insight: Fine-tune từ COCO cho mAP_mask cao hơn ~10% vì transfer knowledge.
         From scratch cần dataset lớn hơn nhiều và epochs gấp đôi.
```

---

## 6. Visualization Module (chi tiết)

### 6.1 Yêu cầu visualization

User đã yêu cầu vẽ kết quả YOLO. Module cần draw:
1. **Bounding box** với label + confidence
2. **Segmentation mask** overlay (bán trong suốt, color-coded)
3. **4 corners** trích từ mask (cho Step 2 Perspective)
4. **Confidence score + FPS** info
5. **Class name** (nếu multi-class)

### 6.2 Code visualization đầy đủ

```python
# ml2/yolo_seg/visualize.py
import cv2
import numpy as np

class YOLODocVisualizer:
    # Palette màu cho multi-class
    COLORS = {
        0: {"name": "document", "rgb": (0, 255, 0)},      # Xanh lá
        1: {"name": "text_region", "rgb": (255, 255, 0)},  # Vàng
        2: {"name": "figure", "rgb": (255, 0, 255)},       # Hồng
    }
    BBOX_COLOR = (0, 0, 255)       # Đỏ cho bbox
    CORNER_COLOR = (255, 165, 0)   # Cam cho corners
    TEXT_COLOR = (255, 255, 255)   # Trắng cho text
    
    def __init__(self, mask_alpha=0.4, corner_size=15, font_scale=0.7):
        self.mask_alpha = mask_alpha
        self.corner_size = corner_size
        self.font_scale = font_scale
    
    def draw_results(self, image, yolo_results, show_corners=True, show_fps=True):
        """Vẽ toàn bộ output YOLO lên ảnh."""
        annotated = image.copy()
        
        if len(yolo_results[0].boxes) == 0:
            self._draw_text(annotated, "No document detected", (20, 50), 
                            color=(0, 0, 255), size=1.2)
            return annotated
        
        # Lấy thông tin từ results
        boxes = yolo_results[0].boxes
        masks = yolo_results[0].masks
        names = yolo_results[0].names
        
        # 1. Vẽ mask overlay cho từng instance
        if masks is not None:
            mask_overlay = np.zeros_like(image)
            for i in range(len(masks.data)):
                mask = masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_bin = mask_resized > 0.5
                
                cls_id = int(boxes.cls[i])
                color = self.COLORS.get(cls_id, self.COLORS[0])["rgb"]
                mask_overlay[mask_bin] = color
            
            annotated = cv2.addWeighted(annotated, 1 - self.mask_alpha,
                                         mask_overlay, self.mask_alpha, 0)
        
        # 2. Vẽ bounding boxes
        for i in range(len(boxes)):
            xyxy = boxes[i].xyxy[0].cpu().numpy().astype(int)
            conf = float(boxes[i].conf[0])
            cls_id = int(boxes[i].cls[0])
            cls_name = names.get(cls_id, f"class_{cls_id}")
            
            # Box
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]),
                          self.BBOX_COLOR, 3)
            
            # Label
            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           self.font_scale, 2)
            cv2.rectangle(annotated, (xyxy[0], xyxy[1] - lh - 10),
                          (xyxy[0] + lw + 10, xyxy[1]),
                          self.BBOX_COLOR, -1)
            cv2.putText(annotated, label, (xyxy[0] + 5, xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                        self.TEXT_COLOR, 2)
        
        # 3. Vẽ 4 corners cho document chính
        if show_corners and masks is not None:
            # Lấy mask của document chính (class 0, conf cao nhất)
            doc_indices = [i for i in range(len(boxes)) if int(boxes.cls[i]) == 0]
            if doc_indices:
                best_idx = max(doc_indices, key=lambda i: float(boxes.conf[i]))
                mask = masks.data[best_idx].cpu().numpy()
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
                
                corners = self._extract_corners(mask_bin)
                if corners is not None:
                    # Vẽ tứ giác nối 4 góc
                    pts = corners.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [pts], True, self.CORNER_COLOR, 2)
                    
                    # Vẽ 4 điểm có số
                    for i, pt in enumerate(corners):
                        center = tuple(pt.astype(int))
                        cv2.circle(annotated, center, self.corner_size,
                                   self.CORNER_COLOR, -1)
                        cv2.circle(annotated, center, self.corner_size,
                                   (0, 0, 0), 2)
                        cv2.putText(annotated, str(i + 1),
                                    (center[0] - 5, center[1] + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    self.TEXT_COLOR, 2)
        
        # 4. Info box ở góc trên
        if show_fps:
            speed = yolo_results[0].speed
            fps = 1000 / speed['inference']
            info_lines = [
                f"FPS: {fps:.1f}",
                f"Inference: {speed['inference']:.1f}ms",
                f"Detections: {len(boxes)}",
            ]
            y_offset = 30
            for line in info_lines:
                self._draw_text(annotated, line, (10, y_offset),
                                color=(255, 255, 0), size=0.7)
                y_offset += 30
        
        return annotated
    
    def _extract_corners(self, mask_bin):
        """Trích 4 góc từ binary mask."""
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        
        # Thử approxPolyDP với epsilon biến đổi để bắt 4 góc
        for eps_ratio in np.linspace(0.01, 0.1, 10):
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, eps_ratio * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                return self._sort_corners(corners)
        
        # Fallback: minAreaRect
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return self._sort_corners(box.astype(np.float32))
    
    def _sort_corners(self, pts):
        """Sort 4 góc theo thứ tự TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def _draw_text(self, img, text, pos, color=(255, 255, 255), size=0.7):
        """Vẽ text với background đen."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2)
        cv2.rectangle(img, (pos[0]-3, pos[1]-th-3),
                      (pos[0]+tw+3, pos[1]+3), (0, 0, 0), -1)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
```

### 6.3 Demo script tạo visualization batch

```python
# ml2/yolo_seg/demo_viz.py
from ultralytics import YOLO
from visualize import YOLODocVisualizer
import cv2, glob, os
from tqdm import tqdm

model = YOLO("ml2/yolo_seg/weights/yolo11s_seg_doc_best.pt")
viz = YOLODocVisualizer()

os.makedirs("ml2/yolo_seg/demos", exist_ok=True)

# Sample 50 ảnh từ test set
test_imgs = glob.glob("ml2/datasets/nhom6_1020/images/**/*.jpg", recursive=True)[:50]

for img_path in tqdm(test_imgs):
    img = cv2.imread(img_path)
    results = model(img, verbose=False)
    annotated = viz.draw_results(img, results)
    
    name = os.path.basename(img_path).replace('.jpg', '_viz.jpg')
    cv2.imwrite(f"ml2/yolo_seg/demos/{name}", annotated)

# Optional: tạo grid 3x3 montage
imgs_for_grid = sorted(glob.glob("ml2/yolo_seg/demos/*.jpg"))[:9]
grid = np.zeros((1920, 1920, 3), dtype=np.uint8)
for i, img_p in enumerate(imgs_for_grid):
    img = cv2.imread(img_p)
    img_resized = cv2.resize(img, (640, 640))
    row, col = i // 3, i % 3
    grid[row*640:(row+1)*640, col*640:(col+1)*640] = img_resized
cv2.imwrite("ml2/yolo_seg/demos/grid_9.jpg", grid)
```

---

## 7. Inference + Export

### 7.1 Test-Time Augmentation

```python
# ml2/yolo_seg/infer_tta.py
from ultralytics import YOLO

model = YOLO("weights/best.pt")

# Built-in TTA
results = model(image, augment=True, verbose=False)
```

### 7.2 Multi-format Export

```python
# ml2/yolo_seg/export_all.py
from ultralytics import YOLO

model = YOLO("weights/best.pt")

# Export sang nhiều format
formats = ["onnx", "openvino", "coreml", "tflite", "engine"]
for fmt in formats:
    try:
        model.export(format=fmt, imgsz=640, half=True, simplify=True)
        print(f"✓ Exported to {fmt}")
    except Exception as e:
        print(f"✗ {fmt} failed: {e}")
```

---

## 8. Deliverables

| Output | Mô tả |
|---|---|
| `ml2/yolo_seg/prepare_dataset.py` | Mask → YOLO polygon converter |
| `ml2/yolo_seg/train.py` | Training wrapper với 3-phase schedule |
| `ml2/yolo_seg/sweep.py` | Hyperparameter sweep script |
| `ml2/yolo_seg/eval.py` | Custom evaluation với 4 nhóm metrics |
| `ml2/yolo_seg/visualize.py` | Visualization module |
| `ml2/yolo_seg/demo_viz.py` | Batch demo script |
| `ml2/yolo_seg/infer_tta.py` | Inference với TTA |
| `ml2/yolo_seg/export_all.py` | Multi-format export |
| `weights/yolo11n_seg_doc_best.pt` | Final mobile model |
| `weights/yolo11s_seg_doc_best.pt` | Final balanced model |
| `weights/yolo11m_seg_doc_best.pt` | Final quality model |
| `weights/yolo11s_seg_doc_best.onnx/coreml/openvino` | Exported formats |
| `runs/` | TensorBoard logs |
| `demos/` | ~50 ảnh visualization + grid montage |
| `report_yolo.md` | Báo cáo Phase 2 |

---

## 9. Risk Matrix

| Rủi ro | Mức | Phòng tránh |
|---|---|---|
| mAP@0.5 không đạt 0.90 | Trung bình | Tăng model size (n → s → m), tăng imgsz, augment mạnh hơn |
| Mask coefficient không đủ chi tiết viền | Trung bình | Tăng mask_ratio=2 (thay 4), tăng imgsz lên 1024+ |
| From-scratch convergence chậm | Cao | Cần epochs gấp 2-3 lần fine-tune, hoặc warmup dài hơn |
| Auto-label noise ảnh hưởng training | Cao | Manual verify nhiều hơn, dùng confidence threshold filter |
| Multi-class confusion | Trung bình | Class balanced sampling, focal loss |
| OOM khi batch lớn | Trung bình | Gradient accumulation, giảm imgsz |

---

## 10. Checklist hoàn thành

### Data Preparation
- [ ] Convert toàn bộ mask → YOLO polygon format
- [ ] Train/val/test split 6000+/300/200
- [ ] (Optional) Sinh synthetic data
- [ ] (Optional) Tạo multi-class labels

### Training
- [ ] Train YOLOv11n-seg baseline (200 epoch, imgsz 640)
- [ ] Train YOLOv11s-seg main (200 epoch, imgsz 640 → 1024)
- [ ] Train YOLOv11m-seg quality (150 epoch, imgsz 1024)
- [ ] Train YOLOv11n từ scratch (300 epoch) — ablation novelty
- [ ] (Optional) Hyperparameter sweep top 5

### Evaluation
- [ ] mAP@0.5/0.95 trên test
- [ ] mIoU custom vs U2NET cùng test set
- [ ] Speed benchmark đa backend
- [ ] Per-category evaluation 7 nhóm
- [ ] Ablation table ≥ 7 experiments

### Visualization & Export
- [ ] Visualization module với bbox + mask + corners + info
- [ ] Demo 50+ ảnh
- [ ] Export ONNX, CoreML, OpenVINO, TFLite

### Reporting
- [ ] Figures: training curves, mAP plot, per-class AP
- [ ] Comparison table size n vs s vs m
- [ ] Comparison fine-tune vs from-scratch
- [ ] Viết báo cáo Markdown đầy đủ

---

*Phase 2 hoàn thành = 3 YOLO variants trained + visualization module + export multi-format.*
