# Phase 1 — Build & Train U2NET from Scratch (Chi tiết)

> **Mục tiêu:** Implement U2NET từ đầu, train trên dataset đa nguồn (DUTS-TR + SmartDoc + 1020 ảnh nhóm), đạt mIoU ≥ 0.85 trên test set
> **Đầu ra:** Model weights + training logs + ablation studies + báo cáo

---

## 1. Lựa chọn Kiến trúc U2NET

### 1.1 So sánh 2 variant của U2NET

| | **U2NET full** | **U2NETp (lite)** |
|---|---|---|
| Params | 44M | 1.1M |
| Model size | 176MB | 4.7MB |
| FLOPs | 87G | 1.2G |
| Inference (GPU) | ~15ms | ~5ms |
| Inference (CPU) | ~500ms | ~80ms |
| Inference (Mobile) | ~200ms | ~30ms |
| mIoU (DUTS-TE, paper) | 0.886 | 0.851 |
| Memory train (batch=16) | ~14GB VRAM | ~4GB VRAM |
| Train time DUTS-TR (600 ep) | ~3 ngày 1×V100 | ~1 ngày 1×V100 |

### 1.2 Khuyến nghị: Train CẢ HAI variant

Train cả 2 model là **đóng góp khoa học mạnh** vì:
- So sánh accuracy/speed trade-off trực tiếp
- Cho phép chọn variant phù hợp use case (cloud server vs mobile)
- Tạo bảng ablation thuyết phục

### 1.3 Kiến trúc U2NET tổng thể

```
        ┌──────────────────────────────────────────────┐
        │  U2NET / U2NETp                              │
        │                                              │
        │  Encoder (6 levels, downsampling):           │
        │    En_1: RSU-7  (3 → 64/16)                  │
        │    En_2: RSU-6  (64/16 → 128/64)             │
        │    En_3: RSU-5  (128/64 → 256/64)            │
        │    En_4: RSU-4  (256/64 → 512/64)            │
        │    En_5: RSU-4F (512/64 → 512/64) no pool    │
        │    En_6: RSU-4F (512/64 → 512/64)            │
        │                                              │
        │  Decoder (5 levels, upsampling):             │
        │    De_5 → De_4 → De_3 → De_2 → De_1          │
        │                                              │
        │  Side outputs:                               │
        │    6 supervisions từ En_6 đến De_1 + fused   │
        │                                              │
        │  Output: 1 mask (H×W×1) sigmoid              │
        └──────────────────────────────────────────────┘
```

### 1.4 Khối RSU (Residual U-block)

RSU-N: U-Net mini bên trong với N levels, dùng dilated conv ở bottleneck.

```
RSU-7 (E1):
    Input (C_in)
       ▼
    Conv 3×3 → BN → ReLU (C_mid)
       ▼
    ──── 7-level inner U-Net ──────────
    │                                  │
    │  E1: Conv (C_mid → C_mid)        │
    │      ▼ pool                      │
    │  E2: Conv                        │
    │      ▼ pool                      │
    │  ... đến E7                      │
    │                                  │
    │  Bottleneck: Dilated Conv (d=2)  │
    │                                  │
    │  Decoder: D7, D6, ..., D1        │
    │  với skip connection             │
    │                                  │
    └──────────────────────────────────
       ▼
    Conv 3×3 → C_out
       ▼
    Add(Conv_input, residual)
       ▼
    Output (C_out)
```

---

## 2. Dataset Strategy

### 2.1 Cấu trúc dataset đa nguồn

```
ml2/datasets/
├── duts_tr/                 # 10,553 ảnh saliency
│   ├── images/
│   └── masks/
│
├── duts_te/                 # 5,019 ảnh test saliency
│   ├── images/
│   └── masks/
│
├── smartdoc_qa/             # ~150 ảnh document scan
│   ├── images/
│   └── masks/
│
├── midv500/                 # Optional: 500 video clips ID docs
│   ├── images/              # ~15,000 frames sau extract
│   └── masks/
│
├── nhom6_1020/
│   ├── images/              # 1020 ảnh gốc
│   ├── auto_masks/          # Auto-label từ rembg pretrained
│   └── verified_masks/      # 200-400 ảnh manual fix (eval set)
│
└── synthetic/               # Optional: synthetic data augmentation
    ├── images/              # Compose document on random backgrounds
    └── masks/
```

### 2.2 Pipeline auto-label 1020 ảnh

```python
# ml2/scripts/autolabel_nhom6.py
from rembg import remove, new_session
import cv2, os, glob
from tqdm import tqdm

# 3 model ensemble: voting cho quality cao hơn
sessions = [
    new_session("u2net"),
    new_session("u2netp"),
    new_session("isnet-general-use"),
]

src_dir = "ml2/datasets/nhom6_1020/images"
dst_dir = "ml2/datasets/nhom6_1020/auto_masks"
os.makedirs(dst_dir, exist_ok=True)

for img_path in tqdm(glob.glob(f"{src_dir}/**/*.jpg", recursive=True)):
    img = cv2.imread(img_path)
    masks = []
    for s in sessions:
        output = remove(img, session=s)
        if output.shape[2] == 4:
            masks.append(output[:, :, 3])
    
    # Majority voting
    mask_stack = np.stack(masks, axis=0)
    avg_mask = mask_stack.mean(axis=0)
    _, mask_bin = cv2.threshold(avg_mask.astype('uint8'), 127, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    
    rel_path = os.path.relpath(img_path, src_dir)
    out_name = rel_path.replace('.jpg', '_mask.png')
    out_path = os.path.join(dst_dir, out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, mask_bin)
```

### 2.3 Synthetic data generation (option mở rộng)

Tạo thêm 5000-10000 ảnh synthetic để boost training:

```python
# ml2/scripts/synthesize_documents.py
import cv2, numpy as np, glob, random

def synthesize(doc_img, doc_mask, bg_img):
    """Tổng hợp ảnh: dán document lên background random với perspective + lighting."""
    h_bg, w_bg = bg_img.shape[:2]
    h_doc, w_doc = doc_img.shape[:2]
    
    # 1. Random perspective transform
    scale = random.uniform(0.4, 0.8)
    new_h = int(h_doc * scale)
    new_w = int(w_doc * scale)
    
    # 4 corners gốc
    src_pts = np.array([[0,0], [w_doc,0], [w_doc,h_doc], [0,h_doc]], dtype=np.float32)
    # 4 corners đích (perturbed random)
    perturb = random.uniform(20, 80)
    dst_pts = np.array([
        [random.uniform(0,perturb), random.uniform(0,perturb)],
        [new_w-random.uniform(0,perturb), random.uniform(0,perturb)],
        [new_w-random.uniform(0,perturb), new_h-random.uniform(0,perturb)],
        [random.uniform(0,perturb), new_h-random.uniform(0,perturb)],
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc_img, M, (new_w, new_h))
    warped_mask = cv2.warpPerspective(doc_mask, M, (new_w, new_h))
    
    # 2. Random position trên background
    x = random.randint(0, w_bg - new_w)
    y = random.randint(0, h_bg - new_h)
    
    # 3. Composite
    composed = bg_img.copy()
    mask_3 = (warped_mask[:,:,np.newaxis] / 255.0).repeat(3, axis=2)
    composed[y:y+new_h, x:x+new_w] = (
        warped_doc * mask_3 + composed[y:y+new_h, x:x+new_w] * (1 - mask_3)
    ).astype(np.uint8)
    
    # 4. Random lighting variation
    if random.random() < 0.5:
        gamma = random.uniform(0.5, 1.5)
        composed = np.power(composed/255.0, gamma) * 255
        composed = composed.astype(np.uint8)
    
    # 5. Random shadow / blur
    if random.random() < 0.3:
        composed = cv2.GaussianBlur(composed, (5,5), 1.5)
    
    # Build full mask
    full_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
    full_mask[y:y+new_h, x:x+new_w] = warped_mask
    
    return composed, full_mask

# Sinh 10000 ảnh synthetic
docs = glob.glob("ml2/datasets/smartdoc_qa/images/*.jpg")
bgs = glob.glob("ml2/datasets/backgrounds/*.jpg")  # tải từ MIT Indoor Scenes

for i in tqdm(range(10000)):
    doc = cv2.imread(random.choice(docs))
    doc_mask = cv2.imread(random.choice(docs).replace('images', 'masks'), 0)
    bg = cv2.imread(random.choice(bgs))
    
    img, mask = synthesize(doc, doc_mask, bg)
    cv2.imwrite(f"ml2/datasets/synthetic/images/synth_{i:05d}.jpg", img)
    cv2.imwrite(f"ml2/datasets/synthetic/masks/synth_{i:05d}.png", mask)
```

### 2.4 Phân chia train/val/test

| Stage | Train | Val | Test | Tổng train |
|---|---|---|---|---|
| **Stage 1 (Pretrain saliency)** | DUTS-TR (10,553) | DUTS-TE-val (1,000) | DUTS-TE-test (4,019) | 10,553 |
| **Stage 2 (Document fine-tune)** | SmartDoc-QA train (120) + Nhóm6 auto (820) + Synthetic (5,000) | SmartDoc val (30) + Nhóm6 verified (200) | Nhóm6 verified hold-out (200) | 5,940 |

---

## 3. Implementation U2NET

### 3.1 Reference & approach

- **Code gốc:** https://github.com/xuebinqin/U-2-Net
- **Approach:** Tự viết lại từng module với comment giải thích → vừa học, vừa có code chạy

### 3.2 Cấu trúc files

```
ml2/u2net/
├── model.py              # U2NET + U2NETp + REBNCONV + RSU blocks
├── loss.py               # Multi-supervision BCE + IoU + SSIM
├── dataset.py            # DocSegDataset class
├── augmentation.py       # Augmentation pipeline (albumentations)
├── train.py              # Training loop với checkpointing
├── eval.py               # Compute mIoU, F1, MAE, Boundary-F1
├── infer.py              # Inference + post-processing
├── visualize.py          # Plot training curves, sample predictions
├── configs/
│   ├── stage1_duts_full.yaml
│   ├── stage1_duts_lite.yaml
│   ├── stage2_doc_full.yaml
│   └── stage2_doc_lite.yaml
└── weights/
    ├── u2net_stage1.pth
    ├── u2net_stage2.pth
    ├── u2netp_stage1.pth
    └── u2netp_stage2.pth
```

### 3.3 Combo Loss function

```python
# ml2/u2net/loss.py
import torch
import torch.nn as nn
from pytorch_msssim import ssim

class U2NETLoss(nn.Module):
    """
    Multi-supervision loss kết hợp 3 thành phần:
    - BCE: pixel-wise classification
    - IoU: structural overlap
    - SSIM: structural similarity (giữ chi tiết viền)
    """
    def __init__(self, w_bce=1.0, w_iou=1.0, w_ssim=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.w_bce = w_bce
        self.w_iou = w_iou
        self.w_ssim = w_ssim
    
    def iou_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=[1,2,3])
        union = pred.sum(dim=[1,2,3]) + target.sum(dim=[1,2,3]) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        return 1 - iou.mean()
    
    def ssim_loss(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        return 1 - ssim(pred_sig, target, data_range=1.0, size_average=True)
    
    def forward(self, d0, d1, d2, d3, d4, d5, d6, target):
        """
        d0 = fused output
        d1-d6 = side outputs (multi-scale supervision)
        target = ground truth mask (B, 1, H, W)
        """
        outputs = [d0, d1, d2, d3, d4, d5, d6]
        total_loss = 0
        loss_main = 0
        
        for i, d in enumerate(outputs):
            bce_l = self.bce(d, target)
            iou_l = self.iou_loss(d, target)
            ssim_l = self.ssim_loss(d, target)
            
            loss_i = self.w_bce*bce_l + self.w_iou*iou_l + self.w_ssim*ssim_l
            total_loss += loss_i
            
            if i == 0:  # d0 = main output
                loss_main = loss_i
        
        return total_loss, loss_main
```

### 3.4 Unit test trước khi train

```python
def test_u2net_forward_backward():
    """Test forward + backward không lỗi."""
    for ModelCls in [U2NET, U2NETp]:
        model = ModelCls(in_ch=3, out_ch=1).cuda()
        x = torch.randn(2, 3, 320, 320).cuda()
        outputs = model(x)
        assert len(outputs) == 7, f"Expected 7 outputs, got {len(outputs)}"
        assert outputs[0].shape == (2, 1, 320, 320)
        
        # Backward
        target = torch.randint(0, 2, (2, 1, 320, 320)).float().cuda()
        loss_fn = U2NETLoss()
        total_loss, main_loss = loss_fn(*outputs, target)
        total_loss.backward()
        
        print(f"{ModelCls.__name__}: forward OK, total_loss={total_loss.item():.4f}, main={main_loss.item():.4f}")
```

---

## 4. Training Strategy 2 Stage

### 4.1 Stage 1: Pretrain on DUTS-TR

**Mục tiêu:** Học general saliency knowledge (cảm thụ "đâu là object nổi bật").

#### Config tham khảo cho U2NET full

```yaml
# ml2/u2net/configs/stage1_duts_full.yaml
model: U2NET
in_channels: 3
out_channels: 1

dataset:
  train: duts_tr
  val: duts_te_val
  test: duts_te_test
  input_size: 320
  batch_size: 32           # Adjust theo VRAM
  num_workers: 8
  pin_memory: true

augmentation:
  random_resize_crop: [320, 320]
  random_flip: 0.5
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.05
  random_rotate: 15
  random_perspective: 0.1

optimizer:
  name: Adam
  lr: 1e-3
  betas: [0.9, 0.999]
  weight_decay: 0
  eps: 1e-8

scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 50
  T_mult: 2
  eta_min: 1e-6

training:
  epochs: 600              # Theo paper gốc
  amp: true                # Mixed precision FP16
  gradient_accumulation: 1
  gradient_clip: 1.0
  
loss:
  bce_weight: 1.0
  iou_weight: 1.0
  ssim_weight: 1.0

checkpoint:
  save_dir: ml2/u2net/weights
  save_every: 25
  save_best: true
  best_metric: val_mIoU

logging:
  log_every: 50
  val_every: 1
  tensorboard: true
```

#### Config cho U2NETp (nhẹ hơn)

```yaml
# Khác biệt chính so với full:
model: U2NETp
dataset:
  batch_size: 64           # Có thể tăng vì model nhỏ
training:
  epochs: 400              # Hội tụ nhanh hơn
```

### 4.2 Stage 2: Fine-tune trên Document data

```yaml
# ml2/u2net/configs/stage2_doc_full.yaml
model: U2NET
load_from: weights/u2net_stage1.pth

dataset:
  train: smartdoc_qa_train + nhom6_auto + synthetic
  val: smartdoc_qa_val + nhom6_verified_val
  test: nhom6_verified_test
  input_size: 384          # Tăng resolution cho chi tiết viền
  batch_size: 16

augmentation:
  # Strong augmentation cho phone-photo realism
  random_perspective: 0.4
  random_motion_blur: 0.3
  random_gaussian_noise: 0.3
  random_shadow: 0.4
  random_brightness: 0.4
  
optimizer:
  name: Adam
  lr: 1e-4                 # Giảm 10× vì fine-tune
  weight_decay: 1e-5

scheduler:
  name: CosineAnnealingLR
  T_max: 200
  eta_min: 1e-6

training:
  epochs: 200
  amp: true
  early_stopping:
    patience: 30
    metric: val_mIoU
    mode: max
```

### 4.3 Training loop với logging đầy đủ

```python
# ml2/u2net/train.py (simplified)
import torch, json, os
from torch.utils.tensorboard import SummaryWriter

def train(config):
    model = build_model(config.model)
    if config.load_from:
        model.load_state_dict(torch.load(config.load_from))
    model.cuda()
    
    train_loader = build_loader(config, split='train')
    val_loader = build_loader(config, split='val')
    
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    criterion = U2NETLoss(**config.loss)
    scaler = torch.cuda.amp.GradScaler() if config.training.amp else None
    
    writer = SummaryWriter(f"runs/{config.run_name}")
    history = {"epoch": [], "train_loss": [], "val_iou": [], "val_f1": [], "val_mae": [], "val_bf": []}
    best_iou = 0
    
    for epoch in range(config.training.epochs):
        # ─── Train ───
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            imgs, masks = batch['image'].cuda(), batch['mask'].cuda()
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(imgs)
                total_loss, _ = criterion(*outputs, masks)
            
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            epoch_loss += total_loss.item()
        
        scheduler.step()
        epoch_loss /= len(train_loader)
        
        # ─── Validate ───
        metrics = evaluate(model, val_loader)
        
        # ─── Log ───
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_loss)
        history['val_iou'].append(metrics['iou'])
        history['val_f1'].append(metrics['f1'])
        history['val_mae'].append(metrics['mae'])
        history['val_bf'].append(metrics['bf'])
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('IoU/val', metrics['iou'], epoch)
        writer.add_scalar('F1/val', metrics['f1'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch} | Loss {epoch_loss:.4f} | IoU {metrics['iou']:.4f} | "
              f"F1 {metrics['f1']:.4f} | MAE {metrics['mae']:.4f} | BF {metrics['bf']:.4f}")
        
        # ─── Save ───
        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            torch.save(model.state_dict(), f"{config.checkpoint.save_dir}/best.pth")
        if epoch % config.checkpoint.save_every == 0:
            torch.save(model.state_dict(), f"{config.checkpoint.save_dir}/epoch_{epoch}.pth")
        
        # Save history
        with open(f"runs/{config.run_name}/history.json", "w") as f:
            json.dump(history, f, indent=2)
```

### 4.4 Mục tiêu metrics

| Stage | Variant | Dataset | mIoU mục tiêu | F1 mục tiêu | MAE mục tiêu |
|---|---|---|---|---|---|
| Stage 1 | U2NET full | DUTS-TE-test | ≥ 0.87 | ≥ 0.90 | < 0.05 |
| Stage 1 | U2NETp | DUTS-TE-test | ≥ 0.84 | ≥ 0.87 | < 0.06 |
| Stage 2 | U2NET full | Nhóm6 verified | ≥ 0.88 | ≥ 0.91 | < 0.04 |
| Stage 2 | U2NETp | Nhóm6 verified | ≥ 0.85 | ≥ 0.88 | < 0.05 |

---

## 5. Evaluation Metrics chi tiết

### 5.1 4 Metric chính

```python
# ml2/u2net/eval.py
import numpy as np
import cv2
from scipy.ndimage import binary_erosion

def compute_iou(pred, gt, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return inter / (union + 1e-7)

def compute_f1(pred, gt, threshold=0.5):
    """Dice F1 score."""
    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    inter = (pred_bin * gt_bin).sum()
    return 2 * inter / (pred_bin.sum() + gt_bin.sum() + 1e-7)

def compute_mae(pred, gt):
    """Mean Absolute Error trên xác suất."""
    return np.mean(np.abs(pred.astype(float) - gt.astype(float)/255.0))

def compute_boundary_f1(pred, gt, threshold=0.5, boundary_width=2):
    """F1 chỉ tính trên pixel gần viền (cách viền < boundary_width)."""
    pred_bin = (pred > threshold)
    gt_bin = (gt > 127)
    
    # Lấy viền bằng phép trừ erosion
    pred_boundary = pred_bin ^ binary_erosion(pred_bin, iterations=boundary_width)
    gt_boundary = gt_bin ^ binary_erosion(gt_bin, iterations=boundary_width)
    
    inter = (pred_boundary & gt_boundary).sum()
    precision = inter / (pred_boundary.sum() + 1e-7)
    recall = inter / (gt_boundary.sum() + 1e-7)
    return 2 * precision * recall / (precision + recall + 1e-7)

def compute_e_measure(pred, gt):
    """Enhanced-alignment Measure (E-measure) — bonus metric."""
    pred_bin = (pred > 0.5).astype(float)
    gt_bin = (gt > 127).astype(float)
    # ... implementation theo paper Fan et al. 2018
    pass
```

### 5.2 Eval theo 7 categories

```python
CATEGORIES = ["Curved", "Fold", "Incomplete", "Perspective", 
              "Rotate", "Random", "Normal"]

def evaluate_per_category(model, root="ml2/datasets/nhom6_1020"):
    results = {}
    for cat in CATEGORIES:
        imgs = glob(f"{root}/images/{cat}/*.jpg")
        gts = glob(f"{root}/verified_masks/{cat}/*.png")
        if not gts:
            continue
        
        ious, f1s, maes, bfs = [], [], [], []
        for img_p, gt_p in zip(imgs, gts):
            img = cv2.imread(img_p)
            gt = cv2.imread(gt_p, 0)
            pred = infer(model, img)
            
            ious.append(compute_iou(pred, gt))
            f1s.append(compute_f1(pred, gt))
            maes.append(compute_mae(pred, gt))
            bfs.append(compute_boundary_f1(pred, gt))
        
        results[cat] = {
            "n": len(ious),
            "mIoU": np.mean(ious),
            "F1": np.mean(f1s),
            "MAE": np.mean(maes),
            "BF": np.mean(bfs),
        }
    return results
```

---

## 6. Hyperparameter Sweep + Ablation

### 6.1 Ablation table cần báo cáo

| Exp | Model | Input | Loss | Stage1 Aug | Stage2 Aug | mIoU | F1 | BF |
|---|---|---|---|---|---|---|---|---|
| E1 | U2NETp | 320 | BCE only | basic | basic | 0.80 | 0.85 | 0.72 |
| E2 | U2NETp | 320 | BCE+IoU | basic | basic | 0.82 | 0.86 | 0.74 |
| E3 | U2NETp | 320 | BCE+IoU+SSIM | basic | basic | 0.83 | 0.87 | 0.77 |
| E4 | U2NETp | 320 | BCE+IoU+SSIM | basic | strong | 0.85 | 0.88 | 0.78 |
| E5 | U2NETp | 384 | BCE+IoU+SSIM | strong | strong | 0.86 | 0.89 | 0.80 |
| E6 | U2NET full | 384 | BCE+IoU+SSIM | strong | strong | **0.88** | **0.91** | **0.82** |
| E7 | E6 + Synthetic data | 384 | - | - | - | 0.89 | 0.91 | 0.82 |
| E8 | E7 + TTA inference | 384 | - | - | - | 0.90 | 0.92 | 0.83 |

### 6.2 Hyperparameter search space

| HP | Range | Best (mong đợi) |
|---|---|---|
| Learning rate | 1e-4, 5e-4, 1e-3, 5e-3 | 1e-3 (Stage 1), 1e-4 (Stage 2) |
| Batch size | 8, 16, 32, 64 | 32 (full) / 64 (lite) |
| Input size | 256, 288, 320, 384, 512 | 320 (Stage 1), 384 (Stage 2) |
| Optimizer | Adam, AdamW, SGD | Adam |
| Scheduler | Step, Cosine, Plateau, CosineWarmRestarts | CosineWarmRestarts |
| Loss BCE weight | 0.5, 1.0, 2.0 | 1.0 |
| Loss IoU weight | 0.5, 1.0, 2.0 | 1.0 |
| Loss SSIM weight | 0.0, 0.5, 1.0 | 1.0 |
| Aug perspective | 0.0, 0.2, 0.4 | 0.4 |
| Aug motion blur | 0.0, 0.2, 0.4 | 0.3 |

### 6.3 Tinh chỉnh boundary

Khi BF < 0.75, thêm các trick sau:

**a. Edge-aware loss bổ sung:**
```python
class EdgeLoss(nn.Module):
    """Loss khuyến khích pred match viền GT bằng Sobel."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        self.register_buffer('sx', sobel_x.view(1,1,3,3))
        self.register_buffer('sy', sobel_y.view(1,1,3,3))
    
    def forward(self, pred, gt):
        pred_sig = torch.sigmoid(pred)
        pred_edge = torch.sqrt(F.conv2d(pred_sig, self.sx, padding=1)**2 + 
                                F.conv2d(pred_sig, self.sy, padding=1)**2)
        gt_edge = torch.sqrt(F.conv2d(gt, self.sx, padding=1)**2 + 
                              F.conv2d(gt, self.sy, padding=1)**2)
        return F.l1_loss(pred_edge, gt_edge)
```

**b. Post-processing với CRF:**
```python
import pydensecrf.densecrf as dcrf
def refine_with_crf(image, mask_prob):
    """Áp DenseCRF để refine biên."""
    h, w = mask_prob.shape
    crf = dcrf.DenseCRF2D(w, h, 2)
    
    unary = -np.log(np.stack([1-mask_prob, mask_prob]) + 1e-7)
    crf.setUnaryEnergy(unary.reshape(2, -1).astype(np.float32))
    crf.addPairwiseGaussian(sxy=3, compat=3)
    crf.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    
    Q = crf.inference(5)
    return np.array(Q)[1].reshape(h, w)
```

**c. Test-time augmentation:**
```python
def predict_with_tta(model, img):
    """TTA: predict trên 4 phép biến đổi, average."""
    preds = []
    preds.append(infer(model, img))                          # Original
    preds.append(infer(model, np.flip(img, 1))[::, ::-1])    # H-flip
    preds.append(rotate_back(infer(model, rotate(img, 90)), 90))   # Rotate 90
    preds.append(rotate_back(infer(model, rotate(img, -90)), -90)) # Rotate -90
    return np.mean(preds, axis=0)
```

---

## 7. Deliverables

| Output | Mô tả |
|---|---|
| `ml2/u2net/model.py` | Code U2NET + U2NETp tự viết với comment đầy đủ |
| `ml2/u2net/loss.py` | Combo loss (BCE + IoU + SSIM + EdgeLoss) |
| `ml2/u2net/dataset.py` | Multi-source dataset class |
| `ml2/u2net/augmentation.py` | Strong augmentation pipeline |
| `ml2/u2net/train.py` | Training loop hoàn chỉnh với AMP, logging, checkpointing |
| `ml2/u2net/eval.py` | 4 metrics + per-category evaluation |
| `ml2/u2net/infer.py` | Inference + TTA + CRF post-processing |
| `ml2/u2net/visualize.py` | Plot curves, sample predictions, comparison |
| `weights/u2net_stage1.pth` | Pretrained DUTS-TR (full) |
| `weights/u2net_stage2.pth` | Final document model (full) |
| `weights/u2netp_stage1.pth` | Pretrained DUTS-TR (lite) |
| `weights/u2netp_stage2.pth` | Final document model (lite) |
| `runs/` | TensorBoard logs + history JSON |
| `figures/` | Training curves, sample predictions, per-category bar charts |
| `report_u2net.md` | Báo cáo Phase 1 đầy đủ |

---

## 8. Risk Matrix

| Rủi ro | Mức | Phòng tránh |
|---|---|---|
| Stage 1 không converge (loss plateau cao) | Trung bình | Giảm LR 5×, kiểm tra data loading, gradient clipping |
| Stage 2 overfit nhanh (val giảm sau 30 epoch) | Cao | Early stopping, weight decay 1e-4, dropout, augmentation mạnh hơn |
| Auto-label 1020 ảnh chất lượng kém | Cao | Manual verify nhiều hơn (300-400), ensemble 3 rembg models |
| Boundary F1 < 0.75 dù tinh chỉnh | Trung bình | Thêm EdgeLoss, dùng DenseCRF post-process |
| OOM trên hardware cụ thể | Trung bình | Giảm batch size, gradient accumulation, gradient checkpointing |
| Synthetic data làm giảm performance trên real | Trung bình | Curriculum: train synthetic 50% epochs đầu rồi giảm dần |

---

## 9. Checklist hoàn thành

### Data Preparation
- [ ] Tải DUTS-TR (~2GB), DUTS-TE (~500MB)
- [ ] Tải SmartDoc-QA (~150MB)
- [ ] Auto-label 1020 ảnh với ensemble rembg
- [ ] Manual verify 200-400 ảnh subset (eval set chất lượng cao)
- [ ] (Optional) Sinh 5000-10000 ảnh synthetic
- [ ] Phân chia train/val/test rõ ràng, no leakage

### Implementation
- [ ] U2NET architecture với RSU-7, RSU-6, RSU-5, RSU-4, RSU-4F
- [ ] U2NETp (lite) với channels nhỏ hơn
- [ ] Combo loss BCE + IoU + SSIM
- [ ] Unit test forward + backward pass cho cả 2 variant
- [ ] Dataset class với augmentation
- [ ] Training loop với AMP, logging, checkpointing
- [ ] Eval script với 4 metrics

### Training
- [ ] Stage 1 U2NET full trên DUTS-TR — mIoU DUTS-TE ≥ 0.87
- [ ] Stage 1 U2NETp trên DUTS-TR — mIoU DUTS-TE ≥ 0.84
- [ ] Stage 2 U2NET full trên doc data — mIoU verified ≥ 0.88
- [ ] Stage 2 U2NETp trên doc data — mIoU verified ≥ 0.85

### Ablation
- [ ] Thực hiện ≥ 8 ablation experiments (E1-E8)
- [ ] Hyperparameter sweep cho LR, batch, input size, loss weights
- [ ] Eval per-category trên 7 nhóm dataset

### Reporting
- [ ] Tạo figures: training curves, sample predictions, per-category
- [ ] Bảng so sánh ablation
- [ ] Bảng comparison U2NET full vs U2NETp
- [ ] Viết báo cáo Markdown đầy đủ

---

*Phase 1 hoàn thành = 2 variant U2NET trained, ablation đầy đủ, đạt mIoU mục tiêu.*
