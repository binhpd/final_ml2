# Kế hoạch Tổng quan — Train U2NET + YOLO-Seg cho Document Segmentation

> **Mục tiêu cốt lõi:** Train 2 model Deep Learning **U2NET** và **YOLO-Seg** cho bài toán document segmentation, sau đó tích hợp vào pipeline + benchmark KPI

---

## A. Mục tiêu kép

| Task | Model | Output | Vai trò trong pipeline |
|---|---|---|---|
| **Task A** | **U2NET (full + lite, train from scratch)** | Mask nhị phân (tách văn bản khỏi nền) | Thay `rembg` trong Step 1 luồng U²-Net |
| **Task B** | **YOLO-Seg (multi-size, fine-tune + scratch)** | Mask + Bounding Box + Visualization | Thay U2NET, kèm vẽ kết quả trực quan |

### Sự khác biệt cốt lõi 2 model

| Tiêu chí | U2NET | YOLO-Seg |
|---|---|---|
| Loại task | Salient Object Detection | Instance Segmentation |
| Output | 1 mask duy nhất | Multi-instance: mask + bbox + class + conf |
| Training | **From scratch** (2 stage) | Fine-tune COCO + ablation from-scratch |
| Architecture | Nested U-Net (RSU blocks) | CSPNet + PAN-FPN + ProtoNet |
| Tốc độ inference | Trung bình (~5-50ms) | Nhanh (~3-30ms) |
| Khả năng detect multi-doc | Không | Có |
| Visualization sẵn | Không | Có (built-in) |

---

## B. Cấu trúc 4 Phase

```
        ┌─────────────────────────────────────────────┐
        │ PHASE 4: KPI Benchmark & So sánh            │
        │ ─────────────────────────────────────       │
        │ Speed (CPU/GPU/Mobile)                       │
        │ Accuracy (mIoU/F1/MAE/Boundary-F1/mAP)      │
        │ Robustness (per 7 categories)                │
        │ E2E Pipeline Impact (PSNR/SSIM/OCR-CER)     │
        └────────────────────┬────────────────────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │ PHASE 3: Integration                        │
        │ ─────────────────────────────────────       │
        │ • U2NET → pipeline cũ (thay rembg)          │
        │ • YOLO → thay U2NET + Visualization         │
        └────────────────────┬────────────────────────┘
                             │ uses
        ┌────────────────────▼────────────────────────┐
        │ PHASE 2: Build & Train YOLO-Seg             │
        │ ─────────────────────────────────────       │
        │ • 3 variant: YOLOv11n / v11s / v11m         │
        │ • Fine-tune COCO + ablation from-scratch    │
        │ • Visualization module                       │
        └────────────────────┬────────────────────────┘
                             │ parallel possible
        ┌────────────────────▼────────────────────────┐
        │ PHASE 1: Build & Train U2NET (TRỌNG TÂM)   │
        │ ─────────────────────────────────────       │
        │ • 2 variant: U2NET full + U2NETp lite       │
        │ • 2 Stage: DUTS-TR pretrain → Doc fine-tune │
        │ • Combo loss BCE + IoU + SSIM               │
        └─────────────────────────────────────────────┘
```

---

## C. Lịch trình (linh hoạt — không ràng buộc thời gian cứng)

### Timeline tham khảo

| Tuần | Phase | Hoạt động chính | Output |
|---|---|---|---|
| 1 | Setup | Môi trường + tải DUTS-TR + SmartDoc + auto-label 1020 ảnh | Datasets sẵn sàng |
| 2 | P1.1 | Implement U2NET + U2NETp architecture, unit tests | Code model |
| 3-4 | P1.2 | **Train U2NET Stage 1 (DUTS-TR)** — cả full + lite | 2 checkpoint stage 1 |
| 5 | P1.3 | **Train U2NET Stage 2 (Doc data)** — cả full + lite | 2 checkpoint stage 2 |
| 6 | P1.4 | Ablation studies + tinh chỉnh + evaluation per-category | Bảng ablation |
| 7 | P2.1 | Dataset prep cho YOLO (mask → polygon, 3 splits) | YOLO format data |
| 8-9 | P2.2 | **Train YOLO-Seg** 3 variants (n/s/m) + ablation | 3 checkpoint |
| 10 | P2.3 | Visualization module + demo generation | Viz module + demos |
| 11 | P3 | Integration cả 2 model vào pipeline | 2 pipeline scripts |
| 12 | P4 | KPI Benchmark 4 chiều + báo cáo so sánh | Report final |

→ **Linh hoạt**: Phase 1 (U2NET) và Phase 2 (YOLO) có thể chạy song song nếu có 2 GPU/máy.

---

## D. Hardware Configuration tham khảo

Plan này không ràng buộc hardware cụ thể. Các option phổ biến:

### Option 1: Single GPU (recommended)
- NVIDIA RTX 3060/3070/3080/3090/4090
- Linux/Windows, CUDA 11.8+
- VRAM: 8-24GB
- Train time U2NET full Stage 1: ~3 ngày
- Train time YOLOv11s 200 epoch: ~12 giờ

### Option 2: Multi-GPU
- 2-4× RTX/A100/H100
- DDP (DistributedDataParallel)
- Train time U2NET full Stage 1: ~1 ngày
- Train time YOLOv11m 150 epoch: ~6 giờ

### Option 3: Cloud (Colab Pro / Kaggle / Vast.ai)
- Colab Pro+ A100 40GB
- Kaggle TPU v3-8 (cần adapt code)
- Vast.ai RTX 3090 ~$0.3/giờ
- Train Stage 1 mất ~50-80 giờ compute

### Option 4: Apple Silicon (MPS)
- M1/M2/M3 Pro/Max/Ultra với unified memory 32GB+
- PyTorch MPS backend
- Khả thi cho U2NETp (lite) + YOLOv11n/s
- Cần `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Train time U2NETp Stage 1: ~2-3 ngày

---

## E. Cấu trúc thư mục code

```
Nhóm 6/
├─ Pipeline With ML/             # Pipeline hiện có (giữ nguyên)
├─ docs/                         # Tài liệu gốc của nhóm
├─ docs_ml2/                     # ← Kế hoạch + báo cáo task này
│  ├─ 00_TongHop_Claude.md       # Master index
│  ├─ KeHoach_TongQuan_Claude.md # File này
│  ├─ 01_U2NET_ChiTiet_Claude.md
│  ├─ 02_YOLO_ChiTiet_Claude.md
│  └─ 03_Integration_KPI_Claude.md
│
└─ ml2/                          # ← THƯ MỤC CODE MỚI
   ├─ datasets/
   │  ├─ duts_tr/                # DUTS-TR cho U2NET Stage 1
   │  ├─ duts_te/                # DUTS-TE eval
   │  ├─ smartdoc_qa/            # SmartDoc-QA
   │  ├─ midv500/                # Optional ID documents
   │  ├─ nhom6_1020/             # 1020 ảnh nhóm + auto/verified masks
   │  ├─ synthetic/              # Optional synthetic
   │  └─ yolo_format/            # Format YOLO (images/labels)
   │
   ├─ u2net/
   │  ├─ model.py                # U2NET + U2NETp
   │  ├─ loss.py                 # Combo loss
   │  ├─ dataset.py              # DocSegDataset
   │  ├─ augmentation.py         # Strong aug pipeline
   │  ├─ train.py                # Training loop
   │  ├─ eval.py                 # 4 metrics
   │  ├─ infer.py                # TTA + CRF
   │  ├─ visualize.py            # Training curves
   │  ├─ configs/                # 4 YAML configs
   │  ├─ runs/                   # TensorBoard logs
   │  └─ weights/                # 4 final checkpoints
   │
   ├─ yolo_seg/
   │  ├─ prepare_dataset.py      # Mask → YOLO polygon
   │  ├─ train.py                # Training wrapper
   │  ├─ sweep.py                # Hyperparameter sweep
   │  ├─ eval.py                 # Custom evaluation
   │  ├─ visualize.py            # Bbox + mask + corners
   │  ├─ demo_viz.py             # Batch viz demo
   │  ├─ infer_tta.py            # TTA inference
   │  ├─ export_all.py           # Multi-format export
   │  ├─ runs/                   # YOLO logs
   │  ├─ weights/                # 3+ checkpoints
   │  └─ demos/                  # Visualization output
   │
   ├─ pipeline_integration/
   │  ├─ u2net_wrapper.py        # Drop-in replacement cho rembg
   │  ├─ yolo_wrapper.py         # YOLO wrapper với viz
   │  ├─ pipeline_u2net.py       # Pipeline với U2NET
   │  └─ pipeline_yolo.py        # Pipeline với YOLO + viz
   │
   ├─ benchmark/
   │  ├─ kpi_speed.py
   │  ├─ kpi_accuracy.py
   │  ├─ kpi_robustness.py
   │  ├─ kpi_e2e.py
   │  └─ results/
   │     ├─ benchmark.csv
   │     ├─ benchmark.json
   │     └─ figures/
   │
   ├─ scripts/
   │  ├─ autolabel_nhom6.py      # Auto-label dùng rembg ensemble
   │  ├─ verify_masks.py         # Manual verify helper
   │  └─ synthesize_documents.py # Synthetic data generator
   │
   └─ requirements.txt
```

---

## F. Setup môi trường

### F.1 Python environment

```bash
# Tạo venv
cd "Nhóm 6"
python3 -m venv venv_ml2
source venv_ml2/bin/activate   # Linux/Mac
# venv_ml2\Scripts\activate    # Windows

# PyTorch (chọn version phù hợp)
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU/MPS only:
pip install torch torchvision torchaudio

# YOLO framework
pip install ultralytics

# Image processing + augmentation
pip install opencv-python pillow scikit-image albumentations

# Loss SSIM
pip install pytorch-msssim

# Metrics
pip install scikit-learn scipy

# Dataset tools
pip install kaggle gdown datasets

# Logging
pip install tensorboard wandb tqdm

# Auto-labeling
pip install rembg[gpu]    # rembg[cpu] nếu không có GPU

# Post-processing (optional)
pip install pydensecrf    # CRF refinement

# Export
pip install onnx onnxruntime onnxruntime-gpu
pip install openvino-dev
pip install coremltools   # Cho macOS
```

### F.2 Verify hardware

```python
import torch

# CUDA check
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# MPS check (Apple Silicon)
print(f"MPS available: {torch.backends.mps.is_available()}")

# Test tensor
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(2, 3, 256, 256).to(device)
print(f"Tensor on {device}: {x.shape}")
```

### F.3 Dataset download links

| Dataset | Size | URL |
|---|---|---|
| DUTS-TR | ~2GB | http://saliencydetection.net/duts/ |
| DUTS-TE | ~500MB | (cùng nguồn) |
| SmartDoc-QA | ~150MB | https://smartdoc.univ-lr.fr/ |
| MIDV-500 | ~200MB | https://github.com/SmartEngines/midv-500 |
| MIT Indoor Scenes (backgrounds) | ~1.5GB | http://web.mit.edu/torralba/www/indoor.html |
| Nhóm 6 1020 ảnh | local | đã có sẵn |

---

## G. Mục tiêu metrics tổng

### G.1 U2NET Phase 1

| Variant | Stage | mIoU | F1 | BF |
|---|---|---|---|---|
| U2NET full | Stage 1 (DUTS-TE) | ≥ 0.87 | ≥ 0.90 | ≥ 0.78 |
| U2NET full | Stage 2 (Doc verified) | ≥ 0.88 | ≥ 0.91 | ≥ 0.82 |
| U2NETp lite | Stage 1 (DUTS-TE) | ≥ 0.84 | ≥ 0.87 | ≥ 0.74 |
| U2NETp lite | Stage 2 (Doc verified) | ≥ 0.85 | ≥ 0.88 | ≥ 0.78 |

### G.2 YOLO-Seg Phase 2

| Variant | mAP@0.5 (box) | mAP@0.5 (mask) | mIoU vs GT | FPS GPU |
|---|---|---|---|---|
| YOLOv11n-seg | ≥ 0.88 | ≥ 0.84 | ≥ 0.83 | ≥ 130 |
| YOLOv11s-seg | ≥ 0.92 | ≥ 0.88 | ≥ 0.86 | ≥ 80 |
| YOLOv11m-seg | ≥ 0.94 | ≥ 0.90 | ≥ 0.88 | ≥ 40 |

### G.3 KPI Benchmark Phase 4

| Tiêu chí | Mục tiêu |
|---|---|
| Pipeline U2NET vs rembg (E2E speed) | ≥ +20% nhanh hơn |
| Pipeline YOLO vs U2NET (E2E speed) | ≥ +30% nhanh hơn |
| OCR-CER (Vietnamese) U2NET | < 7.5% |
| OCR-CER (Vietnamese) YOLO | < 8.0% |
| Robustness (worst category mIoU) | ≥ 0.70 |

---

## H. Decision tree khi gặp vấn đề

```
U2NET Stage 1 không converge?
  ├─ YES → Giảm LR 5×, check data loading, gradient clipping
  └─ NO  → Tiếp tục

U2NET Stage 2 overfit nhanh?
  ├─ YES → Early stopping, weight decay 1e-4, augmentation mạnh hơn
  └─ NO  → Tiếp tục

YOLO mAP@0.5 < 0.85 sau 100 epoch?
  ├─ YES → Tăng model size (n → s → m), tăng imgsz 640 → 1024
  └─ NO  → Tinh chỉnh thêm

Boundary F1 < 0.75?
  ├─ YES → Thêm EdgeLoss, tăng SSIM weight, dùng CRF post-process
  └─ NO  → Acceptable

OOM trên GPU/MPS?
  ├─ YES → Giảm batch size, gradient accumulation, AMP FP16
  └─ NO  → Tiếp tục

Auto-label 1020 ảnh quá noisy?
  ├─ YES → Tăng manual verify (300-400 ảnh), ensemble 3 rembg models
  └─ NO  → Tiếp tục

Synthetic data làm giảm performance?
  ├─ YES → Curriculum: train synthetic 50% epochs đầu rồi giảm dần
  └─ NO  → Tiếp tục
```

---

## I. Deliverables tổng kết

| Loại | Số lượng | Vị trí |
|---|---|---|
| **Code** | 4 module chính (u2net, yolo_seg, integration, benchmark) | `ml2/` |
| **Pretrained weights** | 4 U2NET (full×2 + lite×2) + 3 YOLO (n/s/m) | `ml2/*/weights/` |
| **Exported models** | ONNX, CoreML, OpenVINO, TFLite | `ml2/yolo_seg/weights/` |
| **Báo cáo Markdown** | 4 file chính + 1 final report | `docs_ml2/` |
| **Visualization outputs** | 50+ ảnh viz, 1 grid montage | `ml2/yolo_seg/demos/` |
| **Training curves** | 7+ training runs với TensorBoard | `ml2/*/runs/` |
| **KPI CSV** | 4 bảng KPI | `ml2/benchmark/results/` |
| **Ablation tables** | 8 cho U2NET + 9 cho YOLO | trong báo cáo |

---

## J. Tài liệu chi tiết

Xem các file riêng:
- **[01_U2NET_ChiTiet_Claude.md](./01_U2NET_ChiTiet_Claude.md)** — Kế hoạch chi tiết build U2NET (2 variant, 2 stage, train from scratch)
- **[02_YOLO_ChiTiet_Claude.md](./02_YOLO_ChiTiet_Claude.md)** — Kế hoạch chi tiết train YOLO-Seg (3 sizes, fine-tune + scratch)
- **[03_Integration_KPI_Claude.md](./03_Integration_KPI_Claude.md)** — Integration + KPI Benchmark Protocol 4 chiều
- **[00_TongHop_Claude.md](./00_TongHop_Claude.md)** — Master index của toàn bộ tài liệu

---

*Kế hoạch tổng quan training-focused. Linh hoạt theo hardware và thời gian thực tế.*
