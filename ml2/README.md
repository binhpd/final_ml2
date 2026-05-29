# ml2/ — U²-Net + YOLO-Seg Document Segmentation

> Đồ án ML2 cuối kỳ — Nhóm 6 | Mac Studio M4 Max 48GB | Plan B

## Cấu trúc

```
ml2/
├── u2net/                  # U²-Netp lite (1.1M, 4.7MB)
│   ├── model.py            # RSU + U2NETp
│   ├── loss.py             # BCE + IoU + SSIM combo
│   ├── dataset.py          # SmartDoc / Doc3D / DocAligner loader
│   ├── augmentation.py     # Albumentations basic + strong
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── visualize.py
│   └── configs/
│       ├── doc_lite_planB.yaml      # Config chính
│       ├── doc_full_optional.yaml   # Cho CUDA users
│       └── mps_mini.yaml            # Test nhanh MPS
│
├── yolo_seg/               # YOLOv11n-seg (2.9M, 6MB)
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── eval.py
│   ├── visualize.py
│   ├── demo_viz.py
│   ├── infer_tta.py
│   └── export_all.py
│
├── pipeline_integration/   # Drop-in vào pipeline cũ
│   ├── u2net_wrapper.py
│   ├── yolo_wrapper.py
│   ├── pipeline_u2net.py
│   ├── pipeline_yolo.py
│   └── test_integration.py
│
├── benchmark/              # 4 chiều KPI
│   ├── kpi_speed.py
│   ├── kpi_accuracy.py
│   ├── kpi_robustness.py
│   ├── kpi_e2e.py
│   └── aggregate_results.py
│
├── scripts/                # Hỗ trợ
│   ├── download_datasets.py
│   ├── prepare_smartdoc.py
│   ├── prepare_doc3d.py
│   ├── prepare_docaligner.py
│   ├── build_dummy_data.py
│   ├── check_environment.py
│   └── caffeinate_train.sh
│
├── notebooks/              # 4 demo
├── checkpoints/            # .pth + .pt
├── data/                   # Datasets (gitignored)
├── results/                # KPI CSV + figures
└── requirements.txt
```

## Quick start

```bash
# 1. Setup
python -m venv venv_ml2
source venv_ml2/bin/activate
pip install -r ml2/requirements.txt

# 2. Verify
python ml2/scripts/check_environment.py

# 3. Test code với dummy data
python ml2/scripts/build_dummy_data.py --n 100
python ml2/u2net/train.py --config ml2/u2net/configs/mps_mini.yaml --dummy --epochs 1

# 4. Tải dataset thật (chạy đêm)
python ml2/scripts/download_datasets.py --smartdoc --doc3d
python ml2/scripts/prepare_smartdoc.py
python ml2/scripts/prepare_doc3d.py

# 5. Train (caffeinate chống sleep)
caffeinate -i python ml2/u2net/train.py --config ml2/u2net/configs/doc_lite_planB.yaml
caffeinate -i python ml2/yolo_seg/train.py --epochs 150 --device mps

# 6. Eval + Benchmark
python ml2/u2net/eval.py --ckpt ml2/checkpoints/u2netp_doc.pth
python ml2/yolo_seg/eval.py --weights ml2/checkpoints/yolo11n_seg_doc.pt
python ml2/benchmark/aggregate_results.py
```

## KPI mục tiêu

| Metric | rembg baseline | U²-Netp lite | YOLOv11n-seg |
|--------|----------------|--------------|--------------|
| mIoU | 0.78 | ≥ 0.83 | ≥ 0.81 |
| F1 | 0.82 | ≥ 0.87 | ≥ 0.85 |
| FPS (MPS) | 8 | ≥ 20 | ≥ 35 |
| Size | 176MB | 4.7MB | 6MB |

## Pipeline

Step 1 (Detection) ← **Code mới ở đây** → Step 2 (Warp) → Step 3 (Enhance)

→ Xem `docs_ml2/01_KeHoach.md`, `02_Spec_KyThuat.md`, `03_Research_Note.md`.
