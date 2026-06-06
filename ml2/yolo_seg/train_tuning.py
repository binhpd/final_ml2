"""Train YOLOv11n-seg on the split dataset with specific tuning hyperparameters.

Optimized for Apple Silicon M4 Max / MPS.
"""

import os
import argparse
from pathlib import Path

def main():
    # Force fallback to CPU for unsupported Metal operators to avoid crashes
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml2/data/tunning/Yolo Segmentation/split/data.yaml")
    ap.add_argument("--model", default="yolo11n-seg.pt", help="Pretrained model weights name or path")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32, help="Batch size (M4 Max 48GB can handle 32 or 64)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--project", default="ml2/runs/yolo")
    ap.add_argument("--name", default="yolo11n_seg_spine_exclusion")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from ultralytics import YOLO

    # Load model (pretrained weights or structure)
    model = YOLO(args.model)
    
    # Train model with user specified tuning parameters
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=False,          # Disable Mixed Precision to prevent NaN Loss on MPS
        workers=args.workers,
        
        # Hyperparameters for Data Augmentations
        degrees=15.0,        # Slight rotation
        perspective=0.0005,  # Perspective distortion
        translate=0.1,       # Image translation
        scale=0.5,           # Image scaling
        hsv_h=0.015,         # Hue alteration
        hsv_s=0.7,           # Saturation alteration
        hsv_v=0.4,           # Brightness/value alteration
        mosaic=1.0,          # Combine 4 images
        close_mosaic=20,     # Disable mosaic for last 20 epochs to clean mask boundaries
        copy_paste=0.3,      # Copy-paste augmentation for background variety
        fliplr=0.5,          # Horizontal flip
        
        # Hyperparameters for Loss Weights
        box=7.5,             # High weight on bounding box location
        dfl=1.5,             # High distribution focal loss weight for clean edges
        cls=0.5,             # Reduced class loss weight as we only have 2 simple classes
    )
    
    # Save the best model weight to checkpoints
    best_ckpt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_ckpt.exists():
        out_path = Path("ml2/checkpoints/yolo11n_seg_spine_exclusion_best.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_ckpt, out_path)
        print(f"[SUCCESS] Copied best weights to: {out_path}")

if __name__ == "__main__":
    main()
