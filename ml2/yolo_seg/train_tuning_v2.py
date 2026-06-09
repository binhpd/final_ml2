"""Train YOLOv11n-seg on the fast 1-class tuning_v2 dataset."""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml2/data/tuning_v2_fast/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="Base model to start from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    print(f"Loading YOLO model from: {args.model}")
    model = YOLO(args.model)

    print(f"Starting FAST training on dataset: {args.data}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project="runs/yolo_tuning_v2",
        name="train_1class_fast",
        exist_ok=True,
        # Data augmentation tuning to handle both flat and curved pages
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )
    
    # Save best model to exported_models
    best_weights = Path("runs/yolo_tuning_v2/train_1class_fast/weights/best.pt")
    if best_weights.exists():
        import shutil
        out_path = Path("exported_models/yolo_tuning_v2.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, out_path)
        print(f"\n✅ Successfully exported best model to {out_path}")
    else:
        print(f"\n❌ Error: Could not find best.pt at {best_weights}")

if __name__ == "__main__":
    main()
