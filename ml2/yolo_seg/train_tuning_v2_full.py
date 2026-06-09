from ultralytics import YOLO
import os
from pathlib import Path

def main():
    # Model configuration
    model_name = "yolo11n-seg.pt"
    
    # Path to dataset (from full blended dataset)
    data_yaml = "/Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/ml2/data/tuning_v2_full/dataset.yaml"
    
    if not Path(data_yaml).exists():
        print(f"Error: {data_yaml} not found!")
        print("Please run 'python ml2/scripts/prepare_tuning_v2_full.py' first.")
        return

    print("="*50)
    print("🚀 HARDWARE OPTIMIZED TRAINING (MAC STUDIO M4 MAX)")
    print("="*50)
    
    model = YOLO(model_name)

    # Khởi động training với các tham số ép xung
    results = model.train(
        data=data_yaml,
        epochs=60,                  # Giới hạn epochs để hoàn thành trong 5 tiếng
        patience=15,                # Dừng sớm nếu không hội tụ thêm
        batch=96,                   # Tận dụng tối đa 48GB Unified Memory
        imgsz=640,
        device="mps",               # Sử dụng Metal Performance Shaders
        amp=True,                   # Bật Mixed Precision (FP16) để tăng gấp đôi tốc độ
        workers=14,                 # M4 Max có 16 cores, dùng 14 cores để load data
        cache=False,                # Tắt cache RAM tránh OOM, SSD Mac Studio đủ nhanh (7GB/s)
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        mosaic=1.0,
        mixup=0.15,
        box=7.5,
        cls=0.5,
        project="runs/segment",
        name="tuning_v2_full_hw_optimized",
        exist_ok=True,
        verbose=True
    )

    print("="*50)
    print("✅ Training Complete!")
    
    # Copy best weights to exported_models
    best_weights = Path(f"runs/segment/tuning_v2_full_hw_optimized/weights/best.pt")
    if best_weights.exists():
        export_dir = Path("exported_models")
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / "yolo_tuning_v2_full.pt"
        
        import shutil
        shutil.copy2(best_weights, export_path)
        print(f"Best weights saved to: {export_path}")

if __name__ == "__main__":
    main()
