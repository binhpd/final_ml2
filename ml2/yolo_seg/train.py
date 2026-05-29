"""Train YOLOv11n-seg cho document segmentation (Ultralytics wrapper).

Fine-tune từ COCO pretrained weight. M4 Max MPS support sẵn từ Ultralytics 8.3+.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml2/data/yolo_doc/doc.yaml")
    ap.add_argument("--model", default="yolo11n-seg.pt", help="Pretrained weight name hoặc path")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--project", default="ml2/runs/yolo")
    ap.add_argument("--name", default="doc_planB")
    ap.add_argument("--from_scratch", action="store_true")
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.dummy:
        args.data = "ml2/data/dummy/dummy.yaml"
        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        Path(args.data).write_text(
            "path: ml2/data/dummy\ntrain: images\nval: images\nnc: 1\nnames:\n  0: document\n"
        )
        args.epochs = 1
        args.imgsz = 256

    weights = args.model if not args.from_scratch else "yolo11n-seg.yaml"
    model = YOLO(weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # Augmentation hợp cho document
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        close_mosaic=20,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Hardware
        workers=4,
        pretrained=not args.from_scratch,
    )
    # Save best weight to checkpoints/
    ckpt = Path(args.project) / args.name / "weights" / "best.pt"
    if ckpt.exists():
        out = Path("ml2/checkpoints/yolo11n_seg_doc.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(ckpt, out)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
