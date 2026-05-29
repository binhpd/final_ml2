"""Export YOLO weights → ONNX (+ CoreML optional cho M4 Max)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--formats", nargs="+", default=["onnx"], choices=["onnx", "coreml", "torchscript"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    for fmt in args.formats:
        try:
            out = model.export(format=fmt, imgsz=args.imgsz)
            print(f"[ok] {fmt} -> {out}")
        except Exception as e:
            print(f"[fail] {fmt}: {e}")


if __name__ == "__main__":
    main()
