"""Test-Time Augmentation cho YOLO seg - horizontal flip + multi-scale."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="ml2/results/yolo_tta")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    inp = Path(args.input)
    paths = sorted(inp.glob("*.*")) if inp.is_dir() else [inp]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # TTA: original + flipped
        r1 = model.predict(img, device=args.device, imgsz=args.imgsz, augment=True, verbose=False)
        flipped = cv2.flip(img, 1)
        r2 = model.predict(flipped, device=args.device, imgsz=args.imgsz, verbose=False)

        mask_acc = np.zeros((h, w), dtype=np.float32)

        def add_masks(results, flip: bool):
            if results and results[0].masks is not None:
                for m in results[0].masks.data.cpu().numpy():
                    mm = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    if flip:
                        mm = cv2.flip(mm, 1)
                    nonlocal mask_acc
                    mask_acc += mm

        add_masks(r1, False)
        add_masks(r2, True)

        final = (mask_acc >= 1).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{p.stem}_tta.png"), final)
        print(f"[ok] {p.name}")


if __name__ == "__main__":
    main()
