"""Batch demo + grid montage cho YOLO visualization."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ml2.yolo_seg.visualize import YOLODocVisualizer


def make_grid(images: list[np.ndarray], cols: int = 3, tile: int = 480) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * tile, cols * tile, 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        resized = cv2.resize(im, (tile, tile))
        canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = resized
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default="ml2/results/yolo_demo_grid.jpg")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    viz = YOLODocVisualizer(args.weights, device=args.device)
    paths = sorted(Path(args.input_dir).glob("*.*"))[:args.n]
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        images.append(viz.visualize(img))
    if not images:
        print("[!] Không có ảnh nào")
        return
    grid = make_grid(images, args.cols)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
