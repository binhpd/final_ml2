"""Visualize U²-Net predictions + training curves."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from ml2.u2net.infer import infer_single, load_model


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.4, color=(0, 255, 0)) -> np.ndarray:
    color_arr = np.array(color, dtype=np.uint8)
    overlay = img.copy()
    bin_mask = (mask > 0.5).astype(np.uint8)
    for c in range(3):
        overlay[..., c] = np.where(bin_mask > 0, (1 - alpha) * img[..., c] + alpha * color_arr[c], img[..., c])
    return overlay


def grid_sample(model: torch.nn.Module, img_paths: list[Path], device, size: int = 320) -> np.ndarray:
    """Tạo grid 3 cột: image | mask | overlay."""
    rows = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_show = cv2.resize(img, (size, size))
        mask = infer_single(model, img, device, size)
        mask_show = (cv2.resize(mask, (size, size)) * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_show, cv2.COLOR_GRAY2BGR)
        ov = overlay(img_show, cv2.resize(mask, (size, size)))
        rows.append(np.hstack([img_show, mask_bgr, ov]))
    return np.vstack(rows) if rows else np.zeros((10, 10, 3), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", default="ml2/results/u2net_grid.png")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    model = load_model(args.ckpt, device)
    paths = sorted(Path(args.input_dir).glob("*.*"))[:args.n]

    grid = grid_sample(model, paths, device, args.size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
