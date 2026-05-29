"""Convert DocAlign12K → format chuẩn (optional pretrain).

DocAligner samples gồm:
  - captured.png    (ảnh giấy chụp)
  - clean.png       (ảnh giấy phẳng/gốc)
  - flow.npy        (flow field)

Output (chỉ giữ captured + binary mask từ flow region):
  ml2/data/docaligner/
    images/<id>.png
    masks/<id>.png
    train.txt val.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def flow_to_mask(flow_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Convert flow field → binary mask: pixel có flow != 0 = doc region."""
    flow = np.load(flow_path)
    if flow.ndim == 3:
        magnitude = np.linalg.norm(flow, axis=-1)
    else:
        magnitude = np.abs(flow)
    mask = (magnitude > 0.1).astype(np.uint8) * 255
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ml2/data/docaligner")
    ap.add_argument("--subset", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    raw = root / "repo" / "DocAlign12K"
    if not raw.exists():
        raw = root / "raw"
    if not raw.exists():
        print(f"[!] Không tìm thấy DocAlign12K data")
        print("Theo README DocAligner repo để generate:")
        print(f"  cd {root / 'repo'} && python synthv2.py")
        return

    img_dst = root / "images"
    mask_dst = root / "masks"
    img_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    # Walk samples
    samples = sorted(raw.rglob("captured.png"))
    if not samples:
        samples = sorted(raw.rglob("*.png"))
    random.shuffle(samples)
    samples = samples[:args.subset]
    print(f"[info] Chọn {len(samples)} samples")

    names = []
    for cap_path in samples:
        flow_path = cap_path.parent / "flow.npy"
        img = cv2.imread(str(cap_path))
        if img is None:
            continue
        if flow_path.exists():
            mask = flow_to_mask(flow_path, img.shape[:2])
        else:
            # Fallback: alpha channel hoặc whole-image mask
            mask = np.full(img.shape[:2], 255, dtype=np.uint8)

        name = f"docalign_{cap_path.parent.name}"
        cv2.imwrite(str(img_dst / f"{name}.png"), img)
        cv2.imwrite(str(mask_dst / f"{name}.png"), mask)
        names.append(name)

    random.shuffle(names)
    n_train = int(len(names) * 0.95)
    (root / "train.txt").write_text("\n".join(names[:n_train]))
    (root / "val.txt").write_text("\n".join(names[n_train:]))
    print(f"[done] train={n_train} val={len(names) - n_train}")


if __name__ == "__main__":
    main()
