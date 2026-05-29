"""Regenerate train/val/test splits cho Doc3D từ images đã có sẵn."""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ml2/data/doc3d")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "images"
    mask_dir = root / "masks"
    names = []
    for p in img_dir.glob("*.png"):
        if (mask_dir / f"{p.stem}.png").exists():
            names.append(p.stem)
    random.seed(args.seed)
    random.shuffle(names)
    n = len(names)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)
    (root / "train.txt").write_text("\n".join(names[:n_train]))
    (root / "val.txt").write_text("\n".join(names[n_train:n_train + n_val]))
    (root / "test.txt").write_text("\n".join(names[n_train + n_val:]))
    print(f"total={n} train={n_train} val={n_val} test={n - n_train - n_val}")


if __name__ == "__main__":
    main()
