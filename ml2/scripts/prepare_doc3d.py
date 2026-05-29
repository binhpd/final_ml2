"""Convert Doc3D raw → format chuẩn (subset 5K).

Raw structure (sau khi tải foreground masks + images):
  ml2/data/doc3d/raw/
    img/<id>.png
    fg/<id>.png         (foreground mask = doc region)

Output (chuẩn DocSegDataset):
  ml2/data/doc3d/
    images/<id>.png
    masks/<id>.png
    train.txt val.txt test.txt
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ml2/data/doc3d")
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    raw = root / "raw"
    if not raw.exists():
        print(f"[!] Không tìm thấy {raw}")
        print("Chạy download_datasets.py --doc3d trước, sau đó extract vào ml2/data/doc3d/raw/")
        return

    img_src = raw / "img"
    fg_src = raw / "fg"
    img_dst = root / "images"
    mask_dst = root / "masks"
    img_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    if not img_src.exists() or not fg_src.exists():
        print(f"[!] Cần {img_src} và {fg_src} (img + fg)")
        return

    all_imgs = sorted(img_src.glob("*.png")) + sorted(img_src.glob("*.jpg"))
    print(f"[info] Tìm thấy {len(all_imgs)} ảnh raw")

    candidates = []
    for p in all_imgs:
        fg = fg_src / f"{p.stem}.png"
        if not fg.exists():
            fg = fg_src / f"{p.stem}.exr"
        if fg.exists():
            candidates.append((p, fg))
    random.shuffle(candidates)
    candidates = candidates[:args.subset]
    print(f"[info] Chọn {len(candidates)} samples")

    names = []
    for img_path, fg_path in candidates:
        name = img_path.stem
        # Copy image
        shutil.copy2(img_path, img_dst / f"{name}.png")

        # Convert fg → binary mask
        if fg_path.suffix == ".exr":
            try:
                fg_arr = cv2.imread(str(fg_path), cv2.IMREAD_UNCHANGED)
                if fg_arr is None:
                    continue
                if fg_arr.ndim == 3:
                    fg_arr = fg_arr[..., 0]
            except Exception:
                continue
        else:
            fg_arr = cv2.imread(str(fg_path), cv2.IMREAD_GRAYSCALE)
            if fg_arr is None:
                continue

        mask = (fg_arr > 0).astype(np.uint8) * 255
        cv2.imwrite(str(mask_dst / f"{name}.png"), mask)
        names.append(name)

    # 90/5/5 split (synthetic - no leakage worry)
    random.shuffle(names)
    n = len(names)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)
    (root / "train.txt").write_text("\n".join(names[:n_train]))
    (root / "val.txt").write_text("\n".join(names[n_train:n_train + n_val]))
    (root / "test.txt").write_text("\n".join(names[n_train + n_val:]))
    print(f"[done] train={n_train} val={n_val} test={n - n_train - n_val}")


if __name__ == "__main__":
    main()
