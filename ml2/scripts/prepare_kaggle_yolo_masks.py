"""Convert Kaggle 'document-image-segmentation-yolo-masks' → DocSegDataset format.

Source:
  ml2/data/yolo_masks_kaggle/raw/raw_data/{images,masks,labels}/<id>.{jpg,png,txt}

Output:
  ml2/data/kaggle_real/
    images/<id>.jpg
    masks/<id>.png
    labels/<id>.txt    (remap class -> 0)
    train.txt val.txt test.txt    (random split 80/10/10)
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="ml2/data/yolo_masks_kaggle/raw/raw_data")
    ap.add_argument("--out", default="ml2/data/kaggle_real")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    img_out = out / "images"
    mask_out = out / "masks"
    lbl_out = out / "labels"
    for d in [img_out, mask_out, lbl_out]:
        d.mkdir(parents=True, exist_ok=True)

    img_src = src / "images"
    mask_src = src / "masks"
    lbl_src = src / "labels"

    names = []
    for img_p in tqdm(sorted(img_src.glob("*.jpg"))):
        stem = img_p.stem
        mask_p = mask_src / f"{stem}.png"
        lbl_p = lbl_src / f"{stem}.txt"
        if not mask_p.exists() or not lbl_p.exists():
            continue
        # Copy image + mask
        shutil.copy2(img_p, img_out / img_p.name)
        shutil.copy2(mask_p, mask_out / mask_p.name)
        # Remap class id -> 0
        parts = lbl_p.read_text().strip().split()
        if len(parts) < 9:
            continue
        parts[0] = "0"
        (lbl_out / lbl_p.name).write_text(" ".join(parts))
        names.append(stem)

    random.seed(args.seed)
    random.shuffle(names)
    n = len(names)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    (out / "train.txt").write_text("\n".join(names[:n_train]))
    (out / "val.txt").write_text("\n".join(names[n_train:n_train + n_val]))
    (out / "test.txt").write_text("\n".join(names[n_train + n_val:]))
    print(f"[done] {n} samples | train={n_train} val={n_val} test={n - n_train - n_val} -> {out}")


if __name__ == "__main__":
    main()
