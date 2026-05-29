"""Convert SmartDoc2-Images (carlosaranda/smartdoc2images) → DocSegDataset format.

Source structure:
  ml2/data/smartdoc2/raw/smartdoc/
    data/<filename>.jpg
    labels/{train,val,test}.json    (COCO format with 4 keypoints [bl, tl, tr, br])

Output:
  ml2/data/smartdoc/
    images/<id>.jpg
    masks/<id>.png
    labels/<id>.txt        (YOLO polygon normalized)
    train.txt val.txt test.txt
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def keypoints_to_polygon(keypoints: list[int]) -> np.ndarray:
    """Convert COCO keypoints [bl_x, bl_y, vis, tl_x, tl_y, vis, tr_x, tr_y, vis, br_x, br_y, vis]
    -> polygon in cyclic order tl -> tr -> br -> bl."""
    bl = (keypoints[0], keypoints[1])
    tl = (keypoints[3], keypoints[4])
    tr = (keypoints[6], keypoints[7])
    br = (keypoints[9], keypoints[10])
    return np.array([tl, tr, br, bl], dtype=np.int32)


def process_split(coco_json: Path, img_src: Path, out_root: Path, split: str, max_n: int | None = None):
    with open(coco_json) as f:
        d = json.load(f)

    img_by_id = {im["id"]: im for im in d["images"]}
    names = []

    img_dir = out_root / "images"
    mask_dir = out_root / "masks"
    label_dir = out_root / "labels"
    for dd in [img_dir, mask_dir, label_dir]:
        dd.mkdir(parents=True, exist_ok=True)

    anns = d["annotations"]
    if max_n:
        anns = anns[:max_n]

    for ann in tqdm(anns, desc=split):
        im = img_by_id.get(ann["image_id"])
        if im is None:
            continue
        src_img = img_src / im["file_name"]
        if not src_img.exists():
            continue
        stem = src_img.stem
        h, w = im["height"], im["width"]
        poly = keypoints_to_polygon(ann["keypoints"])

        # Copy image
        dst_img = img_dir / f"{stem}.jpg"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Render mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)

        # YOLO polygon label normalized
        poly_norm = poly.astype(np.float32) / np.array([w, h])
        with open(label_dir / f"{stem}.txt", "w") as fp:
            fp.write("0 " + " ".join(f"{v:.6f}" for pt in poly_norm for v in pt))

        names.append(stem)

    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="ml2/data/smartdoc2/raw/smartdoc")
    ap.add_argument("--out", default="ml2/data/smartdoc")
    ap.add_argument("--max_per_split", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img_src = src / "data"
    lbl_src = src / "labels"

    splits = {}
    for sp in ["train", "val", "test"]:
        coco = lbl_src / f"{sp}.json"
        if not coco.exists():
            print(f"[skip] {coco} không tồn tại")
            continue
        names = process_split(coco, img_src, out, sp, args.max_per_split)
        splits[sp] = names
        (out / f"{sp}.txt").write_text("\n".join(names))
        print(f"[{sp}] {len(names)} samples saved")

    total = sum(len(v) for v in splits.values())
    print(f"\n[done] Tổng {total} ảnh -> {out}")


if __name__ == "__main__":
    main()
