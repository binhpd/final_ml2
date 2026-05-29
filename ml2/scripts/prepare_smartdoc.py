"""Convert SmartDoc ICDAR 2015 raw → format chuẩn.

Raw structure (sau khi tải):
  ml2/data/smartdoc/raw/
    background01/datasetXX/
      *.avi          (video clip)
      *.gt           (XML ground truth với 4-corner mỗi frame)

Output:
  ml2/data/smartdoc/
    images/<bg>_<doc>_<frame>.jpg
    masks/<bg>_<doc>_<frame>.png    (mask polygon từ 4 góc)
    labels/<bg>_<doc>_<frame>.txt   (YOLO polygon normalized)
    train.txt val.txt test.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np


def parse_gt_xml(gt_path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    """Parse SmartDoc .gt file - return list of (frame_idx, 4 corners)."""
    tree = ET.parse(gt_path)
    root = tree.getroot()
    frames = []
    for frame in root.iter("frame"):
        idx = int(frame.attrib.get("index", -1))
        pts = frame.find("./object/point") if frame.find("./object") is not None else None
        # SmartDoc format có 4 'point' tag
        points = []
        obj = frame.find("./object")
        if obj is None:
            continue
        for p in obj.findall("point"):
            x = float(p.attrib["x"])
            y = float(p.attrib["y"])
            points.append((x, y))
        if len(points) >= 4:
            frames.append((idx, points[:4]))
    return frames


def extract_frames(video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    target = set(frame_indices)
    out: dict[int, np.ndarray] = {}
    idx = 0
    while cap.isOpened() and idx <= max(target):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target:
            out[idx] = frame
        idx += 1
    cap.release()
    return out


def make_mask(corners: list[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    pts = np.array(corners, dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ml2/data/smartdoc")
    ap.add_argument("--n_per_video", type=int, default=15, help="Số frame trích / video")
    ap.add_argument("--total", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    raw = root / "raw"
    img_dir = root / "images"
    mask_dir = root / "masks"
    label_dir = root / "labels"
    for d in [img_dir, mask_dir, label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not raw.exists():
        print(f"[!] Không tìm thấy {raw}")
        print("Chạy download_datasets.py --smartdoc trước.")
        return

    # Walk to find video + gt
    pairs: list[tuple[Path, Path, str]] = []
    for video in raw.rglob("*.avi"):
        gt = video.with_suffix(".gt")
        if not gt.exists():
            gt = video.with_suffix(".xml")
        if not gt.exists():
            continue
        bg = video.parent.parent.name if video.parent.parent.exists() else "bg_unknown"
        pairs.append((video, gt, bg))

    if not pairs:
        print(f"[!] Không tìm thấy file .avi trong {raw}")
        return

    print(f"[info] Tìm thấy {len(pairs)} video")

    saved_per_bg: dict[str, list[str]] = {}
    saved = 0
    for video, gt, bg in pairs:
        if saved >= args.total:
            break
        frames_info = parse_gt_xml(gt)
        if not frames_info:
            continue
        sample = random.sample(frames_info, min(args.n_per_video, len(frames_info)))
        indices = [s[0] for s in sample]
        frames = extract_frames(video, indices)

        for idx, corners in sample:
            if idx not in frames:
                continue
            img = frames[idx]
            mask = make_mask(corners, img.shape[:2])
            name = f"{bg}_{video.stem}_{idx:05d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
            cv2.imwrite(str(mask_dir / f"{name}.png"), mask)

            # YOLO polygon label
            h, w = img.shape[:2]
            norm = [(x / w, y / h) for x, y in corners]
            with open(label_dir / f"{name}.txt", "w") as f:
                f.write("0 " + " ".join(f"{v:.6f}" for pt in norm for v in pt))

            saved_per_bg.setdefault(bg, []).append(name)
            saved += 1
            if saved >= args.total:
                break
        print(f"[ok] {video.name}: +{len(frames)} frames | total {saved}/{args.total}")

    # Split theo background (no leakage)
    all_bgs = sorted(saved_per_bg.keys())
    print(f"\n[split] {len(all_bgs)} backgrounds: {all_bgs}")
    if len(all_bgs) >= 3:
        random.shuffle(all_bgs)
        val_bgs = {all_bgs[0]}
        test_bgs = {all_bgs[1]}
        train_bgs = set(all_bgs[2:])
    else:
        # Fallback: random split
        names_all = [n for lst in saved_per_bg.values() for n in lst]
        random.shuffle(names_all)
        n_train = int(len(names_all) * 0.85)
        n_val = int(len(names_all) * 0.07)
        (root / "train.txt").write_text("\n".join(names_all[:n_train]))
        (root / "val.txt").write_text("\n".join(names_all[n_train:n_train + n_val]))
        (root / "test.txt").write_text("\n".join(names_all[n_train + n_val:]))
        print(f"[done] random split: train={n_train} val={n_val} test={len(names_all) - n_train - n_val}")
        return

    train_names = [n for bg in train_bgs for n in saved_per_bg[bg]]
    val_names = [n for bg in val_bgs for n in saved_per_bg[bg]]
    test_names = [n for bg in test_bgs for n in saved_per_bg[bg]]
    (root / "train.txt").write_text("\n".join(train_names))
    (root / "val.txt").write_text("\n".join(val_names))
    (root / "test.txt").write_text("\n".join(test_names))
    print(f"[done] train={len(train_names)} val={len(val_names)} test={len(test_names)}")


if __name__ == "__main__":
    main()
