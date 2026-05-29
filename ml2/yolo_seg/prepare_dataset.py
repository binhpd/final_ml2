"""Convert SmartDoc/Doc3D format chuẩn → YOLO segmentation format.

YOLO seg structure:
  ml2/data/yolo_doc/
    images/train/<id>.jpg
    images/val/<id>.jpg
    images/test/<id>.jpg
    labels/train/<id>.txt  (cls x1 y1 x2 y2 ... normalized polygon)
    labels/val/<id>.txt
    labels/test/<id>.txt
    doc.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def mask_to_polygon(mask: np.ndarray, max_points: int = 64) -> list[tuple[float, float]] | None:
    """Convert binary mask → normalized polygon (cls 0, doc)."""
    mask_u8 = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    # Simplify
    eps = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).squeeze(1)
    if len(approx) > max_points:
        idx = np.linspace(0, len(approx) - 1, max_points).astype(int)
        approx = approx[idx]
    h, w = mask.shape
    return [(float(p[0]) / w, float(p[1]) / h) for p in approx]


def write_yolo_label(path: Path, poly: list[tuple[float, float]]) -> None:
    parts = ["0"]
    for x, y in poly:
        parts.extend([f"{x:.6f}", f"{y:.6f}"])
    path.write_text(" ".join(parts))


def process_source(src_root: Path, dst: Path, splits: list[str]):
    for split in splits:
        list_file = src_root / f"{split}.txt"
        if not list_file.exists():
            continue
        names = [n.strip() for n in list_file.read_text().splitlines() if n.strip()]
        img_out = dst / "images" / split
        lbl_out = dst / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        ok = 0
        for name in names:
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = src_root / "images" / f"{name}{ext}"
                if p.exists():
                    img_path = p
                    break
            mask_path = src_root / "masks" / f"{name}.png"
            if img_path is None or not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            poly = mask_to_polygon(mask)
            if not poly or len(poly) < 3:
                continue
            shutil.copy2(img_path, img_out / f"{src_root.name}_{name}{img_path.suffix}")
            write_yolo_label(lbl_out / f"{src_root.name}_{name}.txt", poly)
            ok += 1
        print(f"[{src_root.name}/{split}] converted {ok}/{len(names)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["ml2/data/smartdoc", "ml2/data/doc3d"])
    ap.add_argument("--out", default="ml2/data/yolo_doc")
    args = ap.parse_args()

    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    for src in args.sources:
        src_p = Path(src)
        if not src_p.exists():
            print(f"[skip] {src_p} không tồn tại")
            continue
        process_source(src_p, dst, ["train", "val", "test"])

    yaml_path = dst / "doc.yaml"
    yaml_path.write_text(
        f"path: {dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names:\n  0: document\n"
    )
    print(f"\n[done] YOLO dataset tại {dst}")
    print(f"  YAML: {yaml_path}")


if __name__ == "__main__":
    main()
