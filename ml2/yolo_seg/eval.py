"""Eval YOLOv11n-seg trên test set - mAP + custom mIoU."""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np


def custom_miou(weights: str, img_dir: Path, lbl_dir: Path, device: str = "mps") -> dict:
    """Tính mIoU thủ công - so polygon predict với polygon GT."""
    from ultralytics import YOLO
    model = YOLO(weights)

    ious = []
    for img_p in sorted(img_dir.glob("*.*")):
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT mask
        parts = lbl_p.read_text().split()
        coords = [float(x) for x in parts[1:]]
        pts = np.array([(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)], dtype=np.int32)
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_mask, [pts], 1)

        # Pred
        results = model.predict(img, device=device, verbose=False)
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            ious.append(0.0)
            continue
        # Lấy mask có area lớn nhất
        masks = results[0].masks.data.cpu().numpy()
        areas = masks.sum(axis=(1, 2))
        best = masks[areas.argmax()]
        pred_mask = cv2.resize(best.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        inter = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        iou = inter / max(1, union)
        ious.append(iou)

    return {"miou": float(np.mean(ious)) if ious else 0.0, "n": len(ious)}


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--data", default="ml2/data/yolo_doc/doc.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="ml2/results/yolo_eval.csv")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    print("[map] Built-in Ultralytics evaluation...")
    metrics = model.val(data=args.data, split=args.split, device=args.device, verbose=True)
    map50_box = float(metrics.box.map50)
    map5095_box = float(metrics.box.map)
    map50_mask = float(metrics.seg.map50)
    map5095_mask = float(metrics.seg.map)

    # Custom mIoU
    data_root = Path(args.data).parent
    img_dir = data_root / "images" / args.split
    lbl_dir = data_root / "labels" / args.split
    print("\n[miou] Custom mIoU on full split...")
    if img_dir.exists():
        miou = custom_miou(args.weights, img_dir, lbl_dir, args.device)
    else:
        miou = {"miou": 0.0, "n": 0}

    # Latency
    print("\n[speed] Latency benchmark...")
    sample_imgs = sorted(img_dir.glob("*.*"))[:50] if img_dir.exists() else []
    times = []
    for p in sample_imgs:
        img = cv2.imread(str(p))
        t = time.perf_counter()
        model.predict(img, device=args.device, verbose=False)
        times.append((time.perf_counter() - t) * 1000)
    median_ms = float(np.median(times)) if times else 0.0
    fps = 1000.0 / median_ms if median_ms > 0 else 0.0

    print(f"\n=== YOLO Eval Results ===")
    print(f"mAP@0.5 box:  {map50_box:.4f}")
    print(f"mAP@0.5:0.95: {map5095_box:.4f}")
    print(f"mAP@0.5 mask: {map50_mask:.4f}")
    print(f"Custom mIoU:  {miou['miou']:.4f} (N={miou['n']})")
    print(f"Median latency: {median_ms:.1f}ms | FPS: {fps:.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["map50_box", map50_box])
        w.writerow(["map5095_box", map5095_box])
        w.writerow(["map50_mask", map50_mask])
        w.writerow(["map5095_mask", map5095_mask])
        w.writerow(["custom_miou", miou["miou"]])
        w.writerow(["median_ms", median_ms])
        w.writerow(["fps", fps])
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
