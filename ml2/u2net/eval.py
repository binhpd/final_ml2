"""Evaluation U²-Net trên test set - 4 metric: IoU, Dice, MAE, Boundary F1."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml2.u2net.augmentation import get_transform
from ml2.u2net.dataset import DocSegDataset
from ml2.u2net.model import U2NETp


def boundary_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=thickness)
    return (mask_u8 - eroded).astype(bool)


def metrics_pair(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = (pred > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    iou = inter / max(1.0, union)
    dice = 2 * inter / max(1.0, p.sum() + g.sum())
    mae = np.abs(pred - g).mean()

    pb = boundary_mask(p)
    gb = boundary_mask(g)
    tp = (pb & gb).sum()
    fp = (pb & ~gb).sum()
    fn = (~pb & gb).sum()
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    bf = 2 * prec * rec / max(1e-6, prec + rec)
    return {"iou": iou, "dice": dice, "mae": float(mae), "bf": float(bf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--roots", nargs="+", default=["ml2/data/smartdoc", "ml2/data/doc3d"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--per_dataset", action="store_true", help="Báo cáo per-dataset")
    ap.add_argument("--out", default="ml2/results/u2net_eval.csv")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    model = U2NETp().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    tf = get_transform("val", size=args.size)
    ds = DocSegDataset(args.roots, split=args.split, transform=tf, return_meta=True)
    dl = DataLoader(ds, batch_size=1, num_workers=0)

    agg: dict[str, list] = {}
    rows = []
    with torch.no_grad():
        for batch in tqdm(dl):
            img = batch["image"].to(device)
            gt = batch["mask"][0, 0].cpu().numpy()
            out = model(img)[0]
            pred = torch.sigmoid(out)[0, 0].cpu().numpy()
            m = metrics_pair(pred, gt)
            ds_name = batch["dataset"][0]
            rows.append({"dataset": ds_name, "path": batch["path"][0], **m})
            agg.setdefault(ds_name, []).append(m)
            agg.setdefault("ALL", []).append(m)

    print("\n=== Results ===")
    print(f"{'Dataset':<15} {'IoU':>8} {'Dice':>8} {'MAE':>8} {'BF':>8} {'N':>6}")
    for name, lst in agg.items():
        iou = np.mean([m["iou"] for m in lst])
        dice = np.mean([m["dice"] for m in lst])
        mae = np.mean([m["mae"] for m in lst])
        bf = np.mean([m["bf"] for m in lst])
        print(f"{name:<15} {iou:>8.4f} {dice:>8.4f} {mae:>8.4f} {bf:>8.4f} {len(lst):>6}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "path", "iou", "dice", "mae", "bf"])
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
