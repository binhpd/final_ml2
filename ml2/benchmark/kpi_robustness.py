"""KPI 3: Per-dataset robustness - báo cáo riêng cho SmartDoc vs Doc3D."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml2.benchmark.kpi_accuracy import collect_samples, eval_u2net, eval_yolo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u2net_ckpt", default="ml2/checkpoints/u2netp_doc.pth")
    ap.add_argument("--yolo_weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--roots", nargs="+", default=["ml2/data/smartdoc", "ml2/data/doc3d"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="ml2/results/kpi_robustness.csv")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        samples = collect_samples([root], args.split)
        ds_name = Path(root).name
        print(f"\n=== {ds_name} ({len(samples)} samples) ===")
        if Path(args.u2net_ckpt).exists():
            m = eval_u2net(args.u2net_ckpt, samples, args.device)
            rows.append({"dataset": ds_name, "model": "u2net", **m})
            print(f"  U2-Net: IoU={m['iou']:.4f} BF={m['bf']:.4f}")
        if Path(args.yolo_weights).exists():
            m = eval_yolo(args.yolo_weights, samples, args.device)
            rows.append({"dataset": ds_name, "model": "yolo", **m})
            print(f"  YOLO:   IoU={m['iou']:.4f} BF={m['bf']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
