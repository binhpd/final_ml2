"""KPI 2: So sánh accuracy U²-Net vs YOLO vs rembg baseline trên test set."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml2.u2net.eval import metrics_pair


def eval_u2net(ckpt: str, samples: list[tuple[Path, Path]], device: str, size: int = 320) -> dict:
    from ml2.u2net.infer import infer_single, load_model
    dev = torch.device(device if (device != "mps" or torch.backends.mps.is_available()) else "cpu")
    model = load_model(ckpt, dev)
    metrics = []
    for img_p, mask_p in samples:
        img = cv2.imread(str(img_p))
        gt = (cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        pred = infer_single(model, img, dev, size=size)
        metrics.append(metrics_pair(pred, gt))
    return _avg(metrics)


def eval_yolo(weights: str, samples: list[tuple[Path, Path]], device: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(weights)
    metrics = []
    for img_p, mask_p in samples:
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        gt = (cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        r = model.predict(img, device=device, verbose=False)
        if not r or r[0].masks is None or len(r[0].masks.data) == 0:
            metrics.append({"iou": 0, "dice": 0, "mae": 1, "bf": 0})
            continue
        masks = r[0].masks.data.cpu().numpy()
        areas = masks.sum(axis=(1, 2))
        best = masks[areas.argmax()]
        pred = cv2.resize(best, (w, h), interpolation=cv2.INTER_LINEAR)
        metrics.append(metrics_pair(pred, gt))
    return _avg(metrics)


def eval_rembg(samples: list[tuple[Path, Path]]) -> dict:
    try:
        from rembg import new_session, remove
        session = new_session()
    except ImportError:
        print("[!] rembg chưa cài - skip baseline")
        return {"iou": 0, "dice": 0, "mae": 1, "bf": 0, "n": 0}
    metrics = []
    for img_p, mask_p in samples:
        img = cv2.imread(str(img_p))
        gt = (cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        out = remove(img, session=session)
        if out.ndim == 3 and out.shape[2] == 4:
            pred = out[..., 3].astype(np.float32) / 255.0
        else:
            pred = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        metrics.append(metrics_pair(pred, gt))
    return _avg(metrics)


def _avg(metrics: list[dict]) -> dict:
    if not metrics:
        return {"iou": 0, "dice": 0, "mae": 1, "bf": 0, "n": 0}
    return {
        "iou": float(np.mean([m["iou"] for m in metrics])),
        "dice": float(np.mean([m["dice"] for m in metrics])),
        "mae": float(np.mean([m["mae"] for m in metrics])),
        "bf": float(np.mean([m["bf"] for m in metrics])),
        "n": len(metrics),
    }


def collect_samples(roots: list[str], split: str = "test") -> list[tuple[Path, Path]]:
    samples = []
    for root in roots:
        root = Path(root)
        f = root / f"{split}.txt"
        if not f.exists():
            continue
        for name in f.read_text().splitlines():
            name = name.strip()
            mask = root / "masks" / f"{name}.png"
            img = None
            for ext in [".jpg", ".png", ".jpeg"]:
                p = root / "images" / f"{name}{ext}"
                if p.exists():
                    img = p
                    break
            if img and mask.exists():
                samples.append((img, mask))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u2net_ckpt", default="ml2/checkpoints/u2netp_doc.pth")
    ap.add_argument("--yolo_weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--roots", nargs="+", default=["ml2/data/smartdoc", "ml2/data/doc3d"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="ml2/results/kpi_accuracy.csv")
    args = ap.parse_args()

    samples = collect_samples(args.roots, args.split)
    print(f"[samples] {len(samples)}")
    rows = []

    if Path(args.u2net_ckpt).exists():
        m = eval_u2net(args.u2net_ckpt, samples, args.device)
        rows.append({"model": "u2net", **m})
        print(f"U2-Net:  IoU={m['iou']:.4f} Dice={m['dice']:.4f} BF={m['bf']:.4f}")

    if Path(args.yolo_weights).exists():
        m = eval_yolo(args.yolo_weights, samples, args.device)
        rows.append({"model": "yolo", **m})
        print(f"YOLO:    IoU={m['iou']:.4f} Dice={m['dice']:.4f} BF={m['bf']:.4f}")

    m = eval_rembg(samples[:50])
    rows.append({"model": "rembg", **m})
    print(f"rembg:   IoU={m['iou']:.4f} Dice={m['dice']:.4f} BF={m['bf']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
