"""KPI 1: Latency + FPS trên CPU + MPS."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def bench_u2net(ckpt: str, input_dir: Path, device: str, size: int = 320, n: int = 100, warmup: int = 10) -> dict:
    from ml2.u2net.infer import infer_single, load_model
    dev = torch.device(device if (device != "mps" or torch.backends.mps.is_available()) else "cpu")
    model = load_model(ckpt, dev)
    paths = sorted(input_dir.glob("*.*"))[:n + warmup]
    times = []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        if dev.type == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        infer_single(model, img, dev, size=size)
        if dev.type == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t) * 1000
        if i >= warmup:
            times.append(dt)
    return _stats(times)


def bench_yolo(weights: str, input_dir: Path, device: str, imgsz: int = 640, n: int = 100, warmup: int = 10) -> dict:
    from ultralytics import YOLO
    model = YOLO(weights)
    paths = sorted(input_dir.glob("*.*"))[:n + warmup]
    times = []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        t = time.perf_counter()
        model.predict(img, device=device, imgsz=imgsz, verbose=False)
        dt = (time.perf_counter() - t) * 1000
        if i >= warmup:
            times.append(dt)
    return _stats(times)


def _stats(times: list[float]) -> dict:
    if not times:
        return {"median_ms": 0.0, "p95_ms": 0.0, "fps": 0.0, "n": 0}
    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": float(1000 / np.median(arr)),
        "n": len(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u2net_ckpt", default="ml2/checkpoints/u2netp_doc.pth")
    ap.add_argument("--yolo_weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="ml2/results/kpi_speed.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    for model_name, fn, ckpt in [
        ("u2net", bench_u2net, args.u2net_ckpt),
        ("yolo", bench_yolo, args.yolo_weights),
    ]:
        if not Path(ckpt).exists():
            print(f"[skip] {model_name}: {ckpt} không tồn tại")
            continue
        for dev in ["cpu", "mps"]:
            if dev == "mps" and not torch.backends.mps.is_available():
                continue
            print(f"\n[bench] {model_name} on {dev}")
            stats = fn(ckpt, input_dir, dev, n=args.n)
            row = {"model": model_name, "device": dev, **stats}
            print(f"  median={stats['median_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms FPS={stats['fps']:.1f}")
            rows.append(row)

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
