"""Test integration: chạy cả 2 pipeline trên test set, so sánh."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np


def test_u2net(ckpt: str, input_dir: Path, out_dir: Path, device: str, n: int):
    from ml2.pipeline_integration.u2net_wrapper import U2NetDetector
    from ml2.pipeline_integration.pipeline_u2net import run_pipeline

    detector = U2NetDetector(ckpt, device=device)
    paths = sorted(input_dir.glob("*.*"))[:n]
    times = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        t = time.perf_counter()
        run_pipeline(img, detector)
        times.append((time.perf_counter() - t) * 1000)
    return {"n": len(times), "median_ms": float(np.median(times)) if times else 0.0}


def test_yolo(weights: str, input_dir: Path, out_dir: Path, device: str, n: int):
    from ml2.pipeline_integration.yolo_wrapper import YOLODocDetector
    from ml2.pipeline_integration.pipeline_yolo import run_pipeline

    detector = YOLODocDetector(weights, device=device)
    paths = sorted(input_dir.glob("*.*"))[:n]
    times = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        t = time.perf_counter()
        run_pipeline(img, detector)
        times.append((time.perf_counter() - t) * 1000)
    return {"n": len(times), "median_ms": float(np.median(times)) if times else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u2net_ckpt", default="ml2/checkpoints/u2netp_doc.pth")
    ap.add_argument("--yolo_weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", default="ml2/results/integration_test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== U²-Net pipeline ===")
    if Path(args.u2net_ckpt).exists():
        res_u = test_u2net(args.u2net_ckpt, input_dir, out_dir, args.device, args.n)
        print(f"  N={res_u['n']} median={res_u['median_ms']:.1f}ms FPS={1000 / max(1, res_u['median_ms']):.1f}")
    else:
        print(f"  [skip] không tìm thấy {args.u2net_ckpt}")

    print("\n=== YOLO pipeline ===")
    if Path(args.yolo_weights).exists():
        res_y = test_yolo(args.yolo_weights, input_dir, out_dir, args.device, args.n)
        print(f"  N={res_y['n']} median={res_y['median_ms']:.1f}ms FPS={1000 / max(1, res_y['median_ms']):.1f}")
    else:
        print(f"  [skip] không tìm thấy {args.yolo_weights}")


if __name__ == "__main__":
    main()
