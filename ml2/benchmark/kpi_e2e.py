"""KPI 4: End-to-end - PSNR + SSIM + OCR-CER + total time."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse < 1e-6:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as _ssim
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
        return float(_ssim(ga, gb, data_range=255))
    except Exception:
        return 0.0


def ocr_cer(pred_text: str, gt_text: str) -> float:
    if not gt_text:
        return 0.0
    # Levenshtein distance
    m, n = len(pred_text), len(gt_text)
    if m == 0:
        return 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            if pred_text[i - 1] == gt_text[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n] / n


def run_ocr(img: np.ndarray) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img, lang="eng+vie")
    except Exception:
        return ""


def benchmark_pipeline(name: str, pipeline_fn, input_dir: Path, gt_dir: Path | None, gt_text_dir: Path | None) -> dict:
    paths = sorted(input_dir.glob("*.*"))
    psnrs, ssims, cers, times = [], [], [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        t = time.perf_counter()
        result = pipeline_fn(img)
        dt = (time.perf_counter() - t) * 1000
        times.append(dt)

        enhanced = result.get("enhanced", img)
        if gt_dir:
            gt = cv2.imread(str(gt_dir / f"{p.stem}.jpg"))
            if gt is None:
                gt = cv2.imread(str(gt_dir / f"{p.stem}.png"))
            if gt is not None:
                gt_resized = cv2.resize(gt, (enhanced.shape[1], enhanced.shape[0]))
                psnrs.append(psnr(enhanced, gt_resized))
                ssims.append(ssim_simple(enhanced, gt_resized))
        if gt_text_dir:
            gt_text_file = gt_text_dir / f"{p.stem}.txt"
            if gt_text_file.exists():
                gt_text = gt_text_file.read_text()
                pred_text = run_ocr(enhanced)
                cers.append(ocr_cer(pred_text, gt_text))

    return {
        "pipeline": name,
        "n": len(times),
        "median_ms": float(np.median(times)) if times else 0.0,
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim": float(np.mean(ssims)) if ssims else 0.0,
        "cer": float(np.mean(cers)) if cers else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u2net_ckpt", default="ml2/checkpoints/u2netp_doc.pth")
    ap.add_argument("--yolo_weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--gt_dir", default=None, help="GT ảnh scan (cho PSNR/SSIM)")
    ap.add_argument("--gt_text_dir", default=None, help="GT text (cho OCR-CER)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="ml2/results/kpi_e2e.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    gt_text_dir = Path(args.gt_text_dir) if args.gt_text_dir else None

    rows = []

    if Path(args.u2net_ckpt).exists():
        from ml2.pipeline_integration.pipeline_u2net import run_pipeline as u_run
        from ml2.pipeline_integration.u2net_wrapper import U2NetDetector
        detector = U2NetDetector(args.u2net_ckpt, device=args.device)
        res = benchmark_pipeline("u2net", lambda i: u_run(i, detector), input_dir, gt_dir, gt_text_dir)
        rows.append(res)
        print(f"U2-Net: {res}")

    if Path(args.yolo_weights).exists():
        from ml2.pipeline_integration.pipeline_yolo import run_pipeline as y_run
        from ml2.pipeline_integration.yolo_wrapper import YOLODocDetector
        detector = YOLODocDetector(args.yolo_weights, device=args.device)
        res = benchmark_pipeline("yolo", lambda i: y_run(i, detector), input_dir, gt_dir, gt_text_dir)
        rows.append(res)
        print(f"YOLO: {res}")

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
