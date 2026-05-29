"""Pipeline đầy đủ Step 1 → 3 dùng YOLOv11n-seg + visualization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml2.pipeline_integration.pipeline_u2net import enhance, warp_to_rect
from ml2.pipeline_integration.yolo_wrapper import YOLODocDetector


def run_pipeline(img: np.ndarray, detector: YOLODocDetector) -> dict:
    det = detector.detect(img)
    out = {"input": img, "det": det}
    if det["corners"] is None:
        out["warped"] = img
    else:
        out["warped"] = warp_to_rect(img, det["corners"])
    out["enhanced"] = enhance(out["warped"])
    out["viz"] = detector.visualize(img, det)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="ml2/results/pipeline_yolo")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    detector = YOLODocDetector(args.weights, device=args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)
    paths = sorted(inp.glob("*.*")) if inp.is_dir() else [inp]
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = run_pipeline(img, detector)
        if res["det"]["mask"] is not None:
            cv2.imwrite(str(out_dir / f"{p.stem}_mask.png"), res["det"]["mask"])
        cv2.imwrite(str(out_dir / f"{p.stem}_viz.jpg"), res["viz"])
        cv2.imwrite(str(out_dir / f"{p.stem}_warped.jpg"), res["warped"])
        cv2.imwrite(str(out_dir / f"{p.stem}_enhanced.jpg"), res["enhanced"])
        print(f"[ok] {p.name}")


if __name__ == "__main__":
    main()
