"""Pipeline đầy đủ Step 1 → 3 dùng U²-Netp lite thay rembg."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml2.pipeline_integration.u2net_wrapper import U2NetDetector


def warp_to_rect(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Step 2 - Perspective warp 4 góc → rectangle."""
    # Sắp xếp corners theo thứ tự TL TR BR BL
    pts = corners.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered = np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(diff)],  # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(diff)],  # BL
    ], dtype=np.float32)

    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bot = np.linalg.norm(ordered[2] - ordered[3])
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    h_right = np.linalg.norm(ordered[2] - ordered[1])
    W = int(max(w_top, w_bot))
    H = int(max(h_left, h_right))

    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(img, M, (W, H))


def enhance(img: np.ndarray) -> np.ndarray:
    """Step 3 - CLAHE + Binarize."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    binar = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)
    return cv2.cvtColor(binar, cv2.COLOR_GRAY2BGR)


def run_pipeline(img: np.ndarray, detector: U2NetDetector) -> dict:
    """Trả {mask, corners, warped, enhanced}."""
    mask = detector.detect(img)
    corners = detector.get_corners(img)
    out = {"input": img, "mask": mask, "corners": corners}
    if corners is None:
        out["warped"] = img
    else:
        out["warped"] = warp_to_rect(img, corners)
    out["enhanced"] = enhance(out["warped"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml2/checkpoints/u2netp_doc.pth")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="ml2/results/pipeline_u2net")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    detector = U2NetDetector(args.ckpt, device=args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)
    paths = sorted(inp.glob("*.*")) if inp.is_dir() else [inp]
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = run_pipeline(img, detector)
        cv2.imwrite(str(out_dir / f"{p.stem}_mask.png"), res["mask"])
        cv2.imwrite(str(out_dir / f"{p.stem}_warped.jpg"), res["warped"])
        cv2.imwrite(str(out_dir / f"{p.stem}_enhanced.jpg"), res["enhanced"])
        print(f"[ok] {p.name}")


if __name__ == "__main__":
    main()
