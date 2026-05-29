"""Sinh dummy data để smoke-test code chạy không lỗi.

Tạo N ảnh giả mô phỏng tài liệu (giấy trắng + text-like patches) trên nền ngẫu nhiên,
kèm mask nhị phân chứa polygon quadrilateral (4 góc).
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def random_quad(w: int, h: int, margin: int = 40) -> np.ndarray:
    """Sinh quadrilateral ngẫu nhiên trong ảnh."""
    cx, cy = w // 2, h // 2
    sw, sh = random.randint(int(w * 0.35), int(w * 0.45)), random.randint(int(h * 0.4), int(h * 0.55))
    corners = np.float32([
        [cx - sw, cy - sh],
        [cx + sw, cy - sh],
        [cx + sw, cy + sh],
        [cx - sw, cy + sh],
    ])
    jitter = np.random.uniform(-margin, margin, corners.shape).astype(np.float32)
    corners = np.clip(corners + jitter, [10, 10], [w - 10, h - 10])
    return corners.astype(np.int32)


def render_doc(quad: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Render giấy + text-like patches."""
    bg = np.random.randint(40, 180, (h, w, 3), dtype=np.uint8)
    cv2.GaussianBlur(bg, (51, 51), 0, dst=bg)

    canvas = bg.copy()
    paper = np.full((h, w, 3), 245, dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)

    for c in range(3):
        canvas[..., c] = np.where(mask > 0, paper[..., c], canvas[..., c])

    # Add fake text lines inside polygon
    x_min, y_min = quad.min(axis=0)
    x_max, y_max = quad.max(axis=0)
    for _ in range(random.randint(8, 18)):
        y = random.randint(y_min + 20, max(y_min + 25, y_max - 20))
        x1 = random.randint(x_min + 15, max(x_min + 20, x_max - 100))
        x2 = random.randint(x1 + 50, max(x1 + 60, x_max - 15))
        cv2.line(canvas, (x1, y), (x2, y), (40, 40, 40), 2)

    # Add some noise + shadow
    if random.random() < 0.4:
        shadow = np.zeros_like(canvas)
        cv2.fillPoly(shadow, [quad + np.array([20, 20])], (0, 0, 0))
        canvas = cv2.addWeighted(canvas, 1.0, shadow, 0.15, 0)

    noise = np.random.randint(-15, 15, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="ml2/data/dummy")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    meta = []
    for i in range(args.n):
        quad = random_quad(args.size, args.size)
        img, mask = render_doc(quad, args.size, args.size)

        name = f"dummy_{i:04d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img)
        cv2.imwrite(str(out / "masks" / f"{name}.png"), mask)

        # YOLO polygon label (normalized)
        poly_norm = quad.astype(np.float32) / np.array([args.size, args.size])
        with open(out / "labels" / f"{name}.txt", "w") as f:
            f.write("0 " + " ".join(f"{v:.6f}" for v in poly_norm.flatten()))

        meta.append({"name": name, "corners": quad.tolist()})

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Train/val split file
    names = [m["name"] for m in meta]
    random.shuffle(names)
    n_train = int(len(names) * 0.8)
    with open(out / "train.txt", "w") as f:
        f.write("\n".join(names[:n_train]))
    with open(out / "val.txt", "w") as f:
        f.write("\n".join(names[n_train:]))

    print(f"Sinh {args.n} ảnh dummy + masks + labels vào {out}/")


if __name__ == "__main__":
    main()
