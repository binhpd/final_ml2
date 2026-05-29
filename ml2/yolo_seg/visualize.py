"""YOLODocVisualizer - vẽ bbox + mask + 4 góc + info card lên ảnh."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    "doc": (0, 200, 80),
    "corner": (0, 80, 220),
    "bbox": (220, 220, 0),
    "text_bg": (0, 0, 0),
    "text_fg": (255, 255, 255),
}


class YOLODocVisualizer:
    """Module visualize - sản phẩm chính khác biệt YOLO vs U²-Net."""

    def __init__(self, weights: str, device: str = "mps", conf: float = 0.25):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def predict(self, img: np.ndarray):
        return self.model.predict(img, device=self.device, conf=self.conf, verbose=False)

    @staticmethod
    def mask_to_corners(mask: np.ndarray) -> np.ndarray:
        """Approx 4 corners từ binary mask qua approxPolyDP."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.empty((0, 2), dtype=np.int32)
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        for eps in [0.02, 0.015, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True).squeeze(1)
            if len(approx) == 4:
                return approx.astype(np.int32)
        # Fallback: min area rect
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        return box.astype(np.int32)

    def visualize(self, img: np.ndarray, show_corners: bool = True, show_info: bool = True) -> np.ndarray:
        results = self.predict(img)
        canvas = img.copy()
        h, w = img.shape[:2]
        info_lines = []

        if not results or results[0].masks is None:
            cv2.putText(canvas, "No detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["text_fg"], 2)
            return canvas

        r = results[0]
        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else None

        for i, m in enumerate(masks):
            m_resized = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            # Mask overlay
            color_layer = np.zeros_like(canvas)
            color_layer[m_resized > 0] = COLORS["doc"]
            canvas = cv2.addWeighted(canvas, 1.0, color_layer, 0.35, 0)
            # Contour
            cnts, _ = cv2.findContours(m_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, COLORS["doc"], 2)

            # BBox
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS["bbox"], 2)
                conf = float(confs[i]) if confs is not None else 0.0
                label = f"doc {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLORS["bbox"], -1)
                cv2.putText(canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Corners
            if show_corners:
                corners = self.mask_to_corners(m_resized)
                for j, (px, py) in enumerate(corners):
                    cv2.circle(canvas, (int(px), int(py)), 8, COLORS["corner"], -1)
                    cv2.putText(canvas, f"C{j}", (int(px) + 10, int(py) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["corner"], 2)

            info_lines.append(f"det{i}: area={int(m_resized.sum())}px conf={confs[i]:.2f}" if confs is not None else f"det{i}")

        # Info card
        if show_info and info_lines:
            card_h = 24 + 22 * len(info_lines)
            card = canvas[10:10 + card_h, 10:330].copy()
            cv2.rectangle(canvas, (10, 10), (330, 10 + card_h), COLORS["text_bg"], -1)
            canvas[10:10 + card_h, 10:330] = cv2.addWeighted(card, 0.4, canvas[10:10 + card_h, 10:330], 0.6, 0)
            cv2.putText(canvas, "YOLODocVisualizer", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text_fg"], 2)
            for k, line in enumerate(info_lines):
                cv2.putText(canvas, line, (16, 54 + k * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)

        return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ml2/checkpoints/yolo11n_seg_doc.pt")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="ml2/results/yolo_viz")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    viz = YOLODocVisualizer(args.weights, device=args.device)
    inp = Path(args.input)
    paths = sorted(inp.glob("*.*")) if inp.is_dir() else [inp]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = viz.visualize(img)
        cv2.imwrite(str(out_dir / f"viz_{p.stem}.jpg"), result)
        print(f"[ok] {p.name}")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
