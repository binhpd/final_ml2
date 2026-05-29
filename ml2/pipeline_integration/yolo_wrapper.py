"""Wrapper YOLOv11n-seg cho Step 1 + 4-corner extraction + viz API."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


class YOLODocDetector:
    """Wrapper cho YOLOv11n-seg dùng làm Step 1 detection.

    Khác với U²-Net: cho ra multi-instance, bbox + conf + class.
    """

    def __init__(self, weights: str = "ml2/checkpoints/yolo11n_seg_doc.pt", device: str = "mps", conf: float = 0.25):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def detect(self, img: np.ndarray) -> dict:
        """Trả dict: mask (H,W), bbox (x1,y1,x2,y2), conf, corners (4,2) hoặc None."""
        h, w = img.shape[:2]
        results = self.model.predict(img, device=self.device, conf=self.conf, verbose=False)
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return {"mask": None, "bbox": None, "conf": 0.0, "corners": None}

        r = results[0]
        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        # Chọn instance có area lớn nhất
        areas = masks.sum(axis=(1, 2))
        i = int(areas.argmax())
        mask = cv2.resize(masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) * 255
        bbox = boxes[i].astype(int)
        conf = float(confs[i])
        corners = self._mask_to_corners(mask)
        return {"mask": mask, "bbox": bbox, "conf": conf, "corners": corners}

    @staticmethod
    def _mask_to_corners(mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        for eps in [0.02, 0.015, 0.03, 0.04]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True).squeeze(1)
            if len(approx) == 4:
                return approx.astype(np.float32)
        rect = cv2.minAreaRect(cnt)
        return cv2.boxPoints(rect).astype(np.float32)

    def visualize(self, img: np.ndarray, det: dict | None = None) -> np.ndarray:
        if det is None:
            det = self.detect(img)
        canvas = img.copy()
        if det["mask"] is None:
            cv2.putText(canvas, "No detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return canvas
        # Mask overlay
        m = det["mask"]
        color = np.zeros_like(canvas)
        color[m > 0] = (0, 200, 80)
        canvas = cv2.addWeighted(canvas, 1.0, color, 0.35, 0)
        # BBox
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 0), 2)
        cv2.putText(canvas, f"doc {det['conf']:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Corners
        if det["corners"] is not None:
            for j, (cx, cy) in enumerate(det["corners"]):
                cv2.circle(canvas, (int(cx), int(cy)), 8, (0, 80, 220), -1)
                cv2.putText(canvas, f"C{j}", (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 220), 2)
        return canvas
