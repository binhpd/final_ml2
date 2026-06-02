"""Drop-in wrapper U²-Netp lite thay rembg.remove() trong pipeline cũ.

API tương thích rembg: input ảnh BGR/RGB → trả ảnh có alpha mask hoặc binary mask.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ml2.u2net.infer import infer_single, load_model


class U2NetDetector:
    """Wrapper drop-in cho rembg trong Step 1 pipeline."""

    def __init__(self, ckpt: str = "ml2/checkpoints/u2netp_doc.pth", device: str = "mps", size: int = 320, is_lite: bool = True):
        self.device = torch.device(device if (device != "mps" or torch.backends.mps.is_available()) else "cpu")
        self.size = size
        self.model = load_model(ckpt, self.device, is_lite=is_lite)

    def detect(self, img: np.ndarray) -> np.ndarray:
        """Trả về binary mask (H, W) uint8 0/255."""
        mask = infer_single(self.model, img, self.device, size=self.size)
        return (mask > 0.5).astype(np.uint8) * 255

    def remove_background(self, img: np.ndarray) -> np.ndarray:
        """API tương thích rembg.remove() - trả RGBA với alpha = mask."""
        mask = self.detect(img)
        if img.ndim == 3 and img.shape[2] == 3:
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        else:
            rgba = img.copy()
        rgba[..., 3] = mask
        return rgba

    def get_corners(self, img: np.ndarray) -> np.ndarray | None:
        """Trả 4 corner (N, 2) hoặc None nếu fail."""
        mask = self.detect(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
