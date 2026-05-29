"""Combo loss BCE + IoU + SSIM cho U²-Net deep supervision.

Tham khảo: U²-Net paper Eq. (8) — tổng loss qua 7 outputs (1 fused + 6 sides).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as ssim_fn
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False


def iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft IoU loss = 1 - IoU."""
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return (1 - (inter + eps) / (union + eps)).mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not _HAS_MSSSIM:
        # Fallback đơn giản: 1 - mean cosine similarity
        return F.mse_loss(torch.sigmoid(pred), target)
    p = torch.sigmoid(pred)
    return 1 - ssim_fn(p, target, data_range=1.0, size_average=True)


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 trên Sobel edges — boost boundary accuracy."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
    sobel_y = sobel_x.T
    kx = sobel_x.view(1, 1, 3, 3).expand(1, pred.size(1), 3, 3)
    ky = sobel_y.view(1, 1, 3, 3).expand(1, pred.size(1), 3, 3)
    p = torch.sigmoid(pred)
    px = F.conv2d(p, kx, padding=1)
    py = F.conv2d(p, ky, padding=1)
    tx = F.conv2d(target, kx, padding=1)
    ty = F.conv2d(target, ky, padding=1)
    return (px - tx).abs().mean() + (py - ty).abs().mean()


class ComboLoss(nn.Module):
    """BCE + IoU + SSIM với deep supervision qua 7 outputs.

    L_total = Σ_{i=0..6} (w_bce·BCE_i + w_iou·IoU_i + w_ssim·SSIM_i)
    """

    def __init__(
        self,
        w_bce: float = 1.0,
        w_iou: float = 1.0,
        w_ssim: float = 1.0,
        w_edge: float = 0.0,
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_iou = w_iou
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.bce = nn.BCEWithLogitsLoss()

    def _single(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.w_bce * self.bce(pred, target)
        loss = loss + self.w_iou * iou_loss(pred, target)
        loss = loss + self.w_ssim * ssim_loss(pred, target)
        if self.w_edge > 0:
            loss = loss + self.w_edge * edge_loss(pred, target)
        return loss

    def forward(self, outputs: tuple[torch.Tensor, ...], target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """outputs = (fused, s1..s6). target shape (B, 1, H, W) in {0, 1}."""
        total = 0.0
        per_side = {}
        for i, pred in enumerate(outputs):
            l = self._single(pred, target)
            total = total + l
            per_side[f"side{i}" if i > 0 else "fused"] = l.item()
        return total, per_side


if __name__ == "__main__":
    fn = ComboLoss()
    outs = tuple(torch.randn(2, 1, 64, 64) for _ in range(7))
    tgt = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss, per = fn(outs, tgt)
    print(f"Loss: {loss.item():.4f}")
    print(f"Per-side: {per}")
