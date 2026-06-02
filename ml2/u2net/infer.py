"""Inference U²-Net trên 1 ảnh hoặc folder. Hỗ trợ TTA (horizontal flip)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml2.u2net.model import U2NET, U2NETp


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr: np.ndarray, size: int = 320) -> tuple[torch.Tensor, tuple[int, int]]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, (h, w)


def postprocess(mask_logits: torch.Tensor, original_size: tuple[int, int]) -> np.ndarray:
    m = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
    m = cv2.resize(m, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    return m


@torch.no_grad()
def infer_single(model: torch.nn.Module, img_bgr: np.ndarray, device: torch.device, size: int = 320, tta: bool = False) -> np.ndarray:
    tensor, hw = preprocess(img_bgr, size)
    tensor = tensor.to(device)
    out = model(tensor)[0]
    mask = postprocess(out, hw)
    if tta:
        flipped = cv2.flip(img_bgr, 1)
        t2, _ = preprocess(flipped, size)
        t2 = t2.to(device)
        out2 = model(t2)[0]
        m2 = postprocess(out2, hw)
        m2 = cv2.flip(m2, 1)
        mask = (mask + m2) / 2.0
    return mask


def load_model(ckpt: str, device: torch.device, is_lite: bool = True) -> torch.nn.Module:
    model = U2NETp().to(device) if is_lite else U2NET().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="File hoặc folder")
    ap.add_argument("--out_dir", default="ml2/results/u2net_infer")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    model = load_model(args.ckpt, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)
    paths = sorted(inp.glob("*.*")) if inp.is_dir() else [inp]

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        mask = infer_single(model, img, device, args.size, args.tta)
        mask_u8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{p.stem}_mask.png"), mask_u8)
        print(f"[ok] {p.name}")

    print(f"[done] {len(paths)} ảnh -> {out_dir}")


if __name__ == "__main__":
    main()
