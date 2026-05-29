"""DocSegDataset - loader cho SmartDoc + Doc3D + DocAligner + Dummy.

Format chuẩn:
  data/<dataset_name>/
    images/<id>.{jpg,png}
    masks/<id>.png                (binary mask 0/255)
    train.txt / val.txt / test.txt (list ID, 1 per line)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DocSegDataset(Dataset):
    """Universal document segmentation dataset loader."""

    IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(
        self,
        roots: list[str | Path],
        split: str = "train",
        transform=None,
        return_meta: bool = False,
    ):
        self.transform = transform
        self.return_meta = return_meta
        self.samples: list[tuple[Path, Path, str]] = []

        for root in roots:
            root = Path(root)
            split_file = root / f"{split}.txt"
            if not split_file.exists():
                continue
            names = split_file.read_text().strip().splitlines()
            for name in names:
                name = name.strip()
                if not name:
                    continue
                img = self._resolve_image(root / "images", name)
                mask = root / "masks" / f"{name}.png"
                if img and mask.exists():
                    self.samples.append((img, mask, root.name))

        if not self.samples:
            raise RuntimeError(f"Không tìm thấy sample nào trong {roots} split={split}")

    def _resolve_image(self, img_dir: Path, name: str) -> Path | None:
        for ext in self.IMG_EXTS:
            p = img_dir / f"{name}{ext}"
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, ds_name = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img_t = out["image"]
            mask_t = out["mask"].unsqueeze(0) if out["mask"].ndim == 2 else out["mask"]
            mask_t = mask_t.float()
        else:
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0)

        sample = {"image": img_t, "mask": mask_t, "dataset": ds_name}
        if self.return_meta:
            sample["path"] = str(img_path)
        return sample


def build_dataset(cfg: dict, split: str = "train", transform=None) -> DocSegDataset:
    """Build dataset từ config dict.

    cfg.data.roots = list path (e.g. ["ml2/data/smartdoc", "ml2/data/doc3d"])
    """
    roots = cfg["data"]["roots"]
    return DocSegDataset(roots, split=split, transform=transform)
