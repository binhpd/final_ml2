"""Augmentation pipelines cho document segmentation.

Basic: dùng cho val + smoke test.
Strong: simulate ảnh chụp điện thoại (blur, shadow, perspective, noise).
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def basic_train(size: int = 320) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=int(size * 1.2)),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0),
        A.RandomCrop(height=size, width=size, p=0.5),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def strong_train(size: int = 320) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=int(size * 1.3)),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0),
        A.RandomCrop(height=size, width=size, p=0.4),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=7),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(std_range=(0.04, 0.2)),
        ], p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.HueSaturationValue(p=0.3),
        A.RandomShadow(p=0.3),
        A.RandomSunFlare(p=0.15, src_radius=80),
        A.Perspective(scale=(0.04, 0.10), p=0.4),
        A.Rotate(limit=12, p=0.5, border_mode=0),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(int(size * 0.05), int(size * 0.1)), hole_width_range=(int(size * 0.05), int(size * 0.1)), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def val_transform(size: int = 320) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0),
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_transform(mode: str, size: int = 320, strong: bool = True) -> A.Compose:
    if mode == "train":
        return strong_train(size) if strong else basic_train(size)
    return val_transform(size)
