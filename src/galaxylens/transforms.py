"""Consistent model transforms and controlled quality perturbations."""
from __future__ import annotations

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class QualityAugment:
    """Randomized training-only quality variation (not applied to validation/test)."""
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        if random.random() < 0.35:
            factor = random.choice((2, 3, 4))
            size = (max(16, image.width // factor), max(16, image.height // factor))
            image = image.resize(size, Image.Resampling.BILINEAR).resize(
                image.size, Image.Resampling.BICUBIC
            )
        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.30:
            arr = np.asarray(image).astype(np.float32)
            arr += np.random.normal(0, random.uniform(1.0, 7.0), arr.shape)
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return image


def build_transform(image_size: int = 224, training: bool = False,
                    quality_augmentation: bool = False):
    ops = []
    if training:
        # Galaxy orientation is not a class label, so flips are a conservative baseline.
        ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        if quality_augmentation:
            ops.append(QualityAugment())
    ops.extend([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def apply_corruption(image: Image.Image, kind: str, severity: int, seed: int = 0) -> Image.Image:
    """Apply deterministic, documented test-time stress conditions.

    Severity levels are 1-3. These are synthetic stress tests, not a substitute
    for validation on an independent astronomical survey.
    """
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    image = image.convert("RGB")
    rng = np.random.default_rng(seed)
    if kind == "clean":
        return image.copy()
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=(0.5, 1.0, 1.8)[severity - 1]))
    if kind == "resolution":
        factor = (2, 4, 8)[severity - 1]
        small = (max(8, image.width // factor), max(8, image.height // factor))
        return image.resize(small, Image.Resampling.BILINEAR).resize(
            image.size, Image.Resampling.BICUBIC
        )
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance((0.8, 0.6, 0.4)[severity - 1])
    if kind == "noise":
        arr = np.asarray(image).astype(np.float32)
        sigma = (3.0, 8.0, 15.0)[severity - 1]
        arr += rng.normal(0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    raise ValueError(f"Unknown corruption type: {kind}")
