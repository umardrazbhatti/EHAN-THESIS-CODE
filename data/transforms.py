"""
data/transforms.py
==================
Provides get_transforms(mode, frame_size) used by DeepfakeDataset.__getitem__.

Train : random horizontal flip + colour jitter + Gaussian blur
        + JPEG compression augmentation (Phase 20 Fix 5)
        + Gaussian noise (Phase 20 Fix 5)
        + ImageNet normalisation
Val / Test : centre-crop resize + ImageNet normalisation only

Phase 20 augmentations simulate social-media re-encoding pipelines and improve
CelebDF cross-domain generalisation by reducing FF++ c23-specific compression
signature overfitting.
"""

import io
import random
import torch
from PIL import Image
from torchvision import transforms


# ImageNet statistics — used for EfficientNet-B4 pre-trained weights
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ── Phase 20 Fix 5: Compression augmentations ─────────────────────────────────

class RandomJPEGCompression:
    """
    Apply random JPEG compression to a PIL Image (applied before ToTensor).
    Simulates social-media re-encoding to prevent FF++ c23 signature overfitting.
    Applied with probability 0.5 at a randomly chosen quality in [30, 95].
    """

    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buf = io.BytesIO()
            # Ensure RGB before JPEG encoding (JPEG doesn't support RGBA/P mode)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).copy()
        return img


class RandomGaussianNoise:
    """
    Add random Gaussian noise to a float32 tensor (applied after ToTensor, before Normalize).
    Simulates sensor noise and additional compression artefacts.
    Applied with probability p; noise std sampled from [std_min, std_max].
    """

    def __init__(self, std_range=(0.01, 0.05), p=0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(img) * std
            return (img + noise).clamp(0.0, 1.0)
        return img


def get_heavy_transforms(frame_size: int = 224):
    """
    Heavy augmentation for minority-class samples when class imbalance ratio > 3:1.
    Applied only during training, only to the minority class.
    Includes: horizontal flip, stronger colour jitter, small rotation,
              Gaussian blur, and perspective distortion.
    """
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        # Phase 20 Fix 5: JPEG compression aug (higher pressure for minority class)
        RandomJPEGCompression(quality_range=(30, 95), p=0.6),
        transforms.ToTensor(),
        # Phase 20 Fix 5: Gaussian noise aug
        RandomGaussianNoise(std_range=(0.01, 0.05), p=0.3),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_transforms(mode: str, frame_size: int = 224):
    """
    Return a torchvision transform pipeline for the given split.

    Parameters
    ----------
    mode : str
        One of 'train', 'val', or 'test'.
    frame_size : int
        Target spatial resolution (height == width). Default 224.

    Returns
    -------
    torchvision.transforms.Compose
        A callable that accepts a PIL Image and returns a normalised float32 tensor
        of shape (3, frame_size, frame_size).
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            # Phase 20 Fix 5: JPEG compression aug (PIL image, before ToTensor)
            RandomJPEGCompression(quality_range=(30, 95), p=0.5),
            transforms.ToTensor(),
            # Phase 20 Fix 5: Gaussian noise aug (float tensor [0,1], before Normalize)
            RandomGaussianNoise(std_range=(0.01, 0.05), p=0.3),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
    else:
        # val and test: deterministic resize + normalise only
        return transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
