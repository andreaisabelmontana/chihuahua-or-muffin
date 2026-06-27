"""Dataset loading and preprocessing for chihuahua-or-muffin.

Images live as PNGs under  data/dataset/<split>/<class>/*.png  (created by
data/generate.py). We load them with OpenCV, convert BGR->RGB, resize to
64x64, scale to [0,1] and return CHW float32 tensors. The same `preprocess`
function is used by predict.py so training and inference see identical pixels.
"""
from __future__ import annotations

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .model import CLASSES

IMG = 64
_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """np.uint8 HxWx3 BGR image -> float32 CHW tensor in [0,1], 64x64 RGB."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).contiguous()


def load_image_file(path: str) -> torch.Tensor:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return preprocess(bgr)


def _augment(t: torch.Tensor) -> torch.Tensor:
    """Light, label-preserving augmentation (train split only)."""
    if torch.rand(1).item() < 0.5:                      # horizontal flip
        t = torch.flip(t, dims=[2])
    if torch.rand(1).item() < 0.4:                      # brightness jitter
        t = torch.clamp(t * (0.85 + 0.3 * torch.rand(1).item()), 0, 1)
    if torch.rand(1).item() < 0.3:                      # small additive noise
        t = torch.clamp(t + 0.03 * torch.randn_like(t), 0, 1)
    return t


class ImageFolderDataset(Dataset):
    def __init__(self, root_split: str, augment: bool = False):
        self.augment = augment
        self.samples: list[tuple[str, int]] = []
        for cls in CLASSES:
            for p in sorted(glob.glob(os.path.join(root_split, cls, "*.png"))):
                self.samples.append((p, _CLASS_TO_IDX[cls]))
        if not self.samples:
            raise RuntimeError(f"no images found under {root_split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        t = load_image_file(path)
        if self.augment:
            t = _augment(t)
        return t, label


def dataset_root() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "dataset")
