"""A small convolutional network for 2-class 64x64 image classification.

Three conv blocks (conv -> BN -> ReLU -> maxpool) shrink 64x64 -> 8x8 while
growing channels 3 -> 16 -> 32 -> 64, then global average pooling feeds a
small dense head to 2 logits. Deliberately compact (~50k params) so it trains
end-to-end on CPU in seconds, yet deep enough to learn the shape/texture cues
that separate a "chihuahua" face-blob from a "muffin" dome.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CLASSES = ("chihuahua", "muffin")


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),   # 64 -> 32
            block(16, 32),  # 32 -> 16
            block(32, 64),  # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # 8x8 -> 1x1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
