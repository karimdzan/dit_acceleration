from typing import Sequence

import torch
from PIL import Image
from torchvision import transforms


class TargetImageTransform:
    def __init__(self, image_size: int) :
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __call__(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transform(img) for img in images], dim=0)
