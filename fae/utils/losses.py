from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class ReconstructionLossBreakdown:
    total: torch.Tensor
    reconstruction: torch.Tensor
    perceptual: torch.Tensor
    adversarial: torch.Tensor


class VGGPerceptualLoss(nn.Module):
    """Lightweight perceptual loss using frozen VGG16 features."""

    def __init__(self, resize_to: int = 224) :
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_FEATURES
        features = models.vgg16(weights=weights).features[:16]
        self.features = features.eval()
        self.resize_to = resize_to
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        for p in self.features.parameters():
            p.requires_grad = False

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1.0, 1.0).add(1.0).mul(0.5)
        x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._preprocess(pred)
        target = self._preprocess(target)
        return F.l1_loss(self.features(pred), self.features(target))


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def image_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    perceptual_loss: nn.Module | None = None,
    disc_fake_logits: torch.Tensor | None = None,
    recon_weight: float = 1.0,
    perceptual_weight: float = 0.1,
    adversarial_weight: float = 0.01,
) -> ReconstructionLossBreakdown:
    recon = F.l1_loss(pred, target) * recon_weight
    perceptual = pred.new_tensor(0.0)
    adv = pred.new_tensor(0.0)
    if perceptual_loss is not None:
        perceptual = perceptual_loss(pred, target) * perceptual_weight
    if disc_fake_logits is not None:
        adv = generator_hinge_loss(disc_fake_logits) * adversarial_weight
    total = recon + perceptual + adv
    return ReconstructionLossBreakdown(total=total, reconstruction=recon, perceptual=perceptual, adversarial=adv)
