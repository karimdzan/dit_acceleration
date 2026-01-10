from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FIDResult:
    fid: float
    n_real: int
    n_fake: int


def _to_uint8(images_01: torch.Tensor) -> torch.Tensor:
    x = images_01.detach()
    if x.is_cuda:
        x = x.cpu()
    x = x.clamp(0, 1)
    return (x * 255.0).round().to(torch.uint8)

@torch.no_grad()
def compute_fid(
    *,
    real_images_01: torch.Tensor,
    fake_images_01: torch.Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
) -> FIDResult:
    from torchmetrics.image.fid import FrechetInceptionDistance

    assert real_images_01.ndim == 4 and fake_images_01.ndim == 4
    assert real_images_01.shape[1] == 3 and fake_images_01.shape[1] == 3

    device = device or (fake_images_01.device if fake_images_01.is_cuda else torch.device("cpu"))

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    real_u8 = _to_uint8(real_images_01)
    fake_u8 = _to_uint8(fake_images_01)

    n_real = int(real_u8.shape[0])
    n_fake = int(fake_u8.shape[0])

    for i in range(0, n_real, batch_size):
        fid.update(real_u8[i : i + batch_size].to(device, non_blocking=True), real=True)

    for i in range(0, n_fake, batch_size):
        fid.update(fake_u8[i : i + batch_size].to(device, non_blocking=True), real=False)

    score = fid.compute()
    return FIDResult(fid=float(score.item()), n_real=n_real, n_fake=n_fake)
