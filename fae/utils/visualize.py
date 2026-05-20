from pathlib import Path

import torch
from torchvision.utils import save_image


def denorm_unit_interval(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    if float(x.amin()) < -0.05 or float(x.amax()) > 1.05:
        return x.clamp(-1.0, 1.0).add(1.0).mul(0.5).clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0)


def save_stage2_grid(
    path: str | Path,
    target: torch.Tensor,
    pred: torch.Tensor,
    reference: torch.Tensor | None = None,
    nrow: int | None = None,
) :
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    target = denorm_unit_interval(target)
    pred = denorm_unit_interval(pred)
    tiles = [target]
    if reference is not None:
        tiles.append(denorm_unit_interval(reference))
    tiles.append(pred)
    grid = torch.cat(tiles, dim=0)
    if nrow is None:
        nrow = target.shape[0]
    save_image(grid, path, nrow=nrow)
