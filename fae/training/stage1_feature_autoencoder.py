from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fae.backbones.base import FrozenVisionBackbone
from fae.models.fae import FeatureAutoEncoder
from fae.scripts.common import get_autocast_context
from fae.training.common import get_current_lr, prepare_backbone_inputs
from fae.utils.distributed import is_main_process, reduce_mean_dict


def train_stage1_epoch(
    model: FeatureAutoEncoder,
    backbone: FrozenVisionBackbone,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    context: torch.autocast,
    device: torch.device,
    lr_scheduler: Any | None = None,
) -> dict[str, float]:
    model.train()
    backbone.eval()
    running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    count = 0

    for batch in tqdm(dataloader, desc="stage1", leave=False, disable=not is_main_process()):
        if batch is None:
            continue
        images = batch['images']
        backbone_inputs = prepare_backbone_inputs(backbone, images, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with context:
                features = backbone.forward_features(backbone_inputs).tokens
        with context:
            loss, logs, _ = model(features, return_loss=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        for key in running:
            running[key] += logs[key]
        count += 1
    averaged = reduce_mean_dict(running, count=count, device=device)
    averaged["lr"] = get_current_lr(optimizer)
    return averaged
