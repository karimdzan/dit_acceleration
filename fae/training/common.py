from dataclasses import dataclass
from typing import Any

import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from fae.backbones.base import FrozenVisionBackbone


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def prepare_backbone_inputs(backbone: FrozenVisionBackbone, images: list[Any], device: torch.device) -> dict[str, torch.Tensor]:
    batch = backbone.preprocess(images)
    return {k: v.to(device) for k, v in batch.items()}


def format_logs(logs: dict[str, float]) -> str:
    parts = [f"{k}={v:.4f}" for k, v in logs.items()]
    return " ".join(parts)


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    steps_per_epoch: int,
) -> LambdaLR | None:
    scheduler_name = str(train_cfg.get("lr_scheduler", "none")).lower()
    steps_per_epoch = max(int(steps_per_epoch), 1)
    total_steps = max(int(train_cfg.get("epochs", 1)) * steps_per_epoch, 1)

    warmup_steps = int(train_cfg.get("warmup_steps", 0) or 0)
    if warmup_steps <= 0 and train_cfg.get("warmup_epochs") is not None:
        warmup_steps = int(float(train_cfg["warmup_epochs"]) * steps_per_epoch)
    if warmup_steps <= 0 and train_cfg.get("warmup_ratio") is not None:
        warmup_steps = int(float(train_cfg["warmup_ratio"]) * total_steps)
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))

    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))

    if scheduler_name in {"none", "constant"} and warmup_steps == 0:
        return None

    def lr_lambda(step: int) -> float:
        step = min(max(step, 0), total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))

        if scheduler_name in {"none", "constant"}:
            return 1.0

        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)

        if scheduler_name == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        if scheduler_name == "linear":
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

        raise ValueError(
            f"Unsupported train.lr_scheduler={scheduler_name}. "
            "Use one of: none, constant, cosine, linear."
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
