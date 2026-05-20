from dataclasses import dataclass
from typing import Any, Iterable

import copy
import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from fae.backbones.base import FrozenVisionBackbone


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) :
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) :
        decay = self.decay
        model_state = model.state_dict()
        shadow_state = self.shadow.state_dict()
        for key, value in shadow_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(value):
                value.copy_(model_value)
                continue
            value.mul_(decay).add_(model_value.to(device=value.device, dtype=value.dtype), alpha=1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) :
        self.decay = float(state.get("decay", self.decay))
        self.shadow.load_state_dict(state["shadow"])


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


def accumulation_steps_from_config(train_cfg: dict[str, Any]) -> int:
    value = train_cfg.get("grad_accum_steps", train_cfg.get("accumulation_steps", 1))
    return max(int(value), 1)


def clip_grad_norm_(parameters: Iterable[torch.nn.Parameter], max_norm: float | None) -> float:
    if max_norm is None or max_norm <= 0:
        return 0.0
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(norm.detach().cpu()) if isinstance(norm, torch.Tensor) else float(norm)


def _resolve_scheduler_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    if "scheduler" in train_cfg and isinstance(train_cfg["scheduler"], dict):
        cfg = dict(train_cfg["scheduler"])
        if "type" not in cfg:
            cfg["type"] = "none"
        return cfg
    return {
        "type": train_cfg.get("lr_scheduler", "none"),
        "warmup_steps": train_cfg.get("warmup_steps"),
        "warmup_epochs": train_cfg.get("warmup_epochs"),
        "warmup_ratio": train_cfg.get("warmup_ratio"),
        "min_lr_ratio": train_cfg.get("min_lr_ratio", 0.1),
        "final_lr": train_cfg.get("final_lr"),
        "base_lr": train_cfg.get("lr"),
        "decay_end_epoch": train_cfg.get("decay_end_epoch"),
        "warmup_from_zero": train_cfg.get("warmup_from_zero", True),
    }


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    steps_per_epoch: int,
) -> LambdaLR | None:
    cfg = _resolve_scheduler_cfg(train_cfg)
    scheduler_name = str(cfg.get("type", "none")).lower()
    steps_per_epoch = max(int(steps_per_epoch), 1)
    total_steps = max(int(train_cfg.get("epochs", 1)) * steps_per_epoch, 1)
    decay_end_epoch = cfg.get("decay_end_epoch")
    decay_end_steps = total_steps if decay_end_epoch is None else max(int(float(decay_end_epoch) * steps_per_epoch), 1)

    warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    if warmup_steps <= 0 and cfg.get("warmup_epochs") is not None:
        warmup_steps = int(float(cfg["warmup_epochs"]) * steps_per_epoch)
    if warmup_steps <= 0 and cfg.get("warmup_ratio") is not None:
        warmup_steps = int(float(cfg["warmup_ratio"]) * total_steps)
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))

    base_lr = float(cfg.get("base_lr", optimizer.param_groups[0]["lr"]))
    final_lr = cfg.get("final_lr")
    warmup_from_zero = bool(cfg.get("warmup_from_zero", False))
    if final_lr is None:
        min_lr_ratio = float(cfg.get("min_lr_ratio", 0.1))
    else:
        min_lr_ratio = float(final_lr) / max(base_lr, 1e-12)

    if scheduler_name in {"none", "constant"} and warmup_steps == 0:
        return None

    def lr_lambda(step: int) -> float:
        step = min(max(step, 0), total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            if warmup_from_zero:
                return float(step + 1) / float(max(warmup_steps, 1))
            return min_lr_ratio + (1.0 - min_lr_ratio) * float(step + 1) / float(max(warmup_steps, 1))

        if scheduler_name in {"none", "constant"}:
            return 1.0

        progress = float(step - warmup_steps) / float(max(decay_end_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)

        if scheduler_name == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        if scheduler_name == "linear":
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

        raise ValueError(
            f"Unsupported scheduler.type={scheduler_name}. Use one of: none, constant, cosine, linear."
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
