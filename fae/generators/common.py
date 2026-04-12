from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LatentTensorSpec:
    channels: int
    height: int
    width: int


@dataclass
class ConditioningBundle:
    vector: torch.Tensor | None = None
    class_labels: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None
    pooled_projections: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "ConditioningBundle":
        for name in ["vector", "class_labels", "encoder_hidden_states", "pooled_projections", "attention_mask"]:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        for key, value in list(self.extra_kwargs.items()):
            if isinstance(value, torch.Tensor):
                self.extra_kwargs[key] = value.to(device)
        return self


@dataclass
class LossOutput:
    loss: torch.Tensor
    logs: dict[str, float]
