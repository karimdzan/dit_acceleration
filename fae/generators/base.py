from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn

from .common import ConditioningBundle, LatentTensorSpec, LossOutput


class LatentGeneratorBackend(nn.Module, ABC):
    uses_native_prompt_encoder: bool = False

    def forward(self, latents: torch.Tensor, conditioning: ConditioningBundle | None = None) -> LossOutput:
        return self.training_loss(latents, conditioning=conditioning)

    @abstractmethod
    def latent_spec(self) -> LatentTensorSpec:
        raise NotImplementedError

    def encode_prompts(self, prompts: Sequence[str], device: torch.device) -> ConditioningBundle:
        raise RuntimeError(f"{self.__class__.__name__} does not implement native prompt encoding.")

    @abstractmethod
    def training_loss(self, latents: torch.Tensor, conditioning: ConditioningBundle | None = None) -> LossOutput:
        raise NotImplementedError

    @abstractmethod
    def sample_latents(
        self,
        batch_size: int,
        device: torch.device,
        conditioning: ConditioningBundle | None = None,
        num_steps: int = 30,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        raise NotImplementedError
