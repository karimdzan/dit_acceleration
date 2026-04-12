from dataclasses import dataclass

import torch
import torch.nn as nn

from fae.modules.rmsnorm import RMSNorm
from fae.modules.transformer import DiTBlock, SinusoidalTimestepEmbedding

from .base import LatentGeneratorBackend
from .common import ConditioningBundle, LatentTensorSpec, LossOutput
from .objectives import FlowMatchingObjective, SimpleCosineDiffusionObjective


class _InternalLatentNetwork(nn.Module):
    def __init__(self, spec: LatentTensorSpec, model_dim: int = 768, depth: int = 12, num_heads: int = 12, cond_dim: int = 1024, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.spec = spec
        self.in_proj = nn.Linear(spec.channels, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, spec.height * spec.width, model_dim))
        self.time_embed = SinusoidalTimestepEmbedding(cond_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(model_dim, num_heads=num_heads, cond_dim=cond_dim, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, spec.channels)
        self.default_cond = nn.Parameter(torch.zeros(cond_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: ConditioningBundle | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.reshape(b, c, h * w).transpose(1, 2)
        cond = self.time_embed(t)
        if conditioning is not None and conditioning.vector is not None:
            cond = cond + conditioning.vector
        else:
            cond = cond + self.default_cond.unsqueeze(0)
        y = self.in_proj(tokens) + self.pos
        for block in self.blocks:
            y = block(y, cond)
        y = self.out_proj(self.norm(y))
        return y.transpose(1, 2).reshape(b, c, h, w)


class InternalLatentDiTBackend(LatentGeneratorBackend):
    def __init__(
        self,
        spec: LatentTensorSpec,
        model_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        cond_dim: int = 1024,
        mlp_ratio: float = 4.0,
        objective: str = "diffusion",
        prediction_type: str = "v_prediction",
        time_shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.model = _InternalLatentNetwork(spec, model_dim=model_dim, depth=depth, num_heads=num_heads, cond_dim=cond_dim, mlp_ratio=mlp_ratio)
        self.objective = FlowMatchingObjective() if objective == "flow_matching" else SimpleCosineDiffusionObjective(prediction_type=prediction_type)
        self.time_shift = time_shift

    def latent_spec(self) -> LatentTensorSpec:
        return self.spec

    def training_loss(self, latents: torch.Tensor, conditioning: ConditioningBundle | None = None) -> LossOutput:
        if isinstance(self.objective, SimpleCosineDiffusionObjective):
            return self.objective.training_loss(self.model, latents, conditioning, time_shift=self.time_shift)
        return self.objective.training_loss(self.model, latents, conditioning)

    def sample_latents(self, batch_size: int, device: torch.device, conditioning: ConditioningBundle | None = None, num_steps: int = 30, guidance_scale: float = 1.0) -> torch.Tensor:
        shape = (batch_size, self.spec.channels, self.spec.height, self.spec.width)
        if isinstance(self.objective, SimpleCosineDiffusionObjective):
            return self.objective.sample(self.model, shape, device, conditioning=conditioning, num_steps=num_steps, time_shift=self.time_shift)
        return self.objective.sample(self.model, shape, device, conditioning=conditioning, num_steps=num_steps)
