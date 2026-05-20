from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from fae.models.pixel_decoder import ViTPixelDecoder
from fae.utils.image import infer_hw_from_tokens


@dataclass
class RAEOutput:
    z: torch.Tensor
    reconstructed_images: torch.Tensor


class RepresentationAutoEncoder(nn.Module):
    """Decoder that reconstructs images from frozen backbone patch tokens"""

    def __init__(
        self,
        input_dim: int,
        image_size: int = 256,
        patch_size: int = 16,
        decoder_hidden_dim: int | None = None,
        decoder_layers: int = 24,
        decoder_heads: int = 16,
        decoder_head_dim: int | None = None,
        decoder_mlp_ratio: float = 4.0,
        decoder_use_rope_2d: bool = True,
        rope_base: float = 10000.0,
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalize_latents: bool = False,
        latent_mean: torch.Tensor | None = None,
        latent_var: torch.Tensor | None = None,
        latent_eps: float = 1e-6,
    ) :
        super().__init__()
        self.input_dim = int(input_dim)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.noise_tau = float(noise_tau)
        self.reshape_to_2d = bool(reshape_to_2d)
        self.use_latent_normalization = bool(normalize_latents)
        self.latent_eps = float(latent_eps)

        hidden_dim = int(decoder_hidden_dim or input_dim)
        self.decoder = ViTPixelDecoder(
            input_dim=input_dim,
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=int(decoder_layers),
            num_heads=int(decoder_heads),
            head_dim=decoder_head_dim,
            mlp_ratio=float(decoder_mlp_ratio),
            use_rope_2d=bool(decoder_use_rope_2d),
            rope_base=float(rope_base),
        )

        self.register_buffer("latent_mean", torch.zeros(1, input_dim, 1, 1, dtype=torch.float32))
        self.register_buffer("latent_var", torch.ones(1, input_dim, 1, 1, dtype=torch.float32))
        self._latent_stats_mode = "spatial"
        if latent_mean is not None or latent_var is not None:
            self.set_latent_stats(latent_mean, latent_var)

    def set_latent_stats(self, mean: torch.Tensor | None, var: torch.Tensor | None) :
        if mean is None or var is None:
            raise ValueError("Both latent_mean and latent_var must be provided together.")
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        var_t = torch.as_tensor(var, dtype=torch.float32)
        if mean_t.ndim == 1:
            if mean_t.numel() != self.input_dim or var_t.numel() != self.input_dim:
                raise ValueError(
                    f"Channel-only latent stats must have {self.input_dim} values, got "
                    f"mean={mean_t.numel()} var={var_t.numel()}."
                )
            self._latent_stats_mode = "channel"
            self.latent_mean = mean_t.view(1, self.input_dim, 1, 1).clone()
            self.latent_var = var_t.view(1, self.input_dim, 1, 1).clamp(min=self.latent_eps).clone()
            return
        if mean_t.ndim == 3:
            mean_t = mean_t.unsqueeze(0)
            var_t = var_t.unsqueeze(0)
        if mean_t.ndim != 4 or mean_t.shape[1] != self.input_dim:
            raise ValueError(
                f"Spatial latent stats must have shape [C,H,W] or [1,C,H,W] with C={self.input_dim}, got {tuple(mean_t.shape)}."
            )
        if tuple(mean_t.shape) != tuple(var_t.shape):
            raise ValueError(f"Latent mean/var shape mismatch: {tuple(mean_t.shape)} vs {tuple(var_t.shape)}")
        if mean_t.shape[-2:] == (1, 1):
            self._latent_stats_mode = "channel"
        else:
            self._latent_stats_mode = "spatial"
        self.latent_mean = mean_t.clone()
        self.latent_var = var_t.clamp(min=self.latent_eps).clone()

    def _tokens_to_grid(self, z_tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = z_tokens.shape
        if c != self.input_dim:
            raise ValueError(f"Expected token dim {self.input_dim}, got {c}.")
        h, w = infer_hw_from_tokens(n)
        return z_tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()

    def _grid_to_tokens(self, z_grid: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z_grid.shape
        if c != self.input_dim:
            raise ValueError(f"Expected latent channels {self.input_dim}, got {c}.")
        return z_grid.reshape(b, c, h * w).transpose(1, 2).contiguous()

    def _check_spatial_stats_broadcastable(self, stats: torch.Tensor, z_grid: torch.Tensor) :
        stat_h, stat_w = stats.shape[-2:]
        grid_h, grid_w = z_grid.shape[-2:]
        if stat_h not in {1, grid_h} or stat_w not in {1, grid_w}:
            raise ValueError(
                f"Latent spatial stats expect broadcastable grid (*,*,1|{grid_h},1|{grid_w}), got {tuple(stats.shape[-2:])}."
            )

    def _normalize_grid(self, z_grid: torch.Tensor) -> torch.Tensor:
        if not self.use_latent_normalization:
            return z_grid
        mean = self.latent_mean.to(device=z_grid.device, dtype=z_grid.dtype)
        var = self.latent_var.to(device=z_grid.device, dtype=z_grid.dtype)
        if self._latent_stats_mode == "spatial":
            self._check_spatial_stats_broadcastable(mean, z_grid)
            self._check_spatial_stats_broadcastable(var, z_grid)
        return (z_grid - mean) / torch.sqrt(var + self.latent_eps)

    def _denormalize_grid(self, z_grid: torch.Tensor) -> torch.Tensor:
        if not self.use_latent_normalization:
            return z_grid
        mean = self.latent_mean.to(device=z_grid.device, dtype=z_grid.dtype)
        var = self.latent_var.to(device=z_grid.device, dtype=z_grid.dtype)
        if self._latent_stats_mode == "spatial":
            self._check_spatial_stats_broadcastable(mean, z_grid)
            self._check_spatial_stats_broadcastable(var, z_grid)
        return z_grid * torch.sqrt(var + self.latent_eps) + mean

    def normalize_latent_tokens(self, z_tokens: torch.Tensor) -> torch.Tensor:
        z_grid = self._tokens_to_grid(z_tokens) if self.reshape_to_2d else z_tokens.transpose(1, 2).unsqueeze(-1)
        z_grid = self._normalize_grid(z_grid)
        return self._grid_to_tokens(z_grid) if self.reshape_to_2d else z_grid.squeeze(-1).transpose(1, 2)

    def denormalize_latent_tokens(self, z_tokens: torch.Tensor) -> torch.Tensor:
        z_grid = self._tokens_to_grid(z_tokens) if self.reshape_to_2d else z_tokens.transpose(1, 2).unsqueeze(-1)
        z_grid = self._denormalize_grid(z_grid)
        return self._grid_to_tokens(z_grid) if self.reshape_to_2d else z_grid.squeeze(-1).transpose(1, 2)

    def add_latent_noise(self, z_tokens: torch.Tensor) -> torch.Tensor:
        if self.noise_tau <= 0:
            return z_tokens
        sigma = self.noise_tau * torch.rand(
            (z_tokens.shape[0],) + (1,) * (z_tokens.ndim - 1),
            device=z_tokens.device,
            dtype=z_tokens.dtype,
        )
        return z_tokens + sigma * torch.randn_like(z_tokens)

    def encode(self, features: torch.Tensor, add_noise: bool | None = None) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,N,C] features, got shape {tuple(features.shape)}.")
        z = self.normalize_latent_tokens(features)
        should_noise = self.training if add_noise is None else bool(add_noise)
        if should_noise:
            z = self.add_latent_noise(z)
        return z

    def decode(self, z_tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_tokens)

    def forward(self, features: torch.Tensor, add_noise: bool | None = None) -> RAEOutput:
        z = self.encode(features, add_noise=add_noise)
        reconstructed_images = self.decode(z)
        return RAEOutput(z=z, reconstructed_images=reconstructed_images)

    @property
    def latent_channels(self) -> int:
        return self.input_dim

    def latent_hw_from_num_tokens(self, num_tokens: int) -> tuple[int, int]:
        return infer_hw_from_tokens(num_tokens)
