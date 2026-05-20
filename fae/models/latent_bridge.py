from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentBridgeSpec:
    fae_dim: int
    fae_height: int
    fae_width: int
    model_channels: int
    model_height: int
    model_width: int


class BaseLatentAdapter(nn.Module):
    def to_model_latents(self, z_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to_fae_tokens(self, model_latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cycle_loss(self, z_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityLatentBridge(BaseLatentAdapter):
    def __init__(self, spec: LatentBridgeSpec) :
        super().__init__()
        self.spec = spec

    def to_model_latents(self, z_tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = z_tokens.shape
        expected = self.spec.fae_height * self.spec.fae_width
        if n != expected:
            raise ValueError(f"Expected {expected} FAE tokens, got {n}.")
        return z_tokens.transpose(1, 2).reshape(b, c, self.spec.fae_height, self.spec.fae_width).contiguous()

    def to_fae_tokens(self, model_latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = model_latents.shape
        expected = (self.spec.fae_dim, self.spec.fae_height, self.spec.fae_width)
        if (c, h, w) != expected:
            raise ValueError(f"Identity bridge expected model latents {expected}, got {(c, h, w)}.")
        return model_latents.reshape(b, c, h * w).transpose(1, 2).contiguous()

    def cycle_loss(self, z_tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=z_tokens.device, dtype=z_tokens.dtype)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) :
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LatentBridge(BaseLatentAdapter):
    def __init__(
        self,
        spec: LatentBridgeSpec,
        hidden_channels: int | None = None,
        num_res_blocks: int = 2,
        resize_mode: str = "bilinear",
    ) :
        super().__init__()
        self.spec = spec
        hidden_channels = hidden_channels or max(spec.fae_dim, spec.model_channels)
        self.resize_mode = resize_mode

        to_model_layers: list[nn.Module] = [nn.Conv2d(spec.fae_dim, hidden_channels, kernel_size=1)]
        to_model_layers.extend(ResidualConvBlock(hidden_channels) for _ in range(num_res_blocks))
        to_model_layers.append(nn.Conv2d(hidden_channels, spec.model_channels, kernel_size=1))
        self.to_model = nn.Sequential(*to_model_layers)

        to_fae_layers: list[nn.Module] = [nn.Conv2d(spec.model_channels, hidden_channels, kernel_size=1)]
        to_fae_layers.extend(ResidualConvBlock(hidden_channels) for _ in range(num_res_blocks))
        to_fae_layers.append(nn.Conv2d(hidden_channels, spec.fae_dim, kernel_size=1))
        self.to_fae = nn.Sequential(*to_fae_layers)

    def _tokens_to_grid(self, z_tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = z_tokens.shape
        expected = self.spec.fae_height * self.spec.fae_width
        if n != expected:
            raise ValueError(f"Expected {expected} FAE tokens, got {n}.")
        return z_tokens.transpose(1, 2).reshape(b, d, self.spec.fae_height, self.spec.fae_width)

    def _grid_to_tokens(self, grid: torch.Tensor) -> torch.Tensor:
        b, c, h, w = grid.shape
        if (h, w) != (self.spec.fae_height, self.spec.fae_width):
            raise ValueError(f"Expected FAE grid {(self.spec.fae_height, self.spec.fae_width)}, got {(h, w)}.")
        return grid.reshape(b, c, h * w).transpose(1, 2).contiguous()

    def to_model_latents(self, z_tokens: torch.Tensor) -> torch.Tensor:
        x = self._tokens_to_grid(z_tokens)
        if (x.shape[-2], x.shape[-1]) != (self.spec.model_height, self.spec.model_width):
            x = F.interpolate(
                x,
                size=(self.spec.model_height, self.spec.model_width),
                mode=self.resize_mode,
                align_corners=False if self.resize_mode in {"bilinear", "bicubic"} else None,
            )
        return self.to_model(x)

    def to_fae_tokens(self, model_latents: torch.Tensor) -> torch.Tensor:
        x = model_latents
        if (x.shape[-2], x.shape[-1]) != (self.spec.fae_height, self.spec.fae_width):
            x = F.interpolate(
                x,
                size=(self.spec.fae_height, self.spec.fae_width),
                mode=self.resize_mode,
                align_corners=False if self.resize_mode in {"bilinear", "bicubic"} else None,
            )
        x = self.to_fae(x)
        return self._grid_to_tokens(x)

    def cycle_loss(self, z_tokens: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.to_fae_tokens(self.to_model_latents(z_tokens)), z_tokens)
