from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fae.models.posterior import DiagonalGaussianPosterior
from fae.modules.rmsnorm import RMSNorm
from fae.modules.transformer import TransformerBlock


@dataclass
class FAEOutput:
    z: torch.Tensor
    posterior: DiagonalGaussianPosterior
    reconstructed_features: torch.Tensor


class SingleAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_heads: int = 8, hidden_dim: int | None = None,):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim) if hidden_dim != input_dim else nn.Identity()
        self.norm = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.to_stats = nn.Linear(hidden_dim, latent_dim * 2) 

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, DiagonalGaussianPosterior]:
        x = self.in_proj(x)
        h = self.norm(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        mean, logvar = self.to_stats(x).chunk(2, dim=-1)
        posterior = DiagonalGaussianPosterior(mean=mean, logvar=logvar.clamp(min=-30.0, max=20.0))
        z = posterior.sample() if self.training else posterior.mode()
        # z = posterior.mode()
        return z, posterior


class FeatureDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(z)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.norm(x))


class FeatureAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        encoder_heads: int = 8,
        encoder_hidden_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        kl_weight: float = 1e-6,
        normalize_features: bool = False,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        feature_eps: float = 1e-6,
    ):
        super().__init__()
        decoder_hidden_dim = decoder_hidden_dim or input_dim
        self.encoder = SingleAttentionEncoder(
            input_dim=input_dim, 
            latent_dim=latent_dim, 
            num_heads=encoder_heads, 
            hidden_dim=encoder_hidden_dim
        )
        self.decoder = FeatureDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
        )
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.kl_weight = kl_weight
        self.use_feature_normalization = bool(normalize_features)
        self.feature_eps = float(feature_eps)

        self.register_buffer("feature_mean", torch.zeros(1, 1, input_dim, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(1, 1, input_dim, dtype=torch.float32))
        if feature_mean is not None or feature_std is not None:
            self.set_feature_stats(feature_mean, feature_std)

    def set_feature_stats(self, mean: torch.Tensor | None, std: torch.Tensor | None) -> None:
        if mean is None or std is None:
            raise ValueError("Both feature_mean and feature_std must be provided together.")
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, self.input_dim)
        std_tensor = torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, self.input_dim).clamp(min=self.feature_eps)
        self.feature_mean.copy_(mean_tensor)
        self.feature_std.copy_(std_tensor)

    def normalize_feature_tokens(self, features: torch.Tensor) -> torch.Tensor:
        if not self.use_feature_normalization:
            return features
        mean = self.feature_mean.to(device=features.device, dtype=features.dtype)
        std = self.feature_std.to(device=features.device, dtype=features.dtype).clamp(min=self.feature_eps)
        return (features - mean) / std

    def denormalize_feature_tokens(self, features: torch.Tensor) -> torch.Tensor:
        if not self.use_feature_normalization:
            return features
        mean = self.feature_mean.to(device=features.device, dtype=features.dtype)
        std = self.feature_std.to(device=features.device, dtype=features.dtype).clamp(min=self.feature_eps)
        return features * std + mean

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, DiagonalGaussianPosterior]:
        normalized = self.normalize_feature_tokens(features)
        return self.encoder(normalized)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        reconstructed = self.decoder(z)
        return self.denormalize_feature_tokens(reconstructed)

    def forward(self, features: torch.Tensor) -> FAEOutput:
        normalized = self.normalize_feature_tokens(features)
        z, posterior = self.encoder(normalized)
        reconstructed_normalized = self.decoder(z)
        reconstructed = self.denormalize_feature_tokens(reconstructed_normalized)
        return FAEOutput(z=z, posterior=posterior, reconstructed_features=reconstructed)

    def compute_loss(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], FAEOutput]:
        normalized = self.normalize_feature_tokens(features)
        z, posterior = self.encoder(normalized)
        reconstructed_normalized = self.decoder(z)

        recon = F.mse_loss(reconstructed_normalized, normalized)
        kl = posterior.kl()
        total = recon + self.kl_weight * kl

        reconstructed = self.denormalize_feature_tokens(reconstructed_normalized)
        out = FAEOutput(z=z, posterior=posterior, reconstructed_features=reconstructed)
        logs = {
            "loss": float(total.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
        }

        if self.use_feature_normalization:
            raw_recon = F.mse_loss(reconstructed, features)
            logs["recon_raw"] = float(raw_recon.detach().cpu())
        return total, logs, out
