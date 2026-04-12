import torch
import torch.nn as nn

from fae.modules.rmsnorm import RMSNorm
from fae.modules.transformer import TransformerBlock
from fae.utils.image import unpatchify


class ViTPixelDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        image_size: int = 256,
        patch_size: int = 16,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.patch_head = nn.Linear(hidden_dim, patch_size * patch_size * 3)
        self.image_size = image_size
        self.patch_size = patch_size

    def forward(self, reconstructed_features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(reconstructed_features)
        for block in self.blocks:
            x = block(x)
        x = self.patch_head(self.norm(x))
        return unpatchify(x, patch_size=self.patch_size, image_size=(self.image_size, self.image_size)).clamp(-1.0, 1.0)
