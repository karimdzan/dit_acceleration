import torch
import torch.nn as nn

from fae.modules.rmsnorm import RMSNorm
from fae.modules.transformer import TransformerBlock
from fae.utils.image import infer_hw_from_tokens, unpatchify


class ViTPixelDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        image_size: int = 256,
        patch_size: int = 16,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        head_dim: int | None = None,
        mlp_ratio: float = 4.0,
        use_rope_2d: bool = True,
        rope_base: float = 10000.0,
    ) :
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                use_rope_2d=use_rope_2d,
                rope_base=rope_base,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.patch_head = nn.Linear(hidden_dim, patch_size * patch_size * 3)
        self.image_size = image_size
        self.patch_size = patch_size

    def forward(self, reconstructed_features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(reconstructed_features)
        grid_size = infer_hw_from_tokens(x.shape[1])
        for block in self.blocks:
            x = block(x, grid_size=grid_size)
        x = self.patch_head(self.norm(x))
        return unpatchify(x, patch_size=self.patch_size, image_size=(self.image_size, self.image_size))
