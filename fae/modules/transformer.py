import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.value = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate(x)) * self.value(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.gates = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gate_attn, gate_mlp = self.gates(cond).chunk(2, dim=-1)
        h = self.norm1(x, cond)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.norm2(x, cond))
        return x


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)
