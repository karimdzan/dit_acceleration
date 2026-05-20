import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm


def _infer_square_hw(num_tokens: int) -> tuple[int, int] | None:
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        return None
    return side, side


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)


def _apply_1d_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    pair_dim = x.shape[-1] // 2
    if pair_dim == 0:
        return x
    inv_freq = 1.0 / (base ** (torch.arange(pair_dim, device=x.device, dtype=torch.float32) / max(pair_dim, 1)))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(device=x.device, dtype=x.dtype)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(device=x.device, dtype=x.dtype)
    return (x * cos.unsqueeze(0).unsqueeze(0)) + (_rotate_half(x) * sin.unsqueeze(0).unsqueeze(0))


def apply_2d_rope(q: torch.Tensor, k: torch.Tensor, grid_size: tuple[int, int], base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, num_tokens, head_dim = q.shape
    h, w = grid_size
    if h * w != num_tokens:
        raise ValueError(f"Grid size {grid_size} does not match token count {num_tokens}.")
    if head_dim % 4 != 0:
        raise ValueError(f"2D RoPE requires head_dim divisible by 4, got {head_dim}.")

    device = q.device
    pos_y = torch.arange(h, device=device).repeat_interleave(w)
    pos_x = torch.arange(w, device=device).repeat(h)

    half = head_dim // 2
    qx, qy = q[..., :half], q[..., half:]
    kx, ky = k[..., :half], k[..., half:]

    q = torch.cat([_apply_1d_rope(qx, pos_x, base=base), _apply_1d_rope(qy, pos_y, base=base)], dim=-1)
    k = torch.cat([_apply_1d_rope(kx, pos_x, base=base), _apply_1d_rope(ky, pos_y, base=base)], dim=-1)
    return q, k


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) :
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.value = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate(x)) * self.value(x))


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int | None = None,
        use_rope_2d: bool = False,
        rope_base: float = 10000.0,
    ) :
        super().__init__()
        if head_dim is None:
            if dim % num_heads != 0:
                raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads} when head_dim is not set.")
            head_dim = dim // num_heads
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = inner_dim
        self.use_rope_2d = bool(use_rope_2d)
        self.rope_base = float(rope_base)

        self.qkv = nn.Linear(dim, inner_dim * 3)
        self.out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor, grid_size: tuple[int, int] | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        resolved_grid = grid_size
        if self.use_rope_2d and resolved_grid is None:
            resolved_grid = _infer_square_hw(n)
        if self.use_rope_2d and resolved_grid is not None:
            q, k = apply_2d_rope(q, k, resolved_grid, base=self.rope_base)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.permute(0, 2, 1, 3).reshape(b, n, self.inner_dim)
        return self.out(attn)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int | None = None,
        mlp_ratio: float = 4.0,
        use_rmsnorm: bool = True,
        use_rope_2d: bool = False,
        rope_base: float = 10000.0,
    ) :
        super().__init__()
        self.norm1 = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, head_dim=head_dim, use_rope_2d=use_rope_2d, rope_base=rope_base)
        self.norm2 = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor, grid_size: tuple[int, int] | None = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, grid_size=grid_size)
        x = x + self.mlp(self.norm2(x))
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int) :
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        head_dim: int | None = None,
        use_rope_2d: bool = False,
        rope_base: float = 10000.0,
    ) :
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, head_dim=head_dim, use_rope_2d=use_rope_2d, rope_base=rope_base)
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.gates = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, grid_size: tuple[int, int] | None = None) -> torch.Tensor:
        gate_attn, gate_mlp = self.gates(cond).chunk(2, dim=-1)
        h = self.norm1(x, cond)
        x = x + gate_attn.unsqueeze(1) * self.attn(h, grid_size=grid_size)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.norm2(x, cond))
        return x


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) :
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
