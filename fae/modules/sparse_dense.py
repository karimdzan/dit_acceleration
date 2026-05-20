"""Sparse-dense token block wrappers for SANA/SANA-Sprint transformers.

This baseline reduces token count through the *entire wrapped transformer block*,
including self-attention, cross-attention and GLUMBConv/MLP. It keeps a dense
residual stream outside the block:

    x_dense -> spatial downsample -> block on sparse tokens -> sparse delta
    -> spatial upsample -> x_dense + delta_dense

Training-free inference baseline: processes fewer spatial tokens through both token-mixing and feed-forward paths.
"""

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseDenseStats:
    original_tokens: int
    sparse_tokens: int
    height: int
    width: int
    sparse_height: int
    sparse_width: int
    stride: int
    residual_scale: float

    @property
    def keep_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 1.0
        return self.sparse_tokens / self.original_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "sparse_tokens": self.sparse_tokens,
            "height": self.height,
            "width": self.width,
            "sparse_height": self.sparse_height,
            "sparse_width": self.sparse_width,
            "stride": self.stride,
            "keep_ratio": self.keep_ratio,
            "residual_scale": self.residual_scale,
        }


def _as_layer_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _normalize_layer_indices(num_layers: int, cfg: dict[str, Any]) -> list[int]:
    explicit = _as_layer_list(cfg.get("layers"))
    if explicit is not None:
        return sorted({i for i in explicit if 0 <= i < num_layers})

    start = cfg.get("start_layer")
    end = cfg.get("end_layer")
    every = int(cfg.get("every", 1))
    if start is None:
        start = max(0, num_layers // 4)
    if end is None:
        end = min(num_layers, (3 * num_layers) // 4)
    start = int(start)
    end = int(end)
    return list(range(max(0, start), min(num_layers, end), max(1, every)))


def get_block_container(model: nn.Module) -> tuple[str, nn.ModuleList | nn.Sequential | list[nn.Module]]:
    for name in ("transformer_blocks", "blocks", "layers"):
        container = getattr(model, name, None)
        if isinstance(container, (nn.ModuleList, nn.Sequential, list)):
            return name, container
    raise ValueError("Could not find transformer block container: expected transformer_blocks, blocks, or layers.")


def downsample_tokens(
    hidden_states: torch.Tensor,
    height: int,
    width: int,
    stride: int,
    pool_mode: str = "avg",
) -> tuple[torch.Tensor, int, int]:
    """Downsample [B, H*W, C] spatial tokens to [B, Hs*Ws, C]"""

    if hidden_states.ndim != 3:
        raise ValueError(f"Expected [B, N, C], got {tuple(hidden_states.shape)}")
    b, n, c = hidden_states.shape
    height = int(height)
    width = int(width)
    stride = int(stride)
    if n != height * width:
        raise ValueError(f"Token count mismatch: N={n}, H*W={height}*{width}={height * width}")
    if stride <= 1:
        return hidden_states, height, width
    if height % stride != 0 or width % stride != 0:
        raise ValueError(f"H={height}, W={width} must be divisible by stride={stride}.")

    image = hidden_states.view(b, height, width, c).permute(0, 3, 1, 2).contiguous()
    if pool_mode == "avg":
        sparse = F.avg_pool2d(image, kernel_size=stride, stride=stride)
    elif pool_mode == "max":
        sparse = F.max_pool2d(image, kernel_size=stride, stride=stride)
    else:
        raise ValueError(f"Unsupported pool_mode={pool_mode!r}; expected avg or max.")
    sparse_h, sparse_w = int(sparse.shape[-2]), int(sparse.shape[-1])
    sparse_tokens = sparse.permute(0, 2, 3, 1).reshape(b, sparse_h * sparse_w, c)
    return sparse_tokens, sparse_h, sparse_w


def upsample_tokens(
    sparse_tokens: torch.Tensor,
    sparse_height: int,
    sparse_width: int,
    height: int,
    width: int,
    mode: str = "nearest",
) -> torch.Tensor:
    """Upsample [B, Hs*Ws, C] tokens to [B, H*W, C]."""

    if sparse_tokens.ndim != 3:
        raise ValueError(f"Expected [B, N, C], got {tuple(sparse_tokens.shape)}")
    b, n, c = sparse_tokens.shape
    sparse_height = int(sparse_height)
    sparse_width = int(sparse_width)
    if n != sparse_height * sparse_width:
        raise ValueError(
            f"Sparse token count mismatch: N={n}, Hs*Ws={sparse_height}*{sparse_width}={sparse_height * sparse_width}"
        )
    image = sparse_tokens.view(b, sparse_height, sparse_width, c).permute(0, 3, 1, 2).contiguous()
    if mode in {"nearest", "area"}:
        dense = F.interpolate(image, size=(int(height), int(width)), mode=mode)
    elif mode in {"bilinear", "bicubic"}:
        dense = F.interpolate(image, size=(int(height), int(width)), mode=mode, align_corners=False)
    else:
        raise ValueError(f"Unsupported upsample mode={mode!r}.")
    return dense.permute(0, 2, 3, 1).reshape(b, int(height) * int(width), c)


class SparseDenseSanaBlockWrapper(nn.Module):
    """Run a SANA transformer block on sparse spatial tokens, then restore a dense residual delta"""

    def __init__(
        self,
        block: nn.Module,
        layer_idx: int,
        stride: int = 2,
        pool_mode: str = "avg",
        upsample_mode: str = "nearest",
        residual_scale: float = 1.0,
        min_tokens: int = 256,
        enabled: bool = True,
    ) :
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.block = block
        self.layer_idx = int(layer_idx)
        self.stride = int(stride)
        self.pool_mode = str(pool_mode)
        self.upsample_mode = str(upsample_mode)
        self.residual_scale = float(residual_scale)
        self.min_tokens = int(min_tokens)
        self.enabled = bool(enabled)
        self.calls = 0
        self.last_stats: SparseDenseStats | None = None

    def _should_use_sparse(self, hidden_states: torch.Tensor, height: int | None, width: int | None) -> bool:
        if not self.enabled or self.stride <= 1 or height is None or width is None:
            return False
        if hidden_states.ndim != 3:
            return False
        if hidden_states.shape[1] != int(height) * int(width):
            return False
        if hidden_states.shape[1] < self.min_tokens:
            return False
        if int(height) % self.stride != 0 or int(width) % self.stride != 0:
            return False
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        height: int | None = None,
        width: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self._should_use_sparse(hidden_states, height, width):
            return self.block(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                height,
                width,
                *args,
                **kwargs,
            )

        dense_input = hidden_states
        sparse_input, sparse_h, sparse_w = downsample_tokens(
            dense_input,
            height=int(height),
            width=int(width),
            stride=self.stride,
            pool_mode=self.pool_mode,
        )
        sparse_output = self.block(
            sparse_input,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            sparse_h,
            sparse_w,
            *args,
            **kwargs,
        )
        sparse_delta = sparse_output - sparse_input
        dense_delta = upsample_tokens(
            sparse_delta,
            sparse_height=sparse_h,
            sparse_width=sparse_w,
            height=int(height),
            width=int(width),
            mode=self.upsample_mode,
        )
        output = dense_input + self.residual_scale * dense_delta
        self.calls += 1
        self.last_stats = SparseDenseStats(
            original_tokens=dense_input.shape[1],
            sparse_tokens=sparse_input.shape[1],
            height=int(height),
            width=int(width),
            sparse_height=sparse_h,
            sparse_width=sparse_w,
            stride=self.stride,
            residual_scale=self.residual_scale,
        )
        return output


def apply_sparse_dense_token_blocks(model: nn.Module, cfg: dict[str, Any] | None) -> list[int]:
    """Wrap selected transformer blocks with sparse-dense token processing"""

    cfg = cfg or {}
    if not bool(cfg.get("enabled", True)):
        return []
    _, blocks = get_block_container(model)
    indices = _normalize_layer_indices(len(blocks), cfg)
    wrapped: list[int] = []
    for idx in indices:
        block = blocks[idx]
        if isinstance(block, SparseDenseSanaBlockWrapper):
            continue
        blocks[idx] = SparseDenseSanaBlockWrapper(
            block=block,
            layer_idx=idx,
            stride=int(cfg.get("stride", 2)),
            pool_mode=str(cfg.get("pool_mode", "avg")),
            upsample_mode=str(cfg.get("upsample_mode", "nearest")),
            residual_scale=float(cfg.get("residual_scale", 1.0)),
            min_tokens=int(cfg.get("min_tokens", 256)),
            enabled=bool(cfg.get("enabled", True)),
        )
        wrapped.append(idx)
    setattr(model, "sparse_dense_layers", wrapped)
    setattr(model, "sparse_dense_config", dict(cfg))
    return wrapped


def iter_sparse_dense_wrappers(model: nn.Module) -> Iterable[SparseDenseSanaBlockWrapper]:
    for module in model.modules():
        if isinstance(module, SparseDenseSanaBlockWrapper):
            yield module
