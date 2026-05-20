"""Group-sparse SANA/SANA-Sprint transformer patch.

Unlike per-block sparse-dense wrappers, this module pays the dense<->sparse
conversion cost once around a consecutive block group:

    dense prefix blocks
    -> downsample once
    -> sparse block group
    -> upsample once
    -> dense suffix blocks

The sparse group runs the full SANA block at lower spatial resolution, so it
reduces self-attention, cross-attention, GLUMBConv/MLP, normalization, and memory
traffic in that block group. Training-free inference baseline.
"""

from dataclasses import dataclass
from typing import Any

import types

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GroupSparseStats:
    original_layers: int
    sparse_start: int
    sparse_end: int
    num_sparse_layers: int
    original_tokens: int
    sparse_tokens: int
    height: int
    width: int
    sparse_height: int
    sparse_width: int
    stride: int
    residual_scale: float

    @property
    def token_keep_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 1.0
        return self.sparse_tokens / self.original_tokens

    @property
    def layer_sparse_ratio(self) -> float:
        if self.original_layers <= 0:
            return 0.0
        return self.num_sparse_layers / self.original_layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_layers": self.original_layers,
            "sparse_start": self.sparse_start,
            "sparse_end": self.sparse_end,
            "num_sparse_layers": self.num_sparse_layers,
            "original_tokens": self.original_tokens,
            "sparse_tokens": self.sparse_tokens,
            "height": self.height,
            "width": self.width,
            "sparse_height": self.sparse_height,
            "sparse_width": self.sparse_width,
            "stride": self.stride,
            "residual_scale": self.residual_scale,
            "token_keep_ratio": self.token_keep_ratio,
            "layer_sparse_ratio": self.layer_sparse_ratio,
        }


def _downsample_tokens(
    hidden_states: torch.Tensor,
    height: int,
    width: int,
    stride: int,
    pool_mode: str,
) -> tuple[torch.Tensor, int, int]:
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden states [B, N, C], got {tuple(hidden_states.shape)}")
    b, n, c = hidden_states.shape
    if n != height * width:
        raise ValueError(f"Token count mismatch: N={n}, H*W={height}*{width}={height * width}")
    if stride <= 1:
        return hidden_states, height, width
    if height % stride != 0 or width % stride != 0:
        raise ValueError(f"H={height}, W={width} must be divisible by stride={stride}")
    image = hidden_states.view(b, height, width, c).permute(0, 3, 1, 2).contiguous()
    if pool_mode == "avg":
        sparse = F.avg_pool2d(image, kernel_size=stride, stride=stride)
    elif pool_mode == "max":
        sparse = F.max_pool2d(image, kernel_size=stride, stride=stride)
    else:
        raise ValueError(f"Unsupported pool_mode={pool_mode!r}; expected avg or max")
    sparse_h, sparse_w = int(sparse.shape[-2]), int(sparse.shape[-1])
    sparse_tokens = sparse.permute(0, 2, 3, 1).reshape(b, sparse_h * sparse_w, c)
    return sparse_tokens, sparse_h, sparse_w


def _upsample_tokens(
    sparse_tokens: torch.Tensor,
    sparse_height: int,
    sparse_width: int,
    height: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    if sparse_tokens.ndim != 3:
        raise ValueError(f"Expected sparse tokens [B, N, C], got {tuple(sparse_tokens.shape)}")
    b, n, c = sparse_tokens.shape
    if n != sparse_height * sparse_width:
        raise ValueError(
            f"Sparse token count mismatch: N={n}, Hs*Ws={sparse_height}*{sparse_width}={sparse_height * sparse_width}"
        )
    image = sparse_tokens.view(b, sparse_height, sparse_width, c).permute(0, 3, 1, 2).contiguous()
    if mode in {"nearest", "area"}:
        dense = F.interpolate(image, size=(height, width), mode=mode)
    elif mode in {"bilinear", "bicubic"}:
        dense = F.interpolate(image, size=(height, width), mode=mode, align_corners=False)
    else:
        raise ValueError(f"Unsupported upsample_mode={mode!r}")
    return dense.permute(0, 2, 3, 1).reshape(b, height * width, c)


def _convert_attention_masks(hidden_states: torch.Tensor, attention_mask, encoder_attention_mask):
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
    return attention_mask, encoder_attention_mask


def _run_block(block, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, height, width):
    return block(
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep,
        height,
        width,
    )


def _make_group_sparse_forward(cfg: dict[str, Any]):
    sparse_start = int(cfg.get("sparse_start", cfg.get("dense_prefix_end", 6)))
    sparse_end = int(cfg.get("sparse_end", cfg.get("dense_suffix_start", 22)))
    stride = int(cfg.get("stride", 2))
    pool_mode = str(cfg.get("pool_mode", "avg"))
    upsample_mode = str(cfg.get("upsample_mode", "nearest"))
    residual_scale = float(cfg.get("residual_scale", 1.0))
    min_tokens = int(cfg.get("min_tokens", 256))
    enabled = bool(cfg.get("enabled", True))
    fallback_on_controlnet = bool(cfg.get("fallback_on_controlnet", True))

    def group_sparse_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: tuple[torch.Tensor] | None = None,
        return_dict: bool = True,
    ):
        original_forward = getattr(self, "_group_sparse_original_forward", None)
        if (
            not enabled
            or stride <= 1
            or (fallback_on_controlnet and controlnet_block_samples is not None)
            or (torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False))
        ):
            return original_forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                guidance=guidance,
                encoder_attention_mask=encoder_attention_mask,
                attention_mask=attention_mask,
                attention_kwargs=attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                return_dict=return_dict,
            )

        attention_mask, encoder_attention_mask = _convert_attention_masks(hidden_states, attention_mask, encoder_attention_mask)

        batch_size, _, height, width = hidden_states.shape
        patch_size = self.config.patch_size
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        original_tokens = post_patch_height * post_patch_width
        blocks = self.transformer_blocks
        num_layers = len(blocks)
        start = max(0, min(sparse_start, num_layers))
        end = max(start, min(sparse_end, num_layers))

        if original_tokens < min_tokens or post_patch_height % stride != 0 or post_patch_width % stride != 0 or start >= end:
            return original_forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                guidance=guidance,
                encoder_attention_mask=encoder_attention_mask,
                attention_mask=attention_mask,
                attention_kwargs=attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                return_dict=return_dict,
            )

        hidden_states = self.patch_embed(hidden_states)
        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep,
                guidance=guidance,
                hidden_dtype=hidden_states.dtype,
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep,
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        for index_block, block in enumerate(blocks[:start]):
            hidden_states = _run_block(
                block,
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_height,
                post_patch_width,
            )
            if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        dense_before_sparse = hidden_states
        sparse_states, sparse_h, sparse_w = _downsample_tokens(
            dense_before_sparse,
            post_patch_height,
            post_patch_width,
            stride=stride,
            pool_mode=pool_mode,
        )
        sparse_input = sparse_states

        for block in blocks[start:end]:
            sparse_states = _run_block(
                block,
                sparse_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                sparse_h,
                sparse_w,
            )

        sparse_delta = sparse_states - sparse_input
        dense_delta = _upsample_tokens(
            sparse_delta,
            sparse_h,
            sparse_w,
            post_patch_height,
            post_patch_width,
            mode=upsample_mode,
        )
        hidden_states = dense_before_sparse + residual_scale * dense_delta

        for relative_idx, block in enumerate(blocks[end:]):
            index_block = end + relative_idx
            hidden_states = _run_block(
                block,
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_height,
                post_patch_width,
            )
            if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_height,
            post_patch_width,
            self.config.patch_size,
            self.config.patch_size,
            -1,
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * patch_size, post_patch_width * patch_size)

        stats = GroupSparseStats(
            original_layers=num_layers,
            sparse_start=start,
            sparse_end=end,
            num_sparse_layers=end - start,
            original_tokens=original_tokens,
            sparse_tokens=sparse_h * sparse_w,
            height=post_patch_height,
            width=post_patch_width,
            sparse_height=sparse_h,
            sparse_width=sparse_w,
            stride=stride,
            residual_scale=residual_scale,
        )
        self.group_sparse_last_stats = stats.to_dict()
        self.group_sparse_calls = int(getattr(self, "group_sparse_calls", 0)) + 1

        if not return_dict:
            return (output,)
        try:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)
        except Exception:
            return {"sample": output}

    return group_sparse_forward


def apply_group_sparse_sana_transformer(model: nn.Module, cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Patch a diffusers SANA/SANA-Sprint transformer with group-sparse forward"""

    cfg = dict(cfg or {})
    cfg.setdefault("enabled", True)
    if hasattr(model, "_group_sparse_original_forward"):
        model.group_sparse_config = cfg
        return {"already_patched": True, "config": cfg}

    model._group_sparse_original_forward = model.forward
    model.forward = types.MethodType(_make_group_sparse_forward(cfg), model)
    model.group_sparse_config = cfg
    model.group_sparse_calls = 0
    model.group_sparse_last_stats = None
    return {"already_patched": False, "config": cfg}


def remove_group_sparse_sana_transformer(model: nn.Module) -> bool:
    """Restore the original forward if the model was patched."""

    if not hasattr(model, "_group_sparse_original_forward"):
        return False
    model.forward = model._group_sparse_original_forward
    delattr(model, "_group_sparse_original_forward")
    return True
