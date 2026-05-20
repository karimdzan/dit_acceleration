
"""Token-merging baselines for DiT-style latent transformers.

This module contains two implementations:

* ``spatial_pool``: the earlier deterministic average-pooling baseline.
* ``toma``: a ToMA-style attention-only merge for DiT blocks.

The ToMA-style path is designed for class-conditional DiT-XL in diffusers. It
implements the three core ideas:

1. choose destination tokens with a facility-location objective,
2. merge/unmerge via attention-like linear transformations,
3. perform the operations inside local image tiles and only around attention.
"""

from dataclasses import dataclass
from typing import Any, Iterable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TokenMergingStats:
    original_tokens: int
    merged_tokens: int
    prefix_tokens: int
    grid_size: tuple[int, int]
    merged_grid_size: tuple[int, int]

    @property
    def keep_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 1.0
        return self.merged_tokens / self.original_tokens


class SpatialTokenPooler(nn.Module):
    def __init__(self, stride: int = 2, num_prefix_tokens: int = 0, ceil_mode: bool = False):
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")
        if num_prefix_tokens < 0:
            raise ValueError(f"num_prefix_tokens must be >= 0, got {num_prefix_tokens}.")
        self.stride = int(stride)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.ceil_mode = bool(ceil_mode)

    def merge(self, x: torch.Tensor, grid_size: tuple[int, int]) -> tuple[torch.Tensor, TokenMergingStats]:
        if x.ndim != 3:
            raise ValueError(f"Expected token tensor [B, N, C], got shape {tuple(x.shape)}.")
        h, w = int(grid_size[0]), int(grid_size[1])
        expected_image_tokens = h * w
        prefix = self.num_prefix_tokens
        if x.shape[1] != prefix + expected_image_tokens:
            raise ValueError(
                f"Token count mismatch: got N={x.shape[1]}, expected prefix({prefix}) + H*W({h}*{w}={expected_image_tokens})."
            )

        if self.stride == 1:
            stats = TokenMergingStats(
                original_tokens=x.shape[1],
                merged_tokens=x.shape[1],
                prefix_tokens=prefix,
                grid_size=(h, w),
                merged_grid_size=(h, w),
            )
            return x, stats

        prefix_tokens = x[:, :prefix] if prefix else None
        image_tokens = x[:, prefix:]
        b, _, c = image_tokens.shape
        image = image_tokens.transpose(1, 2).reshape(b, c, h, w)
        pooled = F.avg_pool2d(image, kernel_size=self.stride, stride=self.stride, ceil_mode=self.ceil_mode)
        mh, mw = int(pooled.shape[-2]), int(pooled.shape[-1])
        merged_image_tokens = pooled.flatten(2).transpose(1, 2)
        if prefix_tokens is not None:
            merged = torch.cat([prefix_tokens, merged_image_tokens], dim=1)
        else:
            merged = merged_image_tokens
        stats = TokenMergingStats(
            original_tokens=x.shape[1],
            merged_tokens=merged.shape[1],
            prefix_tokens=prefix,
            grid_size=(h, w),
            merged_grid_size=(mh, mw),
        )
        return merged, stats

    def unmerge(self, x: torch.Tensor, stats: TokenMergingStats) -> torch.Tensor:
        if self.stride == 1:
            return x
        prefix = stats.prefix_tokens
        prefix_tokens = x[:, :prefix] if prefix else None
        image_tokens = x[:, prefix:]
        b, _, c = image_tokens.shape
        mh, mw = stats.merged_grid_size
        h, w = stats.grid_size
        image = image_tokens.transpose(1, 2).reshape(b, c, mh, mw)
        restored = F.interpolate(image, size=(h, w), mode="nearest")
        restored_tokens = restored.flatten(2).transpose(1, 2)
        if prefix_tokens is not None:
            return torch.cat([prefix_tokens, restored_tokens], dim=1)
        return restored_tokens


class SpatialTokenMergingWrapper(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        stride: int = 2,
        num_prefix_tokens: int = 0,
        min_image_tokens: int = 64,
        ceil_mode: bool = False,
        enabled: bool = True,
    ):
        super().__init__()
        self.block = block
        self.pooler = SpatialTokenPooler(stride=stride, num_prefix_tokens=num_prefix_tokens, ceil_mode=ceil_mode)
        self.min_image_tokens = int(min_image_tokens)
        self.enabled = bool(enabled)
        self.calls = 0
        self.last_stats: TokenMergingStats | None = None

    def _resolve_grid_size(self, x: torch.Tensor, kwargs: dict[str, Any]) -> tuple[int, int] | None:
        grid_size = kwargs.get("grid_size")
        if grid_size is not None:
            return int(grid_size[0]), int(grid_size[1])
        image_tokens = x.shape[1] - self.pooler.num_prefix_tokens
        side = int(image_tokens ** 0.5)
        if side * side == image_tokens:
            return side, side
        return None

    def _set_grid_size(self, kwargs: dict[str, Any], stats: TokenMergingStats) -> dict[str, Any]:
        if "grid_size" not in kwargs:
            return kwargs
        kwargs = dict(kwargs)
        kwargs["grid_size"] = stats.merged_grid_size
        return kwargs

    @staticmethod
    def _replace_first_tensor_output(output: Any, new_tensor: torch.Tensor) -> Any:
        if isinstance(output, torch.Tensor):
            return new_tensor
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (new_tensor, *output[1:])
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            return [new_tensor, *output[1:]]
        raise TypeError(f"Unsupported transformer block output type: {type(output)!r}.")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], torch.Tensor):
            x = args[0]
            args_tail = args[1:]
            input_mode = "args"
        elif isinstance(kwargs.get("hidden_states"), torch.Tensor):
            x = kwargs["hidden_states"]
            args_tail = args
            input_mode = "hidden_states"
        else:
            return self.block(*args, **kwargs)

        grid_size = self._resolve_grid_size(x, kwargs)
        image_tokens = x.shape[1] - self.pooler.num_prefix_tokens
        if not self.enabled or grid_size is None or self.pooler.stride == 1 or image_tokens < self.min_image_tokens:
            return self.block(*args, **kwargs)

        x_merged, stats = self.pooler.merge(x, grid_size=grid_size)
        block_kwargs = self._set_grid_size(kwargs, stats)
        if input_mode == "args":
            block_output = self.block(x_merged, *args_tail, **block_kwargs)
        else:
            block_kwargs = dict(block_kwargs)
            block_kwargs["hidden_states"] = x_merged
            block_output = self.block(*args_tail, **block_kwargs)

        if isinstance(block_output, torch.Tensor):
            merged_output = block_output
        elif isinstance(block_output, (tuple, list)) and block_output and isinstance(block_output[0], torch.Tensor):
            merged_output = block_output[0]
        else:
            raise TypeError(f"Unsupported transformer block output type: {type(block_output)!r}.")

        restored = self.pooler.unmerge(merged_output, stats)
        self.calls += 1
        self.last_stats = stats
        return self._replace_first_tensor_output(block_output, restored)


@dataclass
class TomaMergePlan:
    merge_matrix: torch.Tensor  # [B, T, D, S]
    unmerge_matrix: torch.Tensor  # [B, T, S, D]
    num_tiles_side: int
    tile_h: int
    tile_w: int
    num_tiles: int
    dst_per_tile: int
    original_tokens: int
    merged_tokens: int

    @property
    def keep_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 1.0
        return self.merged_tokens / self.original_tokens


class TomaMergeController:
    """State shared across wrapped blocks"""

    def __init__(
        self,
        wrapped_layers: list[int],
        recompute_steps: int = 1,
        reuse_across_layers: bool = True,
    ):
        self.wrapped_layers = sorted(wrapped_layers)
        self.first_layer = self.wrapped_layers[0] if self.wrapped_layers else None
        self.recompute_steps = max(1, int(recompute_steps))
        self.reuse_across_layers = bool(reuse_across_layers)
        self.step_index = -1
        self.plan: TomaMergePlan | None = None
        self.plan_recomputes = 0

    def begin_layer(self, layer_idx: int) -> bool:
        if self.first_layer is not None and layer_idx == self.first_layer:
            self.step_index += 1
            if (self.step_index % self.recompute_steps) == 0:
                return True
        return False if self.reuse_across_layers else True

    def update_plan(self, plan: TomaMergePlan):
        self.plan = plan
        self.plan_recomputes += 1


def _square_side(n: int) -> int | None:
    side = int(math.isqrt(int(n)))
    return side if side * side == int(n) else None


def _tile_layout(num_tokens: int, num_tiles: int) -> tuple[int, int, int, int] | None:
    side = _square_side(num_tokens)
    tiles_side = _square_side(num_tiles)
    if side is None or tiles_side is None or tiles_side <= 0 or side % tiles_side != 0:
        return None
    tile_h = side // tiles_side
    tile_w = tile_h
    return side, tiles_side, tile_h, tile_w


def _fold_tiles(x: torch.Tensor, num_tiles: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    b, n, c = x.shape
    layout = _tile_layout(n, num_tiles)
    if layout is None:
        raise ValueError(f"Cannot split N={n} tokens into {num_tiles} square local tiles.")
    side, tiles_side, tile_h, tile_w = layout
    img = x.view(b, side, side, c)
    tiles = img.view(b, tiles_side, tile_h, tiles_side, tile_w, c).permute(0, 1, 3, 2, 4, 5)
    tiles = tiles.reshape(b, tiles_side * tiles_side, tile_h * tile_w, c)
    return tiles, (side, tiles_side, tile_h, tile_w)


def _unfold_tiles(tiles: torch.Tensor, layout: tuple[int, int, int, int]) -> torch.Tensor:
    b, num_tiles, tile_tokens, c = tiles.shape
    side, tiles_side, tile_h, tile_w = layout
    if num_tiles != tiles_side * tiles_side or tile_tokens != tile_h * tile_w:
        raise ValueError("Tile tensor shape does not match the requested layout.")
    img = tiles.view(b, tiles_side, tiles_side, tile_h, tile_w, c).permute(0, 1, 3, 2, 4, 5)
    img = img.reshape(b, side, side, c)
    return img.reshape(b, side * side, c)


def batched_facility_location(x: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy facility-location selection used by ToMA

    Args:
        x: [B, N, C] tensor.
        k: number of destination tokens to select.
    Returns:
        [B, k] token indices.
    """

    b, n, _ = x.shape
    if not (0 < k <= n):
        raise ValueError(f"k must be in [1, N], got k={k}, N={n}.")
    x = F.normalize(x.float(), dim=-1)
    sim = x @ x.transpose(-1, -2)  # [B, N, N]
    row_sums = sim.sum(dim=-1)
    first = row_sums.argmax(dim=-1)  # [B]

    reps = torch.zeros(b, k, dtype=torch.long, device=x.device)
    reps[:, 0] = first
    batch_idx = torch.arange(b, device=x.device)
    max_sim = sim[batch_idx, first]  # [B, N]
    selected_mask = torch.zeros(b, n, dtype=torch.bool, device=x.device)
    selected_mask[batch_idx, first] = True

    for i in range(1, k):
        gains = torch.relu(sim - max_sim.unsqueeze(1)).sum(dim=-1)  # [B, N]
        gains = gains.masked_fill(selected_mask, float("-inf"))
        nxt = gains.argmax(dim=-1)
        reps[:, i] = nxt
        selected_mask[batch_idx, nxt] = True
        max_sim = torch.maximum(max_sim, sim[batch_idx, nxt])
    return reps


def local_tile_wise_facility(
    x: torch.Tensor,
    keep_tokens: int,
    num_tiles: int,
    facility_batch: str = "first",
) -> tuple[torch.Tensor, int]:
    """Select destination tokens within local tiles"""

    tiles, _ = _fold_tiles(x, num_tiles)
    b, t, s, c = tiles.shape
    keep_ratio = keep_tokens / max(1, x.shape[1])
    dst_per_tile = max(1, min(s, int(round(s * keep_ratio))))
    if dst_per_tile >= s:
        dst = torch.arange(s, device=x.device).view(1, 1, s).expand(b, t, s)
        return dst, s

    if facility_batch == "first":
        facility_tiles = tiles[:1].reshape(t, s, c)
        dst_first = batched_facility_location(facility_tiles, dst_per_tile).reshape(1, t, dst_per_tile)
        dst = dst_first.expand(b, -1, -1)
    elif facility_batch == "all":
        flat_tiles = tiles.reshape(b * t, s, c)
        dst = batched_facility_location(flat_tiles, dst_per_tile).reshape(b, t, dst_per_tile)
    else:
        raise ValueError(f"Unknown facility_batch={facility_batch!r}; expected 'first' or 'all'.")
    return dst, dst_per_tile


def build_toma_plan(
    x: torch.Tensor,
    ratio: float,
    num_tiles: int,
    attention_scale: float,
    facility_batch: str = "first",
) -> TomaMergePlan | None:
    b, n, c = x.shape
    if ratio <= 0.0 or n < 4:
        return None
    merge_tokens = int(round(n * float(ratio)))
    keep_tokens = n - merge_tokens
    if keep_tokens <= 0 or keep_tokens >= n:
        return None

    layout = _tile_layout(n, num_tiles)
    if layout is None:
        num_tiles = 1
        layout = _tile_layout(n, num_tiles)
    if layout is None:
        return None

    side, tiles_side, tile_h, tile_w = layout
    tiles, _ = _fold_tiles(x, num_tiles)
    b, t, s, c = tiles.shape

    dst_idx, dst_per_tile = local_tile_wise_facility(
        x,
        keep_tokens=keep_tokens,
        num_tiles=num_tiles,
        facility_batch=facility_batch,
    )
    if dst_per_tile >= s:
        return None

    x_norm = F.normalize(tiles.float(), dim=-1)
    gather_index = dst_idx.unsqueeze(-1).expand(-1, -1, -1, c)
    dst = torch.gather(x_norm, dim=2, index=gather_index)  # [B, T, D, C]

    scores = torch.matmul(dst, x_norm.transpose(-1, -2))  # [B, T, D, S]
    assign = torch.softmax(scores * float(attention_scale), dim=-2)
    count_per_dst = assign.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    merge_matrix = (assign / count_per_dst).to(dtype=x.dtype)
    unmerge_matrix = assign.transpose(-1, -2).to(dtype=x.dtype)
    merged_tokens = int(num_tiles * dst_per_tile)
    return TomaMergePlan(
        merge_matrix=merge_matrix,
        unmerge_matrix=unmerge_matrix,
        num_tiles_side=tiles_side,
        tile_h=tile_h,
        tile_w=tile_w,
        num_tiles=num_tiles,
        dst_per_tile=dst_per_tile,
        original_tokens=n,
        merged_tokens=merged_tokens,
    )


def toma_merge_tokens(x: torch.Tensor, plan: TomaMergePlan) -> torch.Tensor:
    tiles, _ = _fold_tiles(x, plan.num_tiles)
    merged_tiles = torch.matmul(plan.merge_matrix, tiles)
    return merged_tiles.reshape(x.shape[0], plan.num_tiles * plan.dst_per_tile, x.shape[-1])


def toma_unmerge_tokens(x: torch.Tensor, plan: TomaMergePlan) -> torch.Tensor:
    b, _, c = x.shape
    merged_tiles = x.view(b, plan.num_tiles, plan.dst_per_tile, c)
    restored_tiles = torch.matmul(plan.unmerge_matrix, merged_tiles)
    layout = (_square_side(plan.original_tokens), plan.num_tiles_side, plan.tile_h, plan.tile_w)
    return _unfold_tiles(restored_tiles, layout)


class TomaBasicTransformerBlockWrapper(nn.Module):
    """The wrapper merges the attention path. The MLP path remains dense and unchanged"""

    def __init__(
        self,
        block: nn.Module,
        layer_idx: int,
        controller: TomaMergeController,
        ratio: float = 0.25,
        num_tiles: int = 16,
        attention_scale: float = 1000.0,
        min_image_tokens: int = 64,
        enabled: bool = True,
        facility_batch: str = "first",
    ):
        super().__init__()
        self.block = block
        self.layer_idx = int(layer_idx)
        self.controller = controller
        self.ratio = float(ratio)
        self.num_tiles = int(num_tiles)
        self.attention_scale = float(attention_scale)
        self.min_image_tokens = int(min_image_tokens)
        self.enabled = bool(enabled)
        self.facility_batch = str(facility_batch)
        self.calls = 0
        self.last_stats: TomaMergePlan | None = None

    def _compute_or_get_plan(self, norm_hidden_states: torch.Tensor) -> TomaMergePlan | None:
        should_recompute = self.controller.begin_layer(self.layer_idx)
        if should_recompute or self.controller.plan is None:
            plan = build_toma_plan(
                norm_hidden_states,
                ratio=self.ratio,
                num_tiles=self.num_tiles,
                attention_scale=self.attention_scale,
                facility_batch=self.facility_batch,
            )
            self.controller.update_plan(plan)
        return self.controller.plan

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        class_labels: torch.LongTensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        block = self.block
        if not self.enabled or hidden_states.ndim != 3 or hidden_states.shape[1] < self.min_image_tokens:
            return block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                added_cond_kwargs=added_cond_kwargs,
            )

        if getattr(block, "norm_type", None) != "ada_norm_zero" or getattr(block, "attn2", None) is not None:
            return block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                added_cond_kwargs=added_cond_kwargs,
            )

        batch_size = hidden_states.shape[0]
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
            hidden_states,
            timestep,
            class_labels,
            hidden_dtype=hidden_states.dtype,
        )

        if block.pos_embed is not None:
            norm_hidden_states = block.pos_embed(norm_hidden_states)

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        plan = self._compute_or_get_plan(norm_hidden_states)
        if plan is None:
            attn_output = block.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if block.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        else:
            merged_hidden_states = toma_merge_tokens(norm_hidden_states, plan)
            attn_merged = block.attn1(
                merged_hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                **cross_attention_kwargs,
            )
            attn_output = toma_unmerge_tokens(attn_merged, plan)
            self.last_stats = plan

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if gligen_kwargs is not None and hasattr(block, "fuser"):
            hidden_states = block.fuser(hidden_states, gligen_kwargs["objs"])

        if block.norm_type in ["ada_norm_zero", "ada_norm", "layer_norm"]:
            norm_hidden_states = block.norm3(hidden_states)
        elif block.norm_type == "ada_norm_continuous":
            norm_hidden_states = block.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif block.norm_type == "ada_norm_single":
            norm_hidden_states = block.norm2(hidden_states)
        elif block.norm_type == "layer_norm_i2vgen":
            norm_hidden_states = hidden_states
        else:
            raise ValueError(f"Unsupported norm_type {block.norm_type!r}.")

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if block._chunk_size is not None:
            ff_output = _chunked_feed_forward(block.ff, norm_hidden_states, block._chunk_dim, block._chunk_size)
        else:
            ff_output = block.ff(norm_hidden_states)

        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = ff_output + hidden_states
        self.calls += 1
        return hidden_states


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int) -> torch.Tensor:
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension {hidden_states.shape[chunk_dim]} must be divisible by chunk size {chunk_size}."
        )
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    return torch.cat([ff(hid) for hid in hidden_states.chunk(num_chunks, dim=chunk_dim)], dim=chunk_dim)

def _normalize_layer_indices(num_layers: int, cfg: dict[str, Any]) -> list[int]:
    explicit = cfg.get("layers")
    if explicit is not None:
        if isinstance(explicit, str):
            explicit = [int(x.strip()) for x in explicit.split(",") if x.strip()]
        return sorted({int(i) for i in explicit if 0 <= int(i) < num_layers})

    start = cfg.get("start_layer")
    end = cfg.get("end_layer")
    every = int(cfg.get("every", 1))

    if start is None:
        start = max(0, num_layers // 4)
    if end is None:
        end = min(num_layers, (3 * num_layers) // 4)
    start, end = int(start), int(end)
    return list(range(max(0, start), min(num_layers, end), max(1, every)))


def _get_block_container(model: nn.Module) -> tuple[str, nn.ModuleList | nn.Sequential | list[nn.Module]]:
    for name in ("transformer_blocks", "blocks"):
        container = getattr(model, name, None)
        if isinstance(container, (nn.ModuleList, nn.Sequential, list)):
            return name, container
    raise ValueError(
        "Could not find a transformer block container. Expected `model.transformer_blocks` "
        "for diffusers DiT/SANA/SD3 or `model.blocks` for the internal DiT."
    )


def apply_spatial_token_merging(model: nn.Module, cfg: dict[str, Any] | None) -> list[int]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return []

    _, blocks = _get_block_container(model)
    indices = _normalize_layer_indices(len(blocks), cfg)
    wrapped: list[int] = []
    for idx in indices:
        block = blocks[idx]
        if isinstance(block, SpatialTokenMergingWrapper):
            continue
        blocks[idx] = SpatialTokenMergingWrapper(
            block,
            stride=int(cfg.get("stride", 2)),
            num_prefix_tokens=int(cfg.get("num_prefix_tokens", 0)),
            min_image_tokens=int(cfg.get("min_image_tokens", 64)),
            ceil_mode=bool(cfg.get("ceil_mode", False)),
            enabled=bool(cfg.get("enabled", True)),
        )
        wrapped.append(idx)
    setattr(model, "token_merging_layers", wrapped)
    return wrapped


def apply_toma_token_merging(model: nn.Module, cfg: dict[str, Any] | None) -> list[int]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return []

    _, blocks = _get_block_container(model)
    indices = _normalize_layer_indices(len(blocks), cfg)
    if not indices:
        return []
    controller = TomaMergeController(
        wrapped_layers=indices,
        recompute_steps=int(cfg.get("recompute_steps", 1)),
        reuse_across_layers=bool(cfg.get("reuse_across_layers", True)),
    )

    wrapped: list[int] = []
    for idx in indices:
        block = blocks[idx]
        if isinstance(block, TomaBasicTransformerBlockWrapper):
            continue
        blocks[idx] = TomaBasicTransformerBlockWrapper(
            block,
            layer_idx=idx,
            controller=controller,
            ratio=float(cfg.get("ratio", 0.25)),
            num_tiles=int(cfg.get("num_tiles", 16)),
            attention_scale=float(cfg.get("attention_scale", 1000.0)),
            min_image_tokens=int(cfg.get("min_image_tokens", 64)),
            enabled=bool(cfg.get("enabled", True)),
            facility_batch=str(cfg.get("facility_batch", "first")),
        )
        wrapped.append(idx)
    setattr(model, "token_merging_layers", wrapped)
    setattr(model, "token_merging_controller", controller)
    return wrapped


def apply_token_merging(model: nn.Module, cfg: dict[str, Any] | None) -> list[int]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return []
    method = str(cfg.get("method", "toma")).lower()
    if method in {"spatial_pool", "local_pool", "token_pool"}:
        return apply_spatial_token_merging(model, cfg)
    if method in {"toma", "attention_merge", "toma_dit"}:
        return apply_toma_token_merging(model, cfg)
    raise ValueError(f"Unsupported token merging method={method!r}. Supported: spatial_pool, toma.")


def iter_token_merging_wrappers(model: nn.Module) -> Iterable[nn.Module]:
    for module in model.modules():
        if isinstance(module, (SpatialTokenMergingWrapper, TomaBasicTransformerBlockWrapper)):
            yield module
