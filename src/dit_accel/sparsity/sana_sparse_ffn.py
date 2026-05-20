"""Group-sparse Sana GLUMBConv feed-forward acceleration via compact dense convs."""
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F


_ORIG_FORWARD = "_dit_accel_orig_forward_sana_group_sparse"
_OBSERVER_FORWARD = "_dit_accel_orig_forward_sana_group_observer"


@dataclass(frozen=True)
class SanaFFNGroupSparseConfig:
    mode: str = "static"
    keep_ratio: float = 0.90
    group_size: int = 32
    min_skip_ratio: float = 0.05
    plan_path: str | Path | None = None
    verbose: bool = True


class _SanaSparseStats:
    def __init__(self) -> None:
        self.calls = 0
        self.dense_fallbacks = 0
        self.total_groups = 0
        self.kept_groups = 0
        self.layers_installed = 0
        self.mode = ""

    def record(self, groups: int, kept: int, fallback: bool = False) -> None:
        self.calls += 1
        self.total_groups += int(groups)
        self.kept_groups += int(kept)
        self.dense_fallbacks += int(fallback)

    def summary(self) -> dict[str, float | int | str]:
        skipped = self.total_groups - self.kept_groups
        skip_ratio = skipped / max(self.total_groups, 1)
        keep_ratio = self.kept_groups / max(self.total_groups, 1)
        return {
            "mode": self.mode,
            "layers_installed": self.layers_installed,
            "calls": self.calls,
            "dense_fallbacks": self.dense_fallbacks,
            "effective_keep_ratio": keep_ratio,
            "effective_skip_ratio": skip_ratio,
        }


class _GroupObserverState:
    def __init__(self, group_size: int) -> None:
        self.group_size = int(group_size)
        self.sums: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}
        self.hidden_channels: dict[int, int] = {}

    def record(self, layer_idx: int, z_act: torch.Tensor) -> None:
        hidden_channels = z_act.shape[1] // 2
        score = _group_scores_from_inverted_activation(z_act, hidden_channels, self.group_size)
        score_cpu = score.detach().float().cpu()
        if layer_idx not in self.sums:
            self.sums[layer_idx] = score_cpu
            self.counts[layer_idx] = 1
            self.hidden_channels[layer_idx] = hidden_channels
        else:
            self.sums[layer_idx] += score_cpu
            self.counts[layer_idx] += 1

    def build_plan(self, keep_ratio: float) -> dict[str, Any]:
        if not self.sums:
            raise RuntimeError("No Sana FFN group statistics were collected.")
        layers: dict[int, dict[str, Any]] = {}
        for layer_idx, score_sum in self.sums.items():
            avg = score_sum / max(self.counts[layer_idx], 1)
            hidden_channels = self.hidden_channels[layer_idx]
            groups = int(avg.numel())
            keep_groups = _num_keep_groups(groups, keep_ratio)
            active_groups = torch.topk(avg, k=keep_groups, largest=True).indices.sort().values.cpu()
            active_idx = _groups_to_channel_index(
                active_groups, self.group_size, hidden_channels, device=torch.device("cpu"),
            )
            layers[int(layer_idx)] = {
                "score": avg.cpu(),
                "active_groups": active_groups.cpu(),
                "active_idx": active_idx.cpu(),
                "hidden_channels": int(hidden_channels),
                "num_groups": int(groups),
                "keep_groups": int(keep_groups),
            }
        return {
            "format": "dit_accel.sana_ffn_group_plan.v1",
            "keep_ratio": float(keep_ratio),
            "group_size": int(self.group_size),
            "layers": layers,
        }


def save_group_plan(plan: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(plan, path)


def load_group_plan(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _num_keep_groups(num_groups: int, keep_ratio: float) -> int:
    keep_ratio = float(keep_ratio)
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    return max(1, min(num_groups, int(round(num_groups * keep_ratio))))


def _groups_to_channel_index(
    groups: torch.Tensor,
    group_size: int,
    hidden_channels: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    groups = groups.to(device=device, dtype=torch.long)
    offsets = torch.arange(group_size, device=device, dtype=torch.long)
    idx = groups[:, None] * int(group_size) + offsets[None, :]
    idx = idx.reshape(-1)
    return idx[idx < int(hidden_channels)]


def _group_scores_from_inverted_activation(
    z_act: torch.Tensor,
    hidden_channels: int,
    group_size: int,
) -> torch.Tensor:
    value = z_act[:, :hidden_channels]
    gate = z_act[:, hidden_channels: 2 * hidden_channels]
    pad = (-hidden_channels) % int(group_size)
    if pad:
        value = F.pad(value, (0, 0, 0, 0, 0, pad))
        gate = F.pad(gate, (0, 0, 0, 0, 0, pad))
    num_groups = value.shape[1] // int(group_size)
    value = value.view(value.shape[0], num_groups, int(group_size), value.shape[2], value.shape[3])
    gate = gate.view(gate.shape[0], num_groups, int(group_size), gate.shape[2], gate.shape[3])
    return 0.5 * (value.abs().mean(dim=(0, 2, 3, 4)) + gate.abs().mean(dim=(0, 2, 3, 4)))


def _apply_norm_and_residual(
    ff: torch.nn.Module, hidden_states: torch.Tensor, residual: torch.Tensor | None,
) -> torch.Tensor:
    norm_type = getattr(ff, "norm_type", None)
    if norm_type == "rms_norm" and getattr(ff, "norm", None) is not None:
        hidden_states = hidden_states.movedim(1, -1)
        hidden_states = ff.norm(hidden_states)
        hidden_states = hidden_states.movedim(-1, 1)
    if residual is not None:
        hidden_states = hidden_states + residual
    return hidden_states


def _dense_glumb_forward(ff: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    residual = hidden_states if getattr(ff, "residual_connection", False) else None
    hidden_states = ff.conv_inverted(hidden_states)
    hidden_states = ff.nonlinearity(hidden_states)
    hidden_states = ff.conv_depth(hidden_states)
    hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
    hidden_states = hidden_states * ff.nonlinearity(gate)
    hidden_states = ff.conv_point(hidden_states)
    return _apply_norm_and_residual(ff, hidden_states, residual)


def _make_static_forward(
    ff: torch.nn.Module,
    active_idx_cpu: torch.Tensor,
    *,
    group_size: int,
    min_skip_ratio: float,
    stats: _SanaSparseStats,
) -> Any:
    hidden_channels = ff.conv_point.weight.shape[1]
    num_groups = (hidden_channels + group_size - 1) // group_size
    kept_groups = (int(active_idx_cpu.numel()) + group_size - 1) // group_size
    skip_ratio = 1.0 - kept_groups / max(num_groups, 1)
    if skip_ratio < min_skip_ratio or int(active_idx_cpu.numel()) >= hidden_channels:
        def dense_fallback(_self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            stats.record(num_groups, num_groups, fallback=True)
            return _dense_glumb_forward(_self, hidden_states)
        return dense_fallback

    device = ff.conv_point.weight.device
    active_idx = active_idx_cpu.to(device=device, dtype=torch.long)
    inv_idx = torch.cat([active_idx, active_idx + hidden_channels], dim=0)

    inv_w = ff.conv_inverted.weight.index_select(0, inv_idx).contiguous()
    inv_b = ff.conv_inverted.bias.index_select(0, inv_idx).contiguous() if ff.conv_inverted.bias is not None else None

    depth_w = ff.conv_depth.weight.index_select(0, inv_idx).contiguous()
    depth_b = ff.conv_depth.bias.index_select(0, inv_idx).contiguous() if ff.conv_depth.bias is not None else None

    point_w = ff.conv_point.weight.index_select(1, active_idx).contiguous()
    point_b = ff.conv_point.bias if ff.conv_point.bias is not None else None

    inv_stride, inv_padding, inv_dilation = ff.conv_inverted.stride, ff.conv_inverted.padding, ff.conv_inverted.dilation
    depth_stride, depth_padding, depth_dilation = ff.conv_depth.stride, ff.conv_depth.padding, ff.conv_depth.dilation
    point_stride, point_padding, point_dilation = ff.conv_point.stride, ff.conv_point.padding, ff.conv_point.dilation
    kept_channels = int(active_idx.numel())

    def forward(_self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states if getattr(_self, "residual_connection", False) else None
        z = F.conv2d(hidden_states, inv_w, inv_b, inv_stride, inv_padding, inv_dilation, groups=1)
        z = _self.nonlinearity(z)
        z = F.conv2d(z, depth_w, depth_b, depth_stride, depth_padding, depth_dilation, groups=2 * kept_channels)
        value, gate = z.split(kept_channels, dim=1)
        z = value * _self.nonlinearity(gate)
        z = F.conv2d(z, point_w, point_b, point_stride, point_padding, point_dilation, groups=1)
        stats.record(num_groups, kept_groups, fallback=False)
        return _apply_norm_and_residual(_self, z, residual)

    return forward


def _make_dynamic_forward(
    ff: torch.nn.Module,
    *,
    keep_ratio: float,
    group_size: int,
    min_skip_ratio: float,
    stats: _SanaSparseStats,
) -> Any:
    hidden_channels = ff.conv_point.weight.shape[1]
    num_groups = (hidden_channels + group_size - 1) // group_size
    keep_groups = _num_keep_groups(num_groups, keep_ratio)
    skip_ratio = 1.0 - keep_groups / max(num_groups, 1)
    if skip_ratio < min_skip_ratio or keep_groups >= num_groups:
        def dense_fallback(_self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            stats.record(num_groups, num_groups, fallback=True)
            return _dense_glumb_forward(_self, hidden_states)
        return dense_fallback

    depth_stride, depth_padding, depth_dilation = ff.conv_depth.stride, ff.conv_depth.padding, ff.conv_depth.dilation
    point_stride, point_padding, point_dilation = ff.conv_point.stride, ff.conv_point.padding, ff.conv_point.dilation

    def forward(_self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states if getattr(_self, "residual_connection", False) else None
        z = _self.conv_inverted(hidden_states)
        z = _self.nonlinearity(z)

        scores = _group_scores_from_inverted_activation(z, hidden_channels, group_size)
        active_groups = torch.topk(scores, k=keep_groups, largest=True).indices.sort().values
        active_idx = _groups_to_channel_index(active_groups, group_size, hidden_channels, device=z.device)
        kept_channels = int(active_idx.numel())
        inv_idx = torch.cat([active_idx, active_idx + hidden_channels], dim=0)

        z = z.index_select(1, inv_idx).contiguous()
        depth_w = _self.conv_depth.weight.index_select(0, inv_idx).contiguous()
        depth_b = _self.conv_depth.bias.index_select(0, inv_idx).contiguous() if _self.conv_depth.bias is not None else None
        z = F.conv2d(z, depth_w, depth_b, depth_stride, depth_padding, depth_dilation, groups=2 * kept_channels)
        value, gate = z.split(kept_channels, dim=1)
        z = value * _self.nonlinearity(gate)

        point_w = _self.conv_point.weight.index_select(1, active_idx).contiguous()
        point_b = _self.conv_point.bias if _self.conv_point.bias is not None else None
        z = F.conv2d(z, point_w, point_b, point_stride, point_padding, point_dilation, groups=1)
        stats.record(num_groups, keep_groups, fallback=False)
        return _apply_norm_and_residual(_self, z, residual)

    return forward


def _iter_sana_ffns(pipe: Any):
    blocks = getattr(getattr(pipe, "transformer", None), "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("Expected pipe.transformer.transformer_blocks.")
    for layer_idx, block in enumerate(blocks):
        ff = getattr(block, "ff", None)
        if ff is not None and all(hasattr(ff, name) for name in ("conv_inverted", "conv_depth", "conv_point", "nonlinearity")):
            yield layer_idx, ff


def _restore_sparse_patch(ff: torch.nn.Module) -> None:
    if hasattr(ff, _ORIG_FORWARD):
        ff.forward = getattr(ff, _ORIG_FORWARD)
        delattr(ff, _ORIG_FORWARD)


def _restore_observer_patch(ff: torch.nn.Module) -> None:
    if hasattr(ff, _OBSERVER_FORWARD):
        ff.forward = getattr(ff, _OBSERVER_FORWARD)
        delattr(ff, _OBSERVER_FORWARD)


def install_sana_ffn_group_sparse(pipe: Any, config: SanaFFNGroupSparseConfig | None = None) -> _SanaSparseStats:
    config = config or SanaFFNGroupSparseConfig()
    mode = config.mode.lower()
    if mode not in {"static", "dynamic"}:
        raise ValueError(f"Unsupported Sana FFN sparse mode {config.mode!r}.")

    plan = None
    if mode == "static":
        if config.plan_path is None:
            raise ValueError("Static Sana FFN group sparsity requires plan_path.")
        plan = load_group_plan(config.plan_path)
        if plan.get("format") != "dit_accel.sana_ffn_group_plan.v1":
            raise ValueError(f"Unrecognized Sana FFN group plan format: {plan.get('format')!r}")
        # Mismatched group_size silently creates wrong channel indices; fail loudly.
        if int(plan.get("group_size")) != int(config.group_size):
            raise ValueError(
                f"Plan group_size={plan.get('group_size')} but config group_size={config.group_size}."
            )

    stats = _SanaSparseStats()
    stats.mode = mode

    installed = 0
    for layer_idx, ff in _iter_sana_ffns(pipe):
        _restore_observer_patch(ff)
        _restore_sparse_patch(ff)
        setattr(ff, _ORIG_FORWARD, ff.forward)

        if mode == "static":
            assert plan is not None
            layer_payload = plan["layers"].get(layer_idx) or plan["layers"].get(str(layer_idx))
            if layer_payload is None:
                active_idx = torch.arange(ff.conv_point.weight.shape[1], dtype=torch.long)
            else:
                active_idx = layer_payload["active_idx"].to(dtype=torch.long)
            new_forward = _make_static_forward(
                ff,
                active_idx,
                group_size=int(config.group_size),
                min_skip_ratio=float(config.min_skip_ratio),
                stats=stats,
            )
        else:
            new_forward = _make_dynamic_forward(
                ff,
                keep_ratio=float(config.keep_ratio),
                group_size=int(config.group_size),
                min_skip_ratio=float(config.min_skip_ratio),
                stats=stats,
            )

        ff.forward = MethodType(new_forward, ff)
        installed += 1

    stats.layers_installed = installed
    pipe._dit_accel_sana_ffn_sparse = stats
    if config.verbose:
        print(
            f"Installed Sana FFN group sparsity: mode={mode}, layers={installed}, "
            f"keep_ratio={config.keep_ratio}, group_size={config.group_size}, min_skip={config.min_skip_ratio}"
        )
    return stats


def install_sana_ffn_group_observer(pipe: Any, group_size: int = 32) -> _GroupObserverState:
    state = _GroupObserverState(group_size=int(group_size))

    for layer_idx, ff in _iter_sana_ffns(pipe):
        _restore_sparse_patch(ff)
        _restore_observer_patch(ff)
        setattr(ff, _OBSERVER_FORWARD, ff.forward)

        def make_forward(idx: int):
            def forward(_self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
                residual = hidden_states if getattr(_self, "residual_connection", False) else None
                z = _self.conv_inverted(hidden_states)
                z = _self.nonlinearity(z)
                state.record(idx, z)
                z = _self.conv_depth(z)
                value, gate = torch.chunk(z, 2, dim=1)
                z = value * _self.nonlinearity(gate)
                z = _self.conv_point(z)
                return _apply_norm_and_residual(_self, z, residual)
            return forward

        ff.forward = MethodType(make_forward(layer_idx), ff)

    pipe._dit_accel_sana_ffn_observer = state
    print(f"Installed Sana FFN group observer on {len(list(_iter_sana_ffns(pipe)))} layers, group_size={group_size}")
    return state
