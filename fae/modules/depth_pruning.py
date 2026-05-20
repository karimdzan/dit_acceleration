
"""Depth-pruning utilities for SANA/SANA-Sprint/DiT-style transformers.

The primary inference baseline physically removes selected transformer blocks from
``model.transformer_blocks`` or ``model.blocks``. This reduces full block compute:
self-attention, cross-attention, MLP/GLUMBConv, normalization, and memory traffic.

Creates a pruned student from a pretrained teacher for evaluation.
"""

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class DepthPruningResult:
    original_num_layers: int
    kept_layers: list[int]
    dropped_layers: list[int]
    strategy: str

    @property
    def pruned_num_layers(self) -> int:
        return len(self.kept_layers)

    @property
    def drop_ratio(self) -> float:
        if self.original_num_layers <= 0:
            return 0.0
        return len(self.dropped_layers) / self.original_num_layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_num_layers": self.original_num_layers,
            "pruned_num_layers": self.pruned_num_layers,
            "kept_layers": self.kept_layers,
            "dropped_layers": self.dropped_layers,
            "strategy": self.strategy,
            "drop_ratio": self.drop_ratio,
        }


class IdentityBlock(nn.Module):
    """Fallback skip wrapper with a broad block-like forward signature.

    Physical layer removal is preferred for speed. This class is useful for
    debugging and for implementations where the block count must remain fixed.
    """

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


def get_block_container(model: nn.Module) -> tuple[str, nn.ModuleList | nn.Sequential | list[nn.Module]]:
    for name in ("transformer_blocks", "blocks", "layers"):
        container = getattr(model, name, None)
        if isinstance(container, (nn.ModuleList, nn.Sequential, list)):
            return name, container
    raise ValueError(
        "Could not find a transformer block container. Expected `transformer_blocks`, `blocks`, or `layers`."
    )


def parse_layer_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _sanitize_layers(indices: list[int], num_layers: int) -> list[int]:
    return sorted({int(i) for i in indices if 0 <= int(i) < num_layers})


def _default_drop_count(num_layers: int, cfg: dict[str, Any]) -> int:
    if cfg.get("drop_count") is not None:
        return int(cfg["drop_count"])
    ratio = float(cfg.get("drop_ratio", 0.2))
    return max(0, min(num_layers - 1, int(round(num_layers * ratio))))


def select_layers_to_drop(num_layers: int, cfg: dict[str, Any] | None) -> list[int]:
    cfg = cfg or {}
    explicit_drop = parse_layer_list(cfg.get("drop_layers"))
    if explicit_drop is not None:
        return _sanitize_layers(explicit_drop, num_layers)

    explicit_keep = parse_layer_list(cfg.get("keep_layers"))
    if explicit_keep is not None:
        keep = set(_sanitize_layers(explicit_keep, num_layers))
        if not keep:
            raise ValueError("keep_layers resolved to an empty list.")
        return [i for i in range(num_layers) if i not in keep]

    strategy = str(cfg.get("strategy", "uniform")).lower()
    drop_count = _default_drop_count(num_layers, cfg)
    preserve_first = int(cfg.get("preserve_first", 1))
    preserve_last = int(cfg.get("preserve_last", 1))
    candidates = list(range(max(0, preserve_first), max(preserve_first, num_layers - max(0, preserve_last))))
    if drop_count <= 0 or not candidates:
        return []
    drop_count = min(drop_count, len(candidates))

    if strategy in {"uniform", "uniform_drop", "even"}:
        if drop_count == 1:
            return [candidates[len(candidates) // 2]]
        positions = []
        for k in range(drop_count):
            pos = round((k + 1) * (len(candidates) + 1) / (drop_count + 1)) - 1
            pos = max(0, min(len(candidates) - 1, pos))
            positions.append(pos)
        # De-duplicate rare round collisions by filling from remaining candidates near center.
        selected = [candidates[p] for p in positions]
        if len(set(selected)) < drop_count:
            missing = [i for i in candidates if i not in set(selected)]
            selected.extend(missing[: drop_count - len(set(selected))])
        return sorted(set(selected))[:drop_count]

    if strategy in {"middle", "center", "middle_drop"}:
        center = (num_layers - 1) / 2.0
        ranked = sorted(candidates, key=lambda i: (abs(i - center), i))
        return sorted(ranked[:drop_count])

    if strategy in {"late", "tail"}:
        return sorted(candidates[-drop_count:])

    if strategy in {"early", "head"}:
        return sorted(candidates[:drop_count])

    raise ValueError(f"Unknown depth-pruning strategy={strategy!r}.")


def apply_depth_pruning(model: nn.Module, cfg: dict[str, Any] | None) -> DepthPruningResult:
    """Physically remove selected transformer blocks.

    Supported config keys: enabled, strategy, drop_ratio, drop_count,
    drop_layers, keep_layers, preserve_first, preserve_last, mode.
    """

    cfg = cfg or {}
    if not bool(cfg.get("enabled", True)):
        name, blocks = get_block_container(model)
        n = len(blocks)
        result = DepthPruningResult(n, list(range(n)), [], str(cfg.get("strategy", "disabled")))
        setattr(model, "depth_pruning", result.to_dict())
        return result

    name, blocks = get_block_container(model)
    num_layers = len(blocks)
    drop_layers = select_layers_to_drop(num_layers, cfg)
    keep_layers = [i for i in range(num_layers) if i not in set(drop_layers)]
    if not keep_layers:
        raise ValueError("Depth pruning would remove all layers. Reduce drop_ratio/drop_count/drop_layers.")

    mode = str(cfg.get("mode", "remove")).lower()
    if mode == "remove":
        kept_blocks = [blocks[i] for i in keep_layers]
        new_container = nn.ModuleList(kept_blocks) if isinstance(blocks, nn.ModuleList) else nn.Sequential(*kept_blocks)
        setattr(model, name, new_container)
        if hasattr(model, "register_to_config"):
            try:
                model.register_to_config(num_layers=len(keep_layers))
            except Exception:
                pass
        elif hasattr(model, "config") and isinstance(getattr(model, "config"), dict):
            try:
                model.config["num_layers"] = len(keep_layers)
            except Exception:
                pass
    elif mode in {"skip", "identity"}:
        for idx in drop_layers:
            blocks[idx] = IdentityBlock()
    else:
        raise ValueError(f"Unknown depth-pruning mode={mode!r}; expected remove or skip.")

    result = DepthPruningResult(
        original_num_layers=num_layers,
        kept_layers=keep_layers,
        dropped_layers=drop_layers,
        strategy=str(cfg.get("strategy", "uniform")),
    )
    setattr(model, "depth_pruning", result.to_dict())
    return result
