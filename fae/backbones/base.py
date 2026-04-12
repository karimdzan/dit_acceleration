from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from fae.utils.image import infer_hw_from_tokens, infer_prefix_tokens


@dataclass
class BackboneFeatures:
    tokens: torch.Tensor
    spatial_shape: tuple[int, int]
    prefix_tokens: int
    raw_sequence: torch.Tensor


class FrozenVisionBackbone(nn.Module):
    model_name: str
    output_dim: int
    patch_size: int | None = None

    def preprocess(self, images: list[Any]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def forward_features(self, inputs: dict[str, torch.Tensor]) -> BackboneFeatures:
        raise NotImplementedError

    @staticmethod
    def _split_patch_tokens(sequence: torch.Tensor, prefix_tokens: int | None = None) -> BackboneFeatures:
        if prefix_tokens is None:
            prefix_tokens = infer_prefix_tokens(sequence.shape[1])
        patch_tokens = sequence[:, prefix_tokens:]
        spatial = infer_hw_from_tokens(patch_tokens.shape[1])
        return BackboneFeatures(
            tokens=patch_tokens,
            spatial_shape=spatial,
            prefix_tokens=prefix_tokens,
            raw_sequence=sequence,
        )
