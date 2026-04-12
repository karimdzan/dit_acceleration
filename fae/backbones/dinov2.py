from typing import Any

import torch
from transformers import AutoImageProcessor, Dinov2Model

from .base import BackboneFeatures, FrozenVisionBackbone


class DINOv2Backbone(FrozenVisionBackbone):
    def __init__(self, model_name: str = "facebook/dinov2-base", prefix_tokens: int | None = 1, **kwargs) -> None:
        super().__init__()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.output_dim = int(self.model.config.hidden_size)
        self.prefix_tokens = prefix_tokens
        self.patch_size = int(kwargs.get("patch_size", getattr(self.model.config, "patch_size", 16)))

    def preprocess(self, images: list[Any]) -> dict[str, torch.Tensor]:
        return self.processor(images=images, return_tensors="pt")

    @torch.no_grad()
    def forward_features(self, inputs: dict[str, torch.Tensor]) -> BackboneFeatures:
        outputs = self.model(**inputs)
        return self._split_patch_tokens(outputs.last_hidden_state, self.prefix_tokens)
